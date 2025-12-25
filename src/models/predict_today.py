import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.team_mapping import normalize_team_name
from src.utils.betting_advanced import (
    calculate_confidence_interval,
    calculate_expected_value,
    calculate_market_efficiency_score,
    calculate_bankroll_risk
)

# --- CONFIGURATION ---
MODEL_PATH = 'models/nba_xgb_model.joblib' 
STATS_PATH = 'data/processed/nba_model.csv'        
ODDS_PATH = 'data/odds/live_odds.csv'              
PLAYER_PATH = 'data/processed/processed_player.csv' 
INJURY_PATH = 'data/raw/espn_injuries_current.csv'   
OUTPUT_PATH = 'results/todays_bets.csv'

# Required features for model prediction
REQUIRED_FEATURES = [
    'ELO_TEAM', 'ELO_OPP', 'IS_HOME', 'IS_B2B', 'IS_3IN4',
    'ROLL_OFF_RTG', 'ROLL_DEF_RTG', 'ROLL_PACE',
    'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
    'ROLL_ROSTER_TALENT_SCORE'
]

# Betting Configuration
BANKROLL = 1000
KELLY_FRACTION = 0.35  # Fractional Kelly (35% of optimal)
MAX_BET_PCT = 0.05     # Never bet more than 5% of bankroll on one game
MIN_EDGE_THRESHOLD = 0.02  # Minimum 2% edge required to bet
MAX_EDGE_THRESHOLD = 0.15  # Maximum 15% edge (above this = data error)
MIN_WIN_PROB = 0.30    # Don't bet if win probability is below 40% (too risky)
MAX_WIN_PROB = 0.90    # Don't bet if win probability is above 85% (odds too short)

# Probability Calibration (to fix overconfident model)
# Model tends to be overconfident, so we dampen probabilities
PROB_CALIBRATION_FACTOR = 1.0  # Multiply edge by this factor (reduces overconfidence)
# Alternative: Use sigmoid calibration to pull probabilities toward 50%
USE_SIGMOID_CALIBRATION = False
CALIBRATION_STRENGTH = 0.3  # Higher = more calibration (pull toward 50%)

# --- 1. HELPER FUNCTIONS ---

def load_injury_report():
    """
    Load injury report from CSV file.
    Returns dictionary mapping team abbreviations to lists of injured player names.
    """
    if not os.path.exists(INJURY_PATH):
        print(f"⚠️ Injury file not found at {INJURY_PATH}. Continuing without injury data.")
        return {}
    
    try:
        df = pd.read_csv(INJURY_PATH)
        df.columns = df.columns.str.lower().str.strip()
        
        status_col = None
        for col in ['status', 'description', 'severity']:
            if col in df.columns and df[col].astype(str).str.contains('Out', case=False).any():
                status_col = col
                break
        status_col = status_col if status_col else 'status'

        if status_col not in df.columns:
            print("⚠️ Could not find status column in injury data.")
            return {}

        # Only count players with "Out" status (not "Questionable", "Day-to-Day", etc.)
        # Be more strict - must contain "out" but not be part of other words
        out_players = df[
            df[status_col].astype(str).str.lower().str.contains(r'\bout\b', regex=True, na=False)
        ]
        
        # Additional filter: exclude if description contains "questionable", "probable", "day-to-day"
        if 'description' in df.columns:
            out_players = out_players[
                ~out_players['description'].astype(str).str.lower().str.contains(
                    'questionable|probable|day.to.day|dtd', regex=True, na=False
                )
            ]
        
        team_col = 'team' if 'team' in df.columns else 'team_name'
        player_col = 'player_name' if 'player_name' in df.columns else 'player'
        
        if team_col not in df.columns or player_col not in df.columns:
            print(f"⚠️ Missing required columns in injury data: need {team_col} and {player_col}")
            return {}
        
        raw_dict = out_players.groupby(team_col)[player_col].apply(list).to_dict()
        # Normalize team names using centralized mapping
        final_dict = {normalize_team_name(k): v for k, v in raw_dict.items()}
        print(f"✅ Loaded Injury Report: {len(final_dict)} teams have players OUT.")
        return final_dict
    except FileNotFoundError:
        print(f"⚠️ Injury file not found: {INJURY_PATH}")
        return {}
    except pd.errors.EmptyDataError:
        print(f"⚠️ Injury file is empty: {INJURY_PATH}")
        return {}
    except Exception as e:
        print(f"⚠️ Error loading injury report: {e}")
        print("   Continuing without injury data.")
        return {}

def get_roster_strength_simulation(team_abbr, injury_list, df_players):
    roster = df_players[df_players['TEAM_ABBREVIATION'] == team_abbr].copy()
    if roster.empty: return 0 

    if injury_list:
        roster = roster[~roster['PLAYER_NAME'].isin(injury_list)]
    
    roster['MIN_AVG'] = roster['MIN_AVG'].replace(0, 1)
    roster['PPM'] = roster['GAME_SCORE_AVG'] / roster['MIN_AVG']
    
    rotation = roster.sort_values('MIN_AVG', ascending=False).head(9).copy()
    
    total_avg_minutes = rotation['MIN_AVG'].sum()
    scaling = 240 / total_avg_minutes if total_avg_minutes > 0 else 1.0
    
    rotation['PROJ_MIN'] = (rotation['MIN_AVG'] * scaling).clip(upper=42)
    rotation['PROJ_SCORE'] = rotation['PPM'] * rotation['PROJ_MIN']
    
    return rotation['PROJ_SCORE'].sum() * 0.95 

def run_monte_carlo(home_team, away_team, df_players, injury_dict=None, n_sims=1000):
    """
    Run Monte Carlo simulation to estimate win probability.
    Accounts for injuries by excluding injured players from rosters.
    """
    if injury_dict is None:
        injury_dict = {}
    
    h_roster = df_players[df_players['TEAM_ABBREVIATION'] == home_team].copy()
    a_roster = df_players[df_players['TEAM_ABBREVIATION'] == away_team].copy()
    
    if h_roster.empty or a_roster.empty:
        return 0.5
    
    # Remove injured players
    h_injured = injury_dict.get(home_team, [])
    a_injured = injury_dict.get(away_team, [])
    
    if h_injured:
        h_roster = h_roster[~h_roster['PLAYER_NAME'].isin(h_injured)]
    if a_injured:
        a_roster = a_roster[~a_roster['PLAYER_NAME'].isin(a_injured)]
    
    if h_roster.empty or a_roster.empty:
        return 0.5
    
    # Get top 9 players by minutes (rotation players)
    h_rotation = h_roster.nlargest(9, 'MIN_AVG') if len(h_roster) > 9 else h_roster
    a_rotation = a_roster.nlargest(9, 'MIN_AVG') if len(a_roster) > 9 else a_roster
    
    # Calculate points per minute for each player
    h_rotation['PPM'] = h_rotation['GAME_SCORE_AVG'] / h_rotation['MIN_AVG'].replace(0, 1)
    a_rotation['PPM'] = a_rotation['GAME_SCORE_AVG'] / a_rotation['MIN_AVG'].replace(0, 1)
    
    # Project minutes (scale to 240 total team minutes)
    h_total_min = h_rotation['MIN_AVG'].sum()
    a_total_min = a_rotation['MIN_AVG'].sum()
    
    h_scale = 240 / h_total_min if h_total_min > 0 else 1.0
    a_scale = 240 / a_total_min if a_total_min > 0 else 1.0
    
    h_proj_min = (h_rotation['MIN_AVG'] * h_scale).clip(upper=42)
    a_proj_min = (a_rotation['MIN_AVG'] * a_scale).clip(upper=42)
    
    h_proj_score = (h_rotation['PPM'] * h_proj_min).sum()
    a_proj_score = (a_rotation['PPM'] * a_proj_min).sum()
    
    # Use standard deviation based on historical variance
    # Typical NBA game score variance is ~8-12 points per player
    h_std = np.std(h_rotation['GAME_SCORE_AVG'].values) if len(h_rotation) > 1 else 8.0
    a_std = np.std(a_rotation['GAME_SCORE_AVG'].values) if len(a_rotation) > 1 else 8.0
    
    # Run simulations with team-level variance
    h_sims = np.random.normal(h_proj_score, h_std * np.sqrt(len(h_rotation)), n_sims)
    a_sims = np.random.normal(a_proj_score, a_std * np.sqrt(len(a_rotation)), n_sims)
    
    # Add home court advantage (typically 2-4 points)
    h_sims = h_sims + 3.0
    
    # Calculate win probability
    wins = (h_sims > a_sims).sum()
    return wins / n_sims

def calibrate_probability(prob):
    """
    Calibrate overconfident model probabilities.
    Pulls probabilities toward 50% to reduce overconfidence.
    """
    if not USE_SIGMOID_CALIBRATION:
        return prob
    
    # Sigmoid calibration: pull toward 50%
    # Formula: calibrated = 0.5 + (prob - 0.5) * (1 - CALIBRATION_STRENGTH)
    calibrated = 0.5 + (prob - 0.5) * (1 - CALIBRATION_STRENGTH)
    return calibrated

def get_kelly_bet(prob_win, decimal_odds, implied_prob, bankroll=BANKROLL):
    """
    Calculate Kelly bet size with safety constraints and calibration.
    
    Args:
        prob_win: Model's win probability (0-1)
        decimal_odds: Decimal odds (e.g., 2.0 for even money)
        implied_prob: Implied probability from odds (1/decimal_odds)
        bankroll: Current bankroll
    
    Returns:
        Tuple of (bet_amount, raw_bet_amount, edge, calibrated_prob)
        bet_amount: Final bet size after all constraints
        raw_bet_amount: Bet size before max cap (for display)
        edge: Calculated edge
        calibrated_prob: Calibrated probability used
    """
    # Calibrate probability to reduce overconfidence
    calibrated_prob = calibrate_probability(prob_win)
    
    # Calculate edge
    edge = calibrated_prob - implied_prob
    
    # Cap maximum edge (edges >15% are usually data errors)
    if edge > MAX_EDGE_THRESHOLD:
        print(f"⚠️ Edge capped at {MAX_EDGE_THRESHOLD:.1%} (calculated: {edge:.1%})")
        edge = MAX_EDGE_THRESHOLD
        calibrated_prob = implied_prob + edge
    
    # Check minimum edge threshold
    if edge < MIN_EDGE_THRESHOLD:
        return (0.0, 0.0, edge, calibrated_prob)
    
    # Check win probability bounds (avoid extreme bets)
    if calibrated_prob < MIN_WIN_PROB or calibrated_prob > MAX_WIN_PROB:
        return (0.0, 0.0, edge, calibrated_prob)
    
    # Apply edge dampening (additional safety)
    edge = edge * PROB_CALIBRATION_FACTOR
    calibrated_prob = implied_prob + edge
    
    # Kelly formula
    b = decimal_odds - 1
    p = calibrated_prob
    q = 1.0 - p
    
    if b <= 0:
        return (0.0, 0.0, edge, calibrated_prob)
    
    kelly_pct = (b * p - q) / b
    
    # Only bet if positive Kelly
    if kelly_pct <= 0:
        return (0.0, 0.0, edge, calibrated_prob)
    
    # Calculate raw bet (before cap)
    raw_bet_pct = kelly_pct * KELLY_FRACTION
    raw_bet_amount = bankroll * raw_bet_pct
    
    # Apply max bet cap
    bet_pct = min(raw_bet_pct, MAX_BET_PCT)
    bet_amount = bankroll * bet_pct
    
    return (bet_amount, raw_bet_amount, edge, calibrated_prob)

# --- 2. MAIN EXECUTION ---

def predict():
    from datetime import datetime, timezone
    
    print("\n--- 1. LOADING DATA ---")
    try:
        df_stats = pd.read_csv(STATS_PATH)
        df_odds = pd.read_csv(ODDS_PATH)
        df_players = pd.read_csv(PLAYER_PATH)
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"❌ Critical Error: {e}")
        return

    # Parse dates but don't filter - show all games
    df_odds['GAME_DATE'] = pd.to_datetime(df_odds['GAME_DATE'])
    df_odds = df_odds.sort_values('GAME_DATE')
    
    print(f"📅 Found {len(df_odds)} games in odds file")
    if len(df_odds) > 0:
        date_range = f"{df_odds['GAME_DATE'].min().date()} to {df_odds['GAME_DATE'].max().date()}"
        print(f"   Date range: {date_range}")

    current_injuries = load_injury_report()
    predictions = []
    
    print(f"\n--- 2. PREDICTING {len(df_odds)} GAMES ---")
    
    def get_last_stats(abbr):
        subset = df_stats[df_stats['TEAM_ABBREVIATION'] == abbr]
        if subset.empty: return None
        return subset.iloc[-1]

    for index, row in df_odds.iterrows():
        home_raw, away_raw = row.get('HOME_TEAM'), row.get('AWAY_TEAM')
        home_abbr = normalize_team_name(home_raw)
        away_abbr = normalize_team_name(away_raw)
        
        home_hist = get_last_stats(home_abbr)
        away_hist = get_last_stats(away_abbr) 
        
        if home_hist is None or away_hist is None:
            print(f"⚠️ Skipping {home_abbr} vs {away_abbr}: Missing historical data")
            continue

        # Validate required features exist
        missing_features = [f for f in REQUIRED_FEATURES if f not in home_hist.index]
        if missing_features:
            print(f"⚠️ Skipping {home_abbr} vs {away_abbr}: Missing features: {missing_features}")
            continue

        # --- SIMULATION ---
        h_out = current_injuries.get(home_abbr, [])
        h_strength = get_roster_strength_simulation(home_abbr, h_out, df_players)
        
        a_out = current_injuries.get(away_abbr, [])
        a_strength = get_roster_strength_simulation(away_abbr, a_out, df_players)
        
        # Get average strength with fallback, protect against division by zero
        h_avg_str = home_hist.get('ROLL_ROSTER_TALENT_SCORE', 0)
        a_avg_str = away_hist.get('ROLL_ROSTER_TALENT_SCORE', 0)
        
        # Use fallback if value is too low (likely invalid)
        if h_avg_str <= 10:
            h_avg_str = 150
        if a_avg_str <= 10:
            a_avg_str = 150

        # Protect against division by zero
        if h_avg_str <= 0:
            h_avg_str = 150
        if a_avg_str <= 0:
            a_avg_str = 150

        h_health = h_strength / h_avg_str if h_avg_str > 0 else 1.0
        a_health = a_strength / a_avg_str if a_avg_str > 0 else 1.0
        
        # Clamp health to reasonable range
        h_health = np.clip(h_health, 0.5, 1.5)
        a_health = np.clip(a_health, 0.5, 1.5)
        
        # INJURY PENALTY (250 pts)
        h_elo_adj = home_hist.get('ELO_TEAM', 1500) - ((1 - h_health) * 250)
        a_elo_adj = away_hist.get('ELO_OPP', 1500) - ((1 - a_health) * 250)

        # Build feature row with safe .get() calls and defaults
        feature_row = pd.DataFrame([{
            'ELO_TEAM': h_elo_adj,
            'ELO_OPP': a_elo_adj,
            'IS_HOME': 1,
            'IS_B2B': home_hist.get('IS_B2B', 0),
            'IS_3IN4': home_hist.get('IS_3IN4', 0),
            'ROLL_OFF_RTG': home_hist.get('ROLL_OFF_RTG', 100),
            'ROLL_DEF_RTG': home_hist.get('ROLL_DEF_RTG', 100),
            'ROLL_PACE': home_hist.get('ROLL_PACE', 100),
            'ROLL_EFG_PCT': home_hist.get('ROLL_EFG_PCT', 0.5),
            'ROLL_TOV_PCT': home_hist.get('ROLL_TOV_PCT', 0.15),
            'ROLL_ORB_PCT': home_hist.get('ROLL_ORB_PCT', 0.25),
            'ROLL_FTR': home_hist.get('ROLL_FTR', 0.25),
            'ROLL_ROSTER_TALENT_SCORE': h_strength
        }])
        
        # 1. Predictions
        try:
            prob_home_xgb = model.predict_proba(feature_row)[0][1]
        except Exception as e:
            print(f"⚠️ Error in XGBoost prediction for {home_abbr} vs {away_abbr}: {e}")
            continue
        
        # Pass injury dict to Monte Carlo
        injury_dict = {home_abbr: h_out, away_abbr: a_out}
        prob_home_mc = run_monte_carlo(home_abbr, away_abbr, df_players, injury_dict)
        
        # Ensemble prediction with weights
        final_prob_home = (0.7 * prob_home_xgb) + (0.3 * prob_home_mc)
        final_prob_away = 1.0 - final_prob_home  # Away Probability
        
        # Calculate confidence intervals for uncertainty quantification
        ci_home = calculate_confidence_interval(final_prob_home, n_samples=500)
        ci_away = calculate_confidence_interval(final_prob_away, n_samples=500)
        
        # 2. Odds Calculation
        home_ml = row.get('HOME_ML', 0)
        away_ml = row.get('AWAY_ML', 0)
        
        if home_ml == 0 or away_ml == 0:
            print(f"⚠️ Skipping {home_abbr} vs {away_abbr}: Missing odds")
            continue
        
        dec_home = (home_ml/100 + 1) if home_ml > 0 else (100/abs(home_ml) + 1)
        implied_home = 1 / dec_home
        
        dec_away = (away_ml/100 + 1) if away_ml > 0 else (100/abs(away_ml) + 1)
        implied_away = 1 / dec_away
        
        # 3. Calculate Market Efficiency
        market_efficiency_home = calculate_market_efficiency_score(
            final_prob_home, implied_home, historical_accuracy=0.55
        )
        market_efficiency_away = calculate_market_efficiency_score(
            final_prob_away, implied_away, historical_accuracy=0.55
        )
        
        # 4. Calculate Expected Value
        ev_home = calculate_expected_value(final_prob_home, dec_home)
        ev_away = calculate_expected_value(final_prob_away, dec_away)
        
        # 5. Calculate Bets (Check BOTH sides) with edge threshold and calibration
        bet_home, raw_bet_home, edge_home, calib_prob_home = get_kelly_bet(
            final_prob_home, dec_home, implied_home, BANKROLL
        )
        bet_away, raw_bet_away, edge_away, calib_prob_away = get_kelly_bet(
            final_prob_away, dec_away, implied_away, BANKROLL
        )
        
        # 6. Calculate risk metrics for recommended bet
        if bet_home > 0:
            risk_metrics = calculate_bankroll_risk(calib_prob_home, bet_home, BANKROLL, dec_home)
        elif bet_away > 0:
            risk_metrics = calculate_bankroll_risk(calib_prob_away, bet_away, BANKROLL, dec_away)
        else:
            risk_metrics = None
        
        # 4. Decision Logic
        rec = "NO BET"
        edge_display = 0.0
        final_prob_display = calib_prob_home  # Use calibrated probability
        implied_display = implied_home
        bet_size_display = 0.0
        raw_bet_display = 0.0
        
        if bet_home > 0:
            rec = f"BET {home_abbr} ${bet_home:.2f}"
            if raw_bet_home > bet_home:
                rec += f" (capped from ${raw_bet_home:.2f})"
            edge_display = edge_home
            final_prob_display = calib_prob_home
            implied_display = implied_home
            bet_size_display = bet_home
            raw_bet_display = raw_bet_home
            
        elif bet_away > 0:
            rec = f"BET {away_abbr} ${bet_away:.2f}"
            if raw_bet_away > bet_away:
                rec += f" (capped from ${raw_bet_away:.2f})"
            edge_display = edge_away
            final_prob_display = calib_prob_away
            implied_display = implied_away
            bet_size_display = bet_away
            raw_bet_display = raw_bet_away

        if rec == "NO BET":
             edge_display = calib_prob_home - implied_home

        # Get game date for display
        game_date = row.get('GAME_DATE', '')
        if isinstance(game_date, str):
            try:
                game_date = pd.to_datetime(game_date).strftime('%Y-%m-%d')
            except:
                game_date = str(game_date)[:10] if len(str(game_date)) > 10 else str(game_date)
        else:
            game_date = str(game_date)[:10] if len(str(game_date)) > 10 else str(game_date)
        
        # Build prediction row with advanced metrics
        pred_row = {
            'Date': game_date,
            'Game': f"{home_abbr} vs {away_abbr}",
            'Win_Prob': f"{final_prob_display:.1%}",
            'Implied': f"{implied_display:.1%}",
            'Edge': f"{edge_display:+.1%}",
            'Kelly': rec,
            'Injuries': f"{len(h_out)} H / {len(a_out)} A"
        }
        
        # Add advanced metrics if bet is recommended
        if bet_home > 0 or bet_away > 0:
            ev = ev_home if bet_home > 0 else ev_away
            efficiency = market_efficiency_home if bet_home > 0 else market_efficiency_away
            ci = ci_home if bet_home > 0 else ci_away
            
            pred_row['EV'] = f"{ev:+.2%}"
            pred_row['Market_Eff'] = f"{efficiency:.2f}"
            pred_row['CI_95%'] = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
            
            if risk_metrics:
                pred_row['Risk/Reward'] = f"{risk_metrics['risk_reward_ratio']:.2f}"
        else:
            pred_row['EV'] = f"{max(ev_home, ev_away):+.2%}"
            pred_row['Market_Eff'] = f"{min(market_efficiency_home, market_efficiency_away):.2f}"
            pred_row['CI_95%'] = "-"
            pred_row['Risk/Reward'] = "-"
        
        predictions.append(pred_row)

    # --- OUTPUT ---
    if predictions:
        results = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        results.to_csv(OUTPUT_PATH, index=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        # Enhanced output formatting
        print("\n" + "="*100)
        print("🏀 NBA BETTING CARD - KELLY 0.35x FRACTIONAL")
        print("="*100)
        
        # Group by date for better organization
        results['Date'] = pd.to_datetime(results['Date'])
        results = results.sort_values('Date')
        
        # Display by date with enhanced columns
        for date, group in results.groupby(results['Date'].dt.date):
            print(f"\n📅 {date}")
            print("-" * 100)
            # Show different columns based on whether bets are recommended
            if 'EV' in group.columns:
                display_cols = ['Game', 'Kelly', 'Win_Prob', 'Implied', 'Edge', 'EV', 'Market_Eff', 'Injuries']
            else:
                display_cols = ['Game', 'Kelly', 'Win_Prob', 'Implied', 'Edge', 'Injuries']
            print(group[display_cols].to_string(index=False))
        
        # Summary statistics
        bets_df = results[results['Kelly'].str.contains('BET', na=False)]
        total_bets = len(bets_df)
        total_games = len(results)
        
        if total_bets > 0:
            # Calculate total bet amount
            bet_amounts = bets_df['Kelly'].str.extract(r'\$(\d+\.?\d*)')[0].astype(float)
            total_stake = bet_amounts.sum()
            avg_edge = bets_df['Edge'].str.rstrip('%').astype(float).mean()
            
            print("\n" + "="*100)
            print("📊 SUMMARY STATISTICS")
            print("="*100)
            print(f"Total Games Analyzed:     {total_games}")
            print(f"Bets Recommended:         {total_bets} ({total_bets/total_games*100:.1f}%)")
            print(f"Total Stake (if all bet): ${total_stake:.2f}")
            print(f"Average Edge:             {avg_edge:+.2f}%")
            print(f"No Bet:                   {total_games - total_bets} ({100-total_bets/total_games*100:.1f}%)")
        else:
            print("\n" + "="*100)
            print("📊 SUMMARY: No bets recommended")
            print("="*100)
            print(f"Total Games Analyzed: {total_games}")
        
        print(f"\n✅ Results saved to {OUTPUT_PATH}")
    else:
        print("❌ No games found.")

if __name__ == "__main__":
    predict()