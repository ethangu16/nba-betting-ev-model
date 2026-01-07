import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import unicodedata
import re
from datetime import timedelta, datetime

# --- CONFIGURATION ---
MODEL_PATH = 'models/nba_xgb_model.joblib' 
STATS_PATH = 'data/processed/nba_model.csv'        
ODDS_PATH = 'data/odds/live_odds.csv'              
PLAYER_PATH = 'data/raw/nba_player_stats.csv'
INJURY_PATH = 'data/raw/espn_injuries_current.csv'   
OUTPUT_PATH = 'results/todays_bets.csv'

# 🟢 FEATURE LIST
REQUIRED_FEATURES = [
    'ELO_TEAM', 'ELO_OPP', 'IS_HOME', 'IS_B2B', 'IS_3IN4',
    'ROSTER_TALENT_SCORE',
    'SMA_20_OFF_RTG', 'SMA_20_DEF_RTG', 'SMA_20_PACE', 'SMA_20_EFG_PCT',
    'SMA_20_TOV_PCT', 'SMA_20_ORB_PCT', 'SMA_20_FTR', 'SMA_20_ROSTER_TALENT_SCORE',
    'EWMA_10_OFF_RTG', 'EWMA_10_DEF_RTG', 'EWMA_10_PACE', 'EWMA_10_EFG_PCT',
    'EWMA_10_TOV_PCT', 'EWMA_10_ORB_PCT', 'EWMA_10_FTR', 'EWMA_10_ROSTER_TALENT_SCORE',
    'EWMA_5_OFF_RTG', 'EWMA_5_DEF_RTG', 'EWMA_5_PACE', 'EWMA_5_EFG_PCT',
    'EWMA_5_TOV_PCT', 'EWMA_5_ORB_PCT', 'EWMA_5_FTR', 'EWMA_5_ROSTER_TALENT_SCORE',
]

# Betting Parameters
BANKROLL = 1000
KELLY_FRACTION = 0.15 
MAX_BET_PCT = 0.03    
MIN_EDGE_THRESHOLD = 0.02 
MAX_EDGE_THRESHOLD = 0.25 # Raised to catch legit "Black Swan" mispricings
MIN_WIN_PROB = 0.35    
USE_SIGMOID_CALIBRATION = False # Disabled to prevent Underdog bias

# --- HELPER FUNCTIONS ---

def normalize_player_name(name):
    if not isinstance(name, str): return str(name)
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    name = name.lower()
    name = name.replace('.', '').replace("'", "").replace("-", " ")
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_team_name(name):
    if pd.isna(name): return None
    name = str(name).upper().strip()
    if name in ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 
                'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 
                'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']:
        return name
    mapping = {
        'ATLANTA HAWKS': 'ATL', 'BOSTON CELTICS': 'BOS', 'BROOKLYN NETS': 'BKN', 'CHARLOTTE HORNETS': 'CHA',
        'CHICAGO BULLS': 'CHI', 'CLEVELAND CAVALIERS': 'CLE', 'DALLAS MAVERICKS': 'DAL', 'DENVER NUGGETS': 'DEN',
        'DETROIT PISTONS': 'DET', 'GOLDEN STATE WARRIORS': 'GSW', 'HOUSTON ROCKETS': 'HOU', 'INDIANA PACERS': 'IND',
        'LOS ANGELES CLIPPERS': 'LAC', 'LA CLIPPERS': 'LAC', 'LOS ANGELES LAKERS': 'LAL', 'L.A. LAKERS': 'LAL',
        'MEMPHIS GRIZZLIES': 'MEM', 'MIAMI HEAT': 'MIA', 'MILWAUKEE BUCKS': 'MIL', 'MINNESOTA TIMBERWOLVES': 'MIN',
        'NEW ORLEANS PELICANS': 'NOP', 'NEW YORK KNICKS': 'NYK', 'OKLAHOMA CITY THUNDER': 'OKC', 'ORLANDO MAGIC': 'ORL',
        'PHILADELPHIA 76ERS': 'PHI', 'PHOENIX SUNS': 'PHX', 'PORTLAND TRAIL BLAZERS': 'POR', 'SACRAMENTO KINGS': 'SAC',
        'SAN ANTONIO SPURS': 'SAS', 'TORONTO RAPTORS': 'TOR', 'UTAH JAZZ': 'UTA', 'WASHINGTON WIZARDS': 'WAS',
        'HAWKS': 'ATL', 'CELTICS': 'BOS', 'NETS': 'BKN', 'HORNETS': 'CHA', 'BULLS': 'CHI', 'CAVS': 'CLE', 
        'MAVERICKS': 'DAL', 'NUGGETS': 'DEN', 'PISTONS': 'DET', 'WARRIORS': 'GSW', 'ROCKETS': 'HOU', 
        'PACERS': 'IND', 'CLIPPERS': 'LAC', 'LAKERS': 'LAL', 'GRIZZLIES': 'MEM', 'HEAT': 'MIA', 
        'BUCKS': 'MIL', 'WOLVES': 'MIN', 'PELICANS': 'NOP', 'KNICKS': 'NYK', 'THUNDER': 'OKC', 
        'MAGIC': 'ORL', 'SIXERS': 'PHI', 'SUNS': 'PHX', 'BLAZERS': 'POR', 'KINGS': 'SAC', 
        'SPURS': 'SAS', 'RAPTORS': 'TOR', 'JAZZ': 'UTA', 'WIZARDS': 'WAS'
    }
    return mapping.get(name, name)

def load_injury_report():
    if not os.path.exists(INJURY_PATH): return {}
    try:
        df = pd.read_csv(INJURY_PATH)
        df.columns = df.columns.str.lower().str.strip()
        player_col = next((c for c in df.columns if c in ['player', 'player_name', 'name', 'athlete']), None)
        team_col = next((c for c in df.columns if c in ['team', 'team_name', 'squad']), None)
        status_col = next((c for c in df.columns if c in ['status', 'description', 'notes']), None)
        
        if not player_col or not team_col: return {}
        if status_col:
            out_players = df[df[status_col].astype(str).str.contains(r'Out', case=False, na=False)]
        else:
            out_players = df 
        
        raw_dict = out_players.groupby(team_col)[player_col].apply(list).to_dict()
        cleaned = {}
        for team, players in raw_dict.items():
            norm_team = normalize_team_name(team)
            norm_players = [normalize_player_name(p) for p in players]
            cleaned[norm_team] = norm_players
        return cleaned
    except: return {}

def get_roster_strength_simulation(team_abbr, injury_list, df_players, debug=False):
    """
    Calculates Roster Strength using RAPTOR Metrics & Weighted Minutes Projections.
    """
    # 1. Filter Data
    roster_all = df_players[df_players['TEAM_ABBREVIATION'] == team_abbr].copy()
    if roster_all.empty: return 0, 110 

    if 'GAME_DATE' in roster_all.columns:
        roster_all['GAME_DATE'] = pd.to_datetime(roster_all['GAME_DATE'])
        max_date = roster_all['GAME_DATE'].max()
        cutoff_date = max_date - timedelta(days=250) 
        roster_all = roster_all[roster_all['GAME_DATE'] >= cutoff_date]
        if roster_all.empty: return 0, 110

    # 2. Calculate RAPTOR SCORE for every game row (Box + On/Off Blend)
    def calc_raptor_live(row):
        # Base Game Score Calculation
        game_score = (
            row['PTS'] + 0.4 * row['FGM'] - 0.7 * row['FGA'] - 0.4 * (row['FTA'] - row['FTM']) + 
            0.7 * row['OREB'] + 0.3 * row['DREB'] + row['STL'] + 0.7 * row['AST'] + 
            0.7 * row['BLK'] - 0.4 * row['PF'] - row['TOV']
        )
        
        # Don't punish garbage time / low sample size
        if row['MIN'] < 5: return game_score
        
        # Approximate Net Rating
        impact = (row['PLUS_MINUS'] / row['MIN']) * 48
        impact = max(min(impact, 30), -30) # Cap at +/- 30
        
        # 50/50 Blend
        return (game_score * 0.5) + (impact * 0.5)

    roster_all['RAPTOR_SCORE'] = roster_all.apply(calc_raptor_live, axis=1)

    # 3. Group by Player to get Form & Projected Minutes
    player_stats = []
    for pid, group in roster_all.groupby('PLAYER_ID'):
        group = group.sort_values('GAME_DATE')
        
        # Minutes Projection (75% Recency Weight)
        season_mins = group['MIN'].mean()
        recent_mins = group.tail(5)['MIN'].mean()
        if pd.isna(recent_mins): recent_mins = 0
        
        proj_mins = (recent_mins * 0.75) + (season_mins * 0.25)
        
        # Current Form (EWMA-10 of RAPTOR)
        curr_form = group['RAPTOR_SCORE'].ewm(span=10, adjust=False).mean().iloc[-1]
        
        name = group['PLAYER_NAME'].iloc[-1]
        player_stats.append({
            'PLAYER_ID': pid,
            'NAME': name,
            'NORM_NAME': normalize_player_name(name),
            'PROJ_MINS': proj_mins,
            'FORM_RAPTOR': curr_form
        })
        
    df_live = pd.DataFrame(player_stats)
    
    # 4. Filter Injuries
    df_live['IS_INJURED'] = df_live['NORM_NAME'].isin(injury_list)
    
    # 5. Determine Active Rotation (Sorted by Projected Minutes)
    active_roster = df_live[~df_live['IS_INJURED']].copy()
    active_roster = active_roster.sort_values('PROJ_MINS', ascending=False)
    
    # Top 8 Players by MINUTES (this catches benchings better than sorting by talent)
    top_8_active = active_roster.head(8)
    active_sum = top_8_active['FORM_RAPTOR'].sum()
    
    # 6. Theoretical Max (Full Strength based on Mins)
    full_roster = df_live.sort_values('PROJ_MINS', ascending=False).head(8)
    theoretical_max = full_roster['FORM_RAPTOR'].sum()
    if theoretical_max == 0: theoretical_max = 110.0
    
    # 7. Catastrophe Logic
    gap = max(0, theoretical_max - active_sum)
    gap_pct = gap / theoretical_max
    
    usage_fill_rate = 0.50 # Default
    collapse_msg = "Normal"
    
    if gap_pct > 0.40:
        usage_fill_rate = 0.10 # Catastrophe
        collapse_msg = "⚠️ CATASTROPHE (>40% Missing)"
    elif gap_pct > 0.20:
        usage_fill_rate = 0.30 # Major
        collapse_msg = "⚠️ Major Loss (>20% Missing)"
        
    usage_bonus = gap * usage_fill_rate
    final_score = active_sum + usage_bonus

    if debug:
        print(f"\n   🔎 ROSTER AUDIT (RAPTOR + MINS): {team_abbr}")
        print(f"      Injuries: {injury_list}")
        print(f"      {'PLAYER':<20} | {'MINS':<5} | {'RAPTOR'}")
        print("      " + "-"*40)
        for _, row in top_8_active.iterrows():
             print(f"      {row['NAME']:<20} | {row['PROJ_MINS']:.1f}  | {row['FORM_RAPTOR']:.1f}")
        print("      " + "-"*40)
        print(f"      Active: {active_sum:.1f} | Theory: {theoretical_max:.1f}")
        print(f"      Missing: {gap_pct:.1%} -> {collapse_msg}")
        print(f"      Final Strength: {final_score:.1f}")

    return final_score, theoretical_max

def get_kelly_bet(prob_win, decimal_odds, implied_prob, bankroll):
    edge = prob_win - implied_prob
    debug_msg = f"       📊 Edge: {prob_win:.1%} - {implied_prob:.1%} = {edge:+.2%}"
    
    if edge > MAX_EDGE_THRESHOLD:
        return 0, 0, edge, f"{debug_msg} -> SKIP (Suspicious)"
    if edge < MIN_EDGE_THRESHOLD:
        return 0, 0, edge, f"{debug_msg} -> SKIP (Low Edge)"

    b = decimal_odds - 1
    p = prob_win
    q = 1.0 - p
    kelly_pct = (b * p - q) / b
    
    if kelly_pct <= 0:
        return 0, 0, edge, f"{debug_msg} -> SKIP (Neg Kelly)"

    raw_bet_pct = kelly_pct * KELLY_FRACTION
    final_bet = bankroll * min(raw_bet_pct, MAX_BET_PCT)
    return final_bet, 0, edge, f"{debug_msg} -> BET ${final_bet:.2f}"

def run_monte_carlo(prob_win, simulations=1000):
    if prob_win <= 0 or prob_win >= 1: return 0
    implied_margin = -np.log(1/prob_win - 1) * 12.0 
    outcomes = np.random.normal(implied_margin, 13.5, simulations)
    return np.mean(outcomes)

# --- MAIN ENGINE ---

def predict():
    print("\n" + "="*60)
    print("🔍 NBA PREDICTION ENGINE - LIVE")
    print("="*60)
    
    try:
        df_stats = pd.read_csv(STATS_PATH)
        df_odds = pd.read_csv(ODDS_PATH)
        df_players = pd.read_csv(PLAYER_PATH) 
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded Resources")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    injuries = load_injury_report()
    predictions = []

    print(f"\n🚀 Analyzing {len(df_odds)} Games...")

    for i, row in df_odds.iterrows():
        home = normalize_team_name(row['HOME_TEAM'])
        away = normalize_team_name(row['AWAY_TEAM'])
        
        print(f"\n🏀 GAME {i+1}: {home} vs {away}")
        print("-" * 40)

        h_hist = df_stats[df_stats['TEAM_ABBREVIATION'] == home].iloc[-1] if not df_stats[df_stats['TEAM_ABBREVIATION'] == home].empty else None
        a_hist = df_stats[df_stats['TEAM_ABBREVIATION'] == away].iloc[-1] if not df_stats[df_stats['TEAM_ABBREVIATION'] == away].empty else None
        
        if h_hist is None or a_hist is None: continue

        h_out = injuries.get(home, [])
        a_out = injuries.get(away, [])
        
        # 🟢 SIMULATION
        h_strength, h_theoretical = get_roster_strength_simulation(home, h_out, df_players, debug=True)
        a_strength, a_theoretical = get_roster_strength_simulation(away, a_out, df_players, debug=True)
        
        # 🟢 ELO CLIFF LOGIC
        def calc_penalty(strength, theory):
            health = strength / theory if theory > 0 else 1.0
            pen = (1 - health) * 250
            if health < 0.60: pen += (1 - health) * 400 # The Cliff
            return pen

        h_penalty = calc_penalty(h_strength, h_theoretical)
        a_penalty = calc_penalty(a_strength, a_theoretical)
        
        h_elo_adj = h_hist['ELO_TEAM'] - h_penalty
        a_elo_adj = a_hist['ELO_TEAM'] - a_penalty

        input_row = {
            'ELO_TEAM': h_elo_adj,
            'ELO_OPP': a_elo_adj,
            'IS_HOME': 1,
            'IS_B2B': h_hist['IS_B2B'],
            'IS_3IN4': h_hist['IS_3IN4'],
            'ROSTER_TALENT_SCORE': h_strength,
        }
        
        rolling_metrics = ['OFF_RTG', 'DEF_RTG', 'PACE', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'ROSTER_TALENT_SCORE']
        for m in rolling_metrics: 
            input_row[f'SMA_20_{m}'] = h_hist.get(f'SMA_20_{m}', 0)
            input_row[f'EWMA_10_{m}'] = h_hist.get(f'EWMA_10_{m}', 0)
            input_row[f'EWMA_5_{m}'] = h_hist.get(f'EWMA_5_{m}', 0)

        try:
            feature_df = pd.DataFrame([input_row])[REQUIRED_FEATURES]
        except KeyError as e:
            print(f"❌ Feature Error: {e}")
            continue

        prob_xgb = model.predict_proba(feature_df)[0][1]
        
        def get_probs(ml):
            if ml < 0: return (1 + 100/abs(ml)), abs(ml)/(abs(ml)+100)
            else: return (1 + ml/100), 100/(ml+100)
            
        h_dec, h_imp = get_probs(row['HOME_ML'])
        a_dec, a_imp = get_probs(row['AWAY_ML'])
        
        h_bet, _, h_edge, h_rsn = get_kelly_bet(prob_xgb, h_dec, h_imp, BANKROLL)
        a_bet, _, a_edge, a_rsn = get_kelly_bet(1-prob_xgb, a_dec, a_imp, BANKROLL)
        
        print(f"\n   💰 DECISION: {prob_xgb:.1%} {home}")
        print(f"      {home}: {h_rsn}")
        print(f"      {away}: {a_rsn}")

        rec = "NO BET"
        vegas_odds = "-"
        pot_profit = "-"
        
        if h_bet > 0: 
            rec = f"BET {home} ${h_bet:.2f}"
            vegas_odds = f"{row['HOME_ML']}" if row['HOME_ML'] < 0 else f"+{row['HOME_ML']}"
            profit = h_bet * (h_dec - 1)
            pot_profit = f"+${profit:.2f}"
            
        if a_bet > 0: 
            rec = f"BET {away} ${a_bet:.2f}"
            vegas_odds = f"{row['AWAY_ML']}" if row['AWAY_ML'] < 0 else f"+{row['AWAY_ML']}"
            profit = a_bet * (a_dec - 1)
            pot_profit = f"+${profit:.2f}"
            
        if prob_xgb > 0.50:
            winner = home
            conf = prob_xgb
        else:
            winner = away
            conf = 1 - prob_xgb
        
        predictions.append({
            'Game': f"{home} vs {away}",
            'Model_Pick': winner,
            'Win_Prob': f"{conf:.1%}",
            'Vegas_Odds': vegas_odds,
            'Edge': f"{max(h_edge, a_edge):+.1%}",
            'Action': rec,
            'Pot_Profit': pot_profit
        })

    if predictions:
        res_df = pd.DataFrame(predictions)
        res_df.to_csv(OUTPUT_PATH, index=False)
        
        print("\n" + "="*95)
        print(f"📋  BETTING CARD FOR {datetime.now().date()}")
        print("="*95)
        print(f"{'MATCHUP':<18} | {'MODEL PICK':<10} | {'WIN%':<6} | {'VEGAS':<6} | {'EDGE':<6} | {'ACTION':<18} | {'PROFIT'}")
        print("-" * 95)
        
        for p in predictions:
            icon = "⚪️"
            if "BET" in p['Action'] and "NO" not in p['Action']:
                icon = "🟢"
            elif "NO BET" in p['Action']:
                icon = "🛑"
                
            print(f"{icon} {p['Game']:<16} | {p['Model_Pick']:<10} | {p['Win_Prob']:<6} | {p['Vegas_Odds']:<6} | {p['Edge']:<6} | {p['Action']:<18} | {p['Pot_Profit']}")
            
        print("="*95 + "\n")
        print(f"✅ Full details saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    predict()