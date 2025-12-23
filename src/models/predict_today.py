import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys

# --- CONFIGURATION ---
MODEL_PATH = 'models/nba_xgb_model.joblib' 
STATS_PATH = 'data/processed/nba_model.csv'        
ODDS_PATH = 'data/odds/live_odds.csv'              
PLAYER_PATH = 'data/processed/processed_player.csv' 
INJURY_PATH = 'data/raw/espn_injuries_current.csv'   
OUTPUT_PATH = 'results/todays_bets.csv'

# --- NAME MAPPING ---
NAME_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "L.A. Lakers": "LAL", 
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", 
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", 
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", 
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# --- 1. HELPER FUNCTIONS ---

def load_injury_report():
    if not os.path.exists(INJURY_PATH):
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

        if status_col not in df.columns: return {}

        out_players = df[df[status_col].astype(str).str.lower().str.contains('out')]
        
        team_col = 'team' if 'team' in df.columns else 'team_name'
        player_col = 'player_name' if 'player_name' in df.columns else 'player'
        
        raw_dict = out_players.groupby(team_col)[player_col].apply(list).to_dict()
        final_dict = {NAME_MAP.get(k, k): v for k, v in raw_dict.items()}
        print(f"✅ Loaded Injury Report: {len(final_dict)} teams have players OUT.")
        return final_dict
    except:
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

def run_monte_carlo(home_team, away_team, df_players, n_sims=1000):
    h_roster = df_players[df_players['TEAM_ABBREVIATION'] == home_team]
    a_roster = df_players[df_players['TEAM_ABBREVIATION'] == away_team]
    
    if h_roster.empty or a_roster.empty: return 0.5
    
    h_avg = h_roster['GAME_SCORE_AVG'].values
    h_std = h_roster.get('GAME_SCORE_STD', pd.Series([8]*len(h_roster))).fillna(8).values
    
    a_avg = a_roster['GAME_SCORE_AVG'].values
    a_std = a_roster.get('GAME_SCORE_STD', pd.Series([8]*len(a_roster))).fillna(8).values
    
    h_sims = np.random.normal(h_avg[:, None], h_std[:, None], (len(h_avg), n_sims))
    a_sims = np.random.normal(a_avg[:, None], a_std[:, None], (len(a_avg), n_sims))
    
    h_score = h_sims.sum(axis=0) + 3 # +3 Home Court
    a_score = a_sims.sum(axis=0)
    
    return (h_score > a_score).sum() / n_sims

def get_kelly_bet(prob_win, decimal_odds, bankroll=1000):
    b = decimal_odds - 1
    p = prob_win
    q = 1.0 - p
    f = (b * p - q) / b
    
    if f > 0:
        return bankroll * f * 0.35 # 35% Kelly Fraction
    else:
        return 0.0

# --- 2. MAIN EXECUTION ---

def predict():
    print("\n--- 1. LOADING DATA ---")
    try:
        df_stats = pd.read_csv(STATS_PATH)
        df_odds = pd.read_csv(ODDS_PATH)
        df_players = pd.read_csv(PLAYER_PATH)
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"❌ Critical Error: {e}")
        return

    current_injuries = load_injury_report()
    predictions = []
    
    print(f"\n--- 2. PREDICTING {len(df_odds)} GAMES ---")
    
    def get_last_stats(abbr):
        subset = df_stats[df_stats['TEAM_ABBREVIATION'] == abbr]
        if subset.empty: return None
        return subset.iloc[-1]

    for index, row in df_odds.iterrows():
        home_raw, away_raw = row.get('HOME_TEAM'), row.get('AWAY_TEAM')
        home_abbr = NAME_MAP.get(home_raw, home_raw)
        away_abbr = NAME_MAP.get(away_raw, away_raw)
        
        home_hist = get_last_stats(home_abbr)
        away_hist = get_last_stats(away_abbr) 
        
        if home_hist is None or away_hist is None:
            continue

        # --- SIMULATION ---
        h_out = current_injuries.get(home_abbr, [])
        h_strength = get_roster_strength_simulation(home_abbr, h_out, df_players)
        
        a_out = current_injuries.get(away_abbr, [])
        a_strength = get_roster_strength_simulation(away_abbr, a_out, df_players)
        
        h_avg_str = home_hist['ROLL_ROSTER_TALENT_SCORE'] if home_hist['ROLL_ROSTER_TALENT_SCORE'] > 10 else 150
        a_avg_str = away_hist['ROLL_ROSTER_TALENT_SCORE'] if away_hist['ROLL_ROSTER_TALENT_SCORE'] > 10 else 150

        h_health = h_strength / h_avg_str
        a_health = a_strength / a_avg_str
        
        # INJURY PENALTY (250 pts)
        h_elo_adj = home_hist['ELO_TEAM'] - ((1 - h_health) * 250)
        a_elo_adj = away_hist['ELO_TEAM'] - ((1 - a_health) * 250)

        feature_row = pd.DataFrame([{
            'ELO_TEAM': h_elo_adj,
            'ELO_OPP': a_elo_adj,
            'IS_HOME': 1,
            'IS_B2B': home_hist.get('IS_B2B', 0),
            'IS_3IN4': home_hist.get('IS_3IN4', 0),
            'ROLL_OFF_RTG': home_hist['ROLL_OFF_RTG'], 
            'ROLL_DEF_RTG': home_hist['ROLL_DEF_RTG'], 
            'ROLL_PACE': home_hist['ROLL_PACE'],       
            'ROLL_EFG_PCT': home_hist['ROLL_EFG_PCT'],
            'ROLL_TOV_PCT': home_hist['ROLL_TOV_PCT'],
            'ROLL_ORB_PCT': home_hist['ROLL_ORB_PCT'],
            'ROLL_FTR': home_hist['ROLL_FTR'],
            'ROLL_ROSTER_TALENT_SCORE': h_strength
        }])
        
        # 1. Predictions
        prob_home_xgb = model.predict_proba(feature_row)[0][1]
        prob_home_mc = run_monte_carlo(home_abbr, away_abbr, df_players)
        
        final_prob_home = (0.7 * prob_home_xgb) + (0.3 * prob_home_mc)
        final_prob_away = 1.0 - final_prob_home  # Away Probability
        
        # 2. Odds Calculation
        home_ml = row.get('HOME_ML', 0)
        away_ml = row.get('AWAY_ML', 0)
        
        dec_home = (home_ml/100 + 1) if home_ml > 0 else (100/abs(home_ml) + 1)
        implied_home = 1 / dec_home
        
        dec_away = (away_ml/100 + 1) if away_ml > 0 else (100/abs(away_ml) + 1)
        implied_away = 1 / dec_away
        
        # 3. Calculate Bets (Check BOTH sides)
        bet_home = get_kelly_bet(final_prob_home, dec_home, 1000)
        bet_away = get_kelly_bet(final_prob_away, dec_away, 1000)
        
        # 4. Decision Logic
        rec = "NO BET"
        edge_display = 0.0
        final_prob_display = final_prob_home
        implied_display = implied_home
        
        if bet_home > 0:
            rec = f"BET {home_abbr} ${bet_home:.2f}" # <--- UPDATED TO SHOW TEAM NAME
            edge_display = final_prob_home - implied_home
            final_prob_display = final_prob_home
            implied_display = implied_home
            
        elif bet_away > 0:
            rec = f"BET {away_abbr} ${bet_away:.2f}" # <--- UPDATED TO SHOW TEAM NAME
            edge_display = final_prob_away - implied_away
            final_prob_display = final_prob_away
            implied_display = implied_away

        if rec == "NO BET":
             edge_display = final_prob_home - implied_home

        predictions.append({
            'Game': f"{home_abbr} vs {away_abbr}",
            'Win_Prob': f"{final_prob_display:.1%}",
            'Implied': f"{implied_display:.1%}",
            'Edge': f"{edge_display:+.1%}",
            'Kelly': rec,
            'Injuries': f"{len(h_out)} H / {len(a_out)} A"
        })

    # --- OUTPUT ---
    if predictions:
        results = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        results.to_csv(OUTPUT_PATH, index=False)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n" + "="*80)
        print("TODAY'S BETTING CARD (KELLY 0.35x)")
        print("="*80)
        print(results[['Game', 'Kelly', 'Win_Prob', 'Implied', 'Edge', 'Injuries']].to_string(index=False))
        print(f"\n✅ Saved to {OUTPUT_PATH}")
    else:
        print("❌ No games found.")

if __name__ == "__main__":
    predict()