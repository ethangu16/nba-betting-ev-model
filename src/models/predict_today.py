import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
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
    'ROLL_OFF_RTG', 'ROLL_DEF_RTG', 'ROLL_PACE',
    'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
    'ROLL_ROSTER_TALENT_SCORE',
]

# Betting Parameters
BANKROLL = 1000
KELLY_FRACTION = 0.35 
MAX_BET_PCT = 0.05     
MIN_EDGE_THRESHOLD = 0.02 
MAX_EDGE_THRESHOLD = 0.15 
MIN_WIN_PROB = 0.30    
MAX_WIN_PROB = 0.90    
USE_SIGMOID_CALIBRATION = True
CALIBRATION_STRENGTH = 0.3

# --- HELPER FUNCTIONS ---

def normalize_team_name(name):
    name = str(name).upper().strip()
    mapping = {
        'ATLANTA HAWKS': 'ATL', 'BOSTON CELTICS': 'BOS', 'BROOKLYN NETS': 'BKN',
        'CHARLOTTE HORNETS': 'CHA', 'CHICAGO BULLS': 'CHI', 'CLEVELAND CAVALIERS': 'CLE',
        'DALLAS MAVERICKS': 'DAL', 'DENVER NUGGETS': 'DEN', 'DETROIT PISTONS': 'DET',
        'GOLDEN STATE WARRIORS': 'GSW', 'HOUSTON ROCKETS': 'HOU', 'INDIANA PACERS': 'IND',
        'LOS ANGELES CLIPPERS': 'LAC', 'LA CLIPPERS': 'LAC', 'LOS ANGELES LAKERS': 'LAL', 'L.A. LAKERS': 'LAL',
        'MEMPHIS GRIZZLIES': 'MEM', 'MIAMI HEAT': 'MIA', 'MILWAUKEE BUCKS': 'MIL',
        'MINNESOTA TIMBERWOLVES': 'MIN', 'NEW ORLEANS PELICANS': 'NOP', 'NEW YORK KNICKS': 'NYK',
        'OKLAHOMA CITY THUNDER': 'OKC', 'ORLANDO MAGIC': 'ORL', 'PHILADELPHIA 76ERS': 'PHI',
        'PHOENIX SUNS': 'PHX', 'PORTLAND TRAIL BLAZERS': 'POR', 'SACRAMENTO KINGS': 'SAC',
        'SAN ANTONIO SPURS': 'SAS', 'TORONTO RAPTORS': 'TOR', 'UTAH JAZZ': 'UTA',
        'WASHINGTON WIZARDS': 'WAS'
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
            out_players = df[df[status_col].astype(str).str.contains(r'Out|Injured|Game Time|Day-To-Day', case=False, na=False)]
        else:
            out_players = df 
            
        raw_dict = out_players.groupby(team_col)[player_col].apply(list).to_dict()
        return {normalize_team_name(k): v for k, v in raw_dict.items()}
    except: return {}

def get_roster_strength_simulation(team_abbr, injury_list, df_players, debug=False):
    """
    Calculates Roster Strength using LAST 10 GAMES AVERAGE.
    🟢 INCLUDES RECENCY FILTER to remove retired players.
    """
    # 1. Filter for Team
    roster = df_players[df_players['TEAM_ABBREVIATION'] == team_abbr].copy()
    if roster.empty: return 0 

    if 'GAME_DATE' in roster.columns:
        roster['GAME_DATE'] = pd.to_datetime(roster['GAME_DATE'])
        max_date = roster['GAME_DATE'].max()
        cutoff_date = max_date - timedelta(days=90) # Only players active in last 3 months
        
        roster = roster[roster['GAME_DATE'] >= cutoff_date]
        
        if roster.empty:
            if debug: print(f"      ⚠️ No active games found for {team_abbr} in last 90 days.")
            return 0

    # 2. Filter Injuries
    if injury_list:
        roster = roster[~roster['PLAYER_NAME'].isin(injury_list)]

    # 3. Calculate Game Score
    if 'GAME_SCORE' not in roster.columns:
        cols = ['PTS', 'FGM', 'FGA', 'FTA', 'FTM', 'OREB', 'DREB', 'STL', 'AST', 'BLK', 'PF', 'TOV']
        for c in cols:
            if c not in roster.columns: roster[c] = 0
            
        roster['GAME_SCORE'] = (
            roster['PTS'] + 0.4 * roster['FGM'] - 0.7 * roster['FGA'] - 0.4 * (roster['FTA'] - roster['FTM']) + 
            0.7 * roster['OREB'] + 0.3 * roster['DREB'] + roster['STL'] + 0.7 * roster['AST'] + 
            0.7 * roster['BLK'] - 0.4 * roster['PF'] - roster['TOV']
        )

    # 4. Filter for Last 10 Games ONLY (Form)
    # Sort newest first
    roster = roster.sort_values(['PLAYER_ID', 'GAME_DATE'], ascending=[True, False])
    # Take top 10 rows per player
    roster = roster.groupby('PLAYER_ID').head(10)
    
    # 5. Calculate Average Form
    if roster.groupby('PLAYER_ID').ngroups < len(roster):
        player_avgs = roster.groupby('PLAYER_ID')['GAME_SCORE'].mean()
        player_names = roster.groupby('PLAYER_ID')['PLAYER_NAME'].first()
    else:
        player_avgs = roster.set_index('PLAYER_ID')['GAME_SCORE']
        player_names = roster.set_index('PLAYER_ID')['PLAYER_NAME']

    # 6. Select Top 8 Rotation
    top_8_ids = player_avgs.nlargest(8).index
    total_score = player_avgs.loc[top_8_ids].sum()

    # 7. 🔍 DEBUG PRINT
    if debug:
        print(f"\n   🔍 ROSTER AUDIT: {team_abbr}")
        print(f"      Injuries Removed: {injury_list}")
        print("      Active Rotation (Last 90 Days -> Last 10 Avg):")
        print("      " + "-"*40)
        print(f"      {'PLAYER':<20} | {'L10 AVG':<8}")
        print("      " + "-"*40)
        
        for pid in top_8_ids:
            p_name = player_names.loc[pid]
            p_score = player_avgs.loc[pid]
            print(f"      {str(p_name):<20} | {p_score:.1f}")
            
        print("      " + "-"*40)
        print(f"      TOTAL STRENGTH:      {total_score:.1f}")

    return total_score

def calibrate_probability(prob):
    if not USE_SIGMOID_CALIBRATION: return prob
    return 0.5 + (prob - 0.5) * (1 - CALIBRATION_STRENGTH)

def get_kelly_bet(prob_win, decimal_odds, implied_prob, bankroll):
    calibrated_prob = calibrate_probability(prob_win)
    edge = calibrated_prob - implied_prob
    
    debug_msg = f"       📊 Edge: {calibrated_prob:.1%} - {implied_prob:.1%} = {edge:+.2%}"
    
    if edge < MIN_EDGE_THRESHOLD:
        return 0, 0, edge, calibrated_prob, f"{debug_msg} -> SKIP (Low Edge)"
    if calibrated_prob < MIN_WIN_PROB or calibrated_prob > MAX_WIN_PROB:
        return 0, 0, edge, calibrated_prob, f"{debug_msg} -> SKIP (Unsafe Prob)"

    b = decimal_odds - 1
    p = calibrated_prob
    q = 1.0 - p
    kelly_pct = (b * p - q) / b
    
    if kelly_pct <= 0:
        return 0, 0, edge, calibrated_prob, f"{debug_msg} -> SKIP (Neg Kelly)"

    raw_bet_pct = kelly_pct * KELLY_FRACTION
    final_bet = bankroll * min(raw_bet_pct, MAX_BET_PCT)
    
    return final_bet, 0, edge, calibrated_prob, f"{debug_msg} -> BET ${final_bet:.2f}"

# --- MAIN ENGINE ---

def predict():
    print("\n" + "="*60)
    print("🔍 NBA PREDICTION ENGINE - AUDIT MODE")
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

        # A. HISTORY & BASELINE
        h_hist = df_stats[df_stats['TEAM_ABBREVIATION'] == home].iloc[-1] if not df_stats[df_stats['TEAM_ABBREVIATION'] == home].empty else None
        a_hist = df_stats[df_stats['TEAM_ABBREVIATION'] == away].iloc[-1] if not df_stats[df_stats['TEAM_ABBREVIATION'] == away].empty else None
        
        if h_hist is None or a_hist is None: continue

        # B. STRENGTH CALCULATION
        h_out = injuries.get(home, [])
        a_out = injuries.get(away, [])
        
        h_strength = get_roster_strength_simulation(home, h_out, df_players, debug=True)
        a_strength = get_roster_strength_simulation(away, a_out, df_players, debug=True)
        
        h_avg = h_hist.get('ROLL_ROSTER_TALENT_SCORE', 110)
        a_avg = a_hist.get('ROLL_ROSTER_TALENT_SCORE', 110)
        if h_avg <= 0: h_avg = 110
        if a_avg <= 0: a_avg = 110
        
        h_health = h_strength / h_avg
        a_health = a_strength / a_avg
        
        h_elo_orig = h_hist['ELO_TEAM']
        a_elo_orig = a_hist['ELO_TEAM']
        
        # Calculate Penalty
        h_penalty = (1 - h_health) * 250
        a_penalty = (1 - a_health) * 250
        
        h_elo_adj = h_elo_orig - h_penalty
        a_elo_adj = a_elo_orig - a_penalty

        print(f"\n   🏥 HEALTH & ELO LOGIC:")
        print(f"      {home}: {h_strength:.1f}/{h_avg:.1f} ({h_health:.1%}) -> Elo {h_elo_orig:.0f} to {h_elo_adj:.0f}")
        print(f"      {away}: {a_strength:.1f}/{a_avg:.1f} ({a_health:.1%}) -> Elo {a_elo_orig:.0f} to {a_elo_adj:.0f}")

        # C. PREDICT
        input_row = {
            'ELO_TEAM': h_elo_adj,
            'ELO_OPP': a_elo_adj,
            'IS_HOME': 1,
            'IS_B2B': h_hist['IS_B2B'],
            'IS_3IN4': h_hist['IS_3IN4'],
            'ROLL_OFF_RTG': h_hist['ROLL_OFF_RTG'],
            'ROLL_DEF_RTG': h_hist['ROLL_DEF_RTG'],
            'ROLL_PACE': h_hist['ROLL_PACE'],
            'ROLL_EFG_PCT': h_hist['ROLL_EFG_PCT'],
            'ROLL_TOV_PCT': h_hist['ROLL_TOV_PCT'],
            'ROLL_ORB_PCT': h_hist['ROLL_ORB_PCT'],
            'ROLL_FTR': h_hist['ROLL_FTR'],
            'ROLL_ROSTER_TALENT_SCORE': h_hist['ROLL_ROSTER_TALENT_SCORE']
        }
        
        try:
            feature_df = pd.DataFrame([input_row])[REQUIRED_FEATURES]
        except KeyError as e:
            print(f"❌ Feature Error: {e}")
            continue

        prob_xgb = model.predict_proba(feature_df)[0][1]
        
        # D. ODDS
        def get_probs(ml):
            if ml < 0: return (1 + 100/abs(ml)), abs(ml)/(abs(ml)+100)
            else: return (1 + ml/100), 100/(ml+100)
            
        h_dec, h_imp = get_probs(row['HOME_ML'])
        a_dec, a_imp = get_probs(row['AWAY_ML'])
        
        h_bet, _, h_edge, _, h_rsn = get_kelly_bet(prob_xgb, h_dec, h_imp, BANKROLL)
        a_bet, _, a_edge, _, a_rsn = get_kelly_bet(1-prob_xgb, a_dec, a_imp, BANKROLL)
        
        print(f"\n   💰 DECISION: {prob_xgb:.1%} {home}")
        print(f"      {home}: {h_rsn}")
        print(f"      {away}: {a_rsn}")

        rec = "NO BET"
        if h_bet > 0: rec = f"BET {home} ${h_bet:.2f}"
        if a_bet > 0: rec = f"BET {away} ${a_bet:.2f}"
        
        predictions.append({
            'Game': f"{home} vs {away}",
            'Model_Prob': f"{prob_xgb:.1%}",
            'Edge': f"{max(h_edge, a_edge):+.1%}",
            'Recommendation': rec
        })

    if predictions:
        res_df = pd.DataFrame(predictions)
        res_df.to_csv(OUTPUT_PATH, index=False)
        
        # 🟢 FINAL SUMMARY TABLE
        print("\n" + "="*80)
        print(f"📋  BETTING CARD FOR {datetime.now().date()}")
        print("="*80)
        print(f"{'MATCHUP':<25} | {'WIN%':<8} | {'EDGE':<8} | {'ACTION'}")
        print("-" * 80)
        
        for p in predictions:
            icon = "⚪️"
            if "BET" in p['Recommendation'] and "NO" not in p['Recommendation']:
                icon = "🟢"
            elif "NO BET" in p['Recommendation']:
                icon = "🛑"
                
            print(f"{icon} {p['Game']:<23} | {p['Model_Prob']:<8} | {p['Edge']:<8} | {p['Recommendation']}")
            
        print("="*80 + "\n")
        print(f"✅ Full details saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    predict()