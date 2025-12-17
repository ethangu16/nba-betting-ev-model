import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys

# --- CONFIGURATION ---
MODEL_PATH = 'models/nba_xgb_model.joblib' 
STATS_PATH = 'data/processed/nba_model.csv'        # Created by engineer.py
ODDS_PATH = 'data/odds/live_odds.csv'              # Must contain today's odds
PLAYER_PATH = 'data/processed/processed_player.csv' # Created by create_talent_pool.py
INJURY_PATH = 'data/raw/espn_injuries_current.csv'   # Created by scrape_espn_injuries_v3.py
OUTPUT_PATH = 'results/todays_bets.csv'

# --- NAME MAPPING (Global) ---
# Maps Odds/Injury names -> Stats abbreviations
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
    """
    Reads injury CSV and filters strictly for status='Out'.
    """
    if not os.path.exists(INJURY_PATH):
        print(f"⚠️ Warning: No injury file found at {INJURY_PATH}")
        return {}
        
    try:
        df = pd.read_csv(INJURY_PATH)
        
        # FIX: Simple String Check. If 'status' contains "Out", they are out.
        out_players = df[df['description'].astype(str).str.lower().str.contains('out')]
        
        # Create Dictionary: {'Boston Celtics': ['Jayson Tatum', ...]}
        # We assume column names from scrape_espn_injuries_v3.py (TEAM, PLAYER)
        raw_dict = out_players.groupby('team')['player_name'].apply(list).to_dict()
        
        final_dict = {}
        for team_name, players in raw_dict.items():
            # Convert "Atlanta Hawks" -> "ATL" using our map
            abbr = NAME_MAP.get(team_name, team_name) 
            final_dict[abbr] = players
            
        print(f"✅ Loaded Injury Report: {len(final_dict)} teams have players OUT.")
        return final_dict
        
    except Exception as e:
        print(f"❌ Error reading injury file: {e}")
        return {}

def get_roster_strength_simulation(team_abbr, injury_list, df_players):
    """
    Simulates the strength of the team TONIGHT by removing injured players
    and redistributing their minutes to the bench.
    """
    # Filter for this team
    roster = df_players[df_players['TEAM_ABBREVIATION'] == team_abbr].copy()
    
    if roster.empty:
        # If we have no player data for a team (rare), return 0 so we skip the prediction safely
        return 0 

    # 1. Remove Injured Players
    # Note: df_players has 'PLAYER_NAME', injury_list has names from ESPN.
    # We use .isin() for exact matches. 
    roster = roster[~roster['PLAYER_NAME'].isin(injury_list)]
    
    # 2. Calculate Productivity Per Minute (PPM)
    # Avoid division by zero
    roster['MIN_AVG'] = roster['MIN_AVG'].replace(0, 1)
    roster['PPM'] = roster['GAME_SCORE_AVG'] / roster['MIN_AVG']
    
    # 3. Select Rotation (Top 9 remaining players)
    rotation = roster.sort_values('MIN_AVG', ascending=False).head(9).copy()
    
    # 4. Redistribute Minutes (The "240-Minute Container")
    total_avg_minutes = rotation['MIN_AVG'].sum()
    target_minutes = 240
    
    # If the remaining players usually only play 180 mins total, we scale them up
    scaling_factor = target_minutes / total_avg_minutes if total_avg_minutes > 0 else 1.0
    
    rotation['PROJ_MIN'] = rotation['MIN_AVG'] * scaling_factor
    
    # Cap minutes at 42 (Human limit) so a bench warmer doesn't play 48 mins
    rotation['PROJ_MIN'] = rotation['PROJ_MIN'].clip(upper=42)
    
    # 5. Calculate Final Score
    rotation['PROJ_SCORE'] = rotation['PPM'] * rotation['PROJ_MIN']
    
    # Efficiency Tax: Teams play slightly worse when forced to stretch lineups
    tax = 0.95 
    return rotation['PROJ_SCORE'].sum() * tax

def get_implied_prob(moneyline):
    if pd.isna(moneyline) or moneyline == 0: return 0.5
    if moneyline < 0: return (-moneyline) / (-moneyline + 100)
    return 100 / (moneyline + 100)

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

    # Load Injuries
    current_injuries = load_injury_report()
    predictions = []
    
    print(f"\n--- 2. PREDICTING {len(df_odds)} GAMES ---")
    
    # Helper to find latest stats for a team
    def get_last_stats(abbr):
        subset = df_stats[df_stats['TEAM_ABBREVIATION'] == abbr]
        if subset.empty: return None
        return subset.iloc[-1]

    for index, row in df_odds.iterrows():
        # Map Full Names to Abbreviations
        home_raw = row.get('HOME_TEAM')
        away_raw = row.get('AWAY_TEAM')
        
        home_abbr = NAME_MAP.get(home_raw, home_raw) # "Boston Celtics" -> "BOS"
        away_abbr = NAME_MAP.get(away_raw, away_raw)
        
        # Get Historical Stats (Elo, Rolling Points, etc.)
        home_hist = get_last_stats(home_abbr)
        if home_hist is None:
            print(f"Skipping {home_abbr} (Name not found in Stats file)")
            continue

        # --- THE CORE LOGIC: SIMULATE TONIGHT'S ROSTER ---
        h_out = current_injuries.get(home_abbr, [])
        h_strength = get_roster_strength_simulation(home_abbr, h_out, df_players)
        
        a_out = current_injuries.get(away_abbr, [])
        a_strength = get_roster_strength_simulation(away_abbr, a_out, df_players)
        
        if h_strength == 0 or a_strength == 0:
            print(f"Skipping {home_abbr} vs {away_abbr} (Player data missing)")
            continue

        # Build the exact same Feature Row used in training
        # IMPORTANT: 'ROLL_ROSTER_TALENT_SCORE' is swapped with our simulation
        feature_row = pd.DataFrame([{
            'ELO_TEAM': home_hist['ELO_TEAM'],
            'ELO_OPP': 1500, # Ideally fetch away_hist['ELO_TEAM']
            'REST_DAYS': 1,
            'TRAVEL_MILES': 0,
            'IS_HOME': 1,
            'ROLL_PTS': home_hist['ROLL_PTS'],
            'ROLL_EFG_PCT': home_hist['ROLL_EFG_PCT'],
            'ROLL_TOV_PCT': home_hist['ROLL_TOV_PCT'],
            'ROLL_ORB_PCT': home_hist['ROLL_ORB_PCT'],
            'ROLL_FTR': home_hist['ROLL_FTR'],
            'ROLL_PLUS_MINUS': home_hist['ROLL_PLUS_MINUS'],
            'ROLL_ROSTER_TALENT_SCORE': h_strength 
        }])
        
        # Run Model
        prob_home = model.predict_proba(feature_row)[0][1]
        
        # Calculate Value
        home_ml = row.get('HOME_ML', 0)
        away_ml = row.get('AWAY_ML', 0)
        
        implied_home = get_implied_prob(home_ml)
        implied_away = get_implied_prob(away_ml)
        
        edge_home = prob_home - implied_home
        edge_away = (1.0 - prob_home) - implied_away
        
        # Decision
        rec = "NO BET"
        THRESHOLD = 0.02
        
        if edge_home > THRESHOLD:
            rec = f"BET HOME"
        elif edge_away > THRESHOLD:
            rec = f"BET AWAY"
            
        predictions.append({
            'Game': f"{home_abbr} vs {away_abbr}",
            'Home_Win%': f"{prob_home:.1%}",
            'Edge_Home': f"{edge_home:.1%}",
            'Edge_Away': f"{edge_away:.1%}",
            'Rec': rec,
            'H_Odds': home_ml,
            'A_Odds': away_ml,
            'Injuries': f"{len(h_out)} H / {len(a_out)} A"
        })

    # --- 3. OUTPUT ---
    print("\n" + "="*60)
    print("TODAY'S BETTING CARD")
    print("="*60)
    
    if not predictions:
        print("❌ No valid games found. Check mappings or data files.")
    else:
        results = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        results.to_csv(OUTPUT_PATH, index=False)
        
        # Pretty Print
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        view_cols = ['Game', 'Rec', 'Home_Win%', 'Edge_Home', 'Edge_Away', 'Injuries']
        print(results[view_cols].to_string(index=False))
        print("\n✅ Results saved to", OUTPUT_PATH)

if __name__ == "__main__":
    predict()