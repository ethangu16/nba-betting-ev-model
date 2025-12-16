import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime

# --- CONFIGURATION ---
# 1. Your trained model
MODEL_PATH = 'models/nba_xgb_model.json' 
# 2. Your up-to-date stats (for calculating rolling averages)
STATS_PATH = 'data/raw/nba_games_advanced.csv'
# 3. Your Live Odds file (Update this filename to match what you just downloaded!)
ODDS_PATH = 'data/odds/live_odds.csv' 

# 4. Same Elo params as training
K_FACTOR = 25
HOME_ADV = 60

def get_latest_stats(team_abbr, df_stats):
    """
    Grabs the last 10 games for a team to calculate rolling averages.
    """
    # Filter for this team
    team_games = df_stats[df_stats['TEAM_ABBREVIATION'] == team_abbr].sort_values('GAME_DATE')
    
    if len(team_games) < 10:
        print(f"⚠️ Warning: {team_abbr} has fewer than 10 games. Stats might be noisy.")
    
    # Calculate Rolling Averages (Last 10)
    last_10 = team_games.iloc[-10:]
    
    stats = {}
    stats['ROLL_PTS'] = last_10['PTS'].mean()
    stats['ROLL_EFG_PCT'] = last_10['EFG_PCT'].mean()
    stats['ROLL_TOV_PCT'] = last_10['TOV_PCT'].mean()
    stats['ROLL_ORB_PCT'] = last_10['ORB_PCT'].mean()
    stats['ROLL_FTR'] = last_10['FTR'].mean()
    stats['ROLL_PLUS_MINUS'] = last_10['PLUS_MINUS'].mean() # Using raw PM as proxy
    
    # Get Last Game Date for Rest Days calculation
    last_game_date = pd.to_datetime(team_games.iloc[-1]['GAME_DATE'])
    stats['LAST_GAME_DATE'] = last_game_date
    
    return stats

def get_current_elo(df_stats):
    """
    Re-runs the Elo calculation from the start of history to get 
    the CURRENT rating for every team.
    """
    print("re-calculating current Elo ratings...")
    team_elos = {team: 1500 for team in df_stats['TEAM_ABBREVIATION'].unique()}
    
    df_sorted = df_stats.sort_values('GAME_DATE')
    
    for _, row in df_sorted.iterrows():
        team = row['TEAM_ABBREVIATION']
        opp = row['MATCHUP'].split(' ')[-1] # Crude extract if OPP_ABBREVIATION missing
        
        # If we have explicit OPP column use it
        if 'OPP_ABBREVIATION' in row:
            opp = row['OPP_ABBREVIATION']

        r_team = team_elos.get(team, 1500)
        r_opp = team_elos.get(opp, 1500)
        
        # Simple Elo Update (Fast version for Inference)
        # We don't need perfect history, just the end state
        home_boost = HOME_ADV if 'vs.' in row['MATCHUP'] else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        actual_win = 1 if row['WL'] == 'W' else 0
        
        # MOV Multiplier (Simplified)
        mov = abs(row['PLUS_MINUS']) if not pd.isna(row['PLUS_MINUS']) else 0
        mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * (dr if actual_win else -dr))
        
        shift = K_FACTOR * mult * (actual_win - prob_win)
        
        team_elos[team] = r_team + shift
        team_elos[opp] = r_opp - shift
        
    return team_elos

def predict():
    # 1. Load Everything
    print("Loading Data...")
    df_stats = pd.read_csv(STATS_PATH)
    df_odds = pd.read_csv(ODDS_PATH)
    
    # Load Model
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    except:
        model = joblib.load('models/nba_xgb_model.joblib')
    
    # 2. Get Current State (Elo & Stats)
    current_elos = get_current_elo(df_stats)
    
    print(f"\n--- GENERATING PREDICTIONS FOR {len(df_odds)} GAMES ---")
    
    predictions = []
    
    for index, row in df_odds.iterrows():
        home_team = row['HOME_TEAM'] # e.g., 'Boston Celtics'
        away_team = row['AWAY_TEAM'] # e.g., 'Miami Heat'
        
        # Map Full Names to Abbreviations (Reuse your dictionary from merge script!)
        # Simplified map for example:
        name_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        
        home_abbr = name_map.get(home_team)
        away_abbr = name_map.get(away_team)
        
        if not home_abbr or not away_abbr:
            print(f"Skipping {home_team} vs {away_team} (Mapping Error)")
            continue
            
        # 3. Build Feature Vector for HOME Team Perspective
        # (Model was trained on Team vs Opp, so we predict for Home Team)
        
        # A. ELO
        elo_home = current_elos.get(home_abbr, 1500)
        elo_away = current_elos.get(away_abbr, 1500)
        
        # B. Rolling Stats (Home)
        home_stats = get_latest_stats(home_abbr, df_stats)
        
        # C. Rest Days (Today - Last Game)
        today = pd.to_datetime(datetime.now().date())
        rest_days = (today - home_stats['LAST_GAME_DATE']).days
        
        # D. Travel (Simplified: Assume 0 for home unless returning from road trip)
        # For a pro model, you'd calculate this properly. For now, set to 0.
        travel = 0 
        
        # Construct DataFrame Row (Order MUST match training!)
        # Features: ['ELO_TEAM', 'ELO_OPP', 'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME', 'ROLL_PTS', ...]
        feature_row = pd.DataFrame([{
            'ELO_TEAM': elo_home,
            'ELO_OPP': elo_away,
            'REST_DAYS': rest_days,
            'TRAVEL_MILES': travel,
            'IS_HOME': 1, # We are predicting from Home perspective
            'ROLL_PTS': home_stats['ROLL_PTS'],
            'ROLL_EFG_PCT': home_stats['ROLL_EFG_PCT'],
            'ROLL_TOV_PCT': home_stats['ROLL_TOV_PCT'],
            'ROLL_ORB_PCT': home_stats['ROLL_ORB_PCT'],
            'ROLL_FTR': home_stats['ROLL_FTR'],
            'ROLL_PLUS_MINUS': home_stats['ROLL_PLUS_MINUS']
        }])
        
        # 4. Predict
#         win_prob = model.predict_proba(feature_row)[0][1] # Prob of Class 1 (Win)
        
#         # 5. EV Calculation
# # ... (Previous code remains the same: Feature building & Prediction) ...
        # 4. Predict
        prob_home = model.predict_proba(feature_row)[0][1]
        prob_away = 1.0 - prob_home
        
        # 5. EV Calculation
        # Helper to convert American Odds -> Implied Prob
        def get_implied(ml):
            if pd.isna(ml) or ml == 0: return 0.5 # Safety for bad data
            if ml < 0: return (-ml) / (-ml + 100)
            return 100 / (ml + 100)

        # Get Odds
        home_ml = row.get('HOME_ML')
        away_ml = row.get('AWAY_ML')
        
        implied_home = get_implied(home_ml)
        implied_away = get_implied(away_ml)
        
        # Calculate Edges
        edge_home = prob_home - implied_home
        edge_away = prob_away - implied_away
        
        # --- DECISION LOGIC ---
        THRESHOLD = 0.02 # 2% Edge required
        
        if edge_home > THRESHOLD:
            rec = f"BET HOME ({home_abbr})"
            display_odds = home_ml
            display_implied = implied_home
            display_edge = edge_home
        elif edge_away > THRESHOLD:
            rec = f"BET AWAY ({away_abbr})"
            display_odds = away_ml
            display_implied = implied_away
            display_edge = edge_away
        else:
            rec = "NO BET"
            # If no bet, default to showing Home info so you see the "Main Line"
            display_odds = home_ml
            display_implied = implied_home
            display_edge = 0.0 # No edge realized
            
        predictions.append({
            'Game': f"{home_abbr} vs {away_abbr}",
            'Model_Home_Win%': f"{prob_home:.1%}",
            'Implied_Win%': f"{display_implied:.1%}", # <--- NEW COLUMN
            'Edge': f"{display_edge:.1%}",
            'Rec': rec,
            'Odds': int(display_odds) if not pd.isna(display_odds) else "N/A" # <--- FIXED (No more 0)
        })
        
        # predictions.append({
        #     'Game': f"{home_abbr} vs {away_abbr}",
        #     'Model_Prob': f"{win_prob:.1%}",
        #     'Implied_Prob': f"{implied:.1%}",
        #     'Edge': f"{edge:.1%}",
        #     'Rec': "BET HOME" if edge > 0.02 else "NO BET", # 2% Threshold
        #     'Odds': home_ml
        # })

    # Output
    print("\n--- TODAY'S PREDICTIONS ---")
    results = pd.DataFrame(predictions)
    print(results)
    
    # Save
    results.to_csv('results/todays_picks.csv', index=False)

if __name__ == "__main__":
    predict()