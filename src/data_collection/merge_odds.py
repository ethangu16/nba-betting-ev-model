import pandas as pd
import numpy as np

# --- CONFIGURATION ---
STATS_PATH = 'data/raw/nba_games_advanced.csv'
ODDS_PATH = 'data/odds/nba_2008-2025.csv'
OUTPUT_PATH = 'data/processed/nba_final_dataset.csv'

# TEAM MAPPING
TEAM_MAP = {
    'atl': 'ATL', 'bkn': 'BKN', 'nj': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'ch': 'CHA',
    'chi': 'CHI', 'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW', 'gsw': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM', 'mia': 'MIA', 'mil': 'MIL',
    'min': 'MIN', 'no': 'NOP', 'nop': 'NOP', 'noh': 'NOP', 'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC',
    'sea': 'OKC', 'orl': 'ORL', 'phi': 'PHI', 'phx': 'PHX', 'pho': 'PHX', 'por': 'POR', 'sac': 'SAC',
    'sa': 'SAS', 'sas': 'SAS', 'tor': 'TOR', 'utah': 'UTA', 'uta': 'UTA', 'wsh': 'WAS', 'was': 'WAS'
}

def load_and_reshape_odds():
    print(f"Loading odds from {ODDS_PATH}...")
    try:
        df = pd.read_csv(ODDS_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {ODDS_PATH}")
        return None

    # 1. Standardize Date
    if 'date' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['Date'])
    
    # 2. Identify Moneyline Columns Dynamically
    ml_home_col = None
    ml_away_col = None
    
    candidates_home = ['moneyline_home', 'ML_Home', 'Home Odds', 'Home Moneyline', 'home_ml']
    candidates_away = ['moneyline_away', 'ML_Away', 'Away Odds', 'Away Moneyline', 'away_ml']
    
    for c in candidates_home:
        if c in df.columns:
            ml_home_col = c
            break
            
    for c in candidates_away:
        if c in df.columns:
            ml_away_col = c
            break
            
    # SOFT FAIL: If columns missing, don't crash. Just warn.
    if not ml_home_col or not ml_away_col:
        print("⚠️ Warning: Could not find Moneyline columns. Creating empty placeholders.")
    else:
        print(f"Using Moneyline Columns: Home='{ml_home_col}', Away='{ml_away_col}'")

    # 3. Reshape: Split 'Away' and 'Home' into separate rows
    print("Reshaping odds data...")
    
    # Create Home Rows
    home_df = df.copy()
    home_df['TEAM_ABBREVIATION'] = home_df['home'].map(TEAM_MAP)
    home_df['OPP_TEAM_ABBREVIATION'] = home_df['away'].map(TEAM_MAP)
    
    # Use dynamic column if it exists, else NaN
    if ml_home_col:
        home_df['MONEYLINE'] = home_df[ml_home_col] 
    else:
        home_df['MONEYLINE'] = np.nan

    # Handle Spread
    if 'spread' in df.columns:
        home_df['SPREAD'] = home_df['spread'] * -1 
    else:
        home_df['SPREAD'] = np.nan
        
    home_df['IS_HOME_ODDS'] = 1
    
    # Create Away Rows
    away_df = df.copy()
    away_df['TEAM_ABBREVIATION'] = away_df['away'].map(TEAM_MAP)
    away_df['OPP_TEAM_ABBREVIATION'] = away_df['home'].map(TEAM_MAP)
    
    if ml_away_col:
        away_df['MONEYLINE'] = away_df[ml_away_col]
    else:
        away_df['MONEYLINE'] = np.nan
    
    if 'spread' in df.columns:
        away_df['SPREAD'] = away_df['spread']
    else:
        away_df['SPREAD'] = np.nan
        
    away_df['IS_HOME_ODDS'] = 0

    # Combine
    odds_reshaped = pd.concat([home_df, away_df], ignore_index=True)
    
    # Keep useful columns
    cols_to_keep = ['GAME_DATE', 'TEAM_ABBREVIATION', 'MONEYLINE', 'SPREAD']
    if 'total' in df.columns:
        cols_to_keep.append('total')
        odds_reshaped['total'] = df['total']
        
    odds_reshaped = odds_reshaped[cols_to_keep]
    
    # --- THE FIX: DROP ROWS WHERE MONEYLINE IS MISSING ---
    # This "ignores" any game that doesn't have valid odds
    before_drop = len(odds_reshaped)
    odds_reshaped = odds_reshaped.dropna(subset=['TEAM_ABBREVIATION', 'MONEYLINE'])
    after_drop = len(odds_reshaped)
    
    if before_drop > after_drop:
        print(f"⚠️ Ignored {before_drop - after_drop} rows due to missing Moneyline/Teams.")
        
    return odds_reshaped

def merge_data():
    print(f"Loading stats from {STATS_PATH}...")
    try:
        df_stats = pd.read_csv(STATS_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {STATS_PATH}")
        return

    df_stats['GAME_DATE'] = pd.to_datetime(df_stats['GAME_DATE'])
    
    df_odds = load_and_reshape_odds()
    if df_odds is None: return
    
    print("Merging datasets...")
    merged_df = pd.merge(
        df_stats, 
        df_odds, 
        on=['GAME_DATE', 'TEAM_ABBREVIATION'], 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("❌ Merge resulted in 0 rows.")
    else:
        print(f"✅ Merge Complete! Final Rows: {len(merged_df)}")
        merged_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_data()