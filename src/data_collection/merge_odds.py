import pandas as pd
import numpy as np

# --- CONFIGURATION ---
STATS_PATH = 'data/raw/nba_games_advanced.csv'
ODDS_PATH = 'data/odds/nba_2008-2025.csv'
OUTPUT_PATH = 'data/processed/nba_final_dataset.csv'

# UPDATED MAPPING: Handles lowercase & non-standard abbreviations from your file
TEAM_MAP = {
    # File Name -> API Name
    'atl': 'ATL', 
    'bkn': 'BKN', 'nj': 'BKN', # Handle old NJ Nets
    'bos': 'BOS', 
    'cha': 'CHA', 'ch': 'CHA',
    'chi': 'CHI', 
    'cle': 'CLE', 
    'dal': 'DAL', 
    'den': 'DEN', 
    'det': 'DET', 
    'gs': 'GSW', 'gsw': 'GSW', # Fix: gs -> GSW
    'hou': 'HOU', 
    'ind': 'IND', 
    'lac': 'LAC', 
    'lal': 'LAL', 
    'mem': 'MEM', 
    'mia': 'MIA', 
    'mil': 'MIL', 
    'min': 'MIN', 
    'no': 'NOP', 'nop': 'NOP', 'noh': 'NOP', # Fix: no -> NOP
    'ny': 'NYK', 'nyk': 'NYK', 
    'okc': 'OKC', 'sea': 'OKC', # Handle Seattle -> OKC
    'orl': 'ORL', 
    'phi': 'PHI', 
    'phx': 'PHX', 'pho': 'PHX',
    'por': 'POR', 
    'sac': 'SAC', 
    'sa': 'SAS', 'sas': 'SAS', # Fix: sa -> SAS
    'tor': 'TOR', 
    'utah': 'UTA', 'uta': 'UTA', # Fix: utah -> UTA
    'wsh': 'WAS', 'was': 'WAS'   # Fix: wsh -> WAS
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
    
    # 2. Reshape: Split 'Away' and 'Home' into separate rows
    print("Reshaping odds data...")
    
    # Create Home Rows
    home_df = df.copy()
    home_df['TEAM_ABBREVIATION'] = home_df['home'].map(TEAM_MAP)
    home_df['OPP_TEAM_ABBREVIATION'] = home_df['away'].map(TEAM_MAP)
    home_df['MONEYLINE'] = home_df['moneyline_home']
    home_df['SPREAD'] = home_df['spread'] * -1 # Flip spread for home
    home_df['IS_HOME_ODDS'] = 1
    
    # Create Away Rows
    away_df = df.copy()
    away_df['TEAM_ABBREVIATION'] = away_df['away'].map(TEAM_MAP)
    away_df['OPP_TEAM_ABBREVIATION'] = away_df['home'].map(TEAM_MAP)
    away_df['MONEYLINE'] = away_df['moneyline_away']
    away_df['SPREAD'] = away_df['spread']
    away_df['IS_HOME_ODDS'] = 0

    # Combine
    odds_reshaped = pd.concat([home_df, away_df], ignore_index=True)
    
    # Keep useful columns
    cols_to_keep = ['GAME_DATE', 'TEAM_ABBREVIATION', 'MONEYLINE', 'SPREAD', 'total']
    odds_reshaped = odds_reshaped[cols_to_keep]
    
    # Remove rows that didn't map (optional logging)
    unmapped = odds_reshaped[odds_reshaped['TEAM_ABBREVIATION'].isna()]
    if len(unmapped) > 0:
        print(f"⚠️ Warning: {len(unmapped)} rows failed to map. (Likely preseason or old names)")
        
    odds_reshaped = odds_reshaped.dropna(subset=['TEAM_ABBREVIATION'])
    return odds_reshaped

def merge_data():
    # 1. Load Stats
    print(f"Loading stats from {STATS_PATH}...")
    try:
        df_stats = pd.read_csv(STATS_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {STATS_PATH}")
        return

    df_stats['GAME_DATE'] = pd.to_datetime(df_stats['GAME_DATE'])
    
    # 2. Load & Reshape Odds
    df_odds = load_and_reshape_odds()
    if df_odds is None: return
    
    # 3. Merge
    print("Merging datasets...")
    merged_df = pd.merge(
        df_stats, 
        df_odds, 
        on=['GAME_DATE', 'TEAM_ABBREVIATION'], 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("❌ Merge resulted in 0 rows.")
        print("Debugging Check:")
        print(f"Stats Example: {df_stats['TEAM_ABBREVIATION'].iloc[0]} on {df_stats['GAME_DATE'].iloc[0]}")
        print(f"Odds Example: {df_odds['TEAM_ABBREVIATION'].iloc[0]} on {df_odds['GAME_DATE'].iloc[0]}")
    else:
        print(f"✅ Merge Complete! Final Rows: {len(merged_df)}")
        merged_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_data()