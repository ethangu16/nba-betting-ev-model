import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.team_mapping import normalize_team_name

# --- CONFIGURATION ---
STATS_PATH = 'data/processed/nba_model.csv' 
ODDS_PATH = 'data/odds/nba_2008-2025.csv'
OUTPUT_PATH = 'data/processed/nba_model_with_odds.csv'

def merge_data():
    print(f"Loading datasets...")
    
    # 1. Check Files
    if not os.path.exists(STATS_PATH):
        print(f"❌ Error: {STATS_PATH} not found. Run engineer.py first.")
        return
    if not os.path.exists(ODDS_PATH):
        print(f"❌ Error: {ODDS_PATH} not found.")
        return

    # 2. Load Data
    df_main = pd.read_csv(STATS_PATH)
    df_main['GAME_DATE'] = pd.to_datetime(df_main['GAME_DATE'])

    df_odds = pd.read_csv(ODDS_PATH)
    df_odds['date'] = pd.to_datetime(df_odds['date'])

    print("Cleaning and Normalising odds data...")
    df_odds['home_abbr'] = df_odds['home'].apply(normalize_team_name)
    df_odds['away_abbr'] = df_odds['away'].apply(normalize_team_name)
    
    # Drop rows where normalisation failed
    df_odds = df_odds.dropna(subset=['home_abbr', 'away_abbr'])

    # We need to reshape the odds file so it can match BOTH the Home and Away rows
    # in your stats file.
    
    # Perspective 1: "I am the Home Team"
    odds_home = df_odds.copy()
    odds_home['TEAM_ABBREVIATION'] = odds_home['home_abbr']
    
    # Perspective 2: "I am the Away Team"
    odds_away = df_odds.copy()
    odds_away['TEAM_ABBREVIATION'] = odds_away['away_abbr']
    
    # Stack them vertically
    odds_stacked = pd.concat([odds_home, odds_away], axis=0)

    odds_ready = odds_stacked[[
        'date', 'TEAM_ABBREVIATION', 
        'moneyline_home', 'moneyline_away', 'spread', 'total'
    ]].rename(columns={
        'date': 'GAME_DATE',
        'moneyline_home': 'HOME_ML',
        'moneyline_away': 'AWAY_ML',
        'spread': 'SPREAD_LINE',
        'total': 'TOTAL_LINE'
    })

    print("Merging odds into main dataset...")
    
    # 3. The Merge
    merged_df = pd.merge(
        df_main,
        odds_ready,
        on=['GAME_DATE', 'TEAM_ABBREVIATION'],
        how='left'
    )

    # Sort chronologically
    merged_df = merged_df.sort_values(['GAME_DATE', 'GAME_ID'])

    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Success! Saved {len(merged_df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_data()