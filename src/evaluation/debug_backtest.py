import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.team_mapping import normalize_team_name

# --- PATHS ---
STATS_PATH = 'data/processed/nba_model.csv'       # Your engineered features
ODDS_PATH = 'data/odds/nba_2008-2025.csv'                   # Your new odds file

def debug():
    print("--- 🔍 MERGE DIAGNOSTIC TOOL ---")
    
    # 1. Load Data
    if not os.path.exists(STATS_PATH): print(f"❌ Missing {STATS_PATH}"); return
    if not os.path.exists(ODDS_PATH): print(f"❌ Missing {ODDS_PATH}"); return
    
    df_stats = pd.read_csv(STATS_PATH)
    df_odds = pd.read_csv(ODDS_PATH)
    
    # 2. Inspect DATES
    print("\n[1] DATE FORMAT CHECK")
    print(f"Stats File Date Example: {df_stats['GAME_DATE'].iloc[0]} (Type: {type(df_stats['GAME_DATE'].iloc[0])})")
    print(f"Odds File Date Example:  {df_odds['date'].iloc[0]} (Type: {type(df_odds['date'].iloc[0])})")
    
    # Normalize for test
    df_stats['GAME_DATE_DT'] = pd.to_datetime(df_stats['GAME_DATE']).dt.normalize()
    df_odds['date_dt'] = pd.to_datetime(df_odds['date']).dt.normalize()
    
    # 3. Inspect TEAMS
    print("\n[2] TEAM NAME CHECK")
    stats_teams = sorted(df_stats['TEAM_ABBREVIATION'].unique())[:5]
    print(f"Stats File Teams (First 5): {stats_teams}")
    
    # Apply map to odds to see if it works
    df_odds['mapped_home'] = df_odds['home'].apply(normalize_team_name)
    odds_teams = sorted(df_odds['mapped_home'].dropna().unique())[:5]
    print(f"Odds File Mapped Teams (First 5): {odds_teams}")
    
    # 4. TRY TO MATCH A SPECIFIC GAME
    print("\n[3] MATCH ATTEMPT")
    # Pick the last game from stats (most recent)
    sample_game = df_stats.iloc[-1]
    target_date = sample_game['GAME_DATE_DT']
    target_team = sample_game['TEAM_ABBREVIATION']
    
    print(f"Attempting to find match for: {target_date.date()} - {target_team}")
    
    # Look for it in odds
    match = df_odds[
        (df_odds['date_dt'] == target_date) & 
        (df_odds['mapped_home'] == target_team)
    ]
    
    if len(match) > 0:
        print("✅ SUCCESS! Found this match in odds file:")
        print(match[['date', 'home', 'moneyline_home']])
    else:
        print("❌ FAILED. Could not find this game in odds file.")
        # Help user debug WHY
        print("  ... Checking if date exists in odds file?")
        date_match = df_odds[df_odds['date_dt'] == target_date]
        if len(date_match) > 0:
            print(f"  ✅ Yes, found {len(date_match)} games on {target_date.date()}.")
            print(f"  Teams playing that day: {date_match['mapped_home'].unique()}")
            print(f"  We were looking for: {target_team}")
            print("  ⚠️ MAPPING ISSUE likely. Check TEAM_MAP.")
        else:
            print(f"  ❌ No. The date {target_date.date()} is NOT in the odds file.")
            print(f"  Odds file date range: {df_odds['date_dt'].min().date()} to {df_odds['date_dt'].max().date()}")

if __name__ == "__main__":
    debug()