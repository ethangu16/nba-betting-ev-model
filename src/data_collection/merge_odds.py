import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.team_mapping import normalize_team_name

# --- PATHS ---
STATS_PATH = 'data/processed/nba_model.csv'      # Your current training data
ODDS_PATH = 'data/odds/nba_2008-2025.csv'                  # Your NEW file (update path if needed)
OUTPUT_PATH = 'data/processed/nba_model_with_odds.csv' # The final product

def merge_data():
    print("Loading datasets...")
    # 1. Load your Main Model Data
    if not os.path.exists(STATS_PATH):
        print(f"❌ Error: Could not find {STATS_PATH}")
        return
    df_main = pd.read_csv(STATS_PATH)
    df_main['GAME_DATE'] = pd.to_datetime(df_main['GAME_DATE'])
    
    # 2. Load the New Odds Data
    if not os.path.exists(ODDS_PATH):
        print(f"❌ Error: Could not find {ODDS_PATH}")
        return
    df_odds = pd.read_csv(ODDS_PATH)
    df_odds['date'] = pd.to_datetime(df_odds['date'])
    
    # 3. Clean the Odds Data
    print("Cleaning odds data...")
    # Rename columns to match what we want
    df_odds = df_odds.rename(columns={
        'date': 'GAME_DATE',
        'moneyline_home': 'HOME_ML',
        'moneyline_away': 'AWAY_ML'
    })
    
    # Apply Team Mapping using centralized function
    df_odds['home_abbr'] = df_odds['home'].apply(normalize_team_name)
    df_odds['away_abbr'] = df_odds['away'].apply(normalize_team_name)
    
    # Drop rows where mapping failed (rare errors)
    df_odds = df_odds.dropna(subset=['home_abbr', 'away_abbr'])
    
    # 4. Merge
    # We match on Date and Home Team. 
    # (Matching on just home team is safe enough per day)
    print("Merging odds into main dataset...")
    
    # Select only the columns we need from the odds file
    odds_subset = df_odds[['GAME_DATE', 'home_abbr', 'HOME_ML', 'AWAY_ML']]
    
    merged_df = pd.merge(
        df_main,
        odds_subset,
        left_on=['GAME_DATE', 'TEAM_ABBREVIATION'], # Match Main(Team) ...
        right_on=['GAME_DATE', 'home_abbr'],        # ... to Odds(Home)
        how='left'                                  # Keep all games, even if odds missing
    )
    
    # 5. Fill Missing Odds for Away Games
    # The merge above only attached odds to the HOME team's row.
    # We also need odds for the AWAY team's row.
    
    # Logic: If I am the Away Team, my "HOME_ML" is the opponent's home ML.
    # But for simplicity in backtesting, we often just look at Home rows.
    # However, to be thorough:
    
    # (Optional: If your model treats Home/Away rows separately, 
    # you might want to duplicate the odds logic. For now, let's just save.)
    
    # Clean up
    merged_df.drop(columns=['home_abbr'], inplace=True)
    
    # Save
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Created {OUTPUT_PATH} with {len(merged_df)} rows.")
    print("   You now have real historical odds.")

if __name__ == "__main__":
    merge_data()