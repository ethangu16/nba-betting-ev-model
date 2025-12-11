import pandas as pd

# CONFIGURATION
ODDS_PATH = 'data/odds/nba_2008-2025.csv'
STATS_PATH = 'data/raw/nba_games_advanced.csv'

def inspect():
    print("--- 1. INSPECTING ODDS FILE ---")
    try:
        df_odds = pd.read_csv(ODDS_PATH)
        print(f"Total Rows: {len(df_odds)}")
        print(f"Columns: {df_odds.columns.tolist()}")
        
        # Check Date Range
        if 'date' in df_odds.columns:
            print(f"\nRaw Date Sample: {df_odds['date'].iloc[0]} (Type: {type(df_odds['date'].iloc[0])})")
            df_odds['date'] = pd.to_datetime(df_odds['date'])
            print(f"Date Range: {df_odds['date'].min()} to {df_odds['date'].max()}")
        
        # Check Team Names (Crucial)
        print("\nUnique Team Names in 'home' column (First 10):")
        print(df_odds['home'].unique()[:10])
        
    except Exception as e:
        print(f"Error reading odds: {e}")

    print("\n--- 2. INSPECTING STATS FILE ---")
    try:
        df_stats = pd.read_csv(STATS_PATH)
        df_stats['GAME_DATE'] = pd.to_datetime(df_stats['GAME_DATE'])
        print(f"Date Range: {df_stats['GAME_DATE'].min()} to {df_stats['GAME_DATE'].max()}")
        print("\nUnique Team Abbreviations (First 5):")
        print(df_stats['TEAM_ABBREVIATION'].unique()[:5])
        
    except Exception as e:
        print(f"Error reading stats: {e}")

if __name__ == "__main__":
    inspect()