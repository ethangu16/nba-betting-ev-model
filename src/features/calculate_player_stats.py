import pandas as pd

INPUT_PATH = 'data/raw/nba_player_stats.csv'  # The file you just showed me
OUTPUT_PATH = 'data/processed/processed_player.csv' # The file predict_today NEEDS
import os

def create_talent_pool():
    print(f"Loading raw player data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Clean Data
    # Convert 'MIN' to number (sometimes it is '34:12', we need just 34.2)
    # Simple fix: If it's a string containing ':', take the first part. 
    # Usually the API gives an integer or float, so we just ensure numeric.
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
    
    # 2. Calculate Hollinger Game Score (The "Rating")
    # Formula: PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    df['GAME_SCORE'] = (
        df['PTS'] + 
        0.4 * df['FGM'] - 
        0.7 * df['FGA'] - 
        0.4 * (df['FTA'] - df['FTM']) + 
        0.7 * df['OREB'] + 
        0.3 * df['DREB'] + 
        df['STL'] + 
        0.7 * df['AST'] + 
        0.7 * df['BLK'] - 
        0.4 * df['PF'] - 
        df['TOV']
    )
    
    # 3. Calculate Rolling Averages (Current Form)
    # We sort by date so the rolling window is correct
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    print("Calculating rolling averages (Last 10 games)...")
    
    # Calculate average Game Score & Minutes over last 10 games
    # We group by PLAYER_ID so stats don't bleed between players
    df['GAME_SCORE_AVG'] = df.groupby('PLAYER_ID')['GAME_SCORE'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    )
    
    df['MIN_AVG'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean()
    )
    
    # 4. Extract the "Latest" Rating for every player
    # We only care about who they are TODAY, so we take the last row for each player.
    latest_ratings = df.groupby('PLAYER_ID').tail(1)
    
    # 5. Keep only necessary columns
    # We need TEAM_ABBREVIATION to find them in predict_today.py
    final_df = latest_ratings[[
        'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_SCORE_AVG', 'MIN_AVG'
    ]]
    
    # Clean up (remove guys who haven't played recently/have no data)
    final_df = final_df.dropna()
    
    print(f"✅ Created Talent Pool with {len(final_df)} active players.")
    print(final_df.head())
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_talent_pool()