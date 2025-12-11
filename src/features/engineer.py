import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_PATH = 'data/processed/nba_final_dataset.csv'
OUTPUT_PATH = 'data/processed/nba_model_ready.csv'

# YOUR OPTIMIZED PARAMETERS
K_FACTOR = 25
HOME_ADVANTAGE = 60

def get_mov_multiplier(mov, elo_diff):
    """
    538's Margin of Victory Multiplier.
    mov: Margin of Victory (Absolute value)
    elo_diff: Difference in ratings (Winner - Loser)
    """
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def calculate_elo(df):
    """
    Calculates a running Elo rating for every team using:
    1. Optimized K-Factor & Home Advantage
    2. Margin of Victory Multiplier
    """
    print(f"Calculating Elo Ratings (K={K_FACTOR}, HomeAdv={HOME_ADVANTAGE})...")
    
    # Initialize all teams at 1500
    team_elos = {team: 1500 for team in df['TEAM_ABBREVIATION'].unique()}
    
    # Sort chronologically
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Lists to store the PRE-GAME Elo (Feature)
    elo_team = []
    elo_opp = []
    
    for index, row in df.iterrows():
        team = row['TEAM_ABBREVIATION']
        opp = row['OPP_ABBREVIATION']
        
        # Get current ratings (default 1500)
        r_team = team_elos.get(team, 1500)
        r_opp = team_elos.get(opp, 1500)
        
        # Store PRE-GAME ratings (These are what the model sees)
        elo_team.append(r_team)
        elo_opp.append(r_opp)
        
        # --- CALCULATION PHASE ---
        
        # 1. Expected Win Probability
        home_boost = HOME_ADVANTAGE if row['IS_HOME'] == 1 else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        
        # 2. Actual Result
        actual_win = 1 if row['WL'] == 'W' else 0
        
        # 3. Margin of Victory Multiplier
        mov = abs(row['PLUS_MINUS'])
        
        # Elo diff relative to the WINNER
        elo_diff_winner = dr if actual_win == 1 else -dr
        multiplier = get_mov_multiplier(mov, elo_diff_winner)
        
        # 4. Update Ratings
        shift = K_FACTOR * multiplier * (actual_win - prob_win)
        
        # Save new ratings for next time
        team_elos[team] = r_team + shift
        team_elos[opp] = r_opp - shift

    df['ELO_TEAM'] = elo_team
    df['ELO_OPP'] = elo_opp
    return df

def create_rolling_features(df):
    print("Creating Rolling Averages (Last 10 games)...")
    
    # Sort by Team and Date
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # Metrics to average
    metrics = ['PTS', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'PLUS_MINUS']
    
    # Group by Team, then calculate rolling mean of the PREVIOUS 10 games
    for col in metrics:
        df[f'ROLL_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )
        
    return df

def main():
    # 1. Load Data
    print(f"Loading {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print("❌ Error: Run the merge script first!")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Engineer Features
    df = calculate_elo(df)
    df = create_rolling_features(df)
    
    # 3. Create Target Variable
    # 1 if Win, 0 if Loss
    df['TARGET_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # 4. Clean NaNs 
    # (The first 3 games for each team will have NaN rolling stats -> Drop them)
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows due to insufficient history (start of season).")
    
    # 5. Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Saved {len(df)} rows to {OUTPUT_PATH}")
    print("Columns ready for ML:")
    print(df[['GAME_DATE', 'ELO_TEAM', 'ROLL_PTS', 'REST_DAYS', 'TRAVEL_MILES']].head())

if __name__ == "__main__":
    main()