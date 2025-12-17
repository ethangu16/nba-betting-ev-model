import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PATH = 'data/raw/nba_games_stats.csv'       # Full history
PLAYER_PATH = 'data/processed/processed_player.csv'  # Player Ratings
OUTPUT_PATH = 'data/processed/nba_model.csv'         # Final Training File

# PARAMETERS
K_FACTOR = 25
HOME_ADVANTAGE = 60

def get_mov_multiplier(mov, elo_diff):
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def calculate_elo(df):
    print(f"Calculating Elo on {len(df)} games...")
    all_teams = set(df['TEAM_ABBREVIATION'].unique()) | set(df['OPP_ABBREVIATION'].unique())
    team_elos = {team: 1500 for team in all_teams}
    
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    elo_team = []
    elo_opp = []
    
    for index, row in df.iterrows():
        team = row['TEAM_ABBREVIATION']
        opp = row['OPP_ABBREVIATION']
        r_team = team_elos.get(team, 1500)
        r_opp = team_elos.get(opp, 1500)
        
        elo_team.append(r_team)
        elo_opp.append(r_opp)
        
        is_home = row['IS_HOME']
        home_boost = HOME_ADVANTAGE if is_home == 1 else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        actual_win = 1 if row['WL'] == 'W' else 0
        mov = abs(row['PLUS_MINUS']) if not pd.isna(row['PLUS_MINUS']) else 0
        elo_diff_winner = dr if actual_win == 1 else -dr
        multiplier = get_mov_multiplier(mov, elo_diff_winner)
        shift = K_FACTOR * multiplier * (actual_win - prob_win)
        
        team_elos[team] = r_team + shift
        team_elos[opp] = r_opp - shift

    df['ELO_TEAM'] = elo_team
    df['ELO_OPP'] = elo_opp
    return df

def calculate_roster_strength(df_games):
    """
    Creates the ROSTER_TALENT_SCORE column.
    For historical training, we will use a placeholder (0) if we don't have
    historical daily rosters. This ensures the column exists so the code doesn't crash.
    """
    print("Initializing Roster Strength column...")
    # For V1, we initialize with 0 for history. 
    # The 'Predict Today' script will fill this with real data for tonight.
    df_games['ROSTER_TALENT_SCORE'] = 0
    return df_games

def create_rolling_features(df):
    print("Creating Rolling Features...")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # Metrics to roll
    metrics = ['PTS', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'PLUS_MINUS', 'ROSTER_TALENT_SCORE']
    
    for col in metrics:
        if col not in df.columns: 
            print(f"⚠️ Warning: {col} missing, skipping.")
            continue
            
        df[f'ROLL_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )
    return df

def main():
    print(f"Loading {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    # 1. Clean Dates
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Create Target Variable (The "Answer Key")
    # This was missing in the last version!
    df['TARGET_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # 3. Engineer Features
    df = calculate_elo(df)
    df = calculate_roster_strength(df) # Creates ROSTER_TALENT_SCORE
    df = create_rolling_features(df)   # Creates ROLL_ROSTER_TALENT_SCORE
    
    # 4. Clean NaNs (Start of season rows)
    df = df.dropna(subset=['ROLL_PTS', 'ELO_TEAM'])
    
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Saved {len(df)} rows to {OUTPUT_PATH}")
    print("   Columns created: TARGET_WIN, ROLL_ROSTER_TALENT_SCORE")

if __name__ == "__main__":
    main()