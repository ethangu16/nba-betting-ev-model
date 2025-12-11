import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_PATH = 'data/processed/nba_final_dataset.csv'
OUTPUT_PATH = 'data/processed/nba_model_ready.csv'

# PARAMETERS
K_FACTOR = 25
HOME_ADVANTAGE = 60

def get_mov_multiplier(mov, elo_diff):
    # (MOV + 3)^0.8 / (7.5 + 0.006 * diff)
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def calculate_elo(df):
    print(f"Calculating Elo Ratings (K={K_FACTOR}, HomeAdv={HOME_ADVANTAGE})...")
    
    # Initialize all teams at 1500
    # We grab unique teams from BOTH columns to ensure we don't miss anyone
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
        
        # Check which column indicates Home status
        # Merged data might have 'IS_HOME' (from stats) or 'IS_HOME_ODDS' (from odds)
        is_home = row.get('IS_HOME', row.get('IS_HOME_ODDS', 0))
        
        # Calc Elo
        home_boost = HOME_ADVANTAGE if is_home == 1 else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        
        actual_win = 1 if row['WL'] == 'W' else 0
        
        # Margin of Victory (Handle missing PM)
        mov = abs(row['PLUS_MINUS']) if not pd.isna(row['PLUS_MINUS']) else 0
        
        elo_diff_winner = dr if actual_win == 1 else -dr
        multiplier = get_mov_multiplier(mov, elo_diff_winner)
        
        shift = K_FACTOR * multiplier * (actual_win - prob_win)
        
        team_elos[team] = r_team + shift
        team_elos[opp] = r_opp - shift

    df['ELO_TEAM'] = elo_team
    df['ELO_OPP'] = elo_opp
    return df

def create_rolling_features(df):
    print("Creating Rolling Averages (Last 10 games)...")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    metrics = ['PTS', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'PLUS_MINUS']
    
    for col in metrics:
        # Check if column exists before rolling
        if col in df.columns:
            df[f'ROLL_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            )
        else:
            print(f"⚠️ Warning: Column {col} missing. Skipping feature.")
            
    return df

def main():
    print(f"Loading {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print("❌ Error: Run merge_odds.py first!")
        return

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # --- FIX 1: FILL HOLES BEFORE DROPPING ---
    # If Rest Days is missing, assume 3 (average rest)
    if 'REST_DAYS' in df.columns:
        df['REST_DAYS'] = df['REST_DAYS'].fillna(3)
    
    # If Travel is missing, assume 0 (Home game or short trip)
    if 'TRAVEL_MILES' in df.columns:
        df['TRAVEL_MILES'] = df['TRAVEL_MILES'].fillna(0)
        
    # --- ENGINEER ---
    df = calculate_elo(df)
    df = create_rolling_features(df)
    
    df['TARGET_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # --- FIX 2: TARGETED DROP ---
    # Only drop rows where the CRITICAL ML features are missing.
    # We do NOT care if 'MONEYLINE' or 'SPREAD' is missing here (we can still train on the game result)
    # We only need ELO and ROLLING STATS to exist.
    
    features_to_check = ['ELO_TEAM', 'ROLL_PTS', 'ROLL_EFG_PCT']
    # Filter features that actually exist in the dataframe
    existing_features = [f for f in features_to_check if f in df.columns]
    
    initial_len = len(df)
    df = df.dropna(subset=existing_features)
    dropped_count = initial_len - len(df)
    
    print(f"Dropped {dropped_count} rows (Start of season / Insufficient history).")
    print(f"Final Dataset Size: {len(df)}")
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()