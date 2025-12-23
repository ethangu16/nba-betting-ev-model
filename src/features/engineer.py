import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PATH = 'data/raw/nba_games_stats.csv'       # Ensure this points to your raw stats
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

def calculate_advanced_stats(df):
    """
    Calculates Pace (Possessions) and Efficiency (Off/Def Rating).
    This normalizes data so fast teams don't look artificially better.
    """
    print("Calculating Advanced Stats (Pace, OffRtg, DefRtg)...")
    
    # Clean MIN column (handle "240:00" vs 240)
    # If MIN is missing, assume 240 (standard game)
    if 'MIN' in df.columns:
        # Force to numeric, coerce errors to NaN, then fill with 240
        df['MIN_CLEAN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(240)
    else:
        df['MIN_CLEAN'] = 240

    # 1. Estimate Possessions
    # Formula: 0.96 * (FGA + TOV + 0.44*FTA - OREB)
    # We check if columns exist to avoid crashes
    req_cols = ['FGA', 'TOV', 'FTA', 'OREB', 'PTS', 'PLUS_MINUS']
    for c in req_cols:
        if c not in df.columns:
            print(f"⚠️ Missing {c} for advanced stats. Skipping.")
            return df

    df['POSS_EST'] = 0.96 * (df['FGA'] + df['TOV'] + 0.44 * df['FTA'] - df['OREB'])
    
    # 2. Calculate Pace (Possessions per 48 Minutes)
    # We use MIN/5 because MIN is usually total player minutes (e.g. 240 for a full game)
    # If MIN is game duration (48), just use MIN. 
    # Usually in NBA data, 'MIN' for a team row is ~240.
    df['PACE'] = 48 * (df['POSS_EST'] / (df['MIN_CLEAN'] / 5))
    
    # 3. Calculate Efficiency (Per 100 Possessions)
    df['OFF_RTG'] = 100 * (df['PTS'] / df['POSS_EST'])
    
    # Derived Points Allowed (PTS - PLUS_MINUS)
    df['PTS_ALLOWED'] = df['PTS'] - df['PLUS_MINUS']
    df['DEF_RTG'] = 100 * (df['PTS_ALLOWED'] / df['POSS_EST'])
    
    return df

def add_fatigue_features(df):
    """
    Calculates Rest Days, Back-to-Backs, and 3-in-4-Nights.
    """
    print("Calculating Fatigue Features...")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # Calculate days since previous game
    df['DAYS_SINCE_LAST'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days
    
    # Fill NaN (First game of season) with 3+ days rest
    df['DAYS_SINCE_LAST'] = df['DAYS_SINCE_LAST'].fillna(3)
    
    # 1. Back-to-Back (Played Yesterday)
    df['IS_B2B'] = df['DAYS_SINCE_LAST'].apply(lambda x: 1 if x == 1 else 0)
    
    # 2. 3 Games in 4 Nights (High Fatigue)
    # We compare current date with the date from 2 games ago
    df['DATE_MINUS_2'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(2)
    df['DAYS_DIFF_3G'] = (df['GAME_DATE'] - df['DATE_MINUS_2']).dt.days
    
    # If you played 3 games in <= 4 days (e.g. Mon, Tue, Thu), that's a 3-in-4
    df['IS_3IN4'] = df['DAYS_DIFF_3G'].apply(lambda x: 1 if x <= 4 else 0)
    
    # Drop temp column
    df.drop(columns=['DATE_MINUS_2'], inplace=True)
    
    return df

def calculate_roster_strength(df_games):
    print("Initializing Roster Strength column...")
    df_games['ROSTER_TALENT_SCORE'] = 0
    return df_games

def create_rolling_features(df):
    print("Creating Rolling Features...")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # UPDATED: We now roll Advanced Stats instead of just PTS
    metrics = [
        'OFF_RTG', 'DEF_RTG', 'PACE',       # New Advanced Stats
        'EFG_PCT', 'TOV_PCT', 'ORB_PCT',    # Four Factors
        'FTR', 'PLUS_MINUS', 
        'ROSTER_TALENT_SCORE'
    ]
    
    for col in metrics:
        if col not in df.columns: 
            print(f"⚠️ Warning: {col} missing, skipping.")
            continue
            
        # Shift(1) is critical for preventing data leakage
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
    
    # 2. Create Target Variable
    df['TARGET_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # 3. Engineer Features
    df = calculate_advanced_stats(df)  # <--- NEW Step
    df = add_fatigue_features(df)      # <--- NEW Step
    df = calculate_elo(df)
    df = calculate_roster_strength(df) 
    
    # 4. Roll the Features
    df = create_rolling_features(df)
    
    # 5. Clean NaNs (Start of season rows)
    # Check for OFF_RTG rolling since that's our new main stat
    df = df.dropna(subset=['ROLL_OFF_RTG', 'ELO_TEAM'])
    
    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Saved {len(df)} rows to {OUTPUT_PATH}")
    print("   New Columns: ROLL_OFF_RTG, ROLL_DEF_RTG, ROLL_PACE, IS_B2B, IS_3IN4")

if __name__ == "__main__":
    main()