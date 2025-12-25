import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PATH = 'data/raw/nba_games_stats.csv'       # Ensure this points to your raw stats
PLAYER_STATS_PATH = 'data/raw/nba_player_stats.csv'  # Player stats for roster strength
OUTPUT_PATH = 'data/processed/nba_model.csv'         # Final Training File

# PARAMETERS
K_FACTOR = 25
HOME_ADVANTAGE = 60

def get_mov_multiplier(mov, elo_diff):
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def extract_opponent_abbreviation(df):
    """Extract opponent abbreviation from MATCHUP column."""
    def get_opp(matchup, team):
        if '@' in matchup:
            # Format: "TEAM @ OPP"
            parts = matchup.split(' @ ')
            return parts[1] if parts[0] == team else parts[0]
        elif 'vs.' in matchup:
            # Format: "TEAM vs. OPP" 
            parts = matchup.split(' vs. ')
            return parts[1] if parts[0] == team else parts[0]
        else:
            # Fallback
            return None
    
    df['OPP_ABBREVIATION'] = df.apply(lambda row: get_opp(row['MATCHUP'], row['TEAM_ABBREVIATION']), axis=1)
    return df

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
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns for advanced stats: {missing_cols}")
        print("   Some features may be incomplete. Continuing with available data...")
        # Don't return early - continue with what we have

    # Only calculate if we have the required columns
    if all(c in df.columns for c in req_cols):
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
    else:
        # Set defaults if we can't calculate
        df['POSS_EST'] = 100  # Default estimate
        df['PACE'] = 100
        df['OFF_RTG'] = 100
        df['DEF_RTG'] = 100
        df['PTS_ALLOWED'] = df.get('PTS', 100) - df.get('PLUS_MINUS', 0)
    
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
    """
    Calculate roster strength by aggregating player talent scores for each team/game.
    Uses rolling average of player game scores up to that date.
    """
    print("Calculating Roster Strength from player stats...")
    
    # Check if player stats file exists
    if not os.path.exists(PLAYER_STATS_PATH):
        print(f"⚠️ Warning: {PLAYER_STATS_PATH} not found. Setting ROSTER_TALENT_SCORE to 0.")
        df_games['ROSTER_TALENT_SCORE'] = 0
        return df_games
    
    try:
        # Load player stats
        df_players = pd.read_csv(PLAYER_STATS_PATH)
        
        # Ensure GAME_DATE is datetime
        df_players['GAME_DATE'] = pd.to_datetime(df_players['GAME_DATE'])
        df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
        
        # Calculate Hollinger Game Score for each player game
        required_player_cols = ['PTS', 'FGM', 'FGA', 'FTA', 'FTM', 'OREB', 'DREB', 
                                'STL', 'AST', 'BLK', 'PF', 'TOV']
        if all(col in df_players.columns for col in required_player_cols):
            df_players['GAME_SCORE'] = (
                df_players['PTS'] + 
                0.4 * df_players['FGM'] - 
                0.7 * df_players['FGA'] - 
                0.4 * (df_players['FTA'] - df_players['FTM']) + 
                0.7 * df_players['OREB'] + 
                0.3 * df_players['DREB'] + 
                df_players['STL'] + 
                0.7 * df_players['AST'] + 
                0.7 * df_players['BLK'] - 
                0.4 * df_players['PF'] - 
                df_players['TOV']
            )
        else:
            print("⚠️ Missing player stat columns. Using PTS as proxy for game score.")
            df_players['GAME_SCORE'] = df_players.get('PTS', 0)
        
        # Calculate rolling average game score for each player (up to each game date)
        df_players = df_players.sort_values(['PLAYER_ID', 'GAME_DATE'])
        df_players['PLAYER_AVG_SCORE'] = df_players.groupby('PLAYER_ID')['GAME_SCORE'].transform(
            lambda x: x.shift(1).expanding().mean()  # Rolling average up to (but not including) current game
        )
        
        # For each game, aggregate roster strength
        # Get the top 9 players by average game score for each team on each date
        roster_scores = []
        
        for idx, row in df_games.iterrows():
            team = row['TEAM_ABBREVIATION']
            game_date = row['GAME_DATE']
            
            # Get all players on this team who played before or on this date
            team_players = df_players[
                (df_players['TEAM_ABBREVIATION'] == team) & 
                (df_players['GAME_DATE'] <= game_date)
            ]
            
            if team_players.empty:
                roster_scores.append(0)
                continue
            
            # Get the most recent average score for each player (up to this date)
            player_scores = team_players.groupby('PLAYER_ID')['PLAYER_AVG_SCORE'].last()
            
            # Take top 9 players and sum their scores
            top_players = player_scores.nlargest(9)
            roster_score = top_players.sum()
            
            # Scale to approximate full game contribution (240 minutes total)
            # This is a rough estimate - actual would need minute projections
            roster_scores.append(roster_score * 0.95)  # Slight discount for bench depth
        
        df_games['ROSTER_TALENT_SCORE'] = roster_scores
        print(f"✅ Calculated roster strength for {len(df_games)} games.")
        
    except Exception as e:
        print(f"⚠️ Error calculating roster strength: {e}")
        print("   Setting ROSTER_TALENT_SCORE to 0.")
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
    df = extract_opponent_abbreviation(df)  # Extract opponent from matchup
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