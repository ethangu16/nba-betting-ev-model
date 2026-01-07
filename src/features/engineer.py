import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

INPUT_PATH = 'data/raw/nba_games_stats.csv'
PLAYER_STATS_PATH = 'data/raw/nba_player_stats.csv'
OUTPUT_PATH = 'data/processed/nba_model.csv'

# ELO PARAMETERS
K_FACTOR = 25
HOME_ADVANTAGE = 60

def get_mov_multiplier(mov, elo_diff):
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def calculate_elo(df):
    print(f"Calculating Elo on {len(df)} games...")
    # Calculate Elo ratings for each team using FiveThirtyEight's formula
    all_teams = set(df['TEAM_ABBREVIATION'].unique()) | set(df['OPP_ABBREVIATION'].unique())
    team_elos = {team: 1500 for team in all_teams}
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    elo_team, elo_opp = [], []
    
    for index, row in df.iterrows():
        team, opp = row['TEAM_ABBREVIATION'], row['OPP_ABBREVIATION']
        r_team, r_opp = team_elos.get(team, 1500), team_elos.get(opp, 1500)
        elo_team.append(r_team)
        elo_opp.append(r_opp)
        
        is_home = row['IS_HOME']
        home_boost = HOME_ADVANTAGE if is_home == 1 else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        actual_win = 1 if row['WL'] == 'W' else 0
        mov = abs(row['PLUS_MINUS']) if not pd.isna(row['PLUS_MINUS']) else 0
        
        if actual_win: elo_diff_winner = dr
        else: elo_diff_winner = -dr
            
        shift = K_FACTOR * get_mov_multiplier(mov, elo_diff_winner) * (actual_win - prob_win)
        team_elos[team] += shift
        team_elos[opp] -= shift

    df['ELO_TEAM'], df['ELO_OPP'] = elo_team, elo_opp
    return df

def calculate_advanced_stats(df):
    print("Calculating Advanced Stats...")
    if 'MIN' in df.columns: df['MIN_CLEAN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(240)
    else: df['MIN_CLEAN'] = 240

    req = ['FGA', 'TOV', 'FTA', 'OREB', 'PTS', 'PLUS_MINUS']
    if all(c in df.columns for c in req):
        df['POSS_EST'] = 0.96 * (df['FGA'] + df['TOV'] + 0.44 * df['FTA'] - df['OREB'])
        df['POSS_EST'] = df['POSS_EST'].replace(0, 95) 
        df['PACE'] = 48 * (df['POSS_EST'] / (df['MIN_CLEAN'] / 5))
        df['OFF_RTG'] = 100 * (df['PTS'] / df['POSS_EST'])
        df['DEF_RTG'] = 100 * ((df['PTS'] - df['PLUS_MINUS']) / df['POSS_EST'])
    else:
        df['PACE'], df['OFF_RTG'], df['DEF_RTG'] = 98.0, 110.0, 110.0
    return df

def add_fatigue_features(df):
    print("Calculating Fatigue Features...")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # 1. Back-to-Backs
    df['IS_B2B'] = (df['REST_DAYS'].fillna(3) == 1).astype(int)
    
    # 2. 3 Games in 4 Nights
    df['DATE_MINUS_2'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(2)
    df['IS_3IN4'] = ((df['GAME_DATE'] - df['DATE_MINUS_2']).dt.days <= 4).astype(int)
    df.drop(columns=['DATE_MINUS_2'], inplace=True)
    return df

def calculate_roster_strength_gamescore(df_games):
    """
    Calculates roster strength using RAPTOR (Box Score + On/Off Blend).
    """
    print("\n Calculating Roster Strength via RAPTOR Logic...")
    if not os.path.exists(PLAYER_STATS_PATH): return df_games
    
    # 1. Load and Prep Player Stats
    df_players = pd.read_csv(PLAYER_STATS_PATH)
    df_players['GAME_DATE'] = pd.to_datetime(df_players['GAME_DATE'])
    
    # 2. Calculate Hollinger Game Score (The "Box" Component)
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
    
    # 🟢 3. Calculate RAPTOR Score (Box + On/Off Blend)
    def calculate_raptor(row):
        # Avoid noise from garbage time players (<5 mins)
        if row['MIN'] < 5: 
            return row['GAME_SCORE'] 
        
        # Box Component
        box_val = row['GAME_SCORE']
        
        # On/Off Component (Approximate Net Rating)
        # We cap it at +/- 30 to prevent massive outliers from 5-minute stints
        impact_val = (row['PLUS_MINUS'] / row['MIN']) * 48
        impact_val = max(min(impact_val, 30), -30)
        
        # Blend (50/50 Split)
        return (box_val * 0.5) + (impact_val * 0.5)

    df_players['RAPTOR_SCORE'] = df_players.apply(calculate_raptor, axis=1)
    
    # 4. Create Rolling Average (Last 10 games) using RAPTOR
    df_players = df_players.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df_players['ROLLING_SCORE'] = df_players.groupby('PLAYER_ID')['RAPTOR_SCORE'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    ).fillna(0)
    
    # 5. Map to Games
    print("  > Mapping players to games...")
    
    # We create a lookup dict: (GameID, TeamAbbr) -> List of Player Rolling RAPTOR Scores
    df_players_slim = df_players[['GAME_ID', 'TEAM_ABBREVIATION', 'ROLLING_SCORE']]
    game_rosters = df_players_slim.groupby(['GAME_ID', 'TEAM_ABBREVIATION'])['ROLLING_SCORE'].apply(list).to_dict()
    
    roster_scores = []
    for idx, row in df_games.iterrows():
        key = (row['GAME_ID'], row['TEAM_ABBREVIATION'])
        player_scores = game_rosters.get(key, [])
        
        if not player_scores:
            roster_scores.append(0)
            continue
            
        # Sum top 8 active players based on their RECENT RAPTOR form
        player_scores.sort(reverse=True)
        top_8 = sum(player_scores[:8])
        roster_scores.append(top_8)

    df_games['ROSTER_TALENT_SCORE'] = roster_scores
    return df_games

def create_rolling_features(df):
    """
    Creates multiple rolling windows:
    1. SMA_10: Stability/Class (Simple Moving Average)
    2. EWMA_10: Momentum (Exponential Weighted)
    3. EWMA_5: Streak (Exponential Weighted)
    """
    print("Creating Rolling Features")
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    
    metrics = ['OFF_RTG', 'DEF_RTG', 'PACE', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'ROSTER_TALENT_SCORE']
    
    for col in metrics:
        if col not in df.columns: continue
        
        # 1. Stability Feature (SMA-20) - Long term "Class"
        df[f'SMA_20_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=5).mean()
        )
        
        # 2. Form Feature (EWMA-10) - Recent "Momentum"
        df[f'EWMA_10_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
            lambda x: x.shift(1).ewm(span=10, adjust=False).mean()
        )
        
        # 3. Streak Feature (EWMA-5) - Immediate "Hot/Cold"
        df[f'EWMA_5_{col}'] = df.groupby('TEAM_ABBREVIATION')[col].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )
            
    return df

def main():
    print(f"Loading {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError: return
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['TARGET_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # Create OPP_ABBREVIATION by mapping GAME_ID to the other team in that game
    if 'OPP_ABBREVIATION' not in df.columns:
        print("Creating OPP_ABBREVIATION column...")
        opp_map = df.groupby('GAME_ID')['TEAM_ABBREVIATION'].apply(list).to_dict()
        df['OPP_ABBREVIATION'] = df.apply(
            lambda x: [t for t in opp_map.get(x['GAME_ID'], []) if t != x['TEAM_ABBREVIATION']][0] 
            if len(opp_map.get(x['GAME_ID'], [])) > 1 else None, 
            axis=1
        )
    
    df = calculate_advanced_stats(df)
    df = add_fatigue_features(df)
    df = calculate_elo(df)
    df = calculate_roster_strength_gamescore(df)
    df = create_rolling_features(df)
    
    # Drop rows where we don't have enough history for the primary momentum feature
    df_clean = df.dropna(subset=['EWMA_10_OFF_RTG', 'ELO_TEAM'])
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Success! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()