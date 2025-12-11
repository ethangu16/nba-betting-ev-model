from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
import time
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
OUTPUT_FILE = 'data/raw/nba_games_advanced.csv'

# Arena Coordinates
ARENA_COORDS = {
    'ATL': (33.757, -84.396), 'BOS': (42.366, -71.062), 'BKN': (40.682, -73.975),
    'CHA': (35.225, -80.839), 'CHI': (41.880, -87.674), 'CLE': (41.496, -81.688),
    'DAL': (32.790, -96.810), 'DEN': (39.748, -105.007), 'DET': (42.341, -83.055),
    'GSW': (37.768, -122.387), 'HOU': (29.750, -95.362), 'IND': (39.764, -86.155),
    'LAC': (33.945, -118.342), 'LAL': (34.043, -118.267), 'MEM': (35.138, -90.050),
    'MIA': (25.781, -80.187), 'MIL': (43.045, -87.917), 'MIN': (44.979, -93.276),
    'NOP': (29.949, -90.082), 'NYK': (40.750, -73.993), 'OKC': (35.463, -97.515),
    'ORL': (28.539, -81.383), 'PHI': (39.901, -75.172), 'PHX': (33.445, -112.071),
    'POR': (45.531, -122.666), 'SAC': (38.580, -121.499), 'SAS': (29.427, -98.437),
    'TOR': (43.643, -79.379), 'UTA': (40.768, -111.901), 'WAS': (38.898, -77.020)
}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in miles between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956
    return c * r

def get_games_for_season(season_str):
    """Fetch games for a SINGLE season"""
    print(f"Fetching {season_str}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        )
        games = gamefinder.get_data_frames()[0]
        print(f"  -> Found {len(games)} rows.")
        return games
    except Exception as e:
        print(f"Error fetching {season_str}: {e}")
        return pd.DataFrame()

def calculate_advanced_stats(df):
    
    # 1. Self-Join to get Opponent Stats
    df_opp = df[['GAME_ID', 'TEAM_ID', 'FGA', 'FTA', 'TOV', 'OREB', 'DREB', 'PTS']].copy()
    df_opp.columns = ['GAME_ID', 'OPP_TEAM_ID', 'OPP_FGA', 'OPP_FTA', 'OPP_TOV', 'OPP_OREB', 'OPP_DREB', 'OPP_PTS']
    
    df = pd.merge(df, df_opp, on='GAME_ID')
    df = df[df['TEAM_ID'] != df['OPP_TEAM_ID']] # Filter out self-matches
    
    # 2. Four Factors Calculation
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    df['ORB_PCT'] = df['OREB'] / (df['OREB'] + df['OPP_DREB'])
    df['FTR'] = df['FTA'] / df['FGA']

    # 3. Rest Days & Travel Distance
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    # Calculate days since last game (fill first game with 7 days rest)
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(7)
    
    # Coordinates & Travel
    df['TEAM_COORDS'] = df['TEAM_ABBREVIATION'].map(ARENA_COORDS)
    # Extract opponent from MATCHUP
    df['OPP_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    df['OPP_COORDS'] = df['OPP_ABBREVIATION'].map(ARENA_COORDS)
    
    # Determine Game Location
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    
    # If home, use team coords. If away, use opponent coords.
    df['GAME_LOC_LAT'] = np.where(df['IS_HOME'] == 1, 
                                  df['TEAM_COORDS'].apply(lambda x: x[0] if isinstance(x, tuple) else 0), 
                                  df['OPP_COORDS'].apply(lambda x: x[0] if isinstance(x, tuple) else 0))
    df['GAME_LOC_LON'] = np.where(df['IS_HOME'] == 1, 
                                  df['TEAM_COORDS'].apply(lambda x: x[1] if isinstance(x, tuple) else 0), 
                                  df['OPP_COORDS'].apply(lambda x: x[1] if isinstance(x, tuple) else 0))

    # Shift to get previous location
    df['PREV_LAT'] = df.groupby('TEAM_ID')['GAME_LOC_LAT'].shift(1)
    df['PREV_LON'] = df.groupby('TEAM_ID')['GAME_LOC_LON'].shift(1)
    
    # Calculate Travel
    df['TRAVEL_MILES'] = df.apply(
        lambda row: haversine(row['PREV_LON'], row['PREV_LAT'], row['GAME_LOC_LON'], row['GAME_LOC_LAT']) 
        if pd.notnull(row['PREV_LAT']) and row['PREV_LAT'] != 0 else 0, axis=1
    )

    return df

if __name__ == "__main__":
    print("Starting NBA data collection (Multi-Season)...")
    
    # 1. Loop through seasons and collect data
    all_season_dfs = []
    for season in SEASONS:
        season_df = get_games_for_season(season)
        if not season_df.empty:
            all_season_dfs.append(season_df)
            time.sleep(1) 
            
    if not all_season_dfs:
        print("❌ No data collected. Check your internet or API limits.")
    else:
        full_df = pd.concat(all_season_dfs, ignore_index=True)
        print(f"Total raw rows: {len(full_df)}")
        
        # 2. Process
        processed_df = calculate_advanced_stats(full_df)
        
        # 3. Save
        cols = [
            'GAME_DATE', 'SEASON_ID', 'GAME_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL',
            'PTS', 'PLUS_MINUS', 
            'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 
            'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME'
        ]
        
        processed_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✅ Success! Saved {len(processed_df)} rows to {OUTPUT_FILE}")
print(processed_df.groupby('SEASON_ID')['GAME_ID'].count())
