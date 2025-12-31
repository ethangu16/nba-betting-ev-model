from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time
import os

# --- CONFIGURATION ---
DATA_DIR = 'data/raw'
GAMES_FILE = f'{DATA_DIR}/nba_games_stats.csv'
PLAYERS_FILE = f'{DATA_DIR}/nba_player_stats.csv'

START_SEASON = 2015
CURRENT_SEASON = 2025

def get_season_string(year):
    return f"{year}-{str(year + 1)[-2:]}"

def get_fetch_params(existing_df=None):
    """
    Decides whether to fetch history or just updates.
    """
    if existing_df is None or existing_df.empty or 'GAME_DATE' not in existing_df.columns:
        print("No existing data found. Fetching full history...")
        return [get_season_string(y) for y in range(START_SEASON, CURRENT_SEASON + 1)], None

    try:
        # Find the latest date
        dates = pd.to_datetime(existing_df['GAME_DATE'], format='mixed', errors='coerce')
        last_date = dates.max()
        start_date_str = last_date.strftime('%m/%d/%Y')
        print(f"Existing data found up to {last_date.date()}. Checking for games after {start_date_str}...")
        return [get_season_string(CURRENT_SEASON)], start_date_str
    except Exception as e:
        print(f"Error reading dates ({e}). Fetching full history.")
        return [get_season_string(y) for y in range(START_SEASON, CURRENT_SEASON + 1)], None

def fetch_games(seasons, start_date=None, mode='Team'):
    dfs = []
    player_flag = 'P' if mode == 'Player' else 'T'

    for season in seasons:
        print(f"   > Fetching {mode} stats for {season}...")
        for attempt in range(3):
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    date_from_nullable=start_date, 
                    league_id_nullable='00',
                    season_type_nullable='Regular Season',
                    player_or_team_abbreviation=player_flag,
                    timeout=60
                )
                data = gamefinder.get_data_frames()[0]
                if not data.empty:
                    dfs.append(data)
                time.sleep(0.6)
                break
            except Exception as e:
                print(f"   ⚠️ Retry {attempt+1}/3 for {season}: {e}")
                time.sleep((attempt + 1) * 2)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_team_data(df):
    print("   > Processing team advanced stats...")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    cols_to_rename = {
        'TEAM_ID': 'OPP_TEAM_ID', 'PTS': 'OPP_PTS', 'FGA': 'OPP_FGA', 
        'FTA': 'OPP_FTA', 'TOV': 'OPP_TOV', 'OREB': 'OPP_OREB', 'DREB': 'OPP_DREB'
    }
    df_opp = df[['GAME_ID'] + list(cols_to_rename.keys())].rename(columns=cols_to_rename)

    df = pd.merge(df, df_opp, on='GAME_ID')
    df = df[df['TEAM_ID'] != df['OPP_TEAM_ID']].copy()

    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
    df['ORB_PCT'] = df['OREB'] / (df['OREB'] + df['OPP_DREB'])
    df['FTR'] = df['FTA'] / df['FGA']
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(3)

    cols = [
        'GAME_DATE', 'SEASON_ID', 'GAME_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL',
        'PTS', 'PLUS_MINUS', 'EFG_PCT', 'TOV_PCT', 'ORB_PCT', 'FTR', 'REST_DAYS', 'IS_HOME',
        'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
    ]
    return df[cols]

def update_and_save(file_path, new_df, id_cols):
    """
    Simplified pipeline: Load -> Merge -> Deduplicate -> Sort -> Save
    """
    if os.path.exists(file_path):
        # Load existing data (keep IDs as strings to prevent data loss)
        old_df = pd.read_csv(file_path, dtype={'GAME_ID': str})
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Ensure ID columns are strings for consistent deduplication
    for col in id_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype(str)

    # Deduplicate (Keep Last = Keep Newest)
    before = len(combined)
    combined = combined.drop_duplicates(subset=id_cols, keep='last')
    
    # Sort & Format Date
    if 'GAME_DATE' in combined.columns:
        combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'], format='mixed', errors='coerce')
        combined = combined.sort_values(by=['GAME_DATE', 'GAME_ID'], ascending=[True, True])
        combined['GAME_DATE'] = combined['GAME_DATE'].dt.strftime('%Y-%m-%d')

    print(f"   > Saving to {file_path}: {len(combined)} rows ({before - len(combined)} duplicates removed)")
    combined.to_csv(file_path, index=False)

def run_pipeline():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- TEAMS ---
    print("\n🏀 Updating TEAM Games...")
    existing_games = pd.read_csv(GAMES_FILE, dtype={'GAME_ID': str}) if os.path.exists(GAMES_FILE) else None
    
    seasons, start_date = get_fetch_params(existing_games)
    
    raw_games = fetch_games(seasons, start_date=start_date, mode='Team')
    
    if not raw_games.empty:
        processed_games = process_team_data(raw_games)
        update_and_save(GAMES_FILE, processed_games, ['GAME_ID', 'TEAM_ABBREVIATION'])
    else:
        print("⚠️ No new team data found.")

    # --- PLAYERS ---
    print("\n👤 Updating PLAYER Stats...")
    existing_players = pd.read_csv(PLAYERS_FILE, dtype={'GAME_ID': str}) if os.path.exists(PLAYERS_FILE) else None
    
    seasons, start_date = get_fetch_params(existing_players)
    
    raw_players = fetch_games(seasons, start_date=start_date, mode='Player')
    
    if not raw_players.empty:
        if 'GAME_DATE' in raw_players.columns:
            raw_players['GAME_DATE'] = pd.to_datetime(raw_players['GAME_DATE']).dt.strftime('%Y-%m-%d')
            
        update_and_save(PLAYERS_FILE, raw_players, ['GAME_ID', 'TEAM_ABBREVIATION', 'PLAYER_ID'])
    else:
        print("⚠️ No new player data found.")

if __name__ == "__main__":
    run_pipeline()