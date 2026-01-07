import pandas as pd
import numpy as np
import os

# --- CONFIG ---
DATA_PATH = 'data/processed/nba_model.csv'
OUTPUT_CSV = 'results/elo_game_log.csv'
SEASON_START = '2025-10-01'
K_FACTOR = 20  # Standard NBA Elo K-Factor

def generate_elo_table():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Filter for 2025-26 Season
    season_df = df[df['GAME_DATE'] >= pd.Timestamp(SEASON_START)].copy()
    season_df = season_df.sort_values('GAME_DATE', ascending=False) # Newest first

    print(f"📊 Calculating Elo Shifts for {len(season_df)} games...")

    # 3. Calculate Win Probability (Expected Score)
    # Formula: 1 / (1 + 10^((OppElo - TeamElo) / 400))
    season_df['PROB_WIN'] = 1 / (1 + 10 ** ((season_df['ELO_OPP'] - season_df['ELO_TEAM']) / 400))

    # 4. Determine Actual Result (1 for Win, 0 for Loss)
    season_df['ACTUAL_SCORE'] = season_df['WL'].apply(lambda x: 1.0 if x == 'W' else 0.0)

    # 5. Calculate Elo Change
    # Change = K * (Actual - Expected)
    season_df['ELO_CHANGE'] = K_FACTOR * (season_df['ACTUAL_SCORE'] - season_df['PROB_WIN'])
    
    # 6. Calculate Post-Game Elo
    season_df['ELO_POST'] = season_df['ELO_TEAM'] + season_df['ELO_CHANGE']

    # 7. Format for Clean Output
    output_df = season_df[[
        'GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 
        'ELO_TEAM', 'PROB_WIN', 'ELO_CHANGE', 'ELO_POST'
    ]].copy()

    # Rename Columns for Readability
    output_df.columns = [
        'Date', 'Team', 'Matchup', 'Result', 
        'Start Elo', 'Win Prob', 'Elo Change', 'New Elo'
    ]

    # Round values
    output_df['Start Elo'] = output_df['Start Elo'].round(1)
    output_df['New Elo'] = output_df['New Elo'].round(1)
    output_df['Elo Change'] = output_df['Elo Change'].round(2)
    output_df['Win Prob'] = (output_df['Win Prob'] * 100).round(1).astype(str) + '%'

    # Add a "+" sign to positive gains for easier reading
    output_df['Elo Change'] = output_df['Elo Change'].apply(lambda x: f"+{x}" if x > 0 else f"{x}")

    # 8. Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*80)
    print(f"📋 RECENT ELO MOVERS (Last 10 Games)")
    print("="*80)
    print(output_df.head(10).to_string(index=False))
    print("="*80)
    print(f"✅ Full Game Log saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_elo_table()