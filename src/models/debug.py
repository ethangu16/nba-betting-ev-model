import pandas as pd
import numpy as np
import os

# --- CONFIG ---
PLAYER_PATH = 'data/raw/nba_player_stats.csv'
OUTPUT_PATH = 'results/player_form_history.csv'
TARGET_PLAYER = 'Kon Knueppel'
SEASON_START = '2025-10-01'

def audit_player_history():
    if not os.path.exists(PLAYER_PATH):
        print(f"❌ Error: {PLAYER_PATH} not found.")
        return

    df = pd.read_csv(PLAYER_PATH)
    
    # 1. Setup Data
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        season_df = df[df['GAME_DATE'] >= pd.Timestamp(SEASON_START)].copy()
    else:
        season_df = df.copy()

    # 2. Calculate Game Score (The Input)
    # This formula measures "Single Game Productivity"
    season_df['GAME_SCORE'] = (
        season_df['PTS'] + 0.4 * season_df['FGM'] - 0.7 * season_df['FGA'] - 0.4 * (season_df['FTA'] - season_df['FTM']) + 
        0.7 * season_df['OREB'] + 0.3 * season_df['DREB'] + season_df['STL'] + 0.7 * season_df['AST'] + 
        0.7 * season_df['BLK'] - 0.4 * season_df['PF'] - season_df['TOV']
    )

    # 3. Calculate Rolling Form (The Output)
    # We will do this manually to show the math: New = Old + alpha * (Game - Old)
    season_df = season_df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    # Span=10 corresponds to alpha = 2/(10+1) = 0.1818
    alpha = 2 / (10 + 1)
    
    full_history = []

    print(f"🧮 Tracing history for {len(season_df['PLAYER_ID'].unique())} players...")

    for pid, group in season_df.groupby('PLAYER_ID'):
        player_name = group['PLAYER_NAME'].iloc[0]
        team = group['TEAM_ABBREVIATION'].iloc[-1]
        
        # Initialize Form (Start at 0 or first game score)
        current_form = group['GAME_SCORE'].iloc[0] 
        
        for idx, row in group.iterrows():
            game_score = row['GAME_SCORE']
            
            # The EWMA Math Step
            # New_Form = (Current_Game * Alpha) + (Old_Form * (1 - Alpha))
            prev_form = current_form
            current_form = (game_score * alpha) + (prev_form * (1 - alpha))
            
            full_history.append({
                'Date': row['GAME_DATE'].date(),
                'Player': player_name,
                'Team': team,
                'Matchup': row['MATCHUP'],
                'PTS': row['PTS'],
                'Game_Score': round(game_score, 1),
                'Prev_Form': round(prev_form, 1),
                'New_Form': round(current_form, 1),
                'Change': round(current_form - prev_form, 2)
            })

    # 4. Save Full Log
    history_df = pd.DataFrame(full_history)
    history_df.to_csv(OUTPUT_PATH, index=False)
    
    # 5. Print Specific Audit
    target_df = history_df[history_df['Player'].str.contains(TARGET_PLAYER, case=False)]
    
    if target_df.empty:
        print(f"⚠️ Player '{TARGET_PLAYER}' not found.")
    else:
        print("\n" + "="*80)
        print(f"🔎 DEEP DIVE: {TARGET_PLAYER.upper()}")
        print("="*80)
        print(f"{'DATE':<12} | {'MATCHUP':<12} | {'PTS':<4} | {'G_SCORE':<8} | {'PREV':<6} | {'NEW':<6} | {'IMPACT'}")
        print("-" * 80)
        
        # Show last 10 games
        for i, row in target_df.tail(10).iterrows():
            impact_symbol = "🟢" if row['Change'] > 0 else "🔻"
            print(f"{str(row['Date']):<12} | {row['Matchup']:<12} | {row['PTS']:<4} | {row['Game_Score']:<8} | {row['Prev_Form']:<6} | {row['New_Form']:<6} | {impact_symbol} {row['Change']:+.2f}")
            
        print("-" * 80)
        print("math key: New_Form = (Game_Score * 0.1818) + (Prev_Form * 0.8182)")
        
    print(f"\n✅ Full history saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    audit_player_history()