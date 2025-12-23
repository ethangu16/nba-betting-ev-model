import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import math

# Load Data
df = pd.read_csv('data/processed/nba_model.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values('GAME_DATE')

def get_mov_multiplier(mov, elo_diff):
    """
    538's Margin of Victory Multiplier.
    mov: Margin of Victory (Absolute value)
    elo_diff: Difference in ratings (Winner - Loser)
    """
    # The formula essentially gives more credit for blowouts, but has diminishing returns
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)

def run_elo_simulation(k_factor, home_advantage, use_mov=True):
    team_elos = {team: 1500 for team in df['TEAM_ABBREVIATION'].unique()}
    predictions = []
    actuals = []
    
    for index, row in df.iterrows():
        team = row['TEAM_ABBREVIATION']
        opp = row['OPP_ABBREVIATION']
        
        r_team = team_elos.get(team, 1500)
        r_opp = team_elos.get(opp, 1500)
        
        # 1. Prediction Step
        home_boost = home_advantage if row['IS_HOME'] == 1 else 0
        dr = r_team + home_boost - r_opp
        prob_win = 1 / (1 + 10 ** (-dr / 400))
        
        predictions.append(prob_win)
        actual_win = 1 if row['WL'] == 'W' else 0
        actuals.append(actual_win)
        
        # 2. Update Step (with MOV)
        # Note: Calculating MOV on the fly in a loop is tricky with single rows. 
        # Better strategy: Use the pre-calculated 'PLUS_MINUS' column if you have it
        # Assuming 'PLUS_MINUS' exists and is positive for wins:
        
        mov = abs(row['PLUS_MINUS']) if not pd.isna(row['PLUS_MINUS']) else 0
        
        if use_mov:
            # Elo difference relative to the WINNER
            elo_diff_winner = dr if actual_win == 1 else -dr
            multiplier = get_mov_multiplier(mov, elo_diff_winner)
        else:
            multiplier = 1.0
            
        shift = k_factor * multiplier * (actual_win - prob_win)
        
        # Update ratings
        team_elos[team] = r_team + shift
        team_elos[opp] = r_opp - shift
        
    return log_loss(actuals, predictions)

# Run Optimization
print("Searching for optimal MOV Elo settings...")
best_score = float('inf')
best_params = {}

for k in [10, 15, 20, 25]: # K is usually smaller when using MOV multiplier
    for home_adv in [60, 80, 100, 120]:
        score = run_elo_simulation(k, home_adv, use_mov=True)
        print(f"K={k}, HomeAdv={home_adv} -> LogLoss: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_params = {'k': k, 'home_adv': home_adv}

print(f"\n✅ BEST SETTINGS: K={best_params['k']}, HomeAdv={best_params['home_adv']}")