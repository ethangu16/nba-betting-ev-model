import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- CONFIGURATION ---
DATA_PATH = 'data/processed/nba_model_ready.csv'
MODEL_PATH_JSON = 'models/nba_xgb_model.json'
MODEL_PATH_JOBLIB = 'models/nba_xgb_model.joblib'

# BANKROLL SETTINGS
INITIAL_BANKROLL = 10000
KELLY_FRACTION = 0.25  # Quarter Kelly (Safety: Bet 25% of the optimal amount)
MAX_BET_PCT = 0.05     # Never bet more than 5% of bankroll on one game
EDGE_THRESHOLD = 0.01  # Only bet if our edge is > 1% (Filters out noise)

def get_implied_prob(moneyline):
    """
    Converts American Odds (-150, +130) to Implied Probability (0-1).
    """
    if pd.isna(moneyline): return None
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)

def get_decimal_odds(moneyline):
    """
    Converts American Odds to Decimal Odds (e.g. -110 -> 1.91)
    """
    if pd.isna(moneyline): return None
    if moneyline < 0:
        return 1 + (100 / -moneyline)
    else:
        return 1 + (moneyline / 100)

def load_model():
    """Smart loader that checks for json or joblib"""
    if os.path.exists(MODEL_PATH_JSON):
        print(f"Loading XGBoost from {MODEL_PATH_JSON}...")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH_JSON)
        return model
    elif os.path.exists(MODEL_PATH_JOBLIB):
        print(f"Loading XGBoost from {MODEL_PATH_JOBLIB}...")
        return joblib.load(MODEL_PATH_JOBLIB)
    else:
        raise FileNotFoundError("❌ Could not find trained model. Run train_model.py first.")

def backtest():
    print("--- STARTING BACKTEST ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('GAME_DATE')
    
    # 2. Split Test Set (Last 20% - Must match training split!)
    split_index = int(len(df) * 0.80)
    test_df = df.iloc[split_index:].copy()
    print(f"Backtesting on {len(test_df)} games (The 'Future')...")
    
    # 3. Load Model & Predict
    try:
        model = load_model()
    except Exception as e:
        print(e)
        return

    # Define Features (MUST MATCH TRAINING EXACTLY)
    features = [
        'ELO_TEAM', 'ELO_OPP', 
        'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME',
        'ROLL_PTS', 'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 
        'ROLL_ORB_PCT', 'ROLL_FTR', 'ROLL_PLUS_MINUS'
    ]
    
    print("Generating predictions...")
    probs = model.predict_proba(test_df[features])[:, 1]
    test_df['MODEL_PROB'] = probs
    
    # 4. Run Simulation
    bankroll = INITIAL_BANKROLL
    history = [INITIAL_BANKROLL]
    
    stats = {
        'bets_placed': 0,
        'wins': 0,
        'skipped_missing_odds': 0,
        'skipped_low_edge': 0
    }
    
    print("\n--- SIMULATING BETS ---")
    
    for index, row in test_df.iterrows():
        # Get Betting Data
        odds = row.get('MONEYLINE')
        
        # SKIP if odds are missing
        if pd.isna(odds):
            stats['skipped_missing_odds'] += 1
            continue
            
        implied_prob = get_implied_prob(odds)
        decimal_odds = get_decimal_odds(odds)
        model_prob = row['MODEL_PROB']
        
        # Calculate Edge
        edge = model_prob - implied_prob
        
        # DECISION: Bet if Edge > Threshold
        if edge > EDGE_THRESHOLD:
            stats['bets_placed'] += 1
            
            # Kelly Criterion
            b = decimal_odds - 1
            p = model_prob
            q = 1 - p
            
            if b <= 0: continue 
            
            kelly_pct = (b * p - q) / b
            
            # Risk Management
            stake_pct = max(0, kelly_pct * KELLY_FRACTION)
            stake_pct = min(stake_pct, MAX_BET_PCT) # Cap max bet
            
            stake = bankroll * stake_pct
            
            # Outcome
            actual_win = row['TARGET_WIN']
            if actual_win == 1:
                profit = stake * b
                bankroll += profit
                stats['wins'] += 1
            else:
                bankroll -= stake
        
        else:
            stats['skipped_low_edge'] += 1
            
        # Track bankroll history
        history.append(bankroll)

    # 5. Report Results
    total_return = ((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL) * 100
    win_rate = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
    
    print("\n" + "="*30)
    print(f"FINAL RESULT: ${bankroll:,.2f}")
    print(f"Return (ROI): {total_return:.2f}%")
    print("="*30)
    print(f"Total Games in Test Set: {len(test_df)}")
    print(f"Bets Placed:             {stats['bets_placed']}")
    print(f"Win Rate:                {win_rate:.2f}%")
    print(f"Skipped (Missing Odds):  {stats['skipped_missing_odds']} (Fix this by patching odds!)")
    print(f"Skipped (Low Edge):      {stats['skipped_low_edge']}")
    
    # 6. Plot
    if len(history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(history, label='Bankroll Strategy')
        plt.axhline(y=INITIAL_BANKROLL, color='r', linestyle='--', label='Start')
        plt.title(f"Backtest Equity Curve (ROI: {total_return:.2f}%)")
        plt.xlabel("Games Bet")
        plt.ylabel("Bankroll ($)")
        plt.legend()
        plt.grid(True)
        # Check if we are in a notebook or terminal
        try:
            plt.show()
        except:
            print("Plot generated (GUI not available).")
    else:
        print("\n⚠️ No bets were placed, so no graph was generated.")

if __name__ == "__main__":
    backtest()