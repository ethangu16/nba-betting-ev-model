import pandas as pd
import xgboost as xgb
import numpy as np
import joblib

# CONFIGURATION
DATA_PATH = 'data/processed/nba_model_ready.csv'
MODEL_PATH = 'models/nba_xgb_model.json' # Check if it's .json or .joblib

def get_implied_prob(moneyline):
    if moneyline < 0: return (-moneyline) / (-moneyline + 100)
    else: return 100 / (moneyline + 100)

def debug():
    print(f"--- DIAGNOSTIC BACKTEST ---")
    
    # 1. Check Data Volume
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Total Rows in Dataset: {len(df)}")
    except FileNotFoundError:
        print("❌ Error: nba_model_ready.csv not found.")
        return

    # 2. Check Test Split
    split_index = int(len(df) * 0.80)
    test_df = df.iloc[split_index:].copy()
    print(f"Test Set Size (Last 20%): {len(test_df)} games")
    
    if len(test_df) == 0:
        print("❌ Error: Test Set is empty! Your dataset is too small.")
        return

    # 3. Check Odds Data
    if 'MONEYLINE' not in test_df.columns:
        print("❌ Error: 'MONEYLINE' column missing.")
        return
        
    missing_odds = test_df['MONEYLINE'].isna().sum()
    print(f"Rows with missing Moneyline: {missing_odds}")
    
    if missing_odds == len(test_df):
        print("❌ Error: All odds are NaN. Backtest cannot run.")
        return

    # 4. Check Model Predictions & Edge
    print("\n--- CHECKING PREDICTIONS ---")
    try:
        model = xgb.XGBClassifier()
        model.load_model('models/nba_xgb_model.json')
    except:
        try:
            model = joblib.load('models/nba_xgb_model.joblib')
        except:
            print("❌ Error: Could not load model.")
            return

    # Features (Must match training!)
    features = [
        'ELO_TEAM', 'ELO_OPP', 
        'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME',
        'ROLL_PTS', 'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 
        'ROLL_ORB_PCT', 'ROLL_FTR', 'ROLL_PLUS_MINUS'
    ]
    
    # Generate Probs
    try:
        probs = model.predict_proba(test_df[features])[:, 1]
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        print(f"Expected Features: {features}")
        print(f"Found Features: {test_df.columns.tolist()}")
        return

    test_df['MODEL_PROB'] = probs
    
    # Calculate Edges
    edges = []
    for _, row in test_df.iterrows():
        odds = row['MONEYLINE']
        if pd.isna(odds): continue
        
        implied = get_implied_prob(odds)
        edge = row['MODEL_PROB'] - implied
        edges.append(edge)
    
    edges = np.array(edges)
    
    print(f"Min Edge: {edges.min():.4f}")
    print(f"Max Edge: {edges.max():.4f}")
    print(f"Avg Edge: {edges.mean():.4f}")
    
    potential_bets = (edges > 0.01).sum()
    print(f"\nPotential Bets (Edge > 1%): {potential_bets}")
    
    if potential_bets == 0:
        print("⚠️ PROBLEM: Your model is 'hugging' the market line too closely.")
        print("Try lowering the threshold in backtest.py to > 0.00 (0%).")
    else:
        print("✅ Bets exist! If backtest.py shows 0, check your logic loop.")

if __name__ == "__main__":
    debug()