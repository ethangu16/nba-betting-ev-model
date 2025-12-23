import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- CONFIGURATION ---
DATA_PATH = 'data/processed/nba_model_with_odds.csv'
MODEL_PATH_JSON = 'models/nba_xgb_model.json'
MODEL_PATH_JOBLIB = 'models/nba_xgb_model.joblib'

# BANKROLL SETTINGS
INITIAL_BANKROLL = 10000
KELLY_FRACTION = 0.25  # Safety: Bet 25% of the optimal Kelly amount
MAX_BET_PCT = 0.05     # Cap: Never bet more than 5% of bankroll on one game
EDGE_THRESHOLD = 0.01  # Threshold: Only bet if Edge > 1%

def get_implied_prob(moneyline):
    """Converts American Odds (-150, +130) to Implied Probability (0-1)."""
    if pd.isna(moneyline) or moneyline == 0: return None
    if moneyline < 0:
        return (-moneyline) / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)

def get_decimal_odds(moneyline):
    """Converts American Odds to Decimal Odds (e.g. -110 -> 1.91)"""
    if pd.isna(moneyline) or moneyline == 0: return None
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
        raise FileNotFoundError("❌ Could not find trained model.")
def backtest():
    print("--- STARTING BACKTEST ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"📄 Loaded {len(df)} rows from dataset.")

    # --- 🛠️ FEATURE REPAIR BLOCK (Fixing Missing Columns) ---
    # The model needs 'ROLL_PTS', but it's missing. We calculate it from 'PTS'.
    if 'ROLL_PTS' not in df.columns and 'PTS' in df.columns:
        print("⚠️ 'ROLL_PTS' column missing. Calculating it on the fly...")
        # Ensure dates are datetime for correct sorting
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE')
        
        # Calculate 10-game rolling average of points (shifted by 1 to avoid data leakage)
        df['ROLL_PTS'] = df.groupby('TEAM_ID')['PTS'].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )
        print("✅ 'ROLL_PTS' successfully generated.")

    # --- 🛠️ DATA PATCHING BLOCK (Odds) ---
    if 'MONEYLINE' not in df.columns:
        print("🔍 Mapping 'HOME_ML' / 'AWAY_ML' to 'MONEYLINE' column...")
        required_cols = {'HOME_ML', 'AWAY_ML', 'IS_HOME'}
        if required_cols.issubset(df.columns):
            df['MONEYLINE'] = np.where(df['IS_HOME'] == 1, df['HOME_ML'], df['AWAY_ML'])
            print(f"✅ Successfully mapped odds to 'MONEYLINE'.")
        else:
            print(f"❌ Critical: Missing columns in CSV: {required_cols - set(df.columns)}")
            return

    # --- 🧹 FILTERING BLOCK ---
    # Drop rows where MONEYLINE is NaN or 0, AND where ROLL_PTS is NaN (early season games)
    df_clean = df.dropna(subset=['MONEYLINE', 'ROLL_PTS']).copy()
    df_clean = df_clean[df_clean['MONEYLINE'] != 0]

    dropped_count = len(df) - len(df_clean)
    print(f"📉 Filtered Data: {len(df_clean)} valid betting opportunities.")
    print(f"   (Skipped {dropped_count} rows due to missing odds or rolling stats)")

    if len(df_clean) == 0:
        print("❌ No valid games left. Check your data source.")
        return

    df_clean = df_clean.sort_values('GAME_DATE')
    
    # 2. Split Test Set (Last 20% of VALID games)
    split_index = int(len(df_clean) * 0.80)
    test_df = df_clean.iloc[split_index:].copy()
    print(f"🚀 Backtesting on {len(test_df)} games...")
    
    # 3. Load Model
    try:
        model = load_model()
    except Exception as e:
        print(e)
        return

    # Define Features
    features = [
        'ELO_TEAM', 'ELO_OPP', 
        'IS_HOME', 'IS_B2B', 'IS_3IN4',
        'ROLL_OFF_RTG', 'ROLL_DEF_RTG', 'ROLL_PACE',
        'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT',
        'ROLL_FTR', 'ROLL_ROSTER_TALENT_SCORE'
    ]
    
    # Verify features exist
    missing_feats = [f for f in features if f not in test_df.columns]
    if missing_feats:
        print(f"❌ Still missing columns: {missing_feats}")
        return

    print("🧠 Generating AI predictions...")
    probs = model.predict_proba(test_df[features])[:, 1]
    test_df['MODEL_PROB'] = probs
    
    # 4. Run Simulation
    bankroll = INITIAL_BANKROLL
    history = [INITIAL_BANKROLL]
    
    stats = {'bets_placed': 0, 'wins': 0, 'losses': 0, 'skipped_low_edge': 0}
    
    print("\n--- SIMULATING WAGERS ---")
    
    for index, row in test_df.iterrows():
        odds = row['MONEYLINE']
        implied_prob = get_implied_prob(odds)
        decimal_odds = get_decimal_odds(odds)
        model_prob = row['MODEL_PROB']
        
        if implied_prob is None: continue
        
        edge = model_prob - implied_prob
        
        if edge > EDGE_THRESHOLD:
            stats['bets_placed'] += 1
            b = decimal_odds - 1
            p = model_prob
            q = 1 - p
            
            if b <= 0: continue 
            
            kelly_pct = (b * p - q) / b
            stake_pct = max(0, min(kelly_pct * KELLY_FRACTION, MAX_BET_PCT))
            stake = bankroll * stake_pct
            
            if row['TARGET_WIN'] == 1:
                bankroll += stake * b
                stats['wins'] += 1
            else:
                bankroll -= stake
                stats['losses'] += 1
        else:
            stats['skipped_low_edge'] += 1
            
        history.append(bankroll)

    # 5. Report & Plot
    total_return = ((bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL) * 100
    win_rate = (stats['wins'] / stats['bets_placed'] * 100) if stats['bets_placed'] > 0 else 0
    
    print("\n" + "="*40)
    print(f"💰 FINAL BANKROLL: ${bankroll:,.2f}")
    print(f"📈 Total ROI:      {total_return:+.2f}%")
    print("="*40)
    print(f"Bets Placed:       {stats['bets_placed']}")
    print(f"Win Rate:          {win_rate:.2f}%")
    
    if len(history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(history, label='AI Strategy', color='green')
        plt.axhline(y=INITIAL_BANKROLL, color='r', linestyle='--')
        plt.title(f"Backtest ROI: {total_return:.2f}%")
        plt.grid(True, alpha=0.3)
        plt.savefig("results/backtest_chart.png")
        print("✅ Chart saved to results/backtest_chart.png")

if __name__ == "__main__":
    backtest()