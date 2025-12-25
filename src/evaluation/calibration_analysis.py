import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/nba_xgb_model.joblib'
DATA_PATH = 'data/processed/nba_model.csv'
OUTPUT_DIR = 'results'
OUTPUT_IMG = f'{OUTPUT_DIR}/calibration_plot.png'

def check_calibration():
    print("--- 📊 MODEL CALIBRATION CHECK ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("❌ Error: Missing data or model file. Run engineer.py and train.py first.")
        return

    # 2. Define Features (Must match training exactly)
    features = [
        'ELO_TEAM', 'ELO_OPP', 'IS_HOME', 'IS_B2B', 'IS_3IN4',
        'ROLL_OFF_RTG', 'ROLL_DEF_RTG', 'ROLL_PACE',
        'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
        'ROLL_ROSTER_TALENT_SCORE'
    ]
    
    # 3. Prepare Test Set (Last 20% of games)
    # We MUST sort by date to test on the "future"
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE')
    
    # Simulate the exact test split used in training
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # Drop rows with missing features
    test_df = test_df.dropna(subset=features + ['TARGET_WIN'])
    
    X_test = test_df[features]
    y_test = test_df['TARGET_WIN']
    
    print(f"Testing on {len(test_df)} recent games...")

    # 4. Get Probabilities
    # Note: If the model is a CalibratedClassifierCV, predict_proba is already calibrated
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"❌ Error predicting: {e}")
        return

    # 5. Metric 1: Brier Score
    brier = brier_score_loss(y_test, probs)
    print(f"\n📉 Brier Score: {brier:.4f}")
    
    # Benchmark against 50/50 guessing (0.25)
    skill_score = 1 - (brier / 0.25)
    print(f"   Skill Score: {skill_score:.2%} (vs random guessing)")
    
    if brier < 0.20:
        print("   ✅ Excellent (World Class)")
    elif brier < 0.22:
        print("   ✅ Good (Profitable Baseline)")
    elif brier < 0.25:
        print("   ⚠️ Mediocre (Coin Flip territory)")
    else:
        print("   ❌ Poor (Worse than random)")

    # 6. Metric 2: Reliability Diagram 
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    # Your model's line
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Your Model')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual Win Rate)')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(OUTPUT_IMG)
    print(f"\n📈 Saved calibration plot to {OUTPUT_IMG}")
    
    # 7. Confidence Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(probs, bins=20, range=(0,1), edgecolor='black', alpha=0.7)
    plt.title('Prediction Distribution (Where is the model confident?)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.savefig(f'{OUTPUT_DIR}/prediction_hist.png')
    print(f"📊 Saved prediction histogram to {OUTPUT_DIR}/prediction_hist.png")

if __name__ == "__main__":
    check_calibration()