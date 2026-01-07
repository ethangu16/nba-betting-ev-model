import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import os
import numpy as np

# --- CONFIGURATION ---
INPUT_PATH = 'data/processed/nba_model.csv'
MODEL_PATH = 'models/nba_xgb_model.joblib'

def train():
    print(f"\nLOADING DATA from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print("❌ Error: Data file not found. Run engineer.py first.")
        return

    # 1. Define Features
    # 🟢 UPDATE THIS LIST in train_model.py AND predict_today.py
    features = [
        # Core Context
        'ELO_TEAM', 'ELO_OPP', 'IS_HOME', 'IS_B2B', 'IS_3IN4',
        'ROSTER_TALENT_SCORE', # Use the raw score, or a rolling version if you made one
        
        # 🟢 NEW: The 3 Time Windows for Every Stat
        # SMA 20 (Stability)
        'SMA_20_OFF_RTG', 'SMA_20_DEF_RTG', 'SMA_20_PACE', 'SMA_20_EFG_PCT',
        'SMA_20_TOV_PCT', 'SMA_20_ORB_PCT', 'SMA_20_FTR', 'SMA_20_ROSTER_TALENT_SCORE',
        
        # EWMA 10 (Momentum)
        'EWMA_10_OFF_RTG', 'EWMA_10_DEF_RTG', 'EWMA_10_PACE', 'EWMA_10_EFG_PCT',
        'EWMA_10_TOV_PCT', 'EWMA_10_ORB_PCT', 'EWMA_10_FTR', 'EWMA_10_ROSTER_TALENT_SCORE',

        # EWMA 5 (Hot/Cold Streaks)
        'EWMA_5_OFF_RTG', 'EWMA_5_DEF_RTG', 'EWMA_5_PACE', 'EWMA_5_EFG_PCT',
        'EWMA_5_TOV_PCT', 'EWMA_5_ORB_PCT', 'EWMA_5_FTR', 'EWMA_5_ROSTER_TALENT_SCORE',
    ]
    
    target = 'TARGET_WIN'

    # 2. Check & Clean Data
    initial_len = len(df)
    df = df.dropna(subset=features + [target])
    print(f"   Dropped {initial_len - len(df)} rows with missing values.")

    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        print(f"   ✅ Date Range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
    else:
        print("⚠️ Warning: GAME_DATE not found. Using index assuming chronological order.")
    
    X = df[features]
    y = df[target]
    
    # --- 3. SPLIT DATA (TIME SERIES SPLIT) ---
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    train_dates = df['GAME_DATE'].iloc[:split_idx]
    
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    test_dates = df['GAME_DATE'].iloc[split_idx:]
    
    print("\n✂️  SPLIT AUDIT:")
    print(f"   Train: {len(X_train)} games | End Date: {train_dates.max().date()}")
    print(f"   Test:  {len(X_test)} games | Start Date: {test_dates.min().date()}")

    # --- GRID SEARCH ---
    print("\n🔍 STARTING GRID SEARCH...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    xgb_base = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    print(f"   ✅ Best Params: {grid_search.best_params_}")

    # --- FEATURE IMPORTANCE AUDIT ---
    print("\n🧠 MODEL BRAIN (Feature Importance):")
    imps = best_xgb.feature_importances_
    # Pair feature names with importance
    feature_imp = pd.DataFrame({'Feature': features, 'Importance': imps})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    
    # Print bar chart style
    for index, row in feature_imp.iterrows():
        bar_len = int(row['Importance'] * 40) # Scale for visual
        bar = '█' * bar_len
        print(f"   {row['Feature']:<25} | {bar} ({row['Importance']:.1%})")

    # --- CALIBRATION ---
    print("\n🔧 Calibrating Probabilities...")
    calibrated_model = CalibratedClassifierCV(best_xgb, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)

    # --- EVALUATION ---
    print("\n🎯 FINAL EVALUATION (Test Set):")
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   Accuracy:    {acc:.1%}")
    print(f"   Brier Score: {brier:.4f} (Lower is better)")
    print(f"   Win % in Test Set: {y_test.mean():.1%}")
    print(f"\n   Confusion Matrix:")
    print(f"      [ TN {cm[0][0]}  FP {cm[0][1]} ]")
    print(f"      [ FN {cm[1][0]}  TP {cm[1][1]} ]")
    
    # Probability Distribution Check
    print("\n   📊 Probability Check (Are we confident?):")
    print(f"      Avg Confidence: {np.mean(np.abs(y_prob - 0.5)) + 0.5:.1%}")
    print(f"      Bets > 60%:     {np.sum(y_prob > 0.6)} games")
    print(f"      Bets < 40%:     {np.sum(y_prob < 0.4)} games")

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()