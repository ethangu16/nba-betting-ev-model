import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV  # <--- NEW IMPORT
import os

# --- CONFIGURATION ---
INPUT_PATH = 'data/processed/nba_model.csv'
MODEL_PATH = 'models/nba_xgb_model.joblib'

def train():
    print(f"Loading data from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print("❌ Error: Data file not found. Run engineer.py first.")
        return

    # 1. Define Features
    features = [
            'ELO_TEAM', 'ELO_OPP', 
            'IS_HOME', 
            'IS_B2B', 'IS_3IN4',              
            'ROLL_OFF_RTG', 'ROLL_DEF_RTG',   
            'ROLL_PACE', 
            'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
            'ROLL_ROSTER_TALENT_SCORE'
    ]
    
    target = 'TARGET_WIN'

    # 2. Check & Clean Data
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"❌ Critical Error: Missing columns: {missing_cols}")
        return

    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(df)} games...")

    # 3. Split Data (Train / Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- GRID SEARCH ---
    
    # 4. Define Hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # 5. Initialize Base XGBoost
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
    )
    
    # 6. Run Grid Search
    print(f"\n🔍 Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"\n✅ Best XGBoost Params: {grid_search.best_params_}")

    # --- CALIBRATION STEP (NEW) ---
    
    print("\n🔧 Calibrating Model Probabilities (Platt Scaling)...")
    
    # Wrap the best XGBoost model in a Calibrator
    # method='sigmoid' is "Platt Scaling" (best for datasets < 100k rows)
    # cv=5 means it splits Training data 5 ways to learn the calibration map safely
    calibrated_model = CalibratedClassifierCV(best_xgb, method='sigmoid', cv=5)
    
    # Fit the CALIBRATED model on training data
    calibrated_model.fit(X_train, y_train)

    # 9. Evaluate (Using the CALIBRATED model)
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob) # Lower Brier score is better!
    
    print(f"\n🎯 Final Accuracy: {acc:.1%}")
    print(f"📉 Brier Score (Calibration Error): {brier:.4f} (Lower is better)")
    print(classification_report(y_test, y_pred))

    # 10. Save the CALIBRATED model
    # Note: This saves the wrapper, which contains the XGBoost model inside it
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"✅ Saved CALIBRATED model to {MODEL_PATH}")

if __name__ == "__main__":
    train()