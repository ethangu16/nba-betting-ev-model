import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV # <--- Imported GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
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

    # 1. Define Features (Updated with Pace & Fatigue)
    features = [
            'ELO_TEAM', 'ELO_OPP', 
            'IS_HOME', 
            'IS_B2B', 'IS_3IN4',              # Fatigue
            'ROLL_OFF_RTG', 'ROLL_DEF_RTG',   # Advanced Stats
            'ROLL_PACE', 
            'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
            'ROLL_ROSTER_TALENT_SCORE'
    ]
    
    target = 'TARGET_WIN'

    # 2. Check & Clean Data
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"❌ Critical Error: Missing columns in CSV: {missing_cols}")
        print("   Did you run the NEW engineer.py?")
        return

    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(df)} games with {len(features)} features...")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- GRID SEARCH IMPLEMENTATION ---
    
    # 4. Define the "Hyperparameter Grid" (The options to test)
    param_grid = {
        'n_estimators': [50, 100, 200],      # How many trees?
        'max_depth': [3, 4, 5],              # How deep can each tree go?
        'learning_rate': [0.01, 0.1, 0.2],   # How fast to correct errors?
        'subsample': [0.8, 1.0],             # Use 80% or 100% of rows per tree?
        'colsample_bytree': [0.8, 1.0]       # Use 80% or 100% of columns per tree?
    }
    
    # 5. Initialize Base Model
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    # 6. Initialize Grid Search
    # cv=3 means "Cross Validation": Splits training data into 3 chunks to verify accuracy.
    # n_jobs=-1 means "Use all CPU cores" to make it faster.
    print(f"\n🔍 Starting Grid Search (Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinations)...")
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # 7. Run the Search (This takes longer than a normal train)
    grid_search.fit(X_train, y_train)
    
    # 8. Get the Winner
    best_model = grid_search.best_estimator_
    print(f"\n✅ Best Hyperparameters found: {grid_search.best_params_}")

    # 9. Evaluate using the BEST model
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Final Accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred))

    # 10. Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved optimized model to {MODEL_PATH}")

if __name__ == "__main__":
    train()