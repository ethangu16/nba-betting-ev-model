import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# --- CONFIGURATION ---
DATA_PATH = 'data/processed/nba_model_ready.csv'
MODEL_PATH = 'models/nba_xgb_model.joblib' # We will save the trained brain here

def train():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('GAME_DATE') # CRITICAL: Ensure time order
    
    # 2. Define Features (X) and Target (y)
    # We remove ID columns (names, dates) and keep only the "Math"
    features = [
        'ELO_TEAM', 'ELO_OPP', 
        'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME',
        'ROLL_PTS', 'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 
        'ROLL_ORB_PCT', 'ROLL_FTR', 'ROLL_PLUS_MINUS'
    ]
    target = 'TARGET_WIN'
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(features)} features: {features}")
    
    # 3. Time-Series Split
    # We take the last 20% of games as our "Test Set" (Simulating the future)
    split_index = int(len(df) * 0.80)
    
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    # Get the odds for the test set (for evaluation later, not training)
    test_odds = df.iloc[split_index:][['MONEYLINE', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION']]
    
    print(f"Train Set: {len(X_train)} games")
    print(f"Test Set: {len(X_test)} games")
    
    # 4. Train XGBoost
    # Hyperparameters: These are standard "good" starts for sports data
    model = xgb.XGBClassifier(
        n_estimators=100,      # Number of "trees"
        learning_rate=0.05,    # How fast it learns (lower = slower but more precise)
        max_depth=3,           # Shallow trees prevent overfitting
        eval_metric='logloss', # Optimize for PROBABILITY, not just Win/Loss
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    # Get probabilities (e.g., 0.65 chance of winning)
    probs = model.predict_proba(X_test)[:, 1] 
    preds = model.predict(X_test) # Just 0 or 1
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print("\n--- MODEL RESULTS ---")
    print(f"Accuracy: {acc:.4f} (Baseline is usually ~0.60)")
    print(f"Log Loss: {ll:.4f} (Lower is better)")
    
    # 6. Feature Importance (What matters most?)
    # This answers the interview question: "What drives your model?"
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    print("\nTop 5 Important Features:")
    print(feature_imp.head())
    
    # 7. Save
    # Create models folder first if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH} using joblib")

if __name__ == "__main__":
    train()