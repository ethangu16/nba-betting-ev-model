import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
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

    # 1. Define Features (Must include the NEW Roster Score)
    features = [
        'ELO_TEAM', 'ELO_OPP', 
        'REST_DAYS', 'TRAVEL_MILES', 'IS_HOME',
        'ROLL_PTS', 'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR', 'ROLL_PLUS_MINUS',
        'ROLL_ROSTER_TALENT_SCORE' # <--- THE NEW FEATURE
    ]
    
    target = 'TARGET_WIN' # 1 if Team Won, 0 if Lost

    # 2. Drop rows where features are missing
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(df)} games with {len(features)} features...")

    # 3. Split Data (80% Train, 20% Test)
    # We allow random shuffle here since we have rolling features handling the time component
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize & Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Trained! Accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred))

    # 6. Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    train()