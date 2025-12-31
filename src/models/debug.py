import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sklearn

MODEL_PATH = 'models/nba_xgb_model.joblib'

# 🟢 HARDCODED FEATURE NAMES (Must match train_model.py order exactly)
FEATURE_NAMES = [
    'ELO_TEAM', 'ELO_OPP', 
    'IS_HOME', 
    'IS_B2B', 'IS_3IN4',              
    'ROLL_OFF_RTG', 'ROLL_DEF_RTG',   
    'ROLL_PACE', 
    'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
    'ROLL_ROSTER_TALENT_SCORE',
]

def inspect():
    print(f"🔍 INSPECTING MODEL: {MODEL_PATH}")
    print(f"   Scikit-Learn Version: {sklearn.__version__}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ File not found.")
        return

    # 1. Load the "Pickle"
    model_wrapper = joblib.load(MODEL_PATH)
    print(f"✅ Type: {type(model_wrapper)}")

    inner_model = None

    # 2. Extract the Inner XGBoost Brain (Robust Fix)
    try:
        if hasattr(model_wrapper, 'calibrated_classifiers_'):
            print(f"   📊 Calibration: Enabled (sigmoid method)")
            print(f"   📂 Internal Models: {len(model_wrapper.calibrated_classifiers_)} folds")
            
            # Grab the first fold's model
            first_fold = model_wrapper.calibrated_classifiers_[0]
            
            # TRY BOTH ATTRIBUTE NAMES (Compatibility Fix)
            if hasattr(first_fold, 'estimator'):
                inner_model = first_fold.estimator
            elif hasattr(first_fold, 'base_estimator'):
                inner_model = first_fold.base_estimator
            else:
                print("   ⚠️ Could not find 'estimator' inside calibrated classifier.")
                return
        else:
            inner_model = model_wrapper # It was just raw XGB
            
        print("\n⚙️  LEARNED PARAMETERS (Fold 1):")
        print(f"   n_estimators:     {inner_model.n_estimators}")
        print(f"   max_depth:        {inner_model.max_depth}")
        print(f"   learning_rate:    {inner_model.learning_rate}")
        print(f"   colsample_bytree: {inner_model.colsample_bytree}")
        
    except Exception as e:
        print(f"⚠️ Critical extraction error: {e}")
        return

    # 3. Print Feature Importances
    try:
        print("\n🧠 FEATURE IMPORTANCE (The Brain):")
        if hasattr(inner_model, 'feature_importances_'):
            imps = inner_model.feature_importances_
            
            # Combine names with scores
            data = list(zip(FEATURE_NAMES, imps))
            # Sort by importance (highest first)
            data.sort(key=lambda x: x[1], reverse=True)
            
            for name, val in data:
                bar_len = int(val * 40)
                bar = '█' * bar_len
                print(f"   {name:<25} | {bar} ({val:.1%})")
        else:
            print("   (Feature importances not available in this wrapper)")
    except Exception as e:
        print(f"   Could not read importances: {e}")

if __name__ == "__main__":
    inspect()