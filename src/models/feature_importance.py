import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Update this path if your model is in a different folder
MODEL_PATH = 'models/nba_xgb_model.joblib'

# Define the features EXACTLY as they were used in training
FEATURES = [
    'ELO_TEAM', 'ELO_OPP', 
    'IS_HOME', 
    'IS_B2B', 'IS_3IN4',
    'ROLL_OFF_RTG', 'ROLL_DEF_RTG',
    'ROLL_PACE', 
    'ROLL_EFG_PCT', 'ROLL_TOV_PCT', 'ROLL_ORB_PCT', 'ROLL_FTR',
    'ROLL_ROSTER_TALENT_SCORE'
]

def show_importance():
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train it first!")
        return

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get Feature Importances
    importance = model.feature_importances_
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 FEATURE IMPORTANCE RANKING:")
    print(df_imp)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # FIX: Assigned 'y' variable to 'hue' and set legend=False to silence warning
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=df_imp, 
        palette='viridis', 
        hue='Feature', 
        legend=False
    )
    
    plt.title('What Drives the Model?', fontsize=15)
    plt.xlabel('Impact on Prediction')
    plt.tight_layout()
    
    print("Opening plot... (Close the window to exit script)")
    plt.show()

if __name__ == "__main__":
    show_importance()