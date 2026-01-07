import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# --- CONFIGURATION ---
DATA_PATH = 'data/processed/nba_model_with_odds.csv'
MODEL_PATH_JOBLIB = 'models/nba_xgb_model.joblib'
LOG_PATH = 'results/backtest_log.csv'
CHART_PATH = 'results/backtest_chart.png'

# --- STRATEGY SETTINGS (CONSERVATIVE MODE) ---
INITIAL_BANKROLL = 1000
KELLY_FRACTION = 0.15  # 🟢 Reduced from 0.25 to 0.15 (More stable)
MAX_BET_PCT = 0.03     # Global hard cap (3%)

# 🛑 SAFETY VALVES
MIN_EDGE = 0.015       
MAX_EDGE = 0.15        
MAX_LONGSHOT_ODDS = 2.85 # (+185) - Definition of "Low Underdog"

# --- FEATURES ---
FEATURES = [
    'ELO_TEAM', 'ELO_OPP', 'IS_HOME', 'IS_B2B', 'IS_3IN4',
    'ROSTER_TALENT_SCORE',
    'SMA_20_OFF_RTG', 'SMA_20_DEF_RTG', 'SMA_20_PACE', 'SMA_20_EFG_PCT',
    'SMA_20_TOV_PCT', 'SMA_20_ORB_PCT', 'SMA_20_FTR', 'SMA_20_ROSTER_TALENT_SCORE',
    'EWMA_10_OFF_RTG', 'EWMA_10_DEF_RTG', 'EWMA_10_PACE', 'EWMA_10_EFG_PCT',
    'EWMA_10_TOV_PCT', 'EWMA_10_ORB_PCT', 'EWMA_10_FTR', 'EWMA_10_ROSTER_TALENT_SCORE',
    'EWMA_5_OFF_RTG', 'EWMA_5_DEF_RTG', 'EWMA_5_PACE', 'EWMA_5_EFG_PCT',
    'EWMA_5_TOV_PCT', 'EWMA_5_ORB_PCT', 'EWMA_5_FTR', 'EWMA_5_ROSTER_TALENT_SCORE',
]

def load_model():
    if os.path.exists(MODEL_PATH_JOBLIB):
        return joblib.load(MODEL_PATH_JOBLIB)
    raise FileNotFoundError("❌ No trained model found.")

def get_moneyline_probs(ml):
    if pd.isna(ml) or ml == 0: return None, None
    if ml < 0:
        decimal = 1 + (100 / abs(ml))
        implied = abs(ml) / (abs(ml) + 100)
    else:
        decimal = 1 + (ml / 100)
        implied = 100 / (ml + 100)
    return decimal, implied

def run_backtest():
    print("--- STARTING NBA BACKTEST (CONSERVATIVE) ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    if 'HOME_ML' in df.columns:
        df['MY_ML'] = np.where(df['IS_HOME'] == 1, df['HOME_ML'], df['AWAY_ML'])
    
    df = df.dropna(subset=['MY_ML', 'WL'] + FEATURES).copy()
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Test on last 20%
    split_idx = int(len(df) * 0.80)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Backtesting on {len(test_df)} games")
    
    model = load_model()
    probs = model.predict_proba(test_df[FEATURES])[:, 1]
    test_df['MODEL_PROB'] = probs

    # --- SIMULATION ---
    bankroll = INITIAL_BANKROLL
    history = [INITIAL_BANKROLL]
    bets_log = []
    
    wins, losses, skipped = 0, 0, 0
    skipped_high_edge = 0
    total_wagered = 0
    
    for idx, row in test_df.iterrows():
        decimal_odds, implied_prob = get_moneyline_probs(row['MY_ML'])
        if not decimal_odds: continue
            
        my_prob = row['MODEL_PROB']
        edge = my_prob - implied_prob
        
        bet_amount = 0
        
        # --- THE DECISION LOGIC ---
        if edge > MIN_EDGE:
            if edge > MAX_EDGE:
                skipped_high_edge += 1
                continue 
            
            # Standard Kelly Calculation
            b = decimal_odds - 1
            p = my_prob
            q = 1 - p
            kelly = (b * p - q) / b
            
            if kelly > 0:
                # 1. Apply Conservative Fraction
                raw_bet_pct = kelly * KELLY_FRACTION
                
                # 🟢 2. DYNAMIC LONGSHOT CAP
                # If the odds are high (low probability underdog), we force a smaller cap.
                # Logic: Even if value is high, variance is dangerous.
                current_cap = MAX_BET_PCT
                if decimal_odds >= MAX_LONGSHOT_ODDS:
                    current_cap = 0.01  # Hard cap of 1% for longshots (+185 or longer)
                
                final_pct = min(raw_bet_pct, current_cap)
                bet_amount = bankroll * final_pct
        
        if bet_amount > 0:
            total_wagered += bet_amount
            
            profit = 0
            if row['WL'] == 'W':
                profit = bet_amount * (decimal_odds - 1)
                bankroll += profit
                wins += 1
                outcome = "WIN"
            else:
                profit = -bet_amount
                bankroll -= bet_amount
                losses += 1
                outcome = "LOSS"
                
            bets_log.append({
                'Date': row['GAME_DATE'],
                'Team': row['TEAM_ABBREVIATION'],
                'Result': outcome,
                'Odds': row['MY_ML'],
                'My_Prob': f"{my_prob:.1%}",
                'Vegas_Prob': f"{implied_prob:.1%}",
                'Edge': f"{edge:.1%}",
                'Bet_Pct': f"{bet_amount/bankroll:.1%}", # Log how much of bankroll was risked
                'Bet_Amt': f"${bet_amount:.2f}",
                'Profit': profit,
                'Bankroll': bankroll
            })
            
        history.append(bankroll)

    # --- REPORTING ---
    total_bets = wins + losses
    net_profit = bankroll - INITIAL_BANKROLL
    
    bankroll_growth = (net_profit / INITIAL_BANKROLL) * 100
    
    if total_wagered > 0:
        roi_yield = (net_profit / total_wagered) * 100
    else:
        roi_yield = 0
    
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    print("\n" + "="*50)
    print(f"💰 CONSERVATIVE RESULT (Low Risk Mode)")
    print(f"   Starting Bankroll:   ${INITIAL_BANKROLL}")
    print(f"   Ending Bankroll:     ${bankroll:,.2f}")
    print(f"   Total Net Profit:    ${net_profit:,.2f}")
    print("-" * 50)
    print(f"   📈 TOTAL RETURN:     {bankroll_growth:+.2f}%")
    print(f"   💵 YIELD (PER BET):  {roi_yield:+.2f}%")
    print("-" * 50)
    print(f"   Total Bets:          {total_bets}")
    print(f"   Win Rate:            {win_rate:.1f}%")
    print(f"   Skipped (Suspicious):{skipped_high_edge}")
    print("="*50)
    
    pd.DataFrame(bets_log).to_csv(LOG_PATH, index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(f"Equity Curve (Yield: {roi_yield:.1f}%)")
    plt.savefig(CHART_PATH)
    print(f"📈 Chart saved to {CHART_PATH}")

if __name__ == "__main__":
    run_backtest()