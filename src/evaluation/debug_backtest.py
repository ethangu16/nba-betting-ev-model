import pandas as pd
import sys

def audit():
    try:
        print("Loading results/backtest_log.csv...")
        df = pd.read_csv('results/backtest_log.csv')
    except FileNotFoundError:
        print("❌ File not found. Run backtest_roi.py first.")
        return

    # Clean the currency string to float for sorting
    # We use regex to remove '$' and ',' if they exist
    df['Profit_Num'] = df['Profit'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

    print("\n🔎 TOP 5 BIGGEST WINS:")
    # Removed 'Opponent' from this list to fix the error
    cols_to_show = ['Date', 'Team', 'Result', 'Odds', 'My_Prob', 'Vegas_Prob', 'Profit']
    
    # Check if columns exist before printing to avoid future KeyErrors
    available_cols = [c for c in cols_to_show if c in df.columns]
    
    print(df.sort_values('Profit_Num', ascending=False).head(5)[available_cols])

    print("\n🔎 TOP 5 BIGGEST LOSSES:")
    print(df.sort_values('Profit_Num', ascending=True).head(5)[available_cols])

if __name__ == "__main__":
    audit()