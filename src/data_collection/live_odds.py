import requests
import pandas as pd
from datetime import datetime

API_KEY = '464fb90e81e4cf13bd320b90f11d053b' 

def get_live_odds():
    print("Fetching live NBA odds (Moneyline + Spreads)...")
    
    # 1. API Endpoint
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'apiKey': API_KEY,
        'regions': 'us', 
        'markets': 'h2h,spreads', 
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None
        
    data = response.json()
    print(f"Found {len(data)} upcoming games.")
    
    # 2. Parse Data
    rows = []
    for game in data:
        game_date = game['commence_time']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # We will initialize vars to None so we can handle missing lines
        home_ml = None
        away_ml = None
        home_spread = None
        home_spread_odds = None
        away_spread = None
        away_spread_odds = None
        bookmaker_name = None

        # Loop through bookmakers to find the first valid one
        target_books = ['draftkings', 'fanduel', 'betmgm']
        
        selected_book = None
        for book in game['bookmakers']:
            if book['key'] in target_books:
                selected_book = book
                break
        
        # If no target book found, just take the first one available
        if not selected_book and game['bookmakers']:
            selected_book = game['bookmakers'][0]
            
        if selected_book:
            bookmaker_name = selected_book['title']
            
            for market in selected_book['markets']:
                # --- PARSE MONEYLINE (h2h) ---
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            home_ml = outcome['price']
                        elif outcome['name'] == away_team:
                            away_ml = outcome['price']
                            
                # --- PARSE SPREADS ---
                elif market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            home_spread = outcome['point']      # e.g. -5.5
                            home_spread_odds = outcome['price'] # e.g. -110
                        elif outcome['name'] == away_team:
                            away_spread = outcome['point']
                            away_spread_odds = outcome['price']

            # Only add the row if we found at least some odds
            rows.append({
                'GAME_DATE': game_date,
                'HOME_TEAM': home_team,
                'AWAY_TEAM': away_team,
                'HOME_ML': home_ml,
                'AWAY_ML': away_ml,
                'HOME_SPREAD': home_spread,
                'HOME_SPREAD_ODDS': home_spread_odds,
                'AWAY_SPREAD': away_spread,
                'AWAY_SPREAD_ODDS': away_spread_odds,
                'BOOKMAKER': bookmaker_name
            })
    
    # 3. Save to CSV
    if rows:
        df = pd.DataFrame(rows)
        filename = f'data/odds/live_odds.csv'
        df.to_csv(filename, index=False)
        print(f"✅ Success! Saved {len(df)} games to {filename}")
        print(df[['HOME_TEAM', 'HOME_ML', 'HOME_SPREAD']].head())
    else:
        print("⚠️ No odds found. (Are there games scheduled for today?)")

if __name__ == "__main__":
    get_live_odds()