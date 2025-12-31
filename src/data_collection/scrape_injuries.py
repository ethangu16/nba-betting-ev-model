import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

def scrape_espn_nba_injuries():
    url = "https://www.espn.com/nba/injuries"
    print(f"Fetching injuries from {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    injuries = []
    
    # Locate all team sections
    team_sections = soup.find_all('div', class_='ResponsiveTable')
    
    for section in team_sections:
        team_header = section.find('div', class_='Table__Title')
        if team_header:
            team_name = team_header.text.strip()
        else:
            continue
        
        tbody = section.find('tbody', class_='Table__TBODY')
        if not tbody: continue
        
        rows = tbody.find_all('tr')
        
        for row in rows:
            try:
                cells = row.find_all('td')
                
                if len(cells) < 5:
                    continue
                
                # 1. Player Name (Column 0)
                player_cell = cells[0]
                player_link = player_cell.find('a')
                player_name = player_link.text.strip() if player_link else player_cell.text.strip()
                
                # 2. Position (Column 1)
                position = cells[1].text.strip()
                
                # 3. Date (Column 2)
                injury_date = cells[2].text.strip()
                
                # 4. Status (Column 3)
                injury_status = cells[3].text.strip()
                
                injuries.append({
                    'team': team_name,
                    'player_name': player_name,
                    'position': position,
                    'date': injury_date,
                    'status': injury_status,
                })
                
            except Exception as e:
                continue
    
    if not injuries:
        print("No injuries found.")
        return None
    
    df = pd.DataFrame(injuries)
    print(f"Scraped {len(df)} injuries from {df['team'].nunique()} teams")
    return df

if __name__ == "__main__":
    injuries_df = scrape_espn_nba_injuries()
    if injuries_df is not None:
        output_path = 'data/raw/espn_injuries_current.csv'
        injuries_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved to {output_path}")