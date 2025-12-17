import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

def scrape_espn_nba_injuries():
    """
    Scrape current NBA injuries from ESPN

    Returns:
        DataFrame with injury information
    """
    
    url = "https://www.espn.com/nba/injuries"
    
    print(f"Fetching injuries from {url}...")
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    injuries = []
    
    # ESPN structure: Each team has a table with injuries
    team_sections = soup.find_all('div', class_='ResponsiveTable')
    
    print(f"Found {len(team_sections)} team sections")
    
    for section in team_sections:
        # Get team name
        team_header = section.find('div', class_='Table__Title')
        if team_header:
            team_name = team_header.text.strip()
        else:
            team_name = "Unknown"
        
        # Find the table body
        tbody = section.find('tbody', class_='Table__TBODY')
        
        if not tbody:
            continue
        
        # Get all rows (each row is a player)
        rows = tbody.find_all('tr')
        
        for row in rows:
            try:
                # Extract cells
                cells = row.find_all('td')
                
                if len(cells) < 4:
                    continue
                
                # Parse player info
                player_cell = cells[0]
                player_link = player_cell.find('a')
                
                if player_link:
                    player_name = player_link.text.strip()
                    player_position = player_cell.find('div', class_='inline').text.strip() if player_cell.find('div', class_='inline') else ""
                else:
                    player_name = player_cell.text.strip()
                    player_position = ""
                
                # Injury details
                injury_date = cells[1].text.strip() if len(cells) > 1 else ""
                injury_status = cells[2].text.strip() if len(cells) > 2 else ""
                injury_description = cells[3].text.strip() if len(cells) > 3 else ""
                
                injuries.append({
                    'team': team_name,
                    'player_name': player_name,
                    'position': player_position,
                    'date': injury_date,
                    'status': injury_status,
                    'description': injury_description,
                })
                
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
    
    if not injuries:
        print("No injuries found. Page structure may have changed.")
        return None
    
    df = pd.DataFrame(injuries)
    
    print(f"Scraped {len(df)} injuries from {df['team'].nunique()} teams")
    return df



if __name__ == "__main__":
    # Scrape current injuries
    injuries_df = scrape_espn_nba_injuries()
    
    if injuries_df is not None:
        # Save
        output_path = 'data/raw/espn_injuries_current.csv'
        injuries_df.to_csv(output_path, index=False)
        
        print(f"\nSaved to {output_path}")
        