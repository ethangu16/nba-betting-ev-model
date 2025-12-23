"""
Centralized team name mapping for consistent team name conversion across the codebase.
"""

# Standard NBA team abbreviations
TEAM_ABBREVIATIONS = {
    'ATL': 'ATL', 'BOS': 'BOS', 'BKN': 'BKN', 'CHA': 'CHA', 'CHI': 'CHI',
    'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GSW': 'GSW',
    'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC', 'LAL': 'LAL', 'MEM': 'MEM',
    'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN', 'NOP': 'NOP', 'NYK': 'NYK',
    'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI', 'PHX': 'PHX', 'POR': 'POR',
    'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR', 'UTA': 'UTA', 'WAS': 'WAS'
}

# Full team names to abbreviations
FULL_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "L.A. Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# Lowercase/short names to abbreviations (for odds data)
LOWERCASE_TO_ABBR = {
    'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
    'gsw': 'GSW', 'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL',
    'mem': 'MEM', 'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP',
    'nop': 'NOP', 'ny': 'NYK', 'nyk': 'NYK', 'okc': 'OKC', 'orl': 'ORL',
    'phi': 'PHI', 'phx': 'PHX', 'pho': 'PHX', 'por': 'POR', 'sa': 'SAS',
    'sas': 'SAS', 'sac': 'SAC', 'tor': 'TOR', 'utah': 'UTA', 'uta': 'UTA',
    'was': 'WAS', 'nj': 'BKN', 'sea': 'OKC'  # Historic franchises
}

def normalize_team_name(team_name):
    """
    Normalize a team name to standard abbreviation.
    
    Args:
        team_name: Team name in any format (full name, abbreviation, lowercase, etc.)
    
    Returns:
        Standard team abbreviation (e.g., 'GSW') or original if not found
    """
    if not team_name:
        return team_name
    
    # If already a standard abbreviation, return as-is
    team_upper = team_name.upper()
    if team_upper in TEAM_ABBREVIATIONS:
        return team_upper
    
    # Try full name mapping
    if team_name in FULL_NAME_TO_ABBR:
        return FULL_NAME_TO_ABBR[team_name]
    
    # Try lowercase mapping
    team_lower = team_name.lower()
    if team_lower in LOWERCASE_TO_ABBR:
        return LOWERCASE_TO_ABBR[team_lower]
    
    # Return original if no mapping found
    return team_name

