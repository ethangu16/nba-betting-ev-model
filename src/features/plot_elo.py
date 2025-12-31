import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# --- CONFIG ---
DATA_PATH = 'data/processed/nba_model.csv'
OUTPUT_IMAGE = 'results/elo_trajectory_2025_26.png'
SEASON_START = '2025-10-01'  # Start of 2025-26 Season

def plot_elo():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run engineer.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Filter for 2025-2026 Season
    season_df = df[df['GAME_DATE'] >= pd.Timestamp(SEASON_START)].copy()
    
    if season_df.empty:
        print(f"⚠️ No games found after {SEASON_START}. Check your dataset dates.")
        return

    print(f"📈 Plotting Elo for {len(season_df)} games...")

    # 3. Setup Plot
    plt.figure(figsize=(16, 10))
    sns.set_style("darkgrid")
    
    # Get top 5 and bottom 5 teams for highlighting labels
    latest_date = season_df['GAME_DATE'].max()
    current_elos = season_df[season_df['GAME_DATE'] == latest_date].sort_values('ELO_TEAM', ascending=False)
    
    top_teams = current_elos.head(5)['TEAM_ABBREVIATION'].tolist()
    bottom_teams = current_elos.tail(3)['TEAM_ABBREVIATION'].tolist()
    highlight_teams = set(top_teams + bottom_teams)

    # 4. Plot Lines
    # We loop manually to control z-order (highlighted teams on top)
    teams = season_df['TEAM_ABBREVIATION'].unique()
    
    # Define a custom color palette or use a standard one
    palette = sns.color_palette("tab20", n_colors=len(teams))
    team_colors = dict(zip(teams, palette))

    for team in teams:
        team_data = season_df[season_df['TEAM_ABBREVIATION'] == team].sort_values('GAME_DATE')
        
        # Determine style based on rank
        if team in top_teams:
            alpha = 1.0
            linewidth = 3.5
            zorder = 10
            label = team
        elif team in bottom_teams:
            alpha = 0.8
            linewidth = 2.5
            zorder = 9
            label = team
        else:
            alpha = 0.15  # Fade out the middle of the pack
            linewidth = 1.5
            zorder = 1
            label = None # Don't clutter legend
            
        plt.plot(team_data['GAME_DATE'], team_data['ELO_TEAM'], 
                 label=label, color=team_colors[team], 
                 alpha=alpha, linewidth=linewidth, zorder=zorder)

        # Add label at the end of the line
        last_game = team_data.iloc[-1]
        if team in highlight_teams:
            plt.text(last_game['GAME_DATE'], last_game['ELO_TEAM'], f" {team}", 
                     fontsize=10, fontweight='bold', color=team_colors[team], va='center')

    # 5. Formatting
    plt.title('NBA Team Elo Ratings (2025-2026 Season)', fontsize=20, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Elo Rating', fontsize=14)
    
    # Add "Average" line
    plt.axhline(y=1500, color='black', linestyle='--', alpha=0.5, label='League Average (1500)')
    
    # Format X-Axis Dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Key Teams')
    plt.tight_layout()
    
    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Elo Chart saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    plot_elo()