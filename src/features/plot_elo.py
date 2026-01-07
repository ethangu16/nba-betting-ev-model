import pandas as pd
import plotly.graph_objects as go
import os

# --- CONFIG ---
DATA_PATH = 'data/processed/nba_model.csv'
OUTPUT_HTML = 'results/elo_trajectory_interactive.html'
SEASON_START = '2025-10-01'

# NBA Colors (Optional: Makes the chart look much professional)
TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#000000', 'CHA': '#1D1160', 
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#FEC524', 
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62', 
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E', 
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6', 
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160', 
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141', 
    'UTA': '#002B5C', 'WAS': '#002B5C'
}

def plot_elo_interactive():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Filter Season
    season_df = df[df['GAME_DATE'] >= pd.Timestamp(SEASON_START)].copy()
    if season_df.empty:
        print("⚠️ No data found for this season.")
        return

    print(f"📈 Generating Interactive Elo Chart for {len(season_df)} games...")

    # 3. Initialize Plotly Figure
    fig = go.Figure()

    # 4. Loop Through Every Team
    teams = sorted(season_df['TEAM_ABBREVIATION'].unique())
    
    for team in teams:
        team_data = season_df[season_df['TEAM_ABBREVIATION'] == team].sort_values('GAME_DATE')
        
        # Get color (fallback to grey if missing)
        color = TEAM_COLORS.get(team, '#999999')
        
        # Add the Line Trace
        fig.add_trace(go.Scatter(
            x=team_data['GAME_DATE'],
            y=team_data['ELO_TEAM'],
            mode='lines',
            name=team,
            line=dict(color=color, width=2),
            opacity=0.8,
            hovertemplate=f"<b>{team}</b><br>Date: %{{x}}<br>Elo: %{{y:.0f}}<extra></extra>"
        ))

        # Add a text label at the very end of the line
        last_game = team_data.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_game['GAME_DATE']],
            y=[last_game['ELO_TEAM']],
            mode='text',
            text=[f"<b>{team}</b>"],
            textposition="middle right",
            showlegend=False,
            hoverinfo='skip',
            textfont=dict(color=color, size=10)
        ))

    # 5. Add "League Average" Line
    fig.add_shape(
        type="line",
        x0=season_df['GAME_DATE'].min(),
        y0=1500,
        x1=season_df['GAME_DATE'].max(),
        y1=1500,
        line=dict(color="black", width=2, dash="dash"),
    )

    # 6. Formatting Layout
    fig.update_layout(
        title=dict(text="<b>NBA Elo Trajectory (2025-26)</b>", font=dict(size=24)),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Elo Rating", showgrid=True),
        template="plotly_white",
        height=900,  # Tall chart to fit all labels
        width=1400,
        showlegend=True,
        margin=dict(r=50) # Right margin for labels
    )

    # 7. Save
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    fig.write_html(OUTPUT_HTML)
    print(f"✅ Interactive Chart saved to: {OUTPUT_HTML}")
    print("👉 Open this file in your browser to interact and zoom!")

if __name__ == "__main__":
    plot_elo_interactive()