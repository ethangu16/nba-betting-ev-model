# Daily Pipeline Guide

## Overview

The automated daily pipeline (`run_daily_pipeline.py`) orchestrates the entire NBA betting model workflow. It handles incremental data updates, feature engineering, model training, and prediction generation.

## Quick Start

### Basic Daily Run
```bash
python run_daily_pipeline.py
```

This will:
- Update data incrementally (only new games)
- Process features
- Retrain model
- Scrape injuries and odds
- Generate predictions
- Run backtest

### Options

**Skip Model Training** (use existing model):
```bash
python run_daily_pipeline.py --skip-training
```

**Skip Backtesting** (faster, just get predictions):
```bash
python run_daily_pipeline.py --skip-backtest
```

**Data Only** (just update data, don't generate predictions):
```bash
python run_daily_pipeline.py --data-only
```

**Full Refresh** (re-download all data):
```bash
python run_daily_pipeline.py --full
```

## Pipeline Steps

### Phase 1: Data Collection
1. **Update Game Data** (`nba_games.py`)
   - Checks existing data file
   - Only fetches missing seasons or current/future seasons
   - Filters duplicates by GAME_ID
   - Appends new games to existing file
   - **Optimized**: Only reads necessary columns for efficiency

2. **Process Player Stats** (`calculate_player_stats.py`)
   - Calculates Hollinger Game Scores
   - Creates rolling averages
   - Updates player talent pool

### Phase 2: Feature Engineering
3. **Engineer Features** (`engineer.py`)
   - Calculates ELO ratings
   - Computes advanced stats (Pace, OffRtg, DefRtg)
   - Creates rolling features
   - Aggregates roster strength

### Phase 3: Model Training (Optional)
4. **Retrain Model** (`train_model.py`)
   - Loads engineered features
   - Grid search for hyperparameters
   - Trains XGBoost classifier
   - Saves updated model

### Phase 4: Live Data Collection
5. **Scrape Injuries** (`scrape_injuries.py`)
   - Fetches current injury reports from ESPN
   - Filters for "Out" players only

6. **Scrape Odds** (`live_odds.py`)
   - Fetches live betting odds from The Odds API
   - Extracts Moneyline and Spread odds

### Phase 5: Prediction & Betting
7. **Generate Predictions** (`predict_today.py`)
   - Loads latest stats and model
   - Adjusts for injuries
   - Generates ensemble predictions (XGBoost + Monte Carlo)
   - Calculates Kelly bet sizes
   - Outputs betting card

### Phase 6: Evaluation (Optional)
8. **Run Backtest** (`backtest.py`)
   - Simulates historical betting
   - Calculates ROI, win rate
   - Generates equity curve

## Data Collection Optimization

### Incremental Updates

The `nba_games.py` script is optimized for incremental updates:

**First Run**:
- Fetches all seasons (2008-2025)
- Saves to `data/raw/nba_games_stats.csv`
- Takes ~30-60 minutes (depending on API speed)

**Subsequent Runs**:
- Checks existing file
- Only fetches:
  - Current season (2024-25) and future seasons
  - Any missing historical seasons
- Filters duplicates by GAME_ID
- Appends only new games
- Takes ~2-5 minutes (only new data)

### Efficiency Features

1. **Selective Column Reading**: Only reads `GAME_DATE`, `GAME_ID`, `SEASON_ID` to check what exists
2. **Duplicate Filtering**: Uses GAME_ID to avoid re-processing games
3. **Season Tracking**: Tracks which seasons are already in file
4. **Smart Fetching**: Only fetches missing or current/future seasons

### Extended Seasons (2008-2025)

The pipeline now includes 18 seasons (2008-09 through 2025-26) for:
- Better backtesting (more historical data)
- More training data for model
- Longer-term pattern recognition

## Backtest Improvements

### Matching predict_today.py Logic

The backtest now uses **exactly the same betting logic** as `predict_today.py`:

✅ **Same Parameters**:
- Kelly Fraction: 0.35 (was 0.25)
- Min Edge: 2% (was 1%)
- Max Edge: 15% (new)
- Win Prob Bounds: 30%-90% (new)

✅ **Same Calibration**:
- Probability calibration (if enabled)
- Edge dampening
- Max edge capping

✅ **Same Constraints**:
- Maximum bet cap (5% of bankroll)
- Win probability bounds
- Edge thresholds

**Result**: Backtest results now accurately reflect what the production system would do.

## Scheduling (Cron/Windows Task Scheduler)

### Linux/Mac (Cron)

Add to crontab (`crontab -e`):
```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/nba-betting-ev-model && /path/to/venv/bin/python run_daily_pipeline.py --skip-training >> logs/pipeline.log 2>&1

# Run weekly on Sunday at 10 AM (with training)
0 10 * * 0 cd /path/to/nba-betting-ev-model && /path/to/venv/bin/python run_daily_pipeline.py >> logs/pipeline.log 2>&1
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 9:00 AM
4. Action: Start a program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `run_daily_pipeline.py --skip-training`
   - Start in: `C:\path\to\nba-betting-ev-model`

## Monitoring

### Log Files

The pipeline outputs to console. To log to file:

```bash
python run_daily_pipeline.py >> logs/pipeline_$(date +%Y%m%d).log 2>&1
```

### Check Results

After pipeline runs, check:
- `results/todays_bets.csv` - Betting recommendations
- `results/backtest_chart.png` - Performance visualization
- Console output for errors

## Troubleshooting

### "No new games found"
- Normal if data is up to date
- Check if NBA season is active

### "API rate limit"
- Add longer delays between requests
- Check API key limits

### "Missing features"
- Run `engineer.py` manually
- Check if data file is corrupted

### "Model not found"
- Run `train_model.py` first
- Or use `--skip-training` to use existing model

## Performance

**First Run** (full data fetch):
- Time: 30-60 minutes
- Data: ~10,000+ games, ~100,000+ player-game rows

**Daily Run** (incremental):
- Time: 2-5 minutes
- Data: Only new games since last run

**With Training**:
- Add 5-15 minutes for model training

**With Backtest**:
- Add 1-2 minutes for backtesting

## Best Practices

1. **Run Daily**: Keep data fresh
2. **Weekly Training**: Retrain model weekly (or when significant data changes)
3. **Monitor Logs**: Check for errors regularly
4. **Backup Data**: Periodically backup `data/raw/` directory
5. **Version Control**: Don't commit large data files (use .gitignore)


