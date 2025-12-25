# Pipeline Fixes Applied

## Issues Found and Fixed

### 1. ✅ Date Sorting Error

**Error**: `TypeError: '<' not supported between instances of 'Timestamp' and 'str'`

**Cause**: When combining existing and new data, GAME_DATE column had mixed types (datetime vs string).

**Fix**: Added explicit datetime conversion before sorting:
```python
existing_full['GAME_DATE'] = pd.to_datetime(existing_full['GAME_DATE'])
processed_new['GAME_DATE'] = pd.to_datetime(processed_new['GAME_DATE'])
```

**Location**: `src/data_collection/nba_games.py` line 223-224

---

### 2. ⚠️ Long Runtime for Historical Data

**Issue**: Pipeline tries to fetch all missing historical seasons (2008-2025) at once, which takes 30-60+ minutes.

**Fix**: Added limit to fetch only 3 historical seasons at a time:
- First run: Fetches current season + 3 oldest missing seasons
- Subsequent runs: Continue fetching 3 more at a time
- Eventually all seasons will be collected

**Location**: `src/data_collection/nba_games.py` line 175-185

**Recommendation**: 
- For first-time setup, let it run overnight or use `--data-only` flag
- For daily runs, it will only fetch current/future seasons (fast)

---

## Current Status

✅ **Date sorting fixed** - No more type errors  
✅ **Incremental updates working** - Only fetches new data  
✅ **Efficient duplicate filtering** - Uses GAME_ID  
✅ **Historical data limit** - Prevents extremely long runs  

---

## Usage Recommendations

### First Time Setup (Full Historical Data)

```bash
# Option 1: Let it run (will take 30-60 minutes for all seasons)
python run_daily_pipeline.py --data-only

# Option 2: Run in background
nohup python run_daily_pipeline.py --data-only > logs/setup.log 2>&1 &

# Option 3: Run multiple times (fetches 3 seasons at a time)
python run_daily_pipeline.py --data-only  # Run 6 times to get all 18 seasons
```

### Daily Runs (Fast)

```bash
# After initial setup, daily runs are fast (2-5 minutes)
python run_daily_pipeline.py --skip-training --skip-backtest
```

### Weekly Runs (With Training)

```bash
# Full pipeline with model retraining
python run_daily_pipeline.py
```

---

## Next Steps

1. **If pipeline is currently running**: Let it complete (it's fetching historical data)
2. **If you want to stop and resume later**: The data is saved incrementally, so you can stop and continue later
3. **For faster daily runs**: After initial historical data is collected, daily runs will be much faster

---

## Verification

To verify fixes are working:

```bash
# Test date conversion
python -c "import pandas as pd; df = pd.read_csv('data/raw/nba_games_stats.csv', nrows=100); df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']); print('✅ Date conversion works')"

# Check pipeline script
python -c "import run_daily_pipeline; print('✅ Pipeline script loads correctly')"
```

---

## Notes

- The first run with all historical seasons will take a long time (30-60 minutes)
- This is normal and only happens once (or when adding new historical seasons)
- Daily incremental updates are fast (2-5 minutes)
- Data is saved incrementally, so you can stop and resume


