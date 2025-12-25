#!/usr/bin/env python3
"""
Automated Daily Pipeline for NBA Betting Model

This script runs the complete pipeline daily to:
1. Update data incrementally (only new games)
2. Process features
3. Retrain model (optional, can be weekly)
4. Generate predictions
5. Output betting recommendations

Usage:
    python run_daily_pipeline.py [--full] [--skip-training] [--skip-backtest]
    
Options:
    --full: Force full data refresh (ignore incremental updates)
    --skip-training: Skip model retraining (use existing model)
    --skip-backtest: Skip backtesting step
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
import time

def run_step(script_path, step_name, check=True):
    """
    Runs a python script and handles errors.
    
    Args:
        script_path: Path to script to run
        step_name: Human-readable name for logging
        check: If True, stop pipeline on error
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🚀 STEP: {step_name}")
    print(f"   Running: {script_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=check,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        duration = time.time() - start_time
        print(f"\n✅ COMPLETED: {step_name} (took {duration:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in step: {step_name}")
        print(f"   Exit Code: {e.returncode}")
        if check:
            print("   Pipeline stopped.")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: Script not found: {script_path}")
        print("   Check your file paths.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run NBA Betting Model Daily Pipeline')
    parser.add_argument('--full', action='store_true', 
                       help='Force full data refresh (ignore incremental updates)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model retraining (use existing model)')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip backtesting step')
    parser.add_argument('--data-only', action='store_true',
                       help='Only update data, skip everything else')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏀 NBA BETTING MODEL - DAILY PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'FULL REFRESH' if args.full else 'INCREMENTAL UPDATE'}")
    if args.skip_training:
        print("⚠️  Model training will be skipped")
    if args.skip_backtest:
        print("⚠️  Backtesting will be skipped")
    print("="*80)
    
    pipeline_start = time.time()
    steps_completed = 0
    steps_failed = 0
    
    # Step 1: Data Collection
    print("\n📊 PHASE 1: DATA COLLECTION")
    if args.full:
        # For full refresh, we could add a flag to nba_games.py
        # For now, just run normally (it will detect existing data)
        pass
    
    if not run_step("src/data_collection/nba_games.py", 
                   "1. Update Game and Player Data"):
        steps_failed += 1
        if not args.data_only:
            print("❌ Pipeline failed at data collection step")
            print("   Note: If fetching many historical seasons, this may take 30-60 minutes.")
            print("   Consider running with --data-only first, then running again later.")
            return
    steps_completed += 1
    
    if args.data_only:
        print("\n✅ Data-only mode: Stopping after data collection")
        return
    
    # Step 2: Player Stats Processing
    if not run_step("src/features/calculate_player_stats.py",
                   "2. Process Player Talent Scores"):
        steps_failed += 1
        print("⚠️  Continuing despite error...")
    steps_completed += 1
    
    # Step 3: Feature Engineering
    if not run_step("src/features/engineer.py",
                   "3. Engineer Features (ELO, Rolling Stats)"):
        steps_failed += 1
        print("❌ Feature engineering failed - cannot continue")
        return
    steps_completed += 1
    
    # Step 4: Model Training (optional)
    if not args.skip_training:
        print("\n🤖 PHASE 2: MODEL TRAINING")
        if not run_step("src/models/train_model.py",
                       "4. Retrain Model"):
            steps_failed += 1
            print("⚠️  Model training failed - will use existing model")
        else:
            steps_completed += 1
    else:
        print("\n⏭️  PHASE 2: MODEL TRAINING (SKIPPED)")
    
    # Step 5: Live Data Collection
    print("\n📡 PHASE 3: LIVE DATA COLLECTION")
    if not run_step("src/data_collection/scrape_injuries.py",
                   "5. Scrape Injury Reports"):
        steps_failed += 1
        print("⚠️  Continuing without injury data...")
    steps_completed += 1
    
    if not run_step("src/data_collection/live_odds.py",
                   "6. Scrape Live Odds"):
        steps_failed += 1
        print("❌ Cannot generate predictions without odds")
        return
    steps_completed += 1
    
    # Step 6: Generate Predictions
    print("\n🎯 PHASE 4: PREDICTION & BETTING")
    if not run_step("src/models/predict_today.py",
                   "7. Generate Betting Recommendations"):
        steps_failed += 1
        print("❌ Prediction failed")
        return
    steps_completed += 1
    
    # Step 7: Backtesting (optional)
    if not args.skip_backtest:
        print("\n📈 PHASE 5: EVALUATION")
        if not run_step("src/evaluation/backtest.py",
                       "8. Run Backtest"):
            steps_failed += 1
            print("⚠️  Backtest failed - predictions still generated")
        else:
            steps_completed += 1
    else:
        print("\n⏭️  PHASE 5: EVALUATION (SKIPPED)")
    
    # Summary
    pipeline_duration = time.time() - pipeline_start
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"Total Duration: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f} minutes)")
    print(f"Steps Completed: {steps_completed}")
    if steps_failed > 0:
        print(f"⚠️  Steps Failed: {steps_failed}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\n📋 Check results in:")
    print("   - results/todays_bets.csv (Betting recommendations)")
    if not args.skip_backtest:
        print("   - results/backtest_chart.png (Performance chart)")
    print("="*80)

if __name__ == "__main__":
    main()

