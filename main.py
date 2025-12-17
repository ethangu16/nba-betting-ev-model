import subprocess
import sys
import time

def run_step(script_path, step_name):
    """
    Runs a python script and handles errors.
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING STEP: {step_name}")
    print(f"   Running: {script_path}")
    print(f"{'='*60}\n")

    start_time = time.time()
    
    try:
        # run using the same python interpreter
        result = subprocess.run([sys.executable, script_path], check=True)
        
        duration = time.time() - start_time
        print(f"\n✅ FINISHED STEP: {step_name} (took {duration:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in step: {step_name}")
        print(f"   Exit Code: {e.returncode}")
        print("   Pipeline stopped.")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: Script not found: {script_path}")
        print("   Check your file paths.")
        return False

def main():
    print("🏀 NBA PREDICTION PIPELINE INITIATED")
    print("-------------------------------------")

    # 1. Update Raw Data (Assuming you have a script for this)
    # If you don't have this yet, comment it out.
    # We assume 'src/data_collection/nba_games.py' fetches the latest box scores.
    if not run_step("src/data_collection/nba_games.py", "1. Update Game and Player Data"):
        return

    # 2. Create Player Ratings (Update Talent Pool)
    if not run_step("src/features/calculate_player_stats.py", "2. Player Talent Processing"):
        return

    # 3. Engineer Features (Calculate Elo & Rolling Stats)
    if not run_step("src/features/engineer.py", "3. Feature Engineering"):
        return
        
    # 4. Retrain Model (Learn from the newest games)
    if not run_step("src/models/train_model.py", "4. Model Retraining"):
        return

    # 5. Scrape Today's Injuries (Get the "Out" list)
    # Note: Assuming your scraper is named correctly. 
    # If it is inside predict_today, this step might be redundant, 
    # but strictly speaking, data collection should happen before prediction.
    if not run_step("src/data_collection/scrape_injuries.py", "5. Injury Scraping"):
        return

    # 6. Predict Today (Simulate and Bet)
    if not run_step("src/data_collection/live_odds.py", "6. Scrape live odds"):
        return

    # 7. Predict Today (Simulate and Bet)
    if not run_step("src/models/predict_today.py", "7. Generate Predictions"):
        return

    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE! CHECK RESULTS ABOVE.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()