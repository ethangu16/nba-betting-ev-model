# 📘 NBA Algorithmic Betting System: Technical Documentation

## 1. System Architecture Overview

### Objective
This project is an end-to-end machine learning pipeline designed to identify "Positive Expected Value" (+EV) betting opportunities in NBA Moneyline markets. It utilizes historical game data, advanced feature engineering, and a hybrid ensemble model (XGBoost + Monte Carlo) to predict win probabilities and manage risk via the Fractional Kelly Criterion.

### The Pipeline Workflow
The system follows a standard Data Science lifecycle:
1.  **Data Ingestion:** Scrapes raw game logs, player stats, odds, and injury reports.
2.  **Feature Engineering:** Transforms raw data into predictive signals (Rolling Efficiency, Fatigue, ELO).
3.  **Modeling:** Trains an XGBoost Classifier optimized via Grid Search.
4.  **Inference:** Aggregates real-time data to generate daily betting recommendations.

---

## 2. File-by-File Technical Breakdown

### 📂 Module: Data Collection (`src/data_collection/`)

#### `nba_games.py` (Historical Data Scraper)
* **Function:** Connects to the NBA Stats API to fetch historical box scores from 2021 to the present.
* **Key Data Points:** Points, Field Goal %, Turnovers, Rebounds, Possessions.
* **Purpose:** Provides the "Ground Truth" data required to train the model on what "winning basketball" looks like.

#### `live_odds.py` (Market Data Scraper)
* **Function:** Scrapes real-time Moneyline odds from major sportsbooks (e.g., FanDuel, DraftKings) for upcoming games.
* **Purpose:** Establishes the "Market Price" (Implied Probability). The model compares its calculated win probability against this price to find "Edge."

#### `merge_odds.py` (Data Unification)
* **Function:** Performs a relational join between the *Game Stats* dataset and the *Historical Odds* dataset.
* **Logic:** Matches records based on `Team_Name` and `Game_Date`.
* **Importance:** Essential for backtesting. A strategy cannot be evaluated without knowing the specific odds available on the day a game was played.

#### `scrape_injuries.py` (Context Provider)
* **Function:** Parses live injury reports (e.g., ESPN) to identify key players listed as "Out" or "Questionable."
* **Output:** A dictionary of missing players per team.
* **Usage:** Used during inference to penalize a team's strength rating if key contributors are unavailable.

---

### 📂 Module: Feature Engineering (`src/features/`)

#### `engineer.py` (Main Transformation Pipeline)
This is the core processing engine that converts raw descriptive data into predictive signals.
* **Advanced Metrics:** Normalizes stats by "Pace" (Possessions per game) to create efficiency ratings (`OFF_RTG`, `DEF_RTG`) rather than raw point totals.
* **Time-Series Lagging:**
    * **Mechanism:** Uses `shift(1)` to strictly separate past performance from current results.
    * **Goal:** Prevents "Data Leakage" (using the final score of a game to predict that same game).
* **Rolling Windows:** Calculates 10-game moving averages for the "Four Factors" (EFG%, TOV%, ORB%, FTR).
* **Fatigue Flags:**
    * `IS_B2B`: Binary flag for Back-to-Back games (0 days rest).
    * `IS_3IN4`: Binary flag for teams playing their 3rd game in 4 nights (high fatigue).

#### `calculate_player_stats.py` (Player Valuation Engine)
* **Function:** Reduces high-dimensional player box scores into a single scalar "Talent Score."
* **Metric:** Uses John Hollinger’s Game Score formula:
    $$Game Score = PTS + 0.4(FGM) - 0.7(FGA) - 0.4(FTA - FTM) + 0.7(ORB) + 0.3(DRB) + STL + 0.7(AST) + 0.7(BLK) - 0.4(PF) - TOV$$
* **Smoothing:** Applies a 10-game rolling average to determine a player's "Current Form."
* **Usage:** This data feeds the Monte Carlo simulation to model specific roster matchups.

#### `optimise_elo.py` (Hyperparameter Tuning)
* **Function:** Determines the optimal parameters for the ELO rating system via Grid Search.
* **Parameters Tuned:**
    * `K-Factor`: Volatility/Sensitivity of the rating updates (Optimal: 25).
    * `Home Advantage`: Points added to the home team's rating (Optimal: 80).
* **Metric:** Minimizes Log Loss to ensure the ratings reflect true win probabilities.

---

### 📂 Module: Models (`src/models/`)

#### `train_model.py` (XGBoost Trainer)
* **Algorithm:** **XGBoost (Extreme Gradient Boosting)**.
* **Methodology:**
    * **Ensemble Learning:** Builds decision trees sequentially, where each tree attempts to correct the errors (residuals) of the previous one.
    * **Grid Search Cross-Validation:** systematically tests combinations of hyperparameters (Tree Depth, Learning Rate, Subsample) to maximize accuracy on unseen data.
* **Output:** A serialized `.joblib` model file capable of predicting win probabilities based on inputs like ELO difference, Fatigue, and Rolling Efficiency.

#### `predict_today.py` (Inference & Execution Engine)
This is the production script for generating daily bets.
* **Dynamic Adjustment:** Penalizes team ELO ratings in real-time based on the "Talent Share" of injured players.
* **Hybrid Ensemble:** Calculates final win probability as a weighted average:
    * **70% XGBoost:** Historical pattern recognition.
    * **30% Monte Carlo:** Stochastic simulation of player-level variance (1,000 runs).
* **Money Management:**
    * **Algorithm:** **Fractional Kelly Criterion (0.35x)**.
    * **Logic:** $Stake \% = \frac{(Odds \times WinProb - LossProb)}{Odds} \times 0.35$.
    * **Result:** Calculates the mathematically optimal bet size to maximize geometric growth while minimizing ruin.

---

### 📂 Module: Evaluation (`src/evaluation/`)

#### `backtest.py` (Strategy Simulator)
* **Function:** Replays historical betting opportunities using "Out-of-Sample" data (data the model has never seen).
* **Key Metrics:**
    * **ROI (Return on Investment):** Net Profit / Total Wagered.
    * **Win Rate:** Percentage of bets won.
    * **Closing Line Value (CLV):** (Optional) Comparing taken odds vs. final odds.
* **Visualization:** Generates an equity curve graph to visualize bankroll volatility and growth over time.

---

## 3. Key Technical Concepts

### XGBoost (eXtreme Gradient Boosting)
A decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. Unlike Random Forests (which rely on bagging), XGBoost builds trees sequentially. It is chosen for this project due to:
1.  **Non-Linearity:** Capability to model complex interactions (e.g., "Fatigue only matters against High Pace teams").
2.  **Regularization:** Built-in penalties (L1/L2) to prevent overfitting on noisy sports data.
3.  **Sparsity Awareness:** Automatic handling of missing values (e.g., player DNPs).

### The Kelly Criterion
A formula used to determine the optimal size of a series of bets. In this system, we use a **Fractional Kelly (0.35)** approach.
* **Full Kelly:** Maximizes wealth but has high volatility (risk of large drawdowns).
* **Fractional Kelly:** Reduces volatility significantly while retaining 80-90% of the growth potential. It acts as a safety mechanism against model error.

### ELO Rating System (Margin of Victory)
A comparative ranking system adapted from Chess.
* **Standard ELO:** Only cares about Win/Loss.
* **Project ELO:** Includes a **Margin of Victory (MOV)** multiplier.
    * *Logic:* Winning by 20 points indicates higher skill than winning by 1 point.
    * *Formula:* The rating update is scaled by $((MOV + 3)^{0.8}) / (7.5 + 0.006 \times EloDiff)$.