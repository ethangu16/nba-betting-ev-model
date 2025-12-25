# NBA Betting EV Model - Complete Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Pipeline Architecture](#pipeline-architecture)
4. [File-by-File Technical Breakdown](#file-by-file-technical-breakdown)
5. [Advanced Techniques](#advanced-techniques)
6. [Data Flow](#data-flow)
7. [Model Training & Evaluation](#model-training--evaluation)

---

## System Overview

### Purpose
This system is an end-to-end machine learning pipeline designed to identify **Positive Expected Value (+EV)** betting opportunities in NBA Moneyline markets. It combines historical pattern recognition, real-time injury data, and advanced statistical methods to generate mathematically optimal betting recommendations.

### Core Philosophy
- **Expected Value First**: Only bet when model probability > market implied probability
- **Risk Management**: Fractional Kelly Criterion (35%) with maximum bet caps
- **Data Integrity**: Strict time-series lagging to prevent data leakage
- **Market Efficiency**: Bayesian methods to quantify uncertainty

---

## Mathematical Foundations

### 1. Expected Value (EV)

The fundamental metric for any bet:

$$EV = (P_{win} \times Payout) - (P_{loss} \times Stake)$$

Where:
- $P_{win}$ = Model's predicted win probability
- $Payout$ = Net profit if bet wins (decimal_odds - 1)
- $P_{loss} = 1 - P_{win}$
- $Stake$ = Amount wagered

**Decision Rule**: Only bet when $EV > 0$ (positive expected value)

---

### 2. Kelly Criterion

The optimal bet sizing formula to maximize long-term geometric growth:

$$f^* = \frac{bp - q}{b}$$

Where:
- $f^*$ = Optimal fraction of bankroll to bet
- $b$ = Net odds received (decimal_odds - 1)
- $p$ = Probability of winning
- $q = 1 - p$ = Probability of losing

**Fractional Kelly**: We use $f = 0.35 \times f^*$ to reduce volatility while retaining 80-90% of growth potential.

**Constraints Applied**:
- Maximum bet: 5% of bankroll
- Minimum edge: 2%
- Maximum edge: 15% (data error protection)
- Win probability bounds: 30% - 90%

---

### 3. ELO Rating System (Margin of Victory)

Adapted from chess, with NBA-specific modifications:

**Standard ELO Update**:
$$R_{new} = R_{old} + K \times S \times (Actual - Expected)$$

Where:
- $K$ = K-factor (volatility parameter, optimal: 25)
- $S$ = Margin of Victory multiplier
- $Expected = \frac{1}{1 + 10^{-(R_{team} - R_{opp} + H)/400}}$
- $H$ = Home advantage (60 points)

**Margin of Victory Multiplier**:
$$S = \frac{(MOV + 3)^{0.8}}{7.5 + 0.006 \times |EloDiff|}$$

This ensures:
- Blowout wins (20+ points) increase rating more than close wins
- Rating changes scale with game importance (larger EloDiff = smaller multiplier)

---

### 4. Advanced Statistics

#### Offensive/Defensive Rating (Per 100 Possessions)

**Possessions Estimation**:
$$Possessions = 0.96 \times (FGA + TOV + 0.44 \times FTA - OREB)$$

**Offensive Rating**:
$$OffRtg = 100 \times \frac{PTS}{Possessions}$$

**Defensive Rating**:
$$DefRtg = 100 \times \frac{PTS_{allowed}}{Possessions}$$

**Pace**:
$$Pace = 48 \times \frac{Possessions}{Minutes/5}$$

These normalize for pace, allowing fair comparison between fast and slow teams.

---

### 5. Four Factors of Basketball

Four key metrics that predict team success:

1. **Effective Field Goal %**: $eFG\% = \frac{FGM + 0.5 \times FG3M}{FGA}$
2. **Turnover %**: $TOV\% = \frac{TOV}{FGA + 0.44 \times FTA + TOV}$
3. **Offensive Rebound %**: $ORB\% = \frac{OREB}{OREB + OPP_{DREB}}$
4. **Free Throw Rate**: $FTR = \frac{FTA}{FGA}$

---

### 6. Rolling Averages (Time-Series Features)

To prevent data leakage, all features use **shifted rolling averages**:

$$Roll_{t} = \frac{1}{n} \sum_{i=t-n}^{t-1} Feature_i$$

Where:
- $n$ = Window size (typically 10 games)
- $t-1$ = Previous game (shift by 1 to avoid leakage)
- Minimum periods = 3 (for early season games)

**Example**: `ROLL_OFF_RTG` = Average offensive rating over last 10 games (excluding current game)

---

### 7. Player Valuation (Hollinger Game Score)

Single metric to quantify player performance:

$$GameScore = PTS + 0.4 \times FGM - 0.7 \times FGA - 0.4 \times (FTA - FTM)$$
$$+ 0.7 \times OREB + 0.3 \times DREB + STL + 0.7 \times AST + 0.7 \times BLK$$
$$- 0.4 \times PF - TOV$$

**Roster Strength**: Sum of top 9 players' projected Game Scores, adjusted for minutes and injuries.

---

### 8. Monte Carlo Simulation

Stochastic simulation to estimate win probability:

1. For each player, sample performance from normal distribution:
   - Mean = Player's average Game Score
   - Std = Historical variance (default: 8.0)

2. Sum team scores: $TeamScore = \sum_{players} Sample_i + HomeAdvantage$

3. Compare: $Win = HomeScore > AwayScore$

4. Repeat 1000 times, calculate: $P_{win} = \frac{Wins}{1000}$

**Injury Adjustment**: Exclude injured players from simulation.

---

### 9. Ensemble Prediction

Final probability combines two methods:

$$P_{final} = 0.7 \times P_{XGBoost} + 0.3 \times P_{MonteCarlo}$$

**Rationale**:
- XGBoost: Captures historical patterns and feature interactions
- Monte Carlo: Accounts for player-level variance and injuries

---

### 10. Bayesian Confidence Intervals

Quantify uncertainty in probability estimates using Beta distribution:

$$P \sim Beta(\alpha, \beta)$$

Where:
- $\alpha = P_{point} \times n_{samples}$
- $\beta = (1 - P_{point}) \times n_{samples}$

95% Confidence Interval: $[Beta_{0.025}, Beta_{0.975}]$

---

### 11. Market Efficiency Score

Measures how well market prices games:

$$Efficiency = 1 - (|P_{model} - P_{market}| \times Accuracy_{model})$$

Lower scores indicate more inefficiency (more betting opportunities).

---

## Pipeline Architecture

### High-Level Flow

```
1. Data Collection
   ├── Historical Game Stats (NBA API)
   ├── Player Stats (NBA API)
   ├── Live Odds (The Odds API)
   └── Injury Reports (ESPN Scraper)

2. Feature Engineering
   ├── Calculate ELO Ratings
   ├── Compute Advanced Stats (Pace, OffRtg, DefRtg)
   ├── Calculate Rolling Averages
   ├── Aggregate Player Talent Scores
   └── Add Fatigue Features (B2B, 3-in-4)

3. Model Training
   ├── Load Engineered Features
   ├── Grid Search Hyperparameters
   ├── Train XGBoost Classifier
   └── Save Model

4. Prediction & Betting
   ├── Load Latest Stats & Odds
   ├── Adjust for Injuries
   ├── Generate Predictions (XGBoost + Monte Carlo)
   ├── Calculate Expected Value
   ├── Apply Kelly Criterion
   └── Output Betting Recommendations

5. Evaluation
   ├── Backtest Historical Performance
   ├── Calculate ROI, Win Rate, Sharpe Ratio
   └── Generate Equity Curve
```

---

## File-by-File Technical Breakdown

### 📂 Data Collection (`src/data_collection/`)

#### `nba_games.py`
**Purpose**: Fetch historical game statistics from NBA API

**Key Functions**:
- `get_games_for_season(season_str)`: Fetches all games for a season
- `get_player_stats_for_season(season_str)`: Fetches player-level stats
- `calculate_advanced_stats(df)`: Calculates Four Factors and travel distance

**Incremental Updates**: 
- Checks existing data file
- Only fetches current season and new games
- Appends to existing data (no re-download)

**Output**: `data/raw/nba_games_stats.csv`, `data/raw/nba_player_stats.csv`

---

#### `live_odds.py`
**Purpose**: Scrape real-time betting odds

**Key Functions**:
- `get_live_odds()`: Fetches odds from The Odds API
- Filters for major books (DraftKings, FanDuel, BetMGM)
- Extracts Moneyline and Spread odds

**Configuration**: Uses `ODDS_API_KEY` from environment variable

**Output**: `data/odds/live_odds.csv`

---

#### `scrape_injuries.py`
**Purpose**: Scrape current injury reports from ESPN

**Key Functions**:
- `scrape_espn_nba_injuries()`: Parses ESPN injury page
- Filters for players marked "Out" (excludes Questionable, Day-to-Day)
- Groups by team

**Output**: `data/raw/espn_injuries_current.csv`

---

#### `merge_odds.py`
**Purpose**: Merge historical odds with game stats for backtesting

**Key Functions**:
- `merge_data()`: Joins odds data with game stats
- Matches on date and team abbreviation
- Handles team name normalization

**Output**: `data/processed/nba_model_with_odds.csv`

---

### 📂 Feature Engineering (`src/features/`)

#### `engineer.py`
**Purpose**: Transform raw data into predictive features

**Key Functions**:

1. **`calculate_elo(df)`**:
   - Initializes all teams at 1500
   - Updates ratings chronologically
   - Applies Margin of Victory multiplier
   - Adds home advantage (60 points)

2. **`calculate_advanced_stats(df)`**:
   - Estimates possessions
   - Calculates Offensive/Defensive Rating
   - Computes Pace
   - Handles missing data gracefully

3. **`add_fatigue_features(df)`**:
   - Calculates days since last game
   - Flags Back-to-Back games (`IS_B2B`)
   - Flags 3-in-4 nights (`IS_3IN4`)

4. **`calculate_roster_strength(df_games)`**:
   - Loads player stats
   - Calculates Hollinger Game Score for each player
   - Aggregates top 9 players per team per game
   - Uses rolling averages to prevent leakage

5. **`create_rolling_features(df)`**:
   - Calculates 10-game rolling averages
   - Shifts by 1 to prevent data leakage
   - Applies to: OffRtg, DefRtg, Pace, Four Factors, Roster Strength

**Output**: `data/processed/nba_model.csv`

---

#### `calculate_player_stats.py`
**Purpose**: Process player-level data into talent scores

**Key Functions**:
- `create_talent_pool()`:
  - Calculates Hollinger Game Score for each game
  - Computes 10-game rolling averages
  - Extracts latest rating per player

**Output**: `data/processed/processed_player.csv`

---

#### `optimise_elo.py`
**Purpose**: Hyperparameter tuning for ELO system

**Key Functions**:
- `run_elo_simulation(k_factor, home_advantage)`: Tests ELO parameters
- Uses Grid Search to minimize Log Loss
- Optimal values: K=25, Home Advantage=60

---

### 📂 Models (`src/models/`)

#### `train_model.py`
**Purpose**: Train XGBoost classifier

**Key Functions**:
- `train()`:
  - Loads engineered features
  - Splits data (80/20 train/test)
  - Grid Search for hyperparameters:
    - `n_estimators`: [100, 200]
    - `max_depth`: [3, 4, 5]
    - `learning_rate`: [0.01, 0.1]
    - `subsample`: [0.8]
    - `colsample_bytree`: [0.8]
  - Evaluates with accuracy and classification report
  - Saves best model

**Output**: `models/nba_xgb_model.joblib`

**Features Used**:
- ELO_TEAM, ELO_OPP
- IS_HOME, IS_B2B, IS_3IN4
- ROLL_OFF_RTG, ROLL_DEF_RTG, ROLL_PACE
- ROLL_EFG_PCT, ROLL_TOV_PCT, ROLL_ORB_PCT, ROLL_FTR
- ROLL_ROSTER_TALENT_SCORE

---

#### `predict_today.py`
**Purpose**: Generate daily betting recommendations

**Key Functions**:

1. **`load_injury_report()`**:
   - Loads injury CSV
   - Filters for "Out" status only
   - Returns dict: {team: [injured_players]}

2. **`get_roster_strength_simulation(team, injuries, players)`**:
   - Filters roster by injuries
   - Calculates points per minute
   - Projects minutes for top 9 players
   - Returns total projected score

3. **`run_monte_carlo(home, away, players, injuries)`**:
   - Samples player performances
   - Accounts for injuries
   - Runs 1000 simulations
   - Returns win probability

4. **`calibrate_probability(prob)`**:
   - Pulls probabilities toward 50% (if enabled)
   - Reduces overconfidence

5. **`get_kelly_bet(prob, odds, implied, bankroll)`**:
   - Calibrates probability
   - Calculates edge
   - Applies constraints (min/max edge, win prob bounds)
   - Calculates Kelly bet size
   - Returns (final_bet, raw_bet, edge, calibrated_prob)

6. **`predict()`**:
   - Loads all data (stats, odds, players, model)
   - For each game:
     - Gets latest team stats
     - Adjusts ELO for injuries
     - Generates XGBoost prediction
     - Runs Monte Carlo simulation
     - Calculates ensemble probability
     - Computes expected value
     - Calculates market efficiency
     - Applies Kelly Criterion
     - Calculates risk metrics
   - Outputs formatted betting card

**Output**: `results/todays_bets.csv`

---

#### `feature_importance.py`
**Purpose**: Analyze which features drive predictions

**Key Functions**:
- `show_importance()`:
  - Loads trained model
  - Extracts feature importances
  - Creates bar plot

**Output**: Visualization of feature importance

---

### 📂 Evaluation (`src/evaluation/`)

#### `backtest.py`
**Purpose**: Simulate historical betting performance

**Key Functions**:
- `backtest()`:
  - Loads historical data with odds
  - Splits into train/test (80/20)
  - Generates predictions for test set
  - Simulates betting with Kelly Criterion
  - Tracks bankroll over time
  - Calculates ROI, win rate
  - Generates equity curve

**Metrics**:
- ROI: (Final Bankroll - Initial) / Initial
- Win Rate: Wins / Total Bets
- Sharpe Ratio: Mean Return / Std Dev

**Output**: `results/backtest_chart.png`

---

#### `calibration_analysis.py`
**Purpose**: Analyze model calibration (predicted vs actual probabilities)

**Key Functions**:
- Creates calibration curve
- Groups predictions into bins
- Compares predicted vs actual win rates
- Identifies overconfidence/underconfidence

**Output**: `results/calibration_plot.png`

---

### 📂 Utilities (`src/utils/`)

#### `team_mapping.py`
**Purpose**: Centralized team name normalization

**Key Functions**:
- `normalize_team_name(name)`: Converts any team name format to standard abbreviation
- Handles: Full names, abbreviations, lowercase, historic teams

---

#### `betting_advanced.py`
**Purpose**: Advanced betting techniques

**Key Functions**:

1. **`calculate_confidence_interval(prob, n_samples, confidence)`**:
   - Uses Beta distribution
   - Returns 95% confidence interval

2. **`bayesian_probability_update(prior, likelihood, weight)`**:
   - Combines prior belief with new evidence
   - Weighted average approach

3. **`calculate_market_efficiency_score(model_prob, implied_prob, accuracy)`**:
   - Measures market efficiency
   - Lower = more opportunity

4. **`calculate_expected_value(prob, odds)`**:
   - Standard EV calculation
   - Returns as percentage

5. **`calculate_bankroll_risk(prob, bet_size, bankroll, odds)`**:
   - Calculates risk metrics
   - Returns: ruin probability, expected loss, risk/reward ratio

---

### 📂 Main Pipeline (`main.py`)

**Purpose**: Orchestrates entire pipeline

**Steps**:
1. Update game and player data
2. Process player talent scores
3. Engineer features (ELO, rolling stats)
4. Retrain model
5. Scrape injuries
6. Scrape live odds
7. Generate predictions

**Error Handling**: Stops pipeline if any step fails

---

## Advanced Techniques

### 1. Probability Calibration
**Problem**: XGBoost probabilities are often overconfident

**Solution**: 
- Sigmoid calibration: Pulls probabilities toward 50%
- Edge dampening: Reduces calculated edge by calibration factor
- Confidence intervals: Quantify uncertainty

### 2. Market Efficiency Analysis
**Purpose**: Identify when market is mispriced

**Method**: Compare model probability to implied probability
- Large discrepancies = market inefficiency
- Small discrepancies = efficient market

### 3. Risk Management
**Multi-layered approach**:
1. Fractional Kelly (35% of optimal)
2. Maximum bet cap (5% of bankroll)
3. Edge thresholds (min 2%, max 15%)
4. Win probability bounds (30%-90%)
5. Confidence intervals for uncertainty

### 4. Injury Adjustment
**Method**:
1. Load injury report (only "Out" players)
2. Calculate roster strength without injured players
3. Adjust ELO: $ELO_{adj} = ELO_{base} - (1 - HealthRatio) \times 250$
4. Exclude from Monte Carlo simulation

### 5. Ensemble Methods
**Hybrid Approach**:
- 70% XGBoost: Historical patterns
- 30% Monte Carlo: Player-level variance

**Rationale**: Combines strengths of both methods

---

## Data Flow

### Training Data Flow
```
Raw Game Stats → Engineer Features → Train Model
     ↓
Player Stats → Calculate Talent → Aggregate to Teams
     ↓
Historical Odds → Merge with Stats → Backtest
```

### Prediction Data Flow
```
Live Odds → Load Model → Generate Predictions
     ↓
Injury Report → Adjust ELO → Monte Carlo
     ↓
Latest Stats → Feature Engineering → XGBoost
     ↓
Ensemble → Kelly Criterion → Betting Card
```

---

## Model Training & Evaluation

### Training Process
1. **Data Split**: 80% train, 20% test
2. **Grid Search**: Tests hyperparameter combinations
3. **Cross-Validation**: 3-fold CV during grid search
4. **Evaluation**: Accuracy, Log Loss, Classification Report

### Model Performance Metrics
- **Accuracy**: Overall correct predictions
- **Log Loss**: Probability calibration quality
- **Brier Score**: Mean squared error of probabilities

### Backtesting Methodology
1. **Time-Based Split**: Last 20% of games (chronological)
2. **Out-of-Sample**: Model never sees test data during training
3. **Simulation**: Replays historical betting opportunities
4. **Metrics**: ROI, Win Rate, Sharpe Ratio, Max Drawdown

---

## Configuration Parameters

### Betting Parameters
```python
BANKROLL = 1000
KELLY_FRACTION = 0.35
MAX_BET_PCT = 0.05
MIN_EDGE_THRESHOLD = 0.02
MAX_EDGE_THRESHOLD = 0.15
MIN_WIN_PROB = 0.30
MAX_WIN_PROB = 0.90
```

### ELO Parameters
```python
K_FACTOR = 25
HOME_ADVANTAGE = 60
INITIAL_ELO = 1500
```

### Feature Engineering
```python
ROLLING_WINDOW = 10
MIN_PERIODS = 3
INJURY_PENALTY = 250  # ELO points
```

### Ensemble Weights
```python
XGBOOST_WEIGHT = 0.7
MONTE_CARLO_WEIGHT = 0.3
```

---

## Mathematical Formulas Summary

### Core Betting Formulas

**Expected Value**:
$$EV = (P \times (Odds - 1)) - ((1-P) \times 1)$$

**Kelly Criterion**:
$$f^* = \frac{(Odds-1) \times P - (1-P)}{Odds-1}$$

**Fractional Kelly**:
$$f = 0.35 \times f^*$$

**Edge**:
$$Edge = P_{model} - P_{market}$$

### ELO Formulas

**Expected Win Probability**:
$$P_{win} = \frac{1}{1 + 10^{-(R_1 - R_2 + H)/400}}$$

**Rating Update**:
$$R_{new} = R_{old} + K \times S \times (Actual - Expected)$$

**Margin of Victory Multiplier**:
$$S = \frac{(MOV + 3)^{0.8}}{7.5 + 0.006 \times |EloDiff|}$$

### Statistical Formulas

**Offensive Rating**:
$$OffRtg = 100 \times \frac{PTS}{Possessions}$$

**Effective Field Goal %**:
$$eFG\% = \frac{FGM + 0.5 \times FG3M}{FGA}$$

**Rolling Average**:
$$Roll_t = \frac{1}{n} \sum_{i=t-n}^{t-1} Feature_i$$

---

## Best Practices

### Data Integrity
1. Always use `shift(1)` for rolling features
2. Sort by date before calculations
3. Handle missing data explicitly
4. Validate feature existence before prediction

### Model Safety
1. Cap maximum edges (15%)
2. Use fractional Kelly (35%)
3. Set win probability bounds
4. Calculate confidence intervals
5. Monitor calibration

### Code Quality
1. Centralize team name mapping
2. Use environment variables for API keys
3. Comprehensive error handling
4. Clear logging and output
5. Modular, reusable functions

---

## Conclusion

This system combines:
- **Machine Learning**: XGBoost for pattern recognition
- **Statistical Methods**: ELO ratings, Monte Carlo simulation
- **Financial Theory**: Kelly Criterion for optimal bet sizing
- **Data Science Best Practices**: Time-series lagging, calibration, backtesting

The result is a robust, mathematically sound betting system that identifies positive expected value opportunities while managing risk appropriately.