"""
Advanced betting techniques for improved decision making.
Includes Bayesian updating, confidence intervals, and market efficiency analysis.
"""

import numpy as np
from scipy import stats
from scipy.stats import beta


def calculate_confidence_interval(prob, n_samples=1000, confidence=0.95):
    """
    Calculate confidence interval for probability estimate using Bayesian approach.
    
    Uses Beta distribution to model uncertainty in probability estimates.
    
    Args:
        prob: Point estimate of probability
        n_samples: Effective sample size (higher = more confident)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Use Beta distribution: Beta(alpha, beta) where alpha = prob * n, beta = (1-prob) * n
    alpha = prob * n_samples
    beta_param = (1 - prob) * n_samples
    
    # Ensure minimum values for stability
    alpha = max(alpha, 1)
    beta_param = max(beta_param, 1)
    
    # Calculate confidence interval
    lower = beta.ppf((1 - confidence) / 2, alpha, beta_param)
    upper = beta.ppf(1 - (1 - confidence) / 2, alpha, beta_param)
    
    return (lower, upper)


def bayesian_probability_update(prior_prob, likelihood, prior_weight=0.7):
    """
    Bayesian updating of probability estimates.
    
    Combines prior belief with new evidence using Bayes' theorem.
    
    Args:
        prior_prob: Prior probability estimate
        likelihood: New evidence probability
        prior_weight: Weight given to prior (0-1, higher = trust prior more)
    
    Returns:
        Updated probability estimate
    """
    # Simple weighted average (can be made more sophisticated)
    posterior = prior_weight * prior_prob + (1 - prior_weight) * likelihood
    return posterior


def calculate_market_efficiency_score(model_prob, implied_prob, historical_accuracy=0.55):
    """
    Calculate market efficiency score.
    
    Measures how well the market prices games relative to model predictions.
    Lower scores indicate more market inefficiency (more betting opportunities).
    
    Args:
        model_prob: Model's predicted probability
        implied_prob: Market implied probability
        historical_accuracy: Historical model accuracy (for calibration)
    
    Returns:
        Efficiency score (0-1, lower = less efficient = more opportunity)
    """
    # Calculate absolute difference
    diff = abs(model_prob - implied_prob)
    
    # Adjust for model accuracy (if model is less accurate, market is more efficient)
    efficiency = 1 - (diff * historical_accuracy)
    
    return max(0, min(1, efficiency))


def calculate_expected_value(prob_win, decimal_odds):
    """
    Calculate expected value of a bet.
    
    EV = (Probability of Win × Payout) - (Probability of Loss × Stake)
    
    Args:
        prob_win: Probability of winning
        decimal_odds: Decimal odds
    
    Returns:
        Expected value as percentage of stake
    """
    prob_loss = 1 - prob_win
    payout = decimal_odds - 1  # Net profit if win
    
    ev = (prob_win * payout) - (prob_loss * 1)
    return ev


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio for betting strategy.
    
    Measures risk-adjusted returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default 0 for betting)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(returns)


def calculate_kelly_with_uncertainty(prob_win, decimal_odds, prob_lower, prob_upper):
    """
    Calculate Kelly bet size accounting for uncertainty in probability estimate.
    
    Uses conservative approach: takes minimum of Kelly calculations at confidence bounds.
    
    Args:
        prob_win: Point estimate of win probability
        decimal_odds: Decimal odds
        prob_lower: Lower bound of confidence interval
        prob_upper: Upper bound of confidence interval
    
    Returns:
        Conservative Kelly bet percentage
    """
    b = decimal_odds - 1
    
    if b <= 0:
        return 0.0
    
    # Calculate Kelly at different probability levels
    kelly_point = (b * prob_win - (1 - prob_win)) / b
    kelly_lower = (b * prob_lower - (1 - prob_lower)) / b
    kelly_upper = (b * prob_upper - (1 - prob_upper)) / b
    
    # Use conservative estimate (minimum of the three)
    kelly_conservative = min(kelly_point, kelly_lower, kelly_upper)
    
    return max(0, kelly_conservative)


def calculate_bankroll_risk(prob_win, bet_size, bankroll, decimal_odds):
    """
    Calculate risk metrics for a bet.
    
    Args:
        prob_win: Probability of winning
        bet_size: Size of bet
        bankroll: Current bankroll
        decimal_odds: Decimal odds
    
    Returns:
        Dictionary with risk metrics:
        - ruin_prob: Probability of losing entire bet
        - expected_loss: Expected loss if bet fails
        - risk_reward_ratio: Risk/Reward ratio
    """
    prob_loss = 1 - prob_win
    potential_win = bet_size * (decimal_odds - 1)
    potential_loss = bet_size
    
    ruin_prob = prob_loss
    expected_loss = prob_loss * potential_loss
    risk_reward = potential_loss / potential_win if potential_win > 0 else float('inf')
    
    return {
        'ruin_prob': ruin_prob,
        'expected_loss': expected_loss,
        'risk_reward_ratio': risk_reward,
        'potential_win': potential_win,
        'potential_loss': potential_loss
    }

