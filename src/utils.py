"""
Core utility functions for Adaptive Query with AI Persona.

This module provides foundational functions used across all querying methods:
- Posterior computation over personas
- Posterior predictive distributions
- Objective functionals (entropy, variance)
- Evaluation metrics

Optimizations:
- Numba JIT-compiled versions of core functions
- Pre-computed data structures for fast array operations
- Vectorized operations where possible

Data Structure Conventions:
---------------------------
- persona_responses: pd.DataFrame
    Rows are personas (n personas), columns are questions (m questions).
    Index and columns are strings.
    Each entry is either None (missing) or a list of length K representing
    a probability distribution over K possible categorical answers.
    Shape conceptually: (n_personas, n_questions), entries are List[float] or None.

- user_responses: pd.DataFrame
    Rows are users, columns are questions.
    Index and columns are strings.
    Each entry is an np.int64 representing the index of the user's response
    (0-indexed category). Missing entries are marked by -1.
    Shape conceptually: (n_users, n_questions), entries are np.int64.

- posterior_weights: np.ndarray of shape (n_personas,)
    The posterior distribution p(θ | y_{I_t}) over personas.
    Must sum to 1.

- prior_weights: np.ndarray of shape (n_personas,)
    The prior distribution p(θ) over personas.
    Default is uniform: 1/n for each persona.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Optional

# Check for Numba availability
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# =============================================================================
# Constants
# =============================================================================

# Small constant for numerical stability in log computations
EPS = 1e-10


# =============================================================================
# Posterior Computation Functions
# =============================================================================

def compute_posterior_over_personas(
    prior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    asked_questions: List[str],
    observed_answers: List[int],
) -> np.ndarray:
    """
    Compute the posterior distribution over personas given observed answers.
    
    Implements Equation (9) from the framework:
        p(θ | y_{I_t}) ∝ p(θ) ∏_{i ∈ I_t} μ_{θ,i,y_i}
    
    where μ_{θ,i,y_i} is the probability persona θ assigns to answer y_i for question i.
    
    Parameters
    ----------
    prior_weights : np.ndarray of shape (n_personas,)
        Prior distribution p(θ) over personas. Must sum to 1.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
        Each entry is a list representing the probability distribution over answers.
    asked_questions : List[str]
        List of question identifiers (column names) that have been asked.
    observed_answers : List[int]
        List of observed answers (0-indexed) corresponding to asked_questions.
        Length must match asked_questions.
    
    Returns
    -------
    posterior_weights : np.ndarray of shape (n_personas,)
        Normalized posterior distribution p(θ | y_{I_t}) over personas.
    
    Notes
    -----
    - If asked_questions is empty, returns the prior_weights unchanged.
    - Uses log-space computation for numerical stability with many observations.
    - Handles edge cases where some personas assign zero probability to observed answers
      by using a small epsilon for numerical stability.
    
    Example
    -------
    >>> prior = np.ones(100) / 100  # Uniform prior over 100 personas
    >>> posterior = compute_posterior_over_personas(
    ...     prior, persona_df, ['q1', 'q2'], [0, 2]
    ... )
    """
    n_personas = len(prior_weights)
    
    # If no questions asked yet, return the prior
    if len(asked_questions) == 0:
        return prior_weights.copy()
    
    assert len(asked_questions) == len(observed_answers), \
        "asked_questions and observed_answers must have the same length"
    
    # Compute log-likelihood for each persona
    # log p(y_{I_t} | θ) = Σ_{i ∈ I_t} log μ_{θ,i,y_i}
    log_likelihood = np.zeros(n_personas)
    
    for question, answer in zip(asked_questions, observed_answers):
        for theta in range(n_personas):
            prob_dist = persona_responses.iloc[theta][question]
            if prob_dist is None:
                # If persona has no response for this question, assign uniform
                # This shouldn't happen in practice if data is properly prepared
                log_likelihood[theta] += np.log(EPS)
            else:
                prob = prob_dist[answer]
                log_likelihood[theta] += np.log(max(prob, EPS))
    
    # Compute log posterior: log p(θ | y_{I_t}) = log p(θ) + log p(y_{I_t} | θ) + const
    log_prior = np.log(np.maximum(prior_weights, EPS))
    log_posterior_unnorm = log_prior + log_likelihood
    
    # Normalize in log space using log-sum-exp trick for numerical stability
    max_log = np.max(log_posterior_unnorm)
    log_posterior_unnorm_shifted = log_posterior_unnorm - max_log
    posterior_weights = np.exp(log_posterior_unnorm_shifted)
    posterior_weights = posterior_weights / np.sum(posterior_weights)
    
    return posterior_weights


def compute_posterior_predictive(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    question: str,
) -> np.ndarray:
    """
    Compute the posterior predictive distribution for a question.
    
    Implements Equation (10) from the framework:
        p(Y_x = k | y_{I_t}) = Σ_{θ=1}^n μ_{θ,x,k} * p(θ | y_{I_t})
    
    This is a mixture of persona response distributions weighted by the posterior.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas p(θ | y_{I_t}).
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    question : str
        The question identifier (column name) to compute predictive for.
    
    Returns
    -------
    predictive_dist : np.ndarray of shape (K,)
        Posterior predictive distribution over the K possible answers.
        Sums to 1.
    
    Example
    -------
    >>> posterior = compute_posterior_over_personas(prior, df, ['q1'], [0])
    >>> pred_dist = compute_posterior_predictive(posterior, df, 'q2')
    >>> print(f"P(Y_q2 = 0) = {pred_dist[0]:.3f}")
    """
    n_personas = len(posterior_weights)
    
    # Get the number of categories K from the first non-None entry
    K = None
    for theta in range(n_personas):
        prob_dist = persona_responses.iloc[theta][question]
        if prob_dist is not None:
            K = len(prob_dist)
            break
    
    if K is None:
        raise ValueError(f"Question '{question}' has no valid responses in any persona")
    
    # Compute weighted average of persona response distributions
    predictive_dist = np.zeros(K)
    for theta in range(n_personas):
        prob_dist = persona_responses.iloc[theta][question]
        if prob_dist is not None:
            predictive_dist += posterior_weights[theta] * np.array(prob_dist)
        # If prob_dist is None, that persona contributes nothing (weight * 0)
    
    # Normalize to handle any numerical issues
    predictive_dist = predictive_dist / np.sum(predictive_dist)
    
    return predictive_dist


def update_posterior_with_observation(
    current_posterior: np.ndarray,
    persona_responses: pd.DataFrame,
    question: str,
    observed_answer: int,
) -> np.ndarray:
    """
    Update the posterior after observing an answer to a single question.
    
    This is an incremental version of compute_posterior_over_personas,
    useful for efficiency when processing observations one at a time.
    
    Implements:
        p(θ | y_{I_t}, y_x) ∝ p(θ | y_{I_t}) * μ_{θ,x,y_x}
    
    Parameters
    ----------
    current_posterior : np.ndarray of shape (n_personas,)
        Current posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    question : str
        The question that was asked.
    observed_answer : int
        The observed answer (0-indexed).
    
    Returns
    -------
    updated_posterior : np.ndarray of shape (n_personas,)
        Updated posterior distribution after incorporating the new observation.
    
    Example
    -------
    >>> posterior = prior_weights.copy()
    >>> for q, a in zip(questions, answers):
    ...     posterior = update_posterior_with_observation(posterior, df, q, a)
    """
    n_personas = len(current_posterior)
    
    # Compute likelihood for this observation for each persona
    likelihood = np.zeros(n_personas)
    for theta in range(n_personas):
        prob_dist = persona_responses.iloc[theta][question]
        if prob_dist is None:
            likelihood[theta] = EPS
        else:
            likelihood[theta] = max(prob_dist[observed_answer], EPS)
    
    # Update posterior: p_new ∝ p_old * likelihood
    updated_posterior = current_posterior * likelihood
    updated_posterior = updated_posterior / np.sum(updated_posterior)
    
    return updated_posterior


# =============================================================================
# Objective Functionals
# =============================================================================

def entropy(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy of a discrete probability distribution.
    
    H(p) = -Σ p_i log(p_i)
    
    Parameters
    ----------
    probs : np.ndarray
        Probability distribution (must sum to 1).
    
    Returns
    -------
    float
        Shannon entropy in nats (natural log).
    
    Example
    -------
    >>> entropy(np.array([0.5, 0.5]))  # Maximum entropy for 2 outcomes
    0.6931471805599453
    """
    # Filter out zero probabilities to avoid log(0)
    probs_positive = probs[probs > EPS]
    return -np.sum(probs_positive * np.log(probs_positive))


def entropy_over_personas(posterior_weights: np.ndarray) -> float:
    """
    Compute the entropy of the posterior distribution over personas.
    
    H(θ | y_{I_t}) = -Σ_θ p(θ | y_{I_t}) log p(θ | y_{I_t})
    
    Lower entropy means more certainty about which persona the user resembles.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    
    Returns
    -------
    float
        Entropy of the persona posterior in nats.
    """
    return entropy(posterior_weights)


def entropy_over_target_questions(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    target_questions: List[str],
) -> float:
    """
    Compute the mean entropy over predictions for target questions.
    
    (1/|I*|) Σ_{x ∈ I*} H(Y_x | y_{I_t})
    
    where H(Y_x | y_{I_t}) is the entropy of the posterior predictive for question x.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    target_questions : List[str]
        List of target question identifiers.
    
    Returns
    -------
    float
        Mean entropy over all target questions.
    """
    if len(target_questions) == 0:
        return 0.0
    total_entropy = 0.0
    for question in target_questions:
        pred_dist = compute_posterior_predictive(posterior_weights, persona_responses, question)
        total_entropy += entropy(pred_dist)
    return total_entropy / len(target_questions)


def variance_of_categorical(probs: np.ndarray) -> float:
    """
    Compute variance-like measure for a categorical distribution.
    
    Uses Gini impurity: 1 - Σ p_i^2
    This equals 0 when distribution is deterministic (one p_i = 1),
    and maximized when distribution is uniform.
    
    Parameters
    ----------
    probs : np.ndarray
        Probability distribution over categories.
    
    Returns
    -------
    float
        Gini impurity (variance proxy) in [0, 1-1/K].
    """
    return 1.0 - np.sum(probs ** 2)


def entropy_over_target_questions_overlapping(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    target_questions: List[str],
    asked_questions: Optional[List[str]] = None,
) -> float:
    """
    Compute mean entropy over target questions in overlapping mode.
    
    In overlapping mode, asked questions are treated as having point-mass
    predictions at the observed response, so their entropy = 0.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    target_questions : List[str]
        List of all target question identifiers.
    asked_questions : List[str], optional
        Questions that have been asked (entropy = 0 for these).
    
    Returns
    -------
    float
        Mean entropy over all target questions, with asked questions
        contributing 0.
    
    Notes
    -----
    In overlapping mode: target_questions = all questions.
    Asked questions get entropy = 0 (point mass at observed).
    Unasked questions get their posterior predictive entropy.
    """
    if len(target_questions) == 0:
        return 0.0
    
    asked_set = set(asked_questions) if asked_questions else set()
    total_entropy = 0.0
    
    for question in target_questions:
        if question in asked_set:
            # Asked questions: point mass → entropy = 0
            continue
        else:
            # Unasked questions: compute posterior predictive entropy
            pred_dist = compute_posterior_predictive(posterior_weights, persona_responses, question)
            total_entropy += entropy(pred_dist)
    
    return total_entropy / len(target_questions)


def variance_over_target_questions_overlapping(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    target_questions: List[str],
    asked_questions: Optional[List[str]] = None,
) -> float:
    """
    Compute mean variance (Gini impurity) over target questions in overlapping mode.
    
    In overlapping mode, asked questions are treated as having point-mass
    predictions, so their variance = 0.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    target_questions : List[str]
        List of all target question identifiers.
    asked_questions : List[str], optional
        Questions that have been asked (variance = 0 for these).
    
    Returns
    -------
    float
        Mean variance over all target questions.
    """
    if len(target_questions) == 0:
        return 0.0
    
    asked_set = set(asked_questions) if asked_questions else set()
    total_variance = 0.0
    
    for question in target_questions:
        if question in asked_set:
            # Asked questions: point mass → variance = 0
            continue
        else:
            # Unasked questions: compute posterior predictive variance
            pred_dist = compute_posterior_predictive(posterior_weights, persona_responses, question)
            total_variance += variance_of_categorical(pred_dist)
    
    return total_variance / len(target_questions)


def variance_over_target_questions(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    target_questions: List[str],
) -> float:
    """
    Compute mean variance (Gini impurity) over predictions for target questions.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    target_questions : List[str]
        List of target question identifiers.
    
    Returns
    -------
    float
        Mean variance (Gini impurity) over all target questions.
    """
    if len(target_questions) == 0:
        return 0.0
    total_variance = 0.0
    for question in target_questions:
        pred_dist = compute_posterior_predictive(posterior_weights, persona_responses, question)
        total_variance += variance_of_categorical(pred_dist)
    return total_variance / len(target_questions)


def crps_over_target_questions(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    target_questions: List[str],
) -> float:
    """
    Compute mean CRPS uncertainty over predictions for target questions.
    
    CRPS uncertainty is appropriate for ordinal data (like ratings) where
    the ordering of categories matters. It measures how spread out the
    predictive distribution is in an ordinal-aware way.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    target_questions : List[str]
        List of target question identifiers.
    
    Returns
    -------
    float
        Mean CRPS uncertainty over all target questions.
    """
    if len(target_questions) == 0:
        return 0.0
    total_crps = 0.0
    for question in target_questions:
        pred_dist = compute_posterior_predictive(posterior_weights, persona_responses, question)
        total_crps += crps_uncertainty(pred_dist)
    return total_crps / len(target_questions)


# =============================================================================
# Evaluation Metrics
# =============================================================================

def accuracy_score(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
) -> float:
    """
    Compute top-1 accuracy (1 if mode of prediction equals true response, 0 otherwise).
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution over K categories.
    true_response : int or np.ndarray
        Either the true category index (int) or one-hot encoding (array).
    
    Returns
    -------
    float
        1.0 if correct, 0.0 if incorrect.
    
    Example
    -------
    >>> accuracy_score(np.array([0.1, 0.6, 0.3]), 1)  # Predicted mode = 1
    1.0
    """
    if isinstance(true_response, np.ndarray):
        true_idx = int(np.argmax(true_response))
    else:
        true_idx = int(true_response)
    
    predicted_idx = int(np.argmax(predicted_dist))
    return 1.0 if predicted_idx == true_idx else 0.0


def brier_score(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
) -> float:
    """
    Compute the Brier score (mean squared error between prediction and one-hot true).
    
    Brier = (1/K) Σ_k (p_k - y_k)^2
    
    where y is one-hot encoding of true response.
    Lower is better. Range: [0, 2] for multiclass.
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution.
    true_response : int or np.ndarray
        True category index or one-hot encoding.
    
    Returns
    -------
    float
        Brier score.
    
    Example
    -------
    >>> brier_score(np.array([0.1, 0.9, 0.0]), 1)  # Good prediction
    0.006666...
    >>> brier_score(np.array([0.9, 0.1, 0.0]), 1)  # Bad prediction
    0.54
    """
    K = len(predicted_dist)
    
    if isinstance(true_response, np.ndarray):
        true_onehot = true_response
    else:
        true_onehot = np.zeros(K)
        true_onehot[int(true_response)] = 1.0
    
    return np.mean((predicted_dist - true_onehot) ** 2)


def log_loss_score(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
) -> float:
    """
    Compute the log loss (negative log-likelihood of true class).
    
    LogLoss = -log(p_{true})
    
    Lower is better. Range: [0, ∞).
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution.
    true_response : int or np.ndarray
        True category index or one-hot encoding.
    
    Returns
    -------
    float
        Log loss.
    
    Example
    -------
    >>> log_loss_score(np.array([0.1, 0.9, 0.0]), 1)
    0.1053...  # -log(0.9)
    """
    if isinstance(true_response, np.ndarray):
        true_idx = int(np.argmax(true_response))
    else:
        true_idx = int(true_response)
    
    prob = max(predicted_dist[true_idx], EPS)
    return -np.log(prob)


def kl_divergence(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
) -> float:
    """
    Compute KL divergence from predicted distribution to true (one-hot) distribution.
    
    KL(true || predicted) = Σ_k y_k log(y_k / p_k)
                          = -log(p_{true})  (for one-hot y)
    
    This equals log_loss when the true distribution is one-hot.
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution.
    true_response : int or np.ndarray
        True category index or one-hot encoding.
    
    Returns
    -------
    float
        KL divergence.
    
    Notes
    -----
    For one-hot true distribution, KL divergence equals log loss.
    """
    # For one-hot true distribution, KL = log_loss
    return log_loss_score(predicted_dist, true_response)


def crps_score(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for ordinal predictions.
    
    CRPS = Σ_k (F(k) - 1{y ≤ k})²
    
    where F(k) is the CDF at category k and y is the true category.
    
    CRPS is a proper scoring rule that respects the ordering of categories,
    making it particularly appropriate for ordinal data like ratings (1-5 stars).
    Unlike Brier score, CRPS penalizes predictions based on how far they are
    from the truth in the ordinal sense.
    
    Lower is better. Range: [0, K-1] where K is number of categories.
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution over K ordered categories.
    true_response : int or np.ndarray
        True category index (int) or one-hot encoding (array).
    
    Returns
    -------
    float
        CRPS score.
    
    Example
    -------
    >>> # Predicting rating 4 when truth is 3 (close, should have low CRPS)
    >>> crps_score(np.array([0.0, 0.0, 0.1, 0.8, 0.1]), 3)
    0.02
    >>> # Predicting rating 0 when truth is 4 (far, should have high CRPS)
    >>> crps_score(np.array([0.8, 0.1, 0.1, 0.0, 0.0]), 4)
    3.06
    
    Notes
    -----
    CRPS is equivalent to the integral of the squared difference between
    the predicted CDF and the empirical CDF (step function at true value).
    """
    K = len(predicted_dist)
    
    # Get true index
    if isinstance(true_response, np.ndarray):
        true_idx = int(np.argmax(true_response))
    else:
        true_idx = int(true_response)
    
    # Compute CDF: F(k) = P(Y <= k) = sum(p[0:k+1])
    cdf = np.cumsum(predicted_dist)
    
    # Compute CRPS: sum over k of (F(k) - 1{y <= k})^2
    # 1{y <= k} is 1 when k >= true_idx, 0 otherwise
    indicator = np.zeros(K)
    indicator[true_idx:] = 1.0
    
    crps = np.sum((cdf - indicator) ** 2)
    
    return crps


def crps_uncertainty(
    probs: np.ndarray,
) -> float:
    """
    Compute CRPS-based uncertainty for an ordinal probability distribution.
    
    This is the "spread" or "sharpness" component of CRPS:
    
    U(P) = (1/2) * E[|Y - Y'|] = (1/2) * Σ_i Σ_j P(i) * P(j) * |i - j|
    
    where Y, Y' are independent draws from P.
    
    This measures uncertainty in an ordinal-aware way: distributions spread
    across distant categories have higher uncertainty than those spread
    across adjacent categories.
    
    Higher means more uncertain. Range: [0, (K-1)/2] where K is number of categories.
    
    Parameters
    ----------
    probs : np.ndarray of shape (K,)
        Probability distribution over K ordered categories.
    
    Returns
    -------
    float
        CRPS-based uncertainty measure.
    
    Example
    -------
    >>> # Concentrated distribution (low uncertainty)
    >>> crps_uncertainty(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    0.0
    >>> # Spread between adjacent categories (moderate uncertainty)
    >>> crps_uncertainty(np.array([0.0, 0.5, 0.5, 0.0, 0.0]))
    0.25  # = 0.5 * 0.5 * |1-2| * 2 / 2 = 0.25
    >>> # Spread between distant categories (high uncertainty)
    >>> crps_uncertainty(np.array([0.5, 0.0, 0.0, 0.0, 0.5]))
    2.0  # = 0.5 * 0.5 * |0-4| * 2 / 2 = 2.0
    
    Notes
    -----
    This can also be computed efficiently using the CDF:
    U(P) = Σ_k F(k) * (1 - F(k))
    where F(k) is the CDF at category k.
    """
    K = len(probs)
    
    # Efficient computation using CDF formulation:
    # U(P) = Σ_k F(k) * (1 - F(k))
    cdf = np.cumsum(probs)
    uncertainty = np.sum(cdf * (1.0 - cdf))
    
    return uncertainty


def posterior_mean_se(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
    score_values: Optional[np.ndarray] = None,
) -> float:
    """
    Compute squared error between posterior mean score and true score.
    
    The posterior mean is E[score] = Σ_k p_k * score_k.
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution over K categories.
    true_response : int or np.ndarray
        True category index (int) or one-hot encoding (array).
    score_values : np.ndarray of shape (K,), optional
        Score values for each category. If None, uses category indices [0, 1, ..., K-1].
    
    Returns
    -------
    float
        Squared error (predicted_mean - true_score)^2.
    
    Example
    -------
    >>> posterior_mean_se(np.array([0.1, 0.2, 0.3, 0.4]), 2)
    0.04  # (0.1*0 + 0.2*1 + 0.3*2 + 0.4*3 - 2)^2 = (2.0 - 2)^2 = 0
    """
    K = len(predicted_dist)
    
    # Use category indices as scores if not provided
    if score_values is None:
        score_values = np.arange(K, dtype=np.float64)
    
    # Get true score
    if isinstance(true_response, np.ndarray):
        true_idx = int(np.argmax(true_response))
    else:
        true_idx = int(true_response)
    true_score = score_values[true_idx]
    
    # Compute posterior mean
    predicted_mean = np.sum(predicted_dist * score_values)
    
    # Squared error
    return (predicted_mean - true_score) ** 2


def ci_coverage(
    predicted_dist: np.ndarray,
    true_response: Union[int, np.ndarray],
    confidence_level: float = 0.95,
) -> float:
    """
    Check if a confidence interval constructed from the posterior covers the true response.
    
    The CI is constructed by excluding categories from both ends (lowest and highest)
    until we cannot exclude more without the remaining probability mass falling below
    the confidence level. The CI includes all categories that weren't excluded.
    
    Parameters
    ----------
    predicted_dist : np.ndarray of shape (K,)
        Predicted probability distribution over K ordered categories.
    true_response : int or np.ndarray
        True category index (int) or one-hot encoding (array).
    confidence_level : float, default=0.95
        Minimum probability mass to retain in the CI.
    
    Returns
    -------
    float
        1.0 if true response is covered by the CI, 0.0 otherwise.
    
    Notes
    -----
    This assumes categories are ordinal (e.g., ratings 0-4 or scores 0.5-5.0).
    The CI is defined by [lower_bound, upper_bound] where both bounds are 
    category indices.
    
    The algorithm greedily removes the category with smaller probability from
    either end (lower or upper) until removing another would drop below the
    confidence level.
    
    Example
    -------
    >>> # Distribution peaked at category 2
    >>> p = np.array([0.05, 0.15, 0.60, 0.15, 0.05])
    >>> ci_coverage(p, 2, confidence_level=0.90)
    1.0  # CI likely covers categories 1-3, true=2 is inside
    >>> ci_coverage(p, 0, confidence_level=0.90)
    0.0  # Category 0 excluded from CI
    """
    K = len(predicted_dist)
    
    if isinstance(true_response, np.ndarray):
        true_idx = int(np.argmax(true_response))
    else:
        true_idx = int(true_response)
    
    # Start with full range
    lower = 0
    upper = K - 1
    current_mass = 1.0
    
    # Greedily remove from ends
    while lower < upper:
        # Check which end has smaller probability
        prob_lower = predicted_dist[lower]
        prob_upper = predicted_dist[upper]
        
        # Try to remove the smaller one
        if prob_lower <= prob_upper:
            # Try removing lower
            if current_mass - prob_lower >= confidence_level:
                current_mass -= prob_lower
                lower += 1
            else:
                # Cannot remove lower, try upper
                if current_mass - prob_upper >= confidence_level:
                    current_mass -= prob_upper
                    upper -= 1
                else:
                    # Cannot remove either, stop
                    break
        else:
            # Try removing upper
            if current_mass - prob_upper >= confidence_level:
                current_mass -= prob_upper
                upper -= 1
            else:
                # Cannot remove upper, try lower
                if current_mass - prob_lower >= confidence_level:
                    current_mass -= prob_lower
                    lower += 1
                else:
                    # Cannot remove either, stop
                    break
    
    # Check if true response is in [lower, upper]
    return 1.0 if lower <= true_idx <= upper else 0.0


def apply_temperature_scaling(
    probs: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Apply temperature scaling to a probability distribution.
    
    Computes: p_τ(y) ∝ p(y)^{1/τ}
    
    Parameters
    ----------
    probs : np.ndarray of shape (K,)
        Original probability distribution (must sum to 1).
    temperature : float, default=1.0
        Temperature parameter τ.
        - τ = 1.0: No change (returns original distribution)
        - τ > 1.0: Softer/more uniform distribution
        - τ < 1.0: Sharper/more peaked distribution
    
    Returns
    -------
    scaled_probs : np.ndarray of shape (K,)
        Temperature-scaled probability distribution.
    
    Notes
    -----
    Uses log-space computation for numerical stability.
    
    Example
    -------
    >>> p = np.array([0.7, 0.2, 0.1])
    >>> apply_temperature_scaling(p, temperature=1.0)  # No change
    array([0.7, 0.2, 0.1])
    >>> apply_temperature_scaling(p, temperature=2.0)  # Softer
    array([0.553, 0.265, 0.182])  # More uniform
    """
    if temperature == 1.0:
        return probs
    
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Clip probabilities for numerical stability
    probs_clipped = np.maximum(probs, EPS)
    
    # Compute in log space: log(p^{1/τ}) = log(p) / τ
    log_probs = np.log(probs_clipped)
    log_scaled = log_probs / temperature
    
    # Normalize using log-sum-exp for stability
    max_log = np.max(log_scaled)
    log_scaled_shifted = log_scaled - max_log
    scaled_probs = np.exp(log_scaled_shifted)
    scaled_probs = scaled_probs / np.sum(scaled_probs)
    
    return scaled_probs


def create_point_mass(n_categories: int, true_response: int) -> np.ndarray:
    """
    Create a point-mass (one-hot) probability distribution.
    
    Parameters
    ----------
    n_categories : int
        Number of categories.
    true_response : int
        Index of the true category.
    
    Returns
    -------
    np.ndarray
        Point-mass distribution with 1.0 at true_response, 0.0 elsewhere.
    """
    dist = np.zeros(n_categories)
    dist[true_response] = 1.0
    return dist


def evaluate_predictions(
    predicted_distributions: Dict[str, np.ndarray],
    user_response_row: pd.Series,
    target_questions: List[str],
    temperature: float = 1.0,
    score_values: Optional[np.ndarray] = None,
    ci_confidence_level: float = 0.95,
    asked_questions: Optional[List[str]] = None,
    n_categories: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate predictions on target questions using multiple metrics.
    
    Parameters
    ----------
    predicted_distributions : Dict[str, np.ndarray]
        Dictionary mapping question IDs to predicted distributions.
    user_response_row : pd.Series
        User's actual responses. Each entry is an np.int64 representing the
        response index (0-indexed). Missing entries are marked by -1.
    target_questions : List[str]
        List of target questions to evaluate.
    temperature : float, default=1.0
        Temperature for scaling predictions. τ > 1 makes distributions softer.
    score_values : np.ndarray, optional
        Score values for each category (for computing posterior mean MSE).
        If None, uses category indices [0, 1, ..., K-1].
    ci_confidence_level : float, default=0.95
        Confidence level for constructing confidence intervals.
    asked_questions : List[str], optional
        Questions that have been asked (for overlapping mode).
        For these questions, predictions are overridden with point-mass at
        the observed response, yielding perfect scores.
    n_categories : int, optional
        Number of categories (required if asked_questions is provided and
        some asked questions have no predicted_distribution entry).
    
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary with average metrics:
        - 'accuracy': Average top-1 accuracy
        - 'brier_score': Average Brier score
        - 'log_loss': Average log loss
        - 'kl_divergence': Average KL divergence
        - 'mse': Average squared error between posterior mean and true score
        - 'ci_coverage': Coverage rate of confidence intervals
        - 'crps': Average CRPS (ordinal-aware scoring rule)
        - 'n_evaluated': Number of questions evaluated
    
    Notes
    -----
    Only evaluates questions that have both predictions and true responses
    (true response != -1).
    
    In overlapping mode (when asked_questions is provided), asked questions
    are EXCLUDED from evaluation. Metrics are computed only over unasked
    target questions. This means:
    - If 100 target questions and 20 are asked, metrics are averaged over 80
    - Metrics reflect remaining uncertainty, not already-observed answers
    - n_evaluated will be the number of unasked questions with valid predictions
    
    Temperature scaling is applied as: p_τ(y) ∝ p(y)^{1/τ}
    
    Example
    -------
    >>> metrics = evaluate_predictions(
    ...     result['predicted_distributions'],
    ...     user_df.iloc[0],
    ...     ['q10', 'q11'],
    ...     temperature=1.5
    ... )
    >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    >>> print(f"MSE: {metrics['mse']:.4f}")
    >>> print(f"CI Coverage: {metrics['ci_coverage']:.2%}")
    """
    asked_set = set(asked_questions) if asked_questions else set()
    
    accuracies = []
    brier_scores = []
    log_losses = []
    kl_divergences = []
    mse_values = []
    ci_coverages = []
    crps_values = []
    
    for q in target_questions:
        # Skip asked questions - only evaluate on unasked questions
        # This way metrics reflect remaining uncertainty, not already-observed answers
        if q in asked_set:
            continue
        
        # Check if we have true response
        if q not in user_response_row.index or user_response_row[q] == -1:
            continue
        
        true_response = int(user_response_row[q])
        
        # Check if we have a prediction for this question
        if q not in predicted_distributions:
            continue
        
        pred_dist = predicted_distributions[q]
        
        # Apply temperature scaling
        if temperature != 1.0:
            pred_dist = apply_temperature_scaling(pred_dist, temperature)
        
        accuracies.append(accuracy_score(pred_dist, true_response))
        brier_scores.append(brier_score(pred_dist, true_response))
        log_losses.append(log_loss_score(pred_dist, true_response))
        kl_divergences.append(kl_divergence(pred_dist, true_response))
        mse_values.append(posterior_mean_se(pred_dist, true_response, score_values))
        ci_coverages.append(ci_coverage(pred_dist, true_response, ci_confidence_level))
        crps_values.append(crps_score(pred_dist, true_response))
    
    n_evaluated = len(accuracies)
    
    if n_evaluated == 0:
        return {
            'accuracy': np.nan,
            'brier_score': np.nan,
            'log_loss': np.nan,
            'kl_divergence': np.nan,
            'mse': np.nan,
            'ci_coverage': np.nan,
            'crps': np.nan,
            'n_evaluated': 0,
        }
    
    return {
        'accuracy': np.mean(accuracies),
        'brier_score': np.mean(brier_scores),
        'log_loss': np.mean(log_losses),
        'kl_divergence': np.mean(kl_divergences),
        'mse': np.mean(mse_values),
        'ci_coverage': np.mean(ci_coverages),
        'crps': np.mean(crps_values),
        'n_evaluated': n_evaluated,
    }


# =============================================================================
# Pre-computed Data Structures (Shared across all methods)
# =============================================================================

class PrecomputedPersonaData:
    """
    Pre-computed NumPy arrays for fast persona-based computations.
    
    Converts DataFrame lookups to O(1) array indexing.
    This class can be shared across greedy, baselines, CAT, and policy gradient.
    """
    
    def __init__(
        self,
        persona_responses: pd.DataFrame,
        questions: List[str],
    ):
        """
        Pre-compute persona response arrays.
        
        Parameters
        ----------
        persona_responses : pd.DataFrame
            DataFrame with personas as rows, questions as columns.
            Each entry is a list of probabilities.
        questions : List[str]
            List of questions to include.
        """
        self.n_personas = len(persona_responses)
        self.questions = questions
        self.n_questions = len(questions)
        self.question_to_idx = {q: i for i, q in enumerate(questions)}
        self.idx_to_question = {i: q for i, q in enumerate(questions)}
        
        # Determine K from first non-None entry
        self.K = None
        for q in questions:
            if q in persona_responses.columns:
                for persona_idx in range(self.n_personas):
                    entry = persona_responses.iloc[persona_idx][q]
                    if entry is not None:
                        self.K = len(entry)
                        break
            if self.K is not None:
                break
        
        if self.K is None:
            raise ValueError("Could not determine number of categories K")
        
        # Build 3D array: (n_personas, n_questions, K)
        # persona_probs[p, q, k] = P(answer=k | persona=p, question=q)
        self.persona_probs = np.zeros(
            (self.n_personas, self.n_questions, self.K), 
            dtype=np.float64, 
            order='C'  # Ensure C-contiguous for Numba
        )
        
        for q_idx, q in enumerate(questions):
            if q in persona_responses.columns:
                for p_idx in range(self.n_personas):
                    entry = persona_responses.iloc[p_idx][q]
                    if entry is not None:
                        probs = np.array(entry, dtype=np.float64)
                        # Ensure proper normalization
                        probs = probs / (probs.sum() + EPS)
                        self.persona_probs[p_idx, q_idx, :len(probs)] = probs
        
        # Ensure the array is contiguous
        self.persona_probs = np.ascontiguousarray(self.persona_probs)


def precompute_persona_data(
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
) -> PrecomputedPersonaData:
    """
    Create pre-computed persona data for fast computations.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Persona response distributions.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    
    Returns
    -------
    PrecomputedPersonaData
        Pre-computed arrays and mappings.
    """
    all_questions = list(set(feasible_questions) | set(target_questions))
    return PrecomputedPersonaData(persona_responses, all_questions)


# =============================================================================
# Numba JIT-Compiled Core Functions
# =============================================================================

@jit(nopython=True, cache=True)
def compute_posterior_predictive_jit(
    posterior_weights: np.ndarray,
    persona_probs_for_question: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled posterior predictive computation.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Current posterior over personas.
    persona_probs_for_question : np.ndarray of shape (n_personas, K)
        P(answer=k | persona=p) for all personas and answers.
    
    Returns
    -------
    np.ndarray of shape (K,)
        Posterior predictive distribution.
    """
    # Ensure contiguous for better performance
    q_probs = np.ascontiguousarray(persona_probs_for_question)
    predictive = np.dot(posterior_weights, q_probs)
    total = predictive.sum()
    if total > 1e-10:
        predictive = predictive / total
    return predictive


@jit(nopython=True, cache=True)
def update_posterior_jit(
    posterior_weights: np.ndarray,
    persona_probs_for_question: np.ndarray,
    observed_answer: int,
) -> np.ndarray:
    """
    JIT-compiled posterior update after observing an answer.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Current posterior.
    persona_probs_for_question : np.ndarray of shape (n_personas, K)
        P(answer=k | persona=p).
    observed_answer : int
        The observed answer index.
    
    Returns
    -------
    np.ndarray of shape (n_personas,)
        Updated posterior.
    """
    # Ensure contiguous
    q_probs = np.ascontiguousarray(persona_probs_for_question)
    # Likelihood: P(Y_x = k | persona = p)
    likelihoods = q_probs[:, observed_answer]
    
    # Add small epsilon for numerical stability
    likelihoods = np.maximum(likelihoods, 1e-10)
    
    # Bayes update
    updated = posterior_weights * likelihoods
    total = updated.sum()
    if total > 1e-10:
        updated = updated / total
    else:
        # Fallback to uniform if all likelihoods are zero
        updated = np.ones_like(posterior_weights) / len(posterior_weights)
    
    return updated


@jit(nopython=True, cache=True)
def compute_posterior_over_personas_jit(
    prior_weights: np.ndarray,
    persona_probs: np.ndarray,
    asked_question_indices: np.ndarray,
    observed_answers: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled posterior computation over personas.
    
    Parameters
    ----------
    prior_weights : np.ndarray of shape (n_personas,)
        Prior distribution over personas.
    persona_probs : np.ndarray of shape (n_personas, n_questions, K)
        Pre-computed persona probabilities.
    asked_question_indices : np.ndarray of shape (n_asked,)
        Indices of asked questions.
    observed_answers : np.ndarray of shape (n_asked,)
        Observed answers (0-indexed).
    
    Returns
    -------
    np.ndarray of shape (n_personas,)
        Posterior distribution.
    """
    n_personas = len(prior_weights)
    n_asked = len(asked_question_indices)
    
    if n_asked == 0:
        return prior_weights.copy()
    
    # Compute log-likelihood for each persona
    log_likelihood = np.zeros(n_personas)
    
    for i in range(n_asked):
        q_idx = asked_question_indices[i]
        answer = observed_answers[i]
        q_probs = np.ascontiguousarray(persona_probs[:, q_idx, :])
        for p in range(n_personas):
            prob = q_probs[p, answer]
            log_likelihood[p] += np.log(max(prob, 1e-10))
    
    # Compute log posterior
    log_prior = np.zeros(n_personas)
    for p in range(n_personas):
        log_prior[p] = np.log(max(prior_weights[p], 1e-10))
    
    log_posterior_unnorm = log_prior + log_likelihood
    
    # Normalize using log-sum-exp trick
    max_log = np.max(log_posterior_unnorm)
    log_posterior_shifted = log_posterior_unnorm - max_log
    posterior_weights = np.exp(log_posterior_shifted)
    posterior_weights = posterior_weights / np.sum(posterior_weights)
    
    return posterior_weights


@jit(nopython=True, cache=True)
def entropy_jit(probs: np.ndarray) -> float:
    """JIT-compiled Shannon entropy computation."""
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * np.log(p)
    return entropy


@jit(nopython=True, cache=True)
def variance_categorical_jit(probs: np.ndarray) -> float:
    """JIT-compiled Gini impurity (1 - sum(p^2))."""
    return 1.0 - np.sum(probs * probs)


@jit(nopython=True, cache=True)
def entropy_over_target_questions_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    """
    JIT-compiled entropy over target question predictions.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
    persona_probs : np.ndarray of shape (n_personas, n_questions, K)
    target_indices : np.ndarray of shape (n_targets,)
        Indices of target questions.
    
    Returns
    -------
    float
        Mean entropy over target question predictions.
    """
    total_entropy = 0.0
    n_targets = len(target_indices)
    
    for t_idx in range(n_targets):
        q_idx = target_indices[t_idx]
        # Compute posterior predictive for this target question
        q_probs = np.ascontiguousarray(persona_probs[:, q_idx, :])
        predictive = np.dot(posterior_weights, q_probs)
        total = predictive.sum()
        if total > 1e-10:
            predictive = predictive / total
        
        # Compute entropy
        entropy = 0.0
        for p in predictive:
            if p > 1e-10:
                entropy -= p * np.log(p)
        total_entropy += entropy
    
    return total_entropy / max(n_targets, 1)


@jit(nopython=True, cache=True)
def variance_over_target_questions_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    """
    JIT-compiled variance (Gini impurity) over target question predictions.
    
    Returns mean variance over target questions.
    """
    total_variance = 0.0
    n_targets = len(target_indices)
    
    for t_idx in range(n_targets):
        q_idx = target_indices[t_idx]
        q_probs = np.ascontiguousarray(persona_probs[:, q_idx, :])
        predictive = np.dot(posterior_weights, q_probs)
        total = predictive.sum()
        if total > 1e-10:
            predictive = predictive / total
        
        # Gini impurity
        variance = 1.0 - np.sum(predictive * predictive)
        total_variance += variance
    
    return total_variance / max(n_targets, 1)


@jit(nopython=True, cache=True)
def crps_uncertainty_jit(probs: np.ndarray) -> float:
    """
    JIT-compiled CRPS-based uncertainty for ordinal distribution.
    
    U(P) = Σ_k F(k) * (1 - F(k))
    
    where F(k) is the CDF at category k.
    """
    K = len(probs)
    cdf = 0.0
    uncertainty = 0.0
    
    for k in range(K):
        cdf += probs[k]
        uncertainty += cdf * (1.0 - cdf)
    
    return uncertainty


@jit(nopython=True, cache=True)
def crps_over_target_questions_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    """
    JIT-compiled CRPS uncertainty over target question predictions.
    
    Computes mean CRPS-based uncertainty over all target questions.
    This is appropriate for ordinal data like ratings where the
    ordering of categories matters.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
    persona_probs : np.ndarray of shape (n_personas, n_questions, K)
    target_indices : np.ndarray of shape (n_targets,)
        Indices of target questions.
    
    Returns
    -------
    float
        Mean CRPS uncertainty over target question predictions.
    """
    total_crps = 0.0
    n_targets = len(target_indices)
    
    for t_idx in range(n_targets):
        q_idx = target_indices[t_idx]
        q_probs = np.ascontiguousarray(persona_probs[:, q_idx, :])
        predictive = np.dot(posterior_weights, q_probs)
        total = predictive.sum()
        if total > 1e-10:
            predictive = predictive / total
        
        # CRPS uncertainty: sum_k F(k) * (1 - F(k))
        K = len(predictive)
        cdf = 0.0
        uncertainty = 0.0
        for k in range(K):
            cdf += predictive[k]
            uncertainty += cdf * (1.0 - cdf)
        
        total_crps += uncertainty
    
    return total_crps / max(n_targets, 1)


# =============================================================================
# Empirical Bayes: Learning Prior Over Personas
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_user_log_likelihoods_jit(
    user_responses: np.ndarray,
    persona_probs: np.ndarray,
    question_mask: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled computation of log-likelihoods for all user-persona pairs.
    
    Parameters
    ----------
    user_responses : np.ndarray of shape (n_users, n_questions)
        User responses as integer indices. -1 indicates missing.
    persona_probs : np.ndarray of shape (n_personas, n_questions, K)
        Pre-computed persona probabilities.
    question_mask : np.ndarray of shape (n_questions,)
        Boolean mask for which questions to consider (feasible questions only).
    
    Returns
    -------
    log_likelihoods : np.ndarray of shape (n_users, n_personas)
        Log p(y^{(u)} | theta) for each user-persona pair.
    """
    n_users = user_responses.shape[0]
    n_personas = persona_probs.shape[0]
    n_questions = user_responses.shape[1]
    
    log_likelihoods = np.zeros((n_users, n_personas))
    
    for u in range(n_users):
        for p in range(n_personas):
            log_lik = 0.0
            for q in range(n_questions):
                if not question_mask[q]:
                    continue
                response = user_responses[u, q]
                if response >= 0:  # Not missing
                    prob = persona_probs[p, q, response]
                    log_lik += np.log(max(prob, 1e-10))
            log_likelihoods[u, p] = log_lik
    
    return log_likelihoods


@jit(nopython=True, cache=True)
def _em_step_jit(
    log_likelihoods: np.ndarray,
    log_prior: np.ndarray,
) -> tuple:
    """
    JIT-compiled E-step and M-step for empirical Bayes.
    
    Parameters
    ----------
    log_likelihoods : np.ndarray of shape (n_users, n_personas)
        Log p(y^{(u)} | theta) for each user-persona pair.
    log_prior : np.ndarray of shape (n_personas,)
        Current log prior log p(theta).
    
    Returns
    -------
    new_log_prior : np.ndarray of shape (n_personas,)
        Updated log prior.
    marginal_log_likelihood : float
        Sum of log marginal likelihoods across users.
    """
    n_users = log_likelihoods.shape[0]
    n_personas = log_likelihoods.shape[1]
    
    # Compute responsibilities (gamma) and marginal log-likelihood
    # gamma[u, theta] = p(theta) * p(y^u | theta) / sum_theta' p(theta') * p(y^u | theta')
    
    # First compute log p(theta) + log p(y^u | theta) for all u, theta
    log_joint = np.zeros((n_users, n_personas))
    for u in range(n_users):
        for p in range(n_personas):
            log_joint[u, p] = log_prior[p] + log_likelihoods[u, p]
    
    # Normalize to get responsibilities (in log space first, then exp)
    responsibilities = np.zeros((n_users, n_personas))
    marginal_log_likelihood = 0.0
    
    for u in range(n_users):
        # Log-sum-exp trick for this user
        max_log = np.max(log_joint[u, :])
        log_sum = max_log + np.log(np.sum(np.exp(log_joint[u, :] - max_log)))
        
        # Responsibilities
        for p in range(n_personas):
            responsibilities[u, p] = np.exp(log_joint[u, p] - log_sum)
        
        # Accumulate marginal log-likelihood
        marginal_log_likelihood += log_sum
    
    # M-step: update prior
    # p(theta) = (1/N) * sum_u gamma[u, theta]
    new_prior = np.zeros(n_personas)
    for p in range(n_personas):
        total = 0.0
        for u in range(n_users):
            total += responsibilities[u, p]
        new_prior[p] = total / n_users
    
    # Ensure valid probability distribution
    prior_sum = np.sum(new_prior)
    if prior_sum > 1e-10:
        new_prior = new_prior / prior_sum
    else:
        new_prior = np.ones(n_personas) / n_personas
    
    # Convert to log prior
    new_log_prior = np.zeros(n_personas)
    for p in range(n_personas):
        new_log_prior[p] = np.log(max(new_prior[p], 1e-10))
    
    return new_log_prior, marginal_log_likelihood


def learn_empirical_prior(
    train_user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> np.ndarray:
    """
    Learn an empirical prior over personas by maximizing marginal likelihood.
    
    Implements the EM algorithm for the objective:
        max_{p(θ) ∈ Δ^{n-1}} Σ_{u=1}^N log(Σ_θ p(θ) p(y^{(u)} | θ))
    
    E-step: Compute responsibilities
        γ_{u,θ} = p(θ) p(y^{(u)} | θ) / Σ_{θ'} p(θ') p(y^{(u)} | θ')
    
    M-step: Update prior
        p(θ) = (1/N) Σ_u γ_{u,θ}
    
    Parameters
    ----------
    train_user_responses : pd.DataFrame
        Training user responses. Rows are users, columns are questions.
        Each entry is np.int64 (response index, 0-indexed). -1 = missing.
    persona_responses : pd.DataFrame
        Persona response distributions. Rows are personas, columns are questions.
        Each entry is a list of probabilities.
    feasible_questions : List[str]
        List of feasible questions to use for learning (not target questions).
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for marginal log-likelihood improvement.
    verbose : bool, default=False
        Whether to print progress.
    
    Returns
    -------
    prior_weights : np.ndarray of shape (n_personas,)
        Learned prior distribution over personas.
    
    Notes
    -----
    - Only uses non-missing responses from each user.
    - Uses JIT-compiled functions for efficiency.
    - Converges when marginal log-likelihood improvement < tol.
    
    Example
    -------
    >>> prior = learn_empirical_prior(
    ...     train_users, persona_responses, feasible_questions,
    ...     max_iter=50, tol=1e-4, verbose=True
    ... )
    >>> # Use learned prior for persona-based methods
    >>> posterior = compute_posterior_over_personas(prior, persona_df, ...)
    """
    n_personas = len(persona_responses)
    n_users = len(train_user_responses)
    
    if n_users == 0:
        if verbose:
            print("  No training users, returning uniform prior")
        return np.ones(n_personas) / n_personas
    
    # Build precomputed data
    precomputed = PrecomputedPersonaData(persona_responses, feasible_questions)
    
    # Convert user responses to numpy array
    user_responses_array = np.zeros((n_users, precomputed.n_questions), dtype=np.int64)
    for u_idx, (_, user_row) in enumerate(train_user_responses.iterrows()):
        for q_idx, q in enumerate(precomputed.questions):
            if q in user_row.index:
                user_responses_array[u_idx, q_idx] = int(user_row[q])
            else:
                user_responses_array[u_idx, q_idx] = -1
    
    # Question mask (all feasible questions)
    question_mask = np.ones(precomputed.n_questions, dtype=np.bool_)
    
    # Pre-compute log-likelihoods (this is the expensive part, do it once)
    if verbose:
        print("  Computing user-persona log-likelihoods...")
    
    log_likelihoods = _compute_user_log_likelihoods_jit(
        user_responses_array,
        precomputed.persona_probs,
        question_mask
    )
    
    # Initialize with uniform prior
    log_prior = np.log(np.ones(n_personas) / n_personas)
    prev_marginal_ll = -np.inf
    
    if verbose:
        print(f"  Running EM (max_iter={max_iter}, tol={tol})...")
    
    # Use tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(range(max_iter), desc="  EM iterations", disable=not verbose)
    except ImportError:
        iterator = range(max_iter)
    
    for iteration in iterator:
        # Combined E-step and M-step
        log_prior, marginal_ll = _em_step_jit(log_likelihoods, log_prior)
        
        # Check convergence
        improvement = marginal_ll - prev_marginal_ll
        
        if verbose and not hasattr(iterator, 'set_postfix'):
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: marginal LL = {marginal_ll:.4f}, "
                      f"improvement = {improvement:.6f}")
        elif hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({'LL': f'{marginal_ll:.2f}', 'Δ': f'{improvement:.2e}'})
        
        if improvement < tol and iteration > 0:
            if verbose:
                print(f"  Converged at iteration {iteration} "
                      f"(improvement {improvement:.2e} < tol {tol})")
            break
        
        prev_marginal_ll = marginal_ll
    
    # Convert log prior to prior weights
    prior_weights = np.exp(log_prior)
    prior_weights = prior_weights / prior_weights.sum()
    
    if verbose:
        # Report prior concentration
        effective_n = 1.0 / np.sum(prior_weights ** 2)
        max_weight = np.max(prior_weights)
        print(f"  Learned prior: effective N = {effective_n:.1f}, "
              f"max weight = {max_weight:.4f}")
    
    return prior_weights


# =============================================================================
# Posterior Sparsification
# =============================================================================

def sparsify_posterior_top_k(
    posterior: np.ndarray,
    top_k: int,
    min_k: int = 1,
) -> np.ndarray:
    """
    Keep only the top-K personas by probability, zero out the rest, renormalize.
    
    Parameters
    ----------
    posterior : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    top_k : int
        Number of top personas to keep.
    min_k : int, default 1
        Minimum number of personas to keep (safety floor).
        
    Returns
    -------
    sparse_posterior : np.ndarray of shape (n_personas,)
        Sparse posterior with only top-K non-zero entries.
    """
    k = max(min_k, min(top_k, len(posterior)))
    
    if k >= len(posterior):
        return posterior.copy()
    
    sparse = np.zeros_like(posterior)
    top_indices = np.argpartition(posterior, -k)[-k:]
    sparse[top_indices] = posterior[top_indices]
    
    # Renormalize
    total = sparse.sum()
    if total > 0:
        sparse = sparse / total
    else:
        # Fallback to uniform over top-K if all zeros
        sparse[top_indices] = 1.0 / k
    
    return sparse


def sparsify_posterior_top_p(
    posterior: np.ndarray,
    top_p: float = 0.99,
    min_k: int = 10,
) -> np.ndarray:
    """
    Keep the smallest set of personas whose cumulative probability exceeds p.
    
    This is similar to nucleus/top-p sampling in language models. It adaptively
    keeps more personas when the posterior is flat (uncertain) and fewer when
    the posterior is concentrated (confident).
    
    Parameters
    ----------
    posterior : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    top_p : float, default 0.99
        Cumulative probability threshold. Keep personas until this fraction
        of total probability mass is covered.
    min_k : int, default 10
        Minimum number of personas to keep regardless of cumulative probability.
        
    Returns
    -------
    sparse_posterior : np.ndarray of shape (n_personas,)
        Sparse posterior with reduced non-zero entries.
    """
    n = len(posterior)
    
    if min_k >= n:
        return posterior.copy()
    
    # Sort indices by probability (descending)
    sorted_indices = np.argsort(posterior)[::-1]
    sorted_probs = posterior[sorted_indices]
    
    # Compute cumulative sum
    cumsum = np.cumsum(sorted_probs)
    
    # Find cutoff: smallest k where cumsum >= p
    # searchsorted finds first index where cumsum >= top_p
    cutoff_idx = np.searchsorted(cumsum, top_p, side='left')
    k = max(min_k, cutoff_idx + 1)  # +1 because we want to include the threshold element
    k = min(k, n)  # Don't exceed total personas
    
    # Create sparse posterior
    sparse = np.zeros_like(posterior)
    top_indices = sorted_indices[:k]
    sparse[top_indices] = posterior[top_indices]
    
    # Renormalize
    total = sparse.sum()
    if total > 0:
        sparse = sparse / total
    else:
        # Fallback to uniform over selected personas
        sparse[top_indices] = 1.0 / k
    
    return sparse


def sparsify_posterior(
    posterior: np.ndarray,
    method: str = "top_p",
    top_k: int = 100,
    top_p: float = 0.99,
    min_k: int = 10,
) -> np.ndarray:
    """
    Sparsify a posterior distribution over personas.
    
    This is a unified interface that dispatches to the appropriate method.
    
    Parameters
    ----------
    posterior : np.ndarray of shape (n_personas,)
        Posterior distribution over personas.
    method : str, default "top_p"
        Sparsification method: "top_k" or "top_p".
    top_k : int, default 100
        For method="top_k": number of top personas to keep.
    top_p : float, default 0.99
        For method="top_p": cumulative probability threshold.
    min_k : int, default 10
        Minimum number of personas to keep.
        
    Returns
    -------
    sparse_posterior : np.ndarray of shape (n_personas,)
        Sparsified posterior distribution.
    """
    if method == "top_k":
        return sparsify_posterior_top_k(posterior, top_k=top_k, min_k=min_k)
    elif method == "top_p":
        return sparsify_posterior_top_p(posterior, top_p=top_p, min_k=min_k)
    else:
        raise ValueError(f"Unknown sparsification method: {method}. Use 'top_k' or 'top_p'.")
