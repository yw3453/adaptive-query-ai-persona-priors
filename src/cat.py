"""
Computerized Adaptive Testing (CAT) Baselines - 1D Models.

This module implements 1D polytomous CAT methods:
- Graded Response Model (GRM): Cumulative probability model
- Generalized Partial Credit Model (GPCM): Adjacent category transitions

These serve as classical baselines for comparison with persona-based methods.

Key components:
- GRM and GPCM for polytomous responses
- Grid-based posterior computation over latent trait θ
- Item selection criteria: MFI (Maximum Fisher Information), MEPV (Minimum Expected Posterior Variance)
- Adaptive query loop and evaluation

For multidimensional models (MGRM, MGPCM), see cat_mirt.py.

Optimizations:
- Numba JIT-compiled E-step for fast log-likelihood computation
- Parallel fitting across questions using joblib
- Smart initialization from sample statistics
- Reduced optimization iterations with early stopping
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm
from scipy.special import expit  # Sigmoid function
from scipy.optimize import minimize
from tqdm import tqdm

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Check for Numba availability
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .utils import (
    EPS,
    accuracy_score,
    brier_score,
    log_loss_score,
    kl_divergence,
)


# =============================================================================
# Graded Response Model (GRM)
# =============================================================================

@dataclass
class GRMParameters:
    """
    Parameters for a Graded Response Model.
    
    For each question x with K response categories:
    - a_x: Discrimination parameter (positive)
    - b_x: Array of K-1 ordered threshold parameters
    
    The model defines:
        P(Y_x >= k | θ) = sigmoid(a_x * (θ - b_{x,k}))
    """
    questions: List[str]  # Question identifiers
    discriminations: Dict[str, float]  # a_x for each question
    thresholds: Dict[str, np.ndarray]  # b_{x,1:K-1} for each question
    n_categories: int  # K (number of response categories)
    
    def get_params(self, question: str) -> Tuple[float, np.ndarray]:
        """Get (a, b) parameters for a question."""
        return self.discriminations[question], self.thresholds[question]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return expit(x)


def grm_cumulative_prob(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative probabilities P(Y >= k | θ) for GRM (not CDF).
    
    Parameters
    ----------
    theta : np.ndarray of shape (G,) or scalar
        Latent trait values.
    a : float
        Discrimination parameter.
    b : np.ndarray of shape (K-1,)
        Threshold parameters (ordered).
    
    Returns
    -------
    cum_probs : np.ndarray of shape (G, K+1) or (K+1,)
        Cumulative probabilities. cum_probs[:, k] = P(Y >= k | θ).
        cum_probs[:, 0] = 1, cum_probs[:, K] = 0 by definition.
    """
    theta = np.atleast_1d(theta)
    K = len(b) + 1  # Number of categories
    G = len(theta)
    
    cum_probs = np.zeros((G, K + 1))
    cum_probs[:, 0] = 1.0  # P(Y >= 0) = 1
    cum_probs[:, K] = 0.0  # P(Y >= K) = 0
    
    for k in range(1, K):
        # P(Y >= k | θ) = σ(a * (θ - b_k))
        cum_probs[:, k] = sigmoid(a * (theta - b[k - 1]))
    
    return cum_probs


def grm_category_probs(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute category probabilities P(Y = k | θ) for GRM.
    
    P(Y = k | θ) = P(Y >= k | θ) - P(Y >= k+1 | θ)
    
    Parameters
    ----------
    theta : np.ndarray of shape (G,) or scalar
        Latent trait values.
    a : float
        Discrimination parameter.
    b : np.ndarray of shape (K-1,)
        Threshold parameters.
    
    Returns
    -------
    probs : np.ndarray of shape (G, K) or (K,)
        Category probabilities. probs[:, k] = P(Y = k | θ).
        Categories are 0-indexed: k ∈ {0, 1, ..., K-1}.
    """
    cum_probs = grm_cumulative_prob(theta, a, b)
    # P(Y = k) = P(Y >= k) - P(Y >= k+1)
    probs = cum_probs[:, :-1] - cum_probs[:, 1:]
    
    # Ensure non-negative (numerical stability)
    probs = np.maximum(probs, EPS)
    # Renormalize
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


# =============================================================================
# GRM Parameter Estimation (Optimized)
# =============================================================================

# JIT-compiled helper functions for fast E-step computation
@jit(nopython=True, cache=True)
def _sigmoid_jit(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@jit(nopython=True, cache=True)
def _grm_category_probs_single_theta_jit(
    theta: float,
    a: float,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute category probabilities for a single theta value.
    
    Returns array of shape (K,) where K = len(b) + 1.
    """
    K = len(b) + 1
    
    # Compute cumulative probabilities P(Y >= k)
    cum_probs = np.zeros(K + 1)
    cum_probs[0] = 1.0  # P(Y >= 0) = 1
    cum_probs[K] = 0.0  # P(Y >= K) = 0
    
    for k in range(1, K):
        cum_probs[k] = _sigmoid_jit(np.array([a * (theta - b[k - 1])]))[0]
    
    # Category probabilities: P(Y = k) = P(Y >= k) - P(Y >= k+1)
    probs = np.zeros(K)
    for k in range(K):
        probs[k] = max(cum_probs[k] - cum_probs[k + 1], 1e-10)
    
    # Normalize
    total = probs.sum()
    if total > 0:
        probs = probs / total
    
    return probs


@jit(nopython=True, cache=True)
def _compute_e_step_log_likelihood_jit(
    valid_responses: np.ndarray,
    grid_points: np.ndarray,
    a: float,
    b: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled E-step log-likelihood computation.
    
    Parameters
    ----------
    valid_responses : np.ndarray of shape (N,)
        Valid responses as integers.
    grid_points : np.ndarray of shape (G,)
        Grid points for theta.
    a : float
        Discrimination parameter.
    b : np.ndarray of shape (K-1,)
        Threshold parameters.
    
    Returns
    -------
    log_likelihood : np.ndarray of shape (N, G)
        Log-likelihood for each response and grid point.
    """
    N = len(valid_responses)
    G = len(grid_points)
    log_likelihood = np.zeros((N, G))
    
    for g in range(G):
        theta_g = grid_points[g]
        probs = _grm_category_probs_single_theta_jit(theta_g, a, b)
        
        for i in range(N):
            y = valid_responses[i]
            log_likelihood[i, g] = np.log(max(probs[y], 1e-10))
    
    return log_likelihood


@jit(nopython=True, cache=True)
def _compute_expected_log_likelihood_jit(
    posterior: np.ndarray,
    valid_responses: np.ndarray,
    grid_points: np.ndarray,
    a: float,
    b: np.ndarray,
) -> float:
    """
    JIT-compiled expected log-likelihood for M-step.
    
    Parameters
    ----------
    posterior : np.ndarray of shape (N, G)
        Posterior weights.
    valid_responses : np.ndarray of shape (N,)
        Valid responses.
    grid_points : np.ndarray of shape (G,)
        Grid points.
    a : float
        Discrimination.
    b : np.ndarray of shape (K-1,)
        Thresholds.
    
    Returns
    -------
    float
        Expected log-likelihood.
    """
    N = len(valid_responses)
    G = len(grid_points)
    total_ll = 0.0
    
    for g in range(G):
        theta_g = grid_points[g]
        probs = _grm_category_probs_single_theta_jit(theta_g, a, b)
        
        for i in range(N):
            y = valid_responses[i]
            total_ll += posterior[i, g] * np.log(max(probs[y], 1e-10))
    
    return total_ll


def _initialize_grm_params_from_data(
    valid_responses: np.ndarray,
    n_categories: int,
) -> Tuple[float, np.ndarray]:
    """
    Initialize GRM parameters from sample statistics.
    
    Uses:
    - a (discrimination): Based on response variance (higher variance = higher discrimination)
    - b (thresholds): Based on cumulative response proportions
    
    Parameters
    ----------
    valid_responses : np.ndarray of shape (N,)
        Valid responses (0-indexed).
    n_categories : int
        Number of categories K.
    
    Returns
    -------
    a : float
        Initial discrimination estimate.
    b : np.ndarray of shape (K-1,)
        Initial threshold estimates.
    """
    N = len(valid_responses)
    K = n_categories
    
    if N == 0:
        return 1.0, np.linspace(-2, 2, K - 1)
    
    # Initialize discrimination based on response variance
    # Higher variance suggests higher discrimination
    resp_std = np.std(valid_responses)
    if resp_std > 0:
        # Scale to reasonable range [0.5, 2.5]
        a = 0.5 + 2.0 * min(resp_std / (K - 1), 1.0)
    else:
        a = 1.0
    
    # Initialize thresholds from cumulative response proportions
    # Use inverse normal CDF of cumulative proportions
    b = np.zeros(K - 1)
    cumulative_count = 0
    
    for k in range(K - 1):
        # Count responses <= k
        cumulative_count = np.sum(valid_responses <= k)
        prop = cumulative_count / N
        
        # Clip to avoid infinity
        prop = np.clip(prop, 0.01, 0.99)
        
        # Inverse normal CDF (probit)
        b[k] = norm.ppf(prop)
    
    # Ensure thresholds are ordered
    b = np.sort(b)
    
    return a, b


def fit_grm_single_question(
    responses: np.ndarray,
    n_categories: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> Tuple[float, np.ndarray]:
    """
    Fit GRM parameters for a single question using marginal MLE with EM.
    
    Optimized with:
    - Smart initialization from sample statistics
    - JIT-compiled E-step
    - Reduced iterations with early stopping
    
    Parameters
    ----------
    responses : np.ndarray of shape (N,)
        Observed responses (0-indexed categories). NaN for missing.
    n_categories : int
        Number of response categories K.
    grid_points : np.ndarray of shape (G,)
        Grid points for θ.
    prior_weights : np.ndarray of shape (G,)
        Prior weights on grid points.
    max_iter : int
        Maximum EM iterations (default: 20).
    tol : float
        Convergence tolerance (default: 1e-3).
    
    Returns
    -------
    a : float
        Estimated discrimination.
    b : np.ndarray of shape (K-1,)
        Estimated thresholds.
    """
    # Filter out missing responses
    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask].astype(np.int64)
    N = len(valid_responses)
    G = len(grid_points)
    K = n_categories
    
    if N == 0:
        # No data: return default parameters
        return 1.0, np.linspace(-2, 2, K - 1)
    
    # Smart initialization from sample statistics
    a, b = _initialize_grm_params_from_data(valid_responses, K)
    b = np.ascontiguousarray(b)
    grid_points = np.ascontiguousarray(grid_points)
    prior_weights = np.ascontiguousarray(prior_weights)
    log_prior = np.log(prior_weights + EPS)
    
    for iteration in range(max_iter):
        # E-step: Compute posterior over θ for each person (JIT-compiled)
        if NUMBA_AVAILABLE:
            log_likelihood = _compute_e_step_log_likelihood_jit(
                valid_responses, grid_points, a, b
            )
        else:
            # Fallback to non-JIT version
            log_likelihood = np.zeros((N, G))
            for g in range(G):
                theta_g = grid_points[g]
                probs = grm_category_probs(np.array([theta_g]), a, b)[0]
                for i, y in enumerate(valid_responses):
                    log_likelihood[i, g] = np.log(max(probs[y], EPS))
        
        # Posterior: add log prior and normalize
        log_posterior = log_likelihood + log_prior
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        
        # M-step: Maximize expected complete log-likelihood
        def neg_expected_ll(params):
            a_new = np.exp(params[0])  # Ensure a > 0
            b_new = np.sort(params[1:])  # Ensure ordered thresholds
            
            if NUMBA_AVAILABLE:
                return -_compute_expected_log_likelihood_jit(
                    posterior, valid_responses, grid_points, a_new, 
                    np.ascontiguousarray(b_new)
                )
            else:
                total_ll = 0.0
                for g in range(G):
                    theta_g = grid_points[g]
                    probs = grm_category_probs(np.array([theta_g]), a_new, b_new)[0]
                    for i, y in enumerate(valid_responses):
                        total_ll += posterior[i, g] * np.log(max(probs[y], EPS))
                return -total_ll
        
        # Initial parameters for optimization
        init_params = np.concatenate([[np.log(a)], b])
        
        try:
            result = minimize(
                neg_expected_ll, init_params,
                method='L-BFGS-B',
                options={'maxiter': 20, 'disp': False}
            )
            a_new = np.exp(result.x[0])
            b_new = np.sort(result.x[1:])
        except Exception:
            # If optimization fails, keep current parameters
            a_new, b_new = a, b
        
        # Check convergence
        if abs(a_new - a) < tol and np.max(np.abs(b_new - b)) < tol:
            break
        
        a, b = a_new, np.ascontiguousarray(b_new)
    
    return a, b


def _fit_single_question_wrapper(
    question: str,
    responses: np.ndarray,
    n_categories: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[str, float, np.ndarray]:
    """Wrapper for parallel fitting of a single question."""
    a, b = fit_grm_single_question(
        responses, n_categories, grid_points, prior_weights, max_iter, tol
    )
    return question, a, b


def fit_grm(
    user_responses: pd.DataFrame,
    questions: List[str],
    n_categories: int,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    max_iter: int = 20,
    tol: float = 1e-3,
    n_jobs: int = -1,
    verbose: bool = False,
) -> GRMParameters:
    """
    Fit GRM parameters for multiple questions.
    
    Optimized with parallel processing across questions.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        Training data with users as rows and questions as columns.
        Entries are np.int64 response indices. Missing entries are marked by -1.
    questions : List[str]
        Questions to fit.
    n_categories : int
        Number of response categories K.
    grid_range : float
        Range for θ grid: [-grid_range, grid_range].
    n_grid_points : int
        Number of grid points G.
    max_iter : int
        Maximum EM iterations per question (default: 20).
    tol : float
        Convergence tolerance (default: 1e-3).
    n_jobs : int
        Number of parallel jobs (-1 = all cores, 1 = sequential).
    verbose : bool
        Print progress.
    
    Returns
    -------
    params : GRMParameters
        Fitted GRM parameters.
    
    Example
    -------
    >>> params = fit_grm(train_user_df, feasible_questions, n_categories=10)
    """
    # Set up grid and prior
    grid_points = np.ascontiguousarray(
        np.linspace(-grid_range, grid_range, n_grid_points)
    )
    prior_density = norm.pdf(grid_points)
    prior_weights = np.ascontiguousarray(prior_density / prior_density.sum())
    
    # Pre-extract responses for all questions (vectorized)
    if verbose:
        print(f"    Extracting responses for {len(questions)} questions...")
    
    question_responses = {}
    for question in questions:
        if question in user_responses.columns:
            col = user_responses[question].values
            # Replace -1 with NaN for missing
            responses = np.where(col == -1, np.nan, col.astype(float))
            question_responses[question] = responses
        else:
            question_responses[question] = np.array([])
    
    # Parallel fitting
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(questions) > 1:
        if verbose:
            print(f"    Fitting GRM for {len(questions)} questions in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_fit_single_question_wrapper)(
                question,
                question_responses[question],
                n_categories,
                grid_points,
                prior_weights,
                max_iter,
                tol,
            )
            for question in questions
        )
        
        discriminations = {}
        thresholds = {}
        for question, a, b in results:
            discriminations[question] = a
            thresholds[question] = b
    else:
        # Sequential fitting with progress bar
        discriminations = {}
        thresholds = {}
        
        iterator = tqdm(questions, desc="    Fitting GRM", disable=not verbose)
        
        for question in iterator:
            responses = question_responses[question]
            
            a, b = fit_grm_single_question(
                responses, n_categories, grid_points, prior_weights,
                max_iter, tol
            )
            
            discriminations[question] = a
            thresholds[question] = b
    
    return GRMParameters(
        questions=questions,
        discriminations=discriminations,
        thresholds=thresholds,
        n_categories=n_categories,
    )


# =============================================================================
# Generalized Partial Credit Model (GPCM)
# =============================================================================

@dataclass
class GPCMParameters:
    """
    Parameters for a Generalized Partial Credit Model.
    
    For each question x with K response categories:
    - a_x: Discrimination parameter (positive)
    - d_x: Array of K step parameters (d_0 = 0 by convention)
    
    The model defines:
        P(Y_x = k | θ) = exp(Σ_{j=0}^k a_x(θ - d_{x,j})) / Z(θ)
    
    where Z(θ) is the normalizing constant.
    """
    questions: List[str]  # Question identifiers
    discriminations: Dict[str, float]  # a_x for each question
    step_parameters: Dict[str, np.ndarray]  # d_{x,1:K-1} for each question
    n_categories: int  # K (number of response categories)
    
    def get_params(self, question: str) -> Tuple[float, np.ndarray]:
        """Get (a, d) parameters for a question."""
        return self.discriminations[question], self.step_parameters[question]


def gpcm_category_probs(
    theta: np.ndarray,
    a: float,
    d: np.ndarray,
) -> np.ndarray:
    """
    Compute category probabilities P(Y = k | θ) for GPCM.
    
    P(Y = k | θ) = exp(k*a*θ - Σ_{j=1}^k a*d_j) / Z(θ)
    
    Parameters
    ----------
    theta : np.ndarray of shape (G,) or scalar
        Latent trait values.
    a : float
        Discrimination parameter.
    d : np.ndarray of shape (K-1,)
        Step parameters (d_1, ..., d_{K-1}).
    
    Returns
    -------
    probs : np.ndarray of shape (G, K)
        Category probabilities. probs[:, k] = P(Y = k | θ).
    """
    theta = np.atleast_1d(theta)
    K = len(d) + 1  # Number of categories
    G = len(theta)
    
    # Compute cumulative step sums: c_k = Σ_{j=1}^k d_j
    # c_0 = 0, c_1 = d_1, c_2 = d_1 + d_2, etc.
    c = np.zeros(K)
    c[1:] = np.cumsum(d)
    
    # Log-numerators: k*a*θ - a*c_k for each k
    # Shape: (G, K)
    log_numerators = np.zeros((G, K))
    for k in range(K):
        log_numerators[:, k] = k * a * theta - a * c[k]
    
    # Log-sum-exp for numerical stability
    log_max = log_numerators.max(axis=1, keepdims=True)
    log_numerators_shifted = log_numerators - log_max
    log_Z = log_max.squeeze() + np.log(np.exp(log_numerators_shifted).sum(axis=1))
    
    # Probabilities
    probs = np.exp(log_numerators - log_Z[:, np.newaxis])
    
    # Ensure non-negative and normalized (numerical stability)
    probs = np.maximum(probs, EPS)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


@jit(nopython=True, cache=True)
def _gpcm_category_probs_single_theta_jit(
    theta: float,
    a: float,
    d: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled GPCM category probabilities for a single theta.
    
    Returns array of shape (K,) where K = len(d) + 1.
    """
    K = len(d) + 1
    
    # Compute cumulative step sums
    c = np.zeros(K)
    for k in range(1, K):
        c[k] = c[k-1] + d[k-1]
    
    # Log-numerators
    log_nums = np.zeros(K)
    for k in range(K):
        log_nums[k] = k * a * theta - a * c[k]
    
    # Log-sum-exp
    log_max = log_nums.max()
    log_Z = log_max + np.log(np.sum(np.exp(log_nums - log_max)))
    
    # Probabilities
    probs = np.exp(log_nums - log_Z)
    
    # Ensure positive
    for k in range(K):
        probs[k] = max(probs[k], 1e-10)
    
    # Normalize
    total = probs.sum()
    if total > 0:
        probs = probs / total
    
    return probs


@jit(nopython=True, cache=True)
def _compute_gpcm_e_step_log_likelihood_jit(
    valid_responses: np.ndarray,
    grid_points: np.ndarray,
    a: float,
    d: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled E-step log-likelihood for GPCM.
    """
    N = len(valid_responses)
    G = len(grid_points)
    log_likelihood = np.zeros((N, G))
    
    for g in range(G):
        theta_g = grid_points[g]
        probs = _gpcm_category_probs_single_theta_jit(theta_g, a, d)
        
        for i in range(N):
            y = valid_responses[i]
            log_likelihood[i, g] = np.log(max(probs[y], 1e-10))
    
    return log_likelihood


@jit(nopython=True, cache=True)
def _compute_gpcm_expected_log_likelihood_jit(
    posterior: np.ndarray,
    valid_responses: np.ndarray,
    grid_points: np.ndarray,
    a: float,
    d: np.ndarray,
) -> float:
    """
    JIT-compiled expected log-likelihood for GPCM M-step.
    """
    N = len(valid_responses)
    G = len(grid_points)
    total_ll = 0.0
    
    for g in range(G):
        theta_g = grid_points[g]
        probs = _gpcm_category_probs_single_theta_jit(theta_g, a, d)
        
        for i in range(N):
            y = valid_responses[i]
            total_ll += posterior[i, g] * np.log(max(probs[y], 1e-10))
    
    return total_ll


def _initialize_gpcm_params_from_data(
    valid_responses: np.ndarray,
    n_categories: int,
) -> Tuple[float, np.ndarray]:
    """
    Initialize GPCM parameters from sample statistics.
    """
    N = len(valid_responses)
    K = n_categories
    
    if N == 0:
        return 1.0, np.zeros(K - 1)
    
    # Initialize discrimination based on response variance
    resp_std = np.std(valid_responses)
    if resp_std > 0:
        a = 0.5 + 2.0 * min(resp_std / (K - 1), 1.0)
    else:
        a = 1.0
    
    # Initialize step parameters from category frequencies
    # d_k ≈ log(P(Y=k-1)/P(Y=k)) (simplified)
    d = np.zeros(K - 1)
    counts = np.array([np.sum(valid_responses == k) for k in range(K)])
    counts = np.maximum(counts, 1)  # Avoid log(0)
    
    for k in range(1, K):
        d[k-1] = np.log(counts[k-1] / counts[k]) / max(a, 0.1)
    
    return a, d


def fit_gpcm_single_question(
    responses: np.ndarray,
    n_categories: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> Tuple[float, np.ndarray]:
    """
    Fit GPCM parameters for a single question using marginal MLE with EM.
    
    Parameters
    ----------
    responses : np.ndarray of shape (N,)
        Observed responses (0-indexed categories). NaN for missing.
    n_categories : int
        Number of response categories K.
    grid_points : np.ndarray of shape (G,)
        Grid points for θ.
    prior_weights : np.ndarray of shape (G,)
        Prior weights on grid points.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    a : float
        Estimated discrimination.
    d : np.ndarray of shape (K-1,)
        Estimated step parameters.
    """
    # Filter out missing responses
    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask].astype(np.int64)
    N = len(valid_responses)
    G = len(grid_points)
    K = n_categories
    
    if N == 0:
        return 1.0, np.zeros(K - 1)
    
    # Smart initialization
    a, d = _initialize_gpcm_params_from_data(valid_responses, K)
    d = np.ascontiguousarray(d)
    grid_points = np.ascontiguousarray(grid_points)
    prior_weights = np.ascontiguousarray(prior_weights)
    log_prior = np.log(prior_weights + EPS)
    
    for iteration in range(max_iter):
        # E-step: Compute posterior over θ
        if NUMBA_AVAILABLE:
            log_likelihood = _compute_gpcm_e_step_log_likelihood_jit(
                valid_responses, grid_points, a, d
            )
        else:
            log_likelihood = np.zeros((N, G))
            for g in range(G):
                theta_g = grid_points[g]
                probs = gpcm_category_probs(np.array([theta_g]), a, d)[0]
                for i, y in enumerate(valid_responses):
                    log_likelihood[i, g] = np.log(max(probs[y], EPS))
        
        # Posterior
        log_posterior = log_likelihood + log_prior
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        
        # M-step
        def neg_expected_ll(params):
            a_new = np.exp(params[0])  # Ensure a > 0
            d_new = params[1:]
            
            if NUMBA_AVAILABLE:
                return -_compute_gpcm_expected_log_likelihood_jit(
                    posterior, valid_responses, grid_points, a_new,
                    np.ascontiguousarray(d_new)
                )
            else:
                total_ll = 0.0
                for g in range(G):
                    theta_g = grid_points[g]
                    probs = gpcm_category_probs(np.array([theta_g]), a_new, d_new)[0]
                    for i, y in enumerate(valid_responses):
                        total_ll += posterior[i, g] * np.log(max(probs[y], EPS))
                return -total_ll
        
        init_params = np.concatenate([[np.log(a)], d])
        
        try:
            result = minimize(
                neg_expected_ll, init_params,
                method='L-BFGS-B',
                options={'maxiter': 20, 'disp': False}
            )
            a_new = np.exp(result.x[0])
            d_new = result.x[1:]
        except Exception:
            a_new, d_new = a, d
        
        # Check convergence
        if abs(a_new - a) < tol and np.max(np.abs(d_new - d)) < tol:
            break
        
        a, d = a_new, np.ascontiguousarray(d_new)
    
    return a, d


def _fit_gpcm_single_question_wrapper(
    question: str,
    responses: np.ndarray,
    n_categories: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[str, float, np.ndarray]:
    """Wrapper for parallel GPCM fitting."""
    a, d = fit_gpcm_single_question(
        responses, n_categories, grid_points, prior_weights, max_iter, tol
    )
    return question, a, d


def fit_gpcm(
    user_responses: pd.DataFrame,
    questions: List[str],
    n_categories: int,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    max_iter: int = 20,
    tol: float = 1e-3,
    n_jobs: int = -1,
    verbose: bool = False,
) -> GPCMParameters:
    """
    Fit GPCM parameters for multiple questions.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        Training data with users as rows and questions as columns.
    questions : List[str]
        Questions to fit.
    n_categories : int
        Number of response categories K.
    grid_range : float
        Range for θ grid: [-grid_range, grid_range].
    n_grid_points : int
        Number of grid points G.
    max_iter : int
        Maximum EM iterations per question.
    tol : float
        Convergence tolerance.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    verbose : bool
        Print progress.
    
    Returns
    -------
    params : GPCMParameters
        Fitted GPCM parameters.
    """
    grid_points = np.ascontiguousarray(
        np.linspace(-grid_range, grid_range, n_grid_points)
    )
    prior_density = norm.pdf(grid_points)
    prior_weights = np.ascontiguousarray(prior_density / prior_density.sum())
    
    # Pre-extract responses
    if verbose:
        print(f"    Extracting responses for {len(questions)} questions...")
    
    question_responses = {}
    for question in questions:
        if question in user_responses.columns:
            col = user_responses[question].values
            responses = np.where(col == -1, np.nan, col.astype(float))
            question_responses[question] = responses
        else:
            question_responses[question] = np.array([])
    
    # Parallel fitting
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(questions) > 1:
        if verbose:
            print(f"    Fitting GPCM for {len(questions)} questions in parallel...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_fit_gpcm_single_question_wrapper)(
                question,
                question_responses[question],
                n_categories,
                grid_points,
                prior_weights,
                max_iter,
                tol,
            )
            for question in questions
        )
        
        discriminations = {}
        step_parameters = {}
        for question, a, d in results:
            discriminations[question] = a
            step_parameters[question] = d
    else:
        discriminations = {}
        step_parameters = {}
        
        iterator = tqdm(questions, desc="    Fitting GPCM", disable=not verbose)
        
        for question in iterator:
            responses = question_responses[question]
            a, d = fit_gpcm_single_question(
                responses, n_categories, grid_points, prior_weights,
                max_iter, tol
            )
            discriminations[question] = a
            step_parameters[question] = d
    
    return GPCMParameters(
        questions=questions,
        discriminations=discriminations,
        step_parameters=step_parameters,
        n_categories=n_categories,
    )


def gpcm_posterior_predictive(
    state: 'CATState',
    gpcm_params: GPCMParameters,
    question: str,
) -> np.ndarray:
    """
    Compute posterior predictive distribution for GPCM.
    
    Parameters
    ----------
    state : CATState
        Current CAT state.
    gpcm_params : GPCMParameters
        GPCM parameters.
    question : str
        Question to predict.
    
    Returns
    -------
    pred_dist : np.ndarray of shape (K,)
        Posterior predictive distribution.
    """
    a, d = gpcm_params.get_params(question)
    probs = gpcm_category_probs(state.grid_points, a, d)  # (G, K)
    pred_dist = np.sum(probs * state.posterior_weights[:, np.newaxis], axis=0)
    pred_dist = pred_dist / pred_dist.sum()
    return pred_dist


def update_cat_posterior_gpcm(
    state: 'CATState',
    gpcm_params: GPCMParameters,
    question: str,
    observed_answer: int,
) -> 'CATState':
    """
    Update CAT posterior after observing an answer (GPCM version).
    """
    a, d = gpcm_params.get_params(question)
    probs = gpcm_category_probs(state.grid_points, a, d)
    likelihood = probs[:, observed_answer]
    
    new_weights = state.posterior_weights * likelihood
    new_weights = new_weights / new_weights.sum()
    
    return CATState(
        grid_points=state.grid_points.copy(),
        posterior_weights=new_weights,
        asked_questions=state.asked_questions + [question],
        observed_answers=state.observed_answers + [observed_answer],
    )


def compute_fisher_information_gpcm(
    theta: float,
    a: float,
    d: np.ndarray,
) -> float:
    """
    Compute Fisher information for a GPCM item at a given θ.
    """
    K = len(d) + 1
    theta_arr = np.array([theta])
    
    # Category probabilities
    probs = gpcm_category_probs(theta_arr, a, d)[0]  # Shape (K,)
    
    # For GPCM: I(θ) = a^2 * Var(Y | θ) where Y is the score
    # More precisely: I(θ) = a^2 * Σ_k k^2 P(k) - (Σ_k k P(k))^2
    expected_k = np.sum(np.arange(K) * probs)
    expected_k2 = np.sum(np.arange(K)**2 * probs)
    var_k = expected_k2 - expected_k**2
    
    fisher_info = a**2 * var_k
    
    return fisher_info


def compute_expected_posterior_variance_gpcm(
    state: 'CATState',
    gpcm_params: GPCMParameters,
    question: str,
) -> float:
    """
    Compute expected posterior variance (MEPV) for GPCM.
    """
    a, d = gpcm_params.get_params(question)
    K = len(d) + 1
    
    pred_dist = gpcm_posterior_predictive(state, gpcm_params, question)
    
    expected_var = 0.0
    probs = gpcm_category_probs(state.grid_points, a, d)
    
    for k in range(K):
        if pred_dist[k] < EPS:
            continue
        
        likelihood = probs[:, k]
        hyp_weights = state.posterior_weights * likelihood
        hyp_weights = hyp_weights / hyp_weights.sum()
        
        hyp_mean = np.sum(hyp_weights * state.grid_points)
        hyp_var = np.sum(hyp_weights * (state.grid_points - hyp_mean) ** 2)
        
        expected_var += pred_dist[k] * hyp_var
    
    return expected_var


# =============================================================================
# Grid-based Posterior Computation
# =============================================================================

@dataclass
class CATState:
    """State for CAT adaptive querying."""
    grid_points: np.ndarray  # θ^{(1)}, ..., θ^{(G)}
    posterior_weights: np.ndarray  # w^{(g)}_t
    asked_questions: List[str]
    observed_answers: List[int]
    
    @property
    def posterior_mean(self) -> float:
        """E[θ | observations]."""
        return np.sum(self.posterior_weights * self.grid_points)
    
    @property
    def posterior_variance(self) -> float:
        """Var(θ | observations)."""
        mean = self.posterior_mean
        return np.sum(self.posterior_weights * (self.grid_points - mean) ** 2)
    
    @property
    def posterior_std(self) -> float:
        """Std(θ | observations)."""
        return np.sqrt(self.posterior_variance)


def initialize_cat_state(
    grid_range: float = 4.0,
    n_grid_points: int = 41,
) -> CATState:
    """
    Initialize CAT state with standard normal prior.
    
    Parameters
    ----------
    grid_range : float
        Range for θ grid: [-grid_range, grid_range].
    n_grid_points : int
        Number of grid points G.
    
    Returns
    -------
    state : CATState
        Initial CAT state with prior weights.
    """
    grid_points = np.linspace(-grid_range, grid_range, n_grid_points)
    
    # Prior weights proportional to N(0,1) density
    prior_density = norm.pdf(grid_points)
    prior_weights = prior_density / prior_density.sum()
    
    return CATState(
        grid_points=grid_points,
        posterior_weights=prior_weights,
        asked_questions=[],
        observed_answers=[],
    )


def update_cat_posterior(
    state: CATState,
    grm_params: GRMParameters,
    question: str,
    observed_answer: int,
) -> CATState:
    """
    Update CAT posterior after observing an answer.
    
    w^{(g)}_{t+1} ∝ w^{(g)}_t * P(Y = y | θ^{(g)})
    
    Parameters
    ----------
    state : CATState
        Current CAT state.
    grm_params : GRMParameters
        GRM parameters.
    question : str
        Question that was asked.
    observed_answer : int
        Observed answer (0-indexed).
    
    Returns
    -------
    new_state : CATState
        Updated CAT state.
    """
    a, b = grm_params.get_params(question)
    
    # Compute likelihood at each grid point
    probs = grm_category_probs(state.grid_points, a, b)
    likelihood = probs[:, observed_answer]
    
    # Update posterior
    new_weights = state.posterior_weights * likelihood
    new_weights = new_weights / new_weights.sum()
    
    return CATState(
        grid_points=state.grid_points.copy(),
        posterior_weights=new_weights,
        asked_questions=state.asked_questions + [question],
        observed_answers=state.observed_answers + [observed_answer],
    )


def cat_posterior_predictive(
    state: CATState,
    grm_params: GRMParameters,
    question: str,
) -> np.ndarray:
    """
    Compute posterior predictive distribution for a question.
    
    P(Y_x = k | observations) = Σ_g P(Y_x = k | θ^{(g)}) * w^{(g)}
    
    Parameters
    ----------
    state : CATState
        Current CAT state.
    grm_params : GRMParameters
        GRM parameters.
    question : str
        Question to predict.
    
    Returns
    -------
    pred_dist : np.ndarray of shape (K,)
        Posterior predictive distribution.
    """
    a, b = grm_params.get_params(question)
    
    # P(Y = k | θ^{(g)}) for all g
    probs = grm_category_probs(state.grid_points, a, b)  # (G, K)
    
    # Weighted sum: P(Y = k) = Σ_g w^{(g)} * P(Y = k | θ^{(g)})
    pred_dist = np.sum(probs * state.posterior_weights[:, np.newaxis], axis=0)
    
    # Normalize
    pred_dist = pred_dist / pred_dist.sum()
    
    return pred_dist


# =============================================================================
# Item Selection Criteria
# =============================================================================

class CATSelectionCriterion(Enum):
    """Item selection criteria for CAT."""
    MFI = "mfi"  # Maximum Fisher Information
    MEPV = "mepv"  # Minimum Expected Posterior Variance


def compute_fisher_information(
    theta: float,
    a: float,
    b: np.ndarray,
) -> float:
    """
    Compute Fisher information for a GRM item at a given θ.
    
    I(θ) = Σ_k P(Y=k|θ) * [∂log P(Y=k|θ)/∂θ]^2
    
    Parameters
    ----------
    theta : float
        Latent trait value.
    a : float
        Discrimination parameter.
    b : np.ndarray of shape (K-1,)
        Threshold parameters.
    
    Returns
    -------
    fisher_info : float
        Fisher information at θ.
    """
    K = len(b) + 1
    theta_arr = np.array([theta])
    
    # Get cumulative probabilities
    cum_probs = grm_cumulative_prob(theta_arr, a, b)[0]  # Shape (K+1,)
    
    # Category probabilities
    cat_probs = grm_category_probs(theta_arr, a, b)[0]  # Shape (K,)
    
    # Derivatives of cumulative probabilities
    # d/dθ P(Y >= k | θ) = a * P(Y >= k) * (1 - P(Y >= k))
    d_cum_probs = np.zeros(K + 1)
    for k in range(1, K):
        P_k = cum_probs[k]
        d_cum_probs[k] = a * P_k * (1 - P_k)
    
    # Derivative of category probabilities
    # d/dθ P(Y = k | θ) = d/dθ P(Y >= k) - d/dθ P(Y >= k+1)
    d_cat_probs = d_cum_probs[:-1] - d_cum_probs[1:]
    
    # Fisher information
    # I(θ) = Σ_k P(Y=k|θ) * [d log P(Y=k|θ) / dθ]^2
    #      = Σ_k [d P(Y=k|θ) / dθ]^2 / P(Y=k|θ)
    fisher_info = 0.0
    for k in range(K):
        if cat_probs[k] > EPS:
            fisher_info += (d_cat_probs[k] ** 2) / cat_probs[k]
    
    return fisher_info


def compute_expected_posterior_variance(
    state: CATState,
    grm_params: GRMParameters,
    question: str,
) -> float:
    """
    Compute expected posterior variance (MEPV) after asking a question.
    
    MEPV(x) = Σ_k P(Y_x = k | observations) * Var(θ | observations, Y_x = k)
    
    Parameters
    ----------
    state : CATState
        Current CAT state.
    grm_params : GRMParameters
        GRM parameters.
    question : str
        Candidate question.
    
    Returns
    -------
    expected_var : float
        Expected posterior variance.
    """
    a, b = grm_params.get_params(question)
    K = len(b) + 1
    
    # Get predictive distribution
    pred_dist = cat_posterior_predictive(state, grm_params, question)
    
    # For each possible answer k, compute posterior variance
    expected_var = 0.0
    
    for k in range(K):
        if pred_dist[k] < EPS:
            continue
        
        # Hypothetical posterior after observing Y = k
        probs = grm_category_probs(state.grid_points, a, b)
        likelihood = probs[:, k]
        
        hyp_weights = state.posterior_weights * likelihood
        hyp_weights = hyp_weights / hyp_weights.sum()
        
        # Posterior variance
        hyp_mean = np.sum(hyp_weights * state.grid_points)
        hyp_var = np.sum(hyp_weights * (state.grid_points - hyp_mean) ** 2)
        
        expected_var += pred_dist[k] * hyp_var
    
    return expected_var


def cat_select_question(
    state: CATState,
    grm_params: GRMParameters,
    feasible_questions: List[str],
    asked_questions: List[str],
    criterion: CATSelectionCriterion,
) -> Tuple[str, float]:
    """
    Select next question using CAT item selection criterion.
    
    Parameters
    ----------
    state : CATState
        Current CAT state.
    grm_params : GRMParameters
        GRM parameters.
    feasible_questions : List[str]
        Available questions.
    asked_questions : List[str]
        Already asked questions.
    criterion : CATSelectionCriterion
        Selection criterion (MFI or MEPV).
    
    Returns
    -------
    best_question : str
        Selected question.
    criterion_value : float
        Value of the selection criterion for the selected question.
    """
    asked_set = set(asked_questions)
    candidates = [q for q in feasible_questions if q not in asked_set]
    
    if len(candidates) == 0:
        raise ValueError("No candidates available")
    
    # Filter to questions with fitted parameters
    candidates = [q for q in candidates if q in grm_params.questions]
    
    if len(candidates) == 0:
        raise ValueError("No candidates with fitted GRM parameters")
    
    if criterion == CATSelectionCriterion.MFI:
        # Maximum Fisher Information at current θ estimate
        theta_hat = state.posterior_mean
        
        best_question = None
        best_info = float('-inf')
        
        for q in candidates:
            a, b = grm_params.get_params(q)
            info = compute_fisher_information(theta_hat, a, b)
            
            if info > best_info:
                best_info = info
                best_question = q
        
        return best_question, best_info
    
    elif criterion == CATSelectionCriterion.MEPV:
        # Minimum Expected Posterior Variance
        best_question = None
        best_epv = float('inf')
        
        for q in candidates:
            epv = compute_expected_posterior_variance(state, grm_params, q)
            
            if epv < best_epv:
                best_epv = epv
                best_question = q
        
        return best_question, best_epv
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


# =============================================================================
# CAT Adaptive Query
# =============================================================================

def cat_adaptive_query(
    user_response_row: pd.Series,
    grm_params: GRMParameters,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    verbose: bool = False,
    exclude_targets: bool = True,
) -> Dict[str, Any]:
    """
    Run CAT adaptive querying for a single user.
    
    Parameters
    ----------
    user_response_row : pd.Series
        A single row from user_responses DataFrame.
        Values are np.int64 response indices. Missing entries are marked by -1.
    grm_params : GRMParameters
        Fitted GRM parameters.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    budget : int
        Maximum number of questions to ask.
    criterion : CATSelectionCriterion
        Item selection criterion (MFI or MEPV).
    grid_range : float
        Range for θ grid.
    n_grid_points : int
        Number of grid points.
    verbose : bool
        Print progress.
    exclude_targets : bool, default True
        If True, exclude targets from querying. Set to False for overlapping mode.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'asked_questions': List[str]
        - 'observed_answers': List[int]
        - 'theta_estimate': float - Final θ estimate
        - 'theta_std': float - Final θ standard deviation
        - 'predicted_distributions': Dict[str, np.ndarray]
        - 'trajectory': List[Dict]
    
    Example
    -------
    >>> result = cat_adaptive_query(
    ...     user_df.iloc[0], grm_params,
    ...     feasible_questions, target_questions,
    ...     budget=10, criterion=CATSelectionCriterion.MEPV
    ... )
    """
    # Get user-specific feasible questions
    target_set = set(target_questions)
    user_feasible = [
        q for q in feasible_questions
        if q in user_response_row.index 
        and user_response_row[q] != -1
        and (not exclude_targets or q not in target_set)
        and q in grm_params.questions
    ]
    
    # Initialize CAT state
    state = initialize_cat_state(grid_range, n_grid_points)
    trajectory = []
    
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'theta_estimate': state.posterior_mean,
        'theta_std': state.posterior_std,
    })
    
    if verbose:
        print(f"User has {len(user_feasible)} feasible questions")
        print(f"Initial θ estimate: {state.posterior_mean:.3f} ± {state.posterior_std:.3f}")
    
    # Main loop
    effective_budget = min(budget, len(user_feasible))
    
    for t in range(effective_budget):
        try:
            question, criterion_value = cat_select_question(
                state=state,
                grm_params=grm_params,
                feasible_questions=user_feasible,
                asked_questions=state.asked_questions,
                criterion=criterion,
            )
        except ValueError:
            break
        
        # Get user's answer (already an integer index)
        observed_answer = int(user_response_row[question])
        
        # Update state
        state = update_cat_posterior(state, grm_params, question, observed_answer)
        
        trajectory.append({
            'step': t + 1,
            'question_asked': question,
            'answer_observed': observed_answer,
            'criterion_value': criterion_value,
            'theta_estimate': state.posterior_mean,
            'theta_std': state.posterior_std,
        })
        
        if verbose:
            print(f"Step {t+1}: Asked '{question}', Answer={observed_answer}, "
                  f"θ={state.posterior_mean:.3f} ± {state.posterior_std:.3f}")
    
    # Compute predictions for target questions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if q in grm_params.questions:
                pred_dist = cat_posterior_predictive(state, grm_params, q)
                predicted_distributions[q] = pred_dist
    
    return {
        'asked_questions': state.asked_questions,
        'observed_answers': state.observed_answers,
        'theta_estimate': state.posterior_mean,
        'theta_std': state.posterior_std,
        'predicted_distributions': predicted_distributions,
        'trajectory': trajectory,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_cat_predictions(
    predicted_distributions: Dict[str, np.ndarray],
    user_response_row: pd.Series,
    target_questions: List[str],
) -> Dict[str, float]:
    """
    Evaluate CAT predictions on target questions.
    
    Parameters
    ----------
    predicted_distributions : Dict[str, np.ndarray]
        Predicted distributions for target questions.
    user_response_row : pd.Series
        User's actual responses. Values are np.int64 response indices.
        Missing entries are marked by -1.
    target_questions : List[str]
        Target questions to evaluate.
    
    Returns
    -------
    metrics : Dict[str, float]
        Evaluation metrics.
    """
    accuracies = []
    brier_scores = []
    log_losses = []
    kl_divergences = []
    
    for q in target_questions:
        if q not in predicted_distributions:
            continue
        if q not in user_response_row.index or user_response_row[q] == -1:
            continue
        
        pred_dist = predicted_distributions[q]
        true_response = int(user_response_row[q])
        
        accuracies.append(accuracy_score(pred_dist, true_response))
        brier_scores.append(brier_score(pred_dist, true_response))
        log_losses.append(log_loss_score(pred_dist, true_response))
        kl_divergences.append(kl_divergence(pred_dist, true_response))
    
    n_evaluated = len(accuracies)
    
    if n_evaluated == 0:
        return {
            'accuracy': np.nan,
            'brier_score': np.nan,
            'log_loss': np.nan,
            'kl_divergence': np.nan,
            'n_evaluated': 0,
        }
    
    return {
        'accuracy': np.mean(accuracies),
        'brier_score': np.mean(brier_scores),
        'log_loss': np.mean(log_losses),
        'kl_divergence': np.mean(kl_divergences),
        'n_evaluated': n_evaluated,
    }


def evaluate_cat_on_users(
    user_responses: pd.DataFrame,
    grm_params: GRMParameters,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    user_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate CAT on multiple users.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        DataFrame with users as rows and questions as columns.
    grm_params : GRMParameters
        Fitted GRM parameters.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    budget : int
        Maximum questions to ask.
    criterion : CATSelectionCriterion
        Item selection criterion.
    grid_range : float
        Range for θ grid.
    n_grid_points : int
        Number of grid points.
    user_indices : List[int], optional
        Specific user indices.
    verbose : bool
        Print progress.
    
    Returns
    -------
    results_df : pd.DataFrame
        Evaluation results.
    
    Example
    -------
    >>> # First fit GRM on training data
    >>> grm_params = fit_grm(train_user_df, feasible_questions, n_categories=10)
    >>> # Then evaluate on test data
    >>> results = evaluate_cat_on_users(
    ...     test_user_df, grm_params,
    ...     feasible_questions, target_questions,
    ...     budget=10, criterion=CATSelectionCriterion.MEPV
    ... )
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    def _evaluate_single_user(user_idx):
        user_row = user_responses.iloc[user_idx]
        
        query_result = cat_adaptive_query(
            user_response_row=user_row,
            grm_params=grm_params,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=budget,
            criterion=criterion,
            grid_range=grid_range,
            n_grid_points=n_grid_points,
            verbose=False,
        )
        
        metrics = evaluate_cat_predictions(
            query_result['predicted_distributions'],
            user_row,
            target_questions,
        )
        
        return {
            'user_idx': user_idx,
            'n_questions_asked': len(query_result['asked_questions']),
            'theta_estimate': query_result['theta_estimate'],
            'theta_std': query_result['theta_std'],
            **metrics,
        }
    
    # Use parallel processing if available
    if JOBLIB_AVAILABLE and len(user_indices) > 1:
        if verbose:
            print(f"  Evaluating {len(user_indices)} users in parallel...")
        results = Parallel(n_jobs=-1, verbose=10 if verbose else 0)(
            delayed(_evaluate_single_user)(user_idx)
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            results.append(_evaluate_single_user(user_idx))
    
    return pd.DataFrame(results)


# =============================================================================
# Full CAT Pipeline
# =============================================================================

def train_and_evaluate_cat(
    train_user_responses: pd.DataFrame,
    test_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    n_categories: int,
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Complete CAT pipeline: fit GRM on training data, evaluate on test data.
    
    Parameters
    ----------
    train_user_responses : pd.DataFrame
        Training users for GRM fitting.
    test_user_responses : pd.DataFrame
        Test users for evaluation.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    n_categories : int
        Number of response categories K.
    budget : int
        Query budget.
    criterion : CATSelectionCriterion
        Item selection criterion.
    grid_range : float
        Range for θ grid.
    n_grid_points : int
        Number of grid points.
    verbose : bool
        Print progress.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'grm_params': Fitted GRM parameters
        - 'results_df': Evaluation results DataFrame
        - 'mean_metrics': Mean metrics across users
    
    Example
    -------
    >>> result = train_and_evaluate_cat(
    ...     train_user_df, test_user_df,
    ...     feasible_questions, target_questions,
    ...     n_categories=10, budget=10
    ... )
    >>> print(f"Mean accuracy: {result['mean_metrics']['accuracy']:.2%}")
    """
    # Fit GRM on training data
    if verbose:
        print("Fitting GRM parameters...")
    
    all_questions = list(set(feasible_questions) | set(target_questions))
    grm_params = fit_grm(
        train_user_responses,
        all_questions,
        n_categories,
        grid_range=grid_range,
        n_grid_points=n_grid_points,
        verbose=verbose,
    )
    
    if verbose:
        print(f"Fitted GRM for {len(grm_params.questions)} questions")
    
    # Evaluate on test data
    if verbose:
        print("Evaluating on test users...")
    
    results_df = evaluate_cat_on_users(
        test_user_responses,
        grm_params,
        feasible_questions,
        target_questions,
        budget,
        criterion=criterion,
        grid_range=grid_range,
        n_grid_points=n_grid_points,
        verbose=verbose,
    )
    
    # Compute mean metrics
    mean_metrics = {
        'accuracy': results_df['accuracy'].mean(),
        'brier_score': results_df['brier_score'].mean(),
        'log_loss': results_df['log_loss'].mean(),
        'kl_divergence': results_df['kl_divergence'].mean(),
        'mean_theta_std': results_df['theta_std'].mean(),
    }
    
    return {
        'grm_params': grm_params,
        'results_df': results_df,
        'mean_metrics': mean_metrics,
    }


# =============================================================================
# GPCM Adaptive Query and Evaluation
# =============================================================================

def cat_select_question_gpcm(
    state: CATState,
    gpcm_params: GPCMParameters,
    feasible_questions: List[str],
    asked_questions: List[str],
    criterion: CATSelectionCriterion,
) -> Tuple[str, float]:
    """
    Select next question using CAT criterion (GPCM version).
    """
    asked_set = set(asked_questions)
    candidates = [q for q in feasible_questions if q not in asked_set]
    
    if len(candidates) == 0:
        raise ValueError("No candidates available")
    
    candidates = [q for q in candidates if q in gpcm_params.questions]
    
    if len(candidates) == 0:
        raise ValueError("No candidates with fitted GPCM parameters")
    
    if criterion == CATSelectionCriterion.MFI:
        theta_hat = state.posterior_mean
        
        best_question = None
        best_info = float('-inf')
        
        for q in candidates:
            a, d = gpcm_params.get_params(q)
            info = compute_fisher_information_gpcm(theta_hat, a, d)
            
            if info > best_info:
                best_info = info
                best_question = q
        
        return best_question, best_info
    
    elif criterion == CATSelectionCriterion.MEPV:
        best_question = None
        best_epv = float('inf')
        
        for q in candidates:
            epv = compute_expected_posterior_variance_gpcm(state, gpcm_params, q)
            
            if epv < best_epv:
                best_epv = epv
                best_question = q
        
        return best_question, best_epv
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def cat_adaptive_query_gpcm(
    user_response_row: pd.Series,
    gpcm_params: GPCMParameters,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    verbose: bool = False,
    exclude_targets: bool = True,
) -> Dict[str, Any]:
    """
    Run CAT adaptive querying for a single user (GPCM version).
    """
    target_set = set(target_questions)
    user_feasible = [
        q for q in feasible_questions
        if q in user_response_row.index 
        and user_response_row[q] != -1
        and (not exclude_targets or q not in target_set)
        and q in gpcm_params.questions
    ]
    
    state = initialize_cat_state(grid_range, n_grid_points)
    trajectory = []
    
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'theta_estimate': state.posterior_mean,
        'theta_std': state.posterior_std,
    })
    
    effective_budget = min(budget, len(user_feasible))
    
    for t in range(effective_budget):
        try:
            question, criterion_value = cat_select_question_gpcm(
                state=state,
                gpcm_params=gpcm_params,
                feasible_questions=user_feasible,
                asked_questions=state.asked_questions,
                criterion=criterion,
            )
        except ValueError:
            break
        
        observed_answer = int(user_response_row[question])
        state = update_cat_posterior_gpcm(state, gpcm_params, question, observed_answer)
        
        trajectory.append({
            'step': t + 1,
            'question_asked': question,
            'answer_observed': observed_answer,
            'criterion_value': criterion_value,
            'theta_estimate': state.posterior_mean,
            'theta_std': state.posterior_std,
        })
    
    # Compute predictions for target questions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if q in gpcm_params.questions:
                pred_dist = gpcm_posterior_predictive(state, gpcm_params, q)
                predicted_distributions[q] = pred_dist
    
    return {
        'asked_questions': state.asked_questions,
        'observed_answers': state.observed_answers,
        'theta_estimate': state.posterior_mean,
        'theta_std': state.posterior_std,
        'predicted_distributions': predicted_distributions,
        'trajectory': trajectory,
    }


def evaluate_cat_on_users_gpcm(
    user_responses: pd.DataFrame,
    gpcm_params: GPCMParameters,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    user_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate GPCM-based CAT on multiple users.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    def _evaluate_single_user(user_idx):
        user_row = user_responses.iloc[user_idx]
        
        query_result = cat_adaptive_query_gpcm(
            user_response_row=user_row,
            gpcm_params=gpcm_params,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=budget,
            criterion=criterion,
            grid_range=grid_range,
            n_grid_points=n_grid_points,
            verbose=False,
        )
        
        metrics = evaluate_cat_predictions(
            query_result['predicted_distributions'],
            user_row,
            target_questions,
        )
        
        return {
            'user_idx': user_idx,
            'n_questions_asked': len(query_result['asked_questions']),
            'theta_estimate': query_result['theta_estimate'],
            'theta_std': query_result['theta_std'],
            **metrics,
        }
    
    if JOBLIB_AVAILABLE and len(user_indices) > 1:
        if verbose:
            print(f"  Evaluating {len(user_indices)} users in parallel...")
        results = Parallel(n_jobs=-1, verbose=10 if verbose else 0)(
            delayed(_evaluate_single_user)(user_idx)
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            results.append(_evaluate_single_user(user_idx))
    
    return pd.DataFrame(results)


def train_and_evaluate_cat_gpcm(
    train_user_responses: pd.DataFrame,
    test_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    n_categories: int,
    budget: int,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Complete GPCM-CAT pipeline: fit GPCM on training data, evaluate on test data.
    """
    if verbose:
        print("Fitting GPCM parameters...")
    
    all_questions = list(set(feasible_questions) | set(target_questions))
    gpcm_params = fit_gpcm(
        train_user_responses,
        all_questions,
        n_categories,
        grid_range=grid_range,
        n_grid_points=n_grid_points,
        verbose=verbose,
    )
    
    if verbose:
        print(f"Fitted GPCM for {len(gpcm_params.questions)} questions")
    
    if verbose:
        print("Evaluating on test users...")
    
    results_df = evaluate_cat_on_users_gpcm(
        test_user_responses,
        gpcm_params,
        feasible_questions,
        target_questions,
        budget,
        criterion=criterion,
        grid_range=grid_range,
        n_grid_points=n_grid_points,
        verbose=verbose,
    )
    
    mean_metrics = {
        'accuracy': results_df['accuracy'].mean(),
        'brier_score': results_df['brier_score'].mean(),
        'log_loss': results_df['log_loss'].mean(),
        'kl_divergence': results_df['kl_divergence'].mean(),
        'mean_theta_std': results_df['theta_std'].mean(),
    }
    
    return {
        'gpcm_params': gpcm_params,
        'results_df': results_df,
        'mean_metrics': mean_metrics,
    }


# =============================================================================
# Unified Interface for 1D CAT Models
# =============================================================================

class CATModelType(Enum):
    """1D CAT model types."""
    GRM = "grm"
    GPCM = "gpcm"


def train_and_evaluate_cat_unified(
    train_user_responses: pd.DataFrame,
    test_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    n_categories: int,
    budget: int,
    model_type: CATModelType = CATModelType.GRM,
    criterion: CATSelectionCriterion = CATSelectionCriterion.MEPV,
    grid_range: float = 4.0,
    n_grid_points: int = 41,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Unified interface for 1D CAT models (GRM or GPCM).
    
    Parameters
    ----------
    train_user_responses : pd.DataFrame
        Training users for model fitting.
    test_user_responses : pd.DataFrame
        Test users for evaluation.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    n_categories : int
        Number of response categories K.
    budget : int
        Query budget.
    model_type : CATModelType
        Model type: GRM or GPCM.
    criterion : CATSelectionCriterion
        Item selection criterion.
    grid_range : float
        Range for θ grid.
    n_grid_points : int
        Number of grid points.
    verbose : bool
        Print progress.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'model_params': Fitted model parameters (GRM or GPCM)
        - 'results_df': Evaluation results DataFrame
        - 'mean_metrics': Mean metrics across users
        - 'model_type': Model type used
    """
    if model_type == CATModelType.GRM:
        result = train_and_evaluate_cat(
            train_user_responses=train_user_responses,
            test_user_responses=test_user_responses,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            n_categories=n_categories,
            budget=budget,
            criterion=criterion,
            grid_range=grid_range,
            n_grid_points=n_grid_points,
            verbose=verbose,
        )
        result['model_params'] = result.pop('grm_params')
        result['model_type'] = 'grm'
        
    elif model_type == CATModelType.GPCM:
        result = train_and_evaluate_cat_gpcm(
            train_user_responses=train_user_responses,
            test_user_responses=test_user_responses,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            n_categories=n_categories,
            budget=budget,
            criterion=criterion,
            grid_range=grid_range,
            n_grid_points=n_grid_points,
            verbose=verbose,
        )
        result['model_params'] = result.pop('gpcm_params')
        result['model_type'] = 'gpcm'
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return result
