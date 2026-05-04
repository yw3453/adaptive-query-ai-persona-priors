"""
Multidimensional Item Response Theory (MIRT) CAT Baselines.

This module implements multidimensional polytomous CAT methods:
- MGRM: Multidimensional Graded Response Model
- MGPCM: Multidimensional Generalized Partial Credit Model

For 1D models (GRM, GPCM), see cat.py.

Key components:
- Multidimensional latent trait estimation
- Grid-based posterior computation over θ ∈ ℝ^D
- MIRT item selection criteria: D-optimality, A-optimality
- Adaptive query loop and evaluation

Computational notes:
- Grid size grows as G^D, so we limit D ≤ 4 for practical computation
- For D > 4, consider using MCMC or variational inference (not implemented)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from scipy.stats import multivariate_normal, norm
from scipy.special import expit
from scipy.optimize import minimize
from tqdm import tqdm
import warnings

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
# Multidimensional Graded Response Model (MGRM)
# =============================================================================

@dataclass
class MGRMParameters:
    """
    Parameters for a Multidimensional Graded Response Model.
    
    For each question x with K response categories and D dimensions:
    - a_x: Discrimination vector of shape (D,) - loadings on each dimension
    - b_x: Array of K-1 ordered threshold parameters
    
    The model defines:
        P(Y_x >= k | θ) = sigmoid(a_x' @ θ - b_{x,k})
    """
    questions: List[str]  # Question identifiers
    discriminations: Dict[str, np.ndarray]  # a_x of shape (D,) for each question
    thresholds: Dict[str, np.ndarray]  # b_{x,1:K-1} for each question
    n_categories: int  # K (number of response categories)
    n_dimensions: int  # D (number of latent dimensions)
    
    def get_params(self, question: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get (a, b) parameters for a question."""
        return self.discriminations[question], self.thresholds[question]


def mgrm_cumulative_prob(
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative probabilities P(Y >= k | θ) for MGRM.
    
    Parameters
    ----------
    theta : np.ndarray of shape (G, D) or (D,)
        Latent trait values.
    a : np.ndarray of shape (D,)
        Discrimination vector.
    b : np.ndarray of shape (K-1,)
        Threshold parameters (ordered).
    
    Returns
    -------
    cum_probs : np.ndarray of shape (G, K+1)
        Cumulative probabilities.
    """
    theta = np.atleast_2d(theta)
    G = theta.shape[0]
    K = len(b) + 1
    
    # Linear combination: a' @ θ for each grid point
    linear = theta @ a  # Shape (G,)
    
    cum_probs = np.zeros((G, K + 1))
    cum_probs[:, 0] = 1.0  # P(Y >= 0) = 1
    cum_probs[:, K] = 0.0  # P(Y >= K) = 0
    
    for k in range(1, K):
        cum_probs[:, k] = expit(linear - b[k - 1])
    
    return cum_probs


def mgrm_category_probs(
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute category probabilities P(Y = k | θ) for MGRM.
    
    Parameters
    ----------
    theta : np.ndarray of shape (G, D) or (D,)
        Latent trait values.
    a : np.ndarray of shape (D,)
        Discrimination vector.
    b : np.ndarray of shape (K-1,)
        Threshold parameters.
    
    Returns
    -------
    probs : np.ndarray of shape (G, K)
        Category probabilities.
    """
    cum_probs = mgrm_cumulative_prob(theta, a, b)
    probs = cum_probs[:, :-1] - cum_probs[:, 1:]
    
    # Ensure non-negative
    probs = np.maximum(probs, EPS)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


# =============================================================================
# Multidimensional GPCM (MGPCM)
# =============================================================================

@dataclass
class MGPCMParameters:
    """
    Parameters for a Multidimensional Generalized Partial Credit Model.
    
    For each question x with K response categories and D dimensions:
    - a_x: Discrimination vector of shape (D,)
    - d_x: Array of K-1 step parameters
    
    The model defines:
        P(Y_x = k | θ) = exp(k * a_x' @ θ - c_{x,k}) / Z(θ)
    
    where c_{x,k} = Σ_{j=1}^k d_{x,j} and Z(θ) is the normalizing constant.
    """
    questions: List[str]
    discriminations: Dict[str, np.ndarray]  # a_x of shape (D,)
    step_parameters: Dict[str, np.ndarray]  # d_{x,1:K-1}
    n_categories: int
    n_dimensions: int
    
    def get_params(self, question: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get (a, d) parameters for a question."""
        return self.discriminations[question], self.step_parameters[question]


def mgpcm_category_probs(
    theta: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Compute category probabilities P(Y = k | θ) for MGPCM.
    
    Parameters
    ----------
    theta : np.ndarray of shape (G, D) or (D,)
        Latent trait values.
    a : np.ndarray of shape (D,)
        Discrimination vector.
    d : np.ndarray of shape (K-1,)
        Step parameters.
    
    Returns
    -------
    probs : np.ndarray of shape (G, K)
        Category probabilities.
    """
    theta = np.atleast_2d(theta)
    G = theta.shape[0]
    K = len(d) + 1
    
    # Linear combination: a' @ θ for each grid point
    linear = theta @ a  # Shape (G,)
    
    # Cumulative step sums
    c = np.zeros(K)
    c[1:] = np.cumsum(d)
    
    # Log-numerators: k * (a' @ θ) - c_k
    log_numerators = np.zeros((G, K))
    for k in range(K):
        log_numerators[:, k] = k * linear - c[k]
    
    # Log-sum-exp for stability
    log_max = log_numerators.max(axis=1, keepdims=True)
    log_numerators_shifted = log_numerators - log_max
    log_Z = log_max.squeeze() + np.log(np.exp(log_numerators_shifted).sum(axis=1))
    
    probs = np.exp(log_numerators - log_Z[:, np.newaxis])
    probs = np.maximum(probs, EPS)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


# =============================================================================
# Multidimensional Grid and Posterior
# =============================================================================

@dataclass
class MIRTState:
    """State for MIRT adaptive querying."""
    grid_points: np.ndarray  # Shape (G^D, D) - all grid points
    posterior_weights: np.ndarray  # Shape (G^D,) - weights at each grid point
    asked_questions: List[str]
    observed_answers: List[int]
    n_dimensions: int
    
    @property
    def posterior_mean(self) -> np.ndarray:
        """E[θ | observations], shape (D,)."""
        return np.sum(
            self.posterior_weights[:, np.newaxis] * self.grid_points, 
            axis=0
        )
    
    @property
    def posterior_cov(self) -> np.ndarray:
        """Cov(θ | observations), shape (D, D)."""
        mean = self.posterior_mean
        centered = self.grid_points - mean
        # Weighted covariance
        cov = np.zeros((self.n_dimensions, self.n_dimensions))
        for i in range(len(self.posterior_weights)):
            cov += self.posterior_weights[i] * np.outer(centered[i], centered[i])
        return cov
    
    @property
    def posterior_variance_trace(self) -> float:
        """Trace of posterior covariance."""
        return np.trace(self.posterior_cov)


def create_multidimensional_grid(
    n_dimensions: int,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
) -> np.ndarray:
    """
    Create a D-dimensional grid of points.
    
    Parameters
    ----------
    n_dimensions : int
        Number of dimensions D.
    grid_range : float
        Range for each dimension: [-grid_range, grid_range].
    n_grid_points_per_dim : int
        Number of grid points per dimension G.
    
    Returns
    -------
    grid_points : np.ndarray of shape (G^D, D)
        All grid points.
    """
    # Create 1D grids
    grids_1d = [
        np.linspace(-grid_range, grid_range, n_grid_points_per_dim)
        for _ in range(n_dimensions)
    ]
    
    # Create mesh grid
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    
    # Flatten and stack
    grid_points = np.stack([m.ravel() for m in mesh], axis=1)
    
    return grid_points


def initialize_mirt_state(
    n_dimensions: int,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
) -> MIRTState:
    """
    Initialize MIRT state with standard normal prior.
    
    Parameters
    ----------
    n_dimensions : int
        Number of dimensions D.
    grid_range : float
        Range for θ grid.
    n_grid_points_per_dim : int
        Grid points per dimension.
    
    Returns
    -------
    state : MIRTState
        Initial MIRT state.
    """
    grid_points = create_multidimensional_grid(
        n_dimensions, grid_range, n_grid_points_per_dim
    )
    
    # Prior: N(0, I)
    prior_density = multivariate_normal.pdf(
        grid_points, 
        mean=np.zeros(n_dimensions),
        cov=np.eye(n_dimensions)
    )
    prior_weights = prior_density / prior_density.sum()
    
    return MIRTState(
        grid_points=grid_points,
        posterior_weights=prior_weights,
        asked_questions=[],
        observed_answers=[],
        n_dimensions=n_dimensions,
    )


def update_mirt_posterior(
    state: MIRTState,
    model_params: Union[MGRMParameters, MGPCMParameters],
    question: str,
    observed_answer: int,
    model_type: str = "mgrm",
) -> MIRTState:
    """
    Update MIRT posterior after observing an answer.
    
    Parameters
    ----------
    state : MIRTState
        Current state.
    model_params : MGRMParameters or MGPCMParameters
        Model parameters.
    question : str
        Question asked.
    observed_answer : int
        Observed answer.
    model_type : str
        "mgrm" or "mgpcm".
    
    Returns
    -------
    new_state : MIRTState
        Updated state.
    """
    a, params = model_params.get_params(question)
    
    if model_type == "mgrm":
        probs = mgrm_category_probs(state.grid_points, a, params)
    else:  # mgpcm
        probs = mgpcm_category_probs(state.grid_points, a, params)
    
    likelihood = probs[:, observed_answer]
    
    new_weights = state.posterior_weights * likelihood
    new_weights = new_weights / new_weights.sum()
    
    return MIRTState(
        grid_points=state.grid_points.copy(),
        posterior_weights=new_weights,
        asked_questions=state.asked_questions + [question],
        observed_answers=state.observed_answers + [observed_answer],
        n_dimensions=state.n_dimensions,
    )


def mirt_posterior_predictive(
    state: MIRTState,
    model_params: Union[MGRMParameters, MGPCMParameters],
    question: str,
    model_type: str = "mgrm",
) -> np.ndarray:
    """
    Compute posterior predictive distribution.
    
    Returns
    -------
    pred_dist : np.ndarray of shape (K,)
        Posterior predictive distribution.
    """
    a, params = model_params.get_params(question)
    
    if model_type == "mgrm":
        probs = mgrm_category_probs(state.grid_points, a, params)
    else:
        probs = mgpcm_category_probs(state.grid_points, a, params)
    
    pred_dist = np.sum(probs * state.posterior_weights[:, np.newaxis], axis=0)
    pred_dist = pred_dist / pred_dist.sum()
    
    return pred_dist


# =============================================================================
# MIRT Item Selection Criteria
# =============================================================================

class MIRTSelectionCriterion(Enum):
    """Item selection criteria for MIRT-CAT."""
    D_OPTIMALITY = "d_opt"  # Maximize det(Fisher Information Matrix)
    A_OPTIMALITY = "a_opt"  # Minimize expected trace(posterior covariance)
    KL_DIVERGENCE = "kl"    # Maximize expected KL divergence


def compute_mgrm_fisher_information(
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute Fisher Information Matrix for MGRM at a given θ.
    
    Parameters
    ----------
    theta : np.ndarray of shape (D,)
        Latent trait value.
    a : np.ndarray of shape (D,)
        Discrimination vector.
    b : np.ndarray of shape (K-1,)
        Thresholds.
    
    Returns
    -------
    I : np.ndarray of shape (D, D)
        Fisher information matrix.
    """
    D = len(a)
    K = len(b) + 1
    
    theta = theta.reshape(1, -1)  # (1, D)
    
    # Get cumulative probabilities
    cum_probs = mgrm_cumulative_prob(theta, a, b)[0]  # (K+1,)
    cat_probs = mgrm_category_probs(theta, a, b)[0]   # (K,)
    
    # Derivatives of cumulative probabilities w.r.t. θ
    # ∂P(Y >= k)/∂θ = P(Y >= k)(1 - P(Y >= k)) * a
    d_cum_probs = np.zeros((K + 1, D))
    for k in range(1, K):
        P_k = cum_probs[k]
        d_cum_probs[k] = P_k * (1 - P_k) * a
    
    # Derivatives of category probabilities
    d_cat_probs = d_cum_probs[:-1] - d_cum_probs[1:]  # (K, D)
    
    # Fisher information: I = Σ_k [∂P(k)/∂θ][∂P(k)/∂θ]' / P(k)
    I = np.zeros((D, D))
    for k in range(K):
        if cat_probs[k] > EPS:
            I += np.outer(d_cat_probs[k], d_cat_probs[k]) / cat_probs[k]
    
    return I


def compute_mgpcm_fisher_information(
    theta: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Compute Fisher Information Matrix for MGPCM at a given θ.
    
    For MGPCM: I(θ) = a a' * Var(Y | θ)
    """
    D = len(a)
    K = len(d) + 1
    
    theta = theta.reshape(1, -1)
    probs = mgpcm_category_probs(theta, a, d)[0]
    
    # E[Y] and Var(Y)
    expected_k = np.sum(np.arange(K) * probs)
    expected_k2 = np.sum(np.arange(K)**2 * probs)
    var_k = expected_k2 - expected_k**2
    
    # I(θ) = a a' * Var(Y)
    I = var_k * np.outer(a, a)
    
    return I


def compute_expected_posterior_trace(
    state: MIRTState,
    model_params: Union[MGRMParameters, MGPCMParameters],
    question: str,
    model_type: str = "mgrm",
) -> float:
    """
    Compute expected trace of posterior covariance after asking a question.
    
    This is the A-optimality criterion for MIRT.
    """
    a, params = model_params.get_params(question)
    K = model_params.n_categories
    
    # Get predictive distribution
    pred_dist = mirt_posterior_predictive(state, model_params, question, model_type)
    
    # Get category probabilities at all grid points
    if model_type == "mgrm":
        probs = mgrm_category_probs(state.grid_points, a, params)
    else:
        probs = mgpcm_category_probs(state.grid_points, a, params)
    
    expected_trace = 0.0
    
    for k in range(K):
        if pred_dist[k] < EPS:
            continue
        
        # Hypothetical posterior after observing Y = k
        likelihood = probs[:, k]
        hyp_weights = state.posterior_weights * likelihood
        hyp_weights = hyp_weights / hyp_weights.sum()
        
        # Hypothetical posterior mean
        hyp_mean = np.sum(hyp_weights[:, np.newaxis] * state.grid_points, axis=0)
        
        # Hypothetical posterior covariance trace
        centered = state.grid_points - hyp_mean
        hyp_var_trace = np.sum(
            hyp_weights[:, np.newaxis] * (centered ** 2)
        )
        
        expected_trace += pred_dist[k] * hyp_var_trace
    
    return expected_trace


def mirt_select_question(
    state: MIRTState,
    model_params: Union[MGRMParameters, MGPCMParameters],
    feasible_questions: List[str],
    asked_questions: List[str],
    criterion: MIRTSelectionCriterion,
    model_type: str = "mgrm",
) -> Tuple[str, float]:
    """
    Select next question using MIRT item selection criterion.
    """
    asked_set = set(asked_questions)
    candidates = [q for q in feasible_questions if q not in asked_set]
    
    if len(candidates) == 0:
        raise ValueError("No candidates available")
    
    candidates = [q for q in candidates if q in model_params.questions]
    
    if len(candidates) == 0:
        raise ValueError("No candidates with fitted parameters")
    
    theta_hat = state.posterior_mean
    
    if criterion == MIRTSelectionCriterion.D_OPTIMALITY:
        # Maximize det(I(θ)) at current estimate
        best_question = None
        best_det = float('-inf')
        
        for q in candidates:
            a, params = model_params.get_params(q)
            
            if model_type == "mgrm":
                I = compute_mgrm_fisher_information(theta_hat, a, params)
            else:
                I = compute_mgpcm_fisher_information(theta_hat, a, params)
            
            det_I = np.linalg.det(I)
            
            if det_I > best_det:
                best_det = det_I
                best_question = q
        
        return best_question, best_det
    
    elif criterion == MIRTSelectionCriterion.A_OPTIMALITY:
        # Minimize expected trace of posterior covariance
        best_question = None
        best_trace = float('inf')
        
        for q in candidates:
            exp_trace = compute_expected_posterior_trace(
                state, model_params, q, model_type
            )
            
            if exp_trace < best_trace:
                best_trace = exp_trace
                best_question = q
        
        return best_question, best_trace
    
    elif criterion == MIRTSelectionCriterion.KL_DIVERGENCE:
        # Maximize expected KL divergence (simplified: use trace reduction)
        best_question = None
        best_kl = float('-inf')
        
        current_trace = state.posterior_variance_trace
        
        for q in candidates:
            exp_trace = compute_expected_posterior_trace(
                state, model_params, q, model_type
            )
            # Use trace reduction as proxy for KL
            kl_proxy = current_trace - exp_trace
            
            if kl_proxy > best_kl:
                best_kl = kl_proxy
                best_question = q
        
        return best_question, best_kl
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


# =============================================================================
# MIRT Parameter Estimation
# =============================================================================

def _initialize_mgrm_params(
    valid_responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize MGRM parameters."""
    K = n_categories
    D = n_dimensions
    N = len(valid_responses)
    
    if N == 0:
        # Default: equal loading on all dimensions
        a = np.ones(D) / np.sqrt(D)
        b = np.linspace(-2, 2, K - 1)
        return a, b
    
    # Initialize discrimination: small positive loadings
    a = np.random.uniform(0.3, 0.7, size=D)
    a = a / np.linalg.norm(a)  # Normalize
    
    # Initialize thresholds from response proportions
    b = np.zeros(K - 1)
    for k in range(K - 1):
        prop = np.mean(valid_responses <= k)
        prop = np.clip(prop, 0.01, 0.99)
        b[k] = norm.ppf(prop)
    b = np.sort(b)
    
    return a, b


def _initialize_mgpcm_params(
    valid_responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize MGPCM parameters."""
    K = n_categories
    D = n_dimensions
    N = len(valid_responses)
    
    if N == 0:
        a = np.ones(D) / np.sqrt(D)
        d = np.zeros(K - 1)
        return a, d
    
    # Initialize discrimination
    a = np.random.uniform(0.3, 0.7, size=D)
    a = a / np.linalg.norm(a)
    
    # Initialize step parameters from category frequencies
    d = np.zeros(K - 1)
    counts = np.array([np.sum(valid_responses == k) for k in range(K)])
    counts = np.maximum(counts, 1)
    
    for k in range(1, K):
        d[k-1] = np.log(counts[k-1] / counts[k])
    
    return a, d


def fit_mgrm_single_question(
    responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int = 15,
    tol: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit MGRM parameters for a single question using marginal MLE with EM.
    
    Parameters
    ----------
    responses : np.ndarray
        Observed responses. NaN for missing.
    n_categories : int
        Number of categories K.
    n_dimensions : int
        Number of dimensions D.
    grid_points : np.ndarray of shape (G^D, D)
        Grid points.
    prior_weights : np.ndarray of shape (G^D,)
        Prior weights.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    a : np.ndarray of shape (D,)
        Discrimination vector.
    b : np.ndarray of shape (K-1,)
        Thresholds.
    """
    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask].astype(np.int64)
    N = len(valid_responses)
    G = len(grid_points)
    K = n_categories
    D = n_dimensions
    
    if N == 0:
        a = np.ones(D) / np.sqrt(D)
        return a, np.linspace(-2, 2, K - 1)
    
    # Initialize
    a, b = _initialize_mgrm_params(valid_responses, K, D)
    log_prior = np.log(prior_weights + EPS)
    
    for iteration in range(max_iter):
        # E-step: Compute posterior over θ
        probs = mgrm_category_probs(grid_points, a, b)  # (G, K)
        
        log_likelihood = np.zeros((N, G))
        for i, y in enumerate(valid_responses):
            log_likelihood[i] = np.log(np.maximum(probs[:, y], EPS))
        
        log_posterior = log_likelihood + log_prior
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        
        # M-step: Optimize parameters
        def neg_expected_ll(params):
            a_new = params[:D]
            b_new = np.sort(params[D:])
            
            probs_new = mgrm_category_probs(grid_points, a_new, b_new)
            
            total_ll = 0.0
            for i, y in enumerate(valid_responses):
                total_ll += np.sum(
                    posterior[i] * np.log(np.maximum(probs_new[:, y], EPS))
                )
            return -total_ll
        
        init_params = np.concatenate([a, b])
        
        try:
            result = minimize(
                neg_expected_ll, init_params,
                method='L-BFGS-B',
                options={'maxiter': 15, 'disp': False}
            )
            a_new = result.x[:D]
            b_new = np.sort(result.x[D:])
        except Exception:
            a_new, b_new = a, b
        
        # Check convergence
        if np.max(np.abs(a_new - a)) < tol and np.max(np.abs(b_new - b)) < tol:
            break
        
        a, b = a_new, b_new
    
    return a, b


def fit_mgpcm_single_question(
    responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int = 15,
    tol: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit MGPCM parameters for a single question.
    """
    valid_mask = ~np.isnan(responses)
    valid_responses = responses[valid_mask].astype(np.int64)
    N = len(valid_responses)
    G = len(grid_points)
    K = n_categories
    D = n_dimensions
    
    if N == 0:
        a = np.ones(D) / np.sqrt(D)
        return a, np.zeros(K - 1)
    
    a, d = _initialize_mgpcm_params(valid_responses, K, D)
    log_prior = np.log(prior_weights + EPS)
    
    for iteration in range(max_iter):
        probs = mgpcm_category_probs(grid_points, a, d)
        
        log_likelihood = np.zeros((N, G))
        for i, y in enumerate(valid_responses):
            log_likelihood[i] = np.log(np.maximum(probs[:, y], EPS))
        
        log_posterior = log_likelihood + log_prior
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum(axis=1, keepdims=True)
        
        def neg_expected_ll(params):
            a_new = params[:D]
            d_new = params[D:]
            
            probs_new = mgpcm_category_probs(grid_points, a_new, d_new)
            
            total_ll = 0.0
            for i, y in enumerate(valid_responses):
                total_ll += np.sum(
                    posterior[i] * np.log(np.maximum(probs_new[:, y], EPS))
                )
            return -total_ll
        
        init_params = np.concatenate([a, d])
        
        try:
            result = minimize(
                neg_expected_ll, init_params,
                method='L-BFGS-B',
                options={'maxiter': 15, 'disp': False}
            )
            a_new = result.x[:D]
            d_new = result.x[D:]
        except Exception:
            a_new, d_new = a, d
        
        if np.max(np.abs(a_new - a)) < tol and np.max(np.abs(d_new - d)) < tol:
            break
        
        a, d = a_new, d_new
    
    return a, d


def _fit_mgrm_wrapper(
    question: str,
    responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Wrapper for parallel MGRM fitting."""
    a, b = fit_mgrm_single_question(
        responses, n_categories, n_dimensions,
        grid_points, prior_weights, max_iter, tol
    )
    return question, a, b


def _fit_mgpcm_wrapper(
    question: str,
    responses: np.ndarray,
    n_categories: int,
    n_dimensions: int,
    grid_points: np.ndarray,
    prior_weights: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Wrapper for parallel MGPCM fitting."""
    a, d = fit_mgpcm_single_question(
        responses, n_categories, n_dimensions,
        grid_points, prior_weights, max_iter, tol
    )
    return question, a, d


def fit_mgrm(
    user_responses: pd.DataFrame,
    questions: List[str],
    n_categories: int,
    n_dimensions: int,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
    max_iter: int = 15,
    tol: float = 1e-2,
    n_jobs: int = -1,
    verbose: bool = False,
) -> MGRMParameters:
    """
    Fit MGRM parameters for multiple questions.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        Training data.
    questions : List[str]
        Questions to fit.
    n_categories : int
        Number of categories K.
    n_dimensions : int
        Number of dimensions D.
    grid_range : float
        Range for θ grid.
    n_grid_points_per_dim : int
        Grid points per dimension.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    n_jobs : int
        Parallel jobs.
    verbose : bool
        Print progress.
    
    Returns
    -------
    params : MGRMParameters
        Fitted parameters.
    """
    if verbose:
        print(f"    Creating {n_dimensions}D grid with {n_grid_points_per_dim} points per dim...")
    
    grid_points = create_multidimensional_grid(
        n_dimensions, grid_range, n_grid_points_per_dim
    )
    
    prior_density = multivariate_normal.pdf(
        grid_points,
        mean=np.zeros(n_dimensions),
        cov=np.eye(n_dimensions)
    )
    prior_weights = prior_density / prior_density.sum()
    
    if verbose:
        print(f"    Grid size: {len(grid_points)} points")
        print(f"    Extracting responses for {len(questions)} questions...")
    
    question_responses = {}
    for question in questions:
        if question in user_responses.columns:
            col = user_responses[question].values
            responses = np.where(col == -1, np.nan, col.astype(float))
            question_responses[question] = responses
        else:
            question_responses[question] = np.array([])
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(questions) > 1:
        if verbose:
            print(f"    Fitting MGRM for {len(questions)} questions in parallel...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_fit_mgrm_wrapper)(
                question,
                question_responses[question],
                n_categories,
                n_dimensions,
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
        discriminations = {}
        thresholds = {}
        
        iterator = tqdm(questions, desc="    Fitting MGRM", disable=not verbose)
        
        for question in iterator:
            responses = question_responses[question]
            a, b = fit_mgrm_single_question(
                responses, n_categories, n_dimensions,
                grid_points, prior_weights, max_iter, tol
            )
            discriminations[question] = a
            thresholds[question] = b
    
    return MGRMParameters(
        questions=questions,
        discriminations=discriminations,
        thresholds=thresholds,
        n_categories=n_categories,
        n_dimensions=n_dimensions,
    )


def fit_mgpcm(
    user_responses: pd.DataFrame,
    questions: List[str],
    n_categories: int,
    n_dimensions: int,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
    max_iter: int = 15,
    tol: float = 1e-2,
    n_jobs: int = -1,
    verbose: bool = False,
) -> MGPCMParameters:
    """
    Fit MGPCM parameters for multiple questions.
    """
    if verbose:
        print(f"    Creating {n_dimensions}D grid with {n_grid_points_per_dim} points per dim...")
    
    grid_points = create_multidimensional_grid(
        n_dimensions, grid_range, n_grid_points_per_dim
    )
    
    prior_density = multivariate_normal.pdf(
        grid_points,
        mean=np.zeros(n_dimensions),
        cov=np.eye(n_dimensions)
    )
    prior_weights = prior_density / prior_density.sum()
    
    if verbose:
        print(f"    Grid size: {len(grid_points)} points")
        print(f"    Extracting responses for {len(questions)} questions...")
    
    question_responses = {}
    for question in questions:
        if question in user_responses.columns:
            col = user_responses[question].values
            responses = np.where(col == -1, np.nan, col.astype(float))
            question_responses[question] = responses
        else:
            question_responses[question] = np.array([])
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(questions) > 1:
        if verbose:
            print(f"    Fitting MGPCM for {len(questions)} questions in parallel...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_fit_mgpcm_wrapper)(
                question,
                question_responses[question],
                n_categories,
                n_dimensions,
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
        
        iterator = tqdm(questions, desc="    Fitting MGPCM", disable=not verbose)
        
        for question in iterator:
            responses = question_responses[question]
            a, d = fit_mgpcm_single_question(
                responses, n_categories, n_dimensions,
                grid_points, prior_weights, max_iter, tol
            )
            discriminations[question] = a
            step_parameters[question] = d
    
    return MGPCMParameters(
        questions=questions,
        discriminations=discriminations,
        step_parameters=step_parameters,
        n_categories=n_categories,
        n_dimensions=n_dimensions,
    )


# =============================================================================
# MIRT Adaptive Query and Evaluation
# =============================================================================

def mirt_adaptive_query(
    user_response_row: pd.Series,
    model_params: Union[MGRMParameters, MGPCMParameters],
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    model_type: str = "mgrm",
    criterion: MIRTSelectionCriterion = MIRTSelectionCriterion.A_OPTIMALITY,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
    verbose: bool = False,
    exclude_targets: bool = True,
) -> Dict[str, Any]:
    """
    Run MIRT adaptive querying for a single user.
    
    Parameters
    ----------
    user_response_row : pd.Series
        User's responses.
    model_params : MGRMParameters or MGPCMParameters
        Model parameters.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    budget : int
        Query budget.
    model_type : str
        "mgrm" or "mgpcm".
    criterion : MIRTSelectionCriterion
        Selection criterion.
    grid_range : float
        Range for θ grid.
    n_grid_points_per_dim : int
        Grid points per dimension.
    verbose : bool
        Print progress.
    exclude_targets : bool
        Exclude targets from querying.
    
    Returns
    -------
    result : Dict[str, Any]
        Query result.
    """
    target_set = set(target_questions)
    user_feasible = [
        q for q in feasible_questions
        if q in user_response_row.index 
        and user_response_row[q] != -1
        and (not exclude_targets or q not in target_set)
        and q in model_params.questions
    ]
    
    n_dimensions = model_params.n_dimensions
    state = initialize_mirt_state(n_dimensions, grid_range, n_grid_points_per_dim)
    trajectory = []
    
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'theta_estimate': state.posterior_mean.tolist(),
        'posterior_trace': state.posterior_variance_trace,
    })
    
    effective_budget = min(budget, len(user_feasible))
    
    for t in range(effective_budget):
        try:
            question, criterion_value = mirt_select_question(
                state=state,
                model_params=model_params,
                feasible_questions=user_feasible,
                asked_questions=state.asked_questions,
                criterion=criterion,
                model_type=model_type,
            )
        except ValueError:
            break
        
        observed_answer = int(user_response_row[question])
        state = update_mirt_posterior(
            state, model_params, question, observed_answer, model_type
        )
        
        trajectory.append({
            'step': t + 1,
            'question_asked': question,
            'answer_observed': observed_answer,
            'criterion_value': criterion_value,
            'theta_estimate': state.posterior_mean.tolist(),
            'posterior_trace': state.posterior_variance_trace,
        })
    
    # Compute predictions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if q in model_params.questions:
                pred_dist = mirt_posterior_predictive(
                    state, model_params, q, model_type
                )
                predicted_distributions[q] = pred_dist
    
    return {
        'asked_questions': state.asked_questions,
        'observed_answers': state.observed_answers,
        'theta_estimate': state.posterior_mean.tolist(),
        'posterior_cov': state.posterior_cov.tolist(),
        'predicted_distributions': predicted_distributions,
        'trajectory': trajectory,
    }


def evaluate_mirt_predictions(
    predicted_distributions: Dict[str, np.ndarray],
    user_response_row: pd.Series,
    target_questions: List[str],
) -> Dict[str, float]:
    """Evaluate MIRT predictions."""
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


def evaluate_mirt_on_users(
    user_responses: pd.DataFrame,
    model_params: Union[MGRMParameters, MGPCMParameters],
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    model_type: str = "mgrm",
    criterion: MIRTSelectionCriterion = MIRTSelectionCriterion.A_OPTIMALITY,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
    user_indices: Optional[List[int]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate MIRT-CAT on multiple users.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    n_dimensions = model_params.n_dimensions
    
    def _evaluate_single_user(user_idx):
        user_row = user_responses.iloc[user_idx]
        
        query_result = mirt_adaptive_query(
            user_response_row=user_row,
            model_params=model_params,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=budget,
            model_type=model_type,
            criterion=criterion,
            grid_range=grid_range,
            n_grid_points_per_dim=n_grid_points_per_dim,
            verbose=False,
        )
        
        metrics = evaluate_mirt_predictions(
            query_result['predicted_distributions'],
            user_row,
            target_questions,
        )
        
        return {
            'user_idx': user_idx,
            'n_questions_asked': len(query_result['asked_questions']),
            'theta_estimate': query_result['theta_estimate'],
            'posterior_trace': query_result['trajectory'][-1]['posterior_trace'],
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


# =============================================================================
# Full MIRT-CAT Pipeline
# =============================================================================

class MIRTModelType(Enum):
    """MIRT model types."""
    MGRM = "mgrm"
    MGPCM = "mgpcm"


def train_and_evaluate_mirt_cat(
    train_user_responses: pd.DataFrame,
    test_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    n_categories: int,
    n_dimensions: int,
    budget: int,
    model_type: MIRTModelType = MIRTModelType.MGRM,
    criterion: MIRTSelectionCriterion = MIRTSelectionCriterion.A_OPTIMALITY,
    grid_range: float = 4.0,
    n_grid_points_per_dim: int = 11,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Complete MIRT-CAT pipeline.
    
    Parameters
    ----------
    train_user_responses : pd.DataFrame
        Training users.
    test_user_responses : pd.DataFrame
        Test users.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    n_categories : int
        Number of categories K.
    n_dimensions : int
        Number of dimensions D.
    budget : int
        Query budget.
    model_type : MIRTModelType
        MGRM or MGPCM.
    criterion : MIRTSelectionCriterion
        Selection criterion.
    grid_range : float
        Range for θ grid.
    n_grid_points_per_dim : int
        Grid points per dimension.
    verbose : bool
        Print progress.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'model_params': Fitted model parameters
        - 'results_df': Evaluation results
        - 'mean_metrics': Mean metrics
        - 'model_type': Model type
        - 'n_dimensions': Number of dimensions
    """
    all_questions = list(set(feasible_questions) | set(target_questions))
    model_type_str = model_type.value
    
    if verbose:
        print(f"Fitting {model_type_str.upper()} with {n_dimensions} dimensions...")
    
    if model_type == MIRTModelType.MGRM:
        model_params = fit_mgrm(
            train_user_responses,
            all_questions,
            n_categories,
            n_dimensions,
            grid_range=grid_range,
            n_grid_points_per_dim=n_grid_points_per_dim,
            verbose=verbose,
        )
    else:  # MGPCM
        model_params = fit_mgpcm(
            train_user_responses,
            all_questions,
            n_categories,
            n_dimensions,
            grid_range=grid_range,
            n_grid_points_per_dim=n_grid_points_per_dim,
            verbose=verbose,
        )
    
    if verbose:
        print(f"Fitted {model_type_str.upper()} for {len(model_params.questions)} questions")
        print("Evaluating on test users...")
    
    results_df = evaluate_mirt_on_users(
        test_user_responses,
        model_params,
        feasible_questions,
        target_questions,
        budget,
        model_type=model_type_str,
        criterion=criterion,
        grid_range=grid_range,
        n_grid_points_per_dim=n_grid_points_per_dim,
        verbose=verbose,
    )
    
    mean_metrics = {
        'accuracy': results_df['accuracy'].mean(),
        'brier_score': results_df['brier_score'].mean(),
        'log_loss': results_df['log_loss'].mean(),
        'kl_divergence': results_df['kl_divergence'].mean(),
        'mean_posterior_trace': results_df['posterior_trace'].mean(),
    }
    
    return {
        'model_params': model_params,
        'results_df': results_df,
        'mean_metrics': mean_metrics,
        'model_type': model_type_str,
        'n_dimensions': n_dimensions,
    }
