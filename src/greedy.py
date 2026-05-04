"""
Greedy Bayesian Adaptive Querying.

This module implements the greedy policy for sequential question selection,
which minimizes the expected posterior cost at each step.

Optimizations for fast implementation:
- Pre-computed NumPy arrays for persona responses
- Numba JIT compilation for core numerical functions
- Batch posterior updates for all K answers
- Parallel evaluation across candidate questions
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Check for Numba availability
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .utils import (
    EPS,
    compute_posterior_predictive,
    update_posterior_with_observation,
    entropy_over_personas,
    entropy_over_target_questions,
    variance_of_categorical,
    variance_over_target_questions,
    crps_uncertainty,
    crps_over_target_questions,
    evaluate_predictions,
    PrecomputedPersonaData,
    precompute_persona_data,
    # JIT-compiled functions (used in Python context)
    compute_posterior_predictive_jit,
    update_posterior_jit,
    crps_over_target_questions_jit,
    crps_uncertainty_jit,
    # Posterior sparsification
    sparsify_posterior,
)


# =============================================================================
# Objective Type Enumeration
# =============================================================================

class ObjectiveType(Enum):
    """Enumeration of supported objective functionals for greedy selection."""
    ENTROPY_PERSONA = "entropy_persona"  # Minimize entropy over persona posterior
    VARIANCE_PERSONA = "variance_persona"  # Minimize variance-like measure over personas
    ENTROPY_TARGET = "entropy_target"  # Minimize entropy over target question predictions
    VARIANCE_TARGET = "variance_target"  # Minimize variance over target question predictions
    CRPS_TARGET = "crps_target"  # Minimize CRPS uncertainty over target question predictions (ordinal-aware)


# PrecomputedPersonaData and precompute_persona_data are imported from utils.py


# =============================================================================
# Numba JIT-Compiled Core Functions
# =============================================================================

@jit(nopython=True, cache=True)
def _batch_update_posteriors_jit(
    posterior_weights: np.ndarray,
    persona_probs_for_question: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled batch posterior update for ALL possible answers.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Current posterior.
    persona_probs_for_question : np.ndarray of shape (n_personas, K)
        P(answer=k | persona=p).
    
    Returns
    -------
    np.ndarray of shape (K, n_personas)
        Updated posteriors for each possible answer k.
        updated_posteriors[k, :] is the posterior if answer k is observed.
    """
    n_personas = len(posterior_weights)
    K = persona_probs_for_question.shape[1]
    
    updated_posteriors = np.zeros((K, n_personas))
    
    for k in range(K):
        likelihoods = persona_probs_for_question[:, k]
        updated = posterior_weights * likelihoods
        total = updated.sum()
        if total > 1e-10:
            updated_posteriors[k, :] = updated / total
        else:
            updated_posteriors[k, :] = 1.0 / n_personas
    
    return updated_posteriors


@jit(nopython=True, cache=True)
def _entropy_over_target_questions_jit(
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
        # Use contiguous array to avoid Numba performance warning
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
def _variance_over_target_questions_jit(
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
def _crps_over_target_questions_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    target_indices: np.ndarray,
) -> float:
    """
    JIT-compiled CRPS uncertainty over target question predictions.
    
    CRPS uncertainty = Σ_k F(k) * (1 - F(k)) where F is CDF.
    This is ordinal-aware, appropriate for ratings data.
    
    Returns mean CRPS uncertainty over target questions.
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


@jit(nopython=True, cache=True)
def _compute_expected_cost_for_candidate_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    candidate_idx: int,
    target_indices: np.ndarray,
    objective_type: int,  # 0=entropy_persona, 1=variance_persona, 2=entropy_target, 3=variance_target, 4=crps_target
) -> float:
    """
    JIT-compiled expected posterior cost computation for a single candidate.
    
    Uses batch posterior updates for efficiency.
    """
    n_personas = len(posterior_weights)
    # Make contiguous to avoid Numba performance warning
    persona_probs_q = np.ascontiguousarray(persona_probs[:, candidate_idx, :])
    K = persona_probs_q.shape[1]
    
    # Compute posterior predictive for this candidate question
    predictive = np.dot(posterior_weights, persona_probs_q)
    total = predictive.sum()
    if total > 1e-10:
        predictive = predictive / total
    
    # Batch update posteriors for all K answers
    updated_posteriors = np.zeros((K, n_personas))
    for k in range(K):
        likelihoods = persona_probs_q[:, k]
        updated = posterior_weights * likelihoods
        update_total = updated.sum()
        if update_total > 1e-10:
            updated_posteriors[k, :] = updated / update_total
        else:
            updated_posteriors[k, :] = 1.0 / n_personas
    
    # Compute expected objective value
    expected_cost = 0.0
    
    for k in range(K):
        if predictive[k] < 1e-10:
            continue
        
        updated_post = updated_posteriors[k, :]
        
        if objective_type == 0:  # entropy_persona
            obj_value = 0.0
            for p in updated_post:
                if p > 1e-10:
                    obj_value -= p * np.log(p)
        elif objective_type == 1:  # variance_persona
            obj_value = 1.0 - np.sum(updated_post * updated_post)
        elif objective_type == 2:  # entropy_target
            obj_value = _entropy_over_target_questions_jit(
                updated_post, persona_probs, target_indices
            )
        elif objective_type == 3:  # variance_target
            obj_value = _variance_over_target_questions_jit(
                updated_post, persona_probs, target_indices
            )
        else:  # crps_target (objective_type == 4)
            obj_value = _crps_over_target_questions_jit(
                updated_post, persona_probs, target_indices
            )
        
        expected_cost += predictive[k] * obj_value
    
    return expected_cost


@jit(nopython=True, parallel=True, cache=True)
def _evaluate_all_candidates_jit(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    candidate_indices: np.ndarray,
    target_indices: np.ndarray,
    objective_type: int,
) -> np.ndarray:
    """
    JIT-compiled parallel evaluation of all candidate questions.
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
    persona_probs : np.ndarray of shape (n_personas, n_questions, K)
    candidate_indices : np.ndarray of shape (n_candidates,)
        Indices of candidate questions to evaluate.
    target_indices : np.ndarray of shape (n_targets,)
    objective_type : int
        0=entropy_persona, 1=variance_persona, 2=entropy_target, 3=variance_target
    
    Returns
    -------
    np.ndarray of shape (n_candidates,)
        Expected cost for each candidate.
    """
    n_candidates = len(candidate_indices)
    costs = np.zeros(n_candidates)
    
    for i in prange(n_candidates):
        candidate_idx = candidate_indices[i]
        costs[i] = _compute_expected_cost_for_candidate_jit(
            posterior_weights, persona_probs, candidate_idx,
            target_indices, objective_type
        )
    
    return costs


# =============================================================================
# Non-JIT Fallback Functions (for compatibility)
# =============================================================================

def _evaluate_all_candidates_fallback(
    posterior_weights: np.ndarray,
    persona_probs: np.ndarray,
    candidate_indices: np.ndarray,
    target_indices: np.ndarray,
    objective_type: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Fallback parallel evaluation using joblib (when Numba not available).
    """
    def evaluate_single(candidate_idx):
        return _compute_expected_cost_for_candidate_jit(
            posterior_weights, persona_probs, candidate_idx,
            target_indices, objective_type
        )
    
    if JOBLIB_AVAILABLE and n_jobs != 1:
        costs = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_single)(idx) for idx in candidate_indices
        )
        return np.array(costs)
    else:
        # Sequential fallback
        costs = np.array([evaluate_single(idx) for idx in candidate_indices])
        return costs


# =============================================================================
# High-Level Greedy Selection (Optimized)
# =============================================================================

def greedy_select_question_optimized(
    posterior_weights: np.ndarray,
    precomputed: PrecomputedPersonaData,
    feasible_indices: np.ndarray,
    asked_indices_set: set,
    objective_type: ObjectiveType,
    target_indices: np.ndarray,
    use_parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[int, float]:
    """
    Optimized greedy question selection using pre-computed data and JIT.
    
    Parameters
    ----------
    posterior_weights : np.ndarray
        Current posterior over personas.
    precomputed : PrecomputedPersonaData
        Pre-computed persona arrays.
    feasible_indices : np.ndarray
        Indices of feasible questions.
    asked_indices_set : set
        Set of already-asked question indices.
    objective_type : ObjectiveType
        Objective functional to minimize.
    target_indices : np.ndarray
        Indices of target questions.
    use_parallel : bool
        Whether to use parallel evaluation.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    
    Returns
    -------
    best_idx : int
        Index of selected question.
    best_cost : float
        Expected cost for selected question.
    """
    # Get candidate indices (feasible but not asked)
    candidate_indices = np.array(
        [idx for idx in feasible_indices if idx not in asked_indices_set],
        dtype=np.int64
    )
    
    if len(candidate_indices) == 0:
        raise ValueError("No feasible questions remain to ask")
    
    # Map objective type to integer for JIT
    obj_type_map = {
        ObjectiveType.ENTROPY_PERSONA: 0,
        ObjectiveType.VARIANCE_PERSONA: 1,
        ObjectiveType.ENTROPY_TARGET: 2,
        ObjectiveType.VARIANCE_TARGET: 3,
        ObjectiveType.CRPS_TARGET: 4,
    }
    obj_type_int = obj_type_map[objective_type]
    
    # Evaluate all candidates
    if NUMBA_AVAILABLE and use_parallel:
        costs = _evaluate_all_candidates_jit(
            posterior_weights,
            precomputed.persona_probs,
            candidate_indices,
            target_indices,
            obj_type_int,
        )
    else:
        costs = _evaluate_all_candidates_fallback(
            posterior_weights,
            precomputed.persona_probs,
            candidate_indices,
            target_indices,
            obj_type_int,
            n_jobs=n_jobs,
        )
    
    # Find best candidate
    best_local_idx = np.argmin(costs)
    best_idx = candidate_indices[best_local_idx]
    best_cost = costs[best_local_idx]
    
    return best_idx, best_cost


# =============================================================================
# Original (Non-Optimized) Functions for Compatibility
# =============================================================================

def compute_expected_posterior_cost(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    candidate_question: str,
    objective_func,
    objective_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute the expected posterior cost for asking a candidate question.
    
    This is the original non-optimized version for compatibility.
    """
    if objective_kwargs is None:
        objective_kwargs = {}
    
    # Compute posterior predictive for the candidate question
    # p(Y_x = k | y_{I_t}) for all k
    predictive_dist = compute_posterior_predictive(
        posterior_weights, persona_responses, candidate_question
    )
    K = len(predictive_dist)
    
    # Compute expected objective value over all possible answers
    expected_cost = 0.0
    
    for k in range(K):
        if predictive_dist[k] < EPS:
            # This answer has negligible probability, skip
            continue
        
        # Compute posterior after hypothetically observing Y_x = k
        updated_posterior = update_posterior_with_observation(
            posterior_weights, persona_responses, candidate_question, k
        )
        
        # Compute objective functional under the updated posterior
        objective_value = objective_func(updated_posterior, **objective_kwargs)
        
        # Weight by the probability of observing this answer
        expected_cost += predictive_dist[k] * objective_value
    
    return expected_cost


# =============================================================================
# Greedy Selection
# =============================================================================

def greedy_select_question(
    posterior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    asked_questions: List[str],
    objective_type: ObjectiveType,
    target_questions: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Select the next question using the greedy policy.
    
    Implements the greedy selection rule:
        x_{t+1} = argmin_{x ∉ I_t} Δ_f(x | y_{I_t})
    
    Parameters
    ----------
    posterior_weights : np.ndarray of shape (n_personas,)
        Current posterior distribution over personas.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    feasible_questions : List[str]
        Set of questions that can be asked (I_feas).
    asked_questions : List[str]
        Questions already asked (I_t).
    objective_type : ObjectiveType
        Which objective functional to use.
    target_questions : List[str], optional
        Target questions for target-based objectives.
        Required if objective_type is ENTROPY_TARGET or VARIANCE_TARGET.
    
    Returns
    -------
    best_question : str
        The question with minimum expected posterior cost.
    best_cost : float
        The expected posterior cost for the selected question.
    
    Raises
    ------
    ValueError
        If target_questions is required but not provided.
        If no feasible questions remain to ask.
    
    Example
    -------
    >>> next_q, cost = greedy_select_question(
    ...     posterior, persona_df,
    ...     feasible_questions=['q1', 'q2', 'q3', 'q4', 'q5'],
    ...     asked_questions=['q1'],
    ...     objective_type=ObjectiveType.ENTROPY_TARGET,
    ...     target_questions=['q10', 'q11']
    ... )
    """
    # Validate inputs
    if objective_type in [ObjectiveType.ENTROPY_TARGET, ObjectiveType.VARIANCE_TARGET, ObjectiveType.CRPS_TARGET]:
        if target_questions is None or len(target_questions) == 0:
            raise ValueError(
                f"target_questions must be provided for objective type {objective_type}"
            )
    
    # Get candidate questions (feasible but not yet asked)
    asked_set = set(asked_questions)
    candidate_questions = [q for q in feasible_questions if q not in asked_set]
    
    if len(candidate_questions) == 0:
        raise ValueError("No feasible questions remain to ask")
    
    # Set up objective function and kwargs based on objective type
    if objective_type == ObjectiveType.ENTROPY_PERSONA:
        objective_func = entropy_over_personas
        objective_kwargs = {}
    elif objective_type == ObjectiveType.VARIANCE_PERSONA:
        # Use Gini impurity of persona posterior as variance proxy
        objective_func = lambda p: variance_of_categorical(p)
        objective_kwargs = {}
    elif objective_type == ObjectiveType.ENTROPY_TARGET:
        objective_func = entropy_over_target_questions
        objective_kwargs = {
            'persona_responses': persona_responses,
            'target_questions': target_questions,
        }
    elif objective_type == ObjectiveType.VARIANCE_TARGET:
        objective_func = variance_over_target_questions
        objective_kwargs = {
            'persona_responses': persona_responses,
            'target_questions': target_questions,
        }
    elif objective_type == ObjectiveType.CRPS_TARGET:
        objective_func = crps_over_target_questions
        objective_kwargs = {
            'persona_responses': persona_responses,
            'target_questions': target_questions,
        }
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
    
    # Compute expected posterior cost for each candidate
    best_question = None
    best_cost = float('inf')
    
    for question in candidate_questions:
        cost = compute_expected_posterior_cost(
            posterior_weights, persona_responses, question,
            objective_func, objective_kwargs
        )
        
        if cost < best_cost:
            best_cost = cost
            best_question = question
    
    return best_question, best_cost


# =============================================================================
# Main Greedy Algorithm (Optimized)
# =============================================================================

def greedy_adaptive_query(
    user_response_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    objective_type: ObjectiveType,
    prior_weights: Optional[np.ndarray] = None,
    precomputed: Optional[PrecomputedPersonaData] = None,
    use_optimized: bool = True,
    use_parallel: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
    exclude_targets: bool = True,
    # Posterior sparsification parameters
    sparsify_enabled: bool = False,
    sparsify_method: str = "top_p",
    sparsify_top_k: int = 100,
    sparsify_top_p: float = 0.99,
    sparsify_min_k: int = 10,
    sparsify_burn_in: int = 0,
) -> Dict[str, Any]:
    """
    Run the greedy Bayesian adaptive querying algorithm for a single user.
    
    Parameters
    ----------
    user_response_row : pd.Series
        A single row from user_responses DataFrame.
        Index should be question identifiers (strings), values are np.int64
        representing the response index. Missing entries are marked by -1.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
        Each entry is a probability distribution list or None.
    feasible_questions : List[str]
        Questions that can be asked (I_feas).
    target_questions : List[str]
        Questions to predict (I*).
    budget : int
        Maximum number of questions to ask (T).
    objective_type : ObjectiveType
        Which objective functional to use for greedy selection.
    prior_weights : np.ndarray, optional
        Prior distribution over personas. Default is uniform.
    precomputed : PrecomputedPersonaData, optional
        Pre-computed persona data for optimization.
        If None and use_optimized=True, will be computed internally.
    use_optimized : bool, default True
        Whether to use optimized JIT-compiled functions.
    use_parallel : bool, default True
        Whether to use parallel candidate evaluation.
    n_jobs : int, default -1
        Number of parallel jobs (-1 for all cores).
    verbose : bool, default False
        Whether to print progress information.
    exclude_targets : bool, default True
        If True, exclude targets from querying. Set to False for overlapping mode.
    sparsify_enabled : bool, default False
        Whether to apply posterior sparsification after each update.
    sparsify_method : str, default "top_p"
        Sparsification method: "top_k" or "top_p".
    sparsify_top_k : int, default 100
        For method="top_k": number of top personas to keep.
    sparsify_top_p : float, default 0.99
        For method="top_p": cumulative probability threshold.
    sparsify_min_k : int, default 10
        Minimum number of personas to keep.
    sparsify_burn_in : int, default 0
        Number of initial steps before sparsification begins.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing:
        - 'asked_questions': List[str] - Questions that were asked in order
        - 'observed_answers': List[int] - Answers observed (0-indexed)
        - 'posterior_weights': np.ndarray - Final posterior over personas
        - 'predicted_distributions': Dict[str, np.ndarray] - Predictions for target questions
        - 'trajectory': List[Dict] - Full trajectory information for each step
    """
    n_personas = len(persona_responses)
    
    # Initialize prior
    if prior_weights is None:
        prior_weights = np.ones(n_personas) / n_personas
    
    # Get user-specific feasible questions (those with observed responses)
    user_feasible = [
        q for q in feasible_questions
        if q in user_response_row.index and user_response_row[q] != -1
    ]
    
    # Exclude target questions from feasible set (unless in overlapping mode)
    if exclude_targets:
        target_set = set(target_questions)
        user_feasible = [q for q in user_feasible if q not in target_set]
    
    if verbose:
        print(f"User has {len(user_feasible)} feasible questions")
    
    # Setup for optimized path
    if use_optimized:
        if precomputed is None:
            precomputed = precompute_persona_data(
                persona_responses, feasible_questions, target_questions
            )
        
        # Convert to indices
        feasible_indices = np.array(
            [precomputed.question_to_idx[q] for q in user_feasible 
             if q in precomputed.question_to_idx],
            dtype=np.int64
        )
        target_indices = np.array(
            [precomputed.question_to_idx[q] for q in target_questions
             if q in precomputed.question_to_idx],
            dtype=np.int64
        )
        asked_indices_set = set()
        idx_to_question = {v: k for k, v in precomputed.question_to_idx.items()}
    
    # Initialize state
    posterior_weights = prior_weights.copy()
    asked_questions: List[str] = []
    observed_answers: List[int] = []
    trajectory: List[Dict] = []
    
    # Set up objective function for tracking
    if objective_type in [ObjectiveType.ENTROPY_PERSONA, ObjectiveType.VARIANCE_PERSONA]:
        def compute_current_objective(p):
            if objective_type == ObjectiveType.ENTROPY_PERSONA:
                return entropy_over_personas(p)
            else:
                return variance_of_categorical(p)
    else:
        if use_optimized:
            def compute_current_objective(p):
                if objective_type == ObjectiveType.ENTROPY_TARGET:
                    return _entropy_over_target_questions_jit(
                        p, precomputed.persona_probs, target_indices
                    )
                elif objective_type == ObjectiveType.VARIANCE_TARGET:
                    return _variance_over_target_questions_jit(
                        p, precomputed.persona_probs, target_indices
                    )
                else:  # CRPS_TARGET
                    return _crps_over_target_questions_jit(
                        p, precomputed.persona_probs, target_indices
                    )
        else:
            def compute_current_objective(p):
                if objective_type == ObjectiveType.ENTROPY_TARGET:
                    return entropy_over_target_questions(p, persona_responses, target_questions)
                elif objective_type == ObjectiveType.VARIANCE_TARGET:
                    return variance_over_target_questions(p, persona_responses, target_questions)
                else:  # CRPS_TARGET
                    return crps_over_target_questions(p, persona_responses, target_questions)
    
    # Record initial state
    initial_objective = compute_current_objective(posterior_weights)
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'objective_value': initial_objective,
        'posterior_entropy': entropy_over_personas(posterior_weights),
    })
    
    if verbose:
        print(f"Step 0: Initial objective = {initial_objective:.4f}")
    
    # Main loop
    effective_budget = min(budget, len(user_feasible))
    
    for t in range(effective_budget):
        # Select next question
        if use_optimized:
            next_idx, expected_cost = greedy_select_question_optimized(
                posterior_weights=posterior_weights,
                precomputed=precomputed,
                feasible_indices=feasible_indices,
                asked_indices_set=asked_indices_set,
                objective_type=objective_type,
                target_indices=target_indices,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
            )
            next_question = idx_to_question[next_idx]
            asked_indices_set.add(next_idx)
        else:
            next_question, expected_cost = greedy_select_question(
                posterior_weights=posterior_weights,
                persona_responses=persona_responses,
                feasible_questions=user_feasible,
                asked_questions=asked_questions,
                objective_type=objective_type,
                target_questions=target_questions,
            )
        
        # Get user's answer
        observed_answer = int(user_response_row[next_question])
        
        # Update posterior
        if use_optimized:
            q_idx = precomputed.question_to_idx[next_question]
            posterior_weights = update_posterior_jit(
                posterior_weights,
                precomputed.persona_probs[:, q_idx, :],
                observed_answer
            )
        else:
            posterior_weights = update_posterior_with_observation(
                posterior_weights, persona_responses, next_question, observed_answer
            )
        
        # Apply posterior sparsification (if enabled and past burn-in period)
        if sparsify_enabled and (t + 1) >= sparsify_burn_in:
            posterior_weights = sparsify_posterior(
                posterior_weights,
                method=sparsify_method,
                top_k=sparsify_top_k,
                top_p=sparsify_top_p,
                min_k=sparsify_min_k,
            )
        
        # Record
        asked_questions.append(next_question)
        observed_answers.append(observed_answer)
        
        current_objective = compute_current_objective(posterior_weights)
        trajectory.append({
            'step': t + 1,
            'question_asked': next_question,
            'answer_observed': observed_answer,
            'expected_cost_before': expected_cost,
            'objective_value': current_objective,
            'posterior_entropy': entropy_over_personas(posterior_weights),
        })
        
        if verbose:
            print(f"Step {t+1}: Asked '{next_question}', Answer={observed_answer}, "
                  f"Objective={current_objective:.4f}")
    
    # Compute predictions for target questions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if use_optimized and q in precomputed.question_to_idx:
                q_idx = precomputed.question_to_idx[q]
                pred_dist = compute_posterior_predictive_jit(
                    posterior_weights, precomputed.persona_probs[:, q_idx, :]
                )
            else:
                pred_dist = compute_posterior_predictive(
                    posterior_weights, persona_responses, q
                )
            predicted_distributions[q] = pred_dist
    
    return {
        'asked_questions': asked_questions,
        'observed_answers': observed_answers,
        'posterior_weights': posterior_weights,
        'predicted_distributions': predicted_distributions,
        'trajectory': trajectory,
    }


# =============================================================================
# Batch Evaluation with User Parallelization
# =============================================================================

def _evaluate_single_user(
    user_idx: int,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    objective_type: ObjectiveType,
    prior_weights: Optional[np.ndarray],
    precomputed: Optional[PrecomputedPersonaData],
    use_optimized: bool,
    # Posterior sparsification parameters
    sparsify_enabled: bool = False,
    sparsify_method: str = "top_p",
    sparsify_top_k: int = 100,
    sparsify_top_p: float = 0.99,
    sparsify_min_k: int = 10,
    sparsify_burn_in: int = 0,
) -> Dict[str, Any]:
    """Helper function for parallel user evaluation."""
    query_result = greedy_adaptive_query(
        user_response_row=user_row,
        persona_responses=persona_responses,
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        budget=budget,
        objective_type=objective_type,
        prior_weights=prior_weights,
        precomputed=precomputed,
        use_optimized=use_optimized,
        use_parallel=False,  # Don't parallelize candidates within parallel users
        verbose=False,
        sparsify_enabled=sparsify_enabled,
        sparsify_method=sparsify_method,
        sparsify_top_k=sparsify_top_k,
        sparsify_top_p=sparsify_top_p,
        sparsify_min_k=sparsify_min_k,
        sparsify_burn_in=sparsify_burn_in,
    )
    
    metrics = evaluate_predictions(
        query_result['predicted_distributions'],
        user_row,
        target_questions,
    )
    
    return {
        'user_idx': user_idx,
        'n_questions_asked': len(query_result['asked_questions']),
        'final_posterior_entropy': query_result['trajectory'][-1]['posterior_entropy'],
        'query_result': query_result,
        **metrics,
    }


def evaluate_greedy_on_users(
    user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    objective_type: ObjectiveType,
    prior_weights: Optional[np.ndarray] = None,
    user_indices: Optional[List[int]] = None,
    use_optimized: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
    # Posterior sparsification parameters
    sparsify_enabled: bool = False,
    sparsify_method: str = "top_p",
    sparsify_top_k: int = 100,
    sparsify_top_p: float = 0.99,
    sparsify_min_k: int = 10,
    sparsify_burn_in: int = 0,
) -> pd.DataFrame:
    """
    Evaluate the greedy policy on multiple users with optional parallelization.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        DataFrame with users as rows and questions as columns.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    budget : int
        Maximum number of questions to ask.
    objective_type : ObjectiveType
        Objective functional for greedy selection.
    prior_weights : np.ndarray, optional
        Prior over personas. Default is uniform.
    user_indices : List[int], optional
        Specific user indices to evaluate. Default is all users.
    use_optimized : bool, default True
        Whether to use optimized functions.
    n_jobs : int, default -1
        Number of parallel jobs for user evaluation.
        -1 uses all CPU cores.
        1 disables parallelization.
    verbose : bool, default False
        Whether to print progress.
    sparsify_enabled : bool, default False
        Whether to apply posterior sparsification after each update.
    sparsify_method : str, default "top_p"
        Sparsification method: "top_k" or "top_p".
    sparsify_top_k : int, default 100
        For method="top_k": number of top personas to keep.
    sparsify_top_p : float, default 0.99
        For method="top_p": cumulative probability threshold.
    sparsify_min_k : int, default 10
        Minimum number of personas to keep.
    sparsify_burn_in : int, default 0
        Number of initial steps before sparsification begins.
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per user containing metrics.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    # Pre-compute persona data once
    precomputed = None
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(user_indices) > 1:
        if verbose:
            print(f"Evaluating {len(user_indices)} users in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_evaluate_single_user)(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                budget,
                objective_type,
                prior_weights,
                precomputed,
                use_optimized,
                sparsify_enabled,
                sparsify_method,
                sparsify_top_k,
                sparsify_top_p,
                sparsify_min_k,
                sparsify_burn_in,
            )
            for user_idx in user_indices
        )
    else:
        # Sequential evaluation
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            
            result = _evaluate_single_user(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                budget,
                objective_type,
                prior_weights,
                precomputed,
                use_optimized,
                sparsify_enabled,
                sparsify_method,
                sparsify_top_k,
                sparsify_top_p,
                sparsify_min_k,
                sparsify_burn_in,
            )
            results.append(result)
    
    # Remove query_result from DataFrame (too large)
    for r in results:
        r.pop('query_result', None)
    
    return pd.DataFrame(results)
