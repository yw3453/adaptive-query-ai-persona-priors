"""
Baseline policies for Adaptive Query with AI Persona.

This module implements baseline querying strategies for comparison:
- Random (adaptive): Select questions uniformly at random one at a time
- Non-adaptive set: Select a fixed set of questions offline using greedy entropy,
  ask same questions for all users
- Random fixed set: Select a fixed set of questions by uniform random sampling,
  ask same questions for all users (simpler baseline than non-adaptive)
- Full: Use all feasible questions

Optimizations:
- Pre-computed NumPy arrays for persona data
- Numba JIT-compiled core functions
- Parallel user processing with joblib
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .utils import (
    EPS,
    compute_posterior_over_personas,
    compute_posterior_predictive,
    update_posterior_with_observation,
    entropy_over_personas,
    entropy_over_target_questions,
    variance_of_categorical,
    variance_over_target_questions,
    crps_uncertainty,
    crps_over_target_questions,
    evaluate_predictions,
    # Optimized versions
    PrecomputedPersonaData,
    precompute_persona_data,
    compute_posterior_predictive_jit,
    update_posterior_jit,
    compute_posterior_over_personas_jit,
    entropy_jit,
    entropy_over_target_questions_jit,
    variance_over_target_questions_jit,
    crps_over_target_questions_jit,
    NUMBA_AVAILABLE,
)


# =============================================================================
# Random Baseline Policy
# =============================================================================

def random_adaptive_query(
    user_response_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    precomputed: Optional[PrecomputedPersonaData] = None,
    use_optimized: bool = True,
    exclude_targets: bool = True,
) -> Dict[str, Any]:
    """
    Run random (non-adaptive) querying as a baseline.
    
    Selects questions uniformly at random from the feasible set,
    but still updates the posterior based on observed answers for prediction.
    
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
    exclude_targets : bool, default=True
        If True, exclude targets from querying. Set to False for overlapping mode.
    budget : int
        Number of questions to ask (T).
    prior_weights : np.ndarray, optional
        Prior distribution over personas. Default is uniform.
    seed : int, optional
        Random seed for reproducibility.
    precomputed : PrecomputedPersonaData, optional
        Pre-computed persona data for optimization.
    use_optimized : bool, default True
        Whether to use optimized JIT-compiled functions.
    
    Returns
    -------
    result : Dict[str, Any]
        Dictionary containing query results.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_personas = len(persona_responses)
    
    if prior_weights is None:
        prior_weights = np.ones(n_personas) / n_personas
    
    # Get user-specific feasible questions (those with observed responses)
    user_feasible = [
        q for q in feasible_questions
        if q in user_response_row.index and user_response_row[q] != -1
    ]
    if exclude_targets:
        target_set = set(target_questions)
        user_feasible = [q for q in user_feasible if q not in target_set]
    
    # Initialize
    posterior_weights = prior_weights.copy()
    asked_questions: List[str] = []
    observed_answers: List[int] = []
    trajectory: List[Dict] = []
    
    # Initial state
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'posterior_entropy': entropy_jit(posterior_weights) if use_optimized and NUMBA_AVAILABLE else entropy_over_personas(posterior_weights),
    })
    
    # Randomly sample questions
    effective_budget = min(budget, len(user_feasible))
    selected_questions = np.random.choice(
        user_feasible, size=effective_budget, replace=False
    ).tolist()
    
    # Use optimized path if available
    if use_optimized and precomputed is not None and NUMBA_AVAILABLE:
        for t, question in enumerate(selected_questions):
            observed_answer = int(user_response_row[question])
            
            # Update posterior using JIT function
            if question in precomputed.question_to_idx:
                q_idx = precomputed.question_to_idx[question]
                posterior_weights = update_posterior_jit(
                    posterior_weights,
                    precomputed.persona_probs[:, q_idx, :],
                    observed_answer
                )
            else:
                # Fallback to non-optimized
                posterior_weights = update_posterior_with_observation(
                    posterior_weights, persona_responses, question, observed_answer
                )
            
            asked_questions.append(question)
            observed_answers.append(observed_answer)
            
            trajectory.append({
                'step': t + 1,
                'question_asked': question,
                'answer_observed': observed_answer,
                'posterior_entropy': entropy_jit(posterior_weights),
            })
    else:
        # Non-optimized path
        for t, question in enumerate(selected_questions):
            observed_answer = int(user_response_row[question])
            
            posterior_weights = update_posterior_with_observation(
                posterior_weights, persona_responses, question, observed_answer
            )
            
            asked_questions.append(question)
            observed_answers.append(observed_answer)
            
            trajectory.append({
                'step': t + 1,
                'question_asked': question,
                'answer_observed': observed_answer,
                'posterior_entropy': entropy_over_personas(posterior_weights),
            })
    
    # Compute predictions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if use_optimized and precomputed is not None and q in precomputed.question_to_idx and NUMBA_AVAILABLE:
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


def _evaluate_random_single_user(
    user_idx: int,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray],
    precomputed: Optional[PrecomputedPersonaData],
    use_optimized: bool,
) -> Dict[str, Any]:
    """Helper for parallel random user evaluation."""
    query_result = random_adaptive_query(
        user_response_row=user_row,
        persona_responses=persona_responses,
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        budget=budget,
        prior_weights=prior_weights,
        seed=None,
        precomputed=precomputed,
        use_optimized=use_optimized,
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


def evaluate_random_on_users(
    user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray] = None,
    user_indices: Optional[List[int]] = None,
    seed: Optional[int] = None,
    use_optimized: bool = True,
    n_jobs: int = 1,  # Default to 1 for random to preserve reproducibility
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the random policy on multiple users with optional parallelization.
    
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
    prior_weights : np.ndarray, optional
        Prior over personas. Default is uniform.
    user_indices : List[int], optional
        Specific user indices to evaluate. Default is all users.
    seed : int, optional
        Random seed for reproducibility.
    use_optimized : bool, default True
        Whether to use optimized functions.
    n_jobs : int, default 1
        Number of parallel jobs. Default is 1 for reproducibility with random.
    verbose : bool, default False
        Whether to print progress.
    
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per user containing metrics.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    if seed is not None:
        np.random.seed(seed)
    
    # Pre-compute persona data
    precomputed = None
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    # Note: For random, we default to n_jobs=1 to preserve reproducibility
    # since random question selection depends on numpy's random state
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(user_indices) > 1:
        if verbose:
            print(f"Evaluating {len(user_indices)} users in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_evaluate_random_single_user)(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                budget,
                prior_weights,
                precomputed,
                use_optimized,
            )
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            
            result = _evaluate_random_single_user(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                budget,
                prior_weights,
                precomputed,
                use_optimized,
            )
            results.append(result)
    
    # Remove query_result from DataFrame (too large)
    for r in results:
        r.pop('query_result', None)
    
    return pd.DataFrame(results)


# =============================================================================
# Non-Adaptive Set Baseline
# =============================================================================

def select_nonadaptive_question_set(
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray] = None,
    selection_criterion: str = "entropy_persona",
    precomputed: Optional[PrecomputedPersonaData] = None,
    use_optimized: bool = True,
    exclude_targets: bool = True,
) -> List[str]:
    """
    Select a fixed set of questions offline using greedy selection under the prior.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict (excluded from selection if exclude_targets=True).
    budget : int
        Number of questions to select.
    prior_weights : np.ndarray, optional
        Prior over personas. Default is uniform.
    selection_criterion : str, default "entropy_persona"
        Criterion for question selection. Options:
        - "entropy_persona": Minimize entropy over persona posterior
        - "entropy_target": Minimize entropy over target question predictions
        - "variance_persona": Minimize Gini impurity (1 - sum(p^2)) over personas
        - "variance_target": Minimize variance over target question predictions
        - "crps_target": Minimize CRPS uncertainty over target predictions (ordinal-aware)
    precomputed : PrecomputedPersonaData, optional
        Pre-computed persona data.
    use_optimized : bool, default True
        Whether to use optimized functions.
    exclude_targets : bool, default True
        Whether to exclude target questions from selection.
        Set to False for overlapping mode where feasible = target.
    
    Returns
    -------
    selected_questions : List[str]
        Ordered list of selected questions.
    """
    n_personas = len(persona_responses)
    
    if prior_weights is None:
        prior_weights = np.ones(n_personas) / n_personas
    
    # Pre-compute if not provided
    if use_optimized and precomputed is None:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    # Get target indices for JIT functions
    target_indices = None
    if use_optimized and precomputed is not None and NUMBA_AVAILABLE:
        target_indices = np.array(
            [precomputed.question_to_idx[q] for q in target_questions
             if q in precomputed.question_to_idx],
            dtype=np.int64
        )
    
    # Determine candidate pool
    if exclude_targets:
        # Exclude target questions from feasible set (disjoint mode)
        target_set = set(target_questions)
        candidates = [q for q in feasible_questions if q not in target_set]
    else:
        # Use all feasible questions (overlapping mode)
        candidates = list(feasible_questions)
    
    if len(candidates) == 0:
        raise ValueError("No feasible questions available for selection")
    
    selected_questions: List[str] = []
    n_select = min(budget, len(candidates))
    
    # Import tqdm for progress tracking
    try:
        from tqdm import tqdm
        show_progress = n_select > 10  # Only show for large selections
    except ImportError:
        show_progress = False
        tqdm = None
    
    # Greedy selection loop
    iterator = range(n_select)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="    Selecting questions", leave=False)
    
    for _ in iterator:
        best_question = None
        best_expected_cost = float('inf')
        
        remaining = [q for q in candidates if q not in selected_questions]
        if len(remaining) == 0:
            break
        
        for candidate in remaining:
            if use_optimized and precomputed is not None and candidate in precomputed.question_to_idx and NUMBA_AVAILABLE:
                expected_cost = _compute_expected_cost_under_prior_jit(
                    prior_weights=prior_weights,
                    precomputed=precomputed,
                    candidate_idx=precomputed.question_to_idx[candidate],
                    target_indices=target_indices,
                    criterion=selection_criterion,
                )
            else:
                expected_cost = _compute_expected_cost_under_prior(
                    prior_weights=prior_weights,
                    persona_responses=persona_responses,
                    candidate_question=candidate,
                    target_questions=target_questions,
                    criterion=selection_criterion,
                )
            
            if expected_cost < best_expected_cost:
                best_expected_cost = expected_cost
                best_question = candidate
        
        if best_question is not None:
            selected_questions.append(best_question)
    
    return selected_questions


def _compute_expected_cost_under_prior(
    prior_weights: np.ndarray,
    persona_responses: pd.DataFrame,
    candidate_question: str,
    target_questions: List[str],
    criterion: str,
) -> float:
    """Compute expected posterior cost (non-optimized version)."""
    predictive_dist = compute_posterior_predictive(
        prior_weights, persona_responses, candidate_question
    )
    K = len(predictive_dist)
    
    expected_cost = 0.0
    
    for k in range(K):
        if predictive_dist[k] < EPS:
            continue
        
        updated_posterior = update_posterior_with_observation(
            prior_weights, persona_responses, candidate_question, k
        )
        
        if criterion == "entropy_persona":
            cost = entropy_over_personas(updated_posterior)
        elif criterion == "entropy_target":
            cost = entropy_over_target_questions(
                updated_posterior, persona_responses, target_questions
            )
        elif criterion == "variance_persona":
            cost = variance_of_categorical(updated_posterior)
        elif criterion == "variance_target":
            cost = variance_over_target_questions(
                updated_posterior, persona_responses, target_questions
            )
        elif criterion == "crps_target":
            cost = crps_over_target_questions(
                updated_posterior, persona_responses, target_questions
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        expected_cost += predictive_dist[k] * cost
    
    return expected_cost


def _compute_expected_cost_under_prior_jit(
    prior_weights: np.ndarray,
    precomputed: PrecomputedPersonaData,
    candidate_idx: int,
    target_indices: np.ndarray,
    criterion: str,
) -> float:
    """Compute expected posterior cost using JIT-compiled functions."""
    persona_probs_q = np.ascontiguousarray(precomputed.persona_probs[:, candidate_idx, :])
    K = persona_probs_q.shape[1]
    
    # Posterior predictive
    predictive = compute_posterior_predictive_jit(prior_weights, persona_probs_q)
    
    expected_cost = 0.0
    
    for k in range(K):
        if predictive[k] < EPS:
            continue
        
        updated_posterior = update_posterior_jit(prior_weights, persona_probs_q, k)
        
        if criterion == "entropy_persona":
            cost = entropy_jit(updated_posterior)
        elif criterion == "entropy_target":
            cost = entropy_over_target_questions_jit(
                updated_posterior, precomputed.persona_probs, target_indices
            )
        elif criterion == "variance_persona":
            # Gini impurity: 1 - sum(p^2)
            cost = 1.0 - np.sum(updated_posterior * updated_posterior)
        elif criterion == "variance_target":
            cost = variance_over_target_questions_jit(
                updated_posterior, precomputed.persona_probs, target_indices
            )
        elif criterion == "crps_target":
            cost = crps_over_target_questions_jit(
                updated_posterior, precomputed.persona_probs, target_indices
            )
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        expected_cost += predictive[k] * cost
    
    return expected_cost


def nonadaptive_set_query(
    user_response_row: pd.Series,
    persona_responses: pd.DataFrame,
    selected_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray] = None,
    precomputed: Optional[PrecomputedPersonaData] = None,
    use_optimized: bool = True,
) -> Dict[str, Any]:
    """
    Query a user using a pre-selected fixed set of questions.
    
    Parameters
    ----------
    user_response_row : pd.Series
        A single row from user_responses DataFrame.
    persona_responses : pd.DataFrame
        DataFrame with personas as rows and questions as columns.
    selected_questions : List[str]
        Pre-selected questions to ask.
    target_questions : List[str]
        Questions to predict.
    prior_weights : np.ndarray, optional
        Prior over personas. Default is uniform.
    precomputed : PrecomputedPersonaData, optional
        Pre-computed persona data.
    use_optimized : bool, default True
        Whether to use optimized functions.
    
    Returns
    -------
    result : Dict[str, Any]
        Same structure as other query functions.
    """
    n_personas = len(persona_responses)
    
    if prior_weights is None:
        prior_weights = np.ones(n_personas) / n_personas
    
    # Filter to questions the user has answered
    user_questions = [
        q for q in selected_questions
        if q in user_response_row.index and user_response_row[q] != -1
    ]
    
    # Initialize
    posterior_weights = prior_weights.copy()
    asked_questions: List[str] = []
    observed_answers: List[int] = []
    trajectory: List[Dict] = []
    
    trajectory.append({
        'step': 0,
        'question_asked': None,
        'answer_observed': None,
        'posterior_entropy': entropy_jit(posterior_weights) if use_optimized and NUMBA_AVAILABLE else entropy_over_personas(posterior_weights),
    })
    
    # Ask each selected question
    for t, question in enumerate(user_questions):
        observed_answer = int(user_response_row[question])
        
        if use_optimized and precomputed is not None and question in precomputed.question_to_idx and NUMBA_AVAILABLE:
            q_idx = precomputed.question_to_idx[question]
            posterior_weights = update_posterior_jit(
                posterior_weights,
                precomputed.persona_probs[:, q_idx, :],
                observed_answer
            )
            post_entropy = entropy_jit(posterior_weights)
        else:
            posterior_weights = update_posterior_with_observation(
                posterior_weights, persona_responses, question, observed_answer
            )
            post_entropy = entropy_over_personas(posterior_weights)
        
        asked_questions.append(question)
        observed_answers.append(observed_answer)
        
        trajectory.append({
            'step': t + 1,
            'question_asked': question,
            'answer_observed': observed_answer,
            'posterior_entropy': post_entropy,
        })
    
    # Compute predictions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if use_optimized and precomputed is not None and q in precomputed.question_to_idx and NUMBA_AVAILABLE:
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


def _evaluate_nonadaptive_single_user(
    user_idx: int,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    selected_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray],
    precomputed: Optional[PrecomputedPersonaData],
    use_optimized: bool,
) -> Dict[str, Any]:
    """Helper for parallel non-adaptive user evaluation."""
    query_result = nonadaptive_set_query(
        user_response_row=user_row,
        persona_responses=persona_responses,
        selected_questions=selected_questions,
        target_questions=target_questions,
        prior_weights=prior_weights,
        precomputed=precomputed,
        use_optimized=use_optimized,
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
        **metrics,
    }


def evaluate_nonadaptive_set_on_users(
    user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray] = None,
    user_indices: Optional[List[int]] = None,
    selection_criterion: str = "entropy_persona",
    selected_questions: Optional[List[str]] = None,
    use_optimized: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the non-adaptive set policy on multiple users with parallelization.
    
    Parameters
    ----------
    n_jobs : int, default -1
        Number of parallel jobs (-1 = all cores).
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    # Pre-compute persona data
    precomputed = None
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    # Select questions if not provided
    if selected_questions is None:
        if verbose:
            print("Selecting non-adaptive question set...")
        selected_questions = select_nonadaptive_question_set(
            persona_responses=persona_responses,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=budget,
            prior_weights=prior_weights,
            selection_criterion=selection_criterion,
            precomputed=precomputed,
            use_optimized=use_optimized,
        )
        if verbose:
            print(f"Selected {len(selected_questions)} questions")
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(user_indices) > 1:
        if verbose:
            print(f"Evaluating {len(user_indices)} users in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_evaluate_nonadaptive_single_user)(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                selected_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            
            result = _evaluate_nonadaptive_single_user(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                selected_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Random Fixed Set Baseline
# =============================================================================

def select_random_fixed_question_set(
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    seed: Optional[int] = None,
    exclude_targets: bool = True,
) -> List[str]:
    """
    Select a fixed set of questions by uniform random sampling (no replacement).
    
    This is a simple baseline that does not use any prior information or
    optimization. Questions are sampled uniformly at random from the feasible set.
    
    Parameters
    ----------
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict (excluded from selection if exclude_targets=True).
    budget : int
        Number of questions to select.
    seed : int, optional
        Random seed for reproducibility.
    exclude_targets : bool, default=True
        Whether to exclude target questions from selection.
        Set to False for overlapping mode where feasible = target.
    
    Returns
    -------
    selected_questions : List[str]
        Randomly selected list of questions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Determine candidate pool
    if exclude_targets:
        # Exclude target questions from feasible set (disjoint mode)
        target_set = set(target_questions)
        candidates = [q for q in feasible_questions if q not in target_set]
    else:
        # Use all feasible questions (overlapping mode)
        candidates = list(feasible_questions)
    
    if len(candidates) == 0:
        raise ValueError("No feasible questions available for selection")
    
    # Sample without replacement
    n_select = min(budget, len(candidates))
    selected_indices = np.random.choice(len(candidates), size=n_select, replace=False)
    selected_questions = [candidates[i] for i in selected_indices]
    
    return selected_questions


def _evaluate_random_fixed_single_user(
    user_idx: int,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    selected_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray],
    precomputed: Optional[PrecomputedPersonaData],
    use_optimized: bool,
) -> Dict[str, Any]:
    """Evaluate random fixed set baseline on a single user."""
    # Reuse nonadaptive_set_query since the only difference is how questions are selected
    query_result = nonadaptive_set_query(
        user_response_row=user_row,
        persona_responses=persona_responses,
        selected_questions=selected_questions,
        target_questions=target_questions,
        prior_weights=prior_weights,
        precomputed=precomputed,
        use_optimized=use_optimized,
    )
    return {
        "user_idx": user_idx,
        **query_result,
    }


def evaluate_random_fixed_set_on_users(
    user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    prior_weights: Optional[np.ndarray] = None,
    user_indices: Optional[List[int]] = None,
    selected_questions: Optional[List[str]] = None,
    seed: Optional[int] = None,
    use_optimized: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the random fixed set baseline on multiple users.
    
    This baseline selects a fixed set of questions uniformly at random
    (without using any prior or optimization), then asks the same questions
    to all users.
    
    Parameters
    ----------
    user_responses : pd.DataFrame
        User response data.
    persona_responses : pd.DataFrame
        Persona response distributions.
    feasible_questions : List[str]
        Questions that can be asked.
    target_questions : List[str]
        Questions to predict.
    budget : int
        Number of questions to select.
    prior_weights : np.ndarray, optional
        Prior over personas (used for prediction, not selection).
    user_indices : List[int], optional
        Subset of users to evaluate.
    selected_questions : List[str], optional
        Pre-selected questions. If None, randomly selects.
    seed : int, optional
        Random seed for question selection.
    use_optimized : bool, default True
        Whether to use optimized functions.
    n_jobs : int, default -1
        Number of parallel jobs (-1 = all cores).
    verbose : bool, default False
        Whether to print progress.
    
    Returns
    -------
    pd.DataFrame
        Evaluation results for each user.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    # Pre-compute persona data
    precomputed = None
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    # Select questions randomly if not provided
    if selected_questions is None:
        if verbose:
            print("Selecting random fixed question set...")
        selected_questions = select_random_fixed_question_set(
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=budget,
            seed=seed,
        )
        if verbose:
            print(f"Randomly selected {len(selected_questions)} questions")
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(user_indices) > 1:
        if verbose:
            print(f"Evaluating {len(user_indices)} users in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_evaluate_random_fixed_single_user)(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                selected_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            
            result = _evaluate_random_fixed_single_user(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                selected_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Full (All Feasible Questions)
# =============================================================================

def full_query(
    user_response_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray] = None,
    precomputed: Optional[PrecomputedPersonaData] = None,
    use_optimized: bool = True,
    exclude_targets: bool = True,
) -> Dict[str, Any]:
    """
    Query using ALL feasible questions the user has answered.
    
    This uses all available feasible information for prediction.
    
    Parameters
    ----------
    exclude_targets : bool, default=True
        If True, exclude target questions from the questions used for inference.
        Set to False in overlapping evaluation mode where targets are also feasible.
    """
    n_personas = len(persona_responses)
    
    if prior_weights is None:
        prior_weights = np.ones(n_personas) / n_personas
    
    # Get all feasible questions the user has answered
    target_set = set(target_questions)
    if exclude_targets:
        user_feasible = [
            q for q in feasible_questions
            if q in user_response_row.index 
            and user_response_row[q] != -1
            and q not in target_set
        ]
    else:
        # In overlapping mode, all questions (including targets) can be used
        user_feasible = [
            q for q in feasible_questions
            if q in user_response_row.index 
            and user_response_row[q] != -1
        ]
    
    # Collect all answers
    asked_questions: List[str] = []
    observed_answers: List[int] = []
    
    for q in user_feasible:
        observed_answer = int(user_response_row[q])
        asked_questions.append(q)
        observed_answers.append(observed_answer)
    
    # Compute posterior using all observations
    if use_optimized and precomputed is not None and NUMBA_AVAILABLE:
        # Convert to indices
        asked_indices = np.array(
            [precomputed.question_to_idx[q] for q in asked_questions
             if q in precomputed.question_to_idx],
            dtype=np.int64
        )
        answers_array = np.array(
            [observed_answers[i] for i, q in enumerate(asked_questions)
             if q in precomputed.question_to_idx],
            dtype=np.int64
        )
        
        posterior_weights = compute_posterior_over_personas_jit(
            prior_weights,
            precomputed.persona_probs,
            asked_indices,
            answers_array
        )
    else:
        posterior_weights = compute_posterior_over_personas(
            prior_weights, persona_responses, asked_questions, observed_answers
        )
    
    # Compute predictions
    predicted_distributions = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            if use_optimized and precomputed is not None and q in precomputed.question_to_idx and NUMBA_AVAILABLE:
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
        'n_questions_used': len(asked_questions),
        'final_posterior_entropy': entropy_jit(posterior_weights) if use_optimized and NUMBA_AVAILABLE else entropy_over_personas(posterior_weights),
    }


def _evaluate_full_single_user(
    user_idx: int,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray],
    precomputed: Optional[PrecomputedPersonaData],
    use_optimized: bool,
) -> Dict[str, Any]:
    """Helper for parallel full method user evaluation."""
    query_result = full_query(
        user_response_row=user_row,
        persona_responses=persona_responses,
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        prior_weights=prior_weights,
        precomputed=precomputed,
        use_optimized=use_optimized,
    )
    
    metrics = evaluate_predictions(
        query_result['predicted_distributions'],
        user_row,
        target_questions,
    )
    
    return {
        'user_idx': user_idx,
        'n_questions_used': query_result['n_questions_used'],
        'final_posterior_entropy': query_result['final_posterior_entropy'],
        **metrics,
    }


def evaluate_full_on_users(
    user_responses: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    prior_weights: Optional[np.ndarray] = None,
    user_indices: Optional[List[int]] = None,
    use_optimized: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the full method (all feasible questions) on multiple users with parallelization.
    """
    if user_indices is None:
        user_indices = list(range(len(user_responses)))
    
    # Pre-compute persona data
    precomputed = None
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(user_indices) > 1:
        if verbose:
            print(f"Evaluating {len(user_indices)} users in parallel (n_jobs={n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_evaluate_full_single_user)(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            for user_idx in user_indices
        )
    else:
        results = []
        for i, user_idx in enumerate(user_indices):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processing user {i + 1}/{len(user_indices)}")
            
            result = _evaluate_full_single_user(
                user_idx,
                user_responses.iloc[user_idx],
                persona_responses,
                feasible_questions,
                target_questions,
                prior_weights,
                precomputed,
                use_optimized,
            )
            results.append(result)
    
    return pd.DataFrame(results)
