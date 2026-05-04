"""
Results handling, analysis, and visualization for adaptive query experiments.

This module provides:
- Structured experiment result storage
- Summary statistics computation
- Visualization generation (tables, plots)
- Organized output management
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter

import numpy as np
import pandas as pd

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Check for seaborn availability
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class SingleUserResult:
    """Result for a single user query session."""
    user_id: str
    asked_questions: List[str]
    observed_answers: List[int]
    predicted_distributions: Dict[str, List[float]]
    true_responses: Dict[str, int]
    metrics: Dict[str, float]
    trajectory: List[Dict[str, Any]]
    n_questions_asked: int
    n_target_evaluated: int


@dataclass
class MethodResult:
    """Aggregated results for a single method."""
    method_name: str
    user_results: List[SingleUserResult]
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def compute_summary(self):
        """Compute summary statistics from user results."""
        if not self.user_results:
            return
        
        metrics_keys = ["accuracy", "brier_score", "log_loss", "kl_divergence", "mse", "ci_coverage", "crps"]
        
        for key in metrics_keys:
            values = [u.metrics.get(key, np.nan) for u in self.user_results]
            values = [v for v in values if not np.isnan(v)]
            if values:
                self.summary_metrics[f"{key}_mean"] = np.mean(values)
                self.summary_metrics[f"{key}_std"] = np.std(values)
                self.summary_metrics[f"{key}_median"] = np.median(values)
                self.summary_metrics[f"{key}_min"] = np.min(values)
                self.summary_metrics[f"{key}_max"] = np.max(values)
        
        # Question statistics
        all_questions = []
        for u in self.user_results:
            all_questions.extend(u.asked_questions)
        self.summary_metrics["total_questions_asked"] = len(all_questions)
        self.summary_metrics["n_users"] = len(self.user_results)
        
        # Entropy statistics if available
        final_entropies = []
        for u in self.user_results:
            if u.trajectory and "posterior_entropy" in u.trajectory[-1]:
                final_entropies.append(u.trajectory[-1]["posterior_entropy"])
        if final_entropies:
            self.summary_metrics["final_entropy_mean"] = np.mean(final_entropies)
            self.summary_metrics["final_entropy_std"] = np.std(final_entropies)


@dataclass 
class ExperimentResult:
    """Complete experiment results."""
    experiment_id: str
    timestamp: str
    dataset_name: str
    budget: int
    n_train_users: int
    n_test_users: int
    n_feasible_questions: int
    n_target_questions: int
    feasible_questions: List[str]
    target_questions: List[str]
    method_results: Dict[str, MethodResult] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    # One-shot fits performed before any method runs. These are *not* included
    # in any method's `runtime_seconds`; they are separate amortized costs
    # shared across all persona-based methods.
    empirical_bayes_fit_time_seconds: Optional[float] = None
    clustering_fit_time_seconds: Optional[float] = None


# =============================================================================
# Results Collection Utilities
# =============================================================================

def collect_detailed_results(
    query_result: Dict[str, Any],
    user_id: str,
    user_response_row: pd.Series,
    target_questions: List[str],
    temperature: float = 1.0,
    score_values: Optional[np.ndarray] = None,
    ci_confidence_level: float = 0.95,
    evaluation_mode: str = "disjoint",
) -> SingleUserResult:
    """
    Collect detailed results from a single user query session.
    
    Parameters
    ----------
    query_result : Dict
        Output from any query function (greedy, random, etc.)
    user_id : str
        User identifier
    user_response_row : pd.Series
        User's responses (for ground truth)
    target_questions : List[str]
        Target questions being predicted
    temperature : float, default=1.0
        Temperature for scaling predictions. τ > 1 makes distributions softer.
    score_values : np.ndarray, optional
        Score values for each category (for posterior mean MSE).
        If None, uses category indices.
    ci_confidence_level : float, default=0.95
        Confidence level for CI coverage metric.
    evaluation_mode : str, default="disjoint"
        Evaluation mode:
        - "disjoint": Feasible and target questions are separate.
        - "overlapping": Asked questions get point-mass predictions (error=0).
    
    Returns
    -------
    SingleUserResult
        Structured result object
    """
    # Get asked questions for overlapping mode
    asked_questions = query_result.get("asked_questions", [])
    
    # Get true responses for target questions
    true_responses = {}
    for q in target_questions:
        if q in user_response_row.index and user_response_row[q] != -1:
            true_responses[q] = int(user_response_row[q])
    
    # Convert predicted distributions to serializable format
    pred_dists = {}
    for q, dist in query_result.get("predicted_distributions", {}).items():
        if isinstance(dist, np.ndarray):
            pred_dists[q] = dist.tolist()
        else:
            pred_dists[q] = list(dist)
    
    # Compute metrics
    from .utils import evaluate_predictions
    
    # In overlapping mode, pass asked_questions so they get point-mass scores
    asked_for_eval = asked_questions if evaluation_mode == "overlapping" else None
    
    metrics = evaluate_predictions(
        query_result.get("predicted_distributions", {}),
        user_response_row,
        target_questions,
        temperature=temperature,
        score_values=score_values,
        ci_confidence_level=ci_confidence_level,
        asked_questions=asked_for_eval,
    )
    
    return SingleUserResult(
        user_id=user_id,
        asked_questions=query_result.get("asked_questions", []),
        observed_answers=query_result.get("observed_answers", []),
        predicted_distributions=pred_dists,
        true_responses=true_responses,
        metrics=metrics,
        trajectory=query_result.get("trajectory", []),
        n_questions_asked=len(query_result.get("asked_questions", [])),
        n_target_evaluated=metrics.get("n_evaluated", 0),
    )


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_question_frequency(method_result: MethodResult) -> pd.DataFrame:
    """
    Compute how often each question was asked.
    
    Returns DataFrame with question, count, and proportion.
    """
    all_questions = []
    for user_result in method_result.user_results:
        all_questions.extend(user_result.asked_questions)
    
    counter = Counter(all_questions)
    total = len(all_questions)
    
    df = pd.DataFrame([
        {"question": q, "count": c, "proportion": c / total if total > 0 else 0}
        for q, c in counter.most_common()
    ])
    return df


def compute_question_position_stats(method_result: MethodResult) -> pd.DataFrame:
    """
    Compute statistics about when each question is asked (position in sequence).
    """
    question_positions = {}
    
    for user_result in method_result.user_results:
        for pos, q in enumerate(user_result.asked_questions):
            if q not in question_positions:
                question_positions[q] = []
            question_positions[q].append(pos + 1)  # 1-indexed position
    
    rows = []
    for q, positions in question_positions.items():
        rows.append({
            "question": q,
            "mean_position": np.mean(positions),
            "median_position": np.median(positions),
            "std_position": np.std(positions),
            "min_position": np.min(positions),
            "max_position": np.max(positions),
            "times_asked": len(positions),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_position")
    return df


def compute_trajectory_stats(method_result: MethodResult) -> pd.DataFrame:
    """
    Compute statistics about trajectories (entropy reduction over time).
    """
    # Collect entropy at each step
    step_entropies = {}
    
    for user_result in method_result.user_results:
        for step_info in user_result.trajectory:
            step = step_info.get("step", 0)
            entropy = step_info.get("posterior_entropy")
            if entropy is not None:
                if step not in step_entropies:
                    step_entropies[step] = []
                step_entropies[step].append(entropy)
    
    rows = []
    for step in sorted(step_entropies.keys()):
        values = step_entropies[step]
        rows.append({
            "step": step,
            "entropy_mean": np.mean(values),
            "entropy_std": np.std(values),
            "entropy_median": np.median(values),
            "n_users": len(values),
        })
    
    return pd.DataFrame(rows)


def compute_per_question_accuracy(method_result: MethodResult) -> pd.DataFrame:
    """
    Compute accuracy for each target question.
    """
    question_correct = {}
    question_total = {}
    
    for user_result in method_result.user_results:
        for q, pred_dist in user_result.predicted_distributions.items():
            if q in user_result.true_responses:
                true_idx = user_result.true_responses[q]
                pred_idx = int(np.argmax(pred_dist))
                
                if q not in question_correct:
                    question_correct[q] = 0
                    question_total[q] = 0
                
                question_total[q] += 1
                if pred_idx == true_idx:
                    question_correct[q] += 1
    
    rows = []
    for q in question_total:
        rows.append({
            "question": q,
            "correct": question_correct[q],
            "total": question_total[q],
            "accuracy": question_correct[q] / question_total[q] if question_total[q] > 0 else 0,
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("accuracy", ascending=False)
    return df


# =============================================================================
# Performance by Budget Analysis (Optimized)
# =============================================================================

def _compute_user_metrics_by_budget_persona(
    user_result: SingleUserResult,
    precomputed_data: Any,
    prior_weights: np.ndarray,
    max_budget: int,
    temperature: float = 1.0,
    return_per_question: bool = False,
    evaluation_mode: str = "disjoint",
    ci_confidence_level: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute metrics at each budget for a single user (persona-based methods).
    
    Uses JIT-compiled functions for speed.
    Returns dict mapping budget -> {accuracy, brier_score, log_loss, ...}.
    
    For users with fewer questions than max_budget, the final metrics (after
    asking all available questions) are reused for all higher budget levels.
    This ensures all users are included at all budget levels.
    
    If return_per_question=True, also returns per-question metrics lists
    for computing standard errors.
    
    Parameters
    ----------
    evaluation_mode : str, default="disjoint"
        "disjoint": Targets are separate from asked questions.
        "overlapping": Asked questions get perfect scores.
    ci_confidence_level : float, default=0.95
        Confidence level for CI coverage metric.
    """
    from .utils import (
        compute_posterior_over_personas_jit,
        compute_posterior_predictive_jit,
        apply_temperature_scaling,
        accuracy_score,
        brier_score,
        log_loss_score,
        posterior_mean_se,
        ci_coverage,
        crps_score,
        NUMBA_AVAILABLE,
    )
    
    results = {}
    
    asked_questions = user_result.asked_questions
    observed_answers = user_result.observed_answers
    target_questions = list(user_result.true_responses.keys())
    
    if not target_questions:
        return results
    
    # Get target question indices
    target_indices = []
    for q in target_questions:
        if q in precomputed_data.question_to_idx:
            target_indices.append(precomputed_data.question_to_idx[q])
    
    if not target_indices:
        return results
    
    target_indices = np.array(target_indices, dtype=np.int64)
    
    # Number of questions this user actually has
    n_user_questions = len(asked_questions)
    
    # For each budget, compute metrics
    for budget in range(1, max_budget + 1):
        # Use min(budget, n_user_questions) - for budgets beyond user's questions,
        # we use all their available questions (same as their final state)
        effective_budget = min(budget, n_user_questions)
        
        # If we've already computed metrics for this effective budget, reuse them
        if effective_budget in results:
            results[budget] = results[effective_budget].copy()
            continue
        
        if effective_budget == 0:
            continue
        
        # Get asked question indices and answers for this effective budget
        asked_q = asked_questions[:effective_budget]
        asked_a = observed_answers[:effective_budget]
        asked_set = set(asked_q)  # For overlapping mode lookup
        
        asked_indices = []
        valid_answers = []
        for q, a in zip(asked_q, asked_a):
            if q in precomputed_data.question_to_idx:
                asked_indices.append(precomputed_data.question_to_idx[q])
                valid_answers.append(a)
        
        if not asked_indices:
            continue
        
        asked_indices = np.array(asked_indices, dtype=np.int64)
        valid_answers = np.array(valid_answers, dtype=np.int64)
        
        # Compute posterior using JIT function
        if NUMBA_AVAILABLE:
            posterior = compute_posterior_over_personas_jit(
                prior_weights,
                precomputed_data.persona_probs,
                asked_indices,
                valid_answers,
            )
        else:
            from .utils import compute_posterior_over_personas
            posterior = compute_posterior_over_personas(
                prior_weights, None, asked_q, asked_a
            )
        
        # Compute predictions for target questions
        acc_list = []
        brier_list = []
        ll_list = []
        mse_list = []
        ci_list = []
        crps_list = []
        
        for i, q in enumerate(target_questions):
            if q not in user_result.true_responses:
                continue
            true_response = user_result.true_responses[q]
            
            # In overlapping mode, skip asked questions - only evaluate on unasked
            if evaluation_mode == "overlapping" and q in asked_set:
                continue
            
            if q in precomputed_data.question_to_idx:
                q_idx = precomputed_data.question_to_idx[q]
                if NUMBA_AVAILABLE:
                    pred_dist = compute_posterior_predictive_jit(
                        posterior, precomputed_data.persona_probs[:, q_idx, :]
                    )
                else:
                    pred_dist = np.dot(posterior, precomputed_data.persona_probs[:, q_idx, :])
                    pred_dist = pred_dist / (pred_dist.sum() + 1e-10)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    pred_dist = apply_temperature_scaling(pred_dist, temperature)
                
                acc_list.append(accuracy_score(pred_dist, true_response))
                brier_list.append(brier_score(pred_dist, true_response))
                ll_list.append(log_loss_score(pred_dist, true_response))
                mse_list.append(posterior_mean_se(pred_dist, true_response))
                ci_list.append(ci_coverage(pred_dist, true_response, ci_confidence_level))
                crps_list.append(crps_score(pred_dist, true_response))
        
        if acc_list:
            result_entry = {
                "accuracy": np.mean(acc_list),
                "brier_score": np.mean(brier_list),
                "log_loss": np.mean(ll_list),
                "mse": np.mean(mse_list),
                "ci_coverage": np.mean(ci_list),
                "crps": np.mean(crps_list),
            }
            if return_per_question:
                result_entry["accuracy_list"] = acc_list
                result_entry["brier_score_list"] = brier_list
                result_entry["log_loss_list"] = ll_list
                result_entry["mse_list"] = mse_list
                result_entry["ci_coverage_list"] = ci_list
                result_entry["crps_list"] = crps_list
            results[budget] = result_entry
    
    return results


def _compute_user_metrics_by_budget_cat(
    user_result: SingleUserResult,
    grm_params: Any,
    max_budget: int,
    temperature: float = 1.0,
    return_per_question: bool = False,
    evaluation_mode: str = "disjoint",
    ci_confidence_level: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute metrics at each budget for a single CAT user.
    
    Uses trajectory's theta info to recompute predictions at each step.
    
    For users with fewer questions than max_budget, the final metrics (after
    asking all available questions) are reused for all higher budget levels.
    This ensures all users are included at all budget levels.
    
    If return_per_question=True, also returns per-question metrics lists
    for computing standard errors.
    
    Parameters
    ----------
    evaluation_mode : str, default="disjoint"
        "disjoint": Targets are separate from asked questions.
        "overlapping": Asked questions get perfect scores.
    ci_confidence_level : float, default=0.95
        Confidence level for CI coverage metric.
    """
    from .utils import (
        accuracy_score, brier_score, log_loss_score, apply_temperature_scaling,
        posterior_mean_se, ci_coverage
    )
    
    results = {}
    
    trajectory = user_result.trajectory
    target_questions = list(user_result.true_responses.keys())
    asked_questions = user_result.asked_questions
    
    if not target_questions or grm_params is None:
        return results
    
    # Import CAT functions
    try:
        from .cat import grm_category_probs, CATState
    except ImportError:
        return results
    
    # Build a mapping from step -> theta_estimate from trajectory
    step_to_theta = {}
    max_step_in_trajectory = 0
    for step_info in trajectory:
        step = step_info.get("step", 0)
        if step == 0:
            continue
        theta_estimate = step_info.get("theta_estimate")
        if theta_estimate is not None:
            step_to_theta[step] = theta_estimate
            max_step_in_trajectory = max(max_step_in_trajectory, step)
    
    if not step_to_theta:
        return results
    
    # For each budget, compute predictions using theta estimate
    for budget in range(1, max_budget + 1):
        # Use min(budget, max_step) - for budgets beyond user's steps,
        # we use their final theta estimate
        effective_step = min(budget, max_step_in_trajectory)
        
        # Find the closest step <= effective_step that has theta
        actual_step = None
        for s in range(effective_step, 0, -1):
            if s in step_to_theta:
                actual_step = s
                break
        
        if actual_step is None:
            continue
        
        # If we've already computed metrics for this actual_step at a lower budget, reuse
        if actual_step in results and budget > actual_step:
            results[budget] = results[actual_step].copy()
            continue
        
        theta_estimate = step_to_theta[actual_step]
        
        # For overlapping mode, track which questions were asked up to this step
        asked_at_step = set(asked_questions[:actual_step]) if evaluation_mode == "overlapping" else set()
        
        # Compute predictions for target questions using this theta
        acc_list = []
        brier_list = []
        ll_list = []
        mse_list = []
        ci_list = []
        
        for q in target_questions:
            if q not in user_result.true_responses:
                continue
            
            true_response = user_result.true_responses[q]
            
            # In overlapping mode, skip asked questions - only evaluate on unasked
            if evaluation_mode == "overlapping" and q in asked_at_step:
                continue
            
            if q in grm_params.questions:
                a, b = grm_params.get_params(q)
                
                # Compute prediction using GRM
                pred_dist = grm_category_probs(np.array([theta_estimate]), a, b)[0]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    pred_dist = apply_temperature_scaling(pred_dist, temperature)
                
                acc_list.append(accuracy_score(pred_dist, true_response))
                brier_list.append(brier_score(pred_dist, true_response))
                ll_list.append(log_loss_score(pred_dist, true_response))
                mse_list.append(posterior_mean_se(pred_dist, true_response))
                ci_list.append(ci_coverage(pred_dist, true_response, ci_confidence_level))
        
        if acc_list:
            result_entry = {
                "accuracy": np.mean(acc_list),
                "brier_score": np.mean(brier_list),
                "log_loss": np.mean(ll_list),
                "mse": np.mean(mse_list),
                "ci_coverage": np.mean(ci_list),
            }
            if return_per_question:
                result_entry["accuracy_list"] = acc_list
                result_entry["brier_score_list"] = brier_list
                result_entry["log_loss_list"] = ll_list
                result_entry["mse_list"] = mse_list
                result_entry["ci_coverage_list"] = ci_list
            results[budget] = result_entry
    
    return results


def _compute_user_metrics_by_budget_cat_gpcm(
    user_result: SingleUserResult,
    gpcm_params: Any,
    max_budget: int,
    temperature: float = 1.0,
    return_per_question: bool = False,
    evaluation_mode: str = "disjoint",
    ci_confidence_level: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute metrics at each budget for a single CAT-GPCM user.
    
    Uses trajectory's theta info to recompute predictions at each step.
    """
    from .utils import (
        accuracy_score, brier_score, log_loss_score, apply_temperature_scaling,
        posterior_mean_se, ci_coverage
    )
    
    results = {}
    
    trajectory = user_result.trajectory
    target_questions = list(user_result.true_responses.keys())
    asked_questions = user_result.asked_questions
    
    if not target_questions or gpcm_params is None:
        return results
    
    # Import GPCM functions
    try:
        from .cat import gpcm_category_probs
    except ImportError:
        return results
    
    # Build a mapping from step -> theta_estimate from trajectory
    step_to_theta = {}
    max_step_in_trajectory = 0
    for step_info in trajectory:
        step = step_info.get("step", 0)
        if step == 0:
            continue
        theta_estimate = step_info.get("theta_estimate")
        if theta_estimate is not None:
            step_to_theta[step] = theta_estimate
            max_step_in_trajectory = max(max_step_in_trajectory, step)
    
    if not step_to_theta:
        return results
    
    # For each budget, compute predictions using theta estimate
    for budget in range(1, max_budget + 1):
        effective_step = min(budget, max_step_in_trajectory)
        
        actual_step = None
        for s in range(effective_step, 0, -1):
            if s in step_to_theta:
                actual_step = s
                break
        
        if actual_step is None:
            continue
        
        if actual_step in results and budget > actual_step:
            results[budget] = results[actual_step].copy()
            continue
        
        theta_estimate = step_to_theta[actual_step]
        asked_at_step = set(asked_questions[:actual_step]) if evaluation_mode == "overlapping" else set()
        
        acc_list = []
        brier_list = []
        ll_list = []
        mse_list = []
        ci_list = []
        
        for q in target_questions:
            if q not in user_result.true_responses:
                continue
            
            true_response = user_result.true_responses[q]
            
            # In overlapping mode, skip asked questions - only evaluate on unasked
            if evaluation_mode == "overlapping" and q in asked_at_step:
                continue
            
            if q in gpcm_params.questions:
                a, d = gpcm_params.get_params(q)
                
                # Compute prediction using GPCM
                pred_dist = gpcm_category_probs(np.array([theta_estimate]), a, d)[0]
                
                if temperature != 1.0:
                    pred_dist = apply_temperature_scaling(pred_dist, temperature)
                
                acc_list.append(accuracy_score(pred_dist, true_response))
                brier_list.append(brier_score(pred_dist, true_response))
                ll_list.append(log_loss_score(pred_dist, true_response))
                mse_list.append(posterior_mean_se(pred_dist, true_response))
                ci_list.append(ci_coverage(pred_dist, true_response, ci_confidence_level))
        
        if acc_list:
            result_entry = {
                "accuracy": np.mean(acc_list),
                "brier_score": np.mean(brier_list),
                "log_loss": np.mean(ll_list),
                "mse": np.mean(mse_list),
                "ci_coverage": np.mean(ci_list),
            }
            if return_per_question:
                result_entry["accuracy_list"] = acc_list
                result_entry["brier_score_list"] = brier_list
                result_entry["log_loss_list"] = ll_list
                result_entry["mse_list"] = mse_list
                result_entry["ci_coverage_list"] = ci_list
            results[budget] = result_entry
    
    return results


def _compute_user_metrics_by_budget_cat_mirt(
    user_result: SingleUserResult,
    mirt_params: Any,
    model_type: str,  # "mgrm" or "mgpcm"
    max_budget: int,
    temperature: float = 1.0,
    return_per_question: bool = False,
    evaluation_mode: str = "disjoint",
    ci_confidence_level: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute metrics at each budget for a single CAT-MIRT user (MGRM or MGPCM).
    
    Uses trajectory's theta info (D-dimensional vector) to recompute predictions at each step.
    """
    from .utils import (
        accuracy_score, brier_score, log_loss_score, apply_temperature_scaling,
        posterior_mean_se, ci_coverage
    )
    
    results = {}
    
    trajectory = user_result.trajectory
    target_questions = list(user_result.true_responses.keys())
    asked_questions = user_result.asked_questions
    
    if not target_questions or mirt_params is None:
        return results
    
    # Import MIRT functions
    try:
        from .cat_mirt import mgrm_category_probs, mgpcm_category_probs
    except ImportError:
        return results
    
    # Build a mapping from step -> theta_estimate from trajectory
    # For MIRT, theta_estimate is a list/array of D values
    step_to_theta = {}
    max_step_in_trajectory = 0
    for step_info in trajectory:
        step = step_info.get("step", 0)
        if step == 0:
            continue
        theta_estimate = step_info.get("theta_estimate")
        if theta_estimate is not None:
            # Convert to numpy array if needed
            step_to_theta[step] = np.array(theta_estimate)
            max_step_in_trajectory = max(max_step_in_trajectory, step)
    
    if not step_to_theta:
        return results
    
    # For each budget, compute predictions using theta estimate
    for budget in range(1, max_budget + 1):
        effective_step = min(budget, max_step_in_trajectory)
        
        actual_step = None
        for s in range(effective_step, 0, -1):
            if s in step_to_theta:
                actual_step = s
                break
        
        if actual_step is None:
            continue
        
        if actual_step in results and budget > actual_step:
            results[budget] = results[actual_step].copy()
            continue
        
        theta_estimate = step_to_theta[actual_step]  # D-dimensional vector
        asked_at_step = set(asked_questions[:actual_step]) if evaluation_mode == "overlapping" else set()
        
        acc_list = []
        brier_list = []
        ll_list = []
        mse_list = []
        ci_list = []
        
        for q in target_questions:
            if q not in user_result.true_responses:
                continue
            
            true_response = user_result.true_responses[q]
            
            # In overlapping mode, skip asked questions - only evaluate on unasked
            if evaluation_mode == "overlapping" and q in asked_at_step:
                continue
            
            if q in mirt_params.questions:
                # Compute prediction using appropriate MIRT model
                # theta_estimate is shape (D,), need to reshape for the functions
                theta_2d = theta_estimate.reshape(1, -1)  # Shape (1, D)
                
                if model_type == "mgrm":
                    a, b = mirt_params.get_params(q)  # (D-dim discrimination, thresholds)
                    pred_dist = mgrm_category_probs(theta_2d, a, b)[0]  # Extract first row -> shape (K,)
                else:  # mgpcm
                    a, d = mirt_params.get_params(q)  # (D-dim discrimination, step params)
                    pred_dist = mgpcm_category_probs(theta_2d, a, d)[0]  # Extract first row -> shape (K,)
                
                if temperature != 1.0:
                    pred_dist = apply_temperature_scaling(pred_dist, temperature)
                
                acc_list.append(accuracy_score(pred_dist, true_response))
                brier_list.append(brier_score(pred_dist, true_response))
                ll_list.append(log_loss_score(pred_dist, true_response))
                mse_list.append(posterior_mean_se(pred_dist, true_response))
                ci_list.append(ci_coverage(pred_dist, true_response, ci_confidence_level))
        
        if acc_list:
            result_entry = {
                "accuracy": np.mean(acc_list),
                "brier_score": np.mean(brier_list),
                "log_loss": np.mean(ll_list),
                "mse": np.mean(mse_list),
                "ci_coverage": np.mean(ci_list),
            }
            if return_per_question:
                result_entry["accuracy_list"] = acc_list
                result_entry["brier_score_list"] = brier_list
                result_entry["log_loss_list"] = ll_list
                result_entry["mse_list"] = mse_list
                result_entry["ci_coverage_list"] = ci_list
            results[budget] = result_entry
    
    return results


def compute_performance_by_budget(
    experiment: ExperimentResult,
    persona_responses: Any,  # pd.DataFrame
    user_responses: Any,  # pd.DataFrame  
    max_budget: int = 10,
    grm_params: Any = None,  # GRMParameters for CAT (legacy, also accepts via cat_model_params)
    n_jobs: int = -1,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
    ci_confidence_level: float = 0.95,
    cat_model_params: Dict[str, Any] = None,  # Dict mapping model_type -> params for all CAT models
) -> pd.DataFrame:
    """
    Compute performance metrics at each budget level (1, 2, ..., max_budget).
    
    For each method and each budget, computes predictions using only the first
    `budget` questions from each user's trajectory, then evaluates on targets.
    
    Optimized with:
    - JIT-compiled posterior computation
    - Pre-computed persona data arrays
    - Parallel processing across users
    
    Parameters
    ----------
    experiment : ExperimentResult
        Experiment results with trajectories.
    persona_responses : pd.DataFrame
        Persona response distributions (needed to recompute posteriors).
    user_responses : pd.DataFrame
        User responses (needed to get true responses for users in results).
    max_budget : int
        Maximum budget to evaluate (will evaluate 1, 2, ..., max_budget).
    evaluation_mode : str, default="disjoint"
        "disjoint": Targets are separate from asked questions.
        "overlapping": Asked questions get perfect scores (error→0 as budget→total).
    grm_params : GRMParameters, optional
        GRM parameters for CAT method. If None, CAT uses final metrics only.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    prior_weights : np.ndarray, optional
        Prior distribution over personas. If None, uniform prior is used.
    ci_confidence_level : float, default=0.95
        Confidence level for CI coverage metric.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: method, budget, accuracy, brier_score, log_loss, n_users
    """
    from .utils import PrecomputedPersonaData
    
    # Check for joblib
    try:
        from joblib import Parallel, delayed
        joblib_available = True
    except ImportError:
        joblib_available = False
    
    results = []
    
    # Pre-compute persona data once for all methods
    all_questions = list(persona_responses.columns)
    precomputed_data = PrecomputedPersonaData(persona_responses, all_questions)
    n_personas = precomputed_data.n_personas
    
    # Use provided prior or uniform
    if prior_weights is None:
        prior_weights = np.ones(n_personas, dtype=np.float64) / n_personas
    else:
        prior_weights = np.asarray(prior_weights, dtype=np.float64)
    
    for method_name, mr in experiment.method_results.items():
        is_full = method_name.lower() == "full"
        # Match all CAT variants: cat, cat_grm, cat_gpcm, cat_mgrm, cat_mgpcm
        is_cat = method_name.lower().startswith("cat")
        
        if is_full:
            # Full uses all questions - constant performance
            # Collect per-question metrics from all users for SE computation
            all_acc = []
            all_brier = []
            all_ll = []
            all_mse = []
            all_ci = []
            from .utils import accuracy_score, brier_score, log_loss_score, posterior_mean_se, ci_coverage
            for ur in mr.user_results:
                asked_set = set(ur.asked_questions) if ur.asked_questions else set()
                for q, true_resp in ur.true_responses.items():
                    # In overlapping mode, skip asked questions - only evaluate on unasked
                    if evaluation_mode == "overlapping" and q in asked_set:
                        continue
                    if q in ur.predicted_distributions:
                        pred_dist = np.array(ur.predicted_distributions[q])
                        all_acc.append(accuracy_score(pred_dist, true_resp))
                        all_brier.append(brier_score(pred_dist, true_resp))
                        all_ll.append(log_loss_score(pred_dist, true_resp))
                        all_mse.append(posterior_mean_se(pred_dist, true_resp))
                        all_ci.append(ci_coverage(pred_dist, true_resp, ci_confidence_level))
            
            n_questions = len(all_acc)
            if n_questions > 0:
                final_acc = np.mean(all_acc)
                final_brier = np.mean(all_brier)
                final_ll = np.mean(all_ll)
                final_mse = np.mean(all_mse)
                final_ci = np.mean(all_ci)
                acc_se = np.std(all_acc, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                brier_se = np.std(all_brier, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                ll_se = np.std(all_ll, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                mse_se = np.std(all_mse, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                ci_se = np.std(all_ci, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
            else:
                final_acc = mr.summary_metrics.get("accuracy_mean", np.nan)
                final_brier = mr.summary_metrics.get("brier_score_mean", np.nan)
                final_ll = mr.summary_metrics.get("log_loss_mean", np.nan)
                final_mse = mr.summary_metrics.get("mse_mean", np.nan)
                final_ci = mr.summary_metrics.get("ci_coverage_mean", np.nan)
                acc_se = brier_se = ll_se = mse_se = ci_se = 0.0
                n_questions = 0
            
            for budget in range(1, max_budget + 1):
                results.append({
                    "method": method_name,
                    "budget": budget,
                    "accuracy": final_acc,
                    "brier_score": final_brier,
                    "log_loss": final_ll,
                    "mse": final_mse,
                    "ci_coverage": final_ci,
                    "accuracy_se": acc_se,
                    "brier_score_se": brier_se,
                    "log_loss_se": ll_se,
                    "mse_se": mse_se,
                    "ci_coverage_se": ci_se,
                    "n_users": len(mr.user_results),
                    "n_questions": n_questions,
                })
            continue
        
        if is_cat:
            # For CAT, use trajectory theta estimates
            # Determine the CAT model type from method name
            method_lower = method_name.lower()
            cat_model_type = None
            if method_lower == "cat" or method_lower == "cat_grm":
                cat_model_type = "grm"
            elif method_lower == "cat_gpcm":
                cat_model_type = "gpcm"
            elif method_lower == "cat_mgrm":
                cat_model_type = "mgrm"
            elif method_lower == "cat_mgpcm":
                cat_model_type = "mgpcm"
            
            # Get the appropriate model params
            effective_grm_params = grm_params  # Legacy support
            if cat_model_params is not None and cat_model_type in cat_model_params:
                if cat_model_type == "grm":
                    effective_grm_params = cat_model_params["grm"]
            
            # For GRM, use detailed by-budget computation with theta trajectory
            if cat_model_type == "grm" and effective_grm_params is not None:
                if joblib_available and n_jobs != 1 and len(mr.user_results) > 1:
                    user_metrics_list = Parallel(n_jobs=n_jobs)(
                        delayed(_compute_user_metrics_by_budget_cat)(
                            user_result, effective_grm_params, max_budget, temperature, True, evaluation_mode, ci_confidence_level
                        )
                        for user_result in mr.user_results
                    )
                else:
                    user_metrics_list = [
                        _compute_user_metrics_by_budget_cat(ur, effective_grm_params, max_budget, temperature, True, evaluation_mode, ci_confidence_level)
                        for ur in mr.user_results
                    ]
                
                # Aggregate per-question metrics across all users by budget
                budget_metrics = {}  # budget -> {"accuracy_all": [...], ...}
                for user_metrics in user_metrics_list:
                    for budget, metrics in user_metrics.items():
                        if budget not in budget_metrics:
                            budget_metrics[budget] = {
                                "accuracy_all": [], "brier_score_all": [], "log_loss_all": [],
                                "mse_all": [], "ci_coverage_all": [], "crps_all": [],
                                "n_users": 0
                            }
                        # Collect per-question metrics
                        budget_metrics[budget]["accuracy_all"].extend(metrics.get("accuracy_list", [metrics["accuracy"]]))
                        budget_metrics[budget]["brier_score_all"].extend(metrics.get("brier_score_list", [metrics["brier_score"]]))
                        budget_metrics[budget]["log_loss_all"].extend(metrics.get("log_loss_list", [metrics["log_loss"]]))
                        budget_metrics[budget]["mse_all"].extend(metrics.get("mse_list", [metrics.get("mse", 0.0)]))
                        budget_metrics[budget]["ci_coverage_all"].extend(metrics.get("ci_coverage_list", [metrics.get("ci_coverage", 0.0)]))
                        budget_metrics[budget]["crps_all"].extend(metrics.get("crps_list", [metrics.get("crps", 0.0)]))
                        budget_metrics[budget]["n_users"] += 1
                
                for budget in sorted(budget_metrics.keys()):
                    m = budget_metrics[budget]
                    n_questions = len(m["accuracy_all"])
                    results.append({
                        "method": method_name,
                        "budget": budget,
                        "accuracy": np.mean(m["accuracy_all"]),
                        "brier_score": np.mean(m["brier_score_all"]),
                        "log_loss": np.mean(m["log_loss_all"]),
                        "mse": np.mean(m["mse_all"]),
                        "ci_coverage": np.mean(m["ci_coverage_all"]),
                        "crps": np.mean(m["crps_all"]),
                        "accuracy_se": np.std(m["accuracy_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "brier_score_se": np.std(m["brier_score_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "log_loss_se": np.std(m["log_loss_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "mse_se": np.std(m["mse_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "ci_coverage_se": np.std(m["ci_coverage_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "crps_se": np.std(m["crps_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "n_users": m["n_users"],
                        "n_questions": n_questions,
                    })
            elif cat_model_type == "gpcm" and cat_model_params is not None and "gpcm" in cat_model_params:
                # GPCM by-budget computation
                gpcm_params = cat_model_params["gpcm"]
                if joblib_available and n_jobs != 1 and len(mr.user_results) > 1:
                    user_metrics_list = Parallel(n_jobs=n_jobs)(
                        delayed(_compute_user_metrics_by_budget_cat_gpcm)(
                            user_result, gpcm_params, max_budget, temperature, True, evaluation_mode, ci_confidence_level
                        )
                        for user_result in mr.user_results
                    )
                else:
                    user_metrics_list = [
                        _compute_user_metrics_by_budget_cat_gpcm(ur, gpcm_params, max_budget, temperature, True, evaluation_mode, ci_confidence_level)
                        for ur in mr.user_results
                    ]
                
                # Aggregate per-question metrics across all users by budget
                budget_metrics = {}
                for user_metrics in user_metrics_list:
                    for budget, metrics in user_metrics.items():
                        if budget not in budget_metrics:
                            budget_metrics[budget] = {
                                "accuracy_all": [], "brier_score_all": [], "log_loss_all": [],
                                "mse_all": [], "ci_coverage_all": [], "crps_all": [],
                                "n_users": 0
                            }
                        budget_metrics[budget]["accuracy_all"].extend(metrics.get("accuracy_list", [metrics["accuracy"]]))
                        budget_metrics[budget]["brier_score_all"].extend(metrics.get("brier_score_list", [metrics["brier_score"]]))
                        budget_metrics[budget]["log_loss_all"].extend(metrics.get("log_loss_list", [metrics["log_loss"]]))
                        budget_metrics[budget]["mse_all"].extend(metrics.get("mse_list", [metrics.get("mse", 0.0)]))
                        budget_metrics[budget]["ci_coverage_all"].extend(metrics.get("ci_coverage_list", [metrics.get("ci_coverage", 0.0)]))
                        budget_metrics[budget]["crps_all"].extend(metrics.get("crps_list", [metrics.get("crps", 0.0)]))
                        budget_metrics[budget]["n_users"] += 1
                
                for budget in sorted(budget_metrics.keys()):
                    m = budget_metrics[budget]
                    n_questions = len(m["accuracy_all"])
                    results.append({
                        "method": method_name,
                        "budget": budget,
                        "accuracy": np.mean(m["accuracy_all"]),
                        "brier_score": np.mean(m["brier_score_all"]),
                        "log_loss": np.mean(m["log_loss_all"]),
                        "mse": np.mean(m["mse_all"]),
                        "ci_coverage": np.mean(m["ci_coverage_all"]),
                        "crps": np.mean(m["crps_all"]),
                        "accuracy_se": np.std(m["accuracy_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "brier_score_se": np.std(m["brier_score_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "log_loss_se": np.std(m["log_loss_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "mse_se": np.std(m["mse_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "ci_coverage_se": np.std(m["ci_coverage_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "crps_se": np.std(m["crps_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "n_users": m["n_users"],
                        "n_questions": n_questions,
                    })
            elif cat_model_type in ("mgrm", "mgpcm") and cat_model_params is not None and cat_model_type in cat_model_params:
                # MIRT (MGRM/MGPCM) by-budget computation
                mirt_params = cat_model_params[cat_model_type]
                if joblib_available and n_jobs != 1 and len(mr.user_results) > 1:
                    user_metrics_list = Parallel(n_jobs=n_jobs)(
                        delayed(_compute_user_metrics_by_budget_cat_mirt)(
                            user_result, mirt_params, cat_model_type, max_budget, temperature, True, evaluation_mode, ci_confidence_level
                        )
                        for user_result in mr.user_results
                    )
                else:
                    user_metrics_list = [
                        _compute_user_metrics_by_budget_cat_mirt(ur, mirt_params, cat_model_type, max_budget, temperature, True, evaluation_mode, ci_confidence_level)
                        for ur in mr.user_results
                    ]
                
                # Aggregate per-question metrics across all users by budget
                budget_metrics = {}
                for user_metrics in user_metrics_list:
                    for budget, metrics in user_metrics.items():
                        if budget not in budget_metrics:
                            budget_metrics[budget] = {
                                "accuracy_all": [], "brier_score_all": [], "log_loss_all": [],
                                "mse_all": [], "ci_coverage_all": [], "crps_all": [],
                                "n_users": 0
                            }
                        budget_metrics[budget]["accuracy_all"].extend(metrics.get("accuracy_list", [metrics["accuracy"]]))
                        budget_metrics[budget]["brier_score_all"].extend(metrics.get("brier_score_list", [metrics["brier_score"]]))
                        budget_metrics[budget]["log_loss_all"].extend(metrics.get("log_loss_list", [metrics["log_loss"]]))
                        budget_metrics[budget]["mse_all"].extend(metrics.get("mse_list", [metrics.get("mse", 0.0)]))
                        budget_metrics[budget]["ci_coverage_all"].extend(metrics.get("ci_coverage_list", [metrics.get("ci_coverage", 0.0)]))
                        budget_metrics[budget]["crps_all"].extend(metrics.get("crps_list", [metrics.get("crps", 0.0)]))
                        budget_metrics[budget]["n_users"] += 1
                
                for budget in sorted(budget_metrics.keys()):
                    m = budget_metrics[budget]
                    n_questions = len(m["accuracy_all"])
                    results.append({
                        "method": method_name,
                        "budget": budget,
                        "accuracy": np.mean(m["accuracy_all"]),
                        "brier_score": np.mean(m["brier_score_all"]),
                        "log_loss": np.mean(m["log_loss_all"]),
                        "mse": np.mean(m["mse_all"]),
                        "ci_coverage": np.mean(m["ci_coverage_all"]),
                        "crps": np.mean(m["crps_all"]),
                        "accuracy_se": np.std(m["accuracy_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "brier_score_se": np.std(m["brier_score_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "log_loss_se": np.std(m["log_loss_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "mse_se": np.std(m["mse_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "ci_coverage_se": np.std(m["ci_coverage_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "crps_se": np.std(m["crps_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                        "n_users": m["n_users"],
                        "n_questions": n_questions,
                    })
            else:
                # Fallback: use final predictions for all budgets
                from .utils import accuracy_score, brier_score, log_loss_score, posterior_mean_se, ci_coverage, crps_score
                
                all_acc = []
                all_brier = []
                all_ll = []
                all_mse = []
                all_ci = []
                
                for ur in mr.user_results:
                    asked_set = set(ur.asked_questions) if ur.asked_questions else set()
                    for q, true_resp in ur.true_responses.items():
                        # In overlapping mode, skip asked questions - only evaluate on unasked
                        if evaluation_mode == "overlapping" and q in asked_set:
                            continue
                        if q in ur.predicted_distributions:
                            pred_dist = np.array(ur.predicted_distributions[q])
                            all_acc.append(accuracy_score(pred_dist, true_resp))
                            all_brier.append(brier_score(pred_dist, true_resp))
                            all_ll.append(log_loss_score(pred_dist, true_resp))
                            all_mse.append(posterior_mean_se(pred_dist, true_resp))
                            all_ci.append(ci_coverage(pred_dist, true_resp, ci_confidence_level))
                
                n_questions = len(all_acc)
                if n_questions > 0:
                    final_acc = np.mean(all_acc)
                    final_brier = np.mean(all_brier)
                    final_ll = np.mean(all_ll)
                    final_mse = np.mean(all_mse)
                    final_ci = np.mean(all_ci)
                    acc_se = np.std(all_acc, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                    brier_se = np.std(all_brier, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                    ll_se = np.std(all_ll, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                    mse_se = np.std(all_mse, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                    ci_se = np.std(all_ci, ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0
                else:
                    final_acc = mr.summary_metrics.get("accuracy_mean", np.nan)
                    final_brier = mr.summary_metrics.get("brier_score_mean", np.nan)
                    final_ll = mr.summary_metrics.get("log_loss_mean", np.nan)
                    final_mse = mr.summary_metrics.get("mse_mean", np.nan)
                    final_ci = mr.summary_metrics.get("ci_coverage_mean", np.nan)
                    acc_se = brier_se = ll_se = mse_se = ci_se = 0.0
                
                for budget in range(1, max_budget + 1):
                    results.append({
                        "method": method_name,
                        "budget": budget,
                        "accuracy": final_acc,
                        "brier_score": final_brier,
                        "log_loss": final_ll,
                        "mse": final_mse,
                        "ci_coverage": final_ci,
                        "accuracy_se": acc_se,
                        "brier_score_se": brier_se,
                        "log_loss_se": ll_se,
                        "mse_se": mse_se,
                        "ci_coverage_se": ci_se,
                        "n_users": len(mr.user_results),
                        "n_questions": n_questions,
                    })
            continue
        
        # For persona-based methods (greedy, random, nonadaptive)
        # Parallel processing across users
        if joblib_available and n_jobs != 1 and len(mr.user_results) > 1:
            user_metrics_list = Parallel(n_jobs=n_jobs)(
                delayed(_compute_user_metrics_by_budget_persona)(
                    user_result, precomputed_data, prior_weights, max_budget, temperature, True, evaluation_mode, ci_confidence_level
                )
                for user_result in mr.user_results
            )
        else:
            user_metrics_list = [
                _compute_user_metrics_by_budget_persona(ur, precomputed_data, prior_weights, max_budget, temperature, True, evaluation_mode, ci_confidence_level)
                for ur in mr.user_results
            ]
        
        # Aggregate per-question metrics across all users by budget
        budget_metrics = {}  # budget -> {"accuracy_all": [...], ...}
        for user_metrics in user_metrics_list:
            for budget, metrics in user_metrics.items():
                if budget not in budget_metrics:
                    budget_metrics[budget] = {
                        "accuracy_all": [], "brier_score_all": [], "log_loss_all": [],
                        "mse_all": [], "ci_coverage_all": [], "crps_all": [],
                        "n_users": 0
                    }
                # Collect per-question metrics
                budget_metrics[budget]["accuracy_all"].extend(metrics.get("accuracy_list", [metrics["accuracy"]]))
                budget_metrics[budget]["brier_score_all"].extend(metrics.get("brier_score_list", [metrics["brier_score"]]))
                budget_metrics[budget]["log_loss_all"].extend(metrics.get("log_loss_list", [metrics["log_loss"]]))
                budget_metrics[budget]["mse_all"].extend(metrics.get("mse_list", [metrics.get("mse", 0.0)]))
                budget_metrics[budget]["ci_coverage_all"].extend(metrics.get("ci_coverage_list", [metrics.get("ci_coverage", 0.0)]))
                budget_metrics[budget]["crps_all"].extend(metrics.get("crps_list", [metrics.get("crps", 0.0)]))
                budget_metrics[budget]["n_users"] += 1
        
        for budget in sorted(budget_metrics.keys()):
            m = budget_metrics[budget]
            n_questions = len(m["accuracy_all"])
            results.append({
                "method": method_name,
                "budget": budget,
                "accuracy": np.mean(m["accuracy_all"]),
                "brier_score": np.mean(m["brier_score_all"]),
                "log_loss": np.mean(m["log_loss_all"]),
                "mse": np.mean(m["mse_all"]),
                "ci_coverage": np.mean(m["ci_coverage_all"]),
                "crps": np.mean(m["crps_all"]),
                "accuracy_se": np.std(m["accuracy_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "brier_score_se": np.std(m["brier_score_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "log_loss_se": np.std(m["log_loss_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "mse_se": np.std(m["mse_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "ci_coverage_se": np.std(m["ci_coverage_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "crps_se": np.std(m["crps_all"], ddof=1) / np.sqrt(n_questions) if n_questions > 1 else 0.0,
                "n_users": m["n_users"],
                "n_questions": n_questions,
            })
    
    return pd.DataFrame(results)


def plot_performance_by_budget(
    performance_df: pd.DataFrame,
    figures_path: Path,
    analysis_path: Path = None,
    metrics: List[str] = ["accuracy", "brier_score", "log_loss", "mse", "ci_coverage", "crps"],
    confidence_level: float = 1.96,
):
    """
    Plot performance metrics vs budget for each method with confidence bands.
    
    Creates:
    - A combined plot with all metrics (all_metrics.pdf)
    - Individual plots for each metric ({metric}.pdf)
    
    Parameters
    ----------
    performance_df : pd.DataFrame
        Output from compute_performance_by_budget().
    figures_path : Path
        Directory to save figures (PDFs). Should be figures/by_budget/.
    analysis_path : Path, optional
        Directory to save CSV data. If None, uses figures_path parent.
    metrics : List[str]
        Metrics to plot.
    confidence_level : float
        Multiplier for standard error (default 1.96 for 95% CI).
    """
    if analysis_path is None:
        analysis_path = figures_path.parent if figures_path.name == "by_budget" else figures_path
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    if performance_df.empty:
        return
    
    from matplotlib.ticker import MaxNLocator
    
    # Canonical method ordering (matches paper/plan.md tables). Methods not in this
    # list are appended at the end in their first-seen order so nothing is dropped.
    LEGEND_ORDER = [
        "random", "random_fixed", "nonadaptive", "greedy",
        "cat", "cat_grm", "cat_gpcm", "cat_mgrm", "cat_mgpcm",
        "full",
    ]
    present_methods = list(dict.fromkeys(performance_df["method"].tolist()))
    methods = [m for m in LEGEND_ORDER if m in present_methods]
    methods += [m for m in present_methods if m not in LEGEND_ORDER]
    
    # Stable color assignment keyed by canonical order so figures are comparable
    # across runs that include different subsets of methods.
    color_palette = plt.cm.tab10(np.linspace(0, 1, max(len(LEGEND_ORDER), len(methods))))
    color_map = {method: color_palette[i] for i, method in enumerate(methods)}
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in performance_df.columns]
    if not available_metrics:
        print("Warning: No metrics available to plot")
        return
    
    # Font sizes (used for both individual and combined figures)
    LABEL_FS = 16
    LEGEND_FS = 14
    TICK_FS = 13
    TITLE_FS = 16
    
    def get_style(method):
        """Get line style and marker for a method."""
        method_lower = method.lower()
        if method_lower == "full":
            return "--", "s"
        elif method_lower == "greedy":
            return "-", "o"
        elif method_lower == "random":
            return ":", "^"
        elif method_lower == "random_fixed":
            return ":", "v"
        elif method_lower == "nonadaptive":
            return "-.", "d"
        elif method_lower == "cat":
            return "-", "x"
        elif method_lower == "cat_grm":
            return "-", "x"
        elif method_lower == "cat_gpcm":
            return "-", "+"
        elif method_lower == "cat_mgrm":
            return "--", "x"
        elif method_lower == "cat_mgpcm":
            return "--", "+"
        else:
            return "-.", "d"
    
    def plot_single_metric(ax, metric, show_legend=True, show_title=False):
        """Plot a single metric on the given axis.
        
        ``show_title`` controls whether a per-metric subplot caption is drawn;
        the combined ``all_metrics.pdf`` uses titles on each subplot, while the
        standalone per-metric PDFs omit them (paper figures).
        """
        se_col = f"{metric}_se"
        has_se = se_col in performance_df.columns
        
        for method in methods:
            method_data = performance_df[performance_df["method"] == method]
            if method_data.empty:
                continue
            
            method_data = method_data.sort_values("budget")
            
            linestyle, marker = get_style(method)
            budgets = method_data["budget"].values
            means = method_data[metric].values
            color = color_map[method]
            
            # Plot confidence band if SE is available
            if has_se:
                se_values = method_data[se_col].values
                lower = means - confidence_level * se_values
                upper = means + confidence_level * se_values
                ax.fill_between(budgets, lower, upper, color=color, alpha=0.2)
            
            ax.plot(
                budgets, 
                means,
                marker=marker,
                linestyle=linestyle,
                label=method,
                color=color,
                linewidth=2,
                markersize=6,
            )
        
        ax.set_xlabel("Budget (# Questions)", fontsize=LABEL_FS)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=LABEL_FS)
        if show_title:
            ax.set_title(f"{metric.replace('_', ' ').title()} vs Budget", fontsize=TITLE_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        if show_legend:
            ax.legend(fontsize=LEGEND_FS)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set appropriate y-axis limits for bounded metrics
        if metric in ["accuracy", "ci_coverage"]:
            ax.set_ylim(0, 1.05)
    
    # Create combined plot with all metrics
    n_metrics = len(available_metrics)
    if n_metrics <= 3:
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    else:
        # 2 rows for more metrics
        n_cols = (n_metrics + 1) // 2
        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))
        axes = axes.flatten()
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        plot_single_metric(axes[i], metric, show_legend=(i == 0), show_title=True)
    
    # Hide unused subplots
    for j in range(len(available_metrics), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(figures_path / "all_metrics.pdf", bbox_inches="tight")
    plt.close()
    
    # Create individual plots for each metric (paper figures: no titles)
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_single_metric(ax, metric, show_legend=True, show_title=False)
        plt.tight_layout()
        plt.savefig(figures_path / f"{metric}.pdf", bbox_inches="tight")
        plt.close()
    
    # Also save data as CSV to analysis folder
    performance_df.to_csv(analysis_path / "performance_by_budget.csv", index=False)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_metrics_comparison(
    experiment: ExperimentResult,
    output_path: Path,
    metrics: List[str] = ["accuracy", "brier_score", "log_loss", "mse", "ci_coverage"],
):
    """
    Create bar chart comparing methods across all metrics at end of budget.
    
    Saves to output_path (should be figures/final/).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    methods = list(experiment.method_results.keys())
    
    # Filter to metrics that have data
    available_metrics = []
    for m in metrics:
        if any(mr.summary_metrics.get(f"{m}_mean") is not None 
               for mr in experiment.method_results.values()):
            available_metrics.append(m)
    
    if not available_metrics:
        return
    
    n_metrics = len(available_metrics)
    
    # Layout: 2 rows if more than 3 metrics
    if n_metrics <= 3:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    else:
        n_cols = (n_metrics + 1) // 2
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        axes = axes.flatten()
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        means = []
        stds = []
        for method in methods:
            mr = experiment.method_results[method]
            means.append(mr.summary_metrics.get(f"{metric}_mean", 0))
            stds.append(mr.summary_metrics.get(f"{metric}_std", 0))
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Method")
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for j in range(len(available_metrics), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / "metrics_comparison.pdf", bbox_inches="tight")
    plt.close()


def plot_metrics_distribution(
    experiment: ExperimentResult,
    output_path: Path,
    metric: str = "accuracy",
):
    """
    Create box/violin plot showing distribution of metric across users.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    data = []
    for method_name, mr in experiment.method_results.items():
        for user_result in mr.user_results:
            value = user_result.metrics.get(metric)
            if value is not None and not np.isnan(value):
                data.append({"method": method_name, metric: value})
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if SEABORN_AVAILABLE:
        sns.boxplot(data=df, x="method", y=metric, ax=ax)
    else:
        methods = df["method"].unique()
        data_by_method = [df[df["method"] == m][metric].values for m in methods]
        ax.boxplot(data_by_method, labels=methods)
    
    ax.set_xlabel("Method")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} Across Users")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / f"{metric}_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_entropy_trajectory(
    experiment: ExperimentResult,
    output_path: Path,
):
    """
    Plot uncertainty reduction over query steps for each method.
    
    For persona-based methods: plots posterior entropy over personas.
    For CAT: plots theta_std (posterior standard deviation of latent trait).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    # Separate methods by type
    persona_methods = {}  # Methods with posterior_entropy
    cat_methods = {}      # Methods with theta_std
    
    for method_name, mr in experiment.method_results.items():
        # Check what data is available
        has_entropy = False
        has_theta_std = False
        for user_result in mr.user_results[:1]:  # Check first user
            for step_info in user_result.trajectory[:2]:
                if step_info.get("posterior_entropy") is not None:
                    has_entropy = True
                if step_info.get("theta_std") is not None:
                    has_theta_std = True
        
        if has_entropy:
            persona_methods[method_name] = mr
        elif has_theta_std:
            cat_methods[method_name] = mr
    
    # Create plot with possibly two y-axes if we have both types
    has_both = len(persona_methods) > 0 and len(cat_methods) > 0
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    all_colors = plt.cm.tab10(np.linspace(0, 1, len(experiment.method_results)))
    color_map = {name: color for (name, _), color in 
                 zip(experiment.method_results.items(), all_colors)}
    
    # Plot persona-based methods on primary axis
    for method_name, mr in persona_methods.items():
        traj_stats = compute_trajectory_stats(mr)
        if traj_stats.empty:
            continue
        
        color = color_map[method_name]
        ax1.plot(traj_stats["step"], traj_stats["entropy_mean"], 
                marker="o", label=method_name, color=color)
        ax1.fill_between(
            traj_stats["step"],
            traj_stats["entropy_mean"] - traj_stats["entropy_std"],
            traj_stats["entropy_mean"] + traj_stats["entropy_std"],
            alpha=0.2, color=color
        )
    
    ax1.set_xlabel("Query Step")
    ax1.set_ylabel("Posterior Entropy (Persona-based)")
    ax1.grid(True, alpha=0.3)
    
    # Set x-axis to use integer ticks
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot CAT methods on secondary axis if present
    if cat_methods:
        if has_both:
            ax2 = ax1.twinx()
        else:
            ax2 = ax1
            ax2.set_ylabel("Posterior Std (θ)")
        
        for method_name, mr in cat_methods.items():
            # Compute theta_std trajectory stats
            step_values = {}
            for user_result in mr.user_results:
                for step_info in user_result.trajectory:
                    step = step_info.get("step", 0)
                    theta_std = step_info.get("theta_std")
                    if theta_std is not None:
                        if step not in step_values:
                            step_values[step] = []
                        step_values[step].append(theta_std)
            
            if step_values:
                steps = sorted(step_values.keys())
                means = [np.mean(step_values[s]) for s in steps]
                stds = [np.std(step_values[s]) for s in steps]
                
                color = color_map[method_name]
                linestyle = "--" if has_both else "-"
                ax2.plot(steps, means, marker="s", label=f"{method_name}", 
                        color=color, linestyle=linestyle)
                ax2.fill_between(
                    steps,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=color
                )
        
        if has_both:
            ax2.set_ylabel("Posterior Std θ (CAT)", color="gray")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if cat_methods and has_both:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend()
    
    ax1.set_title("Uncertainty Reduction Over Query Steps")
    plt.tight_layout()
    plt.savefig(output_path / "entropy_trajectory.pdf", bbox_inches="tight")
    plt.close()


def plot_question_frequency(
    method_result: MethodResult,
    output_path: Path,
    top_n: int = 20,
):
    """
    Plot most frequently asked questions.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    freq_df = compute_question_frequency(method_result)
    if freq_df.empty:
        return
    
    freq_df = freq_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(freq_df))
    ax.bar(x, freq_df["count"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(freq_df["question"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Question")
    ax.set_ylabel("Times Asked")
    ax.set_title(f"Top {top_n} Most Frequently Asked Questions ({method_result.method_name})")
    plt.tight_layout()
    plt.savefig(
        output_path / f"question_frequency_{method_result.method_name}.pdf",
        bbox_inches="tight"
    )
    plt.close()


def plot_accuracy_by_n_questions(
    experiment: ExperimentResult,
    output_path: Path,
):
    """
    Plot accuracy comparison across methods.
    
    Creates two plots:
    1. Bar chart comparing mean accuracy (±std) across all methods
    2. Box plot showing accuracy distribution by method
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    # Collect data for all methods
    method_data = {}
    for method_name, mr in experiment.method_results.items():
        accuracies = []
        for user_result in mr.user_results:
            acc = user_result.metrics.get("accuracy")
            if acc is not None and not np.isnan(acc):
                accuracies.append(acc)
        if accuracies:
            method_data[method_name] = accuracies
    
    if not method_data:
        return
    
    # Plot 1: Bar chart with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(method_data.keys())
    means = [np.mean(method_data[m]) for m in methods]
    stds = [np.std(method_data[m]) for m in methods]
    
    ax1 = axes[0]
    x = np.arange(len(methods))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods))))
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Mean Accuracy by Method (Budget = {experiment.budget})")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax1.annotate(f'{mean:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, mean + std + 0.02),
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Box plot
    ax2 = axes[1]
    if SEABORN_AVAILABLE:
        data_for_plot = []
        for method in methods:
            for acc in method_data[method]:
                data_for_plot.append({"method": method, "accuracy": acc})
        df = pd.DataFrame(data_for_plot)
        sns.boxplot(data=df, x="method", y="accuracy", ax=ax2)
    else:
        data_by_method = [method_data[m] for m in methods]
        ax2.boxplot(data_by_method, labels=methods)
    
    # Rotate tick labels - need to get current ticks first
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Distribution by Method")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_comparison.pdf", bbox_inches="tight")
    plt.close()
    
    # Also create a simple single bar chart for quick viewing
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                  color=plt.cm.tab10(np.linspace(0, 1, len(methods))))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Mean Accuracy by Method (Budget = {experiment.budget})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.3f}', 
                   xy=(bar.get_x() + bar.get_width() / 2, mean + 0.02),
                   ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_by_method.pdf", bbox_inches="tight")
    plt.close()


def plot_objective_trajectory(
    experiment: ExperimentResult,
    output_path: Path,
):
    """
    Plot method-specific metrics over query steps.
    
    Shows different metrics depending on method type:
    - Greedy: objective_value (entropy/variance over targets)
    - CAT: criterion_value (Fisher info or expected posterior variance)
    - Random/Nonadaptive: posterior_entropy only (no method-specific objective)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment.method_results)))
    
    has_data = False
    
    for (method_name, mr), color in zip(experiment.method_results.items(), colors):
        # Try different metric keys depending on method
        metric_keys = ["objective_value", "criterion_value", "q_value"]
        
        step_values = {}  # step -> list of values
        used_metric = None
        
        for user_result in mr.user_results:
            for traj_step in user_result.trajectory:
                step = traj_step.get("step", 0)
                
                # Try each metric key
                for metric_key in metric_keys:
                    val = traj_step.get(metric_key)
                    if val is not None:
                        used_metric = metric_key
                        if step not in step_values:
                            step_values[step] = []
                        step_values[step].append(val)
                        break
        
        if not step_values:
            continue
        
        has_data = True
        steps = sorted(step_values.keys())
        means = [np.mean(step_values[s]) for s in steps]
        stds = [np.std(step_values[s]) for s in steps]
        
        # Customize label based on metric used
        if used_metric == "objective_value":
            label = f"{method_name} (obj)"
        elif used_metric == "criterion_value":
            label = f"{method_name} (criterion)"
        elif used_metric == "q_value":
            label = f"{method_name} (Q)"
        else:
            label = method_name
        
        ax.plot(steps, means, marker="o", label=label, color=color)
        ax.fill_between(
            steps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color=color
        )
    
    if not has_data:
        plt.close()
        return
    
    ax.set_xlabel("Query Step")
    ax.set_ylabel("Metric Value")
    ax.set_title("Method-Specific Metrics Over Query Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to use integer ticks
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_path / "objective_trajectory.pdf", bbox_inches="tight")
    plt.close()


# =============================================================================
# Output Management
# =============================================================================

class ExperimentOutputManager:
    """
    Manages organized output for experiments.
    
    Creates directory structure:
    output/
    └── {experiment_id}/
        ├── config.yaml
        ├── summary.csv
        ├── detailed/
        │   ├── {method}_user_results.csv
        │   ├── {method}_trajectories.json
        │   └── ...
        ├── analysis/
        │   ├── question_frequency_{method}.csv
        │   ├── trajectory_stats_{method}.csv
        │   └── ...
        └── figures/
            ├── final/                      # End-of-budget performance
            │   ├── metrics_comparison.pdf
            │   └── accuracy_comparison.pdf
            ├── by_budget/                  # Performance vs budget
            │   ├── all_metrics.pdf
            │   ├── accuracy.pdf
            │   └── ...
            ├── question_frequency/         # Question selection patterns
            │   ├── greedy.pdf
            │   ├── random.pdf
            │   └── ...
            ├── entropy_trajectory.pdf
            └── objective_trajectory.pdf
    """
    
    def __init__(self, output_dir: Path, experiment_id: str = None):
        """
        Initialize output manager.
        
        Parameters
        ----------
        output_dir : Path
            Base output directory
        experiment_id : str, optional
            Unique experiment identifier. If None, generates from timestamp.
        """
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_id = experiment_id
        self.base_dir = output_dir / experiment_id
        self.detailed_dir = self.base_dir / "detailed"
        self.analysis_dir = self.base_dir / "analysis"
        self.figures_dir = self.base_dir / "figures"
        self.figures_final_dir = self.figures_dir / "final"
        self.figures_by_budget_dir = self.figures_dir / "by_budget"
        self.figures_question_freq_dir = self.figures_dir / "question_frequency"
        
        # Create directories
        for d in [self.base_dir, self.detailed_dir, self.analysis_dir, 
                  self.figures_dir, self.figures_final_dir, self.figures_by_budget_dir,
                  self.figures_question_freq_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        import yaml
        with open(self.base_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def save_experiment_info(self, experiment: ExperimentResult):
        """Save experiment metadata."""
        info = {
            "experiment_id": experiment.experiment_id,
            "timestamp": experiment.timestamp,
            "dataset_name": experiment.dataset_name,
            "budget": experiment.budget,
            "n_train_users": experiment.n_train_users,
            "n_test_users": experiment.n_test_users,
            "n_feasible_questions": experiment.n_feasible_questions,
            "n_target_questions": experiment.n_target_questions,
            "methods": list(experiment.method_results.keys()),
            # One-shot fit times shared across all persona-based methods.
            # `None` means the corresponding step was not enabled for this run.
            "empirical_bayes_fit_time_seconds": experiment.empirical_bayes_fit_time_seconds,
            "clustering_fit_time_seconds": experiment.clustering_fit_time_seconds,
        }
        with open(self.base_dir / "experiment_info.json", "w") as f:
            json.dump(info, f, indent=2)
    
    def save_summary(self, experiment: ExperimentResult):
        """Save summary statistics for all methods."""
        rows = []
        for method_name, mr in experiment.method_results.items():
            row = {"method": method_name}
            row.update(mr.summary_metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.base_dir / "summary.csv", index=False)
        
        # Also save as formatted text
        with open(self.base_dir / "summary.txt", "w") as f:
            f.write(f"Experiment: {experiment.experiment_id}\n")
            f.write(f"Dataset: {experiment.dataset_name}\n")
            f.write(f"Budget: {experiment.budget}\n")
            f.write(f"Test Users: {experiment.n_test_users}\n")
            f.write(f"Target Questions: {experiment.n_target_questions}\n")
            f.write("\n" + "="*70 + "\n")
            f.write(f"{'Method':<15} {'Accuracy':>12} {'Brier':>12} {'LogLoss':>12}\n")
            f.write("-"*70 + "\n")
            for method_name, mr in experiment.method_results.items():
                acc = mr.summary_metrics.get("accuracy_mean", 0)
                brier = mr.summary_metrics.get("brier_score_mean", 0)
                ll = mr.summary_metrics.get("log_loss_mean", 0)
                f.write(f"{method_name:<15} {acc:>12.4f} {brier:>12.4f} {ll:>12.4f}\n")
    
    def save_method_detailed_results(self, method_result: MethodResult):
        """Save detailed per-user results for a method."""
        method_name = method_result.method_name
        
        # User-level metrics
        user_rows = []
        for ur in method_result.user_results:
            row = {
                "user_id": ur.user_id,
                "n_questions_asked": ur.n_questions_asked,
                "n_target_evaluated": ur.n_target_evaluated,
            }
            row.update(ur.metrics)
            user_rows.append(row)
        
        user_df = pd.DataFrame(user_rows)
        user_df.to_csv(self.detailed_dir / f"{method_name}_user_metrics.csv", index=False)
        
        # Question sequences
        sequences = []
        for ur in method_result.user_results:
            sequences.append({
                "user_id": ur.user_id,
                "questions": ur.asked_questions,
                "answers": ur.observed_answers,
            })
        with open(self.detailed_dir / f"{method_name}_sequences.json", "w") as f:
            json.dump(sequences, f, indent=2)
        
        # Predictions
        predictions = []
        for ur in method_result.user_results:
            for q, pred in ur.predicted_distributions.items():
                true_resp = ur.true_responses.get(q)
                predictions.append({
                    "user_id": ur.user_id,
                    "question": q,
                    "prediction": pred,
                    "true_response": true_resp,
                    "predicted_class": int(np.argmax(pred)),
                    "correct": int(np.argmax(pred)) == true_resp if true_resp is not None else None,
                })
        with open(self.detailed_dir / f"{method_name}_predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        
        # Trajectories (if available)
        trajectories = []
        for ur in method_result.user_results:
            if ur.trajectory:
                trajectories.append({
                    "user_id": ur.user_id,
                    "trajectory": ur.trajectory,
                })
        if trajectories:
            with open(self.detailed_dir / f"{method_name}_trajectories.json", "w") as f:
                json.dump(trajectories, f, indent=2)
    
    def save_analysis_tables(self, experiment: ExperimentResult):
        """Save analysis tables for all methods."""
        for method_name, mr in experiment.method_results.items():
            # Question frequency
            freq_df = compute_question_frequency(mr)
            if not freq_df.empty:
                freq_df.to_csv(
                    self.analysis_dir / f"question_frequency_{method_name}.csv",
                    index=False
                )
            
            # Question position stats
            pos_df = compute_question_position_stats(mr)
            if not pos_df.empty:
                pos_df.to_csv(
                    self.analysis_dir / f"question_positions_{method_name}.csv",
                    index=False
                )
            
            # Trajectory stats
            traj_df = compute_trajectory_stats(mr)
            if not traj_df.empty:
                traj_df.to_csv(
                    self.analysis_dir / f"trajectory_stats_{method_name}.csv",
                    index=False
                )
            
            # Per-question accuracy
            q_acc_df = compute_per_question_accuracy(mr)
            if not q_acc_df.empty:
                q_acc_df.to_csv(
                    self.analysis_dir / f"per_question_accuracy_{method_name}.csv",
                    index=False
                )
        
        # Cross-method comparison table
        comparison_rows = []
        for method_name, mr in experiment.method_results.items():
            row = {"method": method_name}
            for metric in ["accuracy", "brier_score", "log_loss"]:
                row[f"{metric}_mean"] = mr.summary_metrics.get(f"{metric}_mean")
                row[f"{metric}_std"] = mr.summary_metrics.get(f"{metric}_std")
            row["n_users"] = mr.summary_metrics.get("n_users")
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(self.analysis_dir / "method_comparison.csv", index=False)
    
    def save_figures(self, experiment: ExperimentResult):
        """
        Generate and save all figures.
        
        Organization:
        - figures/final/: End-of-budget performance comparison (metrics_comparison, distributions)
        - figures/by_budget/: Performance vs budget plots (generated separately)
        - figures/question_frequency/: Question selection patterns per method
        - figures/: Trajectory plots
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping figures")
            return
        
        # End-of-budget performance plots go to figures/final/
        plot_metrics_comparison(experiment, self.figures_final_dir)
        
        # Metrics distributions (also end-of-budget) go to figures/final/
        for metric in ["accuracy", "brier_score", "log_loss", "mse", "ci_coverage"]:
            plot_metrics_distribution(experiment, self.figures_final_dir, metric)
        
        # Accuracy comparison across methods (end-of-budget) goes to figures/final/
        plot_accuracy_by_n_questions(experiment, self.figures_final_dir)
        
        # Trajectory plots stay in main figures/ folder
        plot_entropy_trajectory(experiment, self.figures_dir)
        plot_objective_trajectory(experiment, self.figures_dir)
        
        # Question frequency for each method goes to figures/question_frequency/
        for method_name, mr in experiment.method_results.items():
            plot_question_frequency(mr, self.figures_question_freq_dir)
    
    def save_all(self, experiment: ExperimentResult, config: Dict[str, Any]):
        """Save everything."""
        print(f"\nSaving results to: {self.base_dir}")
        
        # Config
        self.save_config(config)
        print(f"  Saved: config.yaml")
        
        # Experiment info
        self.save_experiment_info(experiment)
        print(f"  Saved: experiment_info.json")
        
        # Summary
        self.save_summary(experiment)
        print(f"  Saved: summary.csv, summary.txt")
        
        # Detailed results for each method
        for method_name, mr in experiment.method_results.items():
            self.save_method_detailed_results(mr)
            print(f"  Saved: detailed/{method_name}_*.csv/json")
        
        # Analysis tables
        self.save_analysis_tables(experiment)
        print(f"  Saved: analysis/*.csv")
        
        # Figures
        self.save_figures(experiment)
        print(f"  Saved: figures/final/*.pdf, figures/question_frequency/*.pdf, figures/*.pdf")
