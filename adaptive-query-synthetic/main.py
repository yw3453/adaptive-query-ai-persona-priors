"""
Main Script for Running Synthetic Adaptive Query Experiments
============================================================

This script generates synthetic users from persona distributions and evaluates
adaptive query methods under controlled conditions where ground truth is known.

Purpose:
-------
Synthetic experiments allow us to:
1. Verify method correctness when ground truth personas are known
2. Study performance under different data generation models
3. Test robustness to mixture weights and missing data
4. Evaluate persona recovery accuracy

Synthetic User Generation Modes:
-------------------------------
1. pure_uniform: Each user is exactly one persona, sampled uniformly
2. pure_weighted: Each user is one persona, sampled from learned prior
3. mixture_sparse: Each user is a mixture of K personas with Dirichlet weights
4. mixture_dirichlet: Each user is a mixture over ALL personas

Missing Data Models:
-------------------
- none: All questions answered
- rate: Each question missing with probability `rate`
- fixed_count: Exactly N questions answered per user

Ground Truth Tracking:
---------------------
For each synthetic user, we record:
- True persona index (for pure modes)
- Mixture component indices and weights (for mixture modes)
This enables evaluation of persona recovery in addition to prediction accuracy.

Pipeline Overview:
-----------------
1. Load Configuration and source dataset (for personas)
2. Learn Empirical Bayes prior from real users (optional)
3. Generate synthetic users based on configuration
4. Train/Test split of synthetic users
5. Persona Clustering (optional)
6. Run all selected methods
7. Evaluate predictions AND persona recovery
8. Save results including ground truth

Usage Examples:
--------------
    # Run with default configuration (pure_uniform mode)
    uv run adaptive-query-synthetic/main.py
    
    # Test mixture model
    # (modify config.yaml: generation_mode: "mixture_sparse", n_components: 3)
    uv run adaptive-query-synthetic/main.py
    
    # Evaluate with missing data
    # (modify config.yaml: missing.mode: "rate", missing.rate: 0.2)
    uv run adaptive-query-synthetic/main.py

Output Structure:
----------------
output/{experiment_id}/
├── config.yaml
├── experiment_info.json
├── summary.txt
├── detailed/
│   ├── ground_truth.json    # True personas for all synthetic users
│   ├── greedy.json
│   └── ...
├── analysis/
│   ├── persona_recovery.csv  # Persona identification accuracy
│   └── ...
└── figures/

"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.greedy import (
    greedy_adaptive_query,
    ObjectiveType,
    precompute_persona_data,
    PrecomputedPersonaData,
)

# Check for joblib availability
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.baselines import (
    random_adaptive_query,
    nonadaptive_set_query,
    select_nonadaptive_question_set,
    select_random_fixed_question_set,
    full_query,
)
from src.utils import (
    precompute_persona_data as precompute_utils,
    evaluate_predictions,
    learn_empirical_prior,
)
from src.clustering import cluster_personas, ClusteringResult
from src.data_loading import load_dataset
from src.results import (
    SingleUserResult,
    MethodResult,
    ExperimentResult,
    ExperimentOutputManager,
    collect_detailed_results,
    compute_performance_by_budget,
    plot_performance_by_budget,
)

# =============================================================================
# Synthetic User Generation
# =============================================================================

def generate_synthetic_users(
    persona_responses: pd.DataFrame,
    config: dict,
    prior_weights: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic users from personas.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Persona response distributions.
    config : dict
        Synthetic generation configuration.
    prior_weights : np.ndarray, optional
        Learned prior for pure_weighted mode.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    synthetic_users : pd.DataFrame
        Synthetic user responses (same format as real user responses).
    ground_truth : dict
        Ground truth information about each synthetic user.
    """
    synth_config = config["synthetic"]
    n_users = synth_config["n_users"]
    generation_mode = synth_config["generation_mode"]
    seed = 42
    
    rng = np.random.RandomState(seed)
    
    n_personas = len(persona_responses)
    questions = list(persona_responses.columns)
    n_questions = len(questions)
    n_categories = config["dataset"]["n_categories"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating {n_users} Synthetic Users")
        print(f"  Mode: {generation_mode}")
        print(f"  Personas: {n_personas}")
        print(f"  Questions: {n_questions}")
        print(f"{'='*60}")
    
    # Pre-convert persona responses to numpy array for fast access
    # Shape: (n_personas, n_questions, n_categories)
    persona_probs = np.zeros((n_personas, n_questions, n_categories), dtype=np.float64)
    for p_idx in range(n_personas):
        for q_idx, q in enumerate(questions):
            dist = persona_responses.iloc[p_idx][q]
            if dist is not None:
                persona_probs[p_idx, q_idx, :] = np.array(dist)
    
    # Ground truth storage
    ground_truth = {
        "generation_mode": generation_mode,
        "n_users": n_users,
        "seed": seed,
        "users": {},
    }
    
    # Generate user data - maintain original RNG order for reproducibility
    user_data = {}
    
    for user_idx in tqdm(range(n_users), desc="  Generating users", disable=not verbose):
        user_id = f"synthetic_{user_idx}"
        
        # Determine persona weights based on generation mode
        if generation_mode == "pure_uniform":
            # Sample one persona uniformly
            true_persona_idx = rng.randint(0, n_personas)
            persona_weights = np.zeros(n_personas)
            persona_weights[true_persona_idx] = 1.0
            
            ground_truth["users"][user_id] = {
                "mode": "pure",
                "true_persona_idx": int(true_persona_idx),
                "true_persona_id": persona_responses.index[true_persona_idx],
            }
            
        elif generation_mode == "pure_weighted":
            # Sample one persona from learned prior
            if prior_weights is None:
                prior_weights = np.ones(n_personas) / n_personas
            true_persona_idx = rng.choice(n_personas, p=prior_weights)
            persona_weights = np.zeros(n_personas)
            persona_weights[true_persona_idx] = 1.0
            
            ground_truth["users"][user_id] = {
                "mode": "pure",
                "true_persona_idx": int(true_persona_idx),
                "true_persona_id": persona_responses.index[true_persona_idx],
            }
            
        elif generation_mode == "mixture_sparse":
            # Sample K personas and assign random weights
            n_components = synth_config["mixture"]["n_components"]
            n_components = min(n_components, n_personas)
            
            selected_indices = rng.choice(n_personas, size=n_components, replace=False)
            # Generate random weights using Dirichlet
            alpha = synth_config["mixture"].get("dirichlet_alpha", 1.0)
            raw_weights = rng.dirichlet([alpha] * n_components)
            
            persona_weights = np.zeros(n_personas)
            for i, idx in enumerate(selected_indices):
                persona_weights[idx] = raw_weights[i]
            
            ground_truth["users"][user_id] = {
                "mode": "mixture_sparse",
                "component_indices": [int(i) for i in selected_indices],
                "component_ids": [persona_responses.index[i] for i in selected_indices],
                "component_weights": raw_weights.tolist(),
            }
            
        elif generation_mode == "mixture_dirichlet":
            # Sample weights over all personas from Dirichlet
            alpha = synth_config["mixture"].get("dirichlet_alpha", 0.5)
            persona_weights = rng.dirichlet([alpha] * n_personas)
            
            # Find top contributors for ground truth
            top_k = min(5, n_personas)
            top_indices = np.argsort(persona_weights)[-top_k:][::-1]
            
            ground_truth["users"][user_id] = {
                "mode": "mixture_dirichlet",
                "all_weights": persona_weights.tolist(),
                "top_persona_indices": [int(i) for i in top_indices],
                "top_persona_weights": [float(persona_weights[i]) for i in top_indices],
            }
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
        
        # Compute mixture distributions for all questions at once (vectorized)
        # Shape: (n_questions, n_categories)
        mixture_dists = np.dot(persona_weights, persona_probs.reshape(n_personas, -1)).reshape(n_questions, n_categories)
        
        # Normalize each distribution
        mixture_sums = mixture_dists.sum(axis=1, keepdims=True)
        zero_mask = (mixture_sums < 1e-10).flatten()
        mixture_sums = np.where(mixture_sums < 1e-10, 1.0, mixture_sums)
        mixture_dists = mixture_dists / mixture_sums
        
        # Replace zero distributions with uniform
        mixture_dists[zero_mask] = 1.0 / n_categories
        
        # Sample answers for all questions at once
        user_responses = {}
        for q_idx, q in enumerate(questions):
            answer = rng.choice(n_categories, p=mixture_dists[q_idx])
            user_responses[q] = int(answer)
        
        user_data[user_id] = user_responses
    
    # Apply missing data model
    missing_config = synth_config.get("missing", {})
    missing_mode = missing_config.get("mode", "none")
    
    if missing_mode == "rate":
        missing_rate = missing_config.get("rate", 0.1)
        if verbose:
            print(f"  Applying missing rate: {missing_rate:.1%}")
        
        for user_id in user_data:
            for q in questions:
                if rng.random() < missing_rate:
                    user_data[user_id][q] = -1
                    
    elif missing_mode == "fixed_count":
        fixed_count = missing_config.get("fixed_count", len(questions))
        fixed_count = min(fixed_count, len(questions))
        if verbose:
            print(f"  Keeping {fixed_count} questions per user")
        
        for user_id in user_data:
            # Randomly select which questions to keep
            keep_indices = rng.choice(len(questions), size=fixed_count, replace=False)
            keep_questions = set(questions[i] for i in keep_indices)
            for q in questions:
                if q not in keep_questions:
                    user_data[user_id][q] = -1
    
    # Convert to DataFrame
    synthetic_users = pd.DataFrame(user_data).T
    synthetic_users = synthetic_users[questions]  # Ensure column order
    synthetic_users = synthetic_users.astype(int)
    synthetic_users.index = synthetic_users.index.astype(str)
    synthetic_users.columns = synthetic_users.columns.astype(str)
    
    if verbose:
        # Count non-missing per user
        non_missing = (synthetic_users != -1).sum(axis=1)
        print(f"  Generated: {len(synthetic_users)} users")
        print(f"  Questions per user: mean={non_missing.mean():.1f}, "
              f"min={non_missing.min()}, max={non_missing.max()}")
    
    return synthetic_users, ground_truth


# =============================================================================
# Question Sampling (for computational efficiency with *_target objectives)
# =============================================================================

def sample_questions(
    all_questions: List[str],
    config: dict,
    random_state: np.random.RandomState,
    verbose: bool = False,
) -> List[str]:
    """
    Sample a subset of questions if configured.
    
    This is useful for running entropy_target/variance_target objectives in
    overlapping mode, where computational cost scales as O(Q^2).
    
    Parameters
    ----------
    all_questions : List[str]
        List of all question IDs.
    config : dict
        Configuration dictionary.
    random_state : np.random.RandomState
        Random state for reproducibility.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    List[str]
        Sampled questions (or all questions if sampling is disabled).
    """
    sample_config = config["dataset"].get("sample_questions", {})
    
    if not sample_config.get("enabled", False):
        return all_questions
    
    n_sample = sample_config.get("n", len(all_questions))
    
    if n_sample >= len(all_questions):
        if verbose:
            print(f"  Note: sample_questions.n ({n_sample}) >= total questions ({len(all_questions)}), using all questions")
        return all_questions
    
    # Sample questions
    indices = random_state.choice(len(all_questions), size=n_sample, replace=False)
    sampled = [all_questions[i] for i in sorted(indices)]
    
    if verbose:
        print(f"  Sampled {len(sampled)} questions from {len(all_questions)} total")
    
    return sampled


# =============================================================================
# Question Set Configuration
# =============================================================================

def get_question_sets(
    config: dict,
    all_questions: List[str],
    random_state: np.random.RandomState
) -> Tuple[List[str], List[str]]:
    """
    Get feasible and target question sets based on configuration.
    
    In "disjoint" mode: feasible and target questions are separate.
    In "overlapping" mode: feasible = target = all questions.
    """
    evaluation_mode = config["dataset"].get("evaluation_mode", "disjoint")
    
    if evaluation_mode == "overlapping":
        # All questions are both feasible and target
        feasible_questions = list(all_questions)
        target_questions = list(all_questions)
        return feasible_questions, target_questions
    
    # Disjoint mode: separate feasible and target sets
    target_config = config["dataset"]["target_questions"]
    mode = target_config["mode"]
    
    if mode == "last_n":
        n = target_config["n"]
        target_questions = all_questions[-n:]
        feasible_questions = all_questions[:-n]
    elif mode == "random_n":
        n = target_config["n"]
        indices = random_state.choice(len(all_questions), size=n, replace=False)
        target_questions = [all_questions[i] for i in sorted(indices)]
        feasible_questions = [q for q in all_questions if q not in target_questions]
    elif mode == "explicit":
        target_questions = target_config["explicit_list"]
        feasible_questions = [q for q in all_questions if q not in target_questions]
    else:
        raise ValueError(f"Unknown target question mode: {mode}")
    
    return feasible_questions, target_questions


# =============================================================================
# Train/Test Split
# =============================================================================

def train_test_split(
    user_responses: pd.DataFrame,
    train_ratio: float,
    random_state: np.random.RandomState
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split user responses into train and test sets."""
    n_users = len(user_responses)
    n_train = int(n_users * train_ratio)
    
    indices = random_state.permutation(n_users)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_users = user_responses.iloc[train_indices].copy()
    test_users = user_responses.iloc[test_indices].copy()
    
    return train_users, test_users


# =============================================================================
# Persona Recovery Evaluation (New for Synthetic)
# =============================================================================

def evaluate_persona_recovery(
    method_result: MethodResult,
    ground_truth: Dict[str, Any],
    persona_responses: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate how well the method recovered the true persona(s).
    
    Only applicable for pure_uniform and pure_weighted modes.
    """
    generation_mode = ground_truth.get("generation_mode", "")
    if not generation_mode.startswith("pure"):
        return {}
    
    n_correct = 0
    total_posterior_mass = 0.0
    n_evaluated = 0
    
    for user_result in method_result.user_results:
        user_id = user_result.user_id
        if user_id not in ground_truth["users"]:
            continue
        
        user_gt = ground_truth["users"][user_id]
        true_persona_idx = user_gt.get("true_persona_idx")
        if true_persona_idx is None:
            continue
        
        # Get final posterior from trajectory
        if user_result.trajectory:
            last_step = user_result.trajectory[-1]
            posterior = last_step.get("posterior_weights")
            if posterior is not None:
                posterior = np.array(posterior)
                
                # Check if mode matches true persona
                predicted_idx = np.argmax(posterior)
                if predicted_idx == true_persona_idx:
                    n_correct += 1
                
                # Posterior mass on true persona
                total_posterior_mass += posterior[true_persona_idx]
                n_evaluated += 1
    
    if n_evaluated == 0:
        return {}
    
    return {
        "persona_accuracy": n_correct / n_evaluated,
        "mean_posterior_on_truth": total_posterior_mass / n_evaluated,
        "n_evaluated": n_evaluated,
    }


# =============================================================================
# Method Runners (Same structure as adaptive-query/main.py)
# =============================================================================

def _process_greedy_user(
    user_id: str,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    budget: int,
    objective_type: ObjectiveType,
    precomputed: PrecomputedPersonaData,
    use_optimized: bool,
    prior_weights: np.ndarray = None,
    exclude_targets: bool = True,
    # Posterior sparsification parameters
    sparsify_enabled: bool = False,
    sparsify_method: str = "top_p",
    sparsify_top_k: int = 100,
    sparsify_top_p: float = 0.99,
    sparsify_min_k: int = 10,
    sparsify_burn_in: int = 0,
) -> Tuple[str, dict, pd.Series]:
    """Helper function for parallel greedy user processing."""
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
        use_parallel=False,
        exclude_targets=exclude_targets,
        sparsify_enabled=sparsify_enabled,
        sparsify_method=sparsify_method,
        sparsify_top_k=sparsify_top_k,
        sparsify_top_p=sparsify_top_p,
        sparsify_min_k=sparsify_min_k,
        sparsify_burn_in=sparsify_burn_in,
    )
    return user_id, query_result, user_row


def run_greedy(
    config: dict,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[MethodResult, float]:
    """Run greedy adaptive querying."""
    objective_map = {
        "entropy_persona": ObjectiveType.ENTROPY_PERSONA,
        "variance_persona": ObjectiveType.VARIANCE_PERSONA,
        "entropy_target": ObjectiveType.ENTROPY_TARGET,
        "variance_target": ObjectiveType.VARIANCE_TARGET,
        "crps_target": ObjectiveType.CRPS_TARGET,
    }
    
    objective_str = config["greedy"]["objective_type"]
    objective_type = objective_map[objective_str]
    
    # Warn about slow *_target objectives in overlapping mode with many questions
    if evaluation_mode == "overlapping" and objective_str in ["entropy_target", "variance_target", "crps_target"]:
        n_questions = len(target_questions)
        if verbose and n_questions > 20:
            print(f"  Warning: {objective_str} with {n_questions} questions in overlapping mode may be slow (O(Q^2)).")
            print(f"  Consider using dataset.sample_questions to reduce question count, or switch to *_persona objective.")
    n_jobs = config.get("greedy", {}).get("n_jobs", -1)
    use_optimized = config.get("greedy", {}).get("use_optimized", True)
    
    # Get posterior sparsification config
    sparsify_config = config.get("posterior_sparsification", {})
    sparsify_enabled = sparsify_config.get("enabled", False)
    sparsify_method = sparsify_config.get("method", "top_p")
    sparsify_top_k = sparsify_config.get("top_k", 100)
    sparsify_top_p = sparsify_config.get("top_p", 0.99)
    sparsify_min_k = sparsify_config.get("min_k", 10)
    sparsify_burn_in = sparsify_config.get("burn_in_steps", 0)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Greedy ({objective_str})")
        if sparsify_enabled:
            print(f"  Posterior sparsification: {sparsify_method} "
                  f"(top_p={sparsify_top_p}, top_k={sparsify_top_k}, "
                  f"min_k={sparsify_min_k}, burn_in={sparsify_burn_in})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    if use_optimized:
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    else:
        precomputed = None
    
    user_results = []
    
    exclude_targets = (evaluation_mode != "overlapping")
    
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(test_users) > 1:
        user_data = [(str(uid), row) for uid, row in test_users.iterrows()]
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_process_greedy_user)(
                uid, row, persona_responses, feasible_questions, target_questions,
                config["budget"], objective_type, precomputed, use_optimized,
                prior_weights, exclude_targets,
                sparsify_enabled, sparsify_method, sparsify_top_k,
                sparsify_top_p, sparsify_min_k, sparsify_burn_in
            )
            for uid, row in user_data
        )
        for user_id, query_result, user_row in results:
            user_result = collect_detailed_results(
                query_result, user_id, user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            user_results.append(user_result)
    else:
        iterator = tqdm(test_users.iterrows(), total=len(test_users),
                        desc="  Greedy", disable=not verbose)
        for user_id, user_row in iterator:
            query_result = greedy_adaptive_query(
                user_response_row=user_row,
                persona_responses=persona_responses,
                feasible_questions=feasible_questions,
                target_questions=target_questions,
                budget=config["budget"],
                objective_type=objective_type,
                prior_weights=prior_weights,
                precomputed=precomputed,
                use_optimized=use_optimized,
                use_parallel=True,
                n_jobs=n_jobs,
                exclude_targets=exclude_targets,
                sparsify_enabled=sparsify_enabled,
                sparsify_method=sparsify_method,
                sparsify_top_k=sparsify_top_k,
                sparsify_top_p=sparsify_top_p,
                sparsify_min_k=sparsify_min_k,
                sparsify_burn_in=sparsify_burn_in,
            )
            user_result = collect_detailed_results(
                query_result, str(user_id), user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            user_results.append(user_result)
    
    elapsed_time = time.time() - start_time
    
    method_result = MethodResult(
        method_name="greedy",
        user_results=user_results,
        config={"objective_type": objective_str},
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time


def run_random(
    config: dict,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[MethodResult, float]:
    """Run random baseline."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running Random Baseline")
        print(f"{'='*60}")
    
    start_time = time.time()
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    np.random.seed(42)
    
    user_results = []
    iterator = tqdm(test_users.iterrows(), total=len(test_users),
                    desc="  Random", disable=not verbose)
    
    for user_id, user_row in iterator:
        query_result = random_adaptive_query(
            user_response_row=user_row,
            persona_responses=persona_responses,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            budget=config["budget"],
            prior_weights=prior_weights,
            seed=None,
            precomputed=precomputed,
            use_optimized=True,
            exclude_targets=(evaluation_mode != "overlapping"),
        )
        user_result = collect_detailed_results(
            query_result, str(user_id), user_row, target_questions, temperature,
            evaluation_mode=evaluation_mode
        )
        user_results.append(user_result)
    
    elapsed_time = time.time() - start_time
    
    method_result = MethodResult(
        method_name="random",
        user_results=user_results,
        config={"seed": 42},
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time


def run_random_fixed(
    config: dict,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[MethodResult, float]:
    """
    Run random fixed set baseline.
    
    Selects a fixed set of questions uniformly at random (no optimization),
    then asks the same questions to all users.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Running Random Fixed Set Baseline")
        print(f"{'='*60}")
    
    start_time = time.time()
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    
    # Select questions randomly (same for all users)
    if verbose:
        print("  Selecting random question set...")
    selected_questions = select_random_fixed_question_set(
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        budget=config["budget"],
        seed=42,
        exclude_targets=(evaluation_mode != "overlapping"),
    )
    if verbose:
        print(f"  Selected {len(selected_questions)} questions")
    
    user_results = []
    iterator = tqdm(test_users.iterrows(), total=len(test_users),
                    desc="  RandomFixed", disable=not verbose)
    
    for user_id, user_row in iterator:
        query_result = nonadaptive_set_query(
            user_response_row=user_row,
            persona_responses=persona_responses,
            selected_questions=selected_questions,
            target_questions=target_questions,
            prior_weights=prior_weights,
            precomputed=precomputed,
            use_optimized=True,
        )
        user_result = collect_detailed_results(
            query_result, str(user_id), user_row, target_questions, temperature,
            evaluation_mode=evaluation_mode
        )
        user_results.append(user_result)
    
    elapsed_time = time.time() - start_time
    
    method_result = MethodResult(
        method_name="random_fixed",
        user_results=user_results,
        config={
            "selected_questions": selected_questions,
            "seed": 42,
        },
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time


def run_nonadaptive(
    config: dict,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[MethodResult, float]:
    """Run non-adaptive set baseline."""
    selection_criterion = config["nonadaptive"]["selection_criterion"]
    
    # Warn about slow *_target objectives in overlapping mode with many questions
    if evaluation_mode == "overlapping" and selection_criterion in ["entropy_target", "variance_target"]:
        n_questions = len(target_questions)
        if verbose and n_questions > 20:
            print(f"  Warning: {selection_criterion} with {n_questions} questions in overlapping mode may be slow (O(Q^2)).")
            print(f"  Consider using dataset.sample_questions to reduce question count, or switch to *_persona objective.")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Non-Adaptive Set ({selection_criterion})")
        print(f"{'='*60}")
    
    start_time = time.time()
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    
    selected_questions = select_nonadaptive_question_set(
        persona_responses=persona_responses,
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        budget=config["budget"],
        prior_weights=prior_weights,
        selection_criterion=selection_criterion,
        precomputed=precomputed,
        use_optimized=True,
        exclude_targets=(evaluation_mode != "overlapping"),
    )
    
    if verbose:
        print(f"  Selected {len(selected_questions)} questions")
    
    user_results = []
    iterator = tqdm(test_users.iterrows(), total=len(test_users),
                    desc="  NonAdaptive", disable=not verbose)
    
    for user_id, user_row in iterator:
        query_result = nonadaptive_set_query(
            user_response_row=user_row,
            persona_responses=persona_responses,
            selected_questions=selected_questions,
            target_questions=target_questions,
            prior_weights=prior_weights,
            precomputed=precomputed,
            use_optimized=True,
        )
        user_result = collect_detailed_results(
            query_result, str(user_id), user_row, target_questions, temperature,
            evaluation_mode=evaluation_mode
        )
        user_results.append(user_result)
    
    elapsed_time = time.time() - start_time
    
    method_result = MethodResult(
        method_name="nonadaptive",
        user_results=user_results,
        config={
            "selection_criterion": selection_criterion,
            "selected_questions": selected_questions,
        },
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time


def run_full(
    config: dict,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[MethodResult, float]:
    """Run full method (all feasible questions)."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running Full (All Feasible Questions)")
        print(f"{'='*60}")
    
    start_time = time.time()
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    
    user_results = []
    iterator = tqdm(test_users.iterrows(), total=len(test_users),
                    desc="  Full", disable=not verbose)
    
    for user_id, user_row in iterator:
        query_result = full_query(
            user_response_row=user_row,
            persona_responses=persona_responses,
            feasible_questions=feasible_questions,
            target_questions=target_questions,
            prior_weights=prior_weights,
            precomputed=precomputed,
            use_optimized=True,
            exclude_targets=(evaluation_mode != "overlapping"),
        )
        user_result = collect_detailed_results(
            query_result, str(user_id), user_row, target_questions, temperature,
            evaluation_mode=evaluation_mode
        )
        user_results.append(user_result)
    
    elapsed_time = time.time() - start_time
    
    method_result = MethodResult(
        method_name="full",
        user_results=user_results,
        config={},
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time


def run_cat_model(
    config: dict,
    train_users: pd.DataFrame,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    model_type: str,  # "grm", "gpcm", "mgrm", "mgpcm"
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[Optional[MethodResult], float, Any]:
    """
    Run a CAT baseline model.
    
    Parameters
    ----------
    model_type : str
        One of "grm", "gpcm", "mgrm", "mgpcm"
    """
    cat_config = config["cat"]
    n_categories = config["dataset"]["n_categories"]
    budget = config["budget"]
    all_questions = list(set(feasible_questions) | set(target_questions))
    
    # Import appropriate modules based on model type
    if model_type in ["grm", "gpcm"]:
        # 1D models
        try:
            from src.cat import (
                fit_grm, fit_gpcm,
                cat_adaptive_query, cat_adaptive_query_gpcm,
                CATSelectionCriterion,
            )
        except ImportError as e:
            print(f"Warning: Could not import CAT module: {e}")
            return None, 0.0, None
        
        criterion_map = {
            "mfi": CATSelectionCriterion.MFI,
            "mepv": CATSelectionCriterion.MEPV,
        }
        criterion_str = cat_config.get("criterion_1d", "mepv")
        criterion = criterion_map.get(criterion_str, CATSelectionCriterion.MEPV)
        grid_points = cat_config.get("n_grid_points_1d", 41)
        
    else:
        # MIRT models
        try:
            from src.cat_mirt import (
                fit_mgrm, fit_mgpcm,
                mirt_adaptive_query,
                MIRTSelectionCriterion,
            )
        except ImportError as e:
            print(f"Warning: Could not import CAT MIRT module: {e}")
            return None, 0.0, None
        
        criterion_map = {
            "d_opt": MIRTSelectionCriterion.D_OPTIMALITY,
            "a_opt": MIRTSelectionCriterion.A_OPTIMALITY,
            "kl": MIRTSelectionCriterion.KL_DIVERGENCE,
        }
        criterion_str = cat_config.get("criterion_mirt", "a_opt")
        criterion = criterion_map.get(criterion_str, MIRTSelectionCriterion.A_OPTIMALITY)
        grid_points = cat_config.get("n_grid_points_mirt", 11)
        n_dimensions = cat_config.get("n_dimensions", 2)
    
    model_name = model_type.upper()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running CAT-{model_name} ({criterion_str.upper()})")
        print(f"{'='*60}")
    
    start_time = time.time()
    grid_range = cat_config.get("grid_range", 4.0)
    max_iter = cat_config.get("max_iter", 50)
    tol = cat_config.get("tol", 0.001)
    n_jobs = cat_config.get("n_jobs", -1)
    
    # Fit model
    if verbose:
        print(f"  Fitting {model_name} parameters...")
    
    fit_start = time.time()
    
    if model_type == "grm":
        model_params = fit_grm(
            user_responses=train_users,
            questions=all_questions,
            n_categories=n_categories,
            grid_range=grid_range,
            n_grid_points=grid_points,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        def _process_user(user_id, user_row):
            query_result = cat_adaptive_query(
                user_response_row=user_row,
                grm_params=model_params,
                feasible_questions=feasible_questions,
                target_questions=target_questions,
                budget=budget,
                criterion=criterion,
                grid_range=grid_range,
                n_grid_points=grid_points,
                exclude_targets=(evaluation_mode != "overlapping"),
            )
            return collect_detailed_results(
                query_result, str(user_id), user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            
    elif model_type == "gpcm":
        model_params = fit_gpcm(
            user_responses=train_users,
            questions=all_questions,
            n_categories=n_categories,
            grid_range=grid_range,
            n_grid_points=grid_points,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        def _process_user(user_id, user_row):
            query_result = cat_adaptive_query_gpcm(
                user_response_row=user_row,
                gpcm_params=model_params,
                feasible_questions=feasible_questions,
                target_questions=target_questions,
                budget=budget,
                criterion=criterion,
                grid_range=grid_range,
                n_grid_points=grid_points,
                exclude_targets=(evaluation_mode != "overlapping"),
            )
            return collect_detailed_results(
                query_result, str(user_id), user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            
    elif model_type == "mgrm":
        model_params = fit_mgrm(
            user_responses=train_users,
            questions=all_questions,
            n_categories=n_categories,
            n_dimensions=n_dimensions,
            grid_range=grid_range,
            n_grid_points_per_dim=grid_points,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        def _process_user(user_id, user_row):
            query_result = mirt_adaptive_query(
                user_response_row=user_row,
                model_params=model_params,
                feasible_questions=feasible_questions,
                target_questions=target_questions,
                budget=budget,
                model_type="mgrm",
                criterion=criterion,
                grid_range=grid_range,
                n_grid_points_per_dim=grid_points,
                exclude_targets=(evaluation_mode != "overlapping"),
            )
            return collect_detailed_results(
                query_result, str(user_id), user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            
    elif model_type == "mgpcm":
        model_params = fit_mgpcm(
            user_responses=train_users,
            questions=all_questions,
            n_categories=n_categories,
            n_dimensions=n_dimensions,
            grid_range=grid_range,
            n_grid_points_per_dim=grid_points,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        
        def _process_user(user_id, user_row):
            query_result = mirt_adaptive_query(
                user_response_row=user_row,
                model_params=model_params,
                feasible_questions=feasible_questions,
                target_questions=target_questions,
                budget=budget,
                model_type="mgpcm",
                criterion=criterion,
                grid_range=grid_range,
                n_grid_points_per_dim=grid_points,
                exclude_targets=(evaluation_mode != "overlapping"),
            )
            return collect_detailed_results(
                query_result, str(user_id), user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
    else:
        raise ValueError(f"Unknown CAT model type: {model_type}")
    
    fit_time = time.time() - fit_start
    
    if verbose:
        print(f"  {model_name} fitting completed in {fit_time:.2f} seconds")
    
    # Evaluate on test users
    if JOBLIB_AVAILABLE and len(test_users) > 1:
        user_data = [(uid, row) for uid, row in test_users.iterrows()]
        user_results = Parallel(n_jobs=-1, verbose=10 if verbose else 0)(
            delayed(_process_user)(uid, row) for uid, row in user_data
        )
    else:
        user_results = []
        iterator = tqdm(test_users.iterrows(), total=len(test_users),
                        desc=f"  CAT-{model_name} Eval", disable=not verbose)
        for user_id, user_row in iterator:
            user_results.append(_process_user(user_id, user_row))
    
    elapsed_time = time.time() - start_time
    
    method_name = f"cat_{model_type}"
    method_result = MethodResult(
        method_name=method_name,
        user_results=user_results,
        config={"model_type": model_type, "criterion": criterion_str},
    )
    method_result.compute_summary()
    method_result.summary_metrics["runtime_seconds"] = elapsed_time
    method_result.summary_metrics["fitting_time_seconds"] = fit_time
    
    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
    
    return method_result, elapsed_time, model_params


def run_cat(
    config: dict,
    train_users: pd.DataFrame,
    test_users: pd.DataFrame,
    persona_responses: pd.DataFrame,
    feasible_questions: List[str],
    target_questions: List[str],
    verbose: bool,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> Tuple[Optional[MethodResult], float, Any]:
    """Run CAT baseline (legacy function for backward compatibility)."""
    return run_cat_model(
        config, train_users, test_users, persona_responses,
        feasible_questions, target_questions, verbose,
        model_type="grm",
        temperature=temperature,
        evaluation_mode=evaluation_mode,
    )


# =============================================================================
# Results Summary
# =============================================================================

def print_summary(experiment: ExperimentResult, runtimes: Dict[str, float],
                  persona_recovery: Dict[str, Dict[str, float]]):
    """Print summary of all results to console."""
    print(f"\n{'='*80}")
    print(f"Results Summary (Budget = {experiment.budget})")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Accuracy':>10} {'Brier':>10} {'LogLoss':>10} "
          f"{'PersonaAcc':>12} {'Time (s)':>10}")
    print(f"{'-'*80}")
    
    for method_name, mr in experiment.method_results.items():
        acc = mr.summary_metrics.get("accuracy_mean", 0)
        brier = mr.summary_metrics.get("brier_score_mean", 0)
        ll = mr.summary_metrics.get("log_loss_mean", 0)
        runtime = runtimes.get(method_name, 0)
        
        # Persona recovery (if available)
        pr = persona_recovery.get(method_name, {})
        persona_acc = pr.get("persona_accuracy", float('nan'))
        
        if np.isnan(persona_acc):
            persona_acc_str = "N/A"
        else:
            persona_acc_str = f"{persona_acc:.4f}"
        
        print(f"{method_name:<15} {acc:>10.4f} {brier:>10.4f} {ll:>10.4f} "
              f"{persona_acc_str:>12} {runtime:>10.2f}")
    
    total_runtime = sum(runtimes.values())
    print(f"{'-'*80}")
    print(f"{'Total runtime:':<55} {total_runtime:>10.2f} seconds")
    print(f"{'='*80}")


# =============================================================================
# Main
# =============================================================================

def main(config_path: str = None):
    """Main function to run synthetic adaptive query experiments."""
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    verbose = config["verbose"]
    temperature = config.get("prediction", {}).get("temperature", 1.0)
    evaluation_mode = config["dataset"].get("evaluation_mode", "disjoint")
    
    # Generate experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    synth_mode = config["synthetic"]["generation_mode"]
    experiment_id = f"{config['output']['prefix']}_{synth_mode}_{config['dataset']['name']}_{timestamp}"
    
    if verbose:
        print(f"Experiment ID: {experiment_id}")
        print(f"Configuration: {config_path}")
        print(f"Source Dataset: {config['dataset']['name']}")
        print(f"Synthetic Mode: {synth_mode}")
        print(f"Evaluation Mode: {evaluation_mode}")
        print(f"Budget: {config['budget']}")
    
    # Set random seeds
    dataset_random_state = np.random.RandomState(42)
    
    # Load source dataset (for personas and optionally real users for prior learning)
    data_dir = PROJECT_ROOT / "data"
    if verbose:
        print(f"\nLoading source dataset from: {data_dir}")
    
    real_user_responses, persona_responses = load_dataset(config, data_dir)
    
    if verbose:
        print(f"  Real users: {len(real_user_responses)}")
        print(f"  Personas: {len(persona_responses)}")
        print(f"  Questions: {len(persona_responses.columns)}")
    
    # Sample questions if configured (for computational efficiency with *_target objectives)
    all_questions = list(persona_responses.columns)
    all_questions = sample_questions(all_questions, config, dataset_random_state, verbose)
    
    # Filter dataframes to only include sampled questions
    if len(all_questions) < len(persona_responses.columns):
        real_user_responses = real_user_responses[all_questions]
        persona_responses = persona_responses[all_questions]
        if verbose:
            print(f"  Filtered to {len(all_questions)} sampled questions")
    
    # Get question sets (feasible/target split for disjoint mode, or all for overlapping)
    feasible_questions, target_questions = get_question_sets(
        config, all_questions, dataset_random_state
    )
    
    if verbose:
        print(f"  Feasible questions: {len(feasible_questions)}")
        print(f"  Target questions: {len(target_questions)}")
    
    # Learn empirical Bayes prior from real users (if enabled and needed)
    prior_weights = None
    eb_elapsed = None  # seconds; remains None when EB is disabled
    eb_config = config.get("empirical_bayes", {})
    
    if eb_config.get("enabled", False):
        if verbose:
            print(f"\n{'='*60}")
            print("Learning Empirical Bayes Prior from Real Users to Guide Synthetic User Generation")
            print(f"{'='*60}")
        
        eb_start_time = time.time()
        prior_weights = learn_empirical_prior(
            train_user_responses=real_user_responses,
            persona_responses=persona_responses,
            feasible_questions=feasible_questions,
            max_iter=eb_config.get("max_iter", 100),
            tol=eb_config.get("tol", 1e-4),
            verbose=verbose,
        )
        eb_elapsed = time.time() - eb_start_time
        
        if verbose:
            print(f"  Completed in {eb_elapsed:.2f} seconds")
    
    # Generate synthetic users
    synthetic_users, ground_truth = generate_synthetic_users(
        persona_responses=persona_responses,
        config=config,
        prior_weights=prior_weights,
        verbose=verbose,
    )
    
    # Train/test split of synthetic users
    train_users, test_users = train_test_split(
        synthetic_users,
        config["dataset"]["train_ratio"],
        np.random.RandomState(42)
    )
    
    if verbose:
        print(f"\n  Synthetic train users: {len(train_users)}")
        print(f"  Synthetic test users: {len(test_users)}")
    
    # Determine prior for persona-based methods
    # For synthetic experiments, we may want to use the empirical Bayes prior
    # learned from real data, or a uniform prior
    method_prior = prior_weights if eb_config.get("enabled", False) else None
    
    # Persona Clustering (optional)
    clustering_config = config.get("clustering", {})
    clustering_result = None
    cluster_elapsed = None  # seconds; remains None when clustering is disabled
    active_persona_responses = persona_responses  # Default: use original personas
    active_prior_weights = method_prior
    
    if clustering_config.get("enabled", False):
        if method_prior is None:
            if verbose:
                print("\n  Warning: Clustering requires empirical Bayes prior. Using uniform prior.")
            method_prior = np.ones(len(persona_responses)) / len(persona_responses)
            active_prior_weights = method_prior
        
        cluster_start_time = time.time()
        
        # Use synthetic train users for cluster validation
        clustering_result = cluster_personas(
            persona_responses=persona_responses,
            prior_weights=method_prior,
            train_user_responses=train_users,
            feasible_questions=feasible_questions,
            n_categories=config["dataset"]["n_categories"],
            target_questions=target_questions,
            n_clusters=clustering_config.get("n_clusters"),
            n_clusters_range=tuple(clustering_config.get("n_clusters_range", [10, 100])),
            n_clusters_step=clustering_config.get("n_clusters_step", 10),
            prune_threshold=clustering_config.get("prune_threshold", 0.001),
            min_personas=clustering_config.get("min_personas", 10),
            method=clustering_config.get("method", "weighted_kmeans"),
            assignment=clustering_config.get("assignment", "hard"),
            soft_temperature=clustering_config.get("soft_temperature", 1.0),
            random_state=42,
            verbose=verbose,
        )
        
        cluster_elapsed = time.time() - cluster_start_time
        
        # Use prototypes instead of original personas
        active_persona_responses = clustering_result.prototype_responses
        active_prior_weights = clustering_result.prototype_prior
        
        if verbose:
            print(f"  Clustering completed in {cluster_elapsed:.2f} seconds")
            print(f"  Using {clustering_result.n_prototypes} prototypes instead of {clustering_result.n_original_personas} personas")
    
    # Initialize experiment result
    experiment = ExperimentResult(
        experiment_id=experiment_id,
        timestamp=timestamp,
        dataset_name=f"synthetic_{config['dataset']['name']}",
        budget=config["budget"],
        n_train_users=len(train_users),
        n_test_users=len(test_users),
        n_feasible_questions=len(feasible_questions),
        n_target_questions=len(target_questions),
        feasible_questions=feasible_questions,
        target_questions=target_questions,
        config=config,
        empirical_bayes_fit_time_seconds=eb_elapsed,
        clustering_fit_time_seconds=cluster_elapsed,
    )
    
    # Run selected methods
    methods = config["methods"]
    runtimes = {}
    persona_recovery = {}
    
    if methods.get("random", False):
        result, runtime = run_random(
            config, test_users, active_persona_responses,
            feasible_questions, target_questions, verbose,
            prior_weights=active_prior_weights,
            temperature=temperature,
            evaluation_mode=evaluation_mode
        )
        if result:
            experiment.method_results["random"] = result
            runtimes["random"] = runtime
            persona_recovery["random"] = evaluate_persona_recovery(
                result, ground_truth, persona_responses
            )
    
    if methods.get("random_fixed", False):
        result, runtime = run_random_fixed(
            config, test_users, active_persona_responses,
            feasible_questions, target_questions, verbose,
            prior_weights=active_prior_weights,
            temperature=temperature,
            evaluation_mode=evaluation_mode
        )
        if result:
            experiment.method_results["random_fixed"] = result
            runtimes["random_fixed"] = runtime
            persona_recovery["random_fixed"] = evaluate_persona_recovery(
                result, ground_truth, persona_responses
            )
    
    if methods.get("nonadaptive", False):
        result, runtime = run_nonadaptive(
            config, test_users, active_persona_responses,
            feasible_questions, target_questions, verbose,
            prior_weights=active_prior_weights,
            temperature=temperature,
            evaluation_mode=evaluation_mode
        )
        if result:
            experiment.method_results["nonadaptive"] = result
            runtimes["nonadaptive"] = runtime
            persona_recovery["nonadaptive"] = evaluate_persona_recovery(
                result, ground_truth, persona_responses
            )
    
    if methods.get("greedy", False):
        result, runtime = run_greedy(
            config, test_users, active_persona_responses,
            feasible_questions, target_questions, verbose,
            prior_weights=active_prior_weights,
            temperature=temperature,
            evaluation_mode=evaluation_mode
        )
        if result:
            experiment.method_results["greedy"] = result
            runtimes["greedy"] = runtime
            persona_recovery["greedy"] = evaluate_persona_recovery(
                result, ground_truth, persona_responses
            )
    
    # Track model params for CAT performance-by-budget
    cat_model_params = {}
    
    # Only run CAT methods if methods.cat is true
    if methods.get("cat", False):
        # Run all enabled CAT models from cat.models config
        cat_models_config = config.get("cat", {}).get("models", {})
        any_model_enabled = any(cat_models_config.get(m, False) for m in ["grm", "gpcm", "mgrm", "mgpcm"])
        
        if any_model_enabled:
            # Run specific models enabled in cat.models
            for cat_model_type in ["grm", "gpcm", "mgrm", "mgpcm"]:
                if cat_models_config.get(cat_model_type, False):
                    result, runtime, model_params = run_cat_model(
                        config, train_users, test_users, persona_responses,
                        feasible_questions, target_questions, verbose,
                        model_type=cat_model_type,
                        temperature=temperature,
                        evaluation_mode=evaluation_mode
                    )
                    if result:
                        method_name = f"cat_{cat_model_type}"
                        experiment.method_results[method_name] = result
                        runtimes[method_name] = runtime
                        cat_model_params[cat_model_type] = model_params
        else:
            # Legacy support: if methods.cat is true but no specific models enabled, run GRM
            result, runtime, model_params = run_cat(
                config, train_users, test_users, persona_responses,
                feasible_questions, target_questions, verbose,
                temperature=temperature,
                evaluation_mode=evaluation_mode
            )
            if result:
                experiment.method_results["cat"] = result
                runtimes["cat"] = runtime
                cat_model_params["grm"] = model_params
    
    if methods.get("full", False):
        result, runtime = run_full(
            config, test_users, active_persona_responses,
            feasible_questions, target_questions, verbose,
            prior_weights=active_prior_weights,
            temperature=temperature,
            evaluation_mode=evaluation_mode
        )
        if result:
            experiment.method_results["full"] = result
            runtimes["full"] = runtime
            persona_recovery["full"] = evaluate_persona_recovery(
                result, ground_truth, persona_responses
            )
    
    # Print summary to console
    print_summary(experiment, runtimes, persona_recovery)
    
    # Save all results
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_manager = ExperimentOutputManager(output_dir, experiment_id)
    output_manager.save_all(experiment, config)
    
    # Save ground truth
    if config["synthetic"].get("save_ground_truth", True):
        ground_truth_path = output_manager.detailed_dir / "ground_truth.json"
        with open(ground_truth_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
        if verbose:
            print(f"  Saved: detailed/ground_truth.json")
    
    # Save persona recovery metrics
    if persona_recovery:
        recovery_path = output_manager.analysis_dir / "persona_recovery.csv"
        recovery_rows = []
        for method_name, metrics in persona_recovery.items():
            if metrics:
                recovery_rows.append({
                    "method": method_name,
                    **metrics
                })
        if recovery_rows:
            pd.DataFrame(recovery_rows).to_csv(recovery_path, index=False)
            if verbose:
                print(f"  Saved: analysis/persona_recovery.csv")
    
    # Compute and save performance by budget analysis
    if verbose:
        print("\nComputing performance by budget...")
    
    ci_confidence_level = config.get("evaluation", {}).get("ci_confidence_level", 0.95)
    # Get GRM params if available (for CAT by-budget analysis)
    grm_params = cat_model_params.get("grm") if cat_model_params else None
    
    # Cap by-budget evaluation at the largest budget that can actually be reached:
    # the trajectory cannot ask more questions than the feasible pool. Any larger
    # budget would just repeat the final value and inflate runtime + CSV rows.
    effective_max_budget = min(int(config["budget"]), len(feasible_questions))
    if verbose and effective_max_budget < int(config["budget"]):
        print(f"  Capping by-budget evaluation at {effective_max_budget} "
              f"(config budget = {config['budget']}, feasible questions = {len(feasible_questions)})")
    
    performance_df = compute_performance_by_budget(
        experiment=experiment,
        persona_responses=active_persona_responses,
        user_responses=test_users,
        max_budget=effective_max_budget,
        grm_params=grm_params,
        n_jobs=config.get("greedy", {}).get("n_jobs", -1),
        prior_weights=active_prior_weights,
        temperature=temperature,
        evaluation_mode=evaluation_mode,
        ci_confidence_level=ci_confidence_level,
        cat_model_params=cat_model_params,
    )
    
    if not performance_df.empty:
        plot_performance_by_budget(
            performance_df,
            figures_path=output_manager.figures_by_budget_dir,
            analysis_path=output_manager.analysis_dir,
        )
        if verbose:
            print(f"  Saved: figures/by_budget/*.pdf, analysis/performance_by_budget.csv")
    
    # Final completion message
    print("\n" + "=" * 70)
    print("🎉  ALL DONE! Synthetic experiment completed successfully!  🎉")
    print(f"    Output saved to: {output_manager.base_dir}")
    print("=" * 70 + "\n")
    
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synthetic adaptive query experiments"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    main(args.config)
