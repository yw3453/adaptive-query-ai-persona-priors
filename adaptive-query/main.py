"""
Main Script for Running Adaptive Query Experiments
===================================================

This script orchestrates the full experimental pipeline for evaluating 
adaptive query methods on real user data.

Pipeline Overview:
-----------------
1. Load Configuration: Read experiment settings from config.yaml
2. Load Dataset: Load user responses and persona response distributions
3. Train/Test Split: Divide users into training and test sets
4. Empirical Bayes Prior Learning: Learn prior over personas from training data
5. Persona Clustering (optional): Reduce persona dimensionality
6. Run Methods: Execute selected query methods (greedy, random, CAT, etc.)
7. Evaluate: Compute accuracy, Brier score, log loss on target questions
8. Save Results: Export detailed results, analysis tables, and figures

Supported Methods:
-----------------
- Greedy: Myopic optimization of expected posterior cost
- Random: Uniform random question selection baseline
- Non-adaptive: Fixed question set selected offline
- Full: Uses all available feasible questions
- CAT: Computerized Adaptive Testing with GRM model

Configuration:
-------------
All experiment parameters are specified in config.yaml, including:
- Dataset selection (WorldValuesBench)
- Budget (max questions per user)
- Method selection and hyperparameters
- Empirical Bayes and clustering settings
- Output directory and format

Usage Examples:
--------------
    # Run with default configuration
    uv run adaptive-query/main.py
    
    # Run with custom configuration
    uv run adaptive-query/main.py --config experiments/config_v2.yaml
    
    # Quick test with small budget
    # (modify config.yaml: budget: 5, methods: {greedy: true, random: true})
    uv run adaptive-query/main.py

Output Structure:
----------------
output/{experiment_id}/
├── config.yaml              # Copy of experiment configuration
├── experiment_info.json     # Metadata (timestamps, counts)
├── summary.txt              # Human-readable results summary
├── summary.csv              # Method comparison table
├── detailed/                # Per-user results by method
│   ├── greedy.json
│   ├── random.json
│   └── ...
├── analysis/                # Analysis tables
│   ├── question_frequency_*.csv
│   ├── per_question_accuracy_*.csv
│   └── performance_by_budget.csv
└── figures/                 # Visualization PDFs
    ├── metrics_comparison.pdf
    ├── performance_by_budget.pdf
    └── ...

"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
from src.utils import precompute_persona_data as precompute_utils
from src.utils import evaluate_predictions, learn_empirical_prior
from src.clustering import cluster_personas, ClusteringResult
from src.data_loading import load_dataset
from src.results import (
    SingleUserResult,
    MethodResult,
    ExperimentResult,
    ExperimentOutputManager,
    collect_detailed_results,
)

# =============================================================================
# Question Sampling (for computational efficiency with *_target objectives)
# =============================================================================

def sample_questions(
    all_questions: list[str],
    config: dict,
    random_state: np.random.RandomState,
    verbose: bool = False,
) -> list[str]:
    """
    Sample a subset of questions if configured.
    
    This is useful for running entropy_target/variance_target objectives in
    overlapping mode, where computational cost scales as O(Q^2).
    
    Parameters
    ----------
    all_questions : list[str]
        List of all question IDs.
    config : dict
        Configuration dictionary.
    random_state : np.random.RandomState
        Random state for reproducibility.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    list[str]
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
    all_questions: list[str],
    random_state: np.random.RandomState
) -> tuple[list[str], list[str]]:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
# Method Runners (with detailed result collection)
# =============================================================================

def _process_greedy_user(
    user_id: str,
    user_row: pd.Series,
    persona_responses: pd.DataFrame,
    feasible_questions: list[str],
    target_questions: list[str],
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
) -> tuple[str, dict, pd.Series]:
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
        use_parallel=False,  # Don't nest parallelism
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult, float]:
    """Run greedy adaptive querying with detailed result collection."""
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
    
    # Get parallelization settings
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
        print(f"  Optimized: {use_optimized}, n_jobs: {n_jobs}")
        if sparsify_enabled:
            print(f"  Posterior sparsification: {sparsify_method} "
                  f"(top_p={sparsify_top_p}, top_k={sparsify_top_k}, "
                  f"min_k={sparsify_min_k}, burn_in={sparsify_burn_in})")
        if prior_weights is not None:
            eff_n = 1.0 / np.sum(prior_weights ** 2)
            print(f"  Using empirical Bayes prior (effective N = {eff_n:.1f})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pre-compute persona data once
    if use_optimized:
        if verbose:
            print("  Pre-computing persona data...")
        precomputed = precompute_persona_data(
            persona_responses, feasible_questions, target_questions
        )
    else:
        precomputed = None
    
    user_results = []
    
    # Parallel processing with joblib
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(test_users) > 1:
        if verbose:
            print(f"  Processing {len(test_users)} users in parallel...")
        
        # Prepare arguments
        user_data = [(str(uid), row) for uid, row in test_users.iterrows()]
        
        exclude_targets = (evaluation_mode != "overlapping")
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
        
        # Collect results
        for user_id, query_result, user_row in results:
            user_result = collect_detailed_results(
                query_result, user_id, user_row, target_questions, temperature,
                evaluation_mode=evaluation_mode
            )
            user_results.append(user_result)
    else:
        # Sequential processing with progress bar
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
                use_parallel=True,  # Parallelize candidates within each user
                n_jobs=n_jobs,
                exclude_targets=(evaluation_mode != "overlapping"),
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
        config={
            "objective_type": objective_str,
            "use_optimized": use_optimized,
            "n_jobs": n_jobs,
            "uses_empirical_bayes": prior_weights is not None,
        },
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult, float]:
    """Run random baseline with detailed result collection."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running Random Baseline")
        if prior_weights is not None:
            eff_n = 1.0 / np.sum(prior_weights ** 2)
            print(f"  Using empirical Bayes prior (effective N = {eff_n:.1f})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pre-compute persona data for optimization
    if verbose:
        print("  Pre-computing persona data...")
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    
    # Set seed for reproducibility across users
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
            seed=None,  # Let master seed control
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
        config={
            "seed": 42,
            "uses_empirical_bayes": prior_weights is not None,
        },
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult, float]:
    """
    Run random fixed set baseline with detailed result collection.
    
    Selects a fixed set of questions uniformly at random (no optimization),
    then asks the same questions to all users. This is a simpler baseline
    than the non-adaptive set which uses greedy entropy selection.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Running Random Fixed Set Baseline")
        if prior_weights is not None:
            eff_n = 1.0 / np.sum(prior_weights ** 2)
            print(f"  Using empirical Bayes prior (effective N = {eff_n:.1f})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pre-compute persona data for optimization
    if verbose:
        print("  Pre-computing persona data...")
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
        # Reuse nonadaptive_set_query since it just takes a pre-selected list
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
            "uses_empirical_bayes": prior_weights is not None,
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult, float]:
    """Run non-adaptive set baseline with detailed result collection."""
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
        if prior_weights is not None:
            eff_n = 1.0 / np.sum(prior_weights ** 2)
            print(f"  Using empirical Bayes prior (effective N = {eff_n:.1f})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pre-compute persona data for optimization
    if verbose:
        print("  Pre-computing persona data...")
    precomputed = precompute_utils(persona_responses, feasible_questions, target_questions)
    
    if verbose:
        print("  Selecting questions offline...")
    
    # Select questions offline with optimization
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
            "uses_empirical_bayes": prior_weights is not None,
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    prior_weights: np.ndarray = None,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult, float]:
    """Run full method (all feasible questions) with detailed result collection."""
    if verbose:
        print(f"\n{'='*60}")
        print("Running Full (All Feasible Questions)")
        if prior_weights is not None:
            eff_n = 1.0 / np.sum(prior_weights ** 2)
            print(f"  Using empirical Bayes prior (effective N = {eff_n:.1f})")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pre-compute persona data for optimization
    if verbose:
        print("  Pre-computing persona data...")
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
        config={
            "uses_empirical_bayes": prior_weights is not None,
        },
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    model_type: str,  # "grm", "gpcm", "mgrm", "mgpcm"
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult | None, float, Any]:
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
    feasible_questions: list[str],
    target_questions: list[str],
    verbose: bool,
    temperature: float = 1.0,
    evaluation_mode: str = "disjoint",
) -> tuple[MethodResult | None, float, Any]:
    """Run CAT baseline (legacy function for backward compatibility)."""
    return run_cat_model(
        config, train_users, test_users, persona_responses,
        feasible_questions, target_questions, verbose,
        model_type="grm",
        temperature=temperature,
        evaluation_mode=evaluation_mode,
    )


# =============================================================================
# Results Summary (Console Output)
# =============================================================================

def print_summary(experiment: ExperimentResult, runtimes: dict[str, float]):
    """Print summary of all results to console."""
    print(f"\n{'='*80}")
    print(f"Results Summary (Budget = {experiment.budget})")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Accuracy':>10} {'Brier':>10} {'LogLoss':>10} {'N Users':>8} {'Time (s)':>10}")
    print(f"{'-'*80}")
    
    for method_name, mr in experiment.method_results.items():
        acc = mr.summary_metrics.get("accuracy_mean", 0)
        brier = mr.summary_metrics.get("brier_score_mean", 0)
        ll = mr.summary_metrics.get("log_loss_mean", 0)
        n_users = mr.summary_metrics.get("n_users", 0)
        runtime = runtimes.get(method_name, 0)
        print(f"{method_name:<15} {acc:>10.4f} {brier:>10.4f} {ll:>10.4f} {n_users:>8} {runtime:>10.2f}")
    
    total_runtime = sum(runtimes.values())
    print(f"{'-'*80}")
    print(f"{'Total runtime:':<55} {total_runtime:>10.2f} seconds")
    print(f"{'='*80}")


# =============================================================================
# Main
# =============================================================================

def main(config_path: str = None):
    """Main function to run adaptive query experiments."""
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
    experiment_id = f"{config['output']['prefix']}_{config['dataset']['name']}_{timestamp}"
    
    if verbose:
        print(f"Experiment ID: {experiment_id}")
        print(f"Loaded configuration from: {config_path}")
        print(f"Dataset: {config['dataset']['name']}")
        print(f"Evaluation Mode: {evaluation_mode}")
        print(f"Budget: {config['budget']}")
    
    # Set random seed
    random_state = np.random.RandomState(42)
    
    # Load dataset
    data_dir = PROJECT_ROOT / "data"
    if verbose:
        print(f"\nLoading dataset from: {data_dir}")
    
    user_responses, persona_responses = load_dataset(config, data_dir)
    
    if verbose:
        print(f"  Users: {len(user_responses)}")
        print(f"  Personas: {len(persona_responses)}")
        print(f"  Questions: {len(user_responses.columns)}")
    
    # Sample questions if configured (for computational efficiency with *_target objectives)
    all_questions = list(user_responses.columns)
    all_questions = sample_questions(all_questions, config, random_state, verbose)
    
    # Filter dataframes to only include sampled questions
    if len(all_questions) < len(user_responses.columns):
        user_responses = user_responses[all_questions]
        persona_responses = persona_responses[all_questions]
        if verbose:
            print(f"  Filtered to {len(all_questions)} sampled questions")
    
    # Get question sets (feasible/target split for disjoint mode, or all for overlapping)
    feasible_questions, target_questions = get_question_sets(
        config, all_questions, random_state
    )
    
    if verbose:
        print(f"  Feasible questions: {len(feasible_questions)}")
        print(f"  Target questions: {len(target_questions)}")
    
    # Subsample users if max_users is set (BEFORE train-test split)
    max_users = config["dataset"].get("max_users", None)
    if max_users is not None and len(user_responses) > max_users:
        if verbose:
            print(f"  Subsampling users: {len(user_responses)} -> {max_users}")
        user_responses = user_responses.sample(n=max_users, random_state=random_state)
    
    # Train/test split
    train_users, test_users = train_test_split(
        user_responses,
        config["dataset"]["train_ratio"],
        random_state
    )
    
    if verbose:
        print(f"  Train users: {len(train_users)}")
        print(f"  Test users: {len(test_users)}")
    
    # Empirical Bayes prior learning
    prior_weights = None
    eb_elapsed = None  # seconds; remains None when EB is disabled
    eb_config = config.get("empirical_bayes", {})
    
    if eb_config.get("enabled", False):
        if verbose:
            print(f"\n{'='*60}")
            print("Learning Empirical Bayes Prior")
            print(f"{'='*60}")
        
        eb_start_time = time.time()
        prior_weights = learn_empirical_prior(
            train_user_responses=train_users,
            persona_responses=persona_responses,
            feasible_questions=feasible_questions,
            max_iter=eb_config.get("max_iter", 100),
            tol=eb_config.get("tol", 1e-4),
            verbose=verbose,
        )
        eb_elapsed = time.time() - eb_start_time
        
        if verbose:
            print(f"  Completed in {eb_elapsed:.2f} seconds")
    else:
        if verbose:
            print("\n  Empirical Bayes prior learning disabled, using uniform prior")
    
    # Persona Clustering (optional)
    clustering_config = config.get("clustering", {})
    clustering_result = None
    cluster_elapsed = None  # seconds; remains None when clustering is disabled
    active_persona_responses = persona_responses  # Default: use original personas
    active_prior_weights = prior_weights
    
    if clustering_config.get("enabled", False):
        if prior_weights is None:
            if verbose:
                print("\n  Warning: Clustering requires empirical Bayes prior. Using uniform prior.")
            prior_weights = np.ones(len(persona_responses)) / len(persona_responses)
            active_prior_weights = prior_weights
        
        cluster_start_time = time.time()
        
        clustering_result = cluster_personas(
            persona_responses=persona_responses,
            prior_weights=prior_weights,
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
        dataset_name=config["dataset"]["name"],
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
    
    # Print summary to console
    print_summary(experiment, runtimes)
    
    # Save all results
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_manager = ExperimentOutputManager(output_dir, experiment_id)
    output_manager.save_all(experiment, config)
    
    # Compute and save performance by budget analysis
    if verbose:
        print("\nComputing performance by budget...")
    
    from src.results import compute_performance_by_budget, plot_performance_by_budget
    
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
        # Save PDFs to figures/by_budget/, CSV to analysis folder
        plot_performance_by_budget(
            performance_df,
            figures_path=output_manager.figures_by_budget_dir,
            analysis_path=output_manager.analysis_dir,
        )
        if verbose:
            print(f"  Saved: figures/by_budget/*.pdf, analysis/performance_by_budget.csv\n")
    
    # Final completion message
    print("\n" + "=" * 70)
    print("🎉  ALL DONE! Experiment completed successfully!  🎉")
    print(f"    Output saved to: {output_manager.base_dir}")
    print("=" * 70 + "\n")
    
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive query experiments"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: config.yaml in same directory)"
    )
    args = parser.parse_args()
    
    main(args.config)
