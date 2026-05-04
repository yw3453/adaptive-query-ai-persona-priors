"""
Persona Clustering Module
=========================

Clusters personas into prototypical personas to reduce dimensionality.
Uses empirical Bayes prior from training data to guide clustering.

Problem:
-------
With many personas (e.g., 2058), computation is expensive and some personas
may be rarely used. Clustering reduces this to a manageable set of prototypes.

Approach:
--------
1. **Pruning**: Remove personas with prior weight below threshold
   - Uses empirical Bayes prior π(θ) learned from training data
   - Keeps only "active" personas that explain real users well

2. **Clustering**: Group similar personas using weighted K-means
   - Distance: Jensen-Shannon divergence (symmetric KL)
   - Weights: Prior weights (important personas influence clusters more)
   - Uses sqrt-transform to approximate JS with Euclidean distance

3. **Prototype Creation**: Create weighted-average distributions
   - Each prototype is weighted average of its cluster members
   - Weights are normalized prior weights within cluster

4. **Prior Re-normalization**: Sum original prior weights per cluster
   - Prototype k gets prior = Σ_{θ∈cluster_k} π(θ)

Key Classes:
-----------
- ClusteringResult: Dataclass containing all clustering outputs

Key Functions:
-------------
- cluster_personas(): Main entry point for clustering pipeline
- prune_personas_by_prior(): Remove low-weight personas
- weighted_kmeans_js(): K-means with prior weights and JS distance
- create_prototype_personas(): Create averaged prototype distributions
- select_n_clusters(): Auto-select K via training data validation

Example:
-------
```python
from src.clustering import cluster_personas

result = cluster_personas(
    persona_responses=personas,
    prior_weights=prior,
    train_user_responses=train_users,
    feasible_questions=feasible_qs,
    n_categories=10,
    target_questions=target_qs,
    n_clusters=50,
    verbose=True
)

# Use prototypes for faster querying
prototypes = result.prototype_responses
proto_prior = result.prototype_prior
```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from tqdm import tqdm

# Small constant for numerical stability
EPS = 1e-10


@dataclass
class ClusteringResult:
    """Result of persona clustering."""
    
    # Prototype personas (n_prototypes x n_questions DataFrame with list entries)
    prototype_responses: pd.DataFrame
    
    # Prior weights over prototypes (sums to 1)
    prototype_prior: np.ndarray
    
    # Mapping from original persona index to prototype index
    persona_to_prototype: Dict[str, int]
    
    # Soft assignment matrix (n_personas x n_prototypes) if soft assignment used
    soft_assignments: Optional[np.ndarray] = None
    
    # Clustering metadata
    n_original_personas: int = 0
    n_pruned_personas: int = 0
    n_prototypes: int = 0
    
    # Validation scores for different K values (if auto-selected)
    validation_scores: Optional[Dict[int, float]] = None
    selected_k: Optional[int] = None


def compute_js_divergence_matrix(
    persona_responses: pd.DataFrame,
    questions: List[str],
) -> np.ndarray:
    """
    Compute pairwise Jensen-Shannon divergence matrix between personas.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Persona response distributions (each entry is a list of probabilities).
    questions : List[str]
        Questions to use for computing divergence.
    
    Returns
    -------
    js_matrix : np.ndarray of shape (n_personas, n_personas)
        Pairwise JS divergence matrix.
    """
    n_personas = len(persona_responses)
    
    # Flatten each persona's distributions into a single vector
    flattened = []
    for idx in persona_responses.index:
        persona_vec = []
        for q in questions:
            dist = persona_responses.loc[idx, q]
            if dist is not None:
                persona_vec.extend(dist)
            else:
                # Uniform if missing
                k = len(persona_responses.iloc[0][questions[0]])
                persona_vec.extend([1.0 / k] * k)
        flattened.append(np.array(persona_vec))
    
    flattened = np.array(flattened)
    
    # Compute pairwise JS divergence
    js_matrix = np.zeros((n_personas, n_personas))
    for i in range(n_personas):
        for j in range(i + 1, n_personas):
            js = jensenshannon(flattened[i], flattened[j])
            js_matrix[i, j] = js
            js_matrix[j, i] = js
    
    return js_matrix


def prune_personas_by_prior(
    persona_responses: pd.DataFrame,
    prior_weights: np.ndarray,
    threshold: float = 0.001,
    min_personas: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prune personas with low prior weight.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Original persona responses.
    prior_weights : np.ndarray
        Prior weights for each persona.
    threshold : float
        Minimum prior weight to keep a persona.
    min_personas : int
        Minimum number of personas to keep (overrides threshold if needed).
    
    Returns
    -------
    pruned_responses : pd.DataFrame
        Pruned persona responses.
    pruned_weights : np.ndarray
        Prior weights for kept personas (re-normalized).
    kept_indices : List[str]
        Indices of kept personas.
    """
    # Find personas above threshold
    mask = prior_weights >= threshold
    
    # Ensure minimum number of personas
    if mask.sum() < min_personas:
        # Keep top min_personas by prior weight
        top_indices = np.argsort(prior_weights)[-min_personas:]
        mask = np.zeros(len(prior_weights), dtype=bool)
        mask[top_indices] = True
    
    kept_indices = list(persona_responses.index[mask])
    pruned_responses = persona_responses.loc[kept_indices].copy()
    pruned_weights = prior_weights[mask]
    
    # Re-normalize weights
    pruned_weights = pruned_weights / pruned_weights.sum()
    
    return pruned_responses, pruned_weights, kept_indices


def weighted_kmeans_js(
    persona_responses: pd.DataFrame,
    weights: np.ndarray,
    n_clusters: int,
    questions: List[str],
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted K-means clustering using Jensen-Shannon divergence.
    
    Since sklearn's K-means uses Euclidean distance, we:
    1. Flatten distributions
    2. Apply sqrt transform (makes Euclidean ≈ Hellinger ≈ JS)
    3. Run weighted K-means
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Persona response distributions.
    weights : np.ndarray
        Prior weights for each persona.
    n_clusters : int
        Number of clusters.
    questions : List[str]
        Questions to use.
    n_init : int
        Number of K-means initializations.
    max_iter : int
        Maximum iterations per run.
    random_state : int
        Random seed.
    
    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each persona.
    centroids : np.ndarray
        Cluster centroids in transformed space.
    """
    # Flatten and sqrt-transform for approximate JS distance
    flattened = []
    for idx in persona_responses.index:
        persona_vec = []
        for q in questions:
            dist = np.array(persona_responses.loc[idx, q])
            dist = np.maximum(dist, EPS)  # Avoid zeros
            persona_vec.extend(np.sqrt(dist))  # Hellinger transform
        flattened.append(persona_vec)
    
    X = np.array(flattened)
    
    # Weighted K-means using sample_weight
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    
    # Fit with sample weights
    kmeans.fit(X, sample_weight=weights)
    
    return kmeans.labels_, kmeans.cluster_centers_


def hierarchical_clustering_js(
    persona_responses: pd.DataFrame,
    n_clusters: int,
    questions: List[str],
) -> np.ndarray:
    """
    Hierarchical clustering using Jensen-Shannon divergence.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Persona response distributions.
    n_clusters : int
        Number of clusters.
    questions : List[str]
        Questions to use.
    
    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each persona.
    """
    # Compute JS divergence matrix
    js_matrix = compute_js_divergence_matrix(persona_responses, questions)
    
    # Convert to condensed form for scipy
    from scipy.spatial.distance import squareform
    condensed = squareform(js_matrix)
    
    # Hierarchical clustering
    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed
    
    return labels


def create_prototype_personas(
    persona_responses: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    questions: List[str],
    n_categories: int,
) -> pd.DataFrame:
    """
    Create prototype personas by weighted averaging within clusters.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Original persona responses.
    labels : np.ndarray
        Cluster assignment for each persona.
    weights : np.ndarray
        Prior weights for each persona.
    questions : List[str]
        Questions to include.
    n_categories : int
        Number of response categories.
    
    Returns
    -------
    prototypes : pd.DataFrame
        Prototype persona responses (same format as input).
    """
    n_clusters = len(np.unique(labels))
    
    # Initialize prototype data
    prototype_data = {}
    
    for k in range(n_clusters):
        cluster_mask = labels == k
        cluster_indices = persona_responses.index[cluster_mask]
        cluster_weights = weights[cluster_mask]
        
        # Normalize weights within cluster
        if cluster_weights.sum() > 0:
            cluster_weights = cluster_weights / cluster_weights.sum()
        else:
            cluster_weights = np.ones(len(cluster_weights)) / len(cluster_weights)
        
        prototype_id = f"prototype_{k}"
        prototype_data[prototype_id] = {}
        
        for q in questions:
            # Weighted average of distributions
            avg_dist = np.zeros(n_categories)
            for i, idx in enumerate(cluster_indices):
                dist = np.array(persona_responses.loc[idx, q])
                avg_dist += cluster_weights[i] * dist
            
            # Normalize
            avg_dist = avg_dist / (avg_dist.sum() + EPS)
            prototype_data[prototype_id][q] = avg_dist.tolist()
    
    prototypes = pd.DataFrame(prototype_data).T
    prototypes = prototypes[questions]  # Ensure column order
    prototypes.index = prototypes.index.astype(str)
    prototypes.columns = prototypes.columns.astype(str)
    
    return prototypes


def compute_prototype_prior(
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute prior weights over prototypes by summing original persona weights.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster assignment for each persona.
    weights : np.ndarray
        Prior weights for each persona.
    
    Returns
    -------
    prototype_prior : np.ndarray
        Prior weights over prototypes.
    """
    n_clusters = len(np.unique(labels))
    prototype_prior = np.zeros(n_clusters)
    
    for k in range(n_clusters):
        prototype_prior[k] = weights[labels == k].sum()
    
    # Normalize
    prototype_prior = prototype_prior / (prototype_prior.sum() + EPS)
    
    return prototype_prior


def compute_soft_assignments(
    persona_responses: pd.DataFrame,
    prototype_responses: pd.DataFrame,
    questions: List[str],
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Compute soft assignments from personas to prototypes.
    
    Uses negative JS divergence with softmax for soft assignment.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Original persona responses.
    prototype_responses : pd.DataFrame
        Prototype responses.
    questions : List[str]
        Questions to use.
    temperature : float
        Softmax temperature (lower = harder assignment).
    
    Returns
    -------
    soft_assignments : np.ndarray of shape (n_personas, n_prototypes)
        Soft assignment matrix (rows sum to 1).
    """
    n_personas = len(persona_responses)
    n_prototypes = len(prototype_responses)
    
    # Compute JS divergence from each persona to each prototype
    divergences = np.zeros((n_personas, n_prototypes))
    
    for i, p_idx in enumerate(persona_responses.index):
        persona_vec = []
        for q in questions:
            persona_vec.extend(persona_responses.loc[p_idx, q])
        persona_vec = np.array(persona_vec)
        
        for j, proto_idx in enumerate(prototype_responses.index):
            proto_vec = []
            for q in questions:
                proto_vec.extend(prototype_responses.loc[proto_idx, q])
            proto_vec = np.array(proto_vec)
            
            divergences[i, j] = jensenshannon(persona_vec, proto_vec)
    
    # Convert to soft assignments via softmax on negative divergence
    neg_div = -divergences / temperature
    neg_div = neg_div - neg_div.max(axis=1, keepdims=True)  # Stability
    exp_neg_div = np.exp(neg_div)
    soft_assignments = exp_neg_div / exp_neg_div.sum(axis=1, keepdims=True)
    
    return soft_assignments


def evaluate_clustering(
    train_user_responses: pd.DataFrame,
    prototype_responses: pd.DataFrame,
    prototype_prior: np.ndarray,
    feasible_questions: List[str],
) -> float:
    """
    Evaluate clustering quality using training data marginal likelihood.
    
    Computes: Σ_u log( Σ_k π_k p(y^(u) | prototype_k) )
    
    Parameters
    ----------
    train_user_responses : pd.DataFrame
        Training user responses.
    prototype_responses : pd.DataFrame
        Prototype responses.
    prototype_prior : np.ndarray
        Prior over prototypes.
    feasible_questions : List[str]
        Questions to use for evaluation.
    
    Returns
    -------
    marginal_ll : float
        Total marginal log-likelihood.
    """
    n_prototypes = len(prototype_responses)
    total_ll = 0.0
    n_users = 0
    
    for user_idx in train_user_responses.index:
        user_row = train_user_responses.loc[user_idx]
        
        # Compute log p(y^(u) | prototype_k) for each prototype
        log_likelihoods = np.zeros(n_prototypes)
        
        for k, proto_idx in enumerate(prototype_responses.index):
            log_lik = 0.0
            n_answered = 0
            
            for q in feasible_questions:
                if q in user_row.index and user_row[q] != -1:
                    answer = int(user_row[q])
                    proto_dist = np.array(prototype_responses.loc[proto_idx, q])
                    proto_dist = np.maximum(proto_dist, EPS)
                    log_lik += np.log(proto_dist[answer])
                    n_answered += 1
            
            if n_answered > 0:
                log_likelihoods[k] = log_lik
            else:
                log_likelihoods[k] = 0.0
        
        # Marginal log-likelihood: log( Σ_k π_k p(y | k) )
        # = log( Σ_k exp( log π_k + log p(y | k) ) )
        log_prior = np.log(np.maximum(prototype_prior, EPS))
        log_joint = log_prior + log_likelihoods
        max_log_joint = np.max(log_joint)
        marginal_ll = max_log_joint + np.log(np.sum(np.exp(log_joint - max_log_joint)))
        
        total_ll += marginal_ll
        n_users += 1
    
    return total_ll


def select_n_clusters(
    persona_responses: pd.DataFrame,
    weights: np.ndarray,
    train_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    n_categories: int,
    k_range: Tuple[int, int] = (10, 100),
    k_step: int = 10,
    method: str = "weighted_kmeans",
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of clusters using validation on training data.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Pruned persona responses.
    weights : np.ndarray
        Prior weights for personas.
    train_user_responses : pd.DataFrame
        Training user responses for validation.
    feasible_questions : List[str]
        Questions to use.
    n_categories : int
        Number of response categories.
    k_range : Tuple[int, int]
        Range of K values to try (min, max).
    k_step : int
        Step size for K values.
    method : str
        Clustering method ("weighted_kmeans" or "hierarchical").
    random_state : int
        Random seed.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    best_k : int
        Optimal number of clusters.
    scores : Dict[int, float]
        Validation scores for each K.
    """
    k_values = list(range(k_range[0], min(k_range[1] + 1, len(persona_responses)), k_step))
    
    # Ensure we have at least a few values to try
    if len(k_values) < 3:
        k_values = list(range(
            max(2, k_range[0]),
            min(len(persona_responses), k_range[1] + 1),
            max(1, (k_range[1] - k_range[0]) // 5)
        ))
    
    scores = {}
    
    iterator = tqdm(k_values, desc="  Selecting K", disable=not verbose)
    for k in iterator:
        if k >= len(persona_responses):
            continue
        
        # Cluster
        if method == "weighted_kmeans":
            labels, _ = weighted_kmeans_js(
                persona_responses, weights, k, feasible_questions,
                random_state=random_state
            )
        else:
            labels = hierarchical_clustering_js(
                persona_responses, k, feasible_questions
            )
        
        # Create prototypes
        prototypes = create_prototype_personas(
            persona_responses, labels, weights, feasible_questions, n_categories
        )
        
        # Compute prototype prior
        proto_prior = compute_prototype_prior(labels, weights)
        
        # Evaluate
        score = evaluate_clustering(
            train_user_responses, prototypes, proto_prior, feasible_questions
        )
        
        scores[k] = score
        
        if verbose:
            iterator.set_postfix({"k": k, "score": f"{score:.2f}"})
    
    # Select K with best score (highest marginal likelihood)
    best_k = max(scores, key=scores.get)
    
    return best_k, scores


def cluster_personas(
    persona_responses: pd.DataFrame,
    prior_weights: np.ndarray,
    train_user_responses: pd.DataFrame,
    feasible_questions: List[str],
    n_categories: int,
    target_questions: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    n_clusters_range: Tuple[int, int] = (10, 100),
    n_clusters_step: int = 10,
    prune_threshold: float = 0.001,
    min_personas: int = 10,
    method: str = "weighted_kmeans",
    assignment: str = "hard",
    soft_temperature: float = 1.0,
    random_state: int = 42,
    verbose: bool = False,
) -> ClusteringResult:
    """
    Main function to cluster personas into prototypes.
    
    Parameters
    ----------
    persona_responses : pd.DataFrame
        Original persona response distributions.
    prior_weights : np.ndarray
        Empirical Bayes prior weights for each persona.
    train_user_responses : pd.DataFrame
        Training user responses for validation.
    feasible_questions : List[str]
        Questions to use for clustering.
    n_categories : int
        Number of response categories.
    n_clusters : int, optional
        Number of clusters. If None, selected by validation.
    n_clusters_range : Tuple[int, int]
        Range of K values to try if n_clusters is None.
    n_clusters_step : int
        Step size for K values.
    prune_threshold : float
        Minimum prior weight to keep a persona.
    min_personas : int
        Minimum number of personas to keep.
    method : str
        Clustering method ("weighted_kmeans" or "hierarchical").
    assignment : str
        Assignment type ("hard" or "soft").
    soft_temperature : float
        Temperature for soft assignment (lower = harder).
    random_state : int
        Random seed.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    ClusteringResult
        Clustering results including prototypes and mappings.
    """
    n_original = len(persona_responses)
    
    # Combine feasible and target questions for prototype creation
    # Clustering is based on feasible_questions, but prototypes include all
    if target_questions is None:
        target_questions = []
    all_questions = list(feasible_questions) + [q for q in target_questions if q not in feasible_questions]
    
    if verbose:
        print(f"\n{'='*60}")
        print("Clustering Personas into Prototypes")
        print(f"{'='*60}")
        print(f"  Original personas: {n_original}")
        print(f"  Clustering on {len(feasible_questions)} feasible questions")
        print(f"  Prototypes will include {len(all_questions)} questions total")
    
    # Step 1: Prune low-weight personas
    pruned_responses, pruned_weights, kept_indices = prune_personas_by_prior(
        persona_responses, prior_weights, prune_threshold, min_personas
    )
    n_pruned = len(pruned_responses)
    
    if verbose:
        print(f"  After pruning (threshold={prune_threshold}): {n_pruned}")
    
    # Step 2: Select number of clusters if not specified
    validation_scores = None
    selected_k = None
    
    if n_clusters is None:
        if verbose:
            print(f"  Selecting K from range {n_clusters_range}...")
        
        selected_k, validation_scores = select_n_clusters(
            pruned_responses, pruned_weights, train_user_responses,
            feasible_questions, n_categories, n_clusters_range,
            n_clusters_step, method, random_state, verbose
        )
        n_clusters = selected_k
        
        if verbose:
            print(f"  Selected K = {n_clusters}")
    
    # Ensure n_clusters doesn't exceed n_pruned
    n_clusters = min(n_clusters, n_pruned)
    
    if verbose:
        print(f"  Creating {n_clusters} prototypes...")
    
    # Step 3: Cluster
    if method == "weighted_kmeans":
        labels, _ = weighted_kmeans_js(
            pruned_responses, pruned_weights, n_clusters, feasible_questions,
            random_state=random_state
        )
    else:
        labels = hierarchical_clustering_js(
            pruned_responses, n_clusters, feasible_questions
        )
    
    # Step 4: Create prototype personas (include ALL questions for predictions)
    prototypes = create_prototype_personas(
        pruned_responses, labels, pruned_weights, all_questions, n_categories
    )
    
    # Step 5: Compute prototype prior
    prototype_prior = compute_prototype_prior(labels, pruned_weights)
    
    # Step 6: Create persona-to-prototype mapping
    persona_to_prototype = {}
    for i, idx in enumerate(pruned_responses.index):
        persona_to_prototype[idx] = int(labels[i])
    
    # For pruned personas, assign to nearest prototype (or -1)
    for idx in persona_responses.index:
        if idx not in persona_to_prototype:
            persona_to_prototype[idx] = -1  # Pruned
    
    # Step 7: Compute soft assignments if requested
    soft_assignments = None
    if assignment == "soft":
        if verbose:
            print("  Computing soft assignments...")
        soft_assignments = compute_soft_assignments(
            pruned_responses, prototypes, feasible_questions, soft_temperature
        )
    
    if verbose:
        print(f"  Done! {n_original} → {n_clusters} prototypes")
    
    return ClusteringResult(
        prototype_responses=prototypes,
        prototype_prior=prototype_prior,
        persona_to_prototype=persona_to_prototype,
        soft_assignments=soft_assignments,
        n_original_personas=n_original,
        n_pruned_personas=n_pruned,
        n_prototypes=n_clusters,
        validation_scores=validation_scores,
        selected_k=selected_k,
    )
