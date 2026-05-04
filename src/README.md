# `src/` — module overview

The `src/` package contains the core implementation of the persona-based
Bayesian adaptive querying framework and all baselines used in the paper.
Each module carries a detailed docstring at the top; this file is a short
map to help navigate the codebase.

| Module | Role | Key exports |
|---|---|---|
| [`utils.py`](utils.py) | Foundational primitives shared by every method: posterior over personas $p(\theta \mid Y_{I_t})$, posterior predictive $p(Y_x \mid Y_{I_t})$, uncertainty functionals (Shannon entropy, variance, CRPS), evaluation metrics (log loss, Brier, ordinal MSE, KL), empirical Bayes prior learning (EM), and posterior sparsification. Also defines the data-structure conventions used across the package. Numba-JIT hot loops. | `compute_posterior_over_personas`, `compute_posterior_predictive`, `update_posterior_with_observation`, `entropy`, `entropy_over_target_questions`, `variance_over_target_questions`, `log_loss_score`, `brier_score`, `evaluate_predictions`, `learn_empirical_prior`, `apply_temperature_scaling`, `sparsify_posterior`, `PrecomputedPersonaData` |
| [`data_loading.py`](data_loading.py) | Loads `worldvalues_real.csv`, `worldvalues_simulated.csv`, and `worldvalues_simulated_deterministic.csv`; converts deterministic answers into $(1-\varepsilon, \varepsilon/(K-1), \dots)$ categorical distributions; applies optional temperature scaling. Produces the `(persona_responses, user_responses)` tuple consumed by `main.py`. | `load_dataset`, `load_worldvaluesbench`, `PersonaSimulationMode` |
| [`greedy.py`](greedy.py) | Greedy one-step-lookahead adaptive querying (Algorithm 2 in the paper): at each step, picks the question that minimizes the expected posterior sum of target-marginal entropies. Numba-accelerated candidate evaluation + joblib parallelism across users. | `greedy_adaptive_query`, `evaluate_greedy_on_users`, `greedy_select_question`, `greedy_select_question_optimized`, `ObjectiveType` |
| [`baselines.py`](baselines.py) | Persona-based baselines: **Random** (adaptive, uniform question selection), **Random fixed** (one fixed uniform set for all users), **Non-adaptive** (greedy forward-selected fixed set, Algorithm 1 in the paper), **Full** (query all feasible questions; oracle upper bound). | `random_adaptive_query`, `evaluate_random_on_users`, `select_nonadaptive_question_set`, `nonadaptive_set_query`, `evaluate_nonadaptive_set_on_users`, `select_random_fixed_question_set`, `evaluate_random_fixed_set_on_users`, `full_query`, `evaluate_full_on_users` |
| [`cat.py`](cat.py) | Unidimensional polytomous CAT baselines: **GRM** (graded response model) and **GPCM** (generalized partial credit model). Item-parameter calibration via marginal maximum likelihood (EM), grid-based posterior over $\theta$, item selection via MFI (maximum Fisher information) or MEPV (minimum expected posterior variance). | `fit_grm`, `fit_gpcm`, `cat_adaptive_query`, `cat_adaptive_query_gpcm`, `cat_select_question`, `train_and_evaluate_cat_unified`, `CATModelType`, `CATSelectionCriterion` |
| [`cat_mirt.py`](cat_mirt.py) | Multidimensional CAT baselines: **MGRM** and **MGPCM** on a $D$-dimensional latent trait (default $D=3$, Cartesian grid). Item selection via D- and A-optimality. | `fit_mgrm`, `fit_mgpcm`, `mirt_adaptive_query`, `mirt_select_question`, `train_and_evaluate_mirt_cat`, `MIRTModelType`, `MIRTSelectionCriterion` |
| [`clustering.py`](clustering.py) | Persona-dictionary compression for the ablation in Section 5.4 of the paper: prune low-weight personas by the empirical Bayes prior, cluster the remainder via prior-weighted $k$-means with Jensen–Shannon distance, and aggregate to prototype distributions whose prior mass sums cluster-member weights. | `cluster_personas`, `prune_personas_by_prior`, `weighted_kmeans_js`, `create_prototype_personas`, `compute_prototype_prior`, `select_n_clusters`, `ClusteringResult` |
| [`results.py`](results.py) | Experiment result aggregation, summary tables, figures (log loss / Brier / ordinal MSE curves), and the on-disk layout of `output/{experiment_id}/`. | `ExperimentResult`, `ExperimentOutputManager`, `compute_performance_by_budget`, `plot_performance_by_budget`, `plot_metrics_comparison`, `plot_metrics_distribution`, `plot_entropy_trajectory`, `plot_question_frequency` |

### Entry points

The two user-facing scripts live outside `src/`:

- [`adaptive-query/main.py`](../adaptive-query/main.py) — real-data
  experiments on WorldValuesBench (driven by
  [`adaptive-query/config.yaml`](../adaptive-query/config.yaml)).
- [`adaptive-query-synthetic/main.py`](../adaptive-query-synthetic/main.py) —
  synthetic experiments with users sampled from the persona model (driven by
  [`adaptive-query-synthetic/config.yaml`](../adaptive-query-synthetic/config.yaml)).

Both import from `src.*` and dispatch to the method-specific `*_adaptive_query` /
`evaluate_*_on_users` / `train_and_evaluate_*` functions listed above.

### Data-structure conventions (quick reference)

Reproduced from `utils.py` for convenience:

- `persona_responses: pd.DataFrame` — rows are personas, columns are questions;
  each entry is a length-$K$ probability vector (or `None` if missing).
- `user_responses: pd.DataFrame` — rows are users, columns are questions;
  each entry is an integer in $\{0, \dots, K-1\}$, with `-1` denoting missing.
