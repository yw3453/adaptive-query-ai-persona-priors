# Adaptive Querying with AI Persona Priors

This repository contains code and data for the paper **"Adaptive Querying with AI Persona Priors"** ([arXiv](https://arxiv.org/abs/2605.00696)).

## What's in this repository

This repository contains a full implementation of the framework together with
the data needed to reproduce the experiments in the paper. Concretely, it
includes:

- `src/` — the Bayesian persona-based core (posterior updates, greedy /
  non-adaptive / random / full baselines, empirical Bayes prior learning,
  persona clustering, CAT/MIRT reference baselines). See
  [`src/README.md`](src/README.md) for a module-by-module overview.
- `adaptive-query/` — the **real-data** experiment entrypoint
  (`main.py` + `config.yaml`) that runs all methods on WorldValuesBench.
- `adaptive-query-synthetic/` — the **synthetic** experiment entrypoint
  (`main.py` + `config.yaml`), in which users are generated from the persona
  model (well-specified prior).
- `data/personas.json` — Twin-2K-500 persona summaries used as the persona
  dictionary (see [Data setup](#data-setup)).
- `data/WorldValuesBench/` — persona response distributions used throughout
  the paper (see [Data setup](#data-setup)).
- `pre-process/` — scripts to (re)construct the real WVS response matrix from
  the raw WVS Wave 7 micro-data; see
  [`pre-process/README.md`](pre-process/README.md).

## Method at a glance

- **Personas** — a dictionary of $n$ profiles, each inducing a categorical
  response distribution $\mu_{\theta,x}$ over question $x$.
- **User model** — a new user is modeled as a (soft) member of the persona
  dictionary; responses are categorical given the persona.
- **Inference** — given observed answers $Y_{I_t}$, the posterior over
  persona membership and the posterior predictive over an unasked questions are updated.
- **Query selection** — the repo implements **greedy** (one-step lookahead on
  target-entropy), **non-adaptive** (forward-selected fixed set),
  **random** / **random fixed**, a **full** oracle baseline, and classical
  **CAT** baselines (GRM / GPCM, unidimensional and multidimensional).

## Installation

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

### Dependencies

- Python 3.13+
- NumPy, Pandas, SciPy
- Numba (JIT compilation)
- Joblib (parallelization)
- tqdm, PyYAML, matplotlib, seaborn, scikit-learn

## Data setup

The experiments use three data artifacts, layered as follows:

1. **Persona profiles** (`data/personas.json`) — the Twin-2K-500 persona bank.
2. **Persona response distributions** (`data/WorldValuesBench/worldvalues_simulated*.csv`)
   — produced offline by prompting GPT-5-mini with each persona profile.
3. **Real user responses** (`data/WorldValuesBench/worldvalues_real.csv`) —
   WorldValuesBench Wave 7; not redistributed, must be reconstructed locally.

All response files use the
[WorldValuesBench](https://github.com/Demon702/WorldValuesBench) question
bank (91 ordinal 4-point Likert questions after filtering).

### Persona profiles (`data/personas.json`, shipped with this repo)

A JSON dictionary of $n = 2{,}058$ text-based persona summaries, keyed by
persona id (`pid_<integer>`). Each value is a prose description of a real
U.S.\ survey participant's demographics, values, beliefs, and background,
drawn from the **Twin-2K-500** persona bank (Toubia et al., 2025).

This file was constructed following the
[Digital-Twin-Simulation demo notebook](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/main/notebooks/demo_simple_simulation.ipynb):
persona summaries are loaded from the Hugging Face dataset
[`LLM-Digital-Twin/Twin-2K-500`](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
(configuration `full_persona`, field `persona_summary`, keyed by `pid`) and
stored as a `{"pid_<id>": "<summary text>"}` mapping.

```python
import json
personas = json.load(open("data/personas.json"))
len(personas)                 # 2058
personas["pid_574"][:120]     # "The following is a description of a person...."
```

### Persona response distributions (shipped with this repo)

These are the persona--question response distributions
$\mu_{\theta,x}$ used by every persona-based method:

```
data/WorldValuesBench/worldvalues_simulated.csv
data/WorldValuesBench/worldvalues_simulated_deterministic.csv
```

**How they were produced.** For each (persona, question) pair, we prompt
**GPT-5-mini** conditioned on the persona profile from `data/personas.json`
and the question text from WorldValuesBench to produce a categorical
distribution over the four Likert responses. The exact system/user prompts
are reproduced in the paper.

The two files differ only in the elicitation strategy:

| File | Elicitation | Format |
|------|-------------|--------|
| `worldvalues_simulated.csv` | **Direct distribution elicitation.** The LLM outputs a 4-entry probability vector per (persona, question). Used as the default persona response model. | Each cell is a JSON list `[p1, p2, p3, p4]`. |
| `worldvalues_simulated_deterministic.csv` | **Deterministic mode.** The LLM outputs a single most-likely answer $\hat{y}$ per (persona, question). At runtime it is converted to a categorical $(1-\varepsilon, \varepsilon/(K-1), \dots)$ to match the interface. | Each cell is an integer answer. |

Both are indexed by Twin-2K-500 persona IDs (rows) × WorldValuesBench question
IDs (columns), matching the keys in `data/personas.json`.

### Real user responses (not redistributed — must be reconstructed locally)

The real-user response matrix lives at

```
data/WorldValuesBench/worldvalues_real.csv
```

It is **not** shipped here because it is derived from the World Values Survey
Wave 7 micro-data, which is distributed only under a signed WVS consent form.
To reconstruct it (88,459 users × 91 questions of ordinal Likert responses),
follow the two-step download-and-run instructions in
[`pre-process/README.md`](pre-process/README.md). After that step,
`data/WorldValuesBench/` will contain the persona files plus
`worldvalues_real.csv`.

You only need this if you want to run the **real-data** experiments; the
synthetic entrypoint works out of the box.

## Quick start

### Synthetic experiments (no WVS download needed)

```bash
uv run adaptive-query-synthetic/main.py
```

Synthetic users are sampled from the persona model.

### Real WorldValuesBench experiments

After following [`pre-process/README.md`](pre-process/README.md):

```bash
uv run adaptive-query/main.py
```

Results are written to `output/real_WorldValuesBench_<timestamp>/` (see
[Output layout](#output-layout)).

## Configuration

Experiments are configured via YAML. The two shipped configs
(`adaptive-query/config.yaml` and `adaptive-query-synthetic/config.yaml`)
correspond to the paper's default settings; key knobs include:

```yaml
dataset:
  name: "WorldValuesBench"
  n_categories: 4                    # K
  target_questions:
    mode: "random_n"                 # held-out target task
    n: 5                             # |I*| = 5
  persona_simulation:
    mode: "distribution"             # or "deterministic_epsilon"
    temperature: 1                   # for calibration
    distribution_file: "worldvalues_simulated.csv"
    deterministic_file: "worldvalues_simulated_deterministic.csv"

budget: 86                          # T

methods:
  greedy: true
  random: true
  random_fixed: true
  nonadaptive: true
  full: true
  cat: true

empirical_bayes:
  enabled: true                      # learn p(theta) via EM from training users

clustering:
  enabled: false                     # optionally compress persona dictionary
```

See the config files for every option and inline documentation.

## Output layout

Each run creates a timestamped directory:

```
output/{experiment_id}/
├── config.yaml              # configuration used
├── experiment_info.json
├── summary.txt              # human-readable summary
├── summary.csv              # method comparison table
├── detailed/                # per-user results
│   ├── greedy.json
│   ├── random.json
│   └── ...
├── analysis/                # analysis tables
│   ├── question_frequency_*.csv
│   ├── performance_by_budget.csv
│   └── ...
└── figures/                 # log-loss, Brier, ordinal MSE plots
```

## Data format

### Persona response distributions (`worldvalues_simulated.csv`)

```python
import ast, pandas as pd
persona_responses = pd.read_csv(
    "data/WorldValuesBench/worldvalues_simulated.csv", index_col=0
).map(ast.literal_eval)
# persona_responses.loc[persona_id, question_id] -> [p1, p2, p3, p4]
```

### User responses (`worldvalues_real.csv`)

```python
user_responses = pd.read_csv(
    "data/WorldValuesBench/worldvalues_real.csv", index_col=0
).astype(int)
# entries in {0, 1, 2, 3}; -1 denotes missing
```

## Citation

```bibtex
@misc{wang2026adaptive,
      title={Adaptive Querying with AI Persona Priors}, 
      author={Kaizheng Wang and Yuhang Wu and Assaf Zeevi},
      year={2026},
      eprint={2605.00696},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.00696}, 
}
```
