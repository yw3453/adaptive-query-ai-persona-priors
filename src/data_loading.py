"""Shared dataset loading utilities for real/synthetic entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd


PersonaSimulationMode = Literal["distribution", "deterministic_epsilon"]


def _load_worldvaluesbench_real(wvb_dir: Path) -> pd.DataFrame:
    """Load real user responses for WorldValuesBench."""
    real_path = wvb_dir / "worldvalues_real.csv"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing real response file: {real_path}")

    real_df = pd.read_csv(real_path, index_col=0)
    real_df = real_df.astype(int)
    real_df.index = real_df.index.astype(str)
    real_df.columns = real_df.columns.astype(str)
    return real_df


def _validate_temperature(temperature: float) -> float:
    """Validate and normalize persona simulation temperature."""
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(
            f"dataset.persona_simulation.temperature must be a positive finite number, got {temperature}"
        )
    return temp


def _apply_temperature_to_distribution(
    dist: np.ndarray,
    temperature: float,
    *,
    row_id: str,
    question_id: str,
) -> np.ndarray:
    """
    Apply temperature scaling to a categorical distribution.

    Uses: p_tau(k) ∝ p(k)^(1 / temperature)
    """
    if temperature == 1.0:
        return dist

    scaled = np.power(dist, 1.0 / temperature)
    total = float(scaled.sum())
    if total <= 0.0:
        raise ValueError(
            f"Temperature scaling produced zero-mass distribution at persona '{row_id}', question '{question_id}'."
        )
    return scaled / total


def _parse_distribution_cell(
    raw_cell: Any,
    *,
    row_id: str,
    question_id: str,
    n_categories: int,
    temperature: float,
) -> list[float]:
    """Parse and validate a JSON-encoded categorical distribution cell."""
    try:
        parsed = json.loads(raw_cell) if isinstance(raw_cell, str) else raw_cell
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed distribution JSON at persona '{row_id}', question '{question_id}': {raw_cell!r}"
        ) from exc

    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(
            f"Expected list-like distribution at persona '{row_id}', question '{question_id}', got {type(parsed).__name__}"
        )
    if len(parsed) != n_categories:
        raise ValueError(
            f"Distribution length mismatch at persona '{row_id}', question '{question_id}': "
            f"expected {n_categories}, got {len(parsed)}"
        )

    dist = np.asarray(parsed, dtype=np.float64)
    if not np.isfinite(dist).all():
        raise ValueError(
            f"Non-finite probability in distribution at persona '{row_id}', question '{question_id}'"
        )
    if (dist < 0).any():
        raise ValueError(
            f"Negative probability in distribution at persona '{row_id}', question '{question_id}'"
        )
    total = float(dist.sum())
    if total <= 0.0:
        raise ValueError(
            f"Distribution sum must be positive at persona '{row_id}', question '{question_id}'"
        )

    # Preserve intended behavior but guard against tiny floating-point drift.
    dist = dist / total
    dist = _apply_temperature_to_distribution(
        dist, temperature, row_id=row_id, question_id=question_id
    )
    return dist.tolist()


def _load_distribution_personas(
    wvb_dir: Path,
    filename: str,
    n_categories: int,
    temperature: float,
) -> pd.DataFrame:
    """Load persona distributions directly from CSV (JSON list cells)."""
    path = wvb_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing persona distribution file: {path}")

    raw_df = pd.read_csv(path, index_col=0)
    raw_df.index = raw_df.index.astype(str)
    raw_df.columns = raw_df.columns.astype(str)

    parsed_df = pd.DataFrame(index=raw_df.index, columns=raw_df.columns, dtype=object)
    for row_id, row in raw_df.iterrows():
        for question_id in raw_df.columns:
            parsed_df.at[row_id, question_id] = _parse_distribution_cell(
                row[question_id],
                row_id=str(row_id),
                question_id=str(question_id),
                n_categories=n_categories,
                temperature=temperature,
            )
    return parsed_df


def _resolve_answer_index_base(
    answer_values: np.ndarray,
    n_categories: int,
    configured_base: Any,
) -> int:
    """Resolve deterministic answer encoding base (0-indexed vs 1-indexed)."""
    if configured_base in ("auto", None):
        vmin = int(answer_values.min())
        vmax = int(answer_values.max())
        if 0 <= vmin and vmax <= n_categories - 1:
            return 0
        if 1 <= vmin and vmax <= n_categories:
            return 1
        raise ValueError(
            "Could not infer deterministic answer encoding base from values "
            f"[{vmin}, {vmax}] with n_categories={n_categories}. "
            "Set dataset.persona_simulation.answer_index_base explicitly to 0 or 1."
        )

    if configured_base in (0, 1):
        return int(configured_base)

    raise ValueError(
        "dataset.persona_simulation.answer_index_base must be one of: 'auto', 0, 1"
    )


def _build_epsilon_distribution_personas(
    wvb_dir: Path,
    filename: str,
    n_categories: int,
    epsilon: float,
    answer_index_base: Any,
    temperature: float,
) -> pd.DataFrame:
    """Load deterministic answers and convert to categorical distributions."""
    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError(f"dataset.persona_simulation.epsilon must be in [0, 1], got {epsilon}")
    if n_categories < 2 and epsilon > 0:
        raise ValueError("deterministic_epsilon mode requires n_categories >= 2 when epsilon > 0")

    path = wvb_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing deterministic persona file: {path}")

    det_df = pd.read_csv(path, index_col=0)
    det_df.index = det_df.index.astype(str)
    det_df.columns = det_df.columns.astype(str)

    if det_df.isna().any().any():
        bad_count = int(det_df.isna().sum().sum())
        raise ValueError(
            f"Deterministic persona file contains {bad_count} missing values; all entries must be valid answers."
        )

    numeric = det_df.to_numpy(dtype=np.float64)
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded):
        raise ValueError(
            "Deterministic persona file must contain integer-coded answers only "
            "(e.g., 1..K or 0..K-1)."
        )
    answer_values = rounded.astype(np.int64)

    base = _resolve_answer_index_base(answer_values, n_categories, answer_index_base)
    encoded = answer_values - base

    if encoded.min() < 0 or encoded.max() > n_categories - 1:
        raise ValueError(
            f"Deterministic answers out of range after base={base} conversion. "
            f"Expected in [0, {n_categories - 1}], got [{encoded.min()}, {encoded.max()}]."
        )

    n_personas, n_questions = encoded.shape
    if n_categories == 1:
        dist_3d = np.ones((n_personas, n_questions, 1), dtype=np.float64)
    else:
        off_prob = float(epsilon) / float(n_categories - 1)
        on_prob = 1.0 - float(epsilon)
        dist_3d = np.full((n_personas, n_questions, n_categories), off_prob, dtype=np.float64)
        row_idx = np.repeat(np.arange(n_personas), n_questions)
        col_idx = np.tile(np.arange(n_questions), n_personas)
        cat_idx = encoded.reshape(-1)
        dist_3d[row_idx, col_idx, cat_idx] = on_prob

    if temperature != 1.0:
        scaled = np.power(dist_3d, 1.0 / temperature)
        denom = scaled.sum(axis=2, keepdims=True)
        if np.any(denom <= 0.0):
            raise ValueError(
                "Temperature scaling produced zero-mass distribution in deterministic_epsilon mode."
            )
        dist_3d = scaled / denom

    persona_df = pd.DataFrame(index=det_df.index, columns=det_df.columns, dtype=object)
    for q_idx, question_id in enumerate(det_df.columns):
        persona_df[question_id] = [dist_3d[p_idx, q_idx, :].tolist() for p_idx in range(n_personas)]

    return persona_df


def _validate_persona_question_alignment(real_df: pd.DataFrame, persona_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure persona matrix covers exactly the same questions as real data."""
    real_questions = set(real_df.columns)
    persona_questions = set(persona_df.columns)

    missing = sorted(real_questions - persona_questions)
    extra = sorted(persona_questions - real_questions)
    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"missing questions in persona file: {missing[:10]}")
        if extra:
            msg_parts.append(f"unexpected extra questions in persona file: {extra[:10]}")
        raise ValueError("Persona question mismatch with real data: " + "; ".join(msg_parts))

    # Reorder persona columns to the same order as real questions for consistency.
    return persona_df[real_df.columns]


def load_worldvaluesbench(config: dict[str, Any], data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load WorldValuesBench with configurable persona simulation mode."""
    dataset_cfg = config.get("dataset", {})
    n_categories = int(dataset_cfg["n_categories"])
    persona_cfg = dataset_cfg.get("persona_simulation", {})

    mode: PersonaSimulationMode = persona_cfg.get("mode", "distribution")
    distribution_file = persona_cfg.get("distribution_file", "worldvalues_simulated.csv")
    deterministic_file = persona_cfg.get("deterministic_file", "worldvalues_simulated_deterministic.csv")
    answer_index_base = persona_cfg.get("answer_index_base", "auto")
    temperature = _validate_temperature(persona_cfg.get("temperature", 1.0))

    wvb_dir = data_dir / "WorldValuesBench"
    if not wvb_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {wvb_dir}")

    real_df = _load_worldvaluesbench_real(wvb_dir)

    if mode == "distribution":
        persona_df = _load_distribution_personas(
            wvb_dir=wvb_dir,
            filename=distribution_file,
            n_categories=n_categories,
            temperature=temperature,
        )
    elif mode == "deterministic_epsilon":
        epsilon = float(persona_cfg.get("epsilon", 0.0))
        persona_df = _build_epsilon_distribution_personas(
            wvb_dir=wvb_dir,
            filename=deterministic_file,
            n_categories=n_categories,
            epsilon=epsilon,
            answer_index_base=answer_index_base,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown dataset.persona_simulation.mode '{mode}'. "
            "Supported modes: 'distribution', 'deterministic_epsilon'."
        )

    persona_df = _validate_persona_question_alignment(real_df, persona_df)
    return real_df, persona_df


def load_dataset(config: dict[str, Any], data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset based on configuration."""
    dataset_name = config["dataset"]["name"]
    if dataset_name == "WorldValuesBench":
        return load_worldvaluesbench(config, data_dir)
    raise ValueError(f"Unknown dataset: {dataset_name}")
