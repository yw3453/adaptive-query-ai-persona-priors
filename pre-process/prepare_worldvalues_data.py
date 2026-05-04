"""
Reconstruct ``data/WorldValuesBench/worldvalues_real.csv`` from the raw
World Values Survey (WVS) Wave 7 dataset.

This is a two-stage pipeline:

Stage 1
    Invokes ``pre-process/dataset_construction/data_preparation.py`` (the
    upstream WorldValuesBench preprocessing script, kept byte-identical) to
    turn the raw WVS csv into an intermediate
    ``pre-process/WorldValuesBench/full/full_value_qa.tsv``.

Stage 2
    Loads that TSV, keeps only the 4-point ordinal value questions defined
    in ``pre-process/dataset_construction/question_metadata.json``, drops
    respondents with more than 20% missing answers, maps ordinal responses
    ``{1, 2, 3, 4} -> {0, 1, 2, 3}`` with ``-1`` for missing, and writes the
    result to ``data/WorldValuesBench/worldvalues_real.csv``.

Run from the repository root::

    uv run pre-process/prepare_worldvalues_data.py

See ``pre-process/README.md`` for the full download + consent-form workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_CONSTRUCTION_DIR = SCRIPT_DIR / "dataset_construction"
DATA_PREPARATION_PY = DATASET_CONSTRUCTION_DIR / "data_preparation.py"
QUESTION_METADATA_JSON = DATASET_CONSTRUCTION_DIR / "question_metadata.json"

# data_preparation.py writes its output to ``{its dir}/../WorldValuesBench``,
# which resolves to ``pre-process/WorldValuesBench`` in this layout.
INTERMEDIATE_DIR = SCRIPT_DIR / "WorldValuesBench"
INTERMEDIATE_VALUE_TSV = INTERMEDIATE_DIR / "full" / "full_value_qa.tsv"

DEFAULT_RAW_CSV = DATASET_CONSTRUCTION_DIR / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "WorldValuesBench"
OUTPUT_REAL_CSV = OUTPUT_DIR / "worldvalues_real.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct worldvalues_real.csv from the raw WVS dataset.",
    )
    parser.add_argument(
        "--raw-dataset-path",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=(
            "Path to the raw WVS csv downloaded from "
            "https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp. "
            f"Defaults to pre-process/{DEFAULT_RAW_CSV.relative_to(SCRIPT_DIR)}."
        ),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help=(
            "Keep the intermediate pre-process/WorldValuesBench/ folder "
            "after the run (useful for debugging). "
            "By default it is deleted on success."
        ),
    )
    return parser.parse_args()


def run_data_preparation(raw_dataset_path: Path) -> None:
    """Stage 1: invoke the upstream WVB data_preparation.py as a subprocess."""
    if not raw_dataset_path.exists():
        sys.exit(
            f"Raw WVS csv not found at {raw_dataset_path}.\n"
            "Download it from https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp "
            "(requires signing a consent form), unzip it, and place it at the default "
            "path above, or pass --raw-dataset-path <path> explicitly."
        )

    print(
        f"[stage 1] Running {DATA_PREPARATION_PY.name} on "
        f"{raw_dataset_path.name} ..."
    )
    subprocess.run(
        [
            sys.executable,
            str(DATA_PREPARATION_PY),
            "--raw-dataset-path",
            str(raw_dataset_path),
        ],
        check=True,
    )
    print(
        f"[stage 1] Done. Intermediate files in "
        f"pre-process/{INTERMEDIATE_DIR.relative_to(SCRIPT_DIR)}/"
    )


def build_worldvalues_real_csv() -> None:
    """Stage 2: produce worldvalues_real.csv from the intermediate TSV."""
    if not INTERMEDIATE_VALUE_TSV.exists():
        sys.exit(
            f"Expected intermediate file missing: {INTERMEDIATE_VALUE_TSV}.\n"
            "Did stage 1 succeed?"
        )

    print(f"[stage 2] Building data/WorldValuesBench/{OUTPUT_REAL_CSV.name} ...")

    full_value_df = pd.read_csv(INTERMEDIATE_VALUE_TSV, sep="\t", low_memory=False)
    with QUESTION_METADATA_JSON.open("r") as f:
        question_metadata = json.load(f)

    kept_cols = [
        k
        for k, v in question_metadata.items()
        if k in full_value_df.columns
        and k.startswith("Q")
        and v["answer_data_type"] == "ordinal"
        and v["answer_scale_max"] == 4
    ]

    full_value_df = full_value_df.set_index("D_INTERVIEW")[kept_cols]

    missing_frac = full_value_df.isnull().mean(axis=1)
    full_value_df = full_value_df[missing_frac < 0.2]

    def to_index(value: object) -> int:
        # Map ordinal 1..4 -> 0..3; anything else (NaN, out-of-range, invalid) -> -1.
        try:
            iv = int(value)
        except (ValueError, TypeError):
            return -1
        if iv < 1 or iv > 4:
            return -1
        return iv - 1

    real_df = full_value_df.apply(lambda col: col.map(to_index)).astype("int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_df.to_csv(OUTPUT_REAL_CSV)
    print(
        f"[stage 2] Wrote data/WorldValuesBench/{OUTPUT_REAL_CSV.name} "
        f"({len(real_df)} rows, {len(real_df.columns)} columns)."
    )


def cleanup_intermediate() -> None:
    if INTERMEDIATE_DIR.exists():
        print(
            f"[cleanup] Removing intermediate folder "
            f"pre-process/{INTERMEDIATE_DIR.relative_to(SCRIPT_DIR)}/"
        )
        shutil.rmtree(INTERMEDIATE_DIR)


def main() -> None:
    args = parse_args()
    run_data_preparation(args.raw_dataset_path)
    build_worldvalues_real_csv()
    if not args.keep_intermediate:
        cleanup_intermediate()
    print("Done.")


if __name__ == "__main__":
    main()
