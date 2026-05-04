# Pre-processing WorldValuesBench real responses

The real-user response file `data/WorldValuesBench/worldvalues_real.csv` is
**not** shipped with this repository. It is derived from the
[World Values Survey (WVS) Wave 7](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp)
micro-data, which WVS only distributes to users who sign its consent form.

To reproduce that file locally, follow the two steps below.

## Step 1 — Download the raw WVS data

1. Go to https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp and sign
   the WVS consent form.
2. Download the CSV archive **WVS Cross-National Wave 7 csv v6.0**
   (file name `WVS_Cross-National_Wave_7_csv_v6_0.zip`).
3. Unzip it and place the file `WVS_Cross-National_Wave_7_csv_v6_0.csv`
   inside `pre-process/dataset_construction/`, so the final path is:

   ```
   pre-process/dataset_construction/WVS_Cross-National_Wave_7_csv_v6_0.csv
   ```

   (If you prefer to keep the raw csv elsewhere, you can pass its path via
   `--raw-dataset-path` in step 2.)

## Step 2 — Run the preprocessing script

From the repository root:

```bash
uv run pre-process/prepare_worldvalues_data.py
```

What this does:

1. **Stage 1** runs `pre-process/dataset_construction/data_preparation.py`
   (the upstream
   [WorldValuesBench](https://github.com/Demon702/WorldValuesBench)
   preprocessing script, kept byte-identical). It produces an intermediate
   folder `pre-process/WorldValuesBench/` containing TSVs with processed
   responses.
2. **Stage 2** reads the intermediate `full/full_value_qa.tsv`, keeps only
   the 4-point ordinal value questions (as defined in
   `pre-process/dataset_construction/question_metadata.json`), drops
   respondents with more than 20% missing answers, maps ordinal responses
   `{1, 2, 3, 4} → {0, 1, 2, 3}` with `-1` for missing, and writes the
   result to `data/WorldValuesBench/worldvalues_real.csv`.
3. The intermediate `pre-process/WorldValuesBench/` folder is deleted on
   success.

### Optional flags

- `--raw-dataset-path <path>` — use a raw WVS csv at a non-default location.
- `--keep-intermediate` — keep the `pre-process/WorldValuesBench/` folder
  after the run (useful for debugging).

## Result

After a successful run, `data/WorldValuesBench/` should contain:

```
data/WorldValuesBench/
├── worldvalues_real.csv                   # produced by this script
├── worldvalues_simulated.csv              # shipped with repo
└── worldvalues_simulated_deterministic.csv # shipped with repo
```

You can now run the main experiments from the repo README, e.g.
`uv run adaptive-query/main.py`.

## Attribution & licensing

- Raw micro-data: © World Values Survey Association. Use of the WVS data
  is governed by the terms you agreed to when downloading from
  https://www.worldvaluessurvey.org/. Do not redistribute the raw csv.
- `pre-process/dataset_construction/` bundles `data_preparation.py`,
  `question_metadata.json`, `codebook.json`, and `answer_adjustment.json`
  unmodified from
  [WorldValuesBench](https://github.com/Demon702/WorldValuesBench). 
  Please cite both WVS and WorldValuesBench if you use the resulting dataset.
