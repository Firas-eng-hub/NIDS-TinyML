# Dataset placement and splits

[Documentation index](README.md) · [Training guide](training.md)

## Put the dataset here

Run commands from the cloned `NIDS-TinyML` repository root, where `README.md`, `config/` and `src/` are located. Create `data/raw/` if it does not exist:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
```

On Linux/macOS, use `mkdir -p data/raw` instead. Obtain the CSE-CIC-IDS2018 **Intrusion CSV** dataset separately, extract it, and copy the original labeled CSV files directly into this folder:

```text
NIDS-TinyML/
  config/
  src/
  data/                             local only; excluded from Git
    raw/
      Bot.csv                       example input filenames
      FTP-BruteForce.csv
      SSH-Bruteforce.csv
      DoS attacks-Hulk.csv
      SQL Injection.csv
      ...remaining labeled CSVs...
    processed/                      generated automatically; never put inputs here
      random_forest/
      logistic_regression/
      xgboost/
  artifacts/                        generated automatically; excluded from Git
```

The loader reads `.csv` and `.CSV` files **directly inside** the configured raw directory. It does not search subfolders, extract ZIP files, or convert PCAPs for training. Filenames do not determine labels; the `Label` column does.

Keep exactly one copy of each input CSV. Keeping a second copy under another name duplicates its rows: the loader does not deduplicate records. Keep generated flow CSVs, exported test rows, alternate datasets, and manually split files outside `data/raw/`. Do not concatenate generated `processed.csv` back into the raw inputs.

## Required schema

Each input needs a `Label` column (matched without regard to case after whitespace trimming) and all 15 feature columns below. Feature names are case-sensitive after trimming. `Timestamp` is optional. Other columns are discarded.

| Position | Feature |
|---|---|
| 0 | Flow Duration |
| 1 | Tot Fwd Pkts |
| 2 | Tot Bwd Pkts |
| 3 | TotLen Fwd Pkts |
| 4 | TotLen Bwd Pkts |
| 5 | Fwd Pkt Len Max |
| 6 | Bwd Pkt Len Max |
| 7 | Pkt Len Min |
| 8 | Pkt Len Max |
| 9 | SYN Flag Cnt |
| 10 | RST Flag Cnt |
| 11 | ACK Flag Cnt |
| 12 | PSH Flag Cnt |
| 13 | Down/Up Ratio |
| 14 | Flow Pkts/s |

Do not reorder the canonical feature list to work around a schema mismatch. Use an explicitly reviewed conversion of another dataset if its names or feature definitions differ. Matching column names alone does not establish matching semantics.

The pipeline converts features to numbers, drops rows with nonfinite or nonnumeric feature values, and uses float32 features. It drops unsupported labels rather than treating them as Normal. Negative finite values are currently retained.

| Target class | Matching raw labels, checked in this order |
|---|---|
| Normal | Exactly `BENIGN`, ignoring case |
| DoS | Contains `dos`, `ddos`, `hoic` or `loic` |
| BruteForce | Contains `brute force`, `bruteforce`, `ssh-bruteforce` or `ftp-bruteforce` |
| Botnet | Contains `bot` or `botnet` |
| WebAttack | Contains `web attack`, `sql injection` or `xss` |

For example, `Brute Force -Web` maps to BruteForce because that check occurs first. An input already labeled `Normal` is not equivalent to raw `BENIGN` under this mapper. Supply enough valid examples of **all five classes**; a single attack CSV is not sufficient for this five-class workflow. Small classes can prevent either stratified split, and missing classes can prevent evaluation or XGBoost training.

## Use a separate local directory when changing datasets

For another dataset version, keep its raw files in a separate directory, such as `data/raw_v2/`. Copy a complete preset to a new configuration and change these values:

```yaml
data:
  raw_dir: data/raw_v2
  processed_dir: data/processed/dataset_v2
  chunksize: 200000
  max_rows: 0
  sample_frac: 1.0
```

This is a fragment, not a complete configuration. The loader does **not** merge configurations; preserve the remaining sections from the copied preset. Use a new run name, such as `rf_dataset_v2_01`.

Relative data and artifact paths resolve against the repository root, even when the configuration file is in another directory. Absolute paths are accepted. For external Windows folders, forward slashes avoid YAML backslash escaping:

```yaml
raw_dir: 'D:/datasets/ids2018/raw'
```

Do not edit, rename, add or remove raw CSVs after training if you want to replay that run. New runs record raw filenames and SHA-256 content hashes. Named evaluation/replay checks those fingerprints before reusing saved indices. Restore the original files or train a new run when they differ. Moving an old run between machines can also require restoring its original configuration paths; snapshots may contain absolute paths.

## What the 70/15/15 split means

The supplied configurations use:

```yaml
split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
train:
  random_state: 42
```

| Split | Approximate share | Purpose |
|---|---|---|
| Training | 70% | Fit preprocessing, class weights and classifier parameters |
| Validation | 15% | Compare algorithms and choose hyperparameters |
| Test | 15% | Assess the final chosen configuration |

The pipeline performs these steps:

1. Discover CSVs in sorted order, clean rows, and map labels.
2. Apply `sample_frac`, then `max_rows`, if configured. These sampling steps are random and are **not stratified**; rare classes can disappear.
3. Reset row indices and separate training data from the remaining 30%, using mapped labels for stratification and the configured seed.
4. Split that remaining 30% into validation and test subsets, again stratified. With equal 0.15 ratios, each receives half of the remainder.
5. Fit log1p selection and MinMax parameters using **training features only**; transform validation and test data using those same parameters.
6. Train with class weights derived from training labels and report validation/test metrics separately.

Ratios refer to the cleaned, optionally sampled dataset, not the original downloaded row count. Exact counts can differ from percentages through integer rounding. Positive ratios must sum to 1; this implementation expects nonempty training, validation and test splits. Do not set validation to zero.

Stratification preserves class proportions approximately. It does not balance classes to equal sizes, remove duplicated flows, or separate capture days and attack campaigns. Random row splits can produce optimistic scores when related traffic occurs on both sides. See [limitations](limitations.md).

## Where the split is stored

You do **not** need an existing `splits.json` to train. Training generates it:

```text
artifacts/metadata/rf_baseline/
  config.json                  full configuration used for that run
  dataset_manifest.json        raw filenames and content fingerprints
  splits.json                  lists named train, val and test
  metadata.json                model parameters, classes and result references
  feature_order.json
  label_map.json
```

The saved integers refer to positions in the cleaned and sampled dataframe, not raw CSV line numbers. Do not edit them or reuse them with different raw files or sampling settings.

The processed directory also receives `processed.csv` and a latest `splits.json`. These shared files are overwritten by another training job using the same processed directory. Named replay/evaluation uses the run-specific records for new runs. Legacy local runs without snapshots emit warnings and fall back to shared files; their correspondence cannot be guaranteed.

## Avoid conflicts between algorithms and runs

The supplied `config/random_forest.yaml`, `config/logistic_regression.yaml` and `config/xgboost.yaml` use the same raw directory, features, seed, sampling and split ratios. They use different processed directories. Changing only the algorithm therefore produces the same split indices in the same pinned software environment.

Always use unique run names and run training commands sequentially. Named artifacts are stored under each run, but unversioned latest model, scaler, metadata and report copies are shared across algorithms. Concurrent jobs can race when writing those latest copies even with separate processed directories. Prefer `--run_name` when replaying/exporting so the intended model is explicit.

All `data/`, `artifacts/`, notebook and capture files remain local. Confirm exclusions before publication:

```powershell
git check-ignore data/raw/Bot.csv data/processed/random_forest/splits.json
python tools/check_publication.py --history
```
