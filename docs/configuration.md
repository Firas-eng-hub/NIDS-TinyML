# Configuration reference

[Documentation index](README.md) · [Training guide](training.md)

## Select a configuration

```powershell
python -m src.main_train --config config/random_forest.yaml --run_name rf_baseline
```

Without `--config`, commands load `config/config.yaml`. A `--config` filename is resolved from the current working directory. Relative data/artifact paths **inside** YAML are resolved against the repository root, not the config file's folder. Absolute paths are accepted. Run all examples from the repository root.

Configurations are complete documents; there is no inheritance/overlay merge and no `--algorithm` or `--model_type` CLI flag. Select an algorithm through `train.model_type` in a complete config. `--run_name` names output artifacts; it does not select an algorithm.

## Supplied presets

| File | Algorithm | Processed directory |
|---|---|---|
| `config/config.yaml` | Random Forest (default) | `data/processed` |
| `config/random_forest.yaml` | Random Forest | `data/processed/random_forest` |
| `config/logistic_regression.yaml` | Logistic Regression | `data/processed/logistic_regression` |
| `config/xgboost.yaml` | XGBoost | `data/processed/xgboost` |

The three algorithm presets share input paths, sampling, feature order, split ratios and seed. Their output roots for models/reports/scalers/metadata are shared, with unique run subdirectories. Run training jobs sequentially because unversioned latest artifact copies are still shared.

Values below describe `config/config.yaml`. The algorithm presets change `train.model_type` and `data.processed_dir`; other initial settings match. Each backend ignores the other backends' parameter families. Class weights are computed from training data; there is no separate YAML class-weight override.

## `data`

| Key | Default | Meaning |
|---|---|---|
| `raw_dir` | `data/raw` | Directory containing only original labeled training CSVs; direct children only. |
| `processed_dir` | `data/processed` | Generated processed.csv and latest splits.json. Separate it by dataset/preset. |
| `chunksize` | `200000` | Rows per CSV read; does not cap total retained data or training memory. |
| `max_rows` | `0` | Random cap after cleaning and sample_frac; 0 disables the cap. Not stratified. |
| `sample_frac` | `1.0` | Random fraction retained after loading/cleaning, before max_rows; keep in (0,1]. Not stratified. |

## `split`

| Key | Default | Meaning |
|---|---|---|
| `train_ratio` | `0.7` | Training proportion after cleaning/sampling. Use a positive value. |
| `val_ratio` | `0.15` | Validation proportion for model selection. Use a positive value. |
| `test_ratio` | `0.15` | Test proportion for final assessment. All three ratios must sum to 1. |

## `features`

| Key | Default | Meaning |
|---|---|---|
| `log1p_skew_threshold` | `1.0` | Select log1p for nonnegative training features whose skew exceeds this value. |
| `fixed_list` | `canonical 15-feature list` | Ordered 15-feature list; must exactly match src/features/feature_list.py. |

## `train`

| Key | Default | Meaning |
|---|---|---|
| `model_type` | `random_forest` | Exactly random_forest, logistic_regression or xgboost. |
| `rf_n_estimators` | `80` | Random Forest tree count. |
| `rf_max_depth` | `10` | Random Forest depth limit; null or 0 means no explicit depth cap. |
| `rf_min_samples_leaf` | `20` | Minimum training samples required in a Random Forest leaf. |
| `rf_n_jobs` | `-1` | Random Forest worker count; -1 requests all available processors. |
| `xgb_n_estimators` | `300` | XGBoost estimator count. |
| `xgb_max_depth` | `8` | XGBoost maximum tree depth. |
| `xgb_learning_rate` | `0.1` | XGBoost boosting step size. |
| `xgb_subsample` | `0.8` | XGBoost row sampling fraction per boosting iteration. |
| `xgb_colsample_bytree` | `0.8` | XGBoost feature sampling fraction per tree. |
| `xgb_min_child_weight` | `1.0` | XGBoost minimum child Hessian-weight sum. |
| `xgb_reg_lambda` | `1.0` | XGBoost L2 regularization coefficient. |
| `xgb_n_jobs` | `-1` | XGBoost worker count; -1 requests all available processors. |
| `xgb_tree_method` | `hist` | Tree construction method; supplied preset uses hist. |
| `xgb_verbosity` | `1` | XGBoost logging verbosity. |
| `lr_c` | `1.0` | Logistic Regression inverse regularization strength. |
| `lr_max_iter` | `1000` | Logistic Regression solver iteration limit. |
| `lr_solver` | `lbfgs` | Logistic Regression solver; validated preset uses lbfgs. |
| `lr_penalty` | `l2` | Validated L2 default. With scikit-learn >=1.8, the factory maps none to C=infinity and otherwise leaves the default penalty; do not assume l1/elasticnet is implemented. |
| `lr_multi_class` | `auto` | Legacy compatibility key; passed only if the installed LogisticRegression signature supports it. |
| `lr_n_jobs` | `-1` | Legacy compatibility key; omitted on scikit-learn >=1.8. |
| `random_state` | `42` | Seed for dataframe sampling, stratified splitting and model randomness; retain across compared runs. |

## `artifacts`

| Key | Default | Meaning |
|---|---|---|
| `models_dir` | `artifacts/models` | Root for named model.joblib files and unversioned latest model copy. |
| `reports_dir` | `artifacts/reports` | Root for named report JSON/images and unversioned latest reports. |
| `scalers_dir` | `artifacts/scalers` | Root for named preprocessor.json files and latest copy. |
| `metadata_dir` | `artifacts/metadata` | Root for per-run config, manifest, splits, model metadata and latest metadata copies. |

## `stream`

| Key | Default | Meaning |
|---|---|---|
| `speed` | `5` | Replay event-rate target, not network packet rate. CLI --speed overrides it. |
| `min_confidence` | `0.5` | Minimum PC replay confidence included in emitted events. |
| `only_attacks` | `false` | Omit Normal predictions during replay; CLI flag can enable it, not disable a true config value. |
| `batch_size` | `1` | PC replay inference batch size; normalized to at least 1 by the command. |

## `dashboard`

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Flask bind host and default host used by local event producers; use 127.0.0.1 for the documented workflow. |
| `port` | `5000` | Flask TCP port and default event-producer destination port. |

## Custom experiment recipe

1. Copy one full preset to another YAML file.
2. Keep raw dataset, features, sampling, seed and ratios unchanged for algorithm/hyperparameter comparisons.
3. Change only the active backend's parameters and select a new run name.
4. For a different dataset, use a different raw directory and processed directory as well.
5. Supply the matching config to export and inference commands, especially when changing artifact roots. Named replay reads the run snapshot for reconstruction, while command paths and dashboard settings still come from the command's selected config.

Do not modify a saved run snapshot to conceal a dataset mismatch. New training is the correct way to establish a new experiment. Keep generated configs and manifests alongside the local run artifacts if they contain private machine paths. See [dataset and split ownership](dataset-and-splits.md) for the exact conflict rules.
