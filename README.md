# NIDS PC Prototype (CSE-CIC-IDS2018)

This prototype ingests the CSE-CIC-IDS2018 Intrusion CSVs, maps labels into 5 classes, applies log1p + MinMax scaling on a fixed feature list, trains a configurable classifier backend (XGBoost, Random Forest, or Logistic Regression), evaluates metrics, and runs a Flask dashboard with a streaming simulator.

## Quick Start

1. Install dependencies

```bash
cd nids_pc_prototype
pip install -r requirements.txt
```

2. Place IDS2018 Intrusion CSVs in `data/raw/`

3. Train + export artifacts

```bash
python -m src.main_train
```

Optional run name (recommended for comparisons):

```bash
python -m src.main_train --run_name xgb_run_01
```

4. Start dashboard server

```bash
python -m src.main_serve
```

5. Stream test set events to the dashboard

```bash
python -m src.main_stream --speed 5 --min_confidence 0.5
```

To stream with a specific trained run:

```bash
python -m src.main_stream --run_name rf_run_01 --speed 5
```

Dashboard: http://127.0.0.1:5000

## Notes

- The fixed feature order is defined in `src/features/feature_list.py` and must be enforced everywhere.
- The preprocessor (log1p + MinMax) is saved to `artifacts/scalers/preprocessor.json`.
- The trained model is saved to `artifacts/models/model.joblib`.
- Reports, confusion matrix plots, ROC curves, and PR curves are saved to `artifacts/reports/`.
- Each training run is also versioned under:
  - `artifacts/models/<run_name>/`
  - `artifacts/scalers/<run_name>/`
  - `artifacts/reports/<run_name>/`
  - `artifacts/metadata/<run_name>/`
- The cleaned and processed dataset is saved to `data/processed/processed.csv` to support streaming.
- Class imbalance is handled with class weights computed from the training split.
- For `xgboost`, those class weights are converted into per-row `sample_weight` during fitting.

## Config

Edit defaults in `config/config.yaml`:

- data paths
- chunk size and sampling
- split ratios
- model backend selection (`xgboost`, `random_forest`, or `logistic_regression`)
- Random Forest, XGBoost, and Logistic Regression hyperparameters
- dashboard host/port

Example:

```yaml
train:
  model_type: xgboost
  random_state: 42
  rf_n_estimators: 200
  rf_max_depth: 20
  rf_min_samples_leaf: 2
  rf_n_jobs: -1
  xgb_n_estimators: 300
  xgb_max_depth: 8
  xgb_learning_rate: 0.1
  xgb_subsample: 0.8
  xgb_colsample_bytree: 0.8
  xgb_min_child_weight: 1.0
  xgb_reg_lambda: 1.0
  xgb_n_jobs: -1
  xgb_tree_method: hist
  xgb_verbosity: 1
  lr_c: 1.0
  lr_max_iter: 1000
  lr_solver: lbfgs
  lr_penalty: l2
  lr_multi_class: auto
  lr_n_jobs: -1
```

## Label Mapping

- Normal: BENIGN -> Normal
- DoS: labels containing "DoS", "DDoS", "HOIC", "LOIC"
- BruteForce: labels containing "Brute Force", "BruteForce", "SSH-Bruteforce", "FTP-Bruteforce"
- Botnet: labels containing "Bot", "Botnet"
- WebAttack: labels containing "Web Attack", "SQL Injection", "XSS"

All other labels are dropped (counts are logged).
