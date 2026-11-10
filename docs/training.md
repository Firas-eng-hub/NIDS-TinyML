# Train, compare and test algorithms

[Documentation index](README.md) · [Dataset and splits](dataset-and-splits.md) · [Configuration reference](configuration.md)

## 1. Install and prepare inputs

From a new clone:

```powershell
git clone https://github.com/Firas-eng-hub/NIDS-TinyML.git
cd NIDS-TinyML
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.13 is the validated environment. On Linux/macOS, activate with `source .venv/bin/activate`. If PowerShell activation is blocked, run `.venv/Scripts/python.exe` in place of `python`; activation is not required when the interpreter is addressed directly.

Place the original labeled CSVs directly in `data/raw/`, following the [dataset guide](dataset-and-splits.md). You do not need downloaded models, preprocessed files or split indices. They are generated locally.

## 2. Train Random Forest

```powershell
python -m src.main_train --config config/random_forest.yaml --run_name rf_baseline
```

This trains the model **and evaluates both validation and test splits**. There is no separate PC test command required after training. The default Random Forest preset uses 80 trees, maximum depth 10 and at least 20 samples per leaf. This is a starting configuration, not a claim of optimal accuracy or fit on your STM32.

## 3. Train Logistic Regression

```powershell
python -m src.main_train --config config/logistic_regression.yaml --run_name lr_baseline
```

It uses the same input features, data selection, split ratios and seed as Random Forest. Its preset uses `C=1.0`, `lbfgs`, and up to 1,000 iterations. The implemented classifier is a PC baseline; the provided plain-C, emlearn and m2cgen commands accept Random Forest only.

## 4. Train XGBoost

Install its optional dependency first:

```powershell
python -m pip install -r requirements-xgboost.txt
python -m src.main_train --config config/xgboost.yaml --run_name xgb_baseline
```

The XGBoost preset uses 300 estimators, maximum depth 8, learning rate 0.1 and histogram tree construction. The wrapper preserves string class labels for inference and applies balanced per-row weights at training time. No XGBoost-to-STM32 exporter is implemented in this repository.

Run these three jobs sequentially. They share unversioned latest artifact copies, so parallel training is not supported. Each run name must be new. If `rf_baseline` already exists, choose `rf_v02`; do not overwrite a run directory or reuse its name after a failed attempt without inspecting its partial outputs.

## 5. Read the results

For `rf_baseline`, the output locations are:

| Output | Path |
|---|---|
| Serialized model | `artifacts/models/rf_baseline/model.joblib` |
| Preprocessor | `artifacts/scalers/rf_baseline/preprocessor.json` |
| Configuration, splits and dataset fingerprints | `artifacts/metadata/rf_baseline/` |
| Validation metrics | `artifacts/reports/rf_baseline/val_report.json` |
| Test metrics | `artifacts/reports/rf_baseline/test_report.json` |
| Confusion matrix, ROC and PR images | `artifacts/reports/rf_baseline/val_*.png` and `test_*.png` |
| Latest processed rows for this preset | `data/processed/random_forest/processed.csv` |

The other algorithms use their own run names and corresponding processed folders. All outputs are ignored by Git.

Read JSON with an editor or PowerShell:

```powershell
Get-Content artifacts/reports/rf_baseline/val_report.json -Raw
```

| Metric | Interpretation |
|---|---|
| `accuracy` | Fraction of correctly classified rows; frequent classes dominate this value |
| `macro_f1` | Mean F1 across the five target classes, giving each class equal weight |
| `per_class` | Precision, recall, F1 and support for individual classes, plus aggregate entries |
| `confusion_matrix` | Rows are true classes, columns are predicted classes |
| `roc_auc_by_class` / `pr_auc_by_class` | One-vs-rest probability curves; absent/undefined classes cannot support all curve calculations |

Training report confusion matrices follow `Normal, DoS, BruteForce, Botnet, WebAttack`. Device reports follow the trained model's class order. Read exported `class_map.json` or `metadata.json` instead of assuming numeric device IDs match the training report order. The Random Forest model normally sorts its string class labels; the XGBoost wrapper uses its configured class order.

Compare **validation** results in PowerShell:

```powershell
$rows = foreach ($run in 'rf_baseline', 'lr_baseline', 'xgb_baseline') {
    $report = Get-Content "artifacts/reports/$run/val_report.json" -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $run
        Accuracy = [math]::Round($report.accuracy, 4)
        MacroF1 = [math]::Round($report.macro_f1, 4)
    }
}
$rows | Sort-Object MacroF1 -Descending | Format-Table -AutoSize
```

Compare class recall and confusion matrices as well. High overall accuracy can hide a weak minority attack class. The script writes test reports for every run, but use validation results for model selection and consult the selected model's test result at the end. Repeatedly tuning against test performance weakens its role as an independent assessment.

For a fair algorithm comparison, keep raw CSV content/names, feature order, sampling, split ratios, seed and dependency versions unchanged. Split indices should match across the baseline runs. Check locally:

```powershell
Get-FileHash artifacts/metadata/rf_baseline/splits.json
Get-FileHash artifacts/metadata/lr_baseline/splits.json
Get-FileHash artifacts/metadata/xgb_baseline/splits.json
```

Matching hashes establish matching index lists, not matching datasets by themselves; the dataset manifests must also match.

## 6. Tune a model safely

Copy a full preset, change the relevant `train` settings, and give the result a new run name:

```powershell
Copy-Item config/random_forest.yaml config/rf_tuned.yaml
# Edit config/rf_tuned.yaml before running the next command.
python -m src.main_train --config config/rf_tuned.yaml --run_name rf_tuned_01
```

For example, test `rf_n_estimators: 40`, `rf_max_depth: 8` and `rf_min_samples_leaf: 20` to study a smaller forest. Measure validation performance and actual board resource use; smaller parameter values do not guarantee a deployable model. Keep data/split settings fixed while tuning to make comparisons meaningful.

Only the chosen algorithm's parameter family is used. `rf_*` settings do not tune XGBoost. See the [configuration reference](configuration.md) for supported keys and version limitations.

For a preliminary reduced-data experiment, edit all compared presets consistently:

```yaml
data:
  # Keep raw_dir, processed_dir and chunksize from the full preset.
  sample_frac: 1.0
  max_rows: 100000
```

This is an example cap, not a memory estimate or quality recommendation. The loader gathers all cleaned rows **before** applying the cap, so `max_rows` does not bound ingestion memory. Random sampling can remove rare classes. Changing the cap creates a different experiment and requires new run names.

## 7. Replay PC predictions

Install hardware dependencies if you will also capture traffic or use UART:

```powershell
python -m pip install -r requirements-hardware.txt
```

Start the dashboard in terminal 1:

```powershell
python -m src.main_serve
```

Open `http://127.0.0.1:5000`. Activate the same environment in terminal 2 and choose the model to replay:

```powershell
python -m src.main_stream --config config/random_forest.yaml --run_name rf_baseline --speed 5
python -m src.main_stream --config config/logistic_regression.yaml --run_name lr_baseline --speed 5
python -m src.main_stream --config config/xgboost.yaml --run_name xgb_baseline --speed 5
```

Run one replay at a time and stop it with Ctrl+C when finished. Replays use the saved test split and require the original raw files. Dashboard events accumulate in memory until the server restarts; mixing replays mixes their displayed counts. Console accuracy can also reflect confidence/attack filters; use the unfiltered saved training reports for algorithm comparison.

## 8. Export and test on STM32

Select a Random Forest run for the supplied plain-C firmware route:

```powershell
python -m src.main_export_rf_c --config config/random_forest.yaml --run_name rf_baseline
```

Follow [firmware integration](../firmware/README.md), copy generated files from that same run, build and flash the board, then evaluate it using its actual port:

```powershell
python -m src.main_rfc_uart_eval --config config/random_forest.yaml --train_run_name rf_baseline --port COM5 --device_preprocess --limit 500
```

This tests up to 500 validation rows and 500 test rows. It is not the full split unless the split is that small; increase `--limit` deliberately for more coverage. UART evaluators require a **positive** limit, so `--limit 0` does not mean all rows. `--skip_val` evaluates test only. `--report_run_name board_rf_01` keeps different board evaluations in separate report folders.

Device accuracy/F1 use accepted device replies; communication failures are recorded as skipped rows. ROC/PR use explicitly labeled **PC reference probabilities**, not probabilities read from STM32. See [command reference](commands.md) for other exports and testing modes.

## 9. Verify the software without the dataset

```powershell
python -m pip install -r requirements-hardware.txt
python -m unittest discover -s tests -v
python tools/check_publication.py --history
```

Tests use generated fixtures. GCC enables host C prediction/preprocessing checks and firmware compilation against test HAL declarations. These checks do not require or validate a physical board. The publication checker rejects local-only files in publishable files and reachable history. Do not force-add ignored datasets or outputs.
