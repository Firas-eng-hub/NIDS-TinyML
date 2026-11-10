# STM32 Network Intrusion Detection Prototype

A research prototype for training a network-flow classifier on CSE-CIC-IDS2018 CSVs, exporting a Random Forest to C, and evaluating STM32 predictions over UART. A local Flask dashboard displays PC or device events.

The supported workflow uses **15 features and five classes**: Normal, DoS, BruteForce, Botnet, and WebAttack. Feature order is defined in `src/features/feature_list.py` and checked against the configuration. Device class IDs follow the exported `class_map.json`, not the order in this paragraph.

## Repository contents

```text
config/                 Training, data paths and dashboard defaults
src/data/               CSV cleaning, label mapping, splits and replay
src/features/           Canonical feature order and preprocessing
src/model/              Model backends, training, evaluation and persistence
src/uart/               Shared UART protocol and device evaluation
src/dashboard/          Flask API and dashboard assets
src/flows.py            Shared Scapy flow aggregation
src/main_*.py           Command-line entry points
firmware/               STM32 UART adapters and integration instructions
tests/                  Dataset-free regression and C export tests
docs/                   Reproducibility and evaluation limitations
```

Datasets, processed rows, split indices, packet captures, models, generated exports, experiment outputs, and **all notebooks** are local-only and excluded from Git. Training creates the necessary directories. No model or dataset is bundled with a fresh checkout.

## Installation

Use Python **3.13**; this is the version used for local validation. From the repository root:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Install optional capabilities as needed:

```powershell
python -m pip install -r requirements-hardware.txt  # UART and Scapy capture
python -m pip install -r requirements-export.txt    # ONNX, emlearn and m2cgen
python -m pip install -r requirements-xgboost.txt   # Optional training backend
```

The custom plain-C Random Forest exporter uses only the core dependencies. Live capture on Windows requires Npcap and appropriate capture permissions. GCC is used by the C equivalence tests; STM32 firmware requires your board's STM32Cube toolchain and HAL.

## Dataset preparation and training

Obtain the CSE-CIC-IDS2018 Intrusion CSV files separately and place them directly in `data/raw/`, or configure `data.raw_dir`. Do not put download credentials in source files.

Cleaning trims column names, selects the 15 configured features, drops unsupported labels and nonfinite values, and converts features to float32. `BENIGN` maps to Normal; DoS/DDoS variants map to DoS; brute-force variants map to BruteForce; Bot maps to Botnet; SQL injection/XSS/web attack labels map to WebAttack. See `src/data/label_map.py` for the exact matching rules. Unsupported labels are dropped. Raw negative values are not currently rejected; see the limitations document.

```powershell
python -m src.main_train --run_name rf_demo
```

Edit `config/config.yaml` before training. The default backend is Random Forest. Logistic Regression is also available; XGBoost requires its optional dependency. Reduce `data.max_rows` or `data.sample_frac` for an experiment, but note that the current loader still gathers the cleaned dataset in memory before this sampling.

Training fits log1p feature selection and MinMax parameters on the training split only. It saves local model, preprocessor, metrics and metadata under `artifacts/`, plus `data/processed/processed.csv` and `splits.json` for local replay.

New runs also save their configuration, raw-file SHA-256 manifest and split indices under `artifacts/metadata/<run_name>/`. Named replay and evaluation commands use those records and reject changed dataset files. Old local runs without these records emit a warning and use the legacy shared configuration/split files; their original correspondence cannot be guaranteed.

## Export and STM32 integration

The primary deployment route is the custom plain-C Random Forest exporter:

```powershell
python -m src.main_export_rf_c --run_name rf_demo
```

It writes `rf_model.h`, `rf_model.c`, `rf_preprocessor.h`, `preprocessor.json`, `class_map.json` and export metadata into `artifacts/embedded_c/rf_demo/`. These generated files remain local.

Follow [firmware/README.md](firmware/README.md) to integrate the model and UART adapter into your own STM32Cube project. The repository provides adapter source, not a complete board-specific firmware project. The primary adapter preprocesses raw float inputs on the device by default.

Evaluate a flashed board using its actual serial port:

```powershell
python -m src.main_rfc_uart_eval --train_run_name rf_demo --port COM5 --device_preprocess --limit 500
```

The evaluator sends validation and test rows, records accepted/skipped responses, and writes local reports. Accuracy, F1 and confusion matrices use device replies. **ROC/PR curves use PC joblib probabilities** and are labeled as PC reference results; they do not measure device probability output.

For a firmware build expecting already-preprocessed inputs, omit `--device_preprocess`. The firmware and host modes must agree.

## Dashboard and replay

Start the dashboard in one terminal:

```powershell
python -m src.main_serve
```

Open `http://127.0.0.1:5000`. In another terminal, replay PC predictions:

```powershell
python -m src.main_stream --run_name rf_demo --speed 5
```

Or post device predictions while replaying test rows:

```powershell
python -m src.main_stm32_uart_test --run_name rf_demo --port COM5 --device_preprocess --limit 100 --dashboard
```

The dashboard is a local demonstration with in-memory events. Its counts describe received predictions, not independently verified attacks.

## Live capture and flow CSV inference

```powershell
python -m src.main_sniff --run_name rf_demo --iface "Ethernet" --bpf "tcp or udp" --dashboard
python -m src.main_flowmeter_sniff --iface "Ethernet" --bpf "tcp or udp" --csv_out data/processed/live_flows_15.csv
python -m src.main_flowmeter --run_name rf_demo --flows_csv data/processed/live_flows_15.csv --dashboard
```

`main_sniff` and `main_flowmeter_sniff` also accept `--pcap` for local offline replay. `main_flowmeter` can invoke an external CICFlowMeter installation through `--cic_cmd`; use `--help` for its command template. Custom Scapy features have not been established as equivalent to the CICFlowMeter features in the training dataset. Live detection accuracy is therefore unverified.

## Alternative exports

```powershell
python -m src.main_export_onnx --run_name rf_demo
python -m src.main_compare_joblib_onnx --run_name rf_demo --limit 1000
python -m src.main_export_emlearn --run_name rf_demo
python -m src.main_export_m2cgen --run_name rf_demo
python -m src.main_m2cgen_uart_eval --train_run_name rf_demo --port COM5 --limit 500
```

These are optional export paths. ONNX export is intended for supported scikit-learn models; the XGBoost wrapper has no corresponding ONNX converter here. emlearn and m2cgen exports are restricted to Random Forest. Export success alone does not demonstrate STM32 memory fit or numerical agreement.

For external desktop validation, export test inputs locally:

```powershell
python -m src.main_export_test_split --run_name rf_demo --preprocessed --cubemx
```

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests use small generated fixtures, never the local dataset or trained model files. GCC-dependent tests are skipped when GCC is unavailable. Hardware UART behavior and board memory/timing require separate device testing.

See [docs/limitations.md](docs/limitations.md) for the interpretation of metrics and deployment constraints.
