# Command reference

[Documentation index](README.md) · [Training walkthrough](training.md)

Run commands from the repository root, with the intended Python environment activated. `rf_baseline` refers to a run created with the Random Forest preset in the training guide. `COM5`, `Ethernet`, and capture paths are examples: replace them with your hardware/interface/local files.

Every command supports `--help` once its listed dependencies are installed; some optional imports occur before argument parsing. Tables list the options defined by the current argument parsers; `--help` is implicit. A default of `None` means unset, not a literal string to pass. Boolean switches are enabled by including the flag without a value. See each command's notes for zero-limit semantics; they are not uniform across commands. The export requirements include setuptools because emlearn needs its distutils compatibility module on Python 3.13.

## `src.main_compare_joblib_onnx`

Requires requirements-export.txt, model.joblib, an exported model.onnx, preprocessor, split records and raw data. Compares a bounded test slice, prints label/probability agreement and mismatches. It does not turn mismatches into a failing exit status.

```powershell
python -m src.main_compare_joblib_onnx --config config/random_forest.yaml --run_name rf_baseline --limit 1000
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Run name, e.g. rf_run_04 |
| `--limit` | `1000` | How many test rows to compare |
| `--start_index` | `0` | Offset inside the test split |

## `src.main_export_emlearn`

Requires requirements-export.txt; RandomForestClassifier only. Writes a generated model header, copied preprocessor, class map and metadata under `artifacts/embedded/<run>`. Select loadable or inline export method. Verify actual generated runtime input types and the selected preprocessing mode before deployment.

```powershell
python -m src.main_export_emlearn --run_name rf_baseline
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Trained RF run to export, e.g. rf_run_04 |
| `--model_name` | `nids_model` | C identifier for the exported model |
| `--output_dir` | `None` | Output directory. Defaults to `artifacts/embedded/<run_name>` |
| `--method` | `loadable` | emlearn export method Choices: loadable, inline. |

## `src.main_export_m2cgen`

Requires requirements-export.txt; RandomForestClassifier only. Writes m2c_model.c, m2c_model_wrapper.h, class_map.json and m2cgen_export.json under `artifacts/embedded_m2c/<run>`. It does not generate a separate m2c_model.h or a complete HAL application. Use the generated wrapper/scorer contract when integrating.

```powershell
python -m src.main_export_m2cgen --run_name rf_baseline
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Trained RF run, e.g. rf_run_04 |
| `--model_name` | `m2c_model` | C identifier for the model |
| `--output_dir` | `None` | Output directory. Defaults to `artifacts/embedded_m2c/<run_name>` |

## `src.main_export_onnx`

Requires requirements-export.txt and a supported scikit-learn model. Writes model.onnx under the selected models run and onnx_export.json under metadata. Inputs must already be preprocessed. Outputs use integer class IDs and a separate class-name mapping. The custom XGBoost wrapper is not supported by this exporter.

```powershell
python -m src.main_export_onnx --run_name rf_baseline
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Trained run to export, e.g. rf_run_03 |
| `--input_name` | `float_input` | ONNX input tensor name |
| `--target_opset` | `17` | ONNX opset version to export |

## `src.main_export_rf_c`

Core dependencies; RandomForestClassifier only. Writes rf_model.h/.c, rf_preprocessor.h, preprocessor.json, class_map.json and rf_c_export.json under `artifacts/embedded_c/<run>` by default. The supplied firmware assumes --model_name rf_model. Changing the name requires adapting firmware includes/symbols. Node-index capacity is checked; hardware fit is not guaranteed.

```powershell
python -m src.main_export_rf_c --run_name rf_baseline
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Trained RF run to export, e.g. rf_run_04 |
| `--model_name` | `rf_model` | Base C identifier/file name for the export |
| `--output_dir` | `None` | Output directory. Defaults to `artifacts/embedded_c/<run_name>` |

## `src.main_export_test_split`

Core dependencies, raw data and split records. --preprocessed requires that run's scaler. By default, the CSV includes labels and optional timestamps. --cubemx removes the header, labels and timestamp, leaving just 15 numeric inputs; it does not itself apply preprocessing unless --preprocessed is also set. Outputs are written under the selected processed directory, or --output.

```powershell
python -m src.main_export_test_split --config config/random_forest.yaml --run_name rf_baseline --preprocessed --cubemx
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Run name used for preprocessing metadata |
| `--output` | `None` | Output CSV path. Defaults to data/processed/test_split_<run_name>.csv |
| `--preprocessed` | `False` | Export scaled/log1p-transformed features instead of raw cleaned feature values |
| `--cubemx` | `False` | Export only the 15 model input columns, ready for X-CUBE-AI desktop validation |

## `src.main_flowmeter`

Core dependencies and a trained run. Reads an existing feature CSV, or invokes an externally installed CICFlowMeter through --pcap plus --cic_cmd. The command template must contain {pcap} and {out} placeholders and quote paths as required by your shell. No universal CICFlowMeter executable or JAR is bundled.

```powershell
python -m src.main_flowmeter --run_name rf_baseline --flows_csv data/processed/live_flows_15.csv --dashboard
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `None` | Use artifacts from a specific run (e.g., rf_run_03) |
| `--flows_csv` | `None` | Path to CICFlowMeter flow CSV to run inference on |
| `--pcap` | `None` | Optional pcap path (if you want this script to run CICFlowMeter) |
| `--cic_cmd` | `None` | Command template to run CICFlowMeter. Use {pcap} and {out} placeholders. |
| `--cic_out_dir` | `data/processed/cicflowmeter_out` | Output dir for CICFlowMeter |
| `--min_confidence` | `0.0` |  |
| `--only_attacks` | `False` |  |
| `--dashboard` | `False` | Post events to Flask dashboard API |
| `--quiet` | `False` | Disable per-flow console output |
| `--limit` | `0` | Max rows to process (0 means unlimited) |

## `src.main_flowmeter_sniff`

Requires requirements-hardware.txt. Writes/appends per-flow CSV rows and metadata without loading a classifier. Use --pcap instead of --iface for offline input. Outputs stay outside data/raw/; appending to an existing file does not clear prior rows.

```powershell
python -m src.main_flowmeter_sniff --iface "Ethernet" --bpf "tcp or udp" --csv_out data/processed/live_flows_15.csv
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--iface` | `None` | Network interface for live sniffing |
| `--pcap` | `None` | Read packets from a pcap file instead of live interface |
| `--bpf` | `ip` | BPF filter (e.g., 'tcp or udp') |
| `--packet_count` | `0` | Stop after N packets (0 means unlimited) |
| `--sniff_timeout` | `0.0` | Sniff duration in seconds (0 means none) |
| `--flow_timeout` | `2.0` | Flow idle timeout in seconds |
| `--max_flow_duration` | `0.0` | If >0, force-rotate long-lived flows after this many seconds. |
| `--min_packets` | `2` | Minimum packets before exporting a flow row |
| `--csv_out` | `data/processed/live_flows_15.csv` | CSV file to append exported flows |
| `--quiet` | `False` | Disable per-flow console output |

## `src.main_m2cgen_uart_eval`

Requires requirements-hardware.txt and a board explicitly integrated with the m2cgen scorer. Uses the same evaluator/flags as plain C; default report name is m2c_<train_run_name>. The shared --device_preprocess flag is usable only if your custom firmware implements the matching raw-input transform. Probability plots are PC reference results.

```powershell
python -m src.main_m2cgen_uart_eval --config config/random_forest.yaml --train_run_name rf_baseline --port COM5 --limit 500 --report_run_name board_m2c_01
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` |  |
| `--train_run_name` | `required` |  |
| `--report_run_name` | `None` |  |
| `--port` | `required` |  |
| `--baudrate` | `115200` |  |
| `--timeout` | `3.0` |  |
| `--limit` | `500` |  |
| `--start_index` | `0` |  |
| `--delay_ms` | `0.0` |  |
| `--skip_val` | `False` |  |
| `--device_preprocess` | `False` | Send raw features; firmware must apply this run's preprocessing |

## `src.main_rfc_uart_eval`

Requires requirements-hardware.txt and a flashed plain-C RF board. Shared evaluator handles validation then test, with accepted/skipped row records. --skip_val selects test only. --limit must be positive and applies to each split. Default report name is rfc_<train_run_name>; choose another name to preserve separate board trials. Accuracy/F1 come from device replies; probability plots use PC reference scores.

```powershell
python -m src.main_rfc_uart_eval --config config/random_forest.yaml --train_run_name rf_baseline --port COM5 --device_preprocess --limit 500 --report_run_name board_rf_01
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` |  |
| `--train_run_name` | `required` |  |
| `--report_run_name` | `None` |  |
| `--port` | `required` |  |
| `--baudrate` | `115200` |  |
| `--timeout` | `3.0` |  |
| `--limit` | `500` |  |
| `--start_index` | `0` |  |
| `--delay_ms` | `0.0` |  |
| `--skip_val` | `False` |  |
| `--device_preprocess` | `False` | Send raw features; firmware must apply this run's preprocessing |

## `src.main_serve`

Core dependencies. Runs the local Flask server using the selected host/port; defaults to 127.0.0.1:5000. Events are stored in memory and cleared on restart.

```powershell
python -m src.main_serve
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |

## `src.main_sniff`

Requires requirements-hardware.txt, a trained model/scaler, and capture permissions/Npcap for live capture. --pcap uses a local capture rather than a live interface. Supports simple Python IP/TCP/UDP offline filters; other expressions can fall back to best-effort processing. --dashboard requires the server to be running. Scapy feature equivalence to CICFlowMeter is unverified.

```powershell
python -m src.main_sniff --run_name rf_baseline --pcap data/captures/example.pcap --bpf "tcp or udp" --dashboard
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `None` | Use artifacts from a specific run (e.g., rf_run_03) |
| `--iface` | `None` | Network interface for live sniffing |
| `--pcap` | `None` | Read packets from a pcap file instead of live interface |
| `--bpf` | `ip` | BPF filter (e.g., 'tcp or udp') |
| `--packet_count` | `0` | Stop after N packets (0 means unlimited) |
| `--sniff_timeout` | `0.0` | Sniff duration in seconds (0 means none) |
| `--flow_timeout` | `2.0` | Flow idle timeout in seconds |
| `--max_flow_duration` | `0.0` | If >0, force-rotate flows after this many seconds (active timeout). |
| `--min_packets` | `2` | Minimum packets before flow inference |
| `--active_infer_interval` | `0.0` | If >0, also run inference periodically on active flows every N seconds. |
| `--min_confidence` | `0.0` | Drop predictions below this confidence |
| `--only_attacks` | `False` | Skip Normal predictions |
| `--dashboard` | `False` | Post events to Flask dashboard API |
| `--quiet` | `False` | Disable per-flow console output |

## `src.main_stm32_uart_test`

Requires requirements-hardware.txt, a flashed board, model/scaler/metadata/split records and original raw data. Replays test rows and optionally posts device predictions to the dashboard; it prints accuracy but does not generate the full val/test report set. --device_preprocess sends raw features; otherwise the host scales them.

```powershell
python -m src.main_stm32_uart_test --config config/random_forest.yaml --run_name rf_baseline --port COM5 --device_preprocess --limit 100 --dashboard
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `required` | Trained run name, e.g. rf_run_04 |
| `--port` | `required` | UART port, e.g. COM6 |
| `--baudrate` | `115200` | UART baudrate |
| `--timeout` | `3.0` | Serial read timeout in seconds |
| `--limit` | `100` | How many test rows to send |
| `--start_index` | `0` | Start offset inside the test split |
| `--delay_ms` | `0.0` | Delay between rows in milliseconds |
| `--dashboard` | `False` | Post STM32 predictions to the dashboard |
| `--dashboard_url` | `None` | Override dashboard events URL |
| `--device_preprocess` | `False` | Send raw cleaned features to STM32; use this when STM32 performs preprocessing on-device |

## `src.main_stream`

Requires a trained local run, raw CSVs for named replay, and a running dashboard. Prefer a named run; without one, the command uses latest processed/model files. It has no --limit option: stop a long replay with Ctrl+C. Filters affect which events are emitted and counted by console comparisons.

```powershell
python -m src.main_stream --config config/random_forest.yaml --run_name rf_baseline --speed 5
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `None` | Use artifacts from a specific run (e.g., rf_run_01) instead of latest files |
| `--speed` | `None` | Events per second |
| `--min_confidence` | `None` |  |
| `--only_attacks` | `False` |  |
| `--batch_size` | `None` |  |
| `--quiet` | `False` | Disable per-flow console output |

## `src.main_train`

Core dependencies; XGBoost additionally needs requirements-xgboost.txt. Training automatically creates both validation and test reports. New run names are mandatory; data/processed latest outputs and unversioned artifact copies can be overwritten.

```powershell
python -m src.main_train --config config/random_forest.yaml --run_name rf_baseline
```

| Option | Default / requirement | Description |
|---|---|---|
| `--config` | `None` | Path to config.yaml |
| `--run_name` | `None` | Optional run name for versioned artifacts |

## Repository checks

```powershell
python -m unittest discover -s tests -v
python tools/check_publication.py --history
```

Install requirements-hardware.txt before the full test suite; optional C checks need GCC. Fixtures are generated locally and do not load your dataset or trained models. The publication command checks currently publishable paths and, with --history, all reachable Git history paths. It also checks publishable small files for the known Kaggle credential pattern. It is a repository-specific guard, not a general secret scanner.
