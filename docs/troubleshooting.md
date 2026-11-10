# Troubleshooting

[Documentation index](README.md) · [Dataset and splits](dataset-and-splits.md)

| Symptom | Cause and action |
|---|---|
| `No module named src` | Run from the repository root and use `python -m src.main_train`, not `python src/main_train.py`. Activate the intended environment. |
| Missing Python dependency | Core: `python -m pip install -r requirements.txt`; capture/UART: `requirements-hardware.txt`; optional exports: `requirements-export.txt`; XGBoost: `requirements-xgboost.txt`. Use the same interpreter for installation and commands. |
| emlearn reports missing `distutils` on Python 3.13 | Install the complete `requirements-export.txt`, including the pinned setuptools compatibility dependency. Installing emlearn alone is insufficient in this environment. |
| PowerShell blocks environment activation | Use `.venv/Scripts/python.exe` instead of `python`, or follow your machine's environment activation policy. Do not change system policy just to run the project. |
| Raw directory is missing or no CSVs are found | Create `data/raw/` and extract labeled CSVs directly inside it. Check `data.raw_dir`; relative data paths resolve against the repository root. ZIP files and nested folders are not scanned. |
| `Label column not found` | The input is not the required labeled training CSV, or its schema differs. Do not place flow exports or generated processed data in the training input directory. |
| Missing required features | Compare headers with the canonical 15-feature list. Whitespace is trimmed but feature spelling/case must match. Another flow extractor may use incompatible feature definitions even after renaming. |
| Rows are unexpectedly dropped | Unsupported labels and nonfinite/nonnumeric feature values are dropped. Inspect the training log and label mapping; do not silently relabel unknown attacks as benign. |
| Stratification complains about class counts or split size | A class has too few surviving examples for one of the two splits. Restore all required classes, reduce aggressive sampling, or collect more examples. Do not remove stratification merely to hide the error. |
| Model probabilities lack expected classes, or XGBoost rejects labels | The workflow expects all five mapped classes. Check raw label coverage and cleaning/sampling; a random row cap can remove rare classes. |
| Memory error even with `max_rows` set | The loader retains cleaned chunks before applying the cap. `chunksize` limits an individual read, not the total dataframe. Use an adequate-memory environment or a separately documented input subset; give the new dataset/selection new run names. |
| Run artifact directory already exists | Run names are immutable identifiers. Choose `rf_v02`, `lr_v02`, etc. A failed training job can leave partial directories; inspect them before any manual cleanup. |
| Raw dataset differs from the training run | Files were renamed, modified, added or removed. Restore the exact originals or train a new run. Do not edit the fingerprint manifest to bypass the check. |
| Legacy run warnings or out-of-range split indices | Older artifacts lack trustworthy per-run config/split snapshots. They may also reflect the earlier duplicate CSV loading behavior on Windows. Retrain with the current loader and an explicit new run name. Do not reuse old indices with a reconstructed dataset of a different length. |
| Model/preprocessor/split not found during export or replay | Train first. Supply the correct `--run_name` and a config whose artifact directories point to that run. Do not combine files from different runs. |
| Restored run fails on another machine | Saved configuration may contain absolute paths to the original raw dataset/artifact directories. Restore the intended layout and inspect its snapshot rather than silently substituting unrelated inputs. |
| Logistic Regression reaches its iteration limit | Inspect the data and validation behavior; increase `lr_max_iter` in a new config/run if appropriate. A convergence warning is not evidence of a good model. |
| ONNX export fails for XGBoost | This repository has no converter for its custom XGBoost wrapper. Use supported scikit-learn models for this export route. |
| Plain-C/emlearn/m2cgen exporter rejects a model | These commands accept Random Forest only. Logistic Regression and XGBoost remain PC baselines for those routes. |
| Plain-C tree exceeds int16 node-index capacity | Train a smaller forest/tree depth in a new run. A successful export still needs board memory and timing measurements. |
| ONNX comparison prints mismatches | Inspect class order, preprocessing and numerical differences. The comparison prints diagnostics; it does not automatically fail its process on disagreement. Do not treat exit status alone as export verification. |
| Dashboard backend unreachable | Start `python -m src.main_serve` in another terminal; match host and port in both configs. Use `127.0.0.1` to connect locally. |
| Dashboard totals seem too high | Events accumulate from previous replays and include received predictions. Restart the server between separate demonstrations. The store is in memory, so restart also clears events. |
| Dashboard tables work but chart is missing | Chart.js is loaded from a CDN. Check browser network access and console errors. |
| Capture interface/filter fails | Install Npcap on Windows, verify the interface name and capture permissions, and use an appropriate filter. Offline Python filtering supports only simple IP/TCP/UDP expressions. |
| UART port cannot open | Use the actual board port, close serial monitors that hold it, and verify the board connection/driver. Linux ports typically use paths such as `/dev/ttyACM0`; `COM5` is just a Windows example. |
| UART timeouts, invalid predictions or row mismatches | Verify baud rate, selected UART handle, flashed adapter/model, line format and shared feature/class order. Record skipped rows rather than treating missing replies as correct predictions. |
| Device accuracy differs sharply from PC | Check raw versus scaled input mode, generated preprocessing header, model version and class map. Default plain-C firmware expects raw inputs and host `--device_preprocess`; a host-preprocessed firmware build requires omitting that flag. |
| UART `--limit 0` fails | UART evaluation uses a positive limit, not zero-as-unlimited. Choose a deliberate row count; it is capped to the split length. |
| C test is skipped | GCC was not found. Install an appropriate compiler if you want host C equivalence and firmware compilation checks. Board tests still require STM32 hardware/tooling. |
| Publication checker reports local-only files | Remove those files from Git tracking while keeping local copies, and resolve any excluded paths in published history. Do not force-add datasets, notebooks, captures, models or split files. |

## Useful diagnostics

```powershell
python --version
python -m pip check
python -m src.main_train --help
python -m src.main_rfc_uart_eval --help
git status --short
git check-ignore data/raw/Bot.csv artifacts/models/rf_baseline/model.joblib
python tools/check_publication.py --history
```

When reporting a problem, include the command, Python version, config file name, run name, and the relevant error. Share configuration structure and aggregate logs; keep raw traffic, dataset rows, model files, credentials and notebooks local.
