# User and developer documentation

Start with [training and algorithm comparison](training.md) for a complete command-based walkthrough. No dataset, trained model, notebook or packet capture is included in the repository.

| Guide | What it covers |
|---|---|
| [Dataset placement and splits](dataset-and-splits.md) | Exact input location/schema, labels, 70/15/15 stratification, saved indices, fingerprints and conflict prevention |
| [Training, comparison and testing](training.md) | Commands for Random Forest, Logistic Regression and XGBoost; reading reports; tuning; PC and STM32 tests |
| [Configuration](configuration.md) | Every YAML section, supported parameter, defaults and preset behavior |
| [Command reference](commands.md) | Every command-line entry point, available flags, examples, dependencies and output behavior |
| [Architecture and dashboard API](architecture.md) | Data flow, module responsibilities, artifact contracts and API examples |
| [STM32 firmware](../firmware/README.md) | Generated files, HAL integration, preprocessing modes, UART protocol and alternative adapters |
| [Troubleshooting](troubleshooting.md) | Dataset, memory, class/split, run, dependency, dashboard, export and serial errors |
| [Evaluation limitations](limitations.md) | What reported metrics and host/device tests do and do not establish |

## Minimal path through the project

1. Clone the repository and install core dependencies in a Python 3.13 environment.
2. Put original labeled Intrusion CSVs directly in `data/raw/`, with one copy of each file.
3. Train sequentially with the standalone algorithm presets and unique run names.
4. Compare validation metrics; inspect the final selected model's test report.
5. Replay a named run on the dashboard, or export a Random Forest and test it on a flashed STM32.

Run all commands from the repository root. Datasets and generated outputs remain local under ignored `data/` and `artifacts/` directories. Always pair a model with its own preprocessor, class map and run-specific split records.
