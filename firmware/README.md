# STM32 integration

[Documentation index](../docs/README.md) · [Train a model first](../docs/training.md) · [Command reference](../docs/commands.md)

The default adapter is `app_nids.c` plus `app_nids.h`. It uses the custom plain-C Random Forest export and applies the selected run's preprocessing on the device.

1. Run `python -m src.main_export_rf_c --run_name <run>` from the repository root.
2. Copy `rf_model.h` and `rf_preprocessor.h` from `artifacts/embedded_c/<run>/` into your STM32 project's `Core/Inc/`, and `rf_model.c` into `Core/Src/`.
3. Copy `firmware/app_nids.h` and `firmware/app_nids.c` to those corresponding directories. Compile only one adapter implementation.
4. Include `app_nids.h` from your `main.c`; after UART initialization call `NIDS_Init(&huartX)` using the actual configured handle. Call `NIDS_ProcessLoop()` inside the main loop. See `main.c.snippet.txt`.
5. Enable the C99 language features and math library required by the adapter. Build and flash with your board's toolchain.
6. Match the host mode: by default send raw features with `--device_preprocess`. To preprocess on the PC, build the adapter with `NIDS_DEVICE_PREPROCESS=0` and omit that host flag.

Never mix generated files from different runs. `NIDS_FEATURE_COUNT` and `NIDS_CLASS_COUNT` in `app_nids.h` must agree with the generated model. Interpret class IDs through `class_map.json`.

## UART protocol

```text
PC -> device: row_id,f0,f1,...,f14\n
device -> PC: row_id,pred_id,confidence_scaled\n
```

Features are ASCII float values in canonical order. Confidence is scaled by 1,000,000. The adapter returns the selected class probability. The PC also accepts two-field replies or JSON objects with `row_id`, `pred_id` and `confidence`.

`\n` denotes an actual newline byte, not the two literal characters backslash and n. Model inputs contain 15 features after the row ID. Device class IDs must be interpreted through that run's `class_map.json`.

## Host commands after flashing

Install `requirements-hardware.txt`, close other serial monitors and replace `COM5` with the actual board port. For the default raw-input adapter:

```powershell
python -m src.main_rfc_uart_eval --config config/random_forest.yaml --train_run_name rf_baseline --port COM5 --device_preprocess --limit 500 --report_run_name board_rf_01
```

It evaluates up to 500 rows per validation/test split. For a dashboard demonstration instead, first start `python -m src.main_serve`, then:

```powershell
python -m src.main_stm32_uart_test --config config/random_forest.yaml --run_name rf_baseline --port COM5 --device_preprocess --limit 100 --dashboard
```

The dashboard replay prints accuracy; the evaluator writes full reports with accepted/skipped row counts. For `NIDS_DEVICE_PREPROCESS=0` firmware, omit `--device_preprocess` in both commands. Never send host-scaled inputs to firmware that will scale them again.

The adapter uses blocking HAL UART I/O and a 256-byte line buffer. Check that this suits your application and maximum payload. This is an integration example, not a complete capture-capable STM32 NIDS. Network capture and flow construction currently run on the PC.

## Optional emlearn adapter

`app_nids_emlearn.c` preserves the alternative emlearn integration. Compile it instead of `app_nids.c`, copy the generated `nids_model.h` and required emlearn runtime headers, and send PC-preprocessed features. Its confidence is a fixed placeholder of 1.0 and must not be interpreted as a calibrated score. Verify the generated runtime's input type and numerical behavior before deployment.

m2cgen exports provide a scorer and wrapper, but this repository does not supply a separate complete m2cgen HAL adapter. Integrate that scorer explicitly before using the m2cgen UART evaluator.
