# STM32 RF Template With emlearn

This template replaces the previous `network.h` / X-CUBE-AI path.

It assumes:

- you keep preprocessing on the PC
- STM32 only receives the 15 already-preprocessed float inputs
- the RandomForest is exported with `emlearn`

## 1. Export the RF model to C

From the project root:

```bash
cd nids_pc_prototype
python -m src.main_export_emlearn --run_name rf_run_04 --model_name nids_model
```

This writes:

- `artifacts/embedded/rf_run_04/nids_model.h`
- `artifacts/embedded/rf_run_04/class_map.json`
- `artifacts/embedded/rf_run_04/preprocessor.json`

## 2. Copy files into STM32 project

Copy these files into your CubeMX project:

- `stm32_templates/app_nids.h` -> `Core/Inc/app_nids.h`
- `stm32_templates/app_nids.c` -> `Core/Src/app_nids.c`
- `artifacts/embedded/rf_run_04/nids_model.h` -> `Core/Inc/nids_model.h`

Also make sure the emlearn runtime headers are in the include path:

- `eml_trees.h`
- `eml_common.h`

If your emlearn export produces extra runtime headers, copy them too.

## 3. main.c integration

Use the same `main.c.snippet.txt` integration as before:

```c
#include "app_nids.h"
```

After UART init:

```c
NIDS_Init(&huart3);
```

Inside `while (1)`:

```c
NIDS_ProcessLoop();
```

## 4. UART protocol

PC -> STM32:

```text
row_id,f1,f2,f3,...,f15\n
```

STM32 -> PC:

```text
row_id,pred_id,confidence_scaled\n
```

`confidence_scaled` is currently fixed to `1000000` for all successful predictions.

## 5. Required adaptations

### A. Model header name

The template includes:

```c
#include "nids_model.h"
```

If your exported header has another name, change that include.

### B. Exported model symbol name

The template calls:

```c
eml_trees_predict(&nids_model, features, feature_count);
```

If you export with a different `--model_name`, update the symbol name too.

### C. Class count

The template assumes `5` classes:

- `0 = Botnet`
- `1 = BruteForce`
- `2 = DoS`
- `3 = Normal`
- `4 = WebAttack`

Do not hardcode the old target order blindly.
Use the real class order from:

- `artifacts/embedded/rf_run_04/class_map.json`

Update `NIDS_CLASS_COUNT` and your downstream interpretation if needed.

## 6. PC-side validation

After flashing the board, run:

```bash
cd nids_pc_prototype
python -m src.main_stm32_uart_test --run_name rf_run_04 --port COM6 --baudrate 115200 --limit 100 --dashboard
```

Replace `COM6` with the actual ST-LINK virtual COM port.
