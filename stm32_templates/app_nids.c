#include "app_nids.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eml_trees.h"
#include "nids_model.h"

static UART_HandleTypeDef *g_uart = NULL;

void NIDS_Init(UART_HandleTypeDef *uart_handle)
{
    g_uart = uart_handle;
}

void NIDS_ProcessLoop(void)
{
    char line[NIDS_UART_LINE_MAX];
    nids_request_t request;
    nids_result_t result;

    if (!NIDS_ReadLine(line, sizeof(line))) {
        return;
    }

    if (NIDS_ParseRequest(line, &request) != 0) {
        const char *msg = "ERR\n";
        HAL_UART_Transmit(g_uart, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
        return;
    }

    result.row_id = request.row_id;
    if (NIDS_RunInference(request.features, NIDS_FEATURE_COUNT, &result.pred_id, &result.confidence) != 0) {
        const char *msg = "INFER_ERR\n";
        HAL_UART_Transmit(g_uart, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
        return;
    }

    NIDS_SendResult(&result);
}

int NIDS_ReadLine(char *out, int max_len)
{
    uint8_t ch = 0;
    int pos = 0;

    if (g_uart == NULL || out == NULL || max_len <= 1) {
        return 0;
    }

    while (pos < max_len - 1) {
        if (HAL_UART_Receive(g_uart, &ch, 1, HAL_MAX_DELAY) != HAL_OK) {
            return 0;
        }

        if (ch == '\n') {
            break;
        }

        if (ch != '\r') {
            out[pos++] = (char)ch;
        }
    }

    out[pos] = '\0';
    return pos > 0;
}

int NIDS_ParseRequest(const char *line, nids_request_t *request)
{
    char buffer[NIDS_UART_LINE_MAX];
    char *token = NULL;
    int i = 0;

    if (line == NULL || request == NULL) {
        return -1;
    }

    strncpy(buffer, line, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';

    token = strtok(buffer, ",");
    if (token == NULL) {
        return -1;
    }

    request->row_id = atoi(token);

    for (i = 0; i < NIDS_FEATURE_COUNT; i++) {
        token = strtok(NULL, ",");
        if (token == NULL) {
            return -1;
        }
        request->features[i] = strtof(token, NULL);
    }

    return 0;
}

int NIDS_RunInference(const float *features, int feature_count, int *pred_id, float *confidence)
{
    int32_t cls = 0;

    if (features == NULL || pred_id == NULL || confidence == NULL) {
        return -1;
    }
    if (feature_count != NIDS_FEATURE_COUNT) {
        return -1;
    }

    /*
     * This template assumes:
     *   - you exported the RF with: python -m src.main_export_emlearn --run_name rf_run_04
     *   - you copied the generated header to Core/Inc/nids_model.h
     *   - the exported C symbol name is the default "nids_model"
     *
     * Inputs are the already-preprocessed 15 float features sent from the PC.
     */
    cls = eml_trees_predict(&nids_model, features, feature_count);

    if (cls < 0 || cls >= NIDS_CLASS_COUNT) {
        return -1;
    }

    *pred_id = (int)cls;

    /*
     * For first validation, use a fixed confidence because plain emlearn RF
     * deployment focuses on class prediction. If you later add probability
     * support from the generated header, replace this with the real value.
     */
    *confidence = 1.0f;
    return 0;
}

void NIDS_SendResult(const nids_result_t *result)
{
    char tx[64];
    int len = 0;
    int conf_scaled = 0;

    if (g_uart == NULL || result == NULL) {
        return;
    }

    if (result->confidence <= 0.0f) {
        conf_scaled = 0;
    } else if (result->confidence >= 1.0f) {
        conf_scaled = 1000000;
    } else {
        conf_scaled = (int)(result->confidence * 1000000.0f);
    }

    len = snprintf(tx, sizeof(tx), "%d,%d,%d\n", result->row_id, result->pred_id, conf_scaled);
    HAL_UART_Transmit(g_uart, (uint8_t *)tx, len, HAL_MAX_DELAY);
}
