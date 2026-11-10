#ifndef APP_NIDS_H
#define APP_NIDS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

#define NIDS_FEATURE_COUNT 15
#define NIDS_CLASS_COUNT 5
#define NIDS_UART_LINE_MAX 256

typedef struct {
    int row_id;
    float features[NIDS_FEATURE_COUNT];
} nids_request_t;

typedef struct {
    int row_id;
    int pred_id;
    float confidence;
} nids_result_t;

void NIDS_Init(UART_HandleTypeDef *uart_handle);
void NIDS_ProcessLoop(void);

int NIDS_ReadLine(char *out, int max_len);
int NIDS_ParseRequest(const char *line, nids_request_t *request);
int NIDS_RunInference(const float *features, int feature_count, int *pred_id, float *confidence);
void NIDS_SendResult(const nids_result_t *result);

#ifdef __cplusplus
}
#endif

#endif
