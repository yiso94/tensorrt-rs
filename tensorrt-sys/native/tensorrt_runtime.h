#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAFE_TRT_MAX_DIMS 8

typedef struct safe_trt_runtime_t safe_trt_runtime_t;
typedef struct safe_trt_engine_t safe_trt_engine_t;
typedef struct safe_trt_execution_context_t safe_trt_execution_context_t;

typedef enum safe_trt_status_t {
    SAFE_TRT_STATUS_OK = 0,
    SAFE_TRT_STATUS_INVALID_ARGUMENT = 1,
    SAFE_TRT_STATUS_ALLOCATION_FAILED = 2,
    SAFE_TRT_STATUS_NULL_POINTER = 3,
    SAFE_TRT_STATUS_TENSORRT_ERROR = 4,
    SAFE_TRT_STATUS_EXCEPTION = 5,
} safe_trt_status_t;

typedef struct safe_trt_dims_t {
    int32_t nb_dims;
    int64_t d[SAFE_TRT_MAX_DIMS];
} safe_trt_dims_t;

int32_t safe_trt_version_major(void);
int32_t safe_trt_version_minor(void);
int32_t safe_trt_version_patch(void);
int32_t safe_trt_version_build(void);

safe_trt_status_t safe_trt_runtime_create(
    int32_t min_severity,
    safe_trt_runtime_t** out_runtime);
void safe_trt_runtime_destroy(safe_trt_runtime_t* runtime);

safe_trt_status_t safe_trt_runtime_deserialize_engine(
    safe_trt_runtime_t* runtime,
    const void* data,
    size_t size,
    safe_trt_engine_t** out_engine);
void safe_trt_engine_destroy(safe_trt_engine_t* engine);

safe_trt_status_t safe_trt_engine_num_io_tensors(
    const safe_trt_engine_t* engine,
    int32_t* out_count);
safe_trt_status_t safe_trt_engine_io_tensor_name(
    const safe_trt_engine_t* engine,
    int32_t index,
    const char** out_name);
safe_trt_status_t safe_trt_engine_tensor_shape(
    const safe_trt_engine_t* engine,
    const char* name,
    safe_trt_dims_t* out_dims);
safe_trt_status_t safe_trt_engine_tensor_data_type(
    const safe_trt_engine_t* engine,
    const char* name,
    int32_t* out_data_type);
safe_trt_status_t safe_trt_engine_tensor_io_mode(
    const safe_trt_engine_t* engine,
    const char* name,
    int32_t* out_io_mode);
safe_trt_status_t safe_trt_engine_create_execution_context(
    const safe_trt_engine_t* engine,
    safe_trt_execution_context_t** out_context);

void safe_trt_execution_context_destroy(safe_trt_execution_context_t* context);
safe_trt_status_t safe_trt_context_set_input_shape(
    safe_trt_execution_context_t* context,
    const char* name,
    safe_trt_dims_t dims);
safe_trt_status_t safe_trt_context_tensor_shape(
    safe_trt_execution_context_t* context,
    const char* name,
    safe_trt_dims_t* out_dims);
safe_trt_status_t safe_trt_context_set_tensor_address(
    safe_trt_execution_context_t* context,
    const char* name,
    void* data);
safe_trt_status_t safe_trt_context_enqueue_v3(
    safe_trt_execution_context_t* context,
    void* stream);

#ifdef __cplusplus
}
#endif
