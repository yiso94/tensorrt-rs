#include "tensorrt_runtime.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <new>
#include <stdexcept>
#include <type_traits>

#ifndef NV_TENSORRT_BUILD
#define NV_TENSORRT_BUILD 0
#endif

namespace {

class SafeLogger final : public nvinfer1::ILogger {
public:
    explicit SafeLogger(int32_t min_severity) noexcept
        : min_severity_(to_severity(min_severity))
    {
    }

    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= min_severity_) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg == nullptr ? "" : msg);
        }
    }

private:
    static Severity to_severity(int32_t value) noexcept
    {
        switch (value) {
        case 0:
            return Severity::kINTERNAL_ERROR;
        case 1:
            return Severity::kERROR;
        case 2:
            return Severity::kWARNING;
        case 3:
            return Severity::kINFO;
        case 4:
            return Severity::kVERBOSE;
        default:
            return Severity::kWARNING;
        }
    }

    Severity min_severity_;
};

template <typename Fn>
safe_trt_status_t guard(Fn&& fn) noexcept
{
    try {
        return fn();
    } catch (...) {
        return SAFE_TRT_STATUS_EXCEPTION;
    }
}

bool valid_name(char const* name) noexcept
{
    return name != nullptr && name[0] != '\0';
}

safe_trt_status_t copy_dims(nvinfer1::Dims dims, safe_trt_dims_t* out_dims) noexcept
{
    if (out_dims == nullptr) {
        return SAFE_TRT_STATUS_INVALID_ARGUMENT;
    }
    if (dims.nbDims < 0 || dims.nbDims > SAFE_TRT_MAX_DIMS) {
        return SAFE_TRT_STATUS_INVALID_ARGUMENT;
    }

    out_dims->nb_dims = dims.nbDims;
    std::fill(std::begin(out_dims->d), std::end(out_dims->d), int64_t{0});
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        out_dims->d[i] = static_cast<int64_t>(dims.d[i]);
    }
    return SAFE_TRT_STATUS_OK;
}

bool make_dims(safe_trt_dims_t input, nvinfer1::Dims* out_dims) noexcept
{
    if (out_dims == nullptr || input.nb_dims < 0 || input.nb_dims > SAFE_TRT_MAX_DIMS) {
        return false;
    }

    nvinfer1::Dims dims{};
    dims.nbDims = input.nb_dims;
    using TrtDim = std::remove_reference_t<decltype(dims.d[0])>;
    for (int32_t i = 0; i < input.nb_dims; ++i) {
        dims.d[i] = static_cast<TrtDim>(input.d[i]);
    }
    *out_dims = dims;
    return true;
}

bool is_tensor(const nvinfer1::ICudaEngine* engine, char const* name) noexcept
{
    return engine != nullptr && valid_name(name)
        && engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kNONE;
}

} // namespace

struct safe_trt_runtime_t {
    explicit safe_trt_runtime_t(int32_t min_severity) noexcept
        : logger(min_severity)
    {
    }

    SafeLogger logger;
    nvinfer1::IRuntime* runtime{nullptr};
};

struct safe_trt_engine_t {
    nvinfer1::ICudaEngine* engine{nullptr};
};

struct safe_trt_execution_context_t {
    nvinfer1::IExecutionContext* context{nullptr};
};

int32_t safe_trt_version_major(void)
{
    return NV_TENSORRT_MAJOR;
}

int32_t safe_trt_version_minor(void)
{
    return NV_TENSORRT_MINOR;
}

int32_t safe_trt_version_patch(void)
{
    return NV_TENSORRT_PATCH;
}

int32_t safe_trt_version_build(void)
{
    return NV_TENSORRT_BUILD;
}

safe_trt_status_t safe_trt_runtime_create(
    int32_t min_severity,
    safe_trt_runtime_t** out_runtime)
{
    return guard([&]() {
        if (out_runtime == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        *out_runtime = nullptr;

        auto* handle = new (std::nothrow) safe_trt_runtime_t(min_severity);
        if (handle == nullptr) {
            return SAFE_TRT_STATUS_ALLOCATION_FAILED;
        }

        handle->runtime = nvinfer1::createInferRuntime(handle->logger);
        if (handle->runtime == nullptr) {
            delete handle;
            return SAFE_TRT_STATUS_NULL_POINTER;
        }

        *out_runtime = handle;
        return SAFE_TRT_STATUS_OK;
    });
}

void safe_trt_runtime_destroy(safe_trt_runtime_t* runtime)
{
    if (runtime != nullptr) {
        delete runtime->runtime;
        delete runtime;
    }
}

safe_trt_status_t safe_trt_runtime_deserialize_engine(
    safe_trt_runtime_t* runtime,
    const void* data,
    size_t size,
    safe_trt_engine_t** out_engine)
{
    return guard([&]() {
        if (runtime == nullptr || runtime->runtime == nullptr || data == nullptr || size == 0
            || out_engine == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        *out_engine = nullptr;

        auto* trt_engine = runtime->runtime->deserializeCudaEngine(data, size);
        if (trt_engine == nullptr) {
            return SAFE_TRT_STATUS_TENSORRT_ERROR;
        }

        auto* handle = new (std::nothrow) safe_trt_engine_t{};
        if (handle == nullptr) {
            delete trt_engine;
            return SAFE_TRT_STATUS_ALLOCATION_FAILED;
        }

        handle->engine = trt_engine;
        *out_engine = handle;
        return SAFE_TRT_STATUS_OK;
    });
}

void safe_trt_engine_destroy(safe_trt_engine_t* engine)
{
    if (engine != nullptr) {
        delete engine->engine;
        delete engine;
    }
}

safe_trt_status_t safe_trt_engine_num_io_tensors(
    const safe_trt_engine_t* engine,
    int32_t* out_count)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || out_count == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        *out_count = engine->engine->getNbIOTensors();
        return SAFE_TRT_STATUS_OK;
    });
}

safe_trt_status_t safe_trt_engine_io_tensor_name(
    const safe_trt_engine_t* engine,
    int32_t index,
    const char** out_name)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || out_name == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        if (index < 0 || index >= engine->engine->getNbIOTensors()) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        *out_name = engine->engine->getIOTensorName(index);
        return *out_name == nullptr ? SAFE_TRT_STATUS_NULL_POINTER : SAFE_TRT_STATUS_OK;
    });
}

safe_trt_status_t safe_trt_engine_tensor_shape(
    const safe_trt_engine_t* engine,
    const char* name,
    safe_trt_dims_t* out_dims)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || !valid_name(name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        if (!is_tensor(engine->engine, name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        return copy_dims(engine->engine->getTensorShape(name), out_dims);
    });
}

safe_trt_status_t safe_trt_engine_tensor_data_type(
    const safe_trt_engine_t* engine,
    const char* name,
    int32_t* out_data_type)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || !valid_name(name)
            || out_data_type == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        if (!is_tensor(engine->engine, name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        *out_data_type = static_cast<int32_t>(engine->engine->getTensorDataType(name));
        return SAFE_TRT_STATUS_OK;
    });
}

safe_trt_status_t safe_trt_engine_tensor_io_mode(
    const safe_trt_engine_t* engine,
    const char* name,
    int32_t* out_io_mode)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || !valid_name(name)
            || out_io_mode == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        *out_io_mode = static_cast<int32_t>(engine->engine->getTensorIOMode(name));
        return SAFE_TRT_STATUS_OK;
    });
}

safe_trt_status_t safe_trt_engine_create_execution_context(
    const safe_trt_engine_t* engine,
    safe_trt_execution_context_t** out_context)
{
    return guard([&]() {
        if (engine == nullptr || engine->engine == nullptr || out_context == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }
        *out_context = nullptr;

        auto* trt_context = engine->engine->createExecutionContext();
        if (trt_context == nullptr) {
            return SAFE_TRT_STATUS_TENSORRT_ERROR;
        }

        auto* handle = new (std::nothrow) safe_trt_execution_context_t{};
        if (handle == nullptr) {
            delete trt_context;
            return SAFE_TRT_STATUS_ALLOCATION_FAILED;
        }

        handle->context = trt_context;
        *out_context = handle;
        return SAFE_TRT_STATUS_OK;
    });
}

void safe_trt_execution_context_destroy(safe_trt_execution_context_t* context)
{
    if (context != nullptr) {
        delete context->context;
        delete context;
    }
}

safe_trt_status_t safe_trt_context_set_input_shape(
    safe_trt_execution_context_t* context,
    const char* name,
    safe_trt_dims_t dims)
{
    return guard([&]() {
        if (context == nullptr || context->context == nullptr || !valid_name(name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        nvinfer1::Dims trt_dims{};
        if (!make_dims(dims, &trt_dims)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        return context->context->setInputShape(name, trt_dims) ? SAFE_TRT_STATUS_OK
                                                               : SAFE_TRT_STATUS_TENSORRT_ERROR;
    });
}

safe_trt_status_t safe_trt_context_tensor_shape(
    safe_trt_execution_context_t* context,
    const char* name,
    safe_trt_dims_t* out_dims)
{
    return guard([&]() {
        if (context == nullptr || context->context == nullptr || !valid_name(name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        return copy_dims(context->context->getTensorShape(name), out_dims);
    });
}

safe_trt_status_t safe_trt_context_set_tensor_address(
    safe_trt_execution_context_t* context,
    const char* name,
    void* data)
{
    return guard([&]() {
        if (context == nullptr || context->context == nullptr || !valid_name(name)) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        return context->context->setTensorAddress(name, data) ? SAFE_TRT_STATUS_OK
                                                              : SAFE_TRT_STATUS_TENSORRT_ERROR;
    });
}

safe_trt_status_t safe_trt_context_enqueue_v3(
    safe_trt_execution_context_t* context,
    void* stream)
{
    return guard([&]() {
        if (context == nullptr || context->context == nullptr) {
            return SAFE_TRT_STATUS_INVALID_ARGUMENT;
        }

        return context->context->enqueueV3(reinterpret_cast<cudaStream_t>(stream))
            ? SAFE_TRT_STATUS_OK
            : SAFE_TRT_STATUS_TENSORRT_ERROR;
    });
}
