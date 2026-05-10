#include "tensorrt_llm_executor.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/runtimeDefaults.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtllm = tensorrt_llm::executor;

struct TensorRtLlmRsExecutor
{
    std::unique_ptr<trtllm::Executor> executor;
};

namespace
{

constexpr int32_t TENSOR_MEMORY_CPU = 0;
constexpr int32_t TENSOR_MEMORY_GPU = 1;

std::vector<std::string> string_list(TensorRtLlmRsStringList const& input, char const* name);
std::optional<std::vector<trtllm::AdditionalModelOutput>> additional_model_outputs(
    TensorRtLlmRsStringList const& input, char const* name);

char* copy_error(std::string const& message)
{
    auto* data = static_cast<char*>(std::malloc(message.size() + 1));
    if (data == nullptr)
    {
        return nullptr;
    }
    std::memcpy(data, message.c_str(), message.size() + 1);
    return data;
}

void set_error(char** out_error, std::string const& message)
{
    if (out_error != nullptr)
    {
        *out_error = copy_error(message);
    }
}

template <typename F>
int run_catching(char** out_error, F&& f)
{
    try
    {
        f();
        return 0;
    }
    catch (std::exception const& error)
    {
        set_error(out_error, error.what());
        return 1;
    }
    catch (...)
    {
        set_error(out_error, "unknown TensorRT-LLM executor error");
        return 1;
    }
}

void check_cuda_result(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

class CudaDeviceGuard
{
public:
    explicit CudaDeviceGuard(int device)
    {
        check_cuda_result(cudaGetDevice(&mPreviousDevice), "cudaGetDevice");
        if (mPreviousDevice != device)
        {
            check_cuda_result(cudaSetDevice(device), "cudaSetDevice");
            mRestore = true;
        }
    }

    ~CudaDeviceGuard()
    {
        if (mRestore)
        {
            (void) cudaSetDevice(mPreviousDevice);
        }
    }

    CudaDeviceGuard(CudaDeviceGuard const&) = delete;
    CudaDeviceGuard& operator=(CudaDeviceGuard const&) = delete;

private:
    int mPreviousDevice{0};
    bool mRestore{false};
};

struct OutputCopyStream
{
    cudaStream_t stream{};
    int device{-1};
    cudaEvent_t completionEvent{};
};

struct TensorRtLlmRsDeferredCopies
{
    std::shared_ptr<std::vector<trtllm::Response>> responses;
    cudaEvent_t completionEvent{};
};

trtllm::DataType data_type_from_i32(int32_t data_type)
{
    switch (data_type)
    {
    case 0:
        return trtllm::DataType::kBOOL;
    case 1:
        return trtllm::DataType::kUINT8;
    case 2:
        return trtllm::DataType::kINT8;
    case 3:
        return trtllm::DataType::kINT32;
    case 4:
        return trtllm::DataType::kINT64;
    case 5:
        return trtllm::DataType::kBF16;
    case 6:
        return trtllm::DataType::kFP8;
    case 7:
        return trtllm::DataType::kFP16;
    case 8:
        return trtllm::DataType::kFP32;
    default:
        return trtllm::DataType::kUNKNOWN;
    }
}

int32_t data_type_to_i32(trtllm::DataType data_type)
{
    switch (data_type)
    {
    case trtllm::DataType::kBOOL:
        return 0;
    case trtllm::DataType::kUINT8:
        return 1;
    case trtllm::DataType::kINT8:
        return 2;
    case trtllm::DataType::kINT32:
        return 3;
    case trtllm::DataType::kINT64:
        return 4;
    case trtllm::DataType::kBF16:
        return 5;
    case trtllm::DataType::kFP8:
        return 6;
    case trtllm::DataType::kFP16:
        return 7;
    case trtllm::DataType::kFP32:
        return 8;
    case trtllm::DataType::kUNKNOWN:
        return 9;
    }
    return 9;
}

std::optional<trtllm::Tensor> tensor_from_data(void const* data, size_t byte_len, size_t element_count,
    int32_t data_type, int64_t const* shape, size_t rank, int32_t memory_type, char const* name)
{
    if (data == nullptr || element_count == 0)
    {
        return std::nullopt;
    }
    if (byte_len == 0)
    {
        throw std::invalid_argument(std::string(name) + " byte length cannot be zero");
    }
    auto dtype = data_type_from_i32(data_type);
    if (dtype == trtllm::DataType::kUNKNOWN)
    {
        throw std::runtime_error(std::string("unsupported ") + name + " TensorRT-LLM executor dtype");
    }
    auto trt_shape = trtllm::Shape(shape, rank);
    if (memory_type == TENSOR_MEMORY_GPU)
    {
        auto tensor = trtllm::Tensor::of(dtype, const_cast<void*>(data), trt_shape);
        if (tensor.getSizeInBytes() != byte_len)
        {
            throw std::invalid_argument(std::string(name) + " byte length does not match tensor shape and dtype");
        }
        if (tensor.getMemoryType() != trtllm::MemoryType::kGPU)
        {
            throw std::runtime_error(std::string(name) + " was not wrapped as a TensorRT-LLM GPU tensor");
        }
        return tensor;
    }
    if (memory_type != TENSOR_MEMORY_CPU)
    {
        throw std::invalid_argument(std::string(name) + " has unsupported TensorRT-LLM executor memory type");
    }

    auto tensor = trtllm::Tensor::cpu(dtype, trt_shape);
    if (tensor.getSizeInBytes() != byte_len)
    {
        throw std::invalid_argument(std::string(name) + " byte length does not match tensor shape and dtype");
    }
    std::memcpy(tensor.getData(), data, byte_len);
    return tensor;
}

std::optional<trtllm::Tensor> tensor_from_view(TensorRtLlmRsTensorView const& view, char const* name)
{
    if (view.data == nullptr || view.byte_len == 0)
    {
        return std::nullopt;
    }
    if (view.rank > 0 && view.shape == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " shape pointer is null");
    }
    auto dtype = data_type_from_i32(view.data_type);
    if (dtype == trtllm::DataType::kUNKNOWN)
    {
        throw std::runtime_error(std::string("unsupported ") + name + " TensorRT-LLM executor dtype");
    }
    return tensor_from_data(
        view.data, view.byte_len, view.byte_len, view.data_type, view.shape, view.rank, view.memory_type, name);
}

std::optional<trtllm::Tensor> prompt_embedding_tensor(TensorRtLlmRsRequest const& request)
{
    int64_t shape[] = {static_cast<int64_t>(request.prompt_embedding_rows),
        static_cast<int64_t>(request.prompt_embedding_hidden_size)};
    return tensor_from_data(request.prompt_embedding_data, request.prompt_embedding_byte_len,
        request.prompt_embedding_rows, request.prompt_embedding_data_type, shape, sizeof(shape) / sizeof(shape[0]),
        request.prompt_embedding_memory_type, "prompt embedding");
}

std::optional<trtllm::Tensor> embedding_bias_tensor(TensorRtLlmRsRequest const& request)
{
    int64_t shape[] = {static_cast<int64_t>(request.embedding_bias_len)};
    return tensor_from_data(request.embedding_bias_data, request.embedding_bias_byte_len, request.embedding_bias_len,
        request.embedding_bias_data_type, shape, sizeof(shape) / sizeof(shape[0]), request.embedding_bias_memory_type,
        "embedding bias");
}

trtllm::VecTokens vec_tokens(int32_t const* tokens, size_t count, char const* name)
{
    if (tokens == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    return trtllm::VecTokens(tokens, tokens + count);
}

std::optional<trtllm::VecTokens> optional_tokens(int32_t const* tokens, size_t count, char const* name)
{
    if (count == 0)
    {
        return std::nullopt;
    }
    return vec_tokens(tokens, count, name);
}

std::optional<std::vector<std::optional<std::string>>> optional_nullable_strings(
    char const* const* values, size_t count, char const* name)
{
    if (count == 0)
    {
        return std::nullopt;
    }
    if (values == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    std::vector<std::optional<std::string>> out;
    out.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        if (values[index] == nullptr)
        {
            out.push_back(std::nullopt);
        }
        else
        {
            out.emplace_back(values[index]);
        }
    }
    return out;
}

std::vector<std::vector<int32_t>> nested_tokens(TensorRtLlmRsWordList const& list, char const* name)
{
    if (list.offset_count == 0)
    {
        return {};
    }
    if (list.tokens == nullptr || list.offsets == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    if (list.offset_count < 2 || list.offsets[0] != 0 || list.offsets[list.offset_count - 1] != list.token_count)
    {
        throw std::invalid_argument(std::string(name) + " offsets are invalid");
    }
    std::vector<std::vector<int32_t>> values;
    values.reserve(list.offset_count - 1);
    for (size_t i = 0; i + 1 < list.offset_count; ++i)
    {
        auto begin = list.offsets[i];
        auto end = list.offsets[i + 1];
        if (begin >= end || end > list.token_count)
        {
            throw std::invalid_argument(std::string(name) + " entry is invalid");
        }
        values.push_back(vec_tokens(list.tokens + begin, end - begin, name));
    }
    return values;
}

std::optional<trtllm::ExternalDraftTokensConfig> external_draft_tokens_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_external_draft_tokens_config)
    {
        return std::nullopt;
    }
    auto const& config = request.external_draft_tokens_config;
    if (config.tokens == nullptr || config.token_count == 0)
    {
        throw std::invalid_argument("external_draft_tokens must contain at least one token");
    }
    return trtllm::ExternalDraftTokensConfig(vec_tokens(config.tokens, config.token_count, "external draft tokens"),
        tensor_from_view(config.logits, "external draft logits"),
        config.has_acceptance_threshold ? std::make_optional(config.acceptance_threshold) : std::nullopt,
        config.has_fast_logits ? std::make_optional(config.fast_logits != 0) : std::nullopt);
}

std::optional<trtllm::MultimodalInput> multimodal_input(TensorRtLlmRsRequest const& request)
{
    if (!request.has_multimodal_input)
    {
        return std::nullopt;
    }
    auto const& input = request.multimodal_input;
    if (input.positions == nullptr || input.lengths == nullptr)
    {
        throw std::invalid_argument("multimodal input positions or lengths pointer is null");
    }
    auto hashes = nested_tokens(input.hashes, "multimodal hashes");
    if (hashes.size() != input.position_count || hashes.size() != input.length_count)
    {
        throw std::invalid_argument("multimodal input hashes, positions, and lengths must have the same length");
    }
    auto uuids = optional_nullable_strings(input.uuids, input.uuid_count, "multimodal uuids");
    if (uuids && uuids->size() != hashes.size())
    {
        throw std::invalid_argument("multimodal input uuids length must match multimodal hashes length");
    }
    return trtllm::MultimodalInput(
        std::move(hashes), std::vector<int32_t>(input.positions, input.positions + input.position_count),
        std::vector<int32_t>(input.lengths, input.lengths + input.length_count), std::move(uuids));
}

std::optional<trtllm::MropeConfig> mrope_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_mrope_config)
    {
        return std::nullopt;
    }
    auto tensor = tensor_from_view(request.mrope_config.rotary_cos_sin, "mrope rotary cos sin");
    if (!tensor)
    {
        throw std::invalid_argument("mrope_config requires rotary cos/sin tensor");
    }
    return trtllm::MropeConfig(std::move(*tensor), request.mrope_config.position_deltas);
}

std::optional<trtllm::LoraConfig> lora_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_lora_config)
    {
        return std::nullopt;
    }
    return trtllm::LoraConfig(request.lora_config.task_id,
        tensor_from_view(request.lora_config.weights, "lora weights"),
        tensor_from_view(request.lora_config.config, "lora config"));
}

std::optional<trtllm::LookaheadDecodingConfig> lookahead_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_lookahead_config)
    {
        return std::nullopt;
    }
    return trtllm::LookaheadDecodingConfig(request.lookahead_config.window_size, request.lookahead_config.ngram_size,
        request.lookahead_config.verification_set_size);
}

std::optional<trtllm::KvCacheRetentionConfig> kv_cache_retention_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_kv_cache_retention_config)
    {
        return std::nullopt;
    }
    auto const& config = request.kv_cache_retention_config;
    if (config.range_count > 0 && config.ranges == nullptr)
    {
        throw std::invalid_argument("kv cache retention ranges pointer is null");
    }
    std::vector<trtllm::KvCacheRetentionConfig::TokenRangeRetentionConfig> ranges;
    ranges.reserve(config.range_count);
    for (size_t i = 0; i < config.range_count; ++i)
    {
        auto const& range = config.ranges[i];
        ranges.emplace_back(range.token_start, range.has_token_end ? std::make_optional(range.token_end) : std::nullopt,
            range.priority, range.has_duration_ms ? std::make_optional(std::chrono::milliseconds(range.duration_ms))
                                                   : std::nullopt);
    }
    auto transfer_mode = static_cast<tensorrt_llm::executor::KvCacheTransferMode>(config.transfer_mode);
    return trtllm::KvCacheRetentionConfig(std::move(ranges), config.decode_retention_priority,
        config.has_decode_duration_ms ? std::make_optional(std::chrono::milliseconds(config.decode_duration_ms))
                                      : std::nullopt,
        transfer_mode, config.directory == nullptr ? "" : config.directory);
}

std::optional<std::string> optional_string(char const* value)
{
    if (value == nullptr)
    {
        return std::nullopt;
    }
    return std::string(value);
}

std::optional<trtllm::ContextPhaseParams> context_phase_params(TensorRtLlmRsRequest const& request)
{
    if (!request.has_context_phase_params)
    {
        return std::nullopt;
    }
    auto const& params = request.context_phase_params;
    if (params.first_gen_tokens == nullptr || params.first_gen_token_count == 0)
    {
        throw std::invalid_argument("context_phase_params requires first generation tokens");
    }
    auto first_gen_tokens = vec_tokens(
        params.first_gen_tokens, params.first_gen_token_count, "context phase first generation tokens");
    std::optional<trtllm::VecTokens> draft_tokens
        = optional_tokens(params.draft_tokens, params.draft_token_count, "context phase draft tokens");
    std::optional<int32_t> ctx_dp_rank
        = params.has_ctx_dp_rank ? std::make_optional(params.ctx_dp_rank) : std::nullopt;
    auto endpoint = optional_string(params.disagg_info_endpoint);
    if (params.serialized_state_len > 0)
    {
        if (params.serialized_state == nullptr)
        {
            throw std::invalid_argument("context_phase_params serialized state pointer is null");
        }
        std::vector<char> serialized_state(params.serialized_state, params.serialized_state + params.serialized_state_len);
        return trtllm::ContextPhaseParams(
            std::move(first_gen_tokens), params.request_id, serialized_state, std::move(draft_tokens), std::move(ctx_dp_rank),
            std::move(endpoint));
    }
    return trtllm::ContextPhaseParams(
        std::move(first_gen_tokens), params.request_id, std::move(draft_tokens), std::move(ctx_dp_rank), std::move(endpoint));
}

std::optional<trtllm::EagleConfig> eagle_config(TensorRtLlmRsRequest const& request)
{
    if (!request.has_eagle_config)
    {
        return std::nullopt;
    }
    auto const& config = request.eagle_config;
    auto choices = nested_tokens(config.choices, "eagle choices");
    std::optional<trtllm::EagleChoices> eagle_choices = choices.empty() ? std::nullopt : std::make_optional(std::move(choices));
    return trtllm::EagleConfig(std::move(eagle_choices), config.greedy_sampling != 0,
        config.has_posterior_threshold ? std::make_optional(config.posterior_threshold) : std::nullopt,
        config.use_dynamic_tree != 0,
        config.has_dynamic_tree_max_top_k ? std::make_optional(config.dynamic_tree_max_top_k) : std::nullopt);
}

std::optional<trtllm::GuidedDecodingParams> guided_decoding_params(TensorRtLlmRsRequest const& request)
{
    if (!request.has_guided_decoding_params)
    {
        return std::nullopt;
    }
    auto const& params = request.guided_decoding_params;
    auto guide_type = static_cast<trtllm::GuidedDecodingParams::GuideType>(params.guide_type);
    return trtllm::GuidedDecodingParams(guide_type, optional_string(params.guide));
}

std::optional<std::vector<int32_t>> optional_int_list(TensorRtLlmRsIntList const& list, char const* name)
{
    if (list.count == 0)
    {
        return std::nullopt;
    }
    if (list.values == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " pointer is null");
    }
    return std::vector<int32_t>(list.values, list.values + list.count);
}

trtllm::SchedulerConfig scheduler_config(TensorRtLlmRsSchedulerConfig const& config)
{
    std::optional<trtllm::DynamicBatchConfig> dynamic_batch = std::nullopt;
    if (config.has_dynamic_batch_config)
    {
        std::vector<std::pair<int32_t, int32_t>> batch_size_table;
        if (config.dynamic_batch_config.batch_size_table_len > 0)
        {
            if (config.dynamic_batch_config.batch_size_table == nullptr)
            {
                throw std::invalid_argument("dynamic batch batch_size_table pointer is null");
            }
            batch_size_table.reserve(config.dynamic_batch_config.batch_size_table_len);
            for (size_t index = 0; index < config.dynamic_batch_config.batch_size_table_len; ++index)
            {
                auto const& entry = config.dynamic_batch_config.batch_size_table[index];
                batch_size_table.emplace_back(entry.first, entry.second);
            }
        }
        dynamic_batch = trtllm::DynamicBatchConfig(config.dynamic_batch_config.enable_batch_size_tuning != 0,
            config.dynamic_batch_config.enable_max_num_tokens_tuning != 0,
            config.dynamic_batch_config.moving_average_window,
            batch_size_table.empty() ? trtllm::DynamicBatchConfig::kDefaultBatchSizeTable : std::move(batch_size_table));
    }
    return trtllm::SchedulerConfig(static_cast<trtllm::CapacitySchedulerPolicy>(config.capacity_scheduler_policy),
        config.has_context_chunking_policy
            ? std::make_optional(static_cast<trtllm::ContextChunkingPolicy>(config.context_chunking_policy))
            : std::nullopt,
        std::move(dynamic_batch));
}

trtllm::KvCacheConfig kv_cache_config(TensorRtLlmRsKvCacheConfig const& config)
{
    trtllm::KvCacheConfig kv;
    kv.setEnableBlockReuse(config.enable_block_reuse != 0);
    kv.setEnablePartialReuse(config.enable_partial_reuse != 0);
    kv.setCopyOnPartialReuse(config.copy_on_partial_reuse != 0);
    if (config.has_max_tokens)
    {
        kv.setMaxTokens(config.max_tokens);
    }
    if (auto max_attention_window = optional_int_list(config.max_attention_window, "max_attention_window"))
    {
        kv.setMaxAttentionWindowVec(std::move(*max_attention_window));
    }
    if (config.has_sink_token_length)
    {
        kv.setSinkTokenLength(config.sink_token_length);
    }
    if (config.has_free_gpu_memory_fraction)
    {
        kv.setFreeGpuMemoryFraction(config.free_gpu_memory_fraction);
    }
    if (config.has_cross_kv_cache_fraction)
    {
        kv.setCrossKvCacheFraction(config.cross_kv_cache_fraction);
    }
    if (config.has_host_cache_size)
    {
        kv.setHostCacheSize(config.host_cache_size);
    }
    if (config.has_secondary_offload_min_priority)
    {
        kv.setSecondaryOffloadMinPriority(config.secondary_offload_min_priority);
    }
    kv.setEventBufferMaxSize(config.event_buffer_max_size);
    kv.setUseUvm(config.use_uvm != 0);
    kv.setAttentionDpEventsGatherPeriodMs(config.attention_dp_events_gather_period_ms);
    if (config.has_runtime_defaults)
    {
        auto defaults = tensorrt_llm::runtime::RuntimeDefaults(
            optional_int_list(
                config.runtime_defaults_max_attention_window, "runtime_defaults max_attention_window"),
            config.has_runtime_defaults_sink_token_length
                ? std::make_optional(config.runtime_defaults_sink_token_length)
                : std::nullopt);
        kv.fillEmptyFieldsFromRuntimeDefaults(defaults);
    }
    kv.setMaxGpuTotalBytes(config.max_gpu_total_bytes);
    return kv;
}

std::optional<trtllm::ParallelConfig> parallel_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_parallel_config)
    {
        return std::nullopt;
    }
    auto const& parallel = config.parallel_config;
    std::optional<trtllm::OrchestratorConfig> orchestrator = std::nullopt;
    if (parallel.has_orchestrator_config)
    {
        auto const& value = parallel.orchestrator_config;
        orchestrator = trtllm::OrchestratorConfig(value.is_orchestrator != 0,
            value.worker_executable_path == nullptr ? "" : value.worker_executable_path, nullptr,
            value.spawn_processes != 0);
    }
    return trtllm::ParallelConfig(static_cast<trtllm::CommunicationType>(parallel.communication_type),
        static_cast<trtllm::CommunicationMode>(parallel.communication_mode),
        optional_int_list(parallel.device_ids, "parallel device_ids"),
        optional_int_list(parallel.participant_ids, "parallel participant_ids"), std::move(orchestrator),
        parallel.has_num_nodes ? std::make_optional(parallel.num_nodes) : std::nullopt);
}

std::optional<trtllm::PeftCacheConfig> peft_cache_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_peft_cache_config)
    {
        return std::nullopt;
    }
    auto const& peft = config.peft_cache_config;
    return trtllm::PeftCacheConfig(peft.num_host_module_layer, peft.num_device_module_layer, peft.optimal_adapter_size,
        peft.max_adapter_size, peft.num_put_workers, peft.num_ensure_workers, peft.num_copy_streams,
        peft.max_pages_per_block_host, peft.max_pages_per_block_device,
        peft.has_device_cache_percent ? std::make_optional(peft.device_cache_percent) : std::nullopt,
        peft.has_host_cache_size ? std::make_optional(peft.host_cache_size) : std::nullopt,
        optional_string(peft.lora_prefetch_dir));
}

trtllm::DecodingMode decoding_mode(int32_t mode)
{
    switch (mode)
    {
    case 0: return trtllm::DecodingMode::Auto();
    case 1: return trtllm::DecodingMode::TopK();
    case 2: return trtllm::DecodingMode::TopP();
    case 3: return trtllm::DecodingMode::TopKTopP();
    case 4: return trtllm::DecodingMode::BeamSearch();
    case 5: return trtllm::DecodingMode::Medusa();
    case 6: return trtllm::DecodingMode::Lookahead();
    case 7: return trtllm::DecodingMode::ExplicitDraftTokens();
    case 8: return trtllm::DecodingMode::ExternalDraftTokens();
    case 9: return trtllm::DecodingMode::Eagle();
    default: throw std::invalid_argument("unknown decoding mode");
    }
}

trtllm::LookaheadDecodingConfig lookahead_config(TensorRtLlmRsLookaheadConfig const& config)
{
    return trtllm::LookaheadDecodingConfig(config.window_size, config.ngram_size, config.verification_set_size);
}

trtllm::EagleConfig eagle_config_value(TensorRtLlmRsEagleConfig const& config)
{
    auto choices = nested_tokens(config.choices, "eagle choices");
    std::optional<trtllm::EagleChoices> eagle_choices = choices.empty() ? std::nullopt : std::make_optional(std::move(choices));
    return trtllm::EagleConfig(std::move(eagle_choices), config.greedy_sampling != 0,
        config.has_posterior_threshold ? std::make_optional(config.posterior_threshold) : std::nullopt,
        config.use_dynamic_tree != 0,
        config.has_dynamic_tree_max_top_k ? std::make_optional(config.dynamic_tree_max_top_k) : std::nullopt);
}

std::optional<trtllm::DecodingConfig> decoding_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_decoding_config)
    {
        return std::nullopt;
    }
    trtllm::DecodingConfig decoding;
    auto const& value = config.decoding_config;
    if (value.has_decoding_mode)
    {
        decoding.setDecodingMode(decoding_mode(value.decoding_mode));
    }
    if (value.has_lookahead_config)
    {
        decoding.setLookaheadDecodingConfig(lookahead_config(value.lookahead_config));
    }
    auto medusa_choices = nested_tokens(value.medusa_choices, "medusa choices");
    if (!medusa_choices.empty())
    {
        decoding.setMedusaChoices(std::move(medusa_choices));
    }
    if (value.has_eagle_config)
    {
        decoding.setEagleConfig(eagle_config_value(value.eagle_config));
    }
    return decoding;
}

std::optional<trtllm::DebugConfig> debug_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_debug_config)
    {
        return std::nullopt;
    }
    auto const& debug = config.debug_config;
    return trtllm::DebugConfig(debug.debug_input_tensors != 0, debug.debug_output_tensors != 0,
        string_list(debug.debug_tensor_names, "debug tensor names"), debug.debug_tensors_max_iterations);
}

std::optional<trtllm::GuidedDecodingConfig> guided_decoding_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_guided_decoding_config)
    {
        return std::nullopt;
    }
    auto const& guided = config.guided_decoding_config;
    return trtllm::GuidedDecodingConfig(static_cast<trtllm::GuidedDecodingConfig::GuidedDecodingBackend>(guided.backend),
        string_list(guided.encoded_vocab, "guided decoding encoded vocab"), optional_string(guided.tokenizer_str),
        optional_int_list(guided.stop_token_ids, "guided decoding stop_token_ids"));
}

std::optional<trtllm::CacheTransceiverConfig> cache_transceiver_config(TensorRtLlmRsExecutorConfig const& config)
{
    if (!config.has_cache_transceiver_config)
    {
        return std::nullopt;
    }
    auto const& cache = config.cache_transceiver_config;
    return trtllm::CacheTransceiverConfig(
        cache.has_backend_type
            ? std::make_optional(static_cast<trtllm::CacheTransceiverConfig::BackendType>(cache.backend_type))
            : std::nullopt,
        cache.has_max_tokens_in_buffer ? std::make_optional(cache.max_tokens_in_buffer) : std::nullopt,
        cache.has_kv_transfer_timeout_ms ? std::make_optional(cache.kv_transfer_timeout_ms) : std::nullopt,
        cache.has_kv_transfer_sender_future_timeout_ms
            ? std::make_optional(cache.kv_transfer_sender_future_timeout_ms)
            : std::nullopt);
}

trtllm::ExecutorConfig executor_config(TensorRtLlmRsExecutorConfig const& config)
{
    auto additional_outputs
        = additional_model_outputs(config.additional_outputs, "executor additional_outputs");

    trtllm::ExtendedRuntimePerfKnobConfig perf_knobs(config.extended_runtime_perf_knob_config.multi_block_mode != 0,
        config.extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc != 0,
        config.extended_runtime_perf_knob_config.cuda_graph_mode != 0,
        config.extended_runtime_perf_knob_config.cuda_graph_cache_size);

    trtllm::ExecutorConfig out(config.max_beam_width, scheduler_config(config.scheduler_config),
        kv_cache_config(config.kv_cache_config), config.enable_chunked_context != 0, config.normalize_log_probs != 0,
        0, 0, static_cast<trtllm::BatchingType>(config.batching_type),
        config.has_max_batch_size ? std::make_optional(config.max_batch_size) : std::nullopt,
        config.has_max_num_tokens ? std::make_optional(config.max_num_tokens) : std::nullopt, parallel_config(config),
        peft_cache_config(config), std::nullopt, decoding_config(config), config.use_gpu_direct_storage != 0,
        config.gpu_weights_percent,
        config.has_max_queue_size ? std::make_optional(config.max_queue_size) : std::nullopt, perf_knobs,
        debug_config(config), config.recv_poll_period_ms, config.max_seq_idle_microseconds,
        config.has_speculative_decoding_config
            ? std::make_optional(trtllm::SpeculativeDecodingConfig(config.speculative_decoding_config.fast_logits != 0))
            : std::nullopt,
        guided_decoding_config(config), std::move(additional_outputs),
        cache_transceiver_config(config), config.gather_generation_logits != 0, config.prompt_table_offloading != 0,
        config.enable_trt_overlap != 0, config.fail_fast_on_attention_window_too_large != 0);
    return out;
}

std::optional<std::vector<int32_t>> position_ids(TensorRtLlmRsRequest const& request)
{
    if (request.position_ids_len == 0)
    {
        return std::nullopt;
    }
    if (request.position_ids == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM executor position_ids pointer is null");
    }
    if (request.position_ids_len != request.input_len)
    {
        throw std::invalid_argument("TensorRT-LLM executor position_ids length must match input_ids length");
    }
    return std::vector<int32_t>(request.position_ids, request.position_ids + request.position_ids_len);
}

std::optional<std::list<trtllm::VecTokens>> word_list(TensorRtLlmRsWordList const& input, char const* name)
{
    if (input.offset_count == 0 && input.token_count == 0)
    {
        return std::nullopt;
    }
    if (input.tokens == nullptr || input.offsets == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " word list has a null pointer");
    }
    if (input.offset_count < 2)
    {
        throw std::invalid_argument(std::string(name) + " word list must contain at least one word");
    }
    if (input.offsets[0] != 0 || input.offsets[input.offset_count - 1] != input.token_count)
    {
        throw std::invalid_argument(std::string(name) + " word list offsets do not match token count");
    }

    std::list<trtllm::VecTokens> words;
    for (size_t index = 0; index + 1 < input.offset_count; ++index)
    {
        size_t begin = input.offsets[index];
        size_t end = input.offsets[index + 1];
        if (end <= begin || end > input.token_count)
        {
            throw std::invalid_argument(std::string(name) + " word list contains an invalid token range");
        }
        words.emplace_back(input.tokens + begin, input.tokens + end);
    }
    return words;
}

std::vector<std::string> string_list(TensorRtLlmRsStringList const& input, char const* name)
{
    if (input.count == 0)
    {
        return {};
    }
    if (input.names == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " string list pointer is null");
    }

    std::vector<std::string> values;
    values.reserve(input.count);
    for (size_t index = 0; index < input.count; ++index)
    {
        if (input.names[index] == nullptr || input.names[index][0] == '\0')
        {
            throw std::invalid_argument(std::string(name) + " contains an empty name");
        }
        values.emplace_back(input.names[index]);
    }
    return values;
}

std::optional<std::vector<trtllm::AdditionalModelOutput>> additional_model_outputs(
    TensorRtLlmRsStringList const& input, char const* name)
{
    if (input.count == 0)
    {
        return std::nullopt;
    }
    if (input.names == nullptr)
    {
        throw std::invalid_argument(std::string(name) + " string list pointer is null");
    }

    std::vector<trtllm::AdditionalModelOutput> outputs;
    outputs.reserve(input.count);
    for (size_t index = 0; index < input.count; ++index)
    {
        if (input.names[index] == nullptr || input.names[index][0] == '\0')
        {
            throw std::invalid_argument(std::string(name) + " contains an empty name");
        }
        outputs.emplace_back(std::string(input.names[index]), true);
    }
    return outputs;
}

trtllm::SamplingConfig sampling_config(TensorRtLlmRsSamplingConfig const& config)
{
    trtllm::SamplingConfig sampling;
    sampling.setBeamWidth(config.beam_width);
    if (config.has_top_k)
    {
        sampling.setTopK(config.top_k);
    }
    if (config.has_top_p)
    {
        sampling.setTopP(config.top_p);
    }
    if (config.has_top_p_min)
    {
        sampling.setTopPMin(config.top_p_min);
    }
    if (config.has_top_p_reset_ids)
    {
        sampling.setTopPResetIds(config.top_p_reset_ids);
    }
    if (config.has_top_p_decay)
    {
        sampling.setTopPDecay(config.top_p_decay);
    }
    if (config.has_temperature)
    {
        sampling.setTemperature(config.temperature);
    }
    if (config.has_seed)
    {
        sampling.setSeed(config.seed);
    }
    if (config.has_min_tokens)
    {
        sampling.setMinTokens(config.min_tokens);
    }
    if (config.has_beam_search_diversity_rate)
    {
        sampling.setBeamSearchDiversityRate(config.beam_search_diversity_rate);
    }
    if (config.has_repetition_penalty)
    {
        sampling.setRepetitionPenalty(config.repetition_penalty);
    }
    if (config.has_presence_penalty)
    {
        sampling.setPresencePenalty(config.presence_penalty);
    }
    if (config.has_frequency_penalty)
    {
        sampling.setFrequencyPenalty(config.frequency_penalty);
    }
    if (config.has_prompt_ignore_length)
    {
        sampling.setPromptIgnoreLength(config.prompt_ignore_length);
    }
    if (config.has_length_penalty)
    {
        sampling.setLengthPenalty(config.length_penalty);
    }
    if (config.has_early_stopping)
    {
        sampling.setEarlyStopping(config.early_stopping);
    }
    if (config.has_no_repeat_ngram_size)
    {
        sampling.setNoRepeatNgramSize(config.no_repeat_ngram_size);
    }
    if (config.has_num_return_sequences)
    {
        sampling.setNumReturnSequences(config.num_return_sequences);
    }
    if (config.has_min_p)
    {
        sampling.setMinP(config.min_p);
    }
    if (config.beam_width_array_len > 0)
    {
        if (config.beam_width_array == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM executor beam_width_array pointer is null");
        }
        sampling.setBeamWidthArray(
            std::vector<int32_t>(config.beam_width_array, config.beam_width_array + config.beam_width_array_len));
    }
    return sampling;
}

trtllm::Request request_from_sys(TensorRtLlmRsRequest const& request)
{
    if (request.input_ids == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM executor request input_ids pointer is null");
    }
    if (request.input_len == 0)
    {
        throw std::invalid_argument("TensorRT-LLM executor request must contain at least one input token");
    }
    if (request.max_new_tokens < 0)
    {
        throw std::invalid_argument("TensorRT-LLM executor max_new_tokens cannot be negative");
    }
    if (request.prompt_embedding_rows > 0 && request.input_token_extra_ids_len != request.input_len)
    {
        throw std::invalid_argument(
            "TensorRT-LLM executor prompt tuning requires one input_token_extra_id per input token");
    }

    // TensorRT-LLM Request owns token ids, so this is the single required copy
    // from Rust's borrowed input_ids slice into the C++ Executor request.
    auto tokens = vec_tokens(request.input_ids, request.input_len, "input_ids");
    auto requested_additional_outputs = additional_model_outputs(request.additional_outputs, "additional_outputs");
    trtllm::OutputConfig output_config(request.return_log_probs != 0, request.return_context_logits != 0,
        request.return_generation_logits != 0, request.exclude_input_from_output != 0,
        request.return_encoder_output != 0, false, std::move(requested_additional_outputs));
    std::optional<int32_t> end_id = request.has_end_id ? std::make_optional(request.end_id) : std::nullopt;
    std::optional<int32_t> pad_id = request.has_pad_id ? std::make_optional(request.pad_id) : std::nullopt;
    std::optional<trtllm::IdType> client_id
        = request.has_client_id ? std::make_optional(static_cast<trtllm::IdType>(request.client_id)) : std::nullopt;
    std::optional<trtllm::MillisecondsType> allotted_time
        = request.has_allotted_time_ms ? std::make_optional(trtllm::MillisecondsType(request.allotted_time_ms))
                                       : std::nullopt;
    auto prompt_embedding = prompt_embedding_tensor(request);
    std::optional<trtllm::PromptTuningConfig> prompt_tuning = std::nullopt;
    if (prompt_embedding)
    {
        std::vector<uint64_t> extra_ids(
            request.input_token_extra_ids, request.input_token_extra_ids + request.input_token_extra_ids_len);
        prompt_tuning = trtllm::PromptTuningConfig(std::move(*prompt_embedding), std::move(extra_ids));
    }
    auto num_return_sequences = request.sampling.has_num_return_sequences ? request.sampling.num_return_sequences : 1;
    return trtllm::Request(std::move(tokens), request.max_new_tokens, request.streaming != 0,
        sampling_config(request.sampling), output_config, end_id, pad_id, position_ids(request),
        word_list(request.bad_words, "bad_words"),
        word_list(request.stop_words, "stop_words"), embedding_bias_tensor(request),
        external_draft_tokens_config(request), std::move(prompt_tuning), multimodal_input(request),
        tensor_from_view(request.multimodal_embedding, "multimodal embedding"), mrope_config(request),
        lora_config(request), lookahead_config(request), kv_cache_retention_config(request),
        optional_string(request.logits_post_processor_name), std::nullopt,
        optional_tokens(request.encoder_input_token_ids, request.encoder_input_token_count, "encoder input token ids"),
        std::move(client_id), request.return_all_generated_tokens != 0,
        request.has_priority ? request.priority : trtllm::Request::kDefaultPriority,
        static_cast<trtllm::RequestType>(request.request_type), context_phase_params(request),
        tensor_from_view(request.encoder_input_features, "encoder input features"),
        request.has_encoder_output_length ? std::make_optional(request.encoder_output_length) : std::nullopt,
        tensor_from_view(request.cross_attention_mask, "cross attention mask"), num_return_sequences,
        eagle_config(request), tensor_from_view(request.skip_cross_attn_blocks, "skip cross attention blocks"),
        guided_decoding_params(request),
        request.has_language_adapter_uid ? std::make_optional(request.language_adapter_uid) : std::nullopt,
        std::move(allotted_time), request.has_cache_salt_id ? std::make_optional(request.cache_salt_id) : std::nullopt,
        request.has_disagg_request_id ? std::make_optional(request.disagg_request_id) : std::nullopt);
}

TensorRtLlmRsOwnedTensor copy_tensor_to_owned(trtllm::Tensor const& tensor)
{
    auto cpu = tensor.copyToCpu();
    TensorRtLlmRsOwnedTensor out{};

    out.byte_len = cpu.getSizeInBytes();
    out.data_type = data_type_to_i32(cpu.getDataType());
    out.rank = cpu.getShape().size();

    if (out.byte_len > 0)
    {
        out.data = std::malloc(out.byte_len);
        if (out.data == nullptr)
        {
            throw std::bad_alloc();
        }
        std::memcpy(out.data, cpu.getData(), out.byte_len);
    }

    if (out.rank > 0)
    {
        out.shape = static_cast<int64_t*>(std::malloc(out.rank * sizeof(int64_t)));
        if (out.shape == nullptr)
        {
            std::free(out.data);
            throw std::bad_alloc();
        }
        for (size_t index = 0; index < out.rank; ++index)
        {
            out.shape[index] = cpu.getShape()[index];
        }
    }

    return out;
}

char* copy_string(std::string const& value)
{
    auto* data = static_cast<char*>(std::malloc(value.size() + 1));
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    std::memcpy(data, value.c_str(), value.size() + 1);
    return data;
}

int32_t* copy_i32_vector(std::vector<int32_t> const& values)
{
    if (values.empty())
    {
        return nullptr;
    }
    auto* data = static_cast<int32_t*>(std::malloc(values.size() * sizeof(int32_t)));
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    std::memcpy(data, values.data(), values.size() * sizeof(int32_t));
    return data;
}

uint8_t* copy_u8_vector(std::vector<char> const& values)
{
    if (values.empty())
    {
        return nullptr;
    }
    auto* data = static_cast<uint8_t*>(std::malloc(values.size()));
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    std::memcpy(data, values.data(), values.size());
    return data;
}

void copy_token_ids(trtllm::Result const& result, TensorRtLlmRsOutput* out, bool borrow_single_beam_tokens)
{
    if (result.outputTokenIds.empty())
    {
        return;
    }
    out->sequence_count = result.outputTokenIds.size();
    if (borrow_single_beam_tokens && result.outputTokenIds.size() == 1)
    {
        auto const& beam = result.outputTokenIds.front();
        out->token_count = beam.size();
        out->token_ids = const_cast<int32_t*>(beam.data());
        out->owns_token_ids = 0;
        out->sequence_count = 0;
        out->sequence_offsets = nullptr;
        out->owns_sequence_offsets = 0;
        return;
    }

    out->sequence_offsets = static_cast<size_t*>(std::malloc((out->sequence_count + 1) * sizeof(size_t)));
    if (out->sequence_offsets == nullptr)
    {
        throw std::bad_alloc();
    }
    out->owns_sequence_offsets = 1;

    size_t total_tokens = 0;
    out->sequence_offsets[0] = 0;
    for (size_t index = 0; index < result.outputTokenIds.size(); ++index)
    {
        total_tokens += result.outputTokenIds[index].size();
        out->sequence_offsets[index + 1] = total_tokens;
    }
    out->token_count = total_tokens;
    if (total_tokens == 0)
    {
        return;
    }

    out->token_ids = static_cast<int32_t*>(std::malloc(total_tokens * sizeof(int32_t)));
    if (out->token_ids == nullptr)
    {
        throw std::bad_alloc();
    }
    out->owns_token_ids = 1;
    size_t offset = 0;
    for (auto const& beam : result.outputTokenIds)
    {
        std::memcpy(out->token_ids + offset, beam.data(), beam.size() * sizeof(int32_t));
        offset += beam.size();
    }
}

void copy_log_probs(trtllm::Result const& result, TensorRtLlmRsOutput* out, bool borrow_single_sequence_log_probs)
{
    if (result.cumLogProbs && !result.cumLogProbs->empty())
    {
        out->cum_log_probs_count = result.cumLogProbs->size();
        if (borrow_single_sequence_log_probs)
        {
            out->cum_log_probs = const_cast<float*>(result.cumLogProbs->data());
            out->owns_cum_log_probs = 0;
        }
        else
        {
            out->cum_log_probs = static_cast<float*>(std::malloc(out->cum_log_probs_count * sizeof(float)));
            if (out->cum_log_probs == nullptr)
            {
                throw std::bad_alloc();
            }
            out->owns_cum_log_probs = 1;
            std::memcpy(out->cum_log_probs, result.cumLogProbs->data(), out->cum_log_probs_count * sizeof(float));
        }
    }

    if (!result.logProbs || result.logProbs->empty())
    {
        return;
    }

    out->log_prob_sequence_count = result.logProbs->size();
    if (borrow_single_sequence_log_probs && result.logProbs->size() == 1)
    {
        auto const& sequence = result.logProbs->front();
        out->log_prob_count = sequence.size();
        out->log_probs = const_cast<float*>(sequence.data());
        out->owns_log_probs = 0;
        out->log_prob_sequence_count = 0;
        out->log_prob_offsets = nullptr;
        out->owns_log_prob_offsets = 0;
        return;
    }

    out->log_prob_offsets = static_cast<size_t*>(std::malloc((out->log_prob_sequence_count + 1) * sizeof(size_t)));
    if (out->log_prob_offsets == nullptr)
    {
        throw std::bad_alloc();
    }
    out->owns_log_prob_offsets = 1;

    size_t total_log_probs = 0;
    out->log_prob_offsets[0] = 0;
    for (size_t index = 0; index < result.logProbs->size(); ++index)
    {
        total_log_probs += (*result.logProbs)[index].size();
        out->log_prob_offsets[index + 1] = total_log_probs;
    }
    out->log_prob_count = total_log_probs;
    if (total_log_probs == 0)
    {
        return;
    }

    out->log_probs = static_cast<float*>(std::malloc(total_log_probs * sizeof(float)));
    if (out->log_probs == nullptr)
    {
        throw std::bad_alloc();
    }
    out->owns_log_probs = 1;
    size_t offset = 0;
    for (auto const& sequence : *result.logProbs)
    {
        std::memcpy(out->log_probs + offset, sequence.data(), sequence.size() * sizeof(float));
        offset += sequence.size();
    }
}

void copy_context_phase_params(trtllm::ContextPhaseParams const& params, TensorRtLlmRsOwnedContextPhaseParams* out)
{
    auto const& first_gen_tokens = params.getFirstGenTokens();
    out->first_gen_tokens = copy_i32_vector(first_gen_tokens);
    out->first_gen_token_count = first_gen_tokens.size();
    out->request_id = params.getReqId();

    auto serialized_state = params.getSerializedState();
    out->serialized_state = copy_u8_vector(serialized_state);
    out->serialized_state_len = serialized_state.size();

    if (auto const& draft_tokens = params.getDraftTokens())
    {
        out->draft_tokens = copy_i32_vector(*draft_tokens);
        out->draft_token_count = draft_tokens->size();
    }
    if (auto ctx_dp_rank = params.getCtxDpRank())
    {
        out->has_ctx_dp_rank = 1;
        out->ctx_dp_rank = *ctx_dp_rank;
    }
    if (auto const& endpoint = params.getDisaggInfoEndpoint())
    {
        out->disagg_info_endpoint = copy_string(*endpoint);
    }
}

std::string_view strip_prefix(std::string_view name, std::string_view prefix)
{
    if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0)
    {
        return name.substr(prefix.size());
    }
    return {};
}

std::string_view output_lookup_name(std::string_view name)
{
    if (auto stripped = strip_prefix(name, "context_"); !stripped.empty())
    {
        return stripped;
    }
    if (auto stripped = strip_prefix(name, "generation_"); !stripped.empty())
    {
        return stripped;
    }
    return std::string_view{name};
}

TensorRtLlmRsNamedTensorView const* find_output_sink(
    TensorRtLlmRsOutputSink const* sink, std::string const& name)
{
    if (sink == nullptr || sink->tensor_count == 0)
    {
        return nullptr;
    }
    if (sink->tensors == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output sink tensor pointer is null");
    }

    auto const original_name = std::string_view{name};
    auto const lookup_name = output_lookup_name(name);
    for (size_t index = 0; index < sink->tensor_count; ++index)
    {
        auto const& candidate = sink->tensors[index];
        if (candidate.name == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM output sink contains a null name");
        }
        auto const candidate_name = std::string_view{candidate.name};
        if (original_name == candidate_name || lookup_name == candidate_name)
        {
            return &candidate;
        }
    }
    return nullptr;
}

bool should_skip_generation_counterpart(std::string const& name, TensorRtLlmRsOutputSink const* sink)
{
    auto stripped = strip_prefix(name, "generation_");
    if (stripped.empty())
    {
        return false;
    }
    if (sink == nullptr || sink->tensor_count == 0)
    {
        return false;
    }
    if (sink->tensors == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output sink tensor pointer is null");
    }
    for (size_t index = 0; index < sink->tensor_count; ++index)
    {
        auto const& candidate = sink->tensors[index];
        if (candidate.name == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM output sink contains a null name");
        }
        if (stripped == std::string_view{candidate.name})
        {
            return true;
        }
    }
    return false;
}

void validate_output_sink(TensorRtLlmRsOutputSink const* sink)
{
    if (sink == nullptr)
    {
        return;
    }
    if (sink->tensor_count == 0)
    {
        throw std::invalid_argument("TensorRT-LLM output sink requires at least one tensor");
    }
    if (sink->tensors == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output sink tensor pointer is null");
    }
    if (sink->stream == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output sink stream is null");
    }
    if (sink->stream_device < 0)
    {
        throw std::invalid_argument("TensorRT-LLM output sink stream device is invalid");
    }

    for (size_t index = 0; index < sink->tensor_count; ++index)
    {
        auto const& candidate = sink->tensors[index];
        if (candidate.name == nullptr || candidate.name[0] == '\0')
        {
            throw std::invalid_argument("TensorRT-LLM output sink contains an empty name");
        }
        if (candidate.written == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM output sink written flag pointer is null");
        }

        auto candidate_name = output_lookup_name(std::string_view{candidate.name});
        for (size_t other_index = 0; other_index < index; ++other_index)
        {
            auto const& other = sink->tensors[other_index];
            auto other_name = output_lookup_name(std::string_view{other.name});
            if (candidate_name == other_name)
            {
                throw std::invalid_argument("TensorRT-LLM output sink contains duplicate output names");
            }
        }

        auto const& view = candidate.tensor;
        if (view.memory_type != TENSOR_MEMORY_GPU)
        {
            throw std::invalid_argument("TensorRT-LLM output sink tensors must be CUDA tensors");
        }
        if (data_type_from_i32(view.data_type) == trtllm::DataType::kUNKNOWN)
        {
            throw std::invalid_argument("TensorRT-LLM output sink contains an unsupported dtype");
        }
        if (view.byte_len > 0 && view.data == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM output sink data pointer is null");
        }
        if (view.rank > 0 && view.shape == nullptr)
        {
            throw std::invalid_argument("TensorRT-LLM output sink shape pointer is null");
        }
        for (size_t dim = 0; dim < view.rank; ++dim)
        {
            if (view.shape[dim] < 0)
            {
                throw std::invalid_argument("TensorRT-LLM output sink shape contains a negative dimension");
            }
        }
    }
}

void validate_tensor_view(
    trtllm::Tensor const& tensor, TensorRtLlmRsTensorView const& view, std::string const& name)
{
    auto dtype = data_type_to_i32(tensor.getDataType());
    if (dtype == data_type_to_i32(trtllm::DataType::kUNKNOWN))
    {
        throw std::runtime_error("TensorRT-LLM output " + name + " has unsupported dtype");
    }
    if (view.data_type != dtype)
    {
        throw std::invalid_argument("TensorRT-LLM output " + name + " dtype does not match output sink");
    }
    if (view.byte_len != tensor.getSizeInBytes())
    {
        throw std::invalid_argument("TensorRT-LLM output " + name + " byte length does not match output sink");
    }
    if (view.byte_len > 0 && view.data == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output " + name + " sink data pointer is null");
    }

    auto shape = tensor.getShape();
    if (view.rank != shape.size())
    {
        throw std::invalid_argument("TensorRT-LLM output " + name + " rank does not match output sink");
    }
    if (view.rank > 0 && view.shape == nullptr)
    {
        throw std::invalid_argument("TensorRT-LLM output " + name + " sink shape pointer is null");
    }
    for (size_t index = 0; index < view.rank; ++index)
    {
        if (view.shape[index] != shape[index])
        {
            std::ostringstream message;
            message << "TensorRT-LLM output " << name << " shape does not match output sink: actual=[";
            for (size_t dim = 0; dim < shape.size(); ++dim)
            {
                if (dim != 0)
                {
                    message << ", ";
                }
                message << shape[dim];
            }
            message << "] sink=[";
            for (size_t dim = 0; dim < view.rank; ++dim)
            {
                if (dim != 0)
                {
                    message << ", ";
                }
                message << view.shape[dim];
            }
            message << "]";
            throw std::invalid_argument(message.str());
        }
    }
}

cudaMemcpyKind copy_kind_for_output(trtllm::MemoryType memory_type)
{
    switch (memory_type)
    {
    case trtllm::MemoryType::kCPU:
    case trtllm::MemoryType::kCPU_PINNED:
    case trtllm::MemoryType::kCPU_PINNEDPOOL:
        return cudaMemcpyHostToDevice;
    case trtllm::MemoryType::kGPU:
        return cudaMemcpyDeviceToDevice;
    case trtllm::MemoryType::kUVM:
        return cudaMemcpyDefault;
    case trtllm::MemoryType::kUNKNOWN:
        break;
    }
    throw std::runtime_error("TensorRT-LLM output has unknown memory type");
}

std::optional<OutputCopyStream> output_stream(TensorRtLlmRsOutputSink const* sink)
{
    if (sink == nullptr)
    {
        return std::nullopt;
    }
    validate_output_sink(sink);
    return OutputCopyStream{reinterpret_cast<cudaStream_t>(sink->stream), sink->stream_device,
        reinterpret_cast<cudaEvent_t>(sink->completion_event)};
}

void copy_tensor_to_sink(trtllm::Tensor const& tensor, TensorRtLlmRsNamedTensorView const& sink,
    std::optional<OutputCopyStream> const& stream, std::string const& name)
{
    if (!stream)
    {
        throw std::invalid_argument("TensorRT-LLM output sink stream is null");
    }
    validate_tensor_view(tensor, sink.tensor, name);
    if (tensor.getSizeInBytes() > 0)
    {
        CudaDeviceGuard guard(stream->device);
        check_cuda_result(cudaMemcpyAsync(const_cast<void*>(sink.tensor.data), tensor.getData(),
                              tensor.getSizeInBytes(), copy_kind_for_output(tensor.getMemoryType()), stream->stream),
            "cudaMemcpyAsync");
        if (stream->completionEvent == nullptr)
        {
            check_cuda_result(cudaStreamSynchronize(stream->stream), "cudaStreamSynchronize");
        }
    }
    if (sink.written != nullptr)
    {
        *sink.written = 1;
    }
}

bool try_copy_tensor_to_sink(std::string const& name, trtllm::Tensor const& tensor, TensorRtLlmRsOutputSink const* sink,
    std::optional<OutputCopyStream> const& stream)
{
    auto const* target = find_output_sink(sink, name);
    if (target == nullptr)
    {
        return false;
    }
    copy_tensor_to_sink(tensor, *target, stream, name);
    return true;
}

void copy_additional_outputs_to_sink_only(trtllm::Result const& result, TensorRtLlmRsOutputSink const* sink,
    std::optional<OutputCopyStream> const& stream)
{
    if (sink == nullptr)
    {
        return;
    }
    for (auto const& additional : result.additionalOutputs)
    {
        if (should_skip_generation_counterpart(additional.name, sink))
        {
            continue;
        }
        (void) try_copy_tensor_to_sink(additional.name, additional.output, sink, stream);
    }
}

void copy_result_to_sink_only(trtllm::Result const& result, TensorRtLlmRsOutputSink const* sink,
    std::optional<OutputCopyStream> const& stream)
{
    if (sink == nullptr)
    {
        return;
    }
    copy_additional_outputs_to_sink_only(result, sink, stream);
    if (result.contextLogits)
    {
        (void) try_copy_tensor_to_sink("context_logits", *result.contextLogits, sink, stream);
    }
    if (result.generationLogits)
    {
        (void) try_copy_tensor_to_sink("generation_logits", *result.generationLogits, sink, stream);
    }
    if (result.encoderOutput)
    {
        (void) try_copy_tensor_to_sink("encoder_output", *result.encoderOutput, sink, stream);
    }
}

void copy_response_to_sink_only(trtllm::Response const& response, TensorRtLlmRsOutputSink const* sink,
    std::optional<OutputCopyStream> const& stream)
{
    if (response.hasError())
    {
        throw std::runtime_error(response.getErrorMsg());
    }
    copy_result_to_sink_only(response.getResult(), sink, stream);
}

void copy_additional_outputs(trtllm::Result const& result, TensorRtLlmRsOutput* out,
    TensorRtLlmRsOutputSink const* sink, std::optional<OutputCopyStream> const& stream)
{
    if (result.additionalOutputs.empty())
    {
        return;
    }
    auto const output_capacity = result.additionalOutputs.size();
    size_t out_index = 0;
    for (size_t index = 0; index < result.additionalOutputs.size(); ++index)
    {
        auto const& additional = result.additionalOutputs[index];
        if (should_skip_generation_counterpart(additional.name, sink))
        {
            continue;
        }
        if (try_copy_tensor_to_sink(additional.name, additional.output, sink, stream))
        {
            continue;
        }
        if (out->named_tensors == nullptr)
        {
            out->named_tensors = static_cast<TensorRtLlmRsNamedTensor*>(
                std::calloc(output_capacity, sizeof(TensorRtLlmRsNamedTensor)));
            if (out->named_tensors == nullptr)
            {
                throw std::bad_alloc();
            }
        }
        out->named_tensors[out_index].name = copy_string(additional.name);
        out->named_tensors[out_index].tensor = copy_tensor_to_owned(additional.output);
        ++out_index;
    }
    out->named_tensor_count = out_index;
}

void copy_result(trtllm::Result const& result, TensorRtLlmRsOutput* out, TensorRtLlmRsOutputSink const* sink = nullptr,
    std::optional<OutputCopyStream> const& stream = std::nullopt, bool borrow_single_sequence_outputs = false)
{
    out->is_final = result.isFinal ? 1 : 0;
    copy_token_ids(result, out, borrow_single_sequence_outputs);
    copy_log_probs(result, out, borrow_single_sequence_outputs);
    copy_additional_outputs(result, out, sink, stream);
    if (result.contextLogits)
    {
        if (!try_copy_tensor_to_sink("context_logits", *result.contextLogits, sink, stream))
        {
            out->context_logits = copy_tensor_to_owned(*result.contextLogits);
            out->has_context_logits = 1;
        }
    }
    if (result.generationLogits)
    {
        if (!try_copy_tensor_to_sink("generation_logits", *result.generationLogits, sink, stream))
        {
            out->generation_logits = copy_tensor_to_owned(*result.generationLogits);
            out->has_generation_logits = 1;
        }
    }
    if (result.encoderOutput)
    {
        if (!try_copy_tensor_to_sink("encoder_output", *result.encoderOutput, sink, stream))
        {
            out->encoder_output = copy_tensor_to_owned(*result.encoderOutput);
            out->has_encoder_output = 1;
        }
    }
    if (result.specDecFastLogitsInfo)
    {
        out->has_spec_dec_fast_logits_info = 1;
        out->spec_dec_fast_logits_info.draft_request_id = result.specDecFastLogitsInfo->draftRequestId;
        out->spec_dec_fast_logits_info.draft_participant_id = result.specDecFastLogitsInfo->draftParticipantId;
    }
    if (!result.finishReasons.empty())
    {
        out->finish_reason_count = result.finishReasons.size();
        out->finish_reasons = static_cast<int32_t*>(std::malloc(out->finish_reason_count * sizeof(int32_t)));
        if (out->finish_reasons == nullptr)
        {
            throw std::bad_alloc();
        }
        for (size_t index = 0; index < result.finishReasons.size(); ++index)
        {
            out->finish_reasons[index] = static_cast<int32_t>(result.finishReasons[index]);
        }
    }
    if (result.contextPhaseParams)
    {
        out->has_context_phase_params = 1;
        copy_context_phase_params(*result.contextPhaseParams, &out->context_phase_params);
    }
    out->sequence_index = result.sequenceIndex;
    out->is_sequence_final = result.isSequenceFinal ? 1 : 0;
}

void copy_response(trtllm::Response const& response, TensorRtLlmRsResponse* out,
    TensorRtLlmRsOutputSink const* sink = nullptr, std::optional<OutputCopyStream> const& stream = std::nullopt,
    bool borrow_single_sequence_outputs = false)
{
    if (response.hasError())
    {
        throw std::runtime_error(response.getErrorMsg());
    }
    out->request_id = response.getRequestId();
    if (auto client_id = response.getClientId())
    {
        out->has_client_id = 1;
        out->client_id = *client_id;
    }
    copy_result(response.getResult(), &out->output, sink, stream, borrow_single_sequence_outputs);
}

void copy_responses_with_stream(std::vector<trtllm::Response> responses, TensorRtLlmRsResponse** out_responses,
    size_t* out_response_count, TensorRtLlmRsOutputSink const* sink,
    std::optional<OutputCopyStream> const& stream)
{
    *out_responses = nullptr;
    *out_response_count = 0;
    if (responses.empty())
    {
        return;
    }

    std::shared_ptr<std::vector<trtllm::Response>> retained_responses;
    auto* data = static_cast<TensorRtLlmRsResponse*>(std::calloc(responses.size(), sizeof(TensorRtLlmRsResponse)));
    if (data == nullptr)
    {
        throw std::bad_alloc();
    }
    size_t copied = 0;
    if (stream && stream->completionEvent != nullptr)
    {
        retained_responses = std::make_shared<std::vector<trtllm::Response>>(std::move(responses));
    }
    auto const& source_responses = retained_responses ? *retained_responses : responses;
    try
    {
        for (; copied < source_responses.size(); ++copied)
        {
            copy_response(source_responses[copied], &data[copied], sink, stream, retained_responses != nullptr);
            if (retained_responses)
            {
                data[copied].output.deferred_copies = new TensorRtLlmRsDeferredCopies{
                    retained_responses,
                    stream->completionEvent,
                };
            }
        }
        if (stream && stream->completionEvent != nullptr)
        {
            CudaDeviceGuard guard(stream->device);
            check_cuda_result(cudaEventRecord(stream->completionEvent, stream->stream), "cudaEventRecord");
        }
    }
    catch (...)
    {
        if (stream && stream->completionEvent != nullptr && retained_responses)
        {
            auto const status = cudaStreamSynchronize(stream->stream);
            (void) status;
        }
        tensorrt_llm_rs_responses_free(data, std::min(copied + 1, source_responses.size()));
        throw;
    }
    *out_responses = data;
    *out_response_count = source_responses.size();
}

bool is_final_or_error(trtllm::Response const& response)
{
    return response.hasError() || response.getResult().isFinal;
}

std::vector<trtllm::Response> await_final_responses(trtllm::Executor& executor, trtllm::IdType request_id,
    TensorRtLlmRsOutputSink const* sink, std::optional<OutputCopyStream> const& stream)
{
    while (true)
    {
        auto responses = executor.awaitResponses(request_id);
        if (responses.empty())
        {
            continue;
        }

        auto const has_final = std::any_of(responses.begin(), responses.end(), is_final_or_error);
        if (sink != nullptr)
        {
            for (auto const& response : responses)
            {
                if (!is_final_or_error(response))
                {
                    copy_response_to_sink_only(response, sink, stream);
                }
            }
        }
        if (has_final)
        {
            return responses;
        }
    }
}

} // namespace

extern "C"
{

int tensorrt_llm_rs_init_plugins(void)
{
    try
    {
        static std::once_flag once;
        static int status = 0;
        std::call_once(once,
            []()
            {
                if (!initTrtLlmPlugins(nullptr, tensorrt_llm::plugins::api::kDefaultNamespace))
                {
                    status = 1;
                }
            });
        return status;
    }
    catch (...)
    {
        return 2;
    }
}

int tensorrt_llm_rs_executor_create(char const* engine_dir, TensorRtLlmRsExecutorConfig const* config,
    TensorRtLlmRsExecutor** out_executor, char** out_error)
{
    return run_catching(out_error,
        [&]
        {
            if (engine_dir == nullptr || config == nullptr || out_executor == nullptr)
            {
                throw std::invalid_argument("tensorrt_llm_rs_executor_create received a null argument");
            }

            std::filesystem::path engine_path(engine_dir);
            auto executor_cfg = executor_config(*config);

            auto wrapper = std::make_unique<TensorRtLlmRsExecutor>();
            wrapper->executor
                = std::make_unique<trtllm::Executor>(engine_path, trtllm::ModelType::kDECODER_ONLY, executor_cfg);
            *out_executor = wrapper.release();
        });
}

void tensorrt_llm_rs_executor_destroy(TensorRtLlmRsExecutor* executor)
{
    delete executor;
}

int tensorrt_llm_rs_executor_run(TensorRtLlmRsExecutor* executor, TensorRtLlmRsRequest const* request,
    TensorRtLlmRsResponse** out_responses, size_t* out_response_count, char** out_error)
{
    return run_catching(out_error,
        [&]
        {
            if (executor == nullptr || request == nullptr || out_responses == nullptr || out_response_count == nullptr)
            {
                throw std::invalid_argument("tensorrt_llm_rs_executor_run received a null argument");
            }
            auto trt_request = request_from_sys(*request);
            auto request_id = executor->executor->enqueueRequest(trt_request);
            auto responses = await_final_responses(*executor->executor, request_id, nullptr, std::nullopt);
            copy_responses_with_stream(std::move(responses), out_responses, out_response_count, nullptr, std::nullopt);
        });
}

int tensorrt_llm_rs_executor_run_into(TensorRtLlmRsExecutor* executor, TensorRtLlmRsRequest const* request,
    TensorRtLlmRsOutputSink const* sink, TensorRtLlmRsResponse** out_responses, size_t* out_response_count,
    char** out_error)
{
    return run_catching(out_error,
        [&]
        {
            if (executor == nullptr || request == nullptr || sink == nullptr || out_responses == nullptr
                || out_response_count == nullptr)
            {
                throw std::invalid_argument("tensorrt_llm_rs_executor_run_into received a null argument");
            }
            auto trt_request = request_from_sys(*request);
            auto stream = output_stream(sink);
            auto request_id = executor->executor->enqueueRequest(trt_request);
            auto responses = await_final_responses(*executor->executor, request_id, sink, stream);
            copy_responses_with_stream(std::move(responses), out_responses, out_response_count, sink, stream);
        });
}

void tensorrt_llm_rs_output_free(TensorRtLlmRsOutput* output)
{
    if (output == nullptr)
    {
        return;
    }
    if (output->deferred_copies != nullptr)
    {
        auto* deferred = static_cast<TensorRtLlmRsDeferredCopies*>(output->deferred_copies);
        if (deferred->completionEvent != nullptr)
        {
            auto const status = cudaEventSynchronize(deferred->completionEvent);
            (void) status;
        }
        delete deferred;
    }
    if (output->owns_token_ids != 0)
    {
        std::free(output->token_ids);
    }
    if (output->owns_sequence_offsets != 0)
    {
        std::free(output->sequence_offsets);
    }
    if (output->owns_cum_log_probs != 0)
    {
        std::free(output->cum_log_probs);
    }
    if (output->owns_log_probs != 0)
    {
        std::free(output->log_probs);
    }
    if (output->owns_log_prob_offsets != 0)
    {
        std::free(output->log_prob_offsets);
    }
    for (size_t index = 0; index < output->named_tensor_count; ++index)
    {
        std::free(output->named_tensors[index].name);
        std::free(output->named_tensors[index].tensor.data);
        std::free(output->named_tensors[index].tensor.shape);
    }
    std::free(output->named_tensors);
    std::free(output->context_logits.data);
    std::free(output->context_logits.shape);
    std::free(output->generation_logits.data);
    std::free(output->generation_logits.shape);
    std::free(output->encoder_output.data);
    std::free(output->encoder_output.shape);
    std::free(output->finish_reasons);
    std::free(output->context_phase_params.first_gen_tokens);
    std::free(output->context_phase_params.serialized_state);
    std::free(output->context_phase_params.draft_tokens);
    std::free(output->context_phase_params.disagg_info_endpoint);
    output->is_final = 0;
    output->token_ids = nullptr;
    output->token_count = 0;
    output->owns_token_ids = 0;
    output->sequence_offsets = nullptr;
    output->sequence_count = 0;
    output->owns_sequence_offsets = 0;
    output->cum_log_probs = nullptr;
    output->cum_log_probs_count = 0;
    output->log_probs = nullptr;
    output->log_prob_count = 0;
    output->log_prob_offsets = nullptr;
    output->log_prob_sequence_count = 0;
    output->named_tensors = nullptr;
    output->named_tensor_count = 0;
    output->has_context_logits = 0;
    output->context_logits = TensorRtLlmRsOwnedTensor{};
    output->has_generation_logits = 0;
    output->generation_logits = TensorRtLlmRsOwnedTensor{};
    output->has_encoder_output = 0;
    output->encoder_output = TensorRtLlmRsOwnedTensor{};
    output->has_spec_dec_fast_logits_info = 0;
    output->spec_dec_fast_logits_info = TensorRtLlmRsSpeculativeDecodingFastLogitsInfo{};
    output->finish_reasons = nullptr;
    output->finish_reason_count = 0;
    output->has_context_phase_params = 0;
    output->context_phase_params = TensorRtLlmRsOwnedContextPhaseParams{};
    output->sequence_index = 0;
    output->is_sequence_final = 0;
    output->deferred_copies = nullptr;
}

void tensorrt_llm_rs_responses_free(TensorRtLlmRsResponse* responses, size_t response_count)
{
    if (responses == nullptr)
    {
        return;
    }
    for (size_t index = 0; index < response_count; ++index)
    {
        tensorrt_llm_rs_output_free(&responses[index].output);
        responses[index].request_id = 0;
        responses[index].has_client_id = 0;
        responses[index].client_id = 0;
    }
    std::free(responses);
}

void tensorrt_llm_rs_error_free(char* error)
{
    std::free(error);
}

} // extern "C"
