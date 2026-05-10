#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct TensorRtLlmRsExecutor TensorRtLlmRsExecutor;

typedef struct TensorRtLlmRsSamplingConfig
{
    int32_t beam_width;
    int32_t has_top_k;
    int32_t top_k;
    int32_t has_top_p;
    float top_p;
    int32_t has_top_p_min;
    float top_p_min;
    int32_t has_top_p_reset_ids;
    int32_t top_p_reset_ids;
    int32_t has_top_p_decay;
    float top_p_decay;
    int32_t has_seed;
    uint64_t seed;
    int32_t has_temperature;
    float temperature;
    int32_t has_min_tokens;
    int32_t min_tokens;
    int32_t has_beam_search_diversity_rate;
    float beam_search_diversity_rate;
    int32_t has_repetition_penalty;
    float repetition_penalty;
    int32_t has_presence_penalty;
    float presence_penalty;
    int32_t has_frequency_penalty;
    float frequency_penalty;
    int32_t has_prompt_ignore_length;
    int32_t prompt_ignore_length;
    int32_t has_length_penalty;
    float length_penalty;
    int32_t has_early_stopping;
    int32_t early_stopping;
    int32_t has_no_repeat_ngram_size;
    int32_t no_repeat_ngram_size;
    int32_t has_num_return_sequences;
    int32_t num_return_sequences;
    int32_t has_min_p;
    float min_p;
    int32_t const* beam_width_array;
    size_t beam_width_array_len;
} TensorRtLlmRsSamplingConfig;

typedef struct TensorRtLlmRsWordList
{
    int32_t const* tokens;
    size_t token_count;
    size_t const* offsets;
    size_t offset_count;
} TensorRtLlmRsWordList;

typedef struct TensorRtLlmRsStringList
{
    char const* const* names;
    size_t count;
} TensorRtLlmRsStringList;

typedef struct TensorRtLlmRsIntList
{
    int32_t const* values;
    size_t count;
} TensorRtLlmRsIntList;

typedef struct TensorRtLlmRsIntPair
{
    int32_t first;
    int32_t second;
} TensorRtLlmRsIntPair;

typedef struct TensorRtLlmRsDynamicBatchConfig
{
    int32_t enable_batch_size_tuning;
    int32_t enable_max_num_tokens_tuning;
    int32_t moving_average_window;
    TensorRtLlmRsIntPair const* batch_size_table;
    size_t batch_size_table_len;
} TensorRtLlmRsDynamicBatchConfig;

typedef struct TensorRtLlmRsSchedulerConfig
{
    int32_t capacity_scheduler_policy;
    int32_t has_context_chunking_policy;
    int32_t context_chunking_policy;
    int32_t has_dynamic_batch_config;
    TensorRtLlmRsDynamicBatchConfig dynamic_batch_config;
} TensorRtLlmRsSchedulerConfig;

typedef struct TensorRtLlmRsKvCacheConfig
{
    int32_t enable_block_reuse;
    int32_t has_max_tokens;
    int32_t max_tokens;
    TensorRtLlmRsIntList max_attention_window;
    int32_t has_sink_token_length;
    int32_t sink_token_length;
    int32_t has_free_gpu_memory_fraction;
    float free_gpu_memory_fraction;
    int32_t has_host_cache_size;
    size_t host_cache_size;
    int32_t has_cross_kv_cache_fraction;
    float cross_kv_cache_fraction;
    int32_t has_secondary_offload_min_priority;
    int32_t secondary_offload_min_priority;
    size_t event_buffer_max_size;
    int32_t enable_partial_reuse;
    int32_t copy_on_partial_reuse;
    int32_t use_uvm;
    int32_t attention_dp_events_gather_period_ms;
    int32_t has_runtime_defaults;
    TensorRtLlmRsIntList runtime_defaults_max_attention_window;
    int32_t has_runtime_defaults_sink_token_length;
    int32_t runtime_defaults_sink_token_length;
    uint64_t max_gpu_total_bytes;
} TensorRtLlmRsKvCacheConfig;

typedef struct TensorRtLlmRsOrchestratorConfig
{
    int32_t is_orchestrator;
    char const* worker_executable_path;
    int32_t spawn_processes;
} TensorRtLlmRsOrchestratorConfig;

typedef struct TensorRtLlmRsParallelConfig
{
    int32_t communication_type;
    int32_t communication_mode;
    TensorRtLlmRsIntList device_ids;
    TensorRtLlmRsIntList participant_ids;
    int32_t has_orchestrator_config;
    TensorRtLlmRsOrchestratorConfig orchestrator_config;
    int32_t has_num_nodes;
    int32_t num_nodes;
} TensorRtLlmRsParallelConfig;

typedef struct TensorRtLlmRsPeftCacheConfig
{
    int32_t num_host_module_layer;
    int32_t num_device_module_layer;
    int32_t optimal_adapter_size;
    int32_t max_adapter_size;
    int32_t num_put_workers;
    int32_t num_ensure_workers;
    int32_t num_copy_streams;
    int32_t max_pages_per_block_host;
    int32_t max_pages_per_block_device;
    int32_t has_device_cache_percent;
    float device_cache_percent;
    int32_t has_host_cache_size;
    size_t host_cache_size;
    char const* lora_prefetch_dir;
} TensorRtLlmRsPeftCacheConfig;

typedef struct TensorRtLlmRsLookaheadConfig
{
    int32_t window_size;
    int32_t ngram_size;
    int32_t verification_set_size;
} TensorRtLlmRsLookaheadConfig;

typedef struct TensorRtLlmRsEagleConfig
{
    TensorRtLlmRsWordList choices;
    int32_t greedy_sampling;
    int32_t has_posterior_threshold;
    float posterior_threshold;
    int32_t use_dynamic_tree;
    int32_t has_dynamic_tree_max_top_k;
    int32_t dynamic_tree_max_top_k;
} TensorRtLlmRsEagleConfig;

typedef struct TensorRtLlmRsDecodingConfig
{
    int32_t has_decoding_mode;
    int32_t decoding_mode;
    int32_t has_lookahead_config;
    TensorRtLlmRsLookaheadConfig lookahead_config;
    TensorRtLlmRsWordList medusa_choices;
    int32_t has_eagle_config;
    TensorRtLlmRsEagleConfig eagle_config;
} TensorRtLlmRsDecodingConfig;

typedef struct TensorRtLlmRsExtendedRuntimePerfKnobConfig
{
    int32_t multi_block_mode;
    int32_t enable_context_fmha_fp32_acc;
    int32_t cuda_graph_mode;
    int32_t cuda_graph_cache_size;
} TensorRtLlmRsExtendedRuntimePerfKnobConfig;

typedef struct TensorRtLlmRsDebugConfig
{
    int32_t debug_input_tensors;
    int32_t debug_output_tensors;
    TensorRtLlmRsStringList debug_tensor_names;
    int32_t debug_tensors_max_iterations;
} TensorRtLlmRsDebugConfig;

typedef struct TensorRtLlmRsSpeculativeDecodingConfig
{
    int32_t fast_logits;
} TensorRtLlmRsSpeculativeDecodingConfig;

typedef struct TensorRtLlmRsGuidedDecodingConfig
{
    int32_t backend;
    TensorRtLlmRsStringList encoded_vocab;
    char const* tokenizer_str;
    TensorRtLlmRsIntList stop_token_ids;
} TensorRtLlmRsGuidedDecodingConfig;

typedef struct TensorRtLlmRsCacheTransceiverConfig
{
    int32_t has_backend_type;
    int32_t backend_type;
    int32_t has_max_tokens_in_buffer;
    size_t max_tokens_in_buffer;
    int32_t has_kv_transfer_timeout_ms;
    int32_t kv_transfer_timeout_ms;
    int32_t has_kv_transfer_sender_future_timeout_ms;
    int32_t kv_transfer_sender_future_timeout_ms;
} TensorRtLlmRsCacheTransceiverConfig;

typedef struct TensorRtLlmRsExecutorConfig
{
    int32_t max_beam_width;
    TensorRtLlmRsSchedulerConfig scheduler_config;
    TensorRtLlmRsKvCacheConfig kv_cache_config;
    int32_t enable_chunked_context;
    int32_t normalize_log_probs;
    int32_t batching_type;
    int32_t has_max_batch_size;
    int32_t max_batch_size;
    int32_t has_max_num_tokens;
    int32_t max_num_tokens;
    int32_t has_parallel_config;
    TensorRtLlmRsParallelConfig parallel_config;
    int32_t has_peft_cache_config;
    TensorRtLlmRsPeftCacheConfig peft_cache_config;
    int32_t has_decoding_config;
    TensorRtLlmRsDecodingConfig decoding_config;
    int32_t use_gpu_direct_storage;
    float gpu_weights_percent;
    int32_t has_max_queue_size;
    int32_t max_queue_size;
    TensorRtLlmRsExtendedRuntimePerfKnobConfig extended_runtime_perf_knob_config;
    int32_t has_debug_config;
    TensorRtLlmRsDebugConfig debug_config;
    int32_t recv_poll_period_ms;
    uint64_t max_seq_idle_microseconds;
    int32_t has_speculative_decoding_config;
    TensorRtLlmRsSpeculativeDecodingConfig speculative_decoding_config;
    int32_t has_guided_decoding_config;
    TensorRtLlmRsGuidedDecodingConfig guided_decoding_config;
    TensorRtLlmRsStringList additional_outputs;
    int32_t has_cache_transceiver_config;
    TensorRtLlmRsCacheTransceiverConfig cache_transceiver_config;
    int32_t gather_generation_logits;
    int32_t prompt_table_offloading;
    int32_t enable_trt_overlap;
    int32_t fail_fast_on_attention_window_too_large;
} TensorRtLlmRsExecutorConfig;

typedef struct TensorRtLlmRsTensorView
{
    void const* data;
    size_t byte_len;
    int32_t data_type;
    int64_t const* shape;
    size_t rank;
    int32_t memory_type;
} TensorRtLlmRsTensorView;

typedef struct TensorRtLlmRsExternalDraftTokensConfig
{
    int32_t const* tokens;
    size_t token_count;
    TensorRtLlmRsTensorView logits;
    int32_t has_acceptance_threshold;
    float acceptance_threshold;
    int32_t has_fast_logits;
    int32_t fast_logits;
} TensorRtLlmRsExternalDraftTokensConfig;

typedef struct TensorRtLlmRsMultimodalInput
{
    TensorRtLlmRsWordList hashes;
    int32_t const* positions;
    size_t position_count;
    int32_t const* lengths;
    size_t length_count;
    char const* const* uuids;
    size_t uuid_count;
} TensorRtLlmRsMultimodalInput;

typedef struct TensorRtLlmRsMropeConfig
{
    TensorRtLlmRsTensorView rotary_cos_sin;
    int32_t position_deltas;
} TensorRtLlmRsMropeConfig;

typedef struct TensorRtLlmRsLoraConfig
{
    uint64_t task_id;
    TensorRtLlmRsTensorView weights;
    TensorRtLlmRsTensorView config;
} TensorRtLlmRsLoraConfig;

typedef struct TensorRtLlmRsKvCacheTokenRangeRetentionConfig
{
    int32_t token_start;
    int32_t has_token_end;
    int32_t token_end;
    int32_t priority;
    int32_t has_duration_ms;
    uint64_t duration_ms;
} TensorRtLlmRsKvCacheTokenRangeRetentionConfig;

typedef struct TensorRtLlmRsKvCacheRetentionConfig
{
    TensorRtLlmRsKvCacheTokenRangeRetentionConfig const* ranges;
    size_t range_count;
    int32_t decode_retention_priority;
    int32_t has_decode_duration_ms;
    uint64_t decode_duration_ms;
    int32_t transfer_mode;
    char const* directory;
} TensorRtLlmRsKvCacheRetentionConfig;

typedef struct TensorRtLlmRsContextPhaseParams
{
    int32_t const* first_gen_tokens;
    size_t first_gen_token_count;
    uint64_t request_id;
    uint8_t const* serialized_state;
    size_t serialized_state_len;
    int32_t const* draft_tokens;
    size_t draft_token_count;
    int32_t has_ctx_dp_rank;
    int32_t ctx_dp_rank;
    char const* disagg_info_endpoint;
} TensorRtLlmRsContextPhaseParams;

typedef struct TensorRtLlmRsGuidedDecodingParams
{
    int32_t guide_type;
    char const* guide;
} TensorRtLlmRsGuidedDecodingParams;

typedef struct TensorRtLlmRsRequest
{
    int32_t const* input_ids;
    size_t input_len;
    int32_t max_new_tokens;
    int32_t streaming;
    int32_t has_end_id;
    int32_t end_id;
    int32_t has_pad_id;
    int32_t pad_id;
    TensorRtLlmRsSamplingConfig sampling;
    void const* prompt_embedding_data;
    size_t prompt_embedding_byte_len;
    size_t prompt_embedding_rows;
    size_t prompt_embedding_hidden_size;
    int32_t prompt_embedding_data_type;
    int32_t prompt_embedding_memory_type;
    uint64_t const* input_token_extra_ids;
    size_t input_token_extra_ids_len;
    int32_t const* position_ids;
    size_t position_ids_len;
    TensorRtLlmRsWordList bad_words;
    TensorRtLlmRsWordList stop_words;
    void const* embedding_bias_data;
    size_t embedding_bias_byte_len;
    size_t embedding_bias_len;
    int32_t embedding_bias_data_type;
    int32_t embedding_bias_memory_type;
    int32_t has_external_draft_tokens_config;
    TensorRtLlmRsExternalDraftTokensConfig external_draft_tokens_config;
    int32_t has_multimodal_input;
    TensorRtLlmRsMultimodalInput multimodal_input;
    TensorRtLlmRsTensorView multimodal_embedding;
    int32_t has_mrope_config;
    TensorRtLlmRsMropeConfig mrope_config;
    int32_t has_lora_config;
    TensorRtLlmRsLoraConfig lora_config;
    int32_t has_lookahead_config;
    TensorRtLlmRsLookaheadConfig lookahead_config;
    int32_t has_kv_cache_retention_config;
    TensorRtLlmRsKvCacheRetentionConfig kv_cache_retention_config;
    char const* logits_post_processor_name;
    int32_t const* encoder_input_token_ids;
    size_t encoder_input_token_count;
    TensorRtLlmRsStringList additional_outputs;
    int32_t return_log_probs;
    int32_t return_context_logits;
    int32_t return_generation_logits;
    int32_t exclude_input_from_output;
    int32_t return_encoder_output;
    int32_t return_all_generated_tokens;
    int32_t has_client_id;
    uint64_t client_id;
    int32_t has_priority;
    float priority;
    int32_t request_type;
    int32_t has_context_phase_params;
    TensorRtLlmRsContextPhaseParams context_phase_params;
    TensorRtLlmRsTensorView encoder_input_features;
    int32_t has_encoder_output_length;
    int32_t encoder_output_length;
    TensorRtLlmRsTensorView cross_attention_mask;
    int32_t has_eagle_config;
    TensorRtLlmRsEagleConfig eagle_config;
    TensorRtLlmRsTensorView skip_cross_attn_blocks;
    int32_t has_guided_decoding_params;
    TensorRtLlmRsGuidedDecodingParams guided_decoding_params;
    int32_t has_language_adapter_uid;
    int32_t language_adapter_uid;
    int32_t has_allotted_time_ms;
    uint64_t allotted_time_ms;
    int32_t has_cache_salt_id;
    uint64_t cache_salt_id;
    int32_t has_disagg_request_id;
    uint64_t disagg_request_id;
} TensorRtLlmRsRequest;

typedef struct TensorRtLlmRsOwnedTensor
{
    void* data;
    size_t byte_len;
    int64_t* shape;
    size_t rank;
    int32_t data_type;
} TensorRtLlmRsOwnedTensor;

typedef struct TensorRtLlmRsNamedTensor
{
    char* name;
    TensorRtLlmRsOwnedTensor tensor;
} TensorRtLlmRsNamedTensor;

typedef struct TensorRtLlmRsNamedTensorView
{
    char const* name;
    TensorRtLlmRsTensorView tensor;
    int32_t* written;
} TensorRtLlmRsNamedTensorView;

typedef struct TensorRtLlmRsOutputSink
{
    TensorRtLlmRsNamedTensorView const* tensors;
    size_t tensor_count;
    void* stream;
    int32_t stream_device;
    void* completion_event;
} TensorRtLlmRsOutputSink;

typedef struct TensorRtLlmRsOwnedContextPhaseParams
{
    int32_t* first_gen_tokens;
    size_t first_gen_token_count;
    uint64_t request_id;
    uint8_t* serialized_state;
    size_t serialized_state_len;
    int32_t* draft_tokens;
    size_t draft_token_count;
    int32_t has_ctx_dp_rank;
    int32_t ctx_dp_rank;
    char* disagg_info_endpoint;
} TensorRtLlmRsOwnedContextPhaseParams;

typedef struct TensorRtLlmRsSpeculativeDecodingFastLogitsInfo
{
    uint64_t draft_request_id;
    int32_t draft_participant_id;
} TensorRtLlmRsSpeculativeDecodingFastLogitsInfo;

typedef struct TensorRtLlmRsOutput
{
    int32_t is_final;
    int32_t* token_ids;
    size_t token_count;
    int32_t owns_token_ids;
    size_t* sequence_offsets;
    size_t sequence_count;
    int32_t owns_sequence_offsets;
    float* cum_log_probs;
    size_t cum_log_probs_count;
    int32_t owns_cum_log_probs;
    float* log_probs;
    size_t log_prob_count;
    int32_t owns_log_probs;
    size_t* log_prob_offsets;
    size_t log_prob_sequence_count;
    int32_t owns_log_prob_offsets;
    TensorRtLlmRsNamedTensor* named_tensors;
    size_t named_tensor_count;
    int32_t has_context_logits;
    TensorRtLlmRsOwnedTensor context_logits;
    int32_t has_generation_logits;
    TensorRtLlmRsOwnedTensor generation_logits;
    int32_t has_encoder_output;
    TensorRtLlmRsOwnedTensor encoder_output;
    int32_t has_spec_dec_fast_logits_info;
    TensorRtLlmRsSpeculativeDecodingFastLogitsInfo spec_dec_fast_logits_info;
    int32_t* finish_reasons;
    size_t finish_reason_count;
    int32_t has_context_phase_params;
    TensorRtLlmRsOwnedContextPhaseParams context_phase_params;
    int32_t sequence_index;
    int32_t is_sequence_final;
    void* deferred_copies;
} TensorRtLlmRsOutput;

typedef struct TensorRtLlmRsResponse
{
    uint64_t request_id;
    int32_t has_client_id;
    uint64_t client_id;
    TensorRtLlmRsOutput output;
} TensorRtLlmRsResponse;

int tensorrt_llm_rs_init_plugins(void);

int tensorrt_llm_rs_executor_create(char const* engine_dir, TensorRtLlmRsExecutorConfig const* config,
    TensorRtLlmRsExecutor** out_executor, char** out_error);

void tensorrt_llm_rs_executor_destroy(TensorRtLlmRsExecutor* executor);

int tensorrt_llm_rs_executor_run(TensorRtLlmRsExecutor* executor, TensorRtLlmRsRequest const* request,
    TensorRtLlmRsResponse** out_responses, size_t* out_response_count, char** out_error);

int tensorrt_llm_rs_executor_run_into(TensorRtLlmRsExecutor* executor, TensorRtLlmRsRequest const* request,
    TensorRtLlmRsOutputSink const* sink, TensorRtLlmRsResponse** out_responses, size_t* out_response_count,
    char** out_error);

void tensorrt_llm_rs_output_free(TensorRtLlmRsOutput* output);

void tensorrt_llm_rs_responses_free(TensorRtLlmRsResponse* responses, size_t response_count);

void tensorrt_llm_rs_error_free(char* error);

#ifdef __cplusplus
}
#endif
