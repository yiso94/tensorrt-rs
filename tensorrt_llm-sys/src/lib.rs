#![deny(unsafe_op_in_unsafe_fn)]

//! Low-level TensorRT-LLM Executor FFI.

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct RawExecutor {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub beam_width: c_int,
    pub has_top_k: c_int,
    pub top_k: c_int,
    pub has_top_p: c_int,
    pub top_p: f32,
    pub has_top_p_min: c_int,
    pub top_p_min: f32,
    pub has_top_p_reset_ids: c_int,
    pub top_p_reset_ids: c_int,
    pub has_top_p_decay: c_int,
    pub top_p_decay: f32,
    pub has_seed: c_int,
    pub seed: u64,
    pub has_temperature: c_int,
    pub temperature: f32,
    pub has_min_tokens: c_int,
    pub min_tokens: c_int,
    pub has_beam_search_diversity_rate: c_int,
    pub beam_search_diversity_rate: f32,
    pub has_repetition_penalty: c_int,
    pub repetition_penalty: f32,
    pub has_presence_penalty: c_int,
    pub presence_penalty: f32,
    pub has_frequency_penalty: c_int,
    pub frequency_penalty: f32,
    pub has_prompt_ignore_length: c_int,
    pub prompt_ignore_length: c_int,
    pub has_length_penalty: c_int,
    pub length_penalty: f32,
    pub has_early_stopping: c_int,
    pub early_stopping: c_int,
    pub has_no_repeat_ngram_size: c_int,
    pub no_repeat_ngram_size: c_int,
    pub has_num_return_sequences: c_int,
    pub num_return_sequences: c_int,
    pub has_min_p: c_int,
    pub min_p: f32,
    pub beam_width_array: *const c_int,
    pub beam_width_array_len: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            beam_width: 1,
            has_top_k: 0,
            top_k: 0,
            has_top_p: 0,
            top_p: 0.0,
            has_top_p_min: 0,
            top_p_min: 0.0,
            has_top_p_reset_ids: 0,
            top_p_reset_ids: 0,
            has_top_p_decay: 0,
            top_p_decay: 0.0,
            has_seed: 0,
            seed: 0,
            has_temperature: 0,
            temperature: 0.0,
            has_min_tokens: 0,
            min_tokens: 0,
            has_beam_search_diversity_rate: 0,
            beam_search_diversity_rate: 0.0,
            has_repetition_penalty: 0,
            repetition_penalty: 0.0,
            has_presence_penalty: 0,
            presence_penalty: 0.0,
            has_frequency_penalty: 0,
            frequency_penalty: 0.0,
            has_prompt_ignore_length: 0,
            prompt_ignore_length: 0,
            has_length_penalty: 0,
            length_penalty: 0.0,
            has_early_stopping: 0,
            early_stopping: 0,
            has_no_repeat_ngram_size: 0,
            no_repeat_ngram_size: 0,
            has_num_return_sequences: 0,
            num_return_sequences: 0,
            has_min_p: 0,
            min_p: 0.0,
            beam_width_array: std::ptr::null(),
            beam_width_array_len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WordList {
    pub tokens: *const i32,
    pub token_count: usize,
    pub offsets: *const usize,
    pub offset_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StringList {
    pub names: *const *const c_char,
    pub count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IntList {
    pub values: *const i32,
    pub count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IntPair {
    pub first: c_int,
    pub second: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicBatchConfig {
    pub enable_batch_size_tuning: c_int,
    pub enable_max_num_tokens_tuning: c_int,
    pub moving_average_window: c_int,
    pub batch_size_table: *const IntPair,
    pub batch_size_table_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerConfig {
    pub capacity_scheduler_policy: c_int,
    pub has_context_chunking_policy: c_int,
    pub context_chunking_policy: c_int,
    pub has_dynamic_batch_config: c_int,
    pub dynamic_batch_config: DynamicBatchConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheConfig {
    pub enable_block_reuse: c_int,
    pub has_max_tokens: c_int,
    pub max_tokens: c_int,
    pub max_attention_window: IntList,
    pub has_sink_token_length: c_int,
    pub sink_token_length: c_int,
    pub has_free_gpu_memory_fraction: c_int,
    pub free_gpu_memory_fraction: f32,
    pub has_host_cache_size: c_int,
    pub host_cache_size: usize,
    pub has_cross_kv_cache_fraction: c_int,
    pub cross_kv_cache_fraction: f32,
    pub has_secondary_offload_min_priority: c_int,
    pub secondary_offload_min_priority: c_int,
    pub event_buffer_max_size: usize,
    pub enable_partial_reuse: c_int,
    pub copy_on_partial_reuse: c_int,
    pub use_uvm: c_int,
    pub attention_dp_events_gather_period_ms: c_int,
    pub has_runtime_defaults: c_int,
    pub runtime_defaults_max_attention_window: IntList,
    pub has_runtime_defaults_sink_token_length: c_int,
    pub runtime_defaults_sink_token_length: c_int,
    pub max_gpu_total_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrchestratorConfig {
    pub is_orchestrator: c_int,
    pub worker_executable_path: *const c_char,
    pub spawn_processes: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParallelConfig {
    pub communication_type: c_int,
    pub communication_mode: c_int,
    pub device_ids: IntList,
    pub participant_ids: IntList,
    pub has_orchestrator_config: c_int,
    pub orchestrator_config: OrchestratorConfig,
    pub has_num_nodes: c_int,
    pub num_nodes: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PeftCacheConfig {
    pub num_host_module_layer: c_int,
    pub num_device_module_layer: c_int,
    pub optimal_adapter_size: c_int,
    pub max_adapter_size: c_int,
    pub num_put_workers: c_int,
    pub num_ensure_workers: c_int,
    pub num_copy_streams: c_int,
    pub max_pages_per_block_host: c_int,
    pub max_pages_per_block_device: c_int,
    pub has_device_cache_percent: c_int,
    pub device_cache_percent: f32,
    pub has_host_cache_size: c_int,
    pub host_cache_size: usize,
    pub lora_prefetch_dir: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LookaheadConfig {
    pub window_size: c_int,
    pub ngram_size: c_int,
    pub verification_set_size: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EagleConfig {
    pub choices: WordList,
    pub greedy_sampling: c_int,
    pub has_posterior_threshold: c_int,
    pub posterior_threshold: f32,
    pub use_dynamic_tree: c_int,
    pub has_dynamic_tree_max_top_k: c_int,
    pub dynamic_tree_max_top_k: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodingConfig {
    pub has_decoding_mode: c_int,
    pub decoding_mode: c_int,
    pub has_lookahead_config: c_int,
    pub lookahead_config: LookaheadConfig,
    pub medusa_choices: WordList,
    pub has_eagle_config: c_int,
    pub eagle_config: EagleConfig,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExtendedRuntimePerfKnobConfig {
    pub multi_block_mode: c_int,
    pub enable_context_fmha_fp32_acc: c_int,
    pub cuda_graph_mode: c_int,
    pub cuda_graph_cache_size: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugConfig {
    pub debug_input_tensors: c_int,
    pub debug_output_tensors: c_int,
    pub debug_tensor_names: StringList,
    pub debug_tensors_max_iterations: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeculativeDecodingConfig {
    pub fast_logits: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GuidedDecodingConfig {
    pub backend: c_int,
    pub encoded_vocab: StringList,
    pub tokenizer_str: *const c_char,
    pub stop_token_ids: IntList,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheTransceiverConfig {
    pub has_backend_type: c_int,
    pub backend_type: c_int,
    pub has_max_tokens_in_buffer: c_int,
    pub max_tokens_in_buffer: usize,
    pub has_kv_transfer_timeout_ms: c_int,
    pub kv_transfer_timeout_ms: c_int,
    pub has_kv_transfer_sender_future_timeout_ms: c_int,
    pub kv_transfer_sender_future_timeout_ms: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutorConfig {
    pub max_beam_width: c_int,
    pub scheduler_config: SchedulerConfig,
    pub kv_cache_config: KvCacheConfig,
    pub enable_chunked_context: c_int,
    pub normalize_log_probs: c_int,
    pub batching_type: c_int,
    pub has_max_batch_size: c_int,
    pub max_batch_size: c_int,
    pub has_max_num_tokens: c_int,
    pub max_num_tokens: c_int,
    pub has_parallel_config: c_int,
    pub parallel_config: ParallelConfig,
    pub has_peft_cache_config: c_int,
    pub peft_cache_config: PeftCacheConfig,
    pub has_decoding_config: c_int,
    pub decoding_config: DecodingConfig,
    pub use_gpu_direct_storage: c_int,
    pub gpu_weights_percent: f32,
    pub has_max_queue_size: c_int,
    pub max_queue_size: c_int,
    pub extended_runtime_perf_knob_config: ExtendedRuntimePerfKnobConfig,
    pub has_debug_config: c_int,
    pub debug_config: DebugConfig,
    pub recv_poll_period_ms: c_int,
    pub max_seq_idle_microseconds: u64,
    pub has_speculative_decoding_config: c_int,
    pub speculative_decoding_config: SpeculativeDecodingConfig,
    pub has_guided_decoding_config: c_int,
    pub guided_decoding_config: GuidedDecodingConfig,
    pub additional_outputs: StringList,
    pub has_cache_transceiver_config: c_int,
    pub cache_transceiver_config: CacheTransceiverConfig,
    pub gather_generation_logits: c_int,
    pub prompt_table_offloading: c_int,
    pub enable_trt_overlap: c_int,
    pub fail_fast_on_attention_window_too_large: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorView {
    pub data: *const c_void,
    pub byte_len: usize,
    pub data_type: c_int,
    pub shape: *const i64,
    pub rank: usize,
    pub memory_type: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExternalDraftTokensConfig {
    pub tokens: *const i32,
    pub token_count: usize,
    pub logits: TensorView,
    pub has_acceptance_threshold: c_int,
    pub acceptance_threshold: f32,
    pub has_fast_logits: c_int,
    pub fast_logits: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MultimodalInput {
    pub hashes: WordList,
    pub positions: *const i32,
    pub position_count: usize,
    pub lengths: *const i32,
    pub length_count: usize,
    pub uuids: *const *const c_char,
    pub uuid_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MropeConfig {
    pub rotary_cos_sin: TensorView,
    pub position_deltas: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LoraConfig {
    pub task_id: u64,
    pub weights: TensorView,
    pub config: TensorView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheTokenRangeRetentionConfig {
    pub token_start: c_int,
    pub has_token_end: c_int,
    pub token_end: c_int,
    pub priority: c_int,
    pub has_duration_ms: c_int,
    pub duration_ms: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheRetentionConfig {
    pub ranges: *const KvCacheTokenRangeRetentionConfig,
    pub range_count: usize,
    pub decode_retention_priority: c_int,
    pub has_decode_duration_ms: c_int,
    pub decode_duration_ms: u64,
    pub transfer_mode: c_int,
    pub directory: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ContextPhaseParams {
    pub first_gen_tokens: *const i32,
    pub first_gen_token_count: usize,
    pub request_id: u64,
    pub serialized_state: *const u8,
    pub serialized_state_len: usize,
    pub draft_tokens: *const i32,
    pub draft_token_count: usize,
    pub has_ctx_dp_rank: c_int,
    pub ctx_dp_rank: c_int,
    pub disagg_info_endpoint: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GuidedDecodingParams {
    pub guide_type: c_int,
    pub guide: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub input_ids: *const i32,
    pub input_len: usize,
    pub max_new_tokens: i32,
    pub streaming: c_int,
    pub has_end_id: c_int,
    pub end_id: i32,
    pub has_pad_id: c_int,
    pub pad_id: i32,
    pub sampling: SamplingConfig,
    pub prompt_embedding_data: *const c_void,
    pub prompt_embedding_byte_len: usize,
    pub prompt_embedding_rows: usize,
    pub prompt_embedding_hidden_size: usize,
    pub prompt_embedding_data_type: i32,
    pub prompt_embedding_memory_type: c_int,
    pub input_token_extra_ids: *const u64,
    pub input_token_extra_ids_len: usize,
    pub position_ids: *const i32,
    pub position_ids_len: usize,
    pub bad_words: WordList,
    pub stop_words: WordList,
    pub embedding_bias_data: *const c_void,
    pub embedding_bias_byte_len: usize,
    pub embedding_bias_len: usize,
    pub embedding_bias_data_type: i32,
    pub embedding_bias_memory_type: c_int,
    pub has_external_draft_tokens_config: c_int,
    pub external_draft_tokens_config: ExternalDraftTokensConfig,
    pub has_multimodal_input: c_int,
    pub multimodal_input: MultimodalInput,
    pub multimodal_embedding: TensorView,
    pub has_mrope_config: c_int,
    pub mrope_config: MropeConfig,
    pub has_lora_config: c_int,
    pub lora_config: LoraConfig,
    pub has_lookahead_config: c_int,
    pub lookahead_config: LookaheadConfig,
    pub has_kv_cache_retention_config: c_int,
    pub kv_cache_retention_config: KvCacheRetentionConfig,
    pub logits_post_processor_name: *const c_char,
    pub encoder_input_token_ids: *const i32,
    pub encoder_input_token_count: usize,
    pub additional_outputs: StringList,
    pub return_log_probs: c_int,
    pub return_context_logits: c_int,
    pub return_generation_logits: c_int,
    pub exclude_input_from_output: c_int,
    pub return_encoder_output: c_int,
    pub return_all_generated_tokens: c_int,
    pub has_client_id: c_int,
    pub client_id: u64,
    pub has_priority: c_int,
    pub priority: f32,
    pub request_type: c_int,
    pub has_context_phase_params: c_int,
    pub context_phase_params: ContextPhaseParams,
    pub encoder_input_features: TensorView,
    pub has_encoder_output_length: c_int,
    pub encoder_output_length: c_int,
    pub cross_attention_mask: TensorView,
    pub has_eagle_config: c_int,
    pub eagle_config: EagleConfig,
    pub skip_cross_attn_blocks: TensorView,
    pub has_guided_decoding_params: c_int,
    pub guided_decoding_params: GuidedDecodingParams,
    pub has_language_adapter_uid: c_int,
    pub language_adapter_uid: c_int,
    pub has_allotted_time_ms: c_int,
    pub allotted_time_ms: u64,
    pub has_cache_salt_id: c_int,
    pub cache_salt_id: u64,
    pub has_disagg_request_id: c_int,
    pub disagg_request_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OwnedTensor {
    pub data: *mut c_void,
    pub byte_len: usize,
    pub shape: *mut i64,
    pub rank: usize,
    pub data_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NamedTensor {
    pub name: *mut c_char,
    pub tensor: OwnedTensor,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NamedTensorView {
    pub name: *const c_char,
    pub tensor: TensorView,
    pub written: *mut c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputSink {
    pub tensors: *const NamedTensorView,
    pub tensor_count: usize,
    pub stream: *mut c_void,
    pub stream_device: c_int,
    pub completion_event: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OwnedContextPhaseParams {
    pub first_gen_tokens: *mut i32,
    pub first_gen_token_count: usize,
    pub request_id: u64,
    pub serialized_state: *mut u8,
    pub serialized_state_len: usize,
    pub draft_tokens: *mut i32,
    pub draft_token_count: usize,
    pub has_ctx_dp_rank: c_int,
    pub ctx_dp_rank: c_int,
    pub disagg_info_endpoint: *mut c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeculativeDecodingFastLogitsInfo {
    pub draft_request_id: u64,
    pub draft_participant_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Output {
    pub is_final: c_int,
    pub token_ids: *mut i32,
    pub token_count: usize,
    pub owns_token_ids: c_int,
    pub sequence_offsets: *mut usize,
    pub sequence_count: usize,
    pub owns_sequence_offsets: c_int,
    pub cum_log_probs: *mut f32,
    pub cum_log_probs_count: usize,
    pub owns_cum_log_probs: c_int,
    pub log_probs: *mut f32,
    pub log_prob_count: usize,
    pub owns_log_probs: c_int,
    pub log_prob_offsets: *mut usize,
    pub log_prob_sequence_count: usize,
    pub owns_log_prob_offsets: c_int,
    pub named_tensors: *mut NamedTensor,
    pub named_tensor_count: usize,
    pub has_context_logits: c_int,
    pub context_logits: OwnedTensor,
    pub has_generation_logits: c_int,
    pub generation_logits: OwnedTensor,
    pub has_encoder_output: c_int,
    pub encoder_output: OwnedTensor,
    pub has_spec_dec_fast_logits_info: c_int,
    pub spec_dec_fast_logits_info: SpeculativeDecodingFastLogitsInfo,
    pub finish_reasons: *mut c_int,
    pub finish_reason_count: usize,
    pub has_context_phase_params: c_int,
    pub context_phase_params: OwnedContextPhaseParams,
    pub sequence_index: c_int,
    pub is_sequence_final: c_int,
    pub deferred_copies: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Response {
    pub request_id: u64,
    pub has_client_id: c_int,
    pub client_id: u64,
    pub output: Output,
}

unsafe extern "C" {
    pub fn tensorrt_llm_rs_init_plugins() -> c_int;

    pub fn tensorrt_llm_rs_executor_create(
        engine_dir: *const c_char,
        config: *const ExecutorConfig,
        out_executor: *mut *mut RawExecutor,
        out_error: *mut *mut c_char,
    ) -> c_int;

    pub fn tensorrt_llm_rs_executor_destroy(executor: *mut RawExecutor);

    pub fn tensorrt_llm_rs_executor_run(
        executor: *mut RawExecutor,
        request: *const Request,
        out_responses: *mut *mut Response,
        out_response_count: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    pub fn tensorrt_llm_rs_executor_run_into(
        executor: *mut RawExecutor,
        request: *const Request,
        sink: *const OutputSink,
        out_responses: *mut *mut Response,
        out_response_count: *mut usize,
        out_error: *mut *mut c_char,
    ) -> c_int;

    pub fn tensorrt_llm_rs_output_free(output: *mut Output);

    pub fn tensorrt_llm_rs_responses_free(responses: *mut Response, response_count: usize);

    pub fn tensorrt_llm_rs_error_free(error: *mut c_char);
}
