use crate::request::LookaheadConfig;
use crate::string::cstring;
use crate::{Error, Executor, Result};
use std::ffi::{CString, c_char};
use std::path::{Path, PathBuf};
use tensorrt_llm_sys as sys;

const DEFAULT_MAX_SEQ_IDLE_MICROSECONDS: u64 = 180_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchingType {
    Static,
    Inflight,
}

impl BatchingType {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Static => 0,
            Self::Inflight => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacitySchedulerPolicy {
    MaxUtilization,
    GuaranteedNoEvict,
    StaticBatch,
}

impl CapacitySchedulerPolicy {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::MaxUtilization => 0,
            Self::GuaranteedNoEvict => 1,
            Self::StaticBatch => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContextChunkingPolicy {
    FirstComeFirstServed,
    EqualProgress,
    ForceChunk,
}

impl ContextChunkingPolicy {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::FirstComeFirstServed => 0,
            Self::EqualProgress => 1,
            Self::ForceChunk => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynamicBatchConfig {
    pub(crate) enable_batch_size_tuning: bool,
    pub(crate) enable_max_num_tokens_tuning: bool,
    pub(crate) moving_average_window: i32,
    pub(crate) batch_size_table: Vec<(i32, i32)>,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            enable_batch_size_tuning: false,
            enable_max_num_tokens_tuning: false,
            moving_average_window: 128,
            batch_size_table: Vec::new(),
        }
    }
}

impl DynamicBatchConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tune_batch_size(mut self, enabled: bool) -> Self {
        self.enable_batch_size_tuning = enabled;
        self
    }

    pub fn tune_max_num_tokens(mut self, enabled: bool) -> Self {
        self.enable_max_num_tokens_tuning = enabled;
        self
    }

    pub fn moving_average_window(mut self, window: i32) -> Self {
        self.moving_average_window = window;
        self
    }

    pub fn batch_size_table<I>(mut self, table: I) -> Self
    where
        I: IntoIterator<Item = (i32, i32)>,
    {
        self.batch_size_table = table.into_iter().collect();
        self
    }
}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub(crate) capacity_scheduler_policy: CapacitySchedulerPolicy,
    pub(crate) context_chunking_policy: Option<ContextChunkingPolicy>,
    pub(crate) dynamic_batch_config: Option<DynamicBatchConfig>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            capacity_scheduler_policy: CapacitySchedulerPolicy::GuaranteedNoEvict,
            context_chunking_policy: None,
            dynamic_batch_config: None,
        }
    }
}

impl SchedulerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capacity_policy(mut self, policy: CapacitySchedulerPolicy) -> Self {
        self.capacity_scheduler_policy = policy;
        self
    }

    pub fn context_chunking_policy(mut self, policy: ContextChunkingPolicy) -> Self {
        self.context_chunking_policy = Some(policy);
        self
    }

    pub fn dynamic_batch(mut self, config: DynamicBatchConfig) -> Self {
        self.dynamic_batch_config = Some(config);
        self
    }
}

#[derive(Clone, Debug)]
pub struct KvCacheConfig {
    pub(crate) enable_block_reuse: bool,
    pub(crate) max_tokens: Option<i32>,
    pub(crate) max_attention_window: Vec<i32>,
    pub(crate) sink_token_length: Option<i32>,
    pub(crate) free_gpu_memory_fraction: Option<f32>,
    pub(crate) host_cache_size: Option<usize>,
    pub(crate) cross_kv_cache_fraction: Option<f32>,
    pub(crate) secondary_offload_min_priority: Option<i32>,
    pub(crate) event_buffer_max_size: usize,
    pub(crate) enable_partial_reuse: bool,
    pub(crate) copy_on_partial_reuse: bool,
    pub(crate) use_uvm: bool,
    pub(crate) attention_dp_events_gather_period_ms: i32,
    pub(crate) runtime_defaults: Option<RuntimeDefaults>,
    pub(crate) max_gpu_total_bytes: u64,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            enable_block_reuse: true,
            max_tokens: None,
            max_attention_window: Vec::new(),
            sink_token_length: None,
            free_gpu_memory_fraction: None,
            host_cache_size: None,
            cross_kv_cache_fraction: None,
            secondary_offload_min_priority: None,
            event_buffer_max_size: 0,
            enable_partial_reuse: true,
            copy_on_partial_reuse: true,
            use_uvm: false,
            attention_dp_events_gather_period_ms: 5,
            runtime_defaults: None,
            max_gpu_total_bytes: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeDefaults {
    pub(crate) max_attention_window: Vec<i32>,
    pub(crate) sink_token_length: Option<i32>,
}

impl RuntimeDefaults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_attention_window<I>(mut self, window: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.max_attention_window = window.into_iter().collect();
        self
    }

    pub fn sink_token_length(mut self, length: i32) -> Self {
        self.sink_token_length = Some(length);
        self
    }
}

impl KvCacheConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn block_reuse(mut self, enabled: bool) -> Self {
        self.enable_block_reuse = enabled;
        self
    }

    pub fn max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn max_attention_window<I>(mut self, window: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.max_attention_window = window.into_iter().collect();
        self
    }

    pub fn sink_token_length(mut self, length: i32) -> Self {
        self.sink_token_length = Some(length);
        self
    }

    pub fn free_gpu_memory_fraction(mut self, fraction: f32) -> Self {
        self.free_gpu_memory_fraction = Some(fraction);
        self
    }

    pub fn host_cache_size(mut self, bytes: usize) -> Self {
        self.host_cache_size = Some(bytes);
        self
    }

    pub fn cross_kv_cache_fraction(mut self, fraction: f32) -> Self {
        self.cross_kv_cache_fraction = Some(fraction);
        self
    }

    pub fn secondary_offload_min_priority(mut self, priority: i32) -> Self {
        self.secondary_offload_min_priority = Some(priority);
        self
    }

    pub fn event_buffer_max_size(mut self, size: usize) -> Self {
        self.event_buffer_max_size = size;
        self
    }

    pub fn partial_block_reuse(mut self, enabled: bool) -> Self {
        self.enable_partial_reuse = enabled;
        self
    }

    pub fn copy_on_partial_reuse(mut self, enabled: bool) -> Self {
        self.copy_on_partial_reuse = enabled;
        self
    }

    pub fn unified_memory(mut self, enabled: bool) -> Self {
        self.use_uvm = enabled;
        self
    }

    pub fn attention_dp_events_gather_period_ms(mut self, ms: i32) -> Self {
        self.attention_dp_events_gather_period_ms = ms;
        self
    }

    pub fn runtime_defaults(mut self, defaults: RuntimeDefaults) -> Self {
        self.runtime_defaults = Some(defaults);
        self
    }

    pub fn max_gpu_total_bytes(mut self, bytes: u64) -> Self {
        self.max_gpu_total_bytes = bytes;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunicationType {
    Mpi,
}

impl CommunicationType {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Mpi => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommunicationMode {
    Leader,
    Orchestrator,
}

impl CommunicationMode {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Leader => 0,
            Self::Orchestrator => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParallelConfig {
    pub(crate) communication_type: CommunicationType,
    pub(crate) communication_mode: CommunicationMode,
    pub(crate) device_ids: Vec<i32>,
    pub(crate) participant_ids: Vec<i32>,
    pub(crate) orchestrator_config: Option<OrchestratorConfig>,
    pub(crate) num_nodes: Option<i32>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            communication_type: CommunicationType::Mpi,
            communication_mode: CommunicationMode::Leader,
            device_ids: Vec::new(),
            participant_ids: Vec::new(),
            orchestrator_config: None,
            num_nodes: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    pub(crate) is_orchestrator: bool,
    pub(crate) worker_executable_path: Option<CString>,
    pub(crate) spawn_processes: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            is_orchestrator: true,
            worker_executable_path: None,
            spawn_processes: true,
        }
    }
}

impl OrchestratorConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.is_orchestrator = enabled;
        self
    }

    pub fn worker_executable_path(mut self, path: impl AsRef<str>) -> Self {
        self.worker_executable_path = Some(cstring(path, "orchestrator worker_executable_path"));
        self
    }

    pub fn spawn_processes(mut self, enabled: bool) -> Self {
        self.spawn_processes = enabled;
        self
    }
}

impl ParallelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn communication_type(mut self, communication_type: CommunicationType) -> Self {
        self.communication_type = communication_type;
        self
    }

    pub fn communication_mode(mut self, communication_mode: CommunicationMode) -> Self {
        self.communication_mode = communication_mode;
        self
    }

    pub fn device_ids<I>(mut self, device_ids: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.device_ids = device_ids.into_iter().collect();
        self
    }

    pub fn participant_ids<I>(mut self, participant_ids: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.participant_ids = participant_ids.into_iter().collect();
        self
    }

    pub fn orchestrator(mut self, config: OrchestratorConfig) -> Self {
        self.orchestrator_config = Some(config);
        self
    }

    pub fn num_nodes(mut self, num_nodes: i32) -> Self {
        self.num_nodes = Some(num_nodes);
        self
    }
}

#[derive(Clone, Debug)]
pub struct PeftCacheConfig {
    pub(crate) num_host_module_layer: i32,
    pub(crate) num_device_module_layer: i32,
    pub(crate) optimal_adapter_size: i32,
    pub(crate) max_adapter_size: i32,
    pub(crate) num_put_workers: i32,
    pub(crate) num_ensure_workers: i32,
    pub(crate) num_copy_streams: i32,
    pub(crate) max_pages_per_block_host: i32,
    pub(crate) max_pages_per_block_device: i32,
    pub(crate) device_cache_percent: Option<f32>,
    pub(crate) host_cache_size: Option<usize>,
    pub(crate) lora_prefetch_dir: Option<CString>,
}

impl Default for PeftCacheConfig {
    fn default() -> Self {
        Self {
            num_host_module_layer: 0,
            num_device_module_layer: 0,
            optimal_adapter_size: 8,
            max_adapter_size: 64,
            num_put_workers: 1,
            num_ensure_workers: 1,
            num_copy_streams: 1,
            max_pages_per_block_host: 24,
            max_pages_per_block_device: 8,
            device_cache_percent: None,
            host_cache_size: None,
            lora_prefetch_dir: None,
        }
    }
}

impl PeftCacheConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn host_module_layers(mut self, value: i32) -> Self {
        self.num_host_module_layer = value;
        self
    }

    pub fn device_module_layers(mut self, value: i32) -> Self {
        self.num_device_module_layer = value;
        self
    }

    pub fn optimal_adapter_size(mut self, value: i32) -> Self {
        self.optimal_adapter_size = value;
        self
    }

    pub fn max_adapter_size(mut self, value: i32) -> Self {
        self.max_adapter_size = value;
        self
    }

    pub fn put_workers(mut self, value: i32) -> Self {
        self.num_put_workers = value;
        self
    }

    pub fn ensure_workers(mut self, value: i32) -> Self {
        self.num_ensure_workers = value;
        self
    }

    pub fn copy_streams(mut self, value: i32) -> Self {
        self.num_copy_streams = value;
        self
    }

    pub fn max_pages_per_block_host(mut self, value: i32) -> Self {
        self.max_pages_per_block_host = value;
        self
    }

    pub fn max_pages_per_block_device(mut self, value: i32) -> Self {
        self.max_pages_per_block_device = value;
        self
    }

    pub fn device_cache_fraction(mut self, value: f32) -> Self {
        self.device_cache_percent = Some(value);
        self
    }

    pub fn host_cache_size(mut self, value: usize) -> Self {
        self.host_cache_size = Some(value);
        self
    }

    pub fn lora_prefetch_dir(mut self, value: impl AsRef<str>) -> Self {
        self.lora_prefetch_dir = Some(cstring(value, "lora_prefetch_dir"));
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodingMode {
    Auto,
    TopK,
    TopP,
    TopKTopP,
    BeamSearch,
    Medusa,
    Lookahead,
    ExplicitDraftTokens,
    ExternalDraftTokens,
    Eagle,
}

impl DecodingMode {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Auto => 0,
            Self::TopK => 1,
            Self::TopP => 2,
            Self::TopKTopP => 3,
            Self::BeamSearch => 4,
            Self::Medusa => 5,
            Self::Lookahead => 6,
            Self::ExplicitDraftTokens => 7,
            Self::ExternalDraftTokens => 8,
            Self::Eagle => 9,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodingConfig {
    pub(crate) decoding_mode: Option<DecodingMode>,
    pub(crate) lookahead_config: Option<LookaheadConfig>,
    pub(crate) medusa_choices: Vec<Vec<i32>>,
    pub(crate) eagle_enabled: bool,
    pub(crate) eagle_choices: Vec<Vec<i32>>,
    pub(crate) eagle_greedy_sampling: bool,
    pub(crate) eagle_posterior_threshold: Option<f32>,
    pub(crate) eagle_use_dynamic_tree: bool,
    pub(crate) eagle_dynamic_tree_max_top_k: Option<i32>,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            decoding_mode: None,
            lookahead_config: None,
            medusa_choices: Vec::new(),
            eagle_enabled: false,
            eagle_choices: Vec::new(),
            eagle_greedy_sampling: true,
            eagle_posterior_threshold: None,
            eagle_use_dynamic_tree: false,
            eagle_dynamic_tree_max_top_k: None,
        }
    }
}

impl DecodingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mode(mut self, mode: DecodingMode) -> Self {
        self.decoding_mode = Some(mode);
        self
    }

    pub fn lookahead(mut self, config: LookaheadConfig) -> Self {
        self.lookahead_config = Some(config);
        self
    }

    pub fn medusa_choices<I, J>(mut self, choices: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = i32>,
    {
        self.medusa_choices = choices
            .into_iter()
            .map(|choice| choice.into_iter().collect())
            .collect();
        self
    }

    pub fn eagle_choices<I, J>(mut self, choices: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = i32>,
    {
        self.eagle_enabled = true;
        self.eagle_choices = choices
            .into_iter()
            .map(|choice| choice.into_iter().collect())
            .collect();
        self
    }

    pub fn eagle_greedy(mut self, enabled: bool) -> Self {
        self.eagle_enabled = true;
        self.eagle_greedy_sampling = enabled;
        self
    }

    pub fn eagle_posterior_threshold(mut self, threshold: f32) -> Self {
        self.eagle_enabled = true;
        self.eagle_posterior_threshold = Some(threshold);
        self
    }

    pub fn eagle_dynamic_tree(mut self, enabled: bool) -> Self {
        self.eagle_enabled = true;
        self.eagle_use_dynamic_tree = enabled;
        self
    }

    pub fn max_eagle_dynamic_tree_top_k(mut self, top_k: i32) -> Self {
        self.eagle_enabled = true;
        self.eagle_dynamic_tree_max_top_k = Some(top_k);
        self
    }
}

#[derive(Clone, Debug)]
pub struct RuntimePerformanceConfig {
    pub(crate) multi_block_mode: bool,
    pub(crate) enable_context_fmha_fp32_acc: bool,
    pub(crate) cuda_graph_mode: bool,
    pub(crate) cuda_graph_cache_size: i32,
}

impl Default for RuntimePerformanceConfig {
    fn default() -> Self {
        Self {
            multi_block_mode: true,
            enable_context_fmha_fp32_acc: false,
            cuda_graph_mode: false,
            cuda_graph_cache_size: 0,
        }
    }
}

impl RuntimePerformanceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn multi_block(mut self, enabled: bool) -> Self {
        self.multi_block_mode = enabled;
        self
    }

    pub fn context_fmha_fp32_accumulation(mut self, enabled: bool) -> Self {
        self.enable_context_fmha_fp32_acc = enabled;
        self
    }

    pub fn cuda_graphs(mut self, enabled: bool) -> Self {
        self.cuda_graph_mode = enabled;
        self
    }

    pub fn cuda_graph_cache_size(mut self, size: i32) -> Self {
        self.cuda_graph_cache_size = size;
        self
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpeculativeDecodingConfig {
    pub(crate) fast_logits: bool,
}

impl SpeculativeDecodingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fast_logits(mut self, enabled: bool) -> Self {
        self.fast_logits = enabled;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuidedDecodingBackend {
    XGrammar,
    LlGuidance,
}

impl GuidedDecodingBackend {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::XGrammar => 0,
            Self::LlGuidance => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuidedDecodingConfig {
    pub(crate) backend: GuidedDecodingBackend,
    pub(crate) encoded_vocab: Vec<CString>,
    pub(crate) tokenizer_str: Option<CString>,
    pub(crate) stop_token_ids: Vec<i32>,
}

impl GuidedDecodingConfig {
    pub fn new(backend: GuidedDecodingBackend) -> Self {
        Self {
            backend,
            encoded_vocab: Vec::new(),
            tokenizer_str: None,
            stop_token_ids: Vec::new(),
        }
    }

    pub fn encoded_vocab<I, S>(mut self, vocab: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.encoded_vocab = vocab
            .into_iter()
            .map(|token| cstring(token, "guided_decoding encoded_vocab"))
            .collect();
        self
    }

    pub fn tokenizer(mut self, tokenizer: impl AsRef<str>) -> Self {
        self.tokenizer_str = Some(cstring(tokenizer, "guided_decoding tokenizer_str"));
        self
    }

    pub fn stop_token_ids<I>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.stop_token_ids = ids.into_iter().collect();
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheTransceiverBackend {
    Default,
    Mpi,
    Ucx,
    Nixl,
    Mooncake,
}

impl CacheTransceiverBackend {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::Mpi => 1,
            Self::Ucx => 2,
            Self::Nixl => 3,
            Self::Mooncake => 4,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CacheTransceiverConfig {
    pub(crate) backend_type: Option<CacheTransceiverBackend>,
    pub(crate) max_tokens_in_buffer: Option<usize>,
    pub(crate) kv_transfer_timeout_ms: Option<i32>,
    pub(crate) kv_transfer_sender_future_timeout_ms: Option<i32>,
}

impl CacheTransceiverConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn backend(mut self, backend: CacheTransceiverBackend) -> Self {
        self.backend_type = Some(backend);
        self
    }

    pub fn max_tokens_in_buffer(mut self, value: usize) -> Self {
        self.max_tokens_in_buffer = Some(value);
        self
    }

    pub fn transfer_timeout_ms(mut self, value: i32) -> Self {
        self.kv_transfer_timeout_ms = Some(value);
        self
    }

    pub fn sender_future_timeout_ms(mut self, value: i32) -> Self {
        self.kv_transfer_sender_future_timeout_ms = Some(value);
        self
    }
}

#[derive(Clone, Debug)]
pub struct ExecutorBuilder {
    pub(crate) engine_dir: PathBuf,
    pub(crate) max_beam_width: i32,
    pub(crate) scheduler_config: SchedulerConfig,
    pub(crate) kv_cache_config: KvCacheConfig,
    pub(crate) enable_chunked_context: bool,
    pub(crate) normalize_log_probs: bool,
    pub(crate) batching_type: BatchingType,
    pub(crate) max_batch_size: Option<i32>,
    pub(crate) max_num_tokens: Option<i32>,
    pub(crate) parallel_config: Option<ParallelConfig>,
    pub(crate) peft_cache_config: Option<PeftCacheConfig>,
    pub(crate) decoding_config: Option<DecodingConfig>,
    pub(crate) use_gpu_direct_storage: bool,
    pub(crate) gpu_weights_percent: f32,
    pub(crate) max_queue_size: Option<i32>,
    pub(crate) extended_runtime_perf_knob_config: RuntimePerformanceConfig,
    pub(crate) recv_poll_period_ms: i32,
    pub(crate) max_seq_idle_microseconds: u64,
    pub(crate) speculative_decoding_config: Option<SpeculativeDecodingConfig>,
    pub(crate) guided_decoding_config: Option<GuidedDecodingConfig>,
    pub(crate) cache_transceiver_config: Option<CacheTransceiverConfig>,
    pub(crate) additional_outputs: Vec<CString>,
    pub(crate) gather_generation_logits: bool,
    pub(crate) prompt_table_offloading: bool,
    pub(crate) enable_trt_overlap: bool,
    pub(crate) fail_fast_on_attention_window_too_large: bool,
}

impl ExecutorBuilder {
    pub fn new(engine_dir: impl AsRef<Path>) -> Self {
        Self {
            engine_dir: engine_dir.as_ref().to_owned(),
            max_beam_width: 1,
            scheduler_config: SchedulerConfig::default(),
            kv_cache_config: KvCacheConfig::default(),
            enable_chunked_context: false,
            normalize_log_probs: false,
            batching_type: BatchingType::Inflight,
            max_batch_size: None,
            max_num_tokens: None,
            parallel_config: None,
            peft_cache_config: None,
            decoding_config: None,
            use_gpu_direct_storage: false,
            gpu_weights_percent: 1.0,
            max_queue_size: None,
            extended_runtime_perf_knob_config: RuntimePerformanceConfig::default(),
            recv_poll_period_ms: 0,
            max_seq_idle_microseconds: DEFAULT_MAX_SEQ_IDLE_MICROSECONDS,
            speculative_decoding_config: None,
            guided_decoding_config: None,
            cache_transceiver_config: None,
            additional_outputs: Vec::new(),
            gather_generation_logits: false,
            prompt_table_offloading: false,
            enable_trt_overlap: false,
            fail_fast_on_attention_window_too_large: false,
        }
    }

    pub fn build(self) -> Result<Executor> {
        Executor::from_builder(self)
    }

    pub fn max_beam_width(mut self, value: i32) -> Self {
        self.max_beam_width = value;
        self
    }

    pub fn scheduler(mut self, value: SchedulerConfig) -> Self {
        self.scheduler_config = value;
        self
    }

    pub fn kv_cache(mut self, value: KvCacheConfig) -> Self {
        self.kv_cache_config = value;
        self
    }

    pub fn chunked_context(mut self, enabled: bool) -> Self {
        self.enable_chunked_context = enabled;
        self
    }

    pub fn normalize_log_probs(mut self, enabled: bool) -> Self {
        self.normalize_log_probs = enabled;
        self
    }

    pub fn batching(mut self, value: BatchingType) -> Self {
        self.batching_type = value;
        self
    }

    pub fn max_batch_size(mut self, value: i32) -> Self {
        self.max_batch_size = Some(value);
        self
    }

    pub fn max_num_tokens(mut self, value: i32) -> Self {
        self.max_num_tokens = Some(value);
        self
    }

    pub fn parallel(mut self, value: ParallelConfig) -> Self {
        self.parallel_config = Some(value);
        self
    }

    pub fn peft_cache(mut self, value: PeftCacheConfig) -> Self {
        self.peft_cache_config = Some(value);
        self
    }

    pub fn decoding(mut self, value: DecodingConfig) -> Self {
        self.decoding_config = Some(value);
        self
    }

    pub fn direct_storage(mut self, enabled: bool) -> Self {
        self.use_gpu_direct_storage = enabled;
        self
    }

    pub fn gpu_weights_fraction(mut self, value: f32) -> Self {
        self.gpu_weights_percent = value;
        self
    }

    pub fn max_queue_size(mut self, value: i32) -> Self {
        self.max_queue_size = Some(value);
        self
    }

    pub fn runtime_performance(mut self, value: RuntimePerformanceConfig) -> Self {
        self.extended_runtime_perf_knob_config = value;
        self
    }

    pub fn receive_poll_period_ms(mut self, value: i32) -> Self {
        self.recv_poll_period_ms = value;
        self
    }

    pub fn max_sequence_idle_microseconds(mut self, value: u64) -> Self {
        self.max_seq_idle_microseconds = value;
        self
    }

    pub fn speculative_decoding(mut self, value: SpeculativeDecodingConfig) -> Self {
        self.speculative_decoding_config = Some(value);
        self
    }

    pub fn guided_decoding(mut self, value: GuidedDecodingConfig) -> Self {
        self.guided_decoding_config = Some(value);
        self
    }

    pub fn cache_transceiver(mut self, value: CacheTransceiverConfig) -> Self {
        self.cache_transceiver_config = Some(value);
        self
    }

    pub fn gather_generation_logits(mut self, enabled: bool) -> Self {
        self.gather_generation_logits = enabled;
        self
    }

    pub fn offload_prompt_table(mut self, enabled: bool) -> Self {
        self.prompt_table_offloading = enabled;
        self
    }

    pub fn tensorrt_overlap(mut self, enabled: bool) -> Self {
        self.enable_trt_overlap = enabled;
        self
    }

    pub fn fail_fast_on_attention_window_too_large(mut self, enabled: bool) -> Self {
        self.fail_fast_on_attention_window_too_large = enabled;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_beam_width <= 0 {
            return Err(Error::InvalidArgument(
                "max_beam_width must be positive".into(),
            ));
        }
        if let Some(value) = self.max_batch_size
            && value <= 0
        {
            return Err(Error::InvalidArgument(
                "max_batch_size must be positive".into(),
            ));
        }
        if let Some(value) = self.max_num_tokens
            && value <= 0
        {
            return Err(Error::InvalidArgument(
                "max_num_tokens must be positive".into(),
            ));
        }
        if !self.gpu_weights_percent.is_finite() || self.gpu_weights_percent < 0.0 {
            return Err(Error::InvalidArgument(
                "gpu_weights_percent must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn sys_parts(&self) -> ExecutorConfigParts<'_> {
        let guided_vocab = self
            .guided_decoding_config
            .as_ref()
            .map(|config| StringListBuffer::new(&config.encoded_vocab));
        let dynamic_batch_table = self
            .scheduler_config
            .dynamic_batch_config
            .as_ref()
            .map(|config| IntPairListBuffer::new(&config.batch_size_table));
        let medusa_choices = self
            .decoding_config
            .as_ref()
            .map(|config| WordListBuffer::from_owned(&config.medusa_choices));
        let eagle_choices = self
            .decoding_config
            .as_ref()
            .filter(|config| config.eagle_enabled)
            .map(|config| WordListBuffer::from_owned(&config.eagle_choices));
        let additional_outputs = StringListBuffer::new(&self.additional_outputs);

        let config = sys::ExecutorConfig {
            max_beam_width: self.max_beam_width,
            scheduler_config: sys::SchedulerConfig {
                capacity_scheduler_policy: self.scheduler_config.capacity_scheduler_policy.as_i32(),
                has_context_chunking_policy: i32::from(
                    self.scheduler_config.context_chunking_policy.is_some(),
                ),
                context_chunking_policy: self
                    .scheduler_config
                    .context_chunking_policy
                    .map_or(0, ContextChunkingPolicy::as_i32),
                has_dynamic_batch_config: i32::from(
                    self.scheduler_config.dynamic_batch_config.is_some(),
                ),
                dynamic_batch_config: self
                    .scheduler_config
                    .dynamic_batch_config
                    .as_ref()
                    .map(|config| sys::DynamicBatchConfig {
                        enable_batch_size_tuning: i32::from(config.enable_batch_size_tuning),
                        enable_max_num_tokens_tuning: i32::from(
                            config.enable_max_num_tokens_tuning,
                        ),
                        moving_average_window: config.moving_average_window,
                        batch_size_table: dynamic_batch_table
                            .as_ref()
                            .map_or(std::ptr::null(), |table| table.values.as_ptr()),
                        batch_size_table_len: dynamic_batch_table
                            .as_ref()
                            .map_or(0, |table| table.values.len()),
                    })
                    .unwrap_or_default(),
            },
            kv_cache_config: sys::KvCacheConfig {
                enable_block_reuse: i32::from(self.kv_cache_config.enable_block_reuse),
                has_max_tokens: i32::from(self.kv_cache_config.max_tokens.is_some()),
                max_tokens: self.kv_cache_config.max_tokens.unwrap_or_default(),
                max_attention_window: int_list(&self.kv_cache_config.max_attention_window),
                has_sink_token_length: i32::from(self.kv_cache_config.sink_token_length.is_some()),
                sink_token_length: self.kv_cache_config.sink_token_length.unwrap_or_default(),
                has_free_gpu_memory_fraction: i32::from(
                    self.kv_cache_config.free_gpu_memory_fraction.is_some(),
                ),
                free_gpu_memory_fraction: self
                    .kv_cache_config
                    .free_gpu_memory_fraction
                    .unwrap_or_default(),
                has_host_cache_size: i32::from(self.kv_cache_config.host_cache_size.is_some()),
                host_cache_size: self.kv_cache_config.host_cache_size.unwrap_or_default(),
                has_cross_kv_cache_fraction: i32::from(
                    self.kv_cache_config.cross_kv_cache_fraction.is_some(),
                ),
                cross_kv_cache_fraction: self
                    .kv_cache_config
                    .cross_kv_cache_fraction
                    .unwrap_or_default(),
                has_secondary_offload_min_priority: i32::from(
                    self.kv_cache_config
                        .secondary_offload_min_priority
                        .is_some(),
                ),
                secondary_offload_min_priority: self
                    .kv_cache_config
                    .secondary_offload_min_priority
                    .unwrap_or_default(),
                event_buffer_max_size: self.kv_cache_config.event_buffer_max_size,
                enable_partial_reuse: i32::from(self.kv_cache_config.enable_partial_reuse),
                copy_on_partial_reuse: i32::from(self.kv_cache_config.copy_on_partial_reuse),
                use_uvm: i32::from(self.kv_cache_config.use_uvm),
                attention_dp_events_gather_period_ms: self
                    .kv_cache_config
                    .attention_dp_events_gather_period_ms,
                has_runtime_defaults: i32::from(self.kv_cache_config.runtime_defaults.is_some()),
                runtime_defaults_max_attention_window: self
                    .kv_cache_config
                    .runtime_defaults
                    .as_ref()
                    .map_or_else(sys::IntList::default, |defaults| {
                        int_list(&defaults.max_attention_window)
                    }),
                has_runtime_defaults_sink_token_length: self
                    .kv_cache_config
                    .runtime_defaults
                    .as_ref()
                    .map_or(0, |defaults| {
                        i32::from(defaults.sink_token_length.is_some())
                    }),
                runtime_defaults_sink_token_length: self
                    .kv_cache_config
                    .runtime_defaults
                    .as_ref()
                    .and_then(|defaults| defaults.sink_token_length)
                    .unwrap_or_default(),
                max_gpu_total_bytes: self.kv_cache_config.max_gpu_total_bytes,
            },
            enable_chunked_context: i32::from(self.enable_chunked_context),
            normalize_log_probs: i32::from(self.normalize_log_probs),
            batching_type: self.batching_type.as_i32(),
            has_max_batch_size: i32::from(self.max_batch_size.is_some()),
            max_batch_size: self.max_batch_size.unwrap_or_default(),
            has_max_num_tokens: i32::from(self.max_num_tokens.is_some()),
            max_num_tokens: self.max_num_tokens.unwrap_or_default(),
            has_parallel_config: i32::from(self.parallel_config.is_some()),
            parallel_config: self
                .parallel_config
                .as_ref()
                .map(|config| sys::ParallelConfig {
                    communication_type: config.communication_type.as_i32(),
                    communication_mode: config.communication_mode.as_i32(),
                    device_ids: int_list(&config.device_ids),
                    participant_ids: int_list(&config.participant_ids),
                    has_orchestrator_config: i32::from(config.orchestrator_config.is_some()),
                    orchestrator_config: config
                        .orchestrator_config
                        .as_ref()
                        .map(|config| sys::OrchestratorConfig {
                            is_orchestrator: i32::from(config.is_orchestrator),
                            worker_executable_path: config
                                .worker_executable_path
                                .as_ref()
                                .map_or(std::ptr::null(), |path| path.as_ptr()),
                            spawn_processes: i32::from(config.spawn_processes),
                        })
                        .unwrap_or_default(),
                    has_num_nodes: i32::from(config.num_nodes.is_some()),
                    num_nodes: config.num_nodes.unwrap_or_default(),
                })
                .unwrap_or_default(),
            has_peft_cache_config: i32::from(self.peft_cache_config.is_some()),
            peft_cache_config: self
                .peft_cache_config
                .as_ref()
                .map(|config| sys::PeftCacheConfig {
                    num_host_module_layer: config.num_host_module_layer,
                    num_device_module_layer: config.num_device_module_layer,
                    optimal_adapter_size: config.optimal_adapter_size,
                    max_adapter_size: config.max_adapter_size,
                    num_put_workers: config.num_put_workers,
                    num_ensure_workers: config.num_ensure_workers,
                    num_copy_streams: config.num_copy_streams,
                    max_pages_per_block_host: config.max_pages_per_block_host,
                    max_pages_per_block_device: config.max_pages_per_block_device,
                    has_device_cache_percent: i32::from(config.device_cache_percent.is_some()),
                    device_cache_percent: config.device_cache_percent.unwrap_or_default(),
                    has_host_cache_size: i32::from(config.host_cache_size.is_some()),
                    host_cache_size: config.host_cache_size.unwrap_or_default(),
                    lora_prefetch_dir: config
                        .lora_prefetch_dir
                        .as_ref()
                        .map_or(std::ptr::null(), |dir| dir.as_ptr()),
                })
                .unwrap_or_default(),
            has_decoding_config: i32::from(self.decoding_config.is_some()),
            decoding_config: self
                .decoding_config
                .as_ref()
                .map(|config| sys::DecodingConfig {
                    has_decoding_mode: i32::from(config.decoding_mode.is_some()),
                    decoding_mode: config.decoding_mode.map_or(0, DecodingMode::as_i32),
                    has_lookahead_config: i32::from(config.lookahead_config.is_some()),
                    lookahead_config: config
                        .lookahead_config
                        .map(|lookahead| sys::LookaheadConfig {
                            window_size: lookahead.window_size,
                            ngram_size: lookahead.ngram_size,
                            verification_set_size: lookahead.verification_set_size,
                        })
                        .unwrap_or_default(),
                    medusa_choices: medusa_choices
                        .as_ref()
                        .map(WordListBuffer::as_sys)
                        .unwrap_or_default(),
                    has_eagle_config: i32::from(config.eagle_enabled),
                    eagle_config: if config.eagle_enabled {
                        sys::EagleConfig {
                            choices: eagle_choices
                                .as_ref()
                                .map(WordListBuffer::as_sys)
                                .unwrap_or_default(),
                            greedy_sampling: i32::from(config.eagle_greedy_sampling),
                            has_posterior_threshold: i32::from(
                                config.eagle_posterior_threshold.is_some(),
                            ),
                            posterior_threshold: config
                                .eagle_posterior_threshold
                                .unwrap_or_default(),
                            use_dynamic_tree: i32::from(config.eagle_use_dynamic_tree),
                            has_dynamic_tree_max_top_k: i32::from(
                                config.eagle_dynamic_tree_max_top_k.is_some(),
                            ),
                            dynamic_tree_max_top_k: config
                                .eagle_dynamic_tree_max_top_k
                                .unwrap_or_default(),
                        }
                    } else {
                        sys::EagleConfig::default()
                    },
                })
                .unwrap_or_default(),
            use_gpu_direct_storage: i32::from(self.use_gpu_direct_storage),
            gpu_weights_percent: self.gpu_weights_percent,
            has_max_queue_size: i32::from(self.max_queue_size.is_some()),
            max_queue_size: self.max_queue_size.unwrap_or_default(),
            extended_runtime_perf_knob_config: sys::ExtendedRuntimePerfKnobConfig {
                multi_block_mode: i32::from(
                    self.extended_runtime_perf_knob_config.multi_block_mode,
                ),
                enable_context_fmha_fp32_acc: i32::from(
                    self.extended_runtime_perf_knob_config
                        .enable_context_fmha_fp32_acc,
                ),
                cuda_graph_mode: i32::from(self.extended_runtime_perf_knob_config.cuda_graph_mode),
                cuda_graph_cache_size: self.extended_runtime_perf_knob_config.cuda_graph_cache_size,
            },
            has_debug_config: 0,
            debug_config: sys::DebugConfig::default(),
            recv_poll_period_ms: self.recv_poll_period_ms,
            max_seq_idle_microseconds: self.max_seq_idle_microseconds,
            has_speculative_decoding_config: i32::from(self.speculative_decoding_config.is_some()),
            speculative_decoding_config: self
                .speculative_decoding_config
                .map(|config| sys::SpeculativeDecodingConfig {
                    fast_logits: i32::from(config.fast_logits),
                })
                .unwrap_or_default(),
            has_guided_decoding_config: i32::from(self.guided_decoding_config.is_some()),
            guided_decoding_config: self
                .guided_decoding_config
                .as_ref()
                .map(|config| sys::GuidedDecodingConfig {
                    backend: config.backend.as_i32(),
                    encoded_vocab: guided_vocab
                        .as_ref()
                        .map(StringListBuffer::as_sys)
                        .unwrap_or_default(),
                    tokenizer_str: config
                        .tokenizer_str
                        .as_ref()
                        .map_or(std::ptr::null(), |tokenizer| tokenizer.as_ptr()),
                    stop_token_ids: int_list(&config.stop_token_ids),
                })
                .unwrap_or_default(),
            additional_outputs: additional_outputs.as_sys(),
            has_cache_transceiver_config: i32::from(self.cache_transceiver_config.is_some()),
            cache_transceiver_config: self
                .cache_transceiver_config
                .as_ref()
                .map(|config| sys::CacheTransceiverConfig {
                    has_backend_type: i32::from(config.backend_type.is_some()),
                    backend_type: config
                        .backend_type
                        .map_or(0, CacheTransceiverBackend::as_i32),
                    has_max_tokens_in_buffer: i32::from(config.max_tokens_in_buffer.is_some()),
                    max_tokens_in_buffer: config.max_tokens_in_buffer.unwrap_or_default(),
                    has_kv_transfer_timeout_ms: i32::from(config.kv_transfer_timeout_ms.is_some()),
                    kv_transfer_timeout_ms: config.kv_transfer_timeout_ms.unwrap_or_default(),
                    has_kv_transfer_sender_future_timeout_ms: i32::from(
                        config.kv_transfer_sender_future_timeout_ms.is_some(),
                    ),
                    kv_transfer_sender_future_timeout_ms: config
                        .kv_transfer_sender_future_timeout_ms
                        .unwrap_or_default(),
                })
                .unwrap_or_default(),
            gather_generation_logits: i32::from(self.gather_generation_logits),
            prompt_table_offloading: i32::from(self.prompt_table_offloading),
            enable_trt_overlap: i32::from(self.enable_trt_overlap),
            fail_fast_on_attention_window_too_large: i32::from(
                self.fail_fast_on_attention_window_too_large,
            ),
        };
        ExecutorConfigParts {
            config,
            _guided_vocab: guided_vocab,
            _dynamic_batch_table: dynamic_batch_table,
            _medusa_choices: medusa_choices,
            _eagle_choices: eagle_choices,
            _additional_outputs: additional_outputs,
        }
    }
}

pub(crate) struct ExecutorConfigParts<'a> {
    pub(crate) config: sys::ExecutorConfig,
    _guided_vocab: Option<StringListBuffer<'a>>,
    _dynamic_batch_table: Option<IntPairListBuffer>,
    _medusa_choices: Option<WordListBuffer>,
    _eagle_choices: Option<WordListBuffer>,
    _additional_outputs: StringListBuffer<'a>,
}

struct StringListBuffer<'a> {
    names: Vec<*const c_char>,
    _strings: &'a [CString],
}

impl<'a> StringListBuffer<'a> {
    fn new(strings: &'a [CString]) -> Self {
        Self {
            names: strings.iter().map(|string| string.as_ptr()).collect(),
            _strings: strings,
        }
    }

    fn as_sys(&self) -> sys::StringList {
        sys::StringList {
            names: self.names.as_ptr(),
            count: self.names.len(),
        }
    }
}

struct WordListBuffer {
    tokens: Vec<i32>,
    offsets: Vec<usize>,
}

struct IntPairListBuffer {
    values: Vec<sys::IntPair>,
}

impl IntPairListBuffer {
    fn new(values: &[(i32, i32)]) -> Self {
        Self {
            values: values
                .iter()
                .map(|(first, second)| sys::IntPair {
                    first: *first,
                    second: *second,
                })
                .collect(),
        }
    }
}

impl WordListBuffer {
    fn from_owned(words: &[Vec<i32>]) -> Self {
        if words.is_empty() {
            return Self {
                tokens: Vec::new(),
                offsets: Vec::new(),
            };
        }
        let mut tokens = Vec::new();
        let mut offsets = Vec::with_capacity(words.len() + 1);
        offsets.push(0);
        for word in words {
            tokens.extend_from_slice(word);
            offsets.push(tokens.len());
        }
        Self { tokens, offsets }
    }

    fn as_sys(&self) -> sys::WordList {
        sys::WordList {
            tokens: self.tokens.as_ptr(),
            token_count: self.tokens.len(),
            offsets: self.offsets.as_ptr(),
            offset_count: self.offsets.len(),
        }
    }
}

fn int_list(values: &[i32]) -> sys::IntList {
    sys::IntList {
        values: values.as_ptr(),
        count: values.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_convert_to_sys_config() {
        let builder = ExecutorBuilder::new("engine")
            .max_batch_size(1)
            .max_num_tokens(8);

        builder.validate().unwrap();
        let parts = builder.sys_parts();

        assert_eq!(parts.config.max_beam_width, 1);
        assert_eq!(parts.config.has_max_batch_size, 1);
        assert_eq!(parts.config.max_batch_size, 1);
        assert_eq!(parts.config.has_max_num_tokens, 1);
        assert_eq!(parts.config.max_num_tokens, 8);
        assert_eq!(parts.config.enable_chunked_context, 0);
        assert_eq!(parts.config.batching_type, BatchingType::Inflight.as_i32());
        assert_eq!(parts.config.additional_outputs.count, 0);
    }

    #[test]
    fn builder_encodes_executor_extensions() {
        let builder = ExecutorBuilder::new("engine")
            .max_batch_size(2)
            .max_num_tokens(16)
            .max_beam_width(4)
            .chunked_context(true)
            .scheduler(
                SchedulerConfig::new().dynamic_batch(
                    DynamicBatchConfig::new()
                        .tune_batch_size(true)
                        .batch_size_table([(4, 2), (8, 4)]),
                ),
            )
            .kv_cache(
                KvCacheConfig::new().runtime_defaults(
                    RuntimeDefaults::new()
                        .max_attention_window([128, 256])
                        .sink_token_length(4),
                ),
            )
            .parallel(
                ParallelConfig::new().orchestrator(
                    OrchestratorConfig::new()
                        .worker_executable_path("/bin/worker")
                        .spawn_processes(false),
                ),
            )
            .decoding(
                DecodingConfig::new()
                    .mode(DecodingMode::Eagle)
                    .medusa_choices(vec![vec![1, 2], vec![3]])
                    .eagle_choices(vec![vec![4, 5], vec![6]])
                    .eagle_greedy(false)
                    .eagle_posterior_threshold(0.25)
                    .eagle_dynamic_tree(true)
                    .max_eagle_dynamic_tree_top_k(8),
            )
            .runtime_performance(RuntimePerformanceConfig::new().cuda_graphs(true));

        builder.validate().unwrap();
        let parts = builder.sys_parts();

        assert_eq!(parts.config.max_beam_width, 4);
        assert_eq!(parts.config.enable_chunked_context, 1);
        assert_eq!(parts.config.additional_outputs.count, 0);
        assert_eq!(
            parts
                .config
                .scheduler_config
                .dynamic_batch_config
                .batch_size_table_len,
            2
        );
        assert_eq!(parts.config.kv_cache_config.has_runtime_defaults, 1);
        assert_eq!(
            parts
                .config
                .kv_cache_config
                .runtime_defaults_max_attention_window
                .count,
            2
        );
        assert_eq!(
            parts
                .config
                .kv_cache_config
                .has_runtime_defaults_sink_token_length,
            1
        );
        assert_eq!(parts.config.has_parallel_config, 1);
        assert_eq!(parts.config.parallel_config.has_orchestrator_config, 1);
        assert_eq!(parts.config.has_decoding_config, 1);
        assert_eq!(parts.config.decoding_config.has_decoding_mode, 1);
        assert_eq!(
            parts.config.decoding_config.decoding_mode,
            DecodingMode::Eagle.as_i32()
        );
        assert_eq!(parts.config.decoding_config.medusa_choices.token_count, 3);
        assert_eq!(parts.config.decoding_config.medusa_choices.offset_count, 3);
        assert_eq!(parts.config.decoding_config.has_eagle_config, 1);
        assert_eq!(
            parts
                .config
                .decoding_config
                .eagle_config
                .choices
                .token_count,
            3
        );
        assert_eq!(
            parts
                .config
                .decoding_config
                .eagle_config
                .has_posterior_threshold,
            1
        );
        assert_eq!(
            parts
                .config
                .decoding_config
                .eagle_config
                .has_dynamic_tree_max_top_k,
            1
        );
        assert_eq!(parts.config.has_debug_config, 0);
        assert_eq!(
            parts
                .config
                .extended_runtime_perf_knob_config
                .cuda_graph_mode,
            1
        );
    }

    #[test]
    fn builder_rejects_invalid_limits() {
        assert!(
            ExecutorBuilder::new("engine")
                .max_beam_width(0)
                .validate()
                .is_err()
        );
        assert!(
            ExecutorBuilder::new("engine")
                .max_batch_size(0)
                .validate()
                .is_err()
        );
        assert!(
            ExecutorBuilder::new("engine")
                .max_num_tokens(0)
                .validate()
                .is_err()
        );
        assert!(
            ExecutorBuilder::new("engine")
                .gpu_weights_fraction(f32::NAN)
                .validate()
                .is_err()
        );
        assert!(
            ExecutorBuilder::new("engine")
                .gpu_weights_fraction(-0.1)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn enum_mappings_match_executor_ffi_contract() {
        assert_eq!(BatchingType::Static.as_i32(), 0);
        assert_eq!(BatchingType::Inflight.as_i32(), 1);

        assert_eq!(CapacitySchedulerPolicy::MaxUtilization.as_i32(), 0);
        assert_eq!(CapacitySchedulerPolicy::GuaranteedNoEvict.as_i32(), 1);
        assert_eq!(CapacitySchedulerPolicy::StaticBatch.as_i32(), 2);

        assert_eq!(ContextChunkingPolicy::FirstComeFirstServed.as_i32(), 0);
        assert_eq!(ContextChunkingPolicy::EqualProgress.as_i32(), 1);
        assert_eq!(ContextChunkingPolicy::ForceChunk.as_i32(), 2);

        assert_eq!(CommunicationType::Mpi.as_i32(), 0);
        assert_eq!(CommunicationMode::Leader.as_i32(), 0);
        assert_eq!(CommunicationMode::Orchestrator.as_i32(), 1);

        assert_eq!(DecodingMode::Auto.as_i32(), 0);
        assert_eq!(DecodingMode::TopK.as_i32(), 1);
        assert_eq!(DecodingMode::TopP.as_i32(), 2);
        assert_eq!(DecodingMode::TopKTopP.as_i32(), 3);
        assert_eq!(DecodingMode::BeamSearch.as_i32(), 4);
        assert_eq!(DecodingMode::Medusa.as_i32(), 5);
        assert_eq!(DecodingMode::Lookahead.as_i32(), 6);
        assert_eq!(DecodingMode::ExplicitDraftTokens.as_i32(), 7);
        assert_eq!(DecodingMode::ExternalDraftTokens.as_i32(), 8);
        assert_eq!(DecodingMode::Eagle.as_i32(), 9);

        assert_eq!(GuidedDecodingBackend::XGrammar.as_i32(), 0);
        assert_eq!(GuidedDecodingBackend::LlGuidance.as_i32(), 1);

        assert_eq!(CacheTransceiverBackend::Default.as_i32(), 0);
        assert_eq!(CacheTransceiverBackend::Mpi.as_i32(), 1);
        assert_eq!(CacheTransceiverBackend::Ucx.as_i32(), 2);
        assert_eq!(CacheTransceiverBackend::Nixl.as_i32(), 3);
        assert_eq!(CacheTransceiverBackend::Mooncake.as_i32(), 4);
    }

    #[test]
    fn builder_encodes_remaining_executor_options() {
        let scheduler = SchedulerConfig::new()
            .capacity_policy(CapacitySchedulerPolicy::StaticBatch)
            .context_chunking_policy(ContextChunkingPolicy::ForceChunk)
            .dynamic_batch(
                DynamicBatchConfig::new()
                    .tune_batch_size(true)
                    .tune_max_num_tokens(true)
                    .moving_average_window(64)
                    .batch_size_table([(1, 8)]),
            );
        let kv_cache = KvCacheConfig::new()
            .block_reuse(false)
            .max_tokens(128)
            .max_attention_window([64, 128])
            .sink_token_length(2)
            .free_gpu_memory_fraction(0.5)
            .host_cache_size(1024)
            .cross_kv_cache_fraction(0.25)
            .secondary_offload_min_priority(10)
            .event_buffer_max_size(32)
            .partial_block_reuse(false)
            .copy_on_partial_reuse(false)
            .unified_memory(true)
            .attention_dp_events_gather_period_ms(9)
            .max_gpu_total_bytes(4096);
        let parallel = ParallelConfig::new()
            .communication_type(CommunicationType::Mpi)
            .communication_mode(CommunicationMode::Orchestrator)
            .device_ids([0, 1])
            .participant_ids([10, 11])
            .num_nodes(2);
        let peft = PeftCacheConfig::new()
            .host_module_layers(1)
            .device_module_layers(2)
            .optimal_adapter_size(16)
            .max_adapter_size(32)
            .put_workers(3)
            .ensure_workers(4)
            .copy_streams(5)
            .max_pages_per_block_host(6)
            .max_pages_per_block_device(7)
            .device_cache_fraction(0.4)
            .host_cache_size(2048)
            .lora_prefetch_dir("/tmp/lora");
        let decoding = DecodingConfig::new()
            .mode(DecodingMode::Lookahead)
            .lookahead(LookaheadConfig::new(2, 3, 4));
        let guided = GuidedDecodingConfig::new(GuidedDecodingBackend::LlGuidance)
            .encoded_vocab(["a", "b"])
            .tokenizer("{}")
            .stop_token_ids([1, 2]);
        let cache_transceiver = CacheTransceiverConfig::new()
            .backend(CacheTransceiverBackend::Ucx)
            .max_tokens_in_buffer(123)
            .transfer_timeout_ms(45)
            .sender_future_timeout_ms(67);

        let builder = ExecutorBuilder::new("engine")
            .scheduler(scheduler)
            .kv_cache(kv_cache)
            .normalize_log_probs(true)
            .batching(BatchingType::Static)
            .parallel(parallel)
            .peft_cache(peft)
            .decoding(decoding)
            .direct_storage(true)
            .gpu_weights_fraction(0.75)
            .max_queue_size(4)
            .runtime_performance(
                RuntimePerformanceConfig::new()
                    .multi_block(false)
                    .context_fmha_fp32_accumulation(true)
                    .cuda_graph_cache_size(8),
            )
            .receive_poll_period_ms(11)
            .max_sequence_idle_microseconds(12)
            .speculative_decoding(SpeculativeDecodingConfig::new().fast_logits(true))
            .guided_decoding(guided)
            .cache_transceiver(cache_transceiver)
            .gather_generation_logits(true)
            .offload_prompt_table(true)
            .tensorrt_overlap(true)
            .fail_fast_on_attention_window_too_large(true);

        builder.validate().unwrap();
        let parts = builder.sys_parts();

        assert_eq!(
            parts.config.scheduler_config.capacity_scheduler_policy,
            CapacitySchedulerPolicy::StaticBatch.as_i32()
        );
        assert_eq!(parts.config.scheduler_config.has_context_chunking_policy, 1);
        assert_eq!(
            parts.config.scheduler_config.context_chunking_policy,
            ContextChunkingPolicy::ForceChunk.as_i32()
        );
        assert_eq!(
            parts
                .config
                .scheduler_config
                .dynamic_batch_config
                .enable_max_num_tokens_tuning,
            1
        );
        assert_eq!(parts.config.kv_cache_config.enable_block_reuse, 0);
        assert_eq!(parts.config.kv_cache_config.has_max_tokens, 1);
        assert_eq!(parts.config.kv_cache_config.max_attention_window.count, 2);
        assert_eq!(parts.config.kv_cache_config.has_sink_token_length, 1);
        assert_eq!(parts.config.kv_cache_config.has_free_gpu_memory_fraction, 1);
        assert_eq!(parts.config.kv_cache_config.has_host_cache_size, 1);
        assert_eq!(parts.config.kv_cache_config.has_cross_kv_cache_fraction, 1);
        assert_eq!(
            parts
                .config
                .kv_cache_config
                .has_secondary_offload_min_priority,
            1
        );
        assert_eq!(parts.config.kv_cache_config.event_buffer_max_size, 32);
        assert_eq!(parts.config.kv_cache_config.enable_partial_reuse, 0);
        assert_eq!(parts.config.kv_cache_config.copy_on_partial_reuse, 0);
        assert_eq!(parts.config.kv_cache_config.use_uvm, 1);
        assert_eq!(parts.config.kv_cache_config.max_gpu_total_bytes, 4096);
        assert_eq!(parts.config.normalize_log_probs, 1);
        assert_eq!(parts.config.batching_type, BatchingType::Static.as_i32());
        assert_eq!(parts.config.parallel_config.communication_mode, 1);
        assert_eq!(parts.config.parallel_config.device_ids.count, 2);
        assert_eq!(parts.config.parallel_config.participant_ids.count, 2);
        assert_eq!(parts.config.parallel_config.has_num_nodes, 1);
        assert_eq!(parts.config.has_peft_cache_config, 1);
        assert_eq!(parts.config.peft_cache_config.num_copy_streams, 5);
        assert_eq!(parts.config.peft_cache_config.has_device_cache_percent, 1);
        assert_eq!(parts.config.peft_cache_config.has_host_cache_size, 1);
        assert_eq!(parts.config.has_decoding_config, 1);
        assert_eq!(parts.config.decoding_config.has_lookahead_config, 1);
        assert_eq!(parts.config.use_gpu_direct_storage, 1);
        assert_eq!(parts.config.gpu_weights_percent, 0.75);
        assert_eq!(parts.config.has_max_queue_size, 1);
        assert_eq!(
            parts
                .config
                .extended_runtime_perf_knob_config
                .multi_block_mode,
            0
        );
        assert_eq!(
            parts
                .config
                .extended_runtime_perf_knob_config
                .enable_context_fmha_fp32_acc,
            1
        );
        assert_eq!(parts.config.recv_poll_period_ms, 11);
        assert_eq!(parts.config.max_seq_idle_microseconds, 12);
        assert_eq!(parts.config.has_speculative_decoding_config, 1);
        assert_eq!(parts.config.speculative_decoding_config.fast_logits, 1);
        assert_eq!(parts.config.has_guided_decoding_config, 1);
        assert_eq!(
            parts.config.guided_decoding_config.backend,
            GuidedDecodingBackend::LlGuidance.as_i32()
        );
        assert_eq!(parts.config.guided_decoding_config.encoded_vocab.count, 2);
        assert_eq!(parts.config.guided_decoding_config.stop_token_ids.count, 2);
        assert_eq!(parts.config.has_cache_transceiver_config, 1);
        assert_eq!(parts.config.cache_transceiver_config.has_backend_type, 1);
        assert_eq!(
            parts
                .config
                .cache_transceiver_config
                .has_max_tokens_in_buffer,
            1
        );
        assert_eq!(parts.config.gather_generation_logits, 1);
        assert_eq!(parts.config.prompt_table_offloading, 1);
        assert_eq!(parts.config.enable_trt_overlap, 1);
        assert_eq!(parts.config.fail_fast_on_attention_window_too_large, 1);
    }
}
