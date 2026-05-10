use crate::sampling::SamplingConfig;
use crate::string::cstring;
use std::ffi::CString;
#[cfg(feature = "cuda")]
use tensorrt::{CudaStream, OutputTensors};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestType {
    ContextAndGeneration,
    ContextOnly,
    GenerationOnly,
}

impl RequestType {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::ContextAndGeneration => 0,
            Self::ContextOnly => 1,
            Self::GenerationOnly => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExternalDraftTokensConfig<'a> {
    pub(crate) tokens: &'a [i32],
    pub(crate) logits: Option<&'a candle_core::Tensor>,
    pub(crate) acceptance_threshold: Option<f32>,
    pub(crate) fast_logits: Option<bool>,
}

impl<'a> ExternalDraftTokensConfig<'a> {
    pub fn new(tokens: &'a [i32]) -> Self {
        Self {
            tokens,
            logits: None,
            acceptance_threshold: None,
            fast_logits: None,
        }
    }

    pub fn logits(mut self, logits: &'a candle_core::Tensor) -> Self {
        self.logits = Some(logits);
        self
    }

    pub fn acceptance_threshold(mut self, acceptance_threshold: f32) -> Self {
        self.acceptance_threshold = Some(acceptance_threshold);
        self
    }

    pub fn fast_logits(mut self, fast_logits: bool) -> Self {
        self.fast_logits = Some(fast_logits);
        self
    }
}

#[derive(Clone, Debug)]
pub struct MultimodalInput<'a> {
    pub(crate) hashes: Vec<&'a [i32]>,
    pub(crate) positions: &'a [i32],
    pub(crate) lengths: &'a [i32],
    pub(crate) uuids: Vec<Option<CString>>,
}

impl<'a> MultimodalInput<'a> {
    pub fn new<I>(hashes: I, positions: &'a [i32], lengths: &'a [i32]) -> Self
    where
        I: IntoIterator<Item = &'a [i32]>,
    {
        Self {
            hashes: hashes.into_iter().collect(),
            positions,
            lengths,
            uuids: Vec::new(),
        }
    }

    pub fn uuids<I, S>(mut self, uuids: I) -> Self
    where
        I: IntoIterator<Item = Option<S>>,
        S: AsRef<str>,
    {
        self.uuids = uuids
            .into_iter()
            .map(|uuid| uuid.map(|uuid| cstring(uuid, "multimodal UUID")))
            .collect();
        self
    }
}

#[derive(Clone, Debug)]
pub struct MropeConfig<'a> {
    pub(crate) rotary_cos_sin: &'a candle_core::Tensor,
    pub(crate) position_deltas: i32,
}

impl<'a> MropeConfig<'a> {
    pub fn new(rotary_cos_sin: &'a candle_core::Tensor, position_deltas: i32) -> Self {
        Self {
            rotary_cos_sin,
            position_deltas,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraConfig<'a> {
    pub(crate) task_id: u64,
    pub(crate) weights: Option<&'a candle_core::Tensor>,
    pub(crate) config: Option<&'a candle_core::Tensor>,
}

impl<'a> LoraConfig<'a> {
    pub fn new(task_id: u64) -> Self {
        Self {
            task_id,
            weights: None,
            config: None,
        }
    }

    pub fn weights(mut self, weights: &'a candle_core::Tensor) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn config(mut self, config: &'a candle_core::Tensor) -> Self {
        self.config = Some(config);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LookaheadConfig {
    pub(crate) window_size: i32,
    pub(crate) ngram_size: i32,
    pub(crate) verification_set_size: i32,
}

impl LookaheadConfig {
    pub fn new(window_size: i32, ngram_size: i32, verification_set_size: i32) -> Self {
        Self {
            window_size,
            ngram_size,
            verification_set_size,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCacheTransferMode {
    Dram,
    Gds,
    PosixDebugFallback,
}

impl KvCacheTransferMode {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Dram => 0,
            Self::Gds => 1,
            Self::PosixDebugFallback => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheTokenRangeRetentionConfig {
    pub(crate) token_start: i32,
    pub(crate) token_end: Option<i32>,
    pub(crate) priority: i32,
    pub(crate) duration_ms: Option<u64>,
}

impl KvCacheTokenRangeRetentionConfig {
    pub fn new(token_start: i32) -> Self {
        Self {
            token_start,
            token_end: None,
            priority: KvCacheRetentionConfig::DEFAULT_RETENTION_PRIORITY,
            duration_ms: None,
        }
    }

    pub fn token_end(mut self, token_end: i32) -> Self {
        self.token_end = Some(token_end);
        self
    }

    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

#[derive(Clone, Debug)]
pub struct KvCacheRetentionConfig {
    pub(crate) ranges: Vec<KvCacheTokenRangeRetentionConfig>,
    pub(crate) decode_retention_priority: i32,
    pub(crate) decode_duration_ms: Option<u64>,
    pub(crate) transfer_mode: KvCacheTransferMode,
    pub(crate) directory: Option<CString>,
}

impl Default for KvCacheRetentionConfig {
    fn default() -> Self {
        Self {
            ranges: Vec::new(),
            decode_retention_priority: Self::DEFAULT_RETENTION_PRIORITY,
            decode_duration_ms: None,
            transfer_mode: KvCacheTransferMode::Dram,
            directory: None,
        }
    }
}

impl KvCacheRetentionConfig {
    pub const MIN_RETENTION_PRIORITY: i32 = 0;
    pub const MAX_RETENTION_PRIORITY: i32 = 100;
    pub const DEFAULT_RETENTION_PRIORITY: i32 = 35;

    pub fn new<I>(ranges: I) -> Self
    where
        I: IntoIterator<Item = KvCacheTokenRangeRetentionConfig>,
    {
        Self {
            ranges: ranges.into_iter().collect(),
            ..Self::default()
        }
    }

    pub fn decode_retention_priority(mut self, priority: i32) -> Self {
        self.decode_retention_priority = priority;
        self
    }

    pub fn decode_duration_ms(mut self, duration_ms: u64) -> Self {
        self.decode_duration_ms = Some(duration_ms);
        self
    }

    pub fn transfer_mode(mut self, transfer_mode: KvCacheTransferMode) -> Self {
        self.transfer_mode = transfer_mode;
        self
    }

    pub fn directory(mut self, directory: impl AsRef<str>) -> Self {
        self.directory = Some(cstring(directory, "KV cache directory"));
        self
    }
}

#[derive(Clone, Debug)]
pub struct ContextPhaseParams<'a> {
    pub(crate) first_gen_tokens: &'a [i32],
    pub(crate) request_id: u64,
    pub(crate) serialized_state: Option<&'a [u8]>,
    pub(crate) draft_tokens: Option<&'a [i32]>,
    pub(crate) ctx_dp_rank: Option<i32>,
    pub(crate) disagg_info_endpoint: Option<CString>,
}

impl<'a> ContextPhaseParams<'a> {
    pub fn new(first_gen_tokens: &'a [i32], request_id: u64) -> Self {
        Self {
            first_gen_tokens,
            request_id,
            serialized_state: None,
            draft_tokens: None,
            ctx_dp_rank: None,
            disagg_info_endpoint: None,
        }
    }

    pub fn serialized_state(mut self, serialized_state: &'a [u8]) -> Self {
        self.serialized_state = Some(serialized_state);
        self
    }

    pub fn draft_tokens(mut self, draft_tokens: &'a [i32]) -> Self {
        self.draft_tokens = Some(draft_tokens);
        self
    }

    pub fn context_data_parallel_rank(mut self, ctx_dp_rank: i32) -> Self {
        self.ctx_dp_rank = Some(ctx_dp_rank);
        self
    }

    pub fn disaggregated_info_endpoint(mut self, endpoint: impl AsRef<str>) -> Self {
        self.disagg_info_endpoint = Some(cstring(endpoint, "disaggregated endpoint"));
        self
    }
}

#[derive(Clone, Debug)]
pub struct EagleConfig<'a> {
    pub(crate) choices: Vec<&'a [i32]>,
    pub(crate) greedy_sampling: bool,
    pub(crate) posterior_threshold: Option<f32>,
    pub(crate) use_dynamic_tree: bool,
    pub(crate) dynamic_tree_max_top_k: Option<i32>,
}

impl<'a> Default for EagleConfig<'a> {
    fn default() -> Self {
        Self {
            choices: Vec::new(),
            greedy_sampling: true,
            posterior_threshold: None,
            use_dynamic_tree: false,
            dynamic_tree_max_top_k: None,
        }
    }
}

impl<'a> EagleConfig<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn choices<I>(mut self, choices: I) -> Self
    where
        I: IntoIterator<Item = &'a [i32]>,
    {
        self.choices = choices.into_iter().collect();
        self
    }

    pub fn greedy(mut self, greedy_sampling: bool) -> Self {
        self.greedy_sampling = greedy_sampling;
        self
    }

    pub fn posterior_threshold(mut self, posterior_threshold: f32) -> Self {
        self.posterior_threshold = Some(posterior_threshold);
        self
    }

    pub fn dynamic_tree(mut self, use_dynamic_tree: bool) -> Self {
        self.use_dynamic_tree = use_dynamic_tree;
        self
    }

    pub fn max_dynamic_tree_top_k(mut self, top_k: i32) -> Self {
        self.dynamic_tree_max_top_k = Some(top_k);
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuidedDecodingType {
    Json,
    JsonSchema,
    Regex,
    EbnfGrammar,
    StructuralTag,
}

impl GuidedDecodingType {
    pub(crate) fn as_i32(self) -> i32 {
        match self {
            Self::Json => 0,
            Self::JsonSchema => 1,
            Self::Regex => 2,
            Self::EbnfGrammar => 3,
            Self::StructuralTag => 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuidedDecodingParams {
    pub(crate) guide_type: GuidedDecodingType,
    pub(crate) guide: Option<CString>,
}

impl GuidedDecodingParams {
    pub fn new(guide_type: GuidedDecodingType) -> Self {
        Self {
            guide_type,
            guide: None,
        }
    }

    pub fn guide(mut self, guide: impl AsRef<str>) -> Self {
        self.guide = Some(cstring(guide, "guided decoding guide"));
        self
    }
}

#[derive(Clone, Debug)]
#[cfg(feature = "cuda")]
pub(crate) struct AdditionalOutputSink<'a> {
    pub(crate) outputs: OutputTensors,
    pub(crate) stream: CudaStream<'a>,
}

#[derive(Clone, Debug)]
pub struct Request<'a> {
    pub(crate) input_ids: &'a [i32],
    pub(crate) max_new_tokens: i32,
    pub(crate) end_id: Option<i32>,
    pub(crate) pad_id: Option<i32>,
    pub(crate) sampling: SamplingConfig,
    pub(crate) prompt_embeddings: Option<&'a candle_core::Tensor>,
    pub(crate) prompt_embedding_token_extra_ids: Option<&'a [u64]>,
    pub(crate) position_ids: Option<&'a [i32]>,
    pub(crate) bad_words: Vec<&'a [i32]>,
    pub(crate) stop_words: Vec<&'a [i32]>,
    pub(crate) embedding_bias: Option<&'a candle_core::Tensor>,
    pub(crate) external_draft_tokens_config: Option<ExternalDraftTokensConfig<'a>>,
    pub(crate) multimodal_input: Option<MultimodalInput<'a>>,
    pub(crate) multimodal_embedding: Option<&'a candle_core::Tensor>,
    pub(crate) mrope_config: Option<MropeConfig<'a>>,
    pub(crate) lora_config: Option<LoraConfig<'a>>,
    pub(crate) lookahead_config: Option<LookaheadConfig>,
    pub(crate) kv_cache_retention_config: Option<KvCacheRetentionConfig>,
    pub(crate) logits_post_processor_name: Option<CString>,
    pub(crate) encoder_input_token_ids: Option<&'a [i32]>,
    pub(crate) additional_outputs: Vec<CString>,
    pub(crate) return_log_probs: bool,
    pub(crate) return_context_logits: bool,
    pub(crate) return_generation_logits: bool,
    pub(crate) exclude_input_from_output: bool,
    pub(crate) return_encoder_output: bool,
    pub(crate) return_all_generated_tokens: bool,
    pub(crate) client_id: Option<u64>,
    pub(crate) priority: Option<f32>,
    pub(crate) request_type: RequestType,
    pub(crate) context_phase_params: Option<ContextPhaseParams<'a>>,
    pub(crate) encoder_input_features: Option<&'a candle_core::Tensor>,
    pub(crate) encoder_output_length: Option<i32>,
    pub(crate) cross_attention_mask: Option<&'a candle_core::Tensor>,
    pub(crate) eagle_config: Option<EagleConfig<'a>>,
    pub(crate) skip_cross_attn_blocks: Option<&'a candle_core::Tensor>,
    pub(crate) guided_decoding_params: Option<GuidedDecodingParams>,
    pub(crate) language_adapter_uid: Option<i32>,
    pub(crate) allotted_time_ms: Option<u64>,
    pub(crate) cache_salt_id: Option<u64>,
    pub(crate) disagg_request_id: Option<u64>,
    #[cfg(feature = "cuda")]
    pub(crate) additional_output_sink: Option<AdditionalOutputSink<'a>>,
}

impl<'a> Request<'a> {
    pub fn new(input_ids: &'a [i32]) -> Self {
        Self {
            input_ids,
            max_new_tokens: 1,
            end_id: None,
            pad_id: None,
            sampling: SamplingConfig::default(),
            prompt_embeddings: None,
            prompt_embedding_token_extra_ids: None,
            position_ids: None,
            bad_words: Vec::new(),
            stop_words: Vec::new(),
            embedding_bias: None,
            external_draft_tokens_config: None,
            multimodal_input: None,
            multimodal_embedding: None,
            mrope_config: None,
            lora_config: None,
            lookahead_config: None,
            kv_cache_retention_config: None,
            logits_post_processor_name: None,
            encoder_input_token_ids: None,
            additional_outputs: Vec::new(),
            return_log_probs: false,
            return_context_logits: false,
            return_generation_logits: false,
            exclude_input_from_output: false,
            return_encoder_output: false,
            return_all_generated_tokens: false,
            client_id: None,
            priority: None,
            request_type: RequestType::ContextAndGeneration,
            context_phase_params: None,
            encoder_input_features: None,
            encoder_output_length: None,
            cross_attention_mask: None,
            eagle_config: None,
            skip_cross_attn_blocks: None,
            guided_decoding_params: None,
            language_adapter_uid: None,
            allotted_time_ms: None,
            cache_salt_id: None,
            disagg_request_id: None,
            #[cfg(feature = "cuda")]
            additional_output_sink: None,
        }
    }

    pub fn max_new_tokens(mut self, max_new_tokens: i32) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub fn end_id(mut self, end_id: i32) -> Self {
        self.end_id = Some(end_id);
        self
    }

    pub fn pad_id(mut self, pad_id: i32) -> Self {
        self.pad_id = Some(pad_id);
        self
    }

    pub fn sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn prompt_embeddings(mut self, prompt_embeddings: &'a candle_core::Tensor) -> Self {
        self.prompt_embeddings = Some(prompt_embeddings);
        self
    }

    pub fn prompt_embedding_token_extra_ids(mut self, token_extra_ids: &'a [u64]) -> Self {
        self.prompt_embedding_token_extra_ids = Some(token_extra_ids);
        self
    }

    pub fn position_ids(mut self, position_ids: &'a [i32]) -> Self {
        self.position_ids = Some(position_ids);
        self
    }

    pub fn bad_words<I>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = &'a [i32]>,
    {
        self.bad_words = words.into_iter().collect();
        self
    }

    pub fn stop_words<I>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = &'a [i32]>,
    {
        self.stop_words = words.into_iter().collect();
        self
    }

    pub fn embedding_bias(mut self, embedding_bias: &'a candle_core::Tensor) -> Self {
        self.embedding_bias = Some(embedding_bias);
        self
    }

    pub fn external_draft_tokens(mut self, config: ExternalDraftTokensConfig<'a>) -> Self {
        self.external_draft_tokens_config = Some(config);
        self
    }

    pub fn multimodal_input(mut self, input: MultimodalInput<'a>) -> Self {
        self.multimodal_input = Some(input);
        self
    }

    pub fn multimodal_embedding(mut self, embedding: &'a candle_core::Tensor) -> Self {
        self.multimodal_embedding = Some(embedding);
        self
    }

    pub fn mrope(mut self, config: MropeConfig<'a>) -> Self {
        self.mrope_config = Some(config);
        self
    }

    pub fn lora(mut self, config: LoraConfig<'a>) -> Self {
        self.lora_config = Some(config);
        self
    }

    pub fn lookahead(mut self, config: LookaheadConfig) -> Self {
        self.lookahead_config = Some(config);
        self
    }

    pub fn kv_cache_retention(mut self, config: KvCacheRetentionConfig) -> Self {
        self.kv_cache_retention_config = Some(config);
        self
    }

    pub fn logits_processor(mut self, name: impl AsRef<str>) -> Self {
        self.logits_post_processor_name = Some(cstring(name, "logits postprocessor name"));
        self
    }

    pub fn encoder_input_token_ids(mut self, token_ids: &'a [i32]) -> Self {
        self.encoder_input_token_ids = Some(token_ids);
        self
    }

    pub fn return_log_probs(mut self) -> Self {
        self.return_log_probs = true;
        self
    }

    pub fn return_context_logits(mut self) -> Self {
        self.return_context_logits = true;
        self
    }

    pub fn return_generation_logits(mut self) -> Self {
        self.return_generation_logits = true;
        self
    }

    pub fn exclude_input_from_output(mut self) -> Self {
        self.exclude_input_from_output = true;
        self
    }

    pub fn return_encoder_output(mut self) -> Self {
        self.return_encoder_output = true;
        self
    }

    pub fn return_all_generated_tokens(mut self) -> Self {
        self.return_all_generated_tokens = true;
        self
    }

    pub fn client_id(mut self, client_id: u64) -> Self {
        self.client_id = Some(client_id);
        self
    }

    pub fn priority(mut self, priority: f32) -> Self {
        self.priority = Some(priority);
        self
    }

    pub fn request_type(mut self, request_type: RequestType) -> Self {
        self.request_type = request_type;
        self
    }

    pub fn context_phase(mut self, params: ContextPhaseParams<'a>) -> Self {
        self.context_phase_params = Some(params);
        self
    }

    pub fn encoder_features(mut self, features: &'a candle_core::Tensor) -> Self {
        self.encoder_input_features = Some(features);
        self
    }

    pub fn encoder_output_length(mut self, length: i32) -> Self {
        self.encoder_output_length = Some(length);
        self
    }

    pub fn cross_attention_mask(mut self, mask: &'a candle_core::Tensor) -> Self {
        self.cross_attention_mask = Some(mask);
        self
    }

    pub fn eagle(mut self, config: EagleConfig<'a>) -> Self {
        self.eagle_config = Some(config);
        self
    }

    pub fn skip_cross_attention_blocks(mut self, blocks: &'a candle_core::Tensor) -> Self {
        self.skip_cross_attn_blocks = Some(blocks);
        self
    }

    pub fn guided_decoding(mut self, params: GuidedDecodingParams) -> Self {
        self.guided_decoding_params = Some(params);
        self
    }

    pub fn language_adapter_uid(mut self, uid: i32) -> Self {
        self.language_adapter_uid = Some(uid);
        self
    }

    pub fn time_budget_ms(mut self, allotted_time_ms: u64) -> Self {
        self.allotted_time_ms = Some(allotted_time_ms);
        self
    }

    pub fn cache_salt_id(mut self, cache_salt_id: u64) -> Self {
        self.cache_salt_id = Some(cache_salt_id);
        self
    }

    pub fn disaggregated_request_id(mut self, disagg_request_id: u64) -> Self {
        self.disagg_request_id = Some(disagg_request_id);
        self
    }

    #[cfg(feature = "cuda")]
    pub fn additional_outputs(mut self, outputs: OutputTensors, stream: CudaStream<'a>) -> Self {
        self.additional_output_sink = Some(AdditionalOutputSink { outputs, stream });
        self
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn infer_output_tensor_name(&mut self, name: &str) {
        match name {
            "context_logits" => self.return_context_logits = true,
            "generation_logits" => self.return_generation_logits = true,
            "encoder_output" => self.return_encoder_output = true,
            _ => self.push_additional_output(name),
        }
    }

    #[cfg(feature = "cuda")]
    fn push_additional_output(&mut self, name: &str) {
        if self
            .additional_outputs
            .iter()
            .any(|existing| existing.as_bytes() == name.as_bytes())
        {
            return;
        }
        self.additional_outputs
            .push(cstring(name, "additional output name"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_to_context_and_generation() {
        let ids = [1, 2, 3];
        let request = Request::new(&ids);
        assert_eq!(request.input_ids, ids);
        assert_eq!(request.max_new_tokens, 1);
        assert_eq!(request.request_type, RequestType::ContextAndGeneration);
        assert!(request.additional_outputs.is_empty());
    }

    #[test]
    fn sampling_config_sets_only_requested_fields() {
        let sampling = SamplingConfig::default()
            .beam_width(2)
            .top_k(4)
            .top_p(0.95)
            .temperature(0.7)
            .num_return_sequences(2)
            .beam_width_array([1, 2]);
        let sys = sampling.as_sys();
        assert_eq!(sys.beam_width, 2);
        assert_eq!(sys.has_top_k, 1);
        assert_eq!(sys.top_k, 4);
        assert_eq!(sys.has_top_p, 1);
        assert_eq!(sys.top_p, 0.95);
        assert_eq!(sys.has_temperature, 1);
        assert_eq!(sys.temperature, 0.7);
        assert_eq!(sys.has_num_return_sequences, 1);
        assert_eq!(sys.num_return_sequences, 2);
        assert_eq!(sys.beam_width_array_len, 2);
    }

    #[test]
    fn sampling_config_rejects_invalid_values() {
        assert!(SamplingConfig::default().beam_width(0).validate().is_err());
        assert!(
            SamplingConfig::default()
                .temperature(f32::NAN)
                .validate()
                .is_err()
        );
        assert!(
            SamplingConfig::default()
                .num_return_sequences(0)
                .validate()
                .is_err()
        );
        assert!(
            SamplingConfig::default()
                .beam_width_array([1, 0])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn request_keeps_standard_executor_extensions_in_one_place() {
        use candle_core::{Device, Tensor};

        let ids = [1, 2, 3];
        let positions = [0, 1, 2];
        let prompt_extra_ids = [0_u64, 1, 0];
        let bad = [13, 14];
        let stop = [2];
        let encoder_tokens = [8, 9];
        let first_gen = [5];
        let draft = [6, 7];
        let eagle = [1, 2, 3];
        let mm_hash = [101, 202];
        let mm_positions = [1];
        let mm_lengths = [2];
        let serialized_state = [9_u8, 8, 7];
        let prompt_embeddings = Tensor::from_vec(vec![0f32; 12], (1, 3, 4), &Device::Cpu).unwrap();
        let logits = Tensor::from_vec(vec![0f32; 6], (2, 3), &Device::Cpu).unwrap();
        let embedding_bias = Tensor::from_vec(vec![0f32; 4], 4, &Device::Cpu).unwrap();
        let multimodal_embedding = Tensor::from_vec(vec![0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let rotary_cos_sin = Tensor::from_vec(vec![0f32; 8], (2, 4), &Device::Cpu).unwrap();
        let lora_weights = Tensor::from_vec(vec![0f32; 2], 2, &Device::Cpu).unwrap();
        let lora_config = Tensor::from_vec(vec![1i32, 2], 2, &Device::Cpu).unwrap();
        let encoder_features = Tensor::from_vec(vec![0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let cross_attention_mask = Tensor::from_vec(vec![1f32; 4], (1, 4), &Device::Cpu).unwrap();
        let skip_cross_attn_blocks = Tensor::from_vec(vec![0i32, 1], 2, &Device::Cpu).unwrap();
        let retention = KvCacheRetentionConfig::new([KvCacheTokenRangeRetentionConfig::new(0)
            .token_end(16)
            .priority(80)
            .duration_ms(100)])
        .decode_retention_priority(70)
        .decode_duration_ms(50)
        .transfer_mode(KvCacheTransferMode::Gds)
        .directory("/tmp/cache");
        let request = Request::new(&ids)
            .max_new_tokens(4)
            .end_id(2)
            .pad_id(0)
            .sampling(SamplingConfig::default().top_k(1))
            .prompt_embeddings(&prompt_embeddings)
            .prompt_embedding_token_extra_ids(&prompt_extra_ids)
            .position_ids(&positions)
            .bad_words([bad.as_slice()])
            .stop_words([stop.as_slice()])
            .embedding_bias(&embedding_bias)
            .external_draft_tokens(
                ExternalDraftTokensConfig::new(&draft)
                    .logits(&logits)
                    .acceptance_threshold(0.8)
                    .fast_logits(true),
            )
            .multimodal_input(
                MultimodalInput::new([mm_hash.as_slice()], &mm_positions, &mm_lengths)
                    .uuids([Some("image-0")]),
            )
            .multimodal_embedding(&multimodal_embedding)
            .mrope(MropeConfig::new(&rotary_cos_sin, 3))
            .lora(
                LoraConfig::new(99)
                    .weights(&lora_weights)
                    .config(&lora_config),
            )
            .lookahead(LookaheadConfig::new(4, 3, 4))
            .kv_cache_retention(retention)
            .logits_processor("processor")
            .encoder_input_token_ids(&encoder_tokens)
            .return_log_probs()
            .return_context_logits()
            .return_generation_logits()
            .exclude_input_from_output()
            .return_encoder_output()
            .return_all_generated_tokens()
            .client_id(7)
            .priority(0.9)
            .request_type(RequestType::ContextOnly)
            .context_phase(
                ContextPhaseParams::new(&first_gen, 42)
                    .serialized_state(&serialized_state)
                    .draft_tokens(&draft)
                    .context_data_parallel_rank(2)
                    .disaggregated_info_endpoint("localhost:1234"),
            )
            .encoder_features(&encoder_features)
            .encoder_output_length(2)
            .cross_attention_mask(&cross_attention_mask)
            .eagle(
                EagleConfig::new()
                    .choices([eagle.as_slice()])
                    .greedy(false)
                    .posterior_threshold(0.1)
                    .dynamic_tree(true)
                    .max_dynamic_tree_top_k(4),
            )
            .skip_cross_attention_blocks(&skip_cross_attn_blocks)
            .guided_decoding(GuidedDecodingParams::new(GuidedDecodingType::Json).guide("{}"))
            .language_adapter_uid(3)
            .time_budget_ms(1000)
            .cache_salt_id(11)
            .disaggregated_request_id(12);

        assert_eq!(request.max_new_tokens, 4);
        assert_eq!(request.end_id, Some(2));
        assert_eq!(request.pad_id, Some(0));
        assert!(request.prompt_embeddings.is_some());
        assert_eq!(
            request.prompt_embedding_token_extra_ids,
            Some(prompt_extra_ids.as_slice())
        );
        assert_eq!(request.position_ids, Some(positions.as_slice()));
        assert_eq!(request.bad_words, vec![bad.as_slice()]);
        assert_eq!(request.stop_words, vec![stop.as_slice()]);
        assert!(request.embedding_bias.is_some());
        let draft_config = request.external_draft_tokens_config.as_ref().unwrap();
        assert!(draft_config.logits.is_some());
        assert_eq!(draft_config.fast_logits, Some(true));
        let multimodal_input = request.multimodal_input.as_ref().unwrap();
        assert_eq!(multimodal_input.hashes, vec![mm_hash.as_slice()]);
        assert_eq!(
            multimodal_input.uuids[0]
                .as_ref()
                .unwrap()
                .to_str()
                .unwrap(),
            "image-0"
        );
        assert!(request.multimodal_embedding.is_some());
        assert!(request.mrope_config.is_some());
        let lora = request.lora_config.as_ref().unwrap();
        assert_eq!(lora.task_id, 99);
        assert!(lora.weights.is_some());
        assert!(lora.config.is_some());
        assert!(request.lookahead_config.is_some());
        assert!(request.kv_cache_retention_config.is_some());
        assert_eq!(
            request.encoder_input_token_ids,
            Some(encoder_tokens.as_slice())
        );
        assert!(request.return_log_probs);
        assert!(request.return_context_logits);
        assert!(request.return_generation_logits);
        assert!(request.exclude_input_from_output);
        assert!(request.return_encoder_output);
        assert!(request.return_all_generated_tokens);
        assert_eq!(request.client_id, Some(7));
        assert_eq!(request.priority, Some(0.9));
        assert_eq!(request.request_type, RequestType::ContextOnly);
        let context_phase = request.context_phase_params.as_ref().unwrap();
        assert_eq!(
            context_phase.serialized_state,
            Some(serialized_state.as_slice())
        );
        assert_eq!(context_phase.ctx_dp_rank, Some(2));
        assert_eq!(
            context_phase
                .disagg_info_endpoint
                .as_ref()
                .unwrap()
                .to_str()
                .unwrap(),
            "localhost:1234"
        );
        assert!(request.encoder_input_features.is_some());
        assert_eq!(request.encoder_output_length, Some(2));
        assert!(request.cross_attention_mask.is_some());
        let eagle = request.eagle_config.as_ref().unwrap();
        assert!(!eagle.greedy_sampling);
        assert!(eagle.use_dynamic_tree);
        assert_eq!(eagle.dynamic_tree_max_top_k, Some(4));
        assert!(request.skip_cross_attn_blocks.is_some());
        let guided = request.guided_decoding_params.as_ref().unwrap();
        assert_eq!(guided.guide.as_ref().unwrap().to_str().unwrap(), "{}");
        assert_eq!(request.language_adapter_uid, Some(3));
        assert_eq!(request.allotted_time_ms, Some(1000));
        assert_eq!(request.cache_salt_id, Some(11));
        assert_eq!(request.disagg_request_id, Some(12));
    }

    #[test]
    fn enum_mappings_match_executor_ffi_contract() {
        assert_eq!(RequestType::ContextAndGeneration.as_i32(), 0);
        assert_eq!(RequestType::ContextOnly.as_i32(), 1);
        assert_eq!(RequestType::GenerationOnly.as_i32(), 2);

        assert_eq!(KvCacheTransferMode::Dram.as_i32(), 0);
        assert_eq!(KvCacheTransferMode::Gds.as_i32(), 1);
        assert_eq!(KvCacheTransferMode::PosixDebugFallback.as_i32(), 2);

        assert_eq!(GuidedDecodingType::Json.as_i32(), 0);
        assert_eq!(GuidedDecodingType::JsonSchema.as_i32(), 1);
        assert_eq!(GuidedDecodingType::Regex.as_i32(), 2);
        assert_eq!(GuidedDecodingType::EbnfGrammar.as_i32(), 3);
        assert_eq!(GuidedDecodingType::StructuralTag.as_i32(), 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn additional_outputs_infer_executor_outputs() {
        let ids = [1, 2, 3];
        let mut request = Request::new(&ids);
        request.infer_output_tensor_name("marked_model_output");
        request.infer_output_tensor_name("context_logits");
        request.infer_output_tensor_name("generation_logits");
        request.infer_output_tensor_name("encoder_output");
        request.infer_output_tensor_name("marked_model_output");

        assert_eq!(request.additional_outputs.len(), 1);
        assert_eq!(
            request.additional_outputs[0].to_str().unwrap(),
            "marked_model_output"
        );
        assert!(request.return_context_logits);
        assert!(request.return_generation_logits);
        assert!(request.return_encoder_output);
    }
}
