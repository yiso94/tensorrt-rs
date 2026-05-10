use candle_core::{Device, Tensor};
#[cfg(feature = "cuda")]
use tensorrt::{CudaStream, OutputTensors};
use tensorrt_llm::{
    BatchingType, CacheTransceiverBackend, CacheTransceiverConfig, CapacitySchedulerPolicy,
    ContextChunkingPolicy, ContextPhaseParams, DecodingConfig, DecodingMode, DynamicBatchConfig,
    EagleConfig, Executor, ExternalDraftTokensConfig, GuidedDecodingBackend, GuidedDecodingConfig,
    GuidedDecodingParams, GuidedDecodingType, KvCacheConfig, KvCacheRetentionConfig,
    KvCacheTokenRangeRetentionConfig, KvCacheTransferMode, LookaheadConfig, LoraConfig,
    MropeConfig, MultimodalInput, OrchestratorConfig, ParallelConfig, PeftCacheConfig, Request,
    RequestType, RuntimeDefaults, RuntimePerformanceConfig, SamplingConfig, SchedulerConfig,
    SpeculativeDecodingConfig,
};

#[test]
fn executor_builder_and_request_are_public_api() {
    let ids = [1, 2, 3];
    let sampling = SamplingConfig::default().top_k(1).temperature(0.7);
    let request = Request::new(&ids)
        .max_new_tokens(8)
        .sampling(sampling)
        .return_log_probs();

    let _builder = Executor::builder("engine")
        .max_beam_width(1)
        .batching(BatchingType::Inflight)
        .scheduler(SchedulerConfig::new())
        .kv_cache(KvCacheConfig::new());

    let _ = request;
}

#[test]
fn executor_builder_extensions_are_public_api() {
    let scheduler = SchedulerConfig::new()
        .capacity_policy(CapacitySchedulerPolicy::GuaranteedNoEvict)
        .context_chunking_policy(ContextChunkingPolicy::FirstComeFirstServed)
        .dynamic_batch(
            DynamicBatchConfig::new()
                .tune_batch_size(true)
                .tune_max_num_tokens(true)
                .moving_average_window(16)
                .batch_size_table([(1, 4), (2, 8)]),
        );
    let kv_cache = KvCacheConfig::new()
        .block_reuse(true)
        .max_tokens(1024)
        .max_attention_window([1024])
        .sink_token_length(4)
        .free_gpu_memory_fraction(0.8)
        .host_cache_size(1 << 20)
        .cross_kv_cache_fraction(0.25)
        .secondary_offload_min_priority(10)
        .event_buffer_max_size(8)
        .partial_block_reuse(true)
        .copy_on_partial_reuse(true)
        .unified_memory(false)
        .attention_dp_events_gather_period_ms(5)
        .runtime_defaults(RuntimeDefaults::new().max_attention_window([1024]))
        .max_gpu_total_bytes(1 << 30);
    let parallel = ParallelConfig::new()
        .device_ids([0])
        .participant_ids([0])
        .orchestrator(
            OrchestratorConfig::new()
                .enabled(true)
                .spawn_processes(false),
        )
        .num_nodes(1);
    let decoding = DecodingConfig::new()
        .mode(DecodingMode::TopKTopP)
        .lookahead(LookaheadConfig::new(1, 1, 1))
        .medusa_choices([[0, 1]])
        .eagle_choices([[0, 1]])
        .eagle_greedy(true)
        .eagle_posterior_threshold(0.1)
        .eagle_dynamic_tree(false)
        .max_eagle_dynamic_tree_top_k(4);

    let _builder = Executor::builder("engine")
        .max_beam_width(1)
        .scheduler(scheduler)
        .kv_cache(kv_cache)
        .chunked_context(true)
        .normalize_log_probs(true)
        .batching(BatchingType::Inflight)
        .max_batch_size(1)
        .max_num_tokens(1024)
        .parallel(parallel)
        .peft_cache(PeftCacheConfig::new().copy_streams(1))
        .decoding(decoding)
        .direct_storage(false)
        .gpu_weights_fraction(1.0)
        .max_queue_size(16)
        .runtime_performance(
            RuntimePerformanceConfig::new()
                .multi_block(true)
                .context_fmha_fp32_accumulation(false)
                .cuda_graphs(true)
                .cuda_graph_cache_size(4),
        )
        .receive_poll_period_ms(1)
        .max_sequence_idle_microseconds(1_000)
        .speculative_decoding(SpeculativeDecodingConfig::new().fast_logits(true))
        .guided_decoding(
            GuidedDecodingConfig::new(GuidedDecodingBackend::XGrammar)
                .encoded_vocab(["a", "b"])
                .tokenizer("{}")
                .stop_token_ids([2]),
        )
        .cache_transceiver(
            CacheTransceiverConfig::new()
                .backend(CacheTransceiverBackend::Default)
                .max_tokens_in_buffer(128)
                .transfer_timeout_ms(100)
                .sender_future_timeout_ms(100),
        )
        .gather_generation_logits(true)
        .offload_prompt_table(true)
        .tensorrt_overlap(true)
        .fail_fast_on_attention_window_too_large(true);
}

#[test]
fn request_extensions_are_public_api() {
    let device = Device::Cpu;
    let input_ids = [10, 11];
    let position_ids = [0, 1];
    let prompt_extra_ids = [0_u64, 1];
    let bad = [4, 5];
    let stop = [2];
    let multimodal_hash = [1, 2];
    let encoder_tokens = [1, 2];
    let first_gen_tokens = [1];
    let draft_tokens = [2];
    let eagle_choice = [0, 1];
    let prompt_embeddings = Tensor::from_vec(vec![0f32; 8], (2, 4), &device).unwrap();
    let embedding_bias = Tensor::from_vec(vec![0f32; 10], 10, &device).unwrap();
    let logits = Tensor::from_vec(vec![0f32; 2], 2, &device).unwrap();
    let multimodal_embedding = Tensor::from_vec(vec![0f32; 4], (1, 4), &device).unwrap();
    let rotary = Tensor::from_vec(vec![0f32; 4], (1, 4), &device).unwrap();
    let lora_weights = Tensor::from_vec(vec![0f32; 4], 4, &device).unwrap();
    let lora_config = Tensor::from_vec(vec![1i32, 2], 2, &device).unwrap();
    let encoder_features = Tensor::from_vec(vec![0f32; 4], (1, 4), &device).unwrap();
    let cross_attention_mask = Tensor::from_vec(vec![1f32; 2], 2, &device).unwrap();
    let skip_blocks = Tensor::from_vec(vec![0i32, 1], 2, &device).unwrap();

    let _request = Request::new(&input_ids)
        .max_new_tokens(8)
        .end_id(2)
        .pad_id(0)
        .sampling(SamplingConfig::default().top_k(1))
        .prompt_embeddings(&prompt_embeddings)
        .prompt_embedding_token_extra_ids(&prompt_extra_ids)
        .position_ids(&position_ids)
        .bad_words([bad.as_slice()])
        .stop_words([stop.as_slice()])
        .embedding_bias(&embedding_bias)
        .external_draft_tokens(
            ExternalDraftTokensConfig::new(&[1, 2])
                .logits(&logits)
                .acceptance_threshold(0.5)
                .fast_logits(true),
        )
        .multimodal_input(
            MultimodalInput::new([multimodal_hash.as_slice()], &[0], &[2]).uuids([Some("image-0")]),
        )
        .multimodal_embedding(&multimodal_embedding)
        .mrope(MropeConfig::new(&rotary, 1))
        .lora(
            LoraConfig::new(7)
                .weights(&lora_weights)
                .config(&lora_config),
        )
        .lookahead(LookaheadConfig::new(1, 1, 1))
        .kv_cache_retention(
            KvCacheRetentionConfig::new([KvCacheTokenRangeRetentionConfig::new(0)
                .token_end(2)
                .priority(50)
                .duration_ms(100)])
            .decode_retention_priority(40)
            .decode_duration_ms(100)
            .transfer_mode(KvCacheTransferMode::Dram),
        )
        .logits_processor("processor")
        .encoder_input_token_ids(&encoder_tokens)
        .return_log_probs()
        .return_context_logits()
        .return_generation_logits()
        .exclude_input_from_output()
        .return_encoder_output()
        .return_all_generated_tokens()
        .client_id(1)
        .priority(0.5)
        .request_type(RequestType::ContextAndGeneration)
        .context_phase(ContextPhaseParams::new(&first_gen_tokens, 42).draft_tokens(&draft_tokens))
        .encoder_features(&encoder_features)
        .encoder_output_length(1)
        .cross_attention_mask(&cross_attention_mask)
        .eagle(
            EagleConfig::new()
                .choices([eagle_choice.as_slice()])
                .greedy(true)
                .posterior_threshold(0.1)
                .dynamic_tree(false)
                .max_dynamic_tree_top_k(4),
        )
        .skip_cross_attention_blocks(&skip_blocks)
        .guided_decoding(GuidedDecodingParams::new(GuidedDecodingType::Json).guide("{}"))
        .language_adapter_uid(1)
        .time_budget_ms(1_000)
        .cache_salt_id(3)
        .disaggregated_request_id(4);
}

#[cfg(feature = "cuda")]
#[test]
fn additional_outputs_use_the_request_api() {
    let device = Device::Cpu;
    let input_ids = [10, 11];
    let output = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
    let outputs = OutputTensors::new([("marked_model_output", output)]);

    let _request = Request::new(&input_ids)
        .max_new_tokens(8)
        .additional_outputs(outputs, CudaStream::DEFAULT);
}
