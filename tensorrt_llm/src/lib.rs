#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]

//! Small TensorRT-LLM Executor bindings.
//!
//! The public API is intentionally narrow: [`Executor::run`] accepts a single
//! [`Request`] type for both normal generation and optional marked-output
//! extraction.

mod builder;
mod config;
mod error;
mod executor;
mod output;
mod request;
mod sampling;
mod string;

pub use builder::{
    BatchingType, CacheTransceiverBackend, CacheTransceiverConfig, CapacitySchedulerPolicy,
    CommunicationMode, CommunicationType, ContextChunkingPolicy, DecodingConfig, DecodingMode,
    DynamicBatchConfig, ExecutorBuilder, GuidedDecodingBackend, GuidedDecodingConfig,
    KvCacheConfig, OrchestratorConfig, ParallelConfig, PeftCacheConfig, RuntimeDefaults,
    RuntimePerformanceConfig, SchedulerConfig, SpeculativeDecodingConfig,
};
pub use error::{Error, Result};
pub use executor::{Executor, Response};
pub use output::{
    ExecutorOutput, FinishReason, OutputContextPhaseParams, SpeculativeDecodingFastLogitsInfo,
};
pub use request::{
    ContextPhaseParams, EagleConfig, ExternalDraftTokensConfig, GuidedDecodingParams,
    GuidedDecodingType, KvCacheRetentionConfig, KvCacheTokenRangeRetentionConfig,
    KvCacheTransferMode, LookaheadConfig, LoraConfig, MropeConfig, MultimodalInput, Request,
    RequestType,
};
pub use sampling::SamplingConfig;
