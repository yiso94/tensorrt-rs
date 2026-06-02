use crate::builder::ExecutorBuilder;
use crate::config::Config;
use crate::output::{ExecutorOutput, RawOutputGuard};
#[cfg(feature = "cuda")]
use crate::request::AdditionalOutputSink;
use crate::request::Request;
use crate::{Error, Result};
#[cfg(feature = "cuda")]
use candle_core::{CudaStorage, DeviceLocation, Storage};
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtr;
use half::{bf16, f16};
use smallvec::SmallVec;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tensorrt::DataType;
use tensorrt::TensorIOMode;
#[cfg(feature = "cuda")]
use tensorrt::{CudaEventHandle, CudaStream, OutputTensors};
use tensorrt_llm_sys as sys;

const TENSOR_MEMORY_CPU: c_int = 0;
#[cfg(feature = "cuda")]
const TENSOR_MEMORY_GPU: c_int = 1;
const INLINE_NAME_LIST: usize = 8;
const INLINE_WORD_LIST_TOKENS: usize = 32;
const INLINE_WORD_LIST_OFFSETS: usize = 8;
const INLINE_KV_RETENTION_RANGES: usize = 8;
const INLINE_TOKEN_EXTRA_IDS: usize = 16;
#[cfg(feature = "cuda")]
const INLINE_OUTPUT_SINKS: usize = 8;
#[cfg(feature = "cuda")]
const INLINE_OUTPUT_SINK_SHAPE_VALUES: usize = INLINE_OUTPUT_SINKS * 8;

#[derive(Clone, Copy, Debug)]
struct ValidationProfile {
    dtype: DType,
    executor_data_type: c_int,
    vocab_size: usize,
    hidden_size: usize,
    max_num_tokens: usize,
    max_prompt_embedding_table_size: usize,
}

impl ValidationProfile {
    fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            dtype: dtype_for_data_type(config.data_type())?,
            executor_data_type: data_type_to_executor_i32(config.data_type())?,
            vocab_size: config.vocab_size(),
            hidden_size: config.hidden_size(),
            max_num_tokens: config.max_num_tokens(),
            max_prompt_embedding_table_size: config.max_prompt_embedding_table_size(),
        })
    }
}

pub struct Executor {
    inner: NonNull<sys::RawExecutor>,
    profile: ValidationProfile,
    request_buffer_pool: Arc<RequestBufferPool>,
    #[cfg(feature = "cuda")]
    event_pool: Arc<CudaEventPool>,
    #[cfg(feature = "cuda")]
    sink_scratch_pool: Arc<OutputSinkScratchPool>,
}

unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

#[derive(Debug)]
pub struct Response {
    request_id: u64,
    client_id: Option<u64>,
    output: ExecutorOutput,
    #[cfg(feature = "cuda")]
    completion_event: Option<Arc<PooledCudaEvent>>,
    _raw_output: Option<RawOutputGuard>,
}

impl Response {
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    pub fn client_id(&self) -> Option<u64> {
        self.client_id
    }

    pub fn outputs(&self) -> &ExecutorOutput {
        &self.output
    }

    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        if let Some(event) = self.completion_event.as_ref() {
            event.synchronize().map_err(Error::from)?;
        }
        Ok(())
    }

    fn from_sys(
        raw: &sys::Response,
        #[cfg(feature = "cuda")] completion_event: Option<Arc<PooledCudaEvent>>,
        raw_output: Option<RawOutputGuard>,
    ) -> Result<Self> {
        Ok(Self {
            request_id: raw.request_id,
            client_id: (raw.has_client_id != 0).then_some(raw.client_id),
            output: crate::output::from_sys(&raw.output, raw_output.clone())?,
            #[cfg(feature = "cuda")]
            completion_event,
            _raw_output: raw_output,
        })
    }
}

#[derive(Debug)]
struct RawResponses {
    raw: NonNull<sys::Response>,
    count: usize,
    #[cfg(feature = "cuda")]
    completion_event: Option<Arc<PooledCudaEvent>>,
}

unsafe impl Send for RawResponses {}
unsafe impl Sync for RawResponses {}

impl Drop for RawResponses {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(event) = self.completion_event.as_ref() {
            let _ = event.synchronize();
        }
        unsafe { sys::tensorrt_llm_rs_responses_free(self.raw.as_ptr(), self.count) };
    }
}

#[derive(Default)]
struct RequestBufferPool {
    // Prepared requests hand raw pointers to TensorRT-LLM. Keeping buffers boxed
    // lets the pool reuse allocation while the pointee address stays stable even
    // if the PreparedRequest itself is moved.
    #[allow(clippy::vec_box)]
    buffers: Mutex<Vec<Box<RequestBuffers>>>,
}

impl RequestBufferPool {
    fn checkout(self: &Arc<Self>) -> Result<PooledRequestBuffers> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| Error::Executor("TensorRT-LLM request buffer pool is poisoned".into()))?
            .pop()
            .unwrap_or_default();
        Ok(PooledRequestBuffers {
            buffers: Some(buffers),
            pool: Arc::clone(self),
        })
    }

    fn recycle(&self, mut buffers: Box<RequestBuffers>) {
        buffers.clear();
        if let Ok(mut pool) = self.buffers.lock() {
            pool.push(buffers);
        }
    }
}

struct PooledRequestBuffers {
    buffers: Option<Box<RequestBuffers>>,
    pool: Arc<RequestBufferPool>,
}

impl std::ops::Deref for PooledRequestBuffers {
    type Target = RequestBuffers;

    fn deref(&self) -> &Self::Target {
        self.buffers
            .as_ref()
            .expect("request buffers are only absent during drop")
    }
}

impl std::ops::DerefMut for PooledRequestBuffers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffers
            .as_mut()
            .expect("request buffers are only absent during drop")
    }
}

impl Drop for PooledRequestBuffers {
    fn drop(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            self.pool.recycle(buffers);
        }
    }
}

#[derive(Default)]
struct RequestBuffers {
    bad_words: WordListBuffer,
    stop_words: WordListBuffer,
    multimodal_hashes: WordListBuffer,
    multimodal_uuids: NullableStringListBuffer,
    eagle_choices: WordListBuffer,
    kv_cache_retention_ranges:
        SmallVec<[sys::KvCacheTokenRangeRetentionConfig; INLINE_KV_RETENTION_RANGES]>,
    additional_outputs: StringListBuffer,
}

// RequestBuffers is shared only through RequestBufferPool's mutex. Pointer
// lists are rebuilt for each PreparedRequest and cleared before recycling.
unsafe impl Send for RequestBuffers {}

impl RequestBuffers {
    fn clear(&mut self) {
        self.bad_words.clear();
        self.stop_words.clear();
        self.multimodal_hashes.clear();
        self.multimodal_uuids.clear();
        self.eagle_choices.clear();
        self.kv_cache_retention_ranges.clear();
        self.additional_outputs.clear();
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
struct CudaEventPool {
    events: Mutex<HashMap<DeviceLocation, Vec<CudaEventHandle>>>,
}

#[cfg(feature = "cuda")]
impl CudaEventPool {
    fn checkout(
        self: &Arc<Self>,
        device_location: DeviceLocation,
        stream: CudaStream<'_>,
    ) -> Result<Arc<PooledCudaEvent>> {
        let event = {
            let mut events = self
                .events
                .lock()
                .map_err(|_| Error::Executor("TensorRT-LLM CUDA event pool is poisoned".into()))?;
            events
                .get_mut(&device_location)
                .and_then(|events| events.pop())
        }
        .map(Ok)
        .unwrap_or_else(|| CudaEventHandle::new_for_stream(stream))?;

        Ok(Arc::new(PooledCudaEvent {
            event: Some(event),
            synchronized: AtomicBool::new(false),
            device_location,
            pool: Arc::clone(self),
        }))
    }

    fn recycle(&self, device_location: DeviceLocation, event: CudaEventHandle) {
        if let Ok(mut events) = self.events.lock() {
            events.entry(device_location).or_default().push(event);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct PooledCudaEvent {
    event: Option<CudaEventHandle>,
    synchronized: AtomicBool,
    device_location: DeviceLocation,
    pool: Arc<CudaEventPool>,
}

#[cfg(feature = "cuda")]
impl PooledCudaEvent {
    fn synchronize(&self) -> tensorrt::Result<()> {
        if self.synchronized.load(Ordering::Acquire) {
            return Ok(());
        }
        if let Some(event) = self.event.as_ref() {
            event.synchronize()?;
            self.synchronized.store(true, Ordering::Release);
            Ok(())
        } else {
            Ok(())
        }
    }

    fn as_raw(&self) -> *mut c_void {
        self.event
            .as_ref()
            .expect("pooled CUDA event is only absent during drop")
            .as_event()
            .as_raw()
    }
}

#[cfg(feature = "cuda")]
impl Drop for PooledCudaEvent {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            if !self.synchronized.load(Ordering::Acquire) {
                let _ = event.synchronize();
            }
            self.pool.recycle(self.device_location, event);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct OutputSinkScratchPool {
    scratch: Mutex<Vec<OutputSinkScratch>>,
}

#[cfg(feature = "cuda")]
impl OutputSinkScratchPool {
    fn checkout(self: &Arc<Self>) -> Result<OutputSinkScratch> {
        let mut scratch = self.scratch.lock().map_err(|_| {
            Error::Executor("TensorRT-LLM output sink scratch pool is poisoned".into())
        })?;
        Ok(scratch.pop().unwrap_or_default())
    }

    fn recycle(&self, mut scratch: OutputSinkScratch) {
        scratch.clear();
        if let Ok(mut pool) = self.scratch.lock() {
            pool.push(scratch);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct OutputSinkScratch {
    names: SmallVec<[CString; INLINE_OUTPUT_SINKS]>,
    shape_values: SmallVec<[i64; INLINE_OUTPUT_SINK_SHAPE_VALUES]>,
    shape_ranges: SmallVec<[(usize, usize); INLINE_OUTPUT_SINKS]>,
    written: SmallVec<[c_int; INLINE_OUTPUT_SINKS]>,
    views: SmallVec<[sys::NamedTensorView; INLINE_OUTPUT_SINKS]>,
}

// OutputSinkScratch is only shared through OutputSinkScratchPool's mutex, and
// the pool clears pointer-bearing views before recycling a scratch buffer.
#[cfg(feature = "cuda")]
unsafe impl Send for OutputSinkScratch {}

#[cfg(feature = "cuda")]
impl OutputSinkScratch {
    fn clear(&mut self) {
        self.names.clear();
        self.shape_values.clear();
        self.shape_ranges.clear();
        self.written.clear();
        self.views.clear();
    }

    fn reset_written(&mut self) {
        for written in &mut self.written {
            *written = 0;
        }
    }

    fn layout_matches(&self, outputs: &OutputTensors) -> bool {
        if self.views.len() != outputs.iter().count() {
            return false;
        }
        outputs.iter().enumerate().all(|(index, (name, tensor))| {
            self.names[index].as_c_str().to_bytes() == name.as_bytes()
                && self.views[index].tensor.data_type
                    == data_type_for_dtype(tensor.dtype())
                        .and_then(data_type_to_executor_i32)
                        .unwrap_or(-1)
                && self.shape_slice(index).is_some_and(|shape| {
                    shape.len() == tensor.dims().len()
                        && shape
                            .iter()
                            .zip(tensor.dims())
                            .all(|(left, right)| i64::try_from(*right) == Ok(*left))
                })
        })
    }

    fn shape_slice(&self, index: usize) -> Option<&[i64]> {
        let (start, len) = *self.shape_ranges.get(index)?;
        self.shape_values.get(start..start + len)
    }

    fn rebuild_layout(&mut self, outputs: &OutputTensors) -> Result<()> {
        self.clear();
        let output_count = outputs.iter().count();
        self.names.reserve(output_count);
        self.shape_ranges.reserve(output_count);
        self.written.resize(output_count, 0);
        self.views.reserve(output_count);

        for (name, tensor) in outputs.iter() {
            if name.is_empty() {
                return Err(Error::InvalidArgument(
                    "Request::additional_outputs output names cannot be empty".into(),
                ));
            }
            let data_type = data_type_for_dtype(tensor.dtype())?;
            let shape_start = self.shape_values.len();
            for dim in tensor.dims() {
                self.shape_values.push(i64::try_from(*dim).map_err(|_| {
                    Error::InvalidArgument(format!(
                        "output tensor {name:?} dimension {dim} cannot fit in i64"
                    ))
                })?);
            }
            let shape_len = tensor.dims().len();
            self.names.push(CString::new(name)?);
            self.shape_ranges.push((shape_start, shape_len));
            self.views.push(sys::NamedTensorView {
                name: std::ptr::null(),
                tensor: sys::TensorView {
                    data: std::ptr::null(),
                    byte_len: 0,
                    data_type: data_type_to_executor_i32(data_type)?,
                    shape: std::ptr::null(),
                    rank: shape_len,
                    memory_type: TENSOR_MEMORY_GPU,
                },
                written: std::ptr::null_mut(),
            });
        }

        self.refresh_stable_pointers();
        Ok(())
    }

    fn refresh_stable_pointers(&mut self) {
        let shape_base = self.shape_values.as_ptr();
        for index in 0..self.views.len() {
            let (shape_start, shape_len) = self.shape_ranges[index];
            self.views[index].name = self.names[index].as_ptr();
            self.views[index].tensor.shape = if shape_len == 0 {
                std::ptr::null()
            } else {
                unsafe { shape_base.add(shape_start) }
            };
            self.views[index].tensor.rank = shape_len;
            self.views[index].written = unsafe { self.written.as_mut_ptr().add(index) };
        }
    }
}

impl Executor {
    pub fn builder(engine_dir: impl AsRef<Path>) -> ExecutorBuilder {
        ExecutorBuilder::new(engine_dir)
    }

    pub(crate) fn from_builder(mut builder: ExecutorBuilder) -> Result<Self> {
        init_plugins()?;
        let config = Config::from_engine_dir(&builder.engine_dir)?;
        let profile = ValidationProfile::new(&config)?;
        if builder.max_batch_size.is_none() {
            builder.max_batch_size = Some(usize_to_i32(config.max_batch_size(), "max_batch_size")?);
        }
        if builder.max_num_tokens.is_none() {
            builder.max_num_tokens = Some(usize_to_i32(config.max_num_tokens(), "max_num_tokens")?);
        }
        if builder.additional_outputs.is_empty() {
            builder.additional_outputs = infer_additional_outputs(&builder.engine_dir)?;
        }
        builder.validate()?;
        let engine_dir = CString::new(builder.engine_dir.as_os_str().as_encoded_bytes())?;
        let parts = builder.sys_parts();

        let mut inner = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        let status = unsafe {
            sys::tensorrt_llm_rs_executor_create(
                engine_dir.as_ptr(),
                &parts.config,
                &mut inner,
                &mut error,
            )
        };
        if status != 0 {
            return Err(take_error(error));
        }
        let inner = NonNull::new(inner).ok_or_else(|| {
            Error::Executor("TensorRT-LLM Executor constructor returned null".to_owned())
        })?;
        Ok(Self {
            inner,
            profile,
            request_buffer_pool: Arc::new(RequestBufferPool::default()),
            #[cfg(feature = "cuda")]
            event_pool: Arc::new(CudaEventPool::default()),
            #[cfg(feature = "cuda")]
            sink_scratch_pool: Arc::new(OutputSinkScratchPool::default()),
        })
    }

    pub fn run(&self, request: Request<'_>) -> Result<Response> {
        #[cfg(feature = "cuda")]
        let mut request = request;

        #[cfg(feature = "cuda")]
        let additional_outputs = request.additional_output_sink.take();

        #[cfg(feature = "cuda")]
        if let Some(additional_outputs) = additional_outputs.as_ref() {
            let outputs = unsafe { additional_outputs.outputs_ref() };
            for (name, _) in outputs.iter() {
                request.infer_output_tensor_name(name);
            }
        }

        #[cfg(feature = "cuda")]
        let sink = additional_outputs
            .map(|additional_outputs| self.prepare_output_sink(additional_outputs))
            .transpose()?;

        let prepared = self.prepare_request(request)?;

        #[cfg(feature = "cuda")]
        if let Some(mut sink) = sink {
            let response = self.run_prepared_with_sink(&prepared, Some(&mut sink))?;
            sink.ensure_all_written()?;
            return Ok(response);
        }

        self.run_prepared_with_sink(&prepared, None)
    }

    #[cfg(feature = "cuda")]
    fn prepare_output_sink<'a>(
        &self,
        mut additional_outputs: AdditionalOutputSink<'a>,
    ) -> Result<OutputSinkParts> {
        let outputs = unsafe { additional_outputs.outputs_mut() };
        OutputSinkParts::new(
            outputs,
            additional_outputs.stream,
            Arc::clone(&self.event_pool),
            Arc::clone(&self.sink_scratch_pool),
        )
    }

    fn run_prepared_with_sink(
        &self,
        prepared: &PreparedRequest<'_>,
        sink: Option<&mut OutputSinkParts>,
    ) -> Result<Response> {
        let mut responses = std::ptr::null_mut();
        let mut response_count = 0;
        let mut error = std::ptr::null_mut();
        #[cfg(feature = "cuda")]
        let completion_event = sink.as_ref().map(|sink| Arc::clone(&sink.completion_event));
        let sink = sink.map(OutputSinkParts::as_sys);
        let status = match sink {
            Some(sink) => unsafe {
                sys::tensorrt_llm_rs_executor_run_into(
                    self.inner.as_ptr(),
                    &prepared.sys_request,
                    sink,
                    &mut responses,
                    &mut response_count,
                    &mut error,
                )
            },
            None => unsafe {
                sys::tensorrt_llm_rs_executor_run(
                    self.inner.as_ptr(),
                    &prepared.sys_request,
                    &mut responses,
                    &mut response_count,
                    &mut error,
                )
            },
        };
        if status != 0 {
            return Err(take_error(error));
        }
        final_response_from_sys(
            responses,
            response_count,
            #[cfg(feature = "cuda")]
            completion_event,
        )?
        .ok_or_else(|| Error::Executor("TensorRT-LLM Executor returned no final response".into()))
    }

    fn prepare_request<'a>(&self, request: Request<'a>) -> Result<PreparedRequest<'a>> {
        #[cfg(feature = "cuda")]
        if request.additional_output_sink.is_some() {
            return Err(Error::InvalidArgument(
                "preallocated additional outputs can only be used with Executor::run".into(),
            ));
        }
        self.validate_request(&request)?;
        let prompt = match request.prompt_embeddings {
            Some(prompt_embeddings) => Some(self.prompt_embedding(
                prompt_embeddings,
                request.input_ids,
                request.prompt_embedding_token_extra_ids,
            )?),
            None => None,
        };
        let embedding_bias = match request.embedding_bias {
            Some(embedding_bias) => Some(self.embedding_bias(embedding_bias)?),
            None => None,
        };
        let external_draft_logits = request
            .external_draft_tokens_config
            .as_ref()
            .and_then(|config| config.logits)
            .map(TensorBuffer::new)
            .transpose()?;
        let multimodal_embedding = request
            .multimodal_embedding
            .map(TensorBuffer::new)
            .transpose()?;
        let mrope_rotary_cos_sin = request
            .mrope_config
            .as_ref()
            .map(|config| TensorBuffer::new(config.rotary_cos_sin))
            .transpose()?;
        let lora_weights = request
            .lora_config
            .as_ref()
            .and_then(|config| config.weights)
            .map(TensorBuffer::new)
            .transpose()?;
        let lora_config_tensor = request
            .lora_config
            .as_ref()
            .and_then(|config| config.config)
            .map(TensorBuffer::new)
            .transpose()?;
        let encoder_input_features = request
            .encoder_input_features
            .map(TensorBuffer::new)
            .transpose()?;
        let cross_attention_mask = request
            .cross_attention_mask
            .map(TensorBuffer::new)
            .transpose()?;
        let skip_cross_attn_blocks = request
            .skip_cross_attn_blocks
            .map(TensorBuffer::new)
            .transpose()?;
        let tensors = Box::new(RequestTensors {
            prompt,
            embedding_bias,
            external_draft_logits,
            multimodal_embedding,
            mrope_rotary_cos_sin,
            lora_weights,
            lora_config_tensor,
            encoder_input_features,
            cross_attention_mask,
            skip_cross_attn_blocks,
        });

        let mut buffers = self.request_buffer_pool.checkout()?;
        buffers.bad_words.replace(&request.bad_words);
        buffers.stop_words.replace(&request.stop_words);
        if let Some(input) = request.multimodal_input.as_ref() {
            buffers.multimodal_hashes.replace(&input.hashes);
            buffers.multimodal_uuids.replace(&input.uuids);
        }
        if let Some(config) = request.eagle_config.as_ref() {
            buffers.eagle_choices.replace(&config.choices);
        }
        if let Some(config) = request.kv_cache_retention_config.as_ref() {
            buffers
                .kv_cache_retention_ranges
                .reserve(config.ranges.len());
            buffers
                .kv_cache_retention_ranges
                .extend(
                    config
                        .ranges
                        .iter()
                        .map(|range| sys::KvCacheTokenRangeRetentionConfig {
                            token_start: range.token_start,
                            has_token_end: i32::from(range.token_end.is_some()),
                            token_end: range.token_end.unwrap_or_default(),
                            priority: range.priority,
                            has_duration_ms: i32::from(range.duration_ms.is_some()),
                            duration_ms: range.duration_ms.unwrap_or_default(),
                        }),
                );
        }
        buffers
            .additional_outputs
            .replace(&request.additional_outputs);
        let (position_ids, position_ids_len) = match request.position_ids {
            Some(position_ids) => (position_ids.as_ptr(), position_ids.len()),
            None => (std::ptr::null(), 0),
        };
        let (encoder_input_token_ids, encoder_input_token_count) =
            match request.encoder_input_token_ids {
                Some(token_ids) => (token_ids.as_ptr(), token_ids.len()),
                None => (std::ptr::null(), 0),
            };

        let (
            prompt_data,
            prompt_byte_len,
            prompt_rows,
            prompt_hidden,
            prompt_dtype,
            prompt_memory_type,
            extra_ids,
        ) = match tensors.prompt.as_ref() {
            Some(prompt) => {
                let view = prompt.tensor.as_sys();
                (
                    view.data,
                    view.byte_len,
                    prompt.rows,
                    self.profile.hidden_size,
                    view.data_type,
                    view.memory_type,
                    prompt.input_token_extra_ids.as_slice(),
                )
            }
            None => (
                std::ptr::null(),
                0,
                0,
                0,
                self.profile.executor_data_type,
                TENSOR_MEMORY_CPU,
                &[][..],
            ),
        };
        let (
            embedding_bias_data,
            embedding_bias_byte_len,
            embedding_bias_len,
            embedding_bias_dtype,
            embedding_bias_memory_type,
        ) = match tensors.embedding_bias.as_ref() {
            Some(embedding_bias) => {
                let view = embedding_bias.tensor.as_sys();
                (
                    view.data,
                    view.byte_len,
                    embedding_bias.len,
                    view.data_type,
                    view.memory_type,
                )
            }
            None => (
                std::ptr::null(),
                0,
                0,
                self.profile.executor_data_type,
                TENSOR_MEMORY_CPU,
            ),
        };
        let external_draft_tokens_config =
            request.external_draft_tokens_config.as_ref().map(|config| {
                sys::ExternalDraftTokensConfig {
                    tokens: config.tokens.as_ptr(),
                    token_count: config.tokens.len(),
                    logits: optional_tensor_view(tensors.external_draft_logits.as_ref()),
                    has_acceptance_threshold: i32::from(config.acceptance_threshold.is_some()),
                    acceptance_threshold: config.acceptance_threshold.unwrap_or_default(),
                    has_fast_logits: i32::from(config.fast_logits.is_some()),
                    fast_logits: i32::from(config.fast_logits.unwrap_or_default()),
                }
            });
        let multimodal_input =
            request
                .multimodal_input
                .as_ref()
                .map(|input| sys::MultimodalInput {
                    hashes: buffers.multimodal_hashes.as_sys(),
                    positions: input.positions.as_ptr(),
                    position_count: input.positions.len(),
                    lengths: input.lengths.as_ptr(),
                    length_count: input.lengths.len(),
                    uuids: buffers.multimodal_uuids.as_ptr(),
                    uuid_count: buffers.multimodal_uuids.len(),
                });
        let mrope_config = request
            .mrope_config
            .as_ref()
            .map(|config| sys::MropeConfig {
                rotary_cos_sin: optional_tensor_view(tensors.mrope_rotary_cos_sin.as_ref()),
                position_deltas: config.position_deltas,
            });
        let lora_config = request.lora_config.as_ref().map(|config| sys::LoraConfig {
            task_id: config.task_id,
            weights: optional_tensor_view(tensors.lora_weights.as_ref()),
            config: optional_tensor_view(tensors.lora_config_tensor.as_ref()),
        });
        let lookahead_config =
            request
                .lookahead_config
                .as_ref()
                .map(|config| sys::LookaheadConfig {
                    window_size: config.window_size,
                    ngram_size: config.ngram_size,
                    verification_set_size: config.verification_set_size,
                });
        let kv_cache_retention_config =
            request
                .kv_cache_retention_config
                .as_ref()
                .map(|config| sys::KvCacheRetentionConfig {
                    ranges: buffers.kv_cache_retention_ranges.as_ptr(),
                    range_count: buffers.kv_cache_retention_ranges.len(),
                    decode_retention_priority: config.decode_retention_priority,
                    has_decode_duration_ms: i32::from(config.decode_duration_ms.is_some()),
                    decode_duration_ms: config.decode_duration_ms.unwrap_or_default(),
                    transfer_mode: config.transfer_mode.as_i32(),
                    directory: config
                        .directory
                        .as_ref()
                        .map_or(std::ptr::null(), |directory| directory.as_ptr()),
                });
        let context_phase_params =
            request
                .context_phase_params
                .as_ref()
                .map(|params| sys::ContextPhaseParams {
                    first_gen_tokens: params.first_gen_tokens.as_ptr(),
                    first_gen_token_count: params.first_gen_tokens.len(),
                    request_id: params.request_id,
                    serialized_state: params
                        .serialized_state
                        .map_or(std::ptr::null(), |state| state.as_ptr()),
                    serialized_state_len: params.serialized_state.map_or(0, <[u8]>::len),
                    draft_tokens: params
                        .draft_tokens
                        .map_or(std::ptr::null(), |tokens| tokens.as_ptr()),
                    draft_token_count: params.draft_tokens.map_or(0, <[i32]>::len),
                    has_ctx_dp_rank: i32::from(params.ctx_dp_rank.is_some()),
                    ctx_dp_rank: params.ctx_dp_rank.unwrap_or_default(),
                    disagg_info_endpoint: params
                        .disagg_info_endpoint
                        .as_ref()
                        .map_or(std::ptr::null(), |endpoint| endpoint.as_ptr()),
                });
        let eagle_config = request
            .eagle_config
            .as_ref()
            .map(|config| sys::EagleConfig {
                choices: buffers.eagle_choices.as_sys(),
                greedy_sampling: i32::from(config.greedy_sampling),
                has_posterior_threshold: i32::from(config.posterior_threshold.is_some()),
                posterior_threshold: config.posterior_threshold.unwrap_or_default(),
                use_dynamic_tree: i32::from(config.use_dynamic_tree),
                has_dynamic_tree_max_top_k: i32::from(config.dynamic_tree_max_top_k.is_some()),
                dynamic_tree_max_top_k: config.dynamic_tree_max_top_k.unwrap_or_default(),
            });
        let guided_decoding_params =
            request
                .guided_decoding_params
                .as_ref()
                .map(|params| sys::GuidedDecodingParams {
                    guide_type: params.guide_type.as_i32(),
                    guide: params
                        .guide
                        .as_ref()
                        .map_or(std::ptr::null(), |guide| guide.as_ptr()),
                });

        let sys_request = sys::Request {
            input_ids: request.input_ids.as_ptr(),
            input_len: request.input_ids.len(),
            max_new_tokens: request.max_new_tokens,
            streaming: 0,
            has_end_id: i32::from(request.end_id.is_some()),
            end_id: request.end_id.unwrap_or_default(),
            has_pad_id: i32::from(request.pad_id.is_some()),
            pad_id: request.pad_id.unwrap_or_default(),
            sampling: request.sampling.as_sys(),
            prompt_embedding_data: prompt_data,
            prompt_embedding_byte_len: prompt_byte_len,
            prompt_embedding_rows: prompt_rows,
            prompt_embedding_hidden_size: prompt_hidden,
            prompt_embedding_data_type: prompt_dtype,
            prompt_embedding_memory_type: prompt_memory_type,
            input_token_extra_ids: extra_ids.as_ptr(),
            input_token_extra_ids_len: extra_ids.len(),
            position_ids,
            position_ids_len,
            bad_words: buffers.bad_words.as_sys(),
            stop_words: buffers.stop_words.as_sys(),
            embedding_bias_data,
            embedding_bias_byte_len,
            embedding_bias_len,
            embedding_bias_data_type: embedding_bias_dtype,
            embedding_bias_memory_type,
            has_external_draft_tokens_config: i32::from(external_draft_tokens_config.is_some()),
            external_draft_tokens_config: external_draft_tokens_config.unwrap_or_default(),
            has_multimodal_input: i32::from(multimodal_input.is_some()),
            multimodal_input: multimodal_input.unwrap_or_default(),
            multimodal_embedding: optional_tensor_view(tensors.multimodal_embedding.as_ref()),
            has_mrope_config: i32::from(mrope_config.is_some()),
            mrope_config: mrope_config.unwrap_or_default(),
            has_lora_config: i32::from(lora_config.is_some()),
            lora_config: lora_config.unwrap_or_default(),
            has_lookahead_config: i32::from(lookahead_config.is_some()),
            lookahead_config: lookahead_config.unwrap_or_default(),
            has_kv_cache_retention_config: i32::from(kv_cache_retention_config.is_some()),
            kv_cache_retention_config: kv_cache_retention_config.unwrap_or_default(),
            logits_post_processor_name: request
                .logits_post_processor_name
                .as_ref()
                .map_or(std::ptr::null(), |name| name.as_ptr()),
            encoder_input_token_ids,
            encoder_input_token_count,
            additional_outputs: buffers.additional_outputs.as_sys(),
            return_log_probs: i32::from(request.return_log_probs),
            return_context_logits: i32::from(request.return_context_logits),
            return_generation_logits: i32::from(request.return_generation_logits),
            exclude_input_from_output: i32::from(request.exclude_input_from_output),
            return_encoder_output: i32::from(request.return_encoder_output),
            return_all_generated_tokens: i32::from(request.return_all_generated_tokens),
            has_client_id: i32::from(request.client_id.is_some()),
            client_id: request.client_id.unwrap_or_default(),
            has_priority: i32::from(request.priority.is_some()),
            priority: request.priority.unwrap_or_default(),
            request_type: request.request_type.as_i32(),
            has_context_phase_params: i32::from(context_phase_params.is_some()),
            context_phase_params: context_phase_params.unwrap_or_default(),
            encoder_input_features: optional_tensor_view(tensors.encoder_input_features.as_ref()),
            has_encoder_output_length: i32::from(request.encoder_output_length.is_some()),
            encoder_output_length: request.encoder_output_length.unwrap_or_default(),
            cross_attention_mask: optional_tensor_view(tensors.cross_attention_mask.as_ref()),
            has_eagle_config: i32::from(eagle_config.is_some()),
            eagle_config: eagle_config.unwrap_or_default(),
            skip_cross_attn_blocks: optional_tensor_view(tensors.skip_cross_attn_blocks.as_ref()),
            has_guided_decoding_params: i32::from(guided_decoding_params.is_some()),
            guided_decoding_params: guided_decoding_params.unwrap_or_default(),
            has_language_adapter_uid: i32::from(request.language_adapter_uid.is_some()),
            language_adapter_uid: request.language_adapter_uid.unwrap_or_default(),
            has_allotted_time_ms: i32::from(request.allotted_time_ms.is_some()),
            allotted_time_ms: request.allotted_time_ms.unwrap_or_default(),
            has_cache_salt_id: i32::from(request.cache_salt_id.is_some()),
            cache_salt_id: request.cache_salt_id.unwrap_or_default(),
            has_disagg_request_id: i32::from(request.disagg_request_id.is_some()),
            disagg_request_id: request.disagg_request_id.unwrap_or_default(),
        };

        Ok(PreparedRequest {
            _request: request,
            _tensors: tensors,
            _buffers: buffers,
            sys_request,
        })
    }

    fn validate_request(&self, request: &Request<'_>) -> Result<()> {
        request.sampling.validate()?;
        if request.input_ids.is_empty() {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM request must contain at least one input token".into(),
            ));
        }
        if request.max_new_tokens < 0 {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM request max_new_tokens cannot be negative".into(),
            ));
        }
        if let Some(position_ids) = request.position_ids
            && position_ids.len() != request.input_ids.len()
        {
            return Err(Error::InvalidArgument(format!(
                "position_ids length mismatch: expected {}, got {}",
                request.input_ids.len(),
                position_ids.len()
            )));
        }
        if let Some(priority) = request.priority
            && !priority.is_finite()
        {
            return Err(Error::InvalidArgument("priority must be finite".into()));
        }
        if let Some(token_extra_ids) = request.prompt_embedding_token_extra_ids
            && token_extra_ids.len() != request.input_ids.len()
        {
            return Err(Error::InvalidArgument(format!(
                "prompt_embedding_token_extra_ids length mismatch: expected {}, got {}",
                request.input_ids.len(),
                token_extra_ids.len()
            )));
        }
        if let Some(config) = request.external_draft_tokens_config.as_ref() {
            if config.tokens.is_empty() {
                return Err(Error::InvalidArgument(
                    "external_draft_tokens must contain at least one token".into(),
                ));
            }
            if let Some(acceptance_threshold) = config.acceptance_threshold
                && (!acceptance_threshold.is_finite()
                    || acceptance_threshold <= 0.0
                    || acceptance_threshold > 1.0)
            {
                return Err(Error::InvalidArgument(
                    "external_draft_tokens acceptance_threshold must be in (0, 1]".into(),
                ));
            }
        }
        if let Some(input) = request.multimodal_input.as_ref() {
            validate_word_list("multimodal_hashes", &input.hashes)?;
            if input.hashes.len() != input.positions.len()
                || input.hashes.len() != input.lengths.len()
            {
                return Err(Error::InvalidArgument(
                    "multimodal_input hashes, positions, and lengths must have the same length"
                        .into(),
                ));
            }
            if !input.uuids.is_empty() && input.uuids.len() != input.hashes.len() {
                return Err(Error::InvalidArgument(
                    "multimodal_input uuids length must match hashes length".into(),
                ));
            }
        }
        if let Some(config) = request.lookahead_config
            && (config.window_size <= 0
                || config.ngram_size <= 0
                || config.verification_set_size <= 0)
        {
            return Err(Error::InvalidArgument(
                "lookahead_config values must be positive".into(),
            ));
        }
        if let Some(config) = request.kv_cache_retention_config.as_ref() {
            validate_retention_priority(
                "kv_cache_retention_config decode_retention_priority",
                config.decode_retention_priority,
            )?;
            for range in &config.ranges {
                if range.token_start < 0 {
                    return Err(Error::InvalidArgument(
                        "kv_cache_retention_config token_start cannot be negative".into(),
                    ));
                }
                if let Some(token_end) = range.token_end
                    && token_end <= range.token_start
                {
                    return Err(Error::InvalidArgument(
                        "kv_cache_retention_config token_end must be greater than token_start"
                            .into(),
                    ));
                }
                validate_retention_priority("kv_cache_retention_config priority", range.priority)?;
            }
        }
        if let Some(params) = request.context_phase_params.as_ref()
            && params.first_gen_tokens.is_empty()
        {
            return Err(Error::InvalidArgument(
                "context_phase_params first_gen_tokens cannot be empty".into(),
            ));
        }
        if let Some(length) = request.encoder_output_length
            && length <= 0
        {
            return Err(Error::InvalidArgument(
                "encoder_output_length must be positive".into(),
            ));
        }
        if let Some(config) = request.eagle_config.as_ref() {
            validate_word_list("eagle_choices", &config.choices)?;
            if let Some(posterior_threshold) = config.posterior_threshold
                && (!posterior_threshold.is_finite() || posterior_threshold < 0.0)
            {
                return Err(Error::InvalidArgument(
                    "eagle_config posterior_threshold must be finite and non-negative".into(),
                ));
            }
            if let Some(top_k) = config.dynamic_tree_max_top_k
                && top_k <= 0
            {
                return Err(Error::InvalidArgument(
                    "eagle_config dynamic_tree_max_top_k must be positive".into(),
                ));
            }
        }
        if let Some(uid) = request.language_adapter_uid
            && uid < 0
        {
            return Err(Error::InvalidArgument(
                "language_adapter_uid cannot be negative".into(),
            ));
        }
        validate_word_list("bad_words", &request.bad_words)?;
        validate_word_list("stop_words", &request.stop_words)?;
        let total_tokens = request
            .input_ids
            .len()
            .checked_add(usize::try_from(request.max_new_tokens).unwrap_or(0))
            .ok_or_else(|| Error::InvalidArgument("TensorRT-LLM token count overflowed".into()))?;
        if total_tokens > self.profile.max_num_tokens {
            return Err(Error::InvalidArgument(format!(
                "TensorRT-LLM request has {total_tokens} total tokens but engine supports at most {}",
                self.profile.max_num_tokens
            )));
        }
        Ok(())
    }

    fn prompt_embedding<'a>(
        &self,
        tensor: &'a Tensor,
        input_ids: &[i32],
        token_extra_ids: Option<&'a [u64]>,
    ) -> Result<PromptEmbedding<'a>> {
        let dims = tensor.dims();
        let [rows, hidden] = <[usize; 2]>::try_from(dims).map_err(|_| {
            Error::InvalidArgument(format!(
                "prompt_embeddings must have shape [virtual_tokens, hidden_size], got {dims:?}"
            ))
        })?;
        if rows == 0 {
            return Err(Error::InvalidArgument(
                "prompt_embeddings must contain at least one virtual token".into(),
            ));
        }
        if rows > self.profile.max_prompt_embedding_table_size {
            return Err(Error::InvalidArgument(format!(
                "prompt_embeddings has {rows} rows but engine supports at most {}",
                self.profile.max_prompt_embedding_table_size
            )));
        }
        if hidden != self.profile.hidden_size {
            return Err(Error::InvalidArgument(format!(
                "prompt_embeddings hidden size mismatch: expected {}, got {hidden}",
                self.profile.hidden_size
            )));
        }
        if !tensor.is_contiguous() {
            return Err(Error::InvalidArgument(
                "prompt_embeddings must be contiguous before TensorRT-LLM Executor inference"
                    .into(),
            ));
        }
        let expected = self.profile.dtype;
        if tensor.dtype() != expected {
            return Err(Error::InvalidArgument(format!(
                "prompt_embeddings dtype mismatch: engine expects {expected:?}, got {:?}",
                tensor.dtype()
            )));
        }
        let input_token_extra_ids = match token_extra_ids {
            Some(token_extra_ids) => {
                if token_extra_ids.len() != input_ids.len() {
                    return Err(Error::InvalidArgument(format!(
                        "prompt_embedding_token_extra_ids length mismatch: expected {}, got {}",
                        input_ids.len(),
                        token_extra_ids.len()
                    )));
                }
                TokenExtraIds::Borrowed(token_extra_ids)
            }
            None => {
                validate_virtual_tokens(input_ids, self.profile.vocab_size, rows)?;
                TokenExtraIds::Owned(input_token_extra_ids(input_ids, self.profile.vocab_size))
            }
        };

        Ok(PromptEmbedding {
            tensor: TensorBuffer::new(tensor)?,
            rows,
            input_token_extra_ids,
        })
    }

    fn embedding_bias<'a>(&self, tensor: &'a Tensor) -> Result<EmbeddingBiasBuffer<'a>> {
        let dims = tensor.dims();
        let [len] = <[usize; 1]>::try_from(dims).map_err(|_| {
            Error::InvalidArgument(format!(
                "embedding_bias must have shape [vocab_size], got {dims:?}"
            ))
        })?;
        if len != self.profile.vocab_size {
            return Err(Error::InvalidArgument(format!(
                "embedding_bias vocab size mismatch: expected {}, got {len}",
                self.profile.vocab_size
            )));
        }
        if !tensor.is_contiguous() {
            return Err(Error::InvalidArgument(
                "embedding_bias must be contiguous before TensorRT-LLM Executor inference".into(),
            ));
        }
        Ok(EmbeddingBiasBuffer {
            tensor: TensorBuffer::new(tensor)?,
            len,
        })
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe { sys::tensorrt_llm_rs_executor_destroy(self.inner.as_ptr()) };
    }
}

#[cfg(feature = "cuda")]
struct OutputSinkParts {
    completion_event: Arc<PooledCudaEvent>,
    scratch: Option<OutputSinkScratch>,
    scratch_pool: Arc<OutputSinkScratchPool>,
    sink: sys::OutputSink,
}

#[cfg(feature = "cuda")]
impl OutputSinkParts {
    fn new(
        outputs: &mut OutputTensors,
        stream: CudaStream<'_>,
        event_pool: Arc<CudaEventPool>,
        scratch_pool: Arc<OutputSinkScratchPool>,
    ) -> Result<Self> {
        let Some(stream_location) = stream.device_location() else {
            return Err(Error::InvalidArgument(
                "Request::additional_outputs requires a stream created with CudaStream::from_device"
                    .into(),
            ));
        };
        let DeviceLocation::Cuda { gpu_id } = stream_location else {
            return Err(Error::InvalidArgument(
                "Request::additional_outputs requires a CUDA stream".into(),
            ));
        };
        let stream_device = c_int::try_from(gpu_id).map_err(|_| {
            Error::InvalidArgument(format!(
                "CUDA device id {gpu_id} cannot be represented as c_int"
            ))
        })?;

        let completion_event = event_pool.checkout(stream_location, stream)?;

        let mut scratch = scratch_pool.checkout()?;
        if !scratch.layout_matches(outputs) {
            scratch.rebuild_layout(outputs)?;
        } else {
            scratch.reset_written();
            scratch.refresh_stable_pointers();
        }

        let output_count = outputs.iter().count();
        let mut ranges: SmallVec<[CudaMemoryRange<'_>; 8]> = SmallVec::with_capacity(output_count);
        for (index, (name, tensor)) in outputs.iter().enumerate() {
            if tensor.device().location() != stream_location {
                return Err(Error::InvalidArgument(format!(
                    "output tensor {name:?} must be on the same CUDA device as the stream"
                )));
            }
            if !tensor.is_contiguous() {
                return Err(Error::InvalidArgument(format!(
                    "output tensor {name:?} must be contiguous for Request::additional_outputs"
                )));
            }
            let data_type = data_type_for_dtype(tensor.dtype())?;
            let shape = scratch.shape_slice(index).ok_or_else(|| {
                Error::InvalidArgument(format!("output tensor {name:?} sink shape is missing"))
            })?;
            let bytes = tensor_byte_len_for_shape(name, shape, data_type)?;
            let ptr = cuda_tensor_ptr(name, tensor, data_type)?;
            ranges.push(CudaMemoryRange::new(name, ptr, bytes)?);
            scratch.views[index].tensor = sys::TensorView {
                data: ptr.cast_const(),
                byte_len: bytes,
                data_type: data_type_to_executor_i32(data_type)?,
                shape: scratch.views[index].tensor.shape,
                rank: scratch.views[index].tensor.rank,
                memory_type: TENSOR_MEMORY_GPU,
            };
        }

        validate_cuda_output_aliases(&ranges)?;
        drop(ranges);

        scratch.refresh_stable_pointers();
        let sink = sys::OutputSink {
            tensors: scratch.views.as_ptr(),
            tensor_count: scratch.views.len(),
            stream: stream.as_raw(),
            stream_device,
            completion_event: completion_event.as_raw(),
        };

        Ok(Self {
            completion_event,
            scratch: Some(scratch),
            scratch_pool,
            sink,
        })
    }

    fn as_sys(&mut self) -> *const sys::OutputSink {
        let scratch = self
            .scratch
            .as_mut()
            .expect("output sink scratch is only absent during drop");
        scratch.refresh_stable_pointers();
        self.sink.tensors = scratch.views.as_ptr();
        self.sink.tensor_count = scratch.views.len();
        &self.sink
    }

    fn ensure_all_written(&self) -> Result<()> {
        let scratch = self
            .scratch
            .as_ref()
            .expect("output sink scratch is only absent during drop");
        for (index, written) in scratch.written.iter().enumerate() {
            if *written == 0 {
                return Err(Error::InvalidArgument(format!(
                    "TensorRT-LLM Executor did not return requested output {:?}",
                    scratch.names[index].to_string_lossy()
                )));
            }
        }
        Ok(())
    }

    fn recycle_scratch(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.scratch_pool.recycle(scratch);
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for OutputSinkParts {
    fn drop(&mut self) {
        self.recycle_scratch();
    }
}

#[cfg(not(feature = "cuda"))]
struct OutputSinkParts;

#[cfg(not(feature = "cuda"))]
impl OutputSinkParts {
    fn as_sys(&mut self) -> *const sys::OutputSink {
        unreachable!("output sinks are only constructed with the cuda feature")
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct CudaMemoryRange<'a> {
    name: &'a str,
    start: usize,
    end: usize,
}

#[cfg(feature = "cuda")]
impl<'a> CudaMemoryRange<'a> {
    fn new(name: &'a str, ptr: *mut c_void, bytes: usize) -> Result<Self> {
        let start = ptr as usize;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::InvalidArgument("CUDA output pointer range overflow".into()))?;
        Ok(Self { name, start, end })
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[cfg(feature = "cuda")]
fn validate_cuda_output_aliases(ranges: &[CudaMemoryRange<'_>]) -> Result<()> {
    for (index, range) in ranges.iter().enumerate() {
        for other in ranges.iter().skip(index + 1) {
            if range.overlaps(*other) {
                return Err(Error::InvalidArgument(format!(
                    "output CUDA memory for {:?} overlaps output {:?}; each output must use distinct storage",
                    range.name, other.name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_tensor_ptr(tensor_name: &str, tensor: &Tensor, data_type: DataType) -> Result<*mut c_void> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(Error::InvalidArgument(format!(
            "output tensor {tensor_name:?} is not backed by CUDA storage"
        )));
    };
    let len = tensor.elem_count();
    let start = layout.start_offset();
    let end = start.checked_add(len).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "output tensor {tensor_name:?} CUDA storage offset overflow"
        ))
    })?;

    let ptr = match data_type {
        DataType::Float => cuda_storage_ptr::<f32>(storage, start, end),
        DataType::Half => cuda_storage_ptr::<f16>(storage, start, end),
        DataType::Bf16 => cuda_storage_ptr::<bf16>(storage, start, end),
        DataType::Int32 => cuda_storage_ptr::<i32>(storage, start, end),
        DataType::Int64 => cuda_storage_ptr::<i64>(storage, start, end),
        DataType::Uint8 => cuda_storage_ptr::<u8>(storage, start, end),
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor output dtype {dtype:?} is not supported for CUDA output"
        ))),
    }?;
    Ok(ptr.cast_mut())
}

#[cfg(feature = "cuda")]
fn cuda_storage_ptr<T: candle_core::cuda::CudaDType>(
    storage: &CudaStorage,
    start: usize,
    end: usize,
) -> Result<*const c_void> {
    let slice = storage.as_cuda_slice::<T>()?;
    let view = slice.slice(start..end);
    let stream = slice.stream().clone();
    let (ptr, _guard) = view.device_ptr(&stream);
    Ok(ptr as usize as *const c_void)
}

struct PreparedRequest<'a> {
    _request: Request<'a>,
    _tensors: Box<RequestTensors<'a>>,
    _buffers: PooledRequestBuffers,
    sys_request: sys::Request,
}

struct RequestTensors<'a> {
    prompt: Option<PromptEmbedding<'a>>,
    embedding_bias: Option<EmbeddingBiasBuffer<'a>>,
    external_draft_logits: Option<TensorBuffer<'a>>,
    multimodal_embedding: Option<TensorBuffer<'a>>,
    mrope_rotary_cos_sin: Option<TensorBuffer<'a>>,
    lora_weights: Option<TensorBuffer<'a>>,
    lora_config_tensor: Option<TensorBuffer<'a>>,
    encoder_input_features: Option<TensorBuffer<'a>>,
    cross_attention_mask: Option<TensorBuffer<'a>>,
    skip_cross_attn_blocks: Option<TensorBuffer<'a>>,
}

struct PromptEmbedding<'a> {
    tensor: TensorBuffer<'a>,
    rows: usize,
    input_token_extra_ids: TokenExtraIds<'a>,
}

struct EmbeddingBiasBuffer<'a> {
    tensor: TensorBuffer<'a>,
    len: usize,
}

enum TokenExtraIds<'a> {
    Borrowed(&'a [u64]),
    Owned(SmallVec<[u64; INLINE_TOKEN_EXTRA_IDS]>),
}

impl<'a> TokenExtraIds<'a> {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Borrowed(ids) => ids,
            Self::Owned(ids) => ids,
        }
    }
}

enum TensorBufferData {
    Cpu(Vec<u8>),
    #[cfg(feature = "cuda")]
    Cuda {
        ptr: *const c_void,
        byte_len: usize,
    },
}

const INLINE_TENSOR_RANK: usize = 8;

#[derive(Clone, Debug)]
enum TensorShapeBuf {
    Inline {
        values: [i64; INLINE_TENSOR_RANK],
        len: usize,
    },
    Heap(Vec<i64>),
}

impl TensorShapeBuf {
    fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() <= INLINE_TENSOR_RANK {
            let mut values = [0; INLINE_TENSOR_RANK];
            for (index, dim) in dims.iter().enumerate() {
                values[index] = i64::try_from(*dim).map_err(|_| {
                    Error::InvalidArgument(
                        "TensorRT-LLM Executor tensor dimension cannot fit in i64".into(),
                    )
                })?;
            }
            return Ok(Self::Inline {
                values,
                len: dims.len(),
            });
        }

        dims.iter()
            .map(|dim| {
                i64::try_from(*dim).map_err(|_| {
                    Error::InvalidArgument(
                        "TensorRT-LLM Executor tensor dimension cannot fit in i64".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Self::Heap)
    }

    fn as_slice(&self) -> &[i64] {
        match self {
            Self::Inline { values, len } => &values[..*len],
            Self::Heap(values) => values,
        }
    }

    fn as_ptr(&self) -> *const i64 {
        if self.len() == 0 {
            std::ptr::null()
        } else {
            self.as_slice().as_ptr()
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[cfg(test)]
    fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }
}

struct TensorBuffer<'a> {
    data: TensorBufferData,
    shape: TensorShapeBuf,
    data_type: i32,
    _tensor: PhantomData<&'a Tensor>,
}

impl<'a> TensorBuffer<'a> {
    fn new(tensor: &'a Tensor) -> Result<Self> {
        if !tensor.is_contiguous() {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM Executor tensor inputs must be contiguous".into(),
            ));
        }
        let data_type = data_type_for_dtype(tensor.dtype())?;
        let shape = TensorShapeBuf::new(tensor.dims())?;
        #[cfg(feature = "cuda")]
        if matches!(&*tensor.storage_and_layout().0, Storage::Cuda(_)) {
            let byte_len = tensor_byte_len_for_shape(
                "TensorRT-LLM Executor tensor input",
                shape.as_slice(),
                data_type,
            )?;
            return Ok(Self {
                data: TensorBufferData::Cuda {
                    ptr: cuda_tensor_ptr("TensorRT-LLM Executor tensor input", tensor, data_type)?
                        .cast_const(),
                    byte_len,
                },
                shape,
                data_type: data_type_to_executor_i32(data_type)?,
                _tensor: PhantomData,
            });
        }

        Ok(Self {
            data: TensorBufferData::Cpu(tensor_bytes(tensor, data_type)?),
            shape,
            data_type: data_type_to_executor_i32(data_type)?,
            _tensor: PhantomData,
        })
    }

    fn as_sys(&self) -> sys::TensorView {
        let (data, byte_len, memory_type) = match &self.data {
            TensorBufferData::Cpu(bytes) => (
                bytes.as_ptr().cast::<c_void>(),
                bytes.len(),
                TENSOR_MEMORY_CPU,
            ),
            #[cfg(feature = "cuda")]
            TensorBufferData::Cuda { ptr, byte_len, .. } => (*ptr, *byte_len, TENSOR_MEMORY_GPU),
        };
        sys::TensorView {
            data,
            byte_len,
            data_type: self.data_type,
            shape: self.shape.as_ptr(),
            rank: self.shape.len(),
            memory_type,
        }
    }
}

fn optional_tensor_view(buffer: Option<&TensorBuffer<'_>>) -> sys::TensorView {
    buffer.map(TensorBuffer::as_sys).unwrap_or_default()
}

#[derive(Default)]
struct WordListBuffer {
    tokens: SmallVec<[i32; INLINE_WORD_LIST_TOKENS]>,
    offsets: SmallVec<[usize; INLINE_WORD_LIST_OFFSETS]>,
}

#[derive(Default)]
struct StringListBuffer {
    names: SmallVec<[*const c_char; INLINE_NAME_LIST]>,
}

#[derive(Default)]
struct NullableStringListBuffer {
    names: SmallVec<[*const c_char; INLINE_NAME_LIST]>,
}

impl NullableStringListBuffer {
    #[cfg(test)]
    fn new(strings: &[Option<CString>]) -> Self {
        let mut buffer = Self::default();
        buffer.replace(strings);
        buffer
    }

    fn clear(&mut self) {
        self.names.clear();
    }

    fn replace(&mut self, strings: &[Option<CString>]) {
        self.clear();
        self.names.reserve(strings.len());
        self.names.extend(strings.iter().map(|string| {
            string
                .as_ref()
                .map_or(std::ptr::null(), |string| string.as_ptr())
        }));
    }

    fn as_ptr(&self) -> *const *const c_char {
        if self.names.is_empty() {
            std::ptr::null()
        } else {
            self.names.as_ptr()
        }
    }

    fn len(&self) -> usize {
        self.names.len()
    }
}

impl StringListBuffer {
    #[cfg(test)]
    fn new(strings: &[CString]) -> Self {
        let mut buffer = Self::default();
        buffer.replace(strings);
        buffer
    }

    fn clear(&mut self) {
        self.names.clear();
    }

    fn replace(&mut self, strings: &[CString]) {
        self.clear();
        self.names.reserve(strings.len());
        self.names
            .extend(strings.iter().map(|string| string.as_ptr()));
    }

    fn as_sys(&self) -> sys::StringList {
        if self.names.is_empty() {
            return sys::StringList::default();
        }
        sys::StringList {
            names: self.names.as_ptr(),
            count: self.names.len(),
        }
    }
}

impl WordListBuffer {
    #[cfg(test)]
    fn new(words: &[&[i32]]) -> Self {
        let mut buffer = Self::default();
        buffer.replace(words);
        buffer
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.offsets.clear();
    }

    fn replace(&mut self, words: &[&[i32]]) {
        self.clear();
        if words.is_empty() {
            return;
        }

        let token_count = words.iter().map(|word| word.len()).sum();
        self.tokens.reserve(token_count);
        self.offsets.reserve(words.len() + 1);
        self.offsets.push(0);
        for word in words {
            self.tokens.extend_from_slice(word);
            self.offsets.push(self.tokens.len());
        }
    }

    fn as_sys(&self) -> sys::WordList {
        if self.tokens.is_empty() {
            return sys::WordList::default();
        }
        sys::WordList {
            tokens: self.tokens.as_ptr(),
            token_count: self.tokens.len(),
            offsets: self.offsets.as_ptr(),
            offset_count: self.offsets.len(),
        }
    }
}

fn init_plugins() -> Result<()> {
    let status = unsafe { sys::tensorrt_llm_rs_init_plugins() };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PluginInit { status })
    }
}

fn infer_additional_outputs(engine_dir: &Path) -> Result<Vec<CString>> {
    let engine_buffer = fs::read(engine_dir.join("rank0.engine"))?;
    let session = tensorrt::Session::new(engine_buffer)?;
    session
        .io_tensors()
        .iter()
        .filter(|info| info.io_mode == TensorIOMode::Output)
        .filter(|info| info.name != "logits")
        .map(|info| CString::new(info.name.as_str()).map_err(Error::from))
        .collect()
}

fn dtype_for_data_type(data_type: DataType) -> Result<DType> {
    match data_type {
        DataType::Bf16 => Ok(DType::BF16),
        DataType::Half => Ok(DType::F16),
        DataType::Float => Ok(DType::F32),
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM model dtype {dtype:?} is not supported"
        ))),
    }
}

fn validate_virtual_tokens(input_ids: &[i32], vocab_size: usize, prompt_rows: usize) -> Result<()> {
    let prompt_end = vocab_size
        .checked_add(prompt_rows)
        .ok_or_else(|| Error::InvalidArgument("prompt virtual token range overflowed".into()))?;
    for token in input_ids {
        let Ok(token) = usize::try_from(*token) else {
            continue;
        };
        if token >= vocab_size && token >= prompt_end {
            return Err(Error::InvalidArgument(format!(
                "virtual token id {token} is outside prompt embedding range {vocab_size}..{prompt_end}"
            )));
        }
    }
    Ok(())
}

fn input_token_extra_ids(
    input_ids: &[i32],
    vocab_size: usize,
) -> SmallVec<[u64; INLINE_TOKEN_EXTRA_IDS]> {
    let vocab_size = i32::try_from(vocab_size).unwrap_or(i32::MAX);
    input_ids
        .iter()
        .map(|token| {
            if *token >= vocab_size {
                u64::try_from(*token - vocab_size).unwrap_or(0)
            } else {
                0
            }
        })
        .collect()
}

fn validate_word_list(name: &str, words: &[&[i32]]) -> Result<()> {
    for (index, word) in words.iter().enumerate() {
        if word.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "{name} entry {index} must contain at least one token"
            )));
        }
    }
    Ok(())
}

fn validate_retention_priority(name: &str, priority: i32) -> Result<()> {
    if !(0..=100).contains(&priority) {
        return Err(Error::InvalidArgument(format!(
            "{name} must be between 0 and 100"
        )));
    }
    Ok(())
}

fn data_type_for_dtype(dtype: DType) -> Result<DataType> {
    match dtype {
        DType::U8 => Ok(DataType::Uint8),
        DType::I32 => Ok(DataType::Int32),
        DType::I64 => Ok(DataType::Int64),
        DType::BF16 => Ok(DataType::Bf16),
        DType::F16 => Ok(DataType::Half),
        DType::F32 => Ok(DataType::Float),
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor tensor dtype {dtype:?} is not supported"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn tensor_byte_len_for_shape(tensor: &str, shape: &[i64], data_type: DataType) -> Result<usize> {
    let elements = shape.iter().try_fold(1usize, |acc, dim| {
        if *dim < 0 {
            return Err(Error::InvalidArgument(format!(
                "{tensor} shape still has dynamic dimension: {shape:?}"
            )));
        }
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| Error::InvalidArgument(format!("{tensor} element count overflow")))
    })?;
    let bytes_per_element = data_type.bytes_per_element().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{tensor} has unsupported TensorRT dtype {data_type:?}"
        ))
    })?;
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| Error::InvalidArgument(format!("{tensor} byte size overflow")))
}

fn tensor_bytes(tensor: &Tensor, data_type: DataType) -> Result<Vec<u8>> {
    let tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
    match data_type {
        DataType::Uint8 => Ok(tensor.flatten_all()?.to_vec1::<u8>()?),
        DataType::Int32 => {
            Ok(tensorrt::as_bytes(&tensor.flatten_all()?.to_vec1::<i32>()?).to_vec())
        }
        DataType::Int64 => {
            Ok(tensorrt::as_bytes(&tensor.flatten_all()?.to_vec1::<i64>()?).to_vec())
        }
        DataType::Bf16 => {
            Ok(tensorrt::as_bytes(&tensor.flatten_all()?.to_vec1::<bf16>()?).to_vec())
        }
        DataType::Half => Ok(tensorrt::as_bytes(&tensor.flatten_all()?.to_vec1::<f16>()?).to_vec()),
        DataType::Float => {
            Ok(tensorrt::as_bytes(&tensor.flatten_all()?.to_vec1::<f32>()?).to_vec())
        }
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor tensor dtype {dtype:?} is not supported"
        ))),
    }
}

fn usize_to_i32(value: usize, field: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::InvalidArgument(format!("TensorRT-LLM {field} {value} cannot fit in i32"))
    })
}

fn data_type_to_executor_i32(data_type: DataType) -> Result<i32> {
    match data_type {
        DataType::Bool => Ok(0),
        DataType::Uint8 => Ok(1),
        DataType::Int8 => Ok(2),
        DataType::Int32 => Ok(3),
        DataType::Int64 => Ok(4),
        DataType::Bf16 => Ok(5),
        DataType::Fp8 => Ok(6),
        DataType::Half => Ok(7),
        DataType::Float => Ok(8),
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor dtype mapping for {dtype:?} is not implemented"
        ))),
    }
}

fn final_response_from_sys(
    raw: *mut sys::Response,
    count: usize,
    #[cfg(feature = "cuda")] completion_event: Option<Arc<PooledCudaEvent>>,
) -> Result<Option<Response>> {
    if count == 0 {
        return Ok(None);
    }
    if raw.is_null() {
        return Err(Error::Executor(
            "TensorRT-LLM Executor returned a null response pointer".into(),
        ));
    }

    let raw_responses = Arc::new(RawResponses {
        raw: NonNull::new(raw).expect("raw response pointer was checked"),
        count,
        #[cfg(feature = "cuda")]
        completion_event: completion_event.as_ref().map(Arc::clone),
    });

    let Some(response) = unsafe { slice::from_raw_parts(raw, count) }
        .iter()
        .find(|response| response.output.is_final != 0)
    else {
        return Ok(None);
    };

    Response::from_sys(
        response,
        #[cfg(feature = "cuda")]
        completion_event,
        Some(RawOutputGuard::new(raw_responses)),
    )
    .map(Some)
}

fn take_error(error: *mut c_char) -> Error {
    if error.is_null() {
        return Error::Executor("unknown TensorRT-LLM Executor error".to_owned());
    }
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    unsafe { sys::tensorrt_llm_rs_error_free(error) };
    Error::Executor(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::ManuallyDrop;

    fn fake_executor() -> ManuallyDrop<Executor> {
        ManuallyDrop::new(Executor {
            inner: NonNull::dangling(),
            profile: ValidationProfile {
                dtype: DType::F32,
                executor_data_type: 8,
                vocab_size: 10,
                hidden_size: 4,
                max_num_tokens: 8,
                max_prompt_embedding_table_size: 2,
            },
            request_buffer_pool: Arc::new(RequestBufferPool::default()),
            #[cfg(feature = "cuda")]
            event_pool: Arc::new(CudaEventPool::default()),
            #[cfg(feature = "cuda")]
            sink_scratch_pool: Arc::new(OutputSinkScratchPool::default()),
        })
    }

    #[test]
    fn extra_ids_follow_virtual_token_range() {
        let ids = input_token_extra_ids(&[4, 5, 6, 9], 5);
        assert_eq!(ids.as_slice(), &[0, 0, 1, 4]);
        assert!(!ids.spilled());
    }

    #[test]
    fn virtual_token_validation_rejects_out_of_range_tokens() {
        let error = validate_virtual_tokens(&[10], 5, 2).unwrap_err();
        assert!(error.to_string().contains("outside prompt embedding range"));
    }

    #[test]
    fn word_list_buffer_flattens_offsets() {
        let first = [1, 2];
        let second = [3];
        let words = [first.as_slice(), second.as_slice()];
        let buffer = WordListBuffer::new(&words);
        assert_eq!(buffer.tokens.as_slice(), &[1, 2, 3]);
        assert_eq!(buffer.offsets.as_slice(), &[0, 2, 3]);
        assert!(!buffer.tokens.spilled());
        assert!(!buffer.offsets.spilled());

        let sys = buffer.as_sys();
        assert_eq!(sys.token_count, 3);
        assert_eq!(sys.offset_count, 3);
    }

    #[test]
    fn empty_ffi_lists_use_null_pointers() {
        let words = WordListBuffer::new(&[]);
        let sys_words = words.as_sys();
        assert!(sys_words.tokens.is_null());
        assert_eq!(sys_words.token_count, 0);
        assert!(sys_words.offsets.is_null());
        assert_eq!(sys_words.offset_count, 0);

        let strings = StringListBuffer::new(&[]);
        let sys_strings = strings.as_sys();
        assert!(sys_strings.names.is_null());
        assert_eq!(sys_strings.count, 0);

        let nullable = NullableStringListBuffer::new(&[]);
        assert!(nullable.as_ptr().is_null());
        assert_eq!(nullable.len(), 0);
    }

    #[test]
    fn string_list_buffers_stay_inline_for_common_sizes() {
        let first = CString::new("first").unwrap();
        let second = CString::new("second").unwrap();
        let strings = [first, second];
        let list = StringListBuffer::new(&strings);

        assert_eq!(list.as_sys().count, 2);
        assert!(!list.names.spilled());

        let nullable = NullableStringListBuffer::new(&[
            Some(CString::new("image").unwrap()),
            None,
            Some(CString::new("video").unwrap()),
        ]);

        assert_eq!(nullable.len(), 3);
        assert!(!nullable.names.spilled());
    }

    #[test]
    fn request_buffer_pool_reuses_scratch_capacity() {
        let pool = Arc::new(RequestBufferPool::default());
        let first = [1; INLINE_WORD_LIST_TOKENS + 1];
        let second = [2; INLINE_WORD_LIST_TOKENS + 1];
        let first_words = [first.as_slice()];
        let second_words = [second.as_slice()];

        let first_capacity = {
            let mut buffers = pool.checkout().unwrap();
            buffers.bad_words.replace(&first_words);
            assert!(buffers.bad_words.tokens.spilled());
            buffers.bad_words.tokens.capacity()
        };

        let buffers = pool.checkout().unwrap();
        assert!(buffers.bad_words.tokens.capacity() >= first_capacity);
        drop(buffers);

        let mut buffers = pool.checkout().unwrap();
        buffers.bad_words.replace(&second_words);
        assert_eq!(buffers.bad_words.tokens.as_slice(), second.as_slice());
        assert!(buffers.bad_words.tokens.capacity() >= first_capacity);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn output_sink_scratch_reuses_matching_layout() {
        let hidden = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let logits = Tensor::zeros((1, 2, 10), DType::F32, &Device::Cpu).unwrap();
        let outputs = OutputTensors::new([("hidden", hidden), ("logits", logits)]);
        let mut scratch = OutputSinkScratch::default();

        scratch.rebuild_layout(&outputs).unwrap();
        let first_name = scratch.names[0].as_ptr();
        let first_shape = scratch.views[0].tensor.shape;

        assert!(scratch.layout_matches(&outputs));
        scratch.reset_written();
        scratch.refresh_stable_pointers();

        assert_eq!(scratch.names[0].as_ptr(), first_name);
        assert_eq!(scratch.views[0].tensor.shape, first_shape);

        let resized = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let resized_outputs = OutputTensors::new([("hidden", resized)]);
        assert!(!scratch.layout_matches(&resized_outputs));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn output_sink_parts_refreshes_inline_shape_pointers_after_move() {
        let hidden = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let outputs = OutputTensors::new([("hidden", hidden)]);
        let mut scratch = OutputSinkScratch::default();
        scratch.rebuild_layout(&outputs).unwrap();

        let event_pool = Arc::new(CudaEventPool::default());
        let completion_event = Arc::new(PooledCudaEvent {
            event: None,
            synchronized: AtomicBool::new(true),
            device_location: DeviceLocation::Cuda { gpu_id: 0 },
            pool: Arc::clone(&event_pool),
        });
        let mut parts = OutputSinkParts {
            completion_event,
            scratch: Some(scratch),
            scratch_pool: Arc::new(OutputSinkScratchPool::default()),
            sink: sys::OutputSink {
                tensors: std::ptr::null(),
                tensor_count: 0,
                stream: std::ptr::null_mut(),
                stream_device: 0,
                completion_event: std::ptr::null_mut(),
            },
        };

        let sys = parts.as_sys();
        let scratch = parts.scratch.as_ref().unwrap();
        assert_eq!(unsafe { (*sys).tensor_count }, 1);
        assert_eq!(scratch.views[0].tensor.shape, scratch.shape_values.as_ptr());
    }

    #[test]
    fn tensor_shape_buffer_avoids_heap_for_common_ranks() {
        let shape = TensorShapeBuf::new(&[1, 2, 3, 4]).unwrap();
        assert!(shape.is_inline());
        assert_eq!(shape.as_slice(), &[1, 2, 3, 4]);
        assert!(!shape.as_ptr().is_null());

        let scalar = TensorShapeBuf::new(&[]).unwrap();
        assert!(scalar.is_inline());
        assert!(scalar.as_ptr().is_null());

        let large = TensorShapeBuf::new(&[1; INLINE_TENSOR_RANK + 1]).unwrap();
        assert!(!large.is_inline());
        assert_eq!(large.len(), INLINE_TENSOR_RANK + 1);
    }

    #[test]
    fn word_list_validation_rejects_empty_words() {
        let error = validate_word_list("stop_words", &[&[]]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must contain at least one token")
        );
    }

    #[test]
    fn validation_rejects_invalid_request_shapes_and_limits() {
        let executor = fake_executor();

        assert!(executor.validate_request(&Request::new(&[])).is_err());
        assert!(
            executor
                .validate_request(&Request::new(&[1]).max_new_tokens(-1))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1, 2]).position_ids(&[0]))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).priority(f32::NAN))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1, 2]).prompt_embedding_token_extra_ids(&[0]))
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1])
                        .external_draft_tokens(crate::request::ExternalDraftTokensConfig::new(&[]))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).external_draft_tokens(
                    crate::request::ExternalDraftTokensConfig::new(&[2]).acceptance_threshold(0.0)
                ))
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1]).multimodal_input(crate::request::MultimodalInput::new(
                        [&[][..]],
                        &[0],
                        &[1]
                    ))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1]).multimodal_input(
                        crate::request::MultimodalInput::new([&[1][..]], &[0], &[1])
                            .uuids([Some("a"), Some("b")])
                    )
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1]).lookahead(crate::request::LookaheadConfig::new(0, 1, 1))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).kv_cache_retention(
                    crate::request::KvCacheRetentionConfig::new([
                        crate::request::KvCacheTokenRangeRetentionConfig::new(-1)
                    ])
                ))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).kv_cache_retention(
                    crate::request::KvCacheRetentionConfig::new([
                        crate::request::KvCacheTokenRangeRetentionConfig::new(4).token_end(4)
                    ])
                ))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).kv_cache_retention(
                    crate::request::KvCacheRetentionConfig::new([
                        crate::request::KvCacheTokenRangeRetentionConfig::new(0).priority(101)
                    ])
                ))
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1])
                        .context_phase(crate::request::ContextPhaseParams::new(&[], 0))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).encoder_output_length(0))
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1])
                        .eagle(crate::request::EagleConfig::new().choices([&[][..]]))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1])
                        .eagle(crate::request::EagleConfig::new().posterior_threshold(f32::NAN))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(
                    &Request::new(&[1])
                        .eagle(crate::request::EagleConfig::new().max_dynamic_tree_top_k(0))
                )
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).language_adapter_uid(-1))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1]).bad_words([&[][..]]))
                .is_err()
        );
        assert!(
            executor
                .validate_request(&Request::new(&[1; 8]).max_new_tokens(1))
                .is_err()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn additional_output_sink_validation_runs_before_request_preparation() {
        let executor = fake_executor();
        let mut outputs = OutputTensors::new([(
            "marked_model_output",
            Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap(),
        )]);
        let mut request =
            Request::new(&[1; 16]).additional_outputs(&mut outputs, CudaStream::DEFAULT);
        let additional_outputs = request.additional_output_sink.take().unwrap();

        let error = match executor.prepare_output_sink(additional_outputs) {
            Ok(_) => panic!("invalid additional output sink should be rejected"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("requires a stream created with CudaStream::from_device")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn prepare_request_encodes_cpu_tensor_extensions() {
        let executor = fake_executor();
        let input_ids = [10, 11];
        let prompt_embeddings = Tensor::from_vec(vec![0f32; 8], (2, 4), &Device::Cpu).unwrap();
        let embedding_bias = Tensor::from_vec(vec![0f32; 10], 10, &Device::Cpu).unwrap();
        let logits = Tensor::from_vec(vec![0f32; 2], 2, &Device::Cpu).unwrap();
        let multimodal_embedding = Tensor::from_vec(vec![0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let rotary = Tensor::from_vec(vec![0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let lora_weights = Tensor::from_vec(vec![0f32; 4], 4, &Device::Cpu).unwrap();
        let lora_config = Tensor::from_vec(vec![1i32, 2], 2, &Device::Cpu).unwrap();
        let encoder_features = Tensor::from_vec(vec![0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let cross_attention_mask = Tensor::from_vec(vec![1f32; 2], 2, &Device::Cpu).unwrap();
        let skip_blocks = Tensor::from_vec(vec![0i32, 1], 2, &Device::Cpu).unwrap();
        let bad = [4, 5];
        let stop = [2];
        let mut outputs = OutputTensors::new([(
            "marked_model_output",
            Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap(),
        )]);
        let request = Request::new(&input_ids)
            .prompt_embeddings(&prompt_embeddings)
            .embedding_bias(&embedding_bias)
            .bad_words([bad.as_slice()])
            .stop_words([stop.as_slice()])
            .external_draft_tokens(
                crate::request::ExternalDraftTokensConfig::new(&[1, 2]).logits(&logits),
            )
            .multimodal_embedding(&multimodal_embedding)
            .mrope(crate::request::MropeConfig::new(&rotary, 1))
            .lora(
                crate::request::LoraConfig::new(3)
                    .weights(&lora_weights)
                    .config(&lora_config),
            )
            .encoder_features(&encoder_features)
            .cross_attention_mask(&cross_attention_mask)
            .skip_cross_attention_blocks(&skip_blocks)
            .additional_outputs(&mut outputs, CudaStream::DEFAULT);

        let error = match executor.prepare_request(request) {
            Ok(_) => panic!("preallocated additional outputs should be rejected outside run"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("preallocated additional outputs can only be used")
        );

        let request = Request::new(&input_ids)
            .prompt_embeddings(&prompt_embeddings)
            .embedding_bias(&embedding_bias)
            .bad_words([bad.as_slice()])
            .stop_words([stop.as_slice()])
            .external_draft_tokens(
                crate::request::ExternalDraftTokensConfig::new(&[1, 2]).logits(&logits),
            )
            .multimodal_embedding(&multimodal_embedding)
            .mrope(crate::request::MropeConfig::new(&rotary, 1))
            .lora(
                crate::request::LoraConfig::new(3)
                    .weights(&lora_weights)
                    .config(&lora_config),
            )
            .encoder_features(&encoder_features)
            .cross_attention_mask(&cross_attention_mask)
            .skip_cross_attention_blocks(&skip_blocks);

        let prepared = executor.prepare_request(request).unwrap();
        assert!(
            prepared
                ._tensors
                .prompt
                .as_ref()
                .unwrap()
                .tensor
                .shape
                .is_inline()
        );
        assert_eq!(prepared.sys_request.streaming, 0);
        assert_eq!(
            prepared.sys_request.bad_words.tokens,
            prepared._buffers.bad_words.tokens.as_ptr()
        );
        assert_eq!(
            prepared.sys_request.stop_words.tokens,
            prepared._buffers.stop_words.tokens.as_ptr()
        );
        assert_eq!(
            prepared.sys_request.prompt_embedding_data,
            prepared
                ._tensors
                .prompt
                .as_ref()
                .unwrap()
                .tensor
                .as_sys()
                .data
        );
        assert_eq!(prepared.sys_request.prompt_embedding_rows, 2);
        assert_eq!(prepared.sys_request.prompt_embedding_hidden_size, 4);
        assert_eq!(
            prepared.sys_request.input_token_extra_ids_len,
            input_ids.len()
        );
        assert_eq!(prepared.sys_request.embedding_bias_len, 10);
        assert_eq!(prepared.sys_request.has_external_draft_tokens_config, 1);
        assert_ne!(
            prepared
                .sys_request
                .external_draft_tokens_config
                .logits
                .data,
            std::ptr::null()
        );
        assert_ne!(
            prepared.sys_request.multimodal_embedding.data,
            std::ptr::null()
        );
        assert_eq!(prepared.sys_request.has_mrope_config, 1);
        assert_eq!(prepared.sys_request.has_lora_config, 1);
        assert_ne!(
            prepared.sys_request.encoder_input_features.data,
            std::ptr::null()
        );
        assert_ne!(
            prepared.sys_request.cross_attention_mask.data,
            std::ptr::null()
        );
        assert_ne!(
            prepared.sys_request.skip_cross_attn_blocks.data,
            std::ptr::null()
        );
    }

    #[test]
    fn tensor_helpers_cover_supported_and_unsupported_types() {
        assert_eq!(dtype_for_data_type(DataType::Bf16).unwrap(), DType::BF16);
        assert_eq!(dtype_for_data_type(DataType::Half).unwrap(), DType::F16);
        assert_eq!(dtype_for_data_type(DataType::Float).unwrap(), DType::F32);
        assert!(dtype_for_data_type(DataType::Int32).is_err());

        assert_eq!(data_type_for_dtype(DType::U8).unwrap(), DataType::Uint8);
        assert_eq!(data_type_for_dtype(DType::I32).unwrap(), DataType::Int32);
        assert_eq!(data_type_for_dtype(DType::I64).unwrap(), DataType::Int64);
        assert_eq!(data_type_for_dtype(DType::BF16).unwrap(), DataType::Bf16);
        assert_eq!(data_type_for_dtype(DType::F16).unwrap(), DataType::Half);
        assert_eq!(data_type_for_dtype(DType::F32).unwrap(), DataType::Float);
        assert!(data_type_for_dtype(DType::F64).is_err());

        assert_eq!(data_type_to_executor_i32(DataType::Bool).unwrap(), 0);
        assert_eq!(data_type_to_executor_i32(DataType::Uint8).unwrap(), 1);
        assert_eq!(data_type_to_executor_i32(DataType::Int8).unwrap(), 2);
        assert_eq!(data_type_to_executor_i32(DataType::Int32).unwrap(), 3);
        assert_eq!(data_type_to_executor_i32(DataType::Int64).unwrap(), 4);
        assert_eq!(data_type_to_executor_i32(DataType::Bf16).unwrap(), 5);
        assert_eq!(data_type_to_executor_i32(DataType::Fp8).unwrap(), 6);
        assert_eq!(data_type_to_executor_i32(DataType::Half).unwrap(), 7);
        assert_eq!(data_type_to_executor_i32(DataType::Float).unwrap(), 8);
        assert!(data_type_to_executor_i32(DataType::Int4).is_err());

        let tensors = [
            (
                Tensor::from_vec(vec![1u8], 1, &Device::Cpu).unwrap(),
                DataType::Uint8,
                1,
            ),
            (
                Tensor::from_vec(vec![1i32], 1, &Device::Cpu).unwrap(),
                DataType::Int32,
                4,
            ),
            (
                Tensor::from_vec(vec![1i64], 1, &Device::Cpu).unwrap(),
                DataType::Int64,
                8,
            ),
            (
                Tensor::from_vec(vec![bf16::from_f32(1.0)], 1, &Device::Cpu).unwrap(),
                DataType::Bf16,
                2,
            ),
            (
                Tensor::from_vec(vec![f16::from_f32(1.0)], 1, &Device::Cpu).unwrap(),
                DataType::Half,
                2,
            ),
            (
                Tensor::from_vec(vec![1f32], 1, &Device::Cpu).unwrap(),
                DataType::Float,
                4,
            ),
        ];
        for (tensor, data_type, byte_len) in tensors {
            assert_eq!(tensor_bytes(&tensor, data_type).unwrap().len(), byte_len);
        }
        let tensor = Tensor::from_vec(vec![1f32], 1, &Device::Cpu).unwrap();
        assert!(tensor_bytes(&tensor, DataType::Int4).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_output_alias_validation_catches_overlaps() {
        let first = CudaMemoryRange::new("first", 0x1000usize as *mut c_void, 16).unwrap();
        let second = CudaMemoryRange::new("second", 0x1008usize as *mut c_void, 16).unwrap();
        assert!(validate_cuda_output_aliases(&[first, second]).is_err());

        let second = CudaMemoryRange::new("second", 0x1010usize as *mut c_void, 16).unwrap();
        assert!(validate_cuda_output_aliases(&[first, second]).is_ok());
    }

    #[test]
    fn responses_and_error_helpers_handle_empty_or_null_pointers() {
        assert!(
            final_response_from_sys(
                std::ptr::null_mut(),
                0,
                #[cfg(feature = "cuda")]
                None,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            final_response_from_sys(
                std::ptr::null_mut(),
                1,
                #[cfg(feature = "cuda")]
                None,
            )
            .is_err()
        );
        assert_eq!(
            take_error(std::ptr::null_mut()).to_string(),
            "TensorRT-LLM executor operation failed: unknown TensorRT-LLM Executor error"
        );
    }

    #[test]
    fn response_from_sys_keeps_standard_outputs_lazy() {
        let mut token_ids = [1, 2, 3];
        let raw = sys::Response {
            request_id: 42,
            output: sys::Output {
                token_ids: token_ids.as_mut_ptr(),
                token_count: token_ids.len(),
                ..sys::Output::default()
            },
            ..sys::Response::default()
        };

        let response = Response::from_sys(
            &raw,
            #[cfg(feature = "cuda")]
            None,
            Some(RawOutputGuard::new(Arc::new(()))),
        )
        .unwrap();

        assert_eq!(response.request_id(), 42);
        assert_eq!(
            response.outputs().tensor_is_materialized("token_ids"),
            Some(false)
        );
        assert_eq!(
            response.outputs().tensor_is_materialized("token_offsets"),
            Some(false)
        );
        assert_eq!(
            response.outputs()["token_ids"].to_vec1::<i32>().unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            response.outputs()["token_offsets"]
                .to_vec1::<i64>()
                .unwrap(),
            vec![0, 3]
        );
        assert_eq!(
            response.outputs().tensor_is_materialized("token_ids"),
            Some(true)
        );
        assert_eq!(
            response.outputs().tensor_is_materialized("token_offsets"),
            Some(true)
        );
    }
}
