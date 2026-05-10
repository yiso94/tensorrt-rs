use crate::{Error, Result};
use candle_core::{Device, Tensor};
use half::{bf16, f16};
use std::ffi::CStr;
use std::fmt;
use std::ops::Index;
use std::slice;
use std::sync::{Arc, OnceLock};
use tensorrt::DataType;
use tensorrt_llm_sys as sys;

#[derive(Clone)]
pub(crate) struct RawOutputGuard {
    _owner: Arc<dyn Send + Sync>,
}

impl RawOutputGuard {
    pub(crate) fn new<T>(owner: Arc<T>) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self { _owner: owner }
    }
}

impl fmt::Debug for RawOutputGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawOutputGuard")
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct ExecutorOutput {
    tensors: Vec<TensorOutput>,
    is_final: bool,
    finish_reasons: Vec<FinishReason>,
    context_phase_params: Option<OutputContextPhaseParams>,
    spec_dec_fast_logits_info: Option<SpeculativeDecodingFastLogitsInfo>,
    sequence_index: i32,
    is_sequence_final: bool,
}

impl ExecutorOutput {
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(TensorOutput::tensor)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.tensor()))
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn is_final(&self) -> bool {
        self.is_final
    }

    pub fn finish_reasons(&self) -> &[FinishReason] {
        &self.finish_reasons
    }

    pub fn context_phase(&self) -> Option<&OutputContextPhaseParams> {
        self.context_phase_params.as_ref()
    }

    pub fn speculative_decoding_fast_logits_info(
        &self,
    ) -> Option<SpeculativeDecodingFastLogitsInfo> {
        self.spec_dec_fast_logits_info
    }

    pub fn sequence_index(&self) -> i32 {
        self.sequence_index
    }

    pub fn is_sequence_final(&self) -> bool {
        self.is_sequence_final
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn extend_tensors(&mut self, tensors: impl IntoIterator<Item = (String, Tensor)>) {
        self.tensors.extend(
            tensors
                .into_iter()
                .map(|(name, tensor)| TensorOutput::ready(name, tensor)),
        );
    }

    #[cfg(test)]
    pub(crate) fn tensor_is_materialized(&self, name: &str) -> Option<bool> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(TensorOutput::is_materialized)
    }
}

impl Index<&str> for ExecutorOutput {
    type Output = Tensor;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("TensorRT-LLM output {name:?} was not returned"))
    }
}

impl IntoIterator for ExecutorOutput {
    type Item = (String, Tensor);
    type IntoIter = ExecutorOutputIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        ExecutorOutputIntoIter {
            inner: self.tensors.into_iter(),
        }
    }
}

pub struct ExecutorOutputIntoIter {
    inner: std::vec::IntoIter<TensorOutput>,
}

impl Iterator for ExecutorOutputIntoIter {
    type Item = (String, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(TensorOutput::into_pair)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for ExecutorOutputIntoIter {}

#[derive(Debug)]
struct TensorOutput {
    name: String,
    tensor: LazyTensor,
}

impl TensorOutput {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn ready(name: impl Into<String>, tensor: Tensor) -> Self {
        Self {
            name: name.into(),
            tensor: LazyTensor::Ready(tensor),
        }
    }

    fn int32(
        name: impl Into<String>,
        raw: *const i32,
        count: usize,
        shape: Vec<usize>,
        guard: Option<RawOutputGuard>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            tensor: LazyTensor::from_typed_raw(DataType::Int32, raw.cast(), count, shape, guard)?,
        })
    }

    fn offsets(
        name: impl Into<String>,
        offsets: *const usize,
        sequence_count: usize,
        data_count: usize,
        guard: Option<RawOutputGuard>,
    ) -> Result<Self> {
        let name = name.into();
        Ok(match guard {
            Some(guard) => Self {
                name,
                tensor: LazyTensor::from_offsets(offsets, sequence_count, data_count, guard)?,
            },
            None => {
                let offsets = output_offsets(offsets, sequence_count, data_count)?;
                let offset_count = offsets.len();
                Self {
                    name,
                    tensor: LazyTensor::new(TensorData::Int64(offsets), vec![offset_count])?,
                }
            }
        })
    }

    fn float32(
        name: impl Into<String>,
        raw: *const f32,
        count: usize,
        shape: Vec<usize>,
        guard: Option<RawOutputGuard>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            tensor: LazyTensor::from_typed_raw(DataType::Float, raw.cast(), count, shape, guard)?,
        })
    }

    fn owned(
        name: impl Into<String>,
        raw: &sys::OwnedTensor,
        guard: Option<RawOutputGuard>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            tensor: LazyTensor::from_owned(raw, guard)?,
        })
    }

    fn tensor(&self) -> &Tensor {
        self.tensor.tensor()
    }

    fn into_pair(self) -> (String, Tensor) {
        (self.name, self.tensor.into_tensor())
    }

    #[cfg(test)]
    fn is_materialized(&self) -> bool {
        self.tensor.is_materialized()
    }
}

#[derive(Debug)]
enum LazyTensor {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Ready(Tensor),
    Deferred {
        data: TensorData,
        shape: Vec<usize>,
        tensor: OnceLock<Tensor>,
    },
    Raw {
        data: RawTensorData,
        shape: Vec<usize>,
        tensor: OnceLock<Tensor>,
        _guard: RawOutputGuard,
    },
    Offsets {
        data: OffsetTensorData,
        tensor: OnceLock<Tensor>,
        _guard: RawOutputGuard,
    },
}

impl LazyTensor {
    fn new(data: TensorData, shape: Vec<usize>) -> Result<Self> {
        validate_tensor_len(data.elem_count(), &shape)?;
        Ok(Self::Deferred {
            data,
            shape,
            tensor: OnceLock::new(),
        })
    }

    fn from_owned(raw: &sys::OwnedTensor, guard: Option<RawOutputGuard>) -> Result<Self> {
        let data_type = data_type_from_executor_i32(raw.data_type)?;
        let shape = shape(raw)?;
        match guard {
            Some(guard) => Self::from_raw_parts(
                RawTensorData::new(data_type, raw.data.cast_const(), raw.byte_len)?,
                shape,
                guard,
            ),
            None => {
                let bytes = bytes(raw)?;
                let data = TensorData::from_bytes(data_type, bytes)?;
                Self::new(data, shape)
            }
        }
    }

    fn from_typed_raw(
        data_type: DataType,
        raw: *const std::ffi::c_void,
        count: usize,
        shape: Vec<usize>,
        guard: Option<RawOutputGuard>,
    ) -> Result<Self> {
        let byte_len = count
            .checked_mul(data_type_size(data_type)?)
            .ok_or_else(|| {
                Error::InvalidArgument("TensorRT-LLM Executor output byte length overflowed".into())
            })?;
        match guard {
            Some(guard) => {
                Self::from_raw_parts(RawTensorData::new(data_type, raw, byte_len)?, shape, guard)
            }
            None => {
                if count > 0 && raw.is_null() {
                    return Err(Error::InvalidArgument(
                        "TensorRT-LLM Executor result pointer is null".into(),
                    ));
                }
                let bytes = if byte_len == 0 {
                    &[]
                } else {
                    unsafe { slice::from_raw_parts(raw.cast::<u8>(), byte_len) }
                };
                Self::new(TensorData::from_bytes(data_type, bytes)?, shape)
            }
        }
    }

    fn from_raw_parts(
        data: RawTensorData,
        shape: Vec<usize>,
        guard: RawOutputGuard,
    ) -> Result<Self> {
        validate_tensor_len(data.elem_count()?, &shape)?;
        Ok(Self::Raw {
            data,
            shape,
            tensor: OnceLock::new(),
            _guard: guard,
        })
    }

    fn from_offsets(
        offsets: *const usize,
        sequence_count: usize,
        data_count: usize,
        guard: RawOutputGuard,
    ) -> Result<Self> {
        Ok(Self::Offsets {
            data: OffsetTensorData::new(offsets, sequence_count, data_count)?,
            tensor: OnceLock::new(),
            _guard: guard,
        })
    }

    fn tensor(&self) -> &Tensor {
        match self {
            Self::Ready(tensor) => tensor,
            Self::Deferred {
                data,
                shape,
                tensor,
            } => tensor.get_or_init(|| data.to_tensor(shape.clone())),
            Self::Raw {
                data,
                shape,
                tensor,
                ..
            } => tensor.get_or_init(|| data.to_tensor(shape.clone())),
            Self::Offsets { data, tensor, .. } => tensor.get_or_init(|| data.to_tensor()),
        }
    }

    fn into_tensor(self) -> Tensor {
        match self {
            Self::Ready(tensor) => tensor,
            Self::Deferred {
                data,
                shape,
                tensor,
            } => tensor
                .into_inner()
                .unwrap_or_else(|| data.into_tensor(shape)),
            Self::Raw {
                data,
                shape,
                tensor,
                ..
            } => tensor
                .into_inner()
                .unwrap_or_else(|| data.into_tensor(shape)),
            Self::Offsets { data, tensor, .. } => {
                tensor.into_inner().unwrap_or_else(|| data.into_tensor())
            }
        }
    }

    #[cfg(test)]
    fn is_materialized(&self) -> bool {
        match self {
            Self::Ready(_) => true,
            Self::Deferred { tensor, .. } => tensor.get().is_some(),
            Self::Raw { tensor, .. } => tensor.get().is_some(),
            Self::Offsets { tensor, .. } => tensor.get().is_some(),
        }
    }
}

#[derive(Debug)]
struct RawTensorData {
    data_type: DataType,
    ptr: usize,
    byte_len: usize,
}

impl RawTensorData {
    fn new(data_type: DataType, raw: *const std::ffi::c_void, byte_len: usize) -> Result<Self> {
        data_type_size(data_type)?;
        if byte_len > 0 && raw.is_null() {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM Executor tensor data pointer is null".into(),
            ));
        }
        Ok(Self {
            data_type,
            ptr: raw as usize,
            byte_len,
        })
    }

    fn elem_count(&self) -> Result<usize> {
        let size = data_type_size(self.data_type)?;
        if !self.byte_len.is_multiple_of(size) {
            return Err(Error::InvalidArgument(format!(
                "{:?} tensor byte length is not divisible by {size}",
                self.data_type
            )));
        }
        Ok(self.byte_len / size)
    }

    fn bytes(&self) -> &[u8] {
        if self.byte_len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr as *const u8, self.byte_len) }
        }
    }

    fn to_tensor(&self, shape: Vec<usize>) -> Tensor {
        TensorData::from_bytes(self.data_type, self.bytes())
            .expect("TensorRT-LLM raw output was validated")
            .into_tensor(shape)
    }

    fn into_tensor(self, shape: Vec<usize>) -> Tensor {
        self.to_tensor(shape)
    }
}

#[derive(Debug)]
struct OffsetTensorData {
    ptr: usize,
    sequence_count: usize,
    data_count: usize,
}

impl OffsetTensorData {
    fn new(offsets: *const usize, sequence_count: usize, data_count: usize) -> Result<Self> {
        if sequence_count == 0 {
            i64::try_from(data_count).map_err(|_| {
                Error::InvalidArgument(
                    "TensorRT-LLM Executor output length cannot fit in i64".into(),
                )
            })?;
            return Ok(Self {
                ptr: 0,
                sequence_count,
                data_count,
            });
        }
        if offsets.is_null() {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM Executor output sequence offsets are null".into(),
            ));
        }
        let offsets = unsafe { slice::from_raw_parts(offsets, sequence_count + 1) };
        if offsets.first() != Some(&0) || offsets.last() != Some(&data_count) {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM Executor output sequence offsets are invalid".into(),
            ));
        }
        for offset in offsets {
            i64::try_from(*offset).map_err(|_| {
                Error::InvalidArgument(
                    "TensorRT-LLM Executor output offset cannot fit in i64".into(),
                )
            })?;
        }
        Ok(Self {
            ptr: offsets.as_ptr() as usize,
            sequence_count,
            data_count,
        })
    }

    fn len(&self) -> usize {
        if self.sequence_count == 0 {
            2
        } else {
            self.sequence_count + 1
        }
    }

    fn to_vec(&self) -> Vec<i64> {
        if self.sequence_count == 0 {
            return vec![
                0,
                i64::try_from(self.data_count)
                    .expect("TensorRT-LLM offset data count was validated"),
            ];
        }
        unsafe { slice::from_raw_parts(self.ptr as *const usize, self.sequence_count + 1) }
            .iter()
            .map(|offset| i64::try_from(*offset).expect("TensorRT-LLM offsets were validated"))
            .collect()
    }

    fn to_tensor(&self) -> Tensor {
        Tensor::from_vec(self.to_vec(), vec![self.len()], &Device::Cpu)
            .expect("TensorRT-LLM output offset shape was validated")
    }

    fn into_tensor(self) -> Tensor {
        self.to_tensor()
    }
}

#[derive(Debug)]
enum TensorData {
    Uint8(Vec<u8>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Bf16(Vec<bf16>),
    Float16(Vec<f16>),
    Float32(Vec<f32>),
}

impl TensorData {
    fn from_bytes(data_type: DataType, bytes: &[u8]) -> Result<Self> {
        Ok(match data_type {
            DataType::Uint8 => Self::Uint8(bytes.to_vec()),
            DataType::Int32 => Self::Int32(read_i32(bytes)?),
            DataType::Int64 => Self::Int64(read_i64(bytes)?),
            DataType::Bf16 => Self::Bf16(read_bf16(bytes)?),
            DataType::Half => Self::Float16(read_f16(bytes)?),
            DataType::Float => Self::Float32(read_f32(bytes)?),
            dtype => {
                return Err(Error::InvalidArgument(format!(
                    "TensorRT-LLM Executor output dtype {dtype:?} is not supported as a Candle tensor"
                )));
            }
        })
    }

    fn elem_count(&self) -> usize {
        match self {
            Self::Uint8(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Bf16(values) => values.len(),
            Self::Float16(values) => values.len(),
            Self::Float32(values) => values.len(),
        }
    }

    fn to_tensor(&self, shape: Vec<usize>) -> Tensor {
        match self {
            Self::Uint8(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
            Self::Int32(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
            Self::Int64(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
            Self::Bf16(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
            Self::Float16(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
            Self::Float32(values) => Tensor::from_vec(values.clone(), shape, &Device::Cpu),
        }
        .expect("TensorRT-LLM output tensor shape was validated")
    }

    fn into_tensor(self, shape: Vec<usize>) -> Tensor {
        match self {
            Self::Uint8(values) => Tensor::from_vec(values, shape, &Device::Cpu),
            Self::Int32(values) => Tensor::from_vec(values, shape, &Device::Cpu),
            Self::Int64(values) => Tensor::from_vec(values, shape, &Device::Cpu),
            Self::Bf16(values) => Tensor::from_vec(values, shape, &Device::Cpu),
            Self::Float16(values) => Tensor::from_vec(values, shape, &Device::Cpu),
            Self::Float32(values) => Tensor::from_vec(values, shape, &Device::Cpu),
        }
        .expect("TensorRT-LLM output tensor shape was validated")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    NotFinished,
    EndId,
    StopWords,
    Length,
    TimedOut,
    Cancelled,
    Unknown(i32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputContextPhaseParams {
    pub first_gen_tokens: Vec<i32>,
    pub request_id: u64,
    pub serialized_state: Vec<u8>,
    pub draft_tokens: Option<Vec<i32>>,
    pub context_data_parallel_rank: Option<i32>,
    pub disaggregated_info_endpoint: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculativeDecodingFastLogitsInfo {
    pub draft_request_id: u64,
    pub draft_participant_id: i32,
}

pub(crate) fn from_sys(raw: &sys::Output, guard: Option<RawOutputGuard>) -> Result<ExecutorOutput> {
    let mut outputs = Vec::new();

    if raw.token_count > 0 {
        outputs.push(TensorOutput::int32(
            "token_ids".to_owned(),
            raw.token_ids,
            raw.token_count,
            vec![raw.token_count],
            guard.clone(),
        )?);
        outputs.push(TensorOutput::offsets(
            "token_offsets".to_owned(),
            raw.sequence_offsets,
            raw.sequence_count,
            raw.token_count,
            guard.clone(),
        )?);
    }

    if raw.cum_log_probs_count > 0 {
        outputs.push(TensorOutput::float32(
            "cum_log_probs".to_owned(),
            raw.cum_log_probs,
            raw.cum_log_probs_count,
            vec![raw.cum_log_probs_count],
            guard.clone(),
        )?);
    }

    if raw.log_prob_count > 0 {
        outputs.push(TensorOutput::float32(
            "log_probs".to_owned(),
            raw.log_probs,
            raw.log_prob_count,
            vec![raw.log_prob_count],
            guard.clone(),
        )?);
        outputs.push(TensorOutput::offsets(
            "log_prob_offsets".to_owned(),
            raw.log_prob_offsets,
            raw.log_prob_sequence_count,
            raw.log_prob_count,
            guard.clone(),
        )?);
    }

    if raw.has_context_logits != 0 {
        outputs.push(TensorOutput::owned(
            "context_logits".to_owned(),
            &raw.context_logits,
            guard.clone(),
        )?);
    }
    if raw.has_generation_logits != 0 {
        outputs.push(TensorOutput::owned(
            "generation_logits".to_owned(),
            &raw.generation_logits,
            guard.clone(),
        )?);
    }
    if raw.has_encoder_output != 0 {
        outputs.push(TensorOutput::owned(
            "encoder_output".to_owned(),
            &raw.encoder_output,
            guard.clone(),
        )?);
    }

    outputs.extend(named_tensors(
        raw.named_tensors,
        raw.named_tensor_count,
        guard,
    )?);

    Ok(ExecutorOutput {
        tensors: outputs,
        is_final: raw.is_final != 0,
        finish_reasons: finish_reasons(raw.finish_reasons, raw.finish_reason_count)?,
        context_phase_params: if raw.has_context_phase_params != 0 {
            Some(context_phase_params(&raw.context_phase_params)?)
        } else {
            None
        },
        spec_dec_fast_logits_info: if raw.has_spec_dec_fast_logits_info != 0 {
            Some(SpeculativeDecodingFastLogitsInfo {
                draft_request_id: raw.spec_dec_fast_logits_info.draft_request_id,
                draft_participant_id: raw.spec_dec_fast_logits_info.draft_participant_id,
            })
        } else {
            None
        },
        sequence_index: raw.sequence_index,
        is_sequence_final: raw.is_sequence_final != 0,
    })
}

fn finish_reasons(raw: *const i32, count: usize) -> Result<Vec<FinishReason>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if raw.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor finish_reasons pointer is null".into(),
        ));
    }
    unsafe { slice::from_raw_parts(raw, count) }
        .iter()
        .map(|reason| {
            Ok(match *reason {
                0 => FinishReason::NotFinished,
                1 => FinishReason::EndId,
                2 => FinishReason::StopWords,
                3 => FinishReason::Length,
                4 => FinishReason::TimedOut,
                5 => FinishReason::Cancelled,
                value => FinishReason::Unknown(value),
            })
        })
        .collect()
}

fn context_phase_params(raw: &sys::OwnedContextPhaseParams) -> Result<OutputContextPhaseParams> {
    Ok(OutputContextPhaseParams {
        first_gen_tokens: i32_vec(raw.first_gen_tokens, raw.first_gen_token_count)?,
        request_id: raw.request_id,
        serialized_state: u8_vec(raw.serialized_state, raw.serialized_state_len)?,
        draft_tokens: if raw.draft_token_count > 0 {
            Some(i32_vec(raw.draft_tokens, raw.draft_token_count)?)
        } else {
            None
        },
        context_data_parallel_rank: (raw.has_ctx_dp_rank != 0).then_some(raw.ctx_dp_rank),
        disaggregated_info_endpoint: if raw.disagg_info_endpoint.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(raw.disagg_info_endpoint) }
                    .to_string_lossy()
                    .into_owned(),
            )
        },
    })
}

fn i32_vec(raw: *const i32, count: usize) -> Result<Vec<i32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if raw.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor i32 result pointer is null".into(),
        ));
    }
    Ok(unsafe { slice::from_raw_parts(raw, count) }.to_vec())
}

fn u8_vec(raw: *const u8, count: usize) -> Result<Vec<u8>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if raw.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor byte result pointer is null".into(),
        ));
    }
    Ok(unsafe { slice::from_raw_parts(raw, count) }.to_vec())
}

fn named_tensors(
    raw: *const sys::NamedTensor,
    count: usize,
    guard: Option<RawOutputGuard>,
) -> Result<Vec<TensorOutput>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if raw.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor named tensor pointer is null".into(),
        ));
    }
    let raw = unsafe { slice::from_raw_parts(raw, count) };
    let mut tensors = Vec::with_capacity(raw.len());
    for named in raw {
        if named.name.is_null() {
            return Err(Error::InvalidArgument(
                "TensorRT-LLM Executor named tensor name is null".into(),
            ));
        }
        let name = unsafe { CStr::from_ptr(named.name) }
            .to_string_lossy()
            .into_owned();
        let name = name.strip_prefix("context_").unwrap_or(&name);
        tensors.push(TensorOutput::owned(
            name.to_owned(),
            &named.tensor,
            guard.clone(),
        )?);
    }
    Ok(tensors)
}

fn output_offsets(
    offsets: *const usize,
    sequence_count: usize,
    data_count: usize,
) -> Result<Vec<i64>> {
    if sequence_count == 0 {
        return Ok(vec![
            0,
            i64::try_from(data_count).map_err(|_| {
                Error::InvalidArgument(
                    "TensorRT-LLM Executor output length cannot fit in i64".into(),
                )
            })?,
        ]);
    }
    if offsets.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor output sequence offsets are null".into(),
        ));
    }
    let offsets = unsafe { slice::from_raw_parts(offsets, sequence_count + 1) };
    if offsets.first() != Some(&0) || offsets.last() != Some(&data_count) {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor output sequence offsets are invalid".into(),
        ));
    }
    offsets
        .iter()
        .map(|offset| {
            i64::try_from(*offset).map_err(|_| {
                Error::InvalidArgument(
                    "TensorRT-LLM Executor output offset cannot fit in i64".into(),
                )
            })
        })
        .collect()
}

#[cfg(test)]
fn tensor_from_owned(raw: &sys::OwnedTensor) -> Result<Tensor> {
    Ok(LazyTensor::from_owned(raw, None)?.into_tensor())
}

fn shape(raw: &sys::OwnedTensor) -> Result<Vec<usize>> {
    if raw.rank == 0 {
        return Ok(Vec::new());
    }
    if raw.shape.is_null() {
        return Err(Error::InvalidArgument(
            "TensorRT-LLM Executor tensor shape pointer is null".into(),
        ));
    }
    unsafe { slice::from_raw_parts(raw.shape, raw.rank) }
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                Error::InvalidArgument(format!(
                    "TensorRT-LLM Executor output dimension {dim} cannot fit in usize"
                ))
            })
        })
        .collect()
}

fn bytes(raw: &sys::OwnedTensor) -> Result<&[u8]> {
    if raw.byte_len == 0 {
        Ok(&[])
    } else if raw.data.is_null() {
        Err(Error::InvalidArgument(
            "TensorRT-LLM Executor tensor data pointer is null".into(),
        ))
    } else {
        Ok(unsafe { slice::from_raw_parts(raw.data.cast::<u8>(), raw.byte_len) })
    }
}

fn validate_tensor_len(elem_count: usize, shape: &[usize]) -> Result<()> {
    let expected = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::InvalidArgument("TensorRT-LLM Executor output shape overflowed".into())
        })
    })?;
    if elem_count != expected {
        return Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor output element count mismatch: shape expects {expected}, buffer has {elem_count}"
        )));
    }
    Ok(())
}

fn data_type_size(data_type: DataType) -> Result<usize> {
    match data_type {
        DataType::Uint8 => Ok(1),
        DataType::Int32 | DataType::Float => Ok(4),
        DataType::Int64 => Ok(8),
        DataType::Bf16 | DataType::Half => Ok(2),
        dtype => Err(Error::InvalidArgument(format!(
            "TensorRT-LLM Executor output dtype {dtype:?} is not supported as a Candle tensor"
        ))),
    }
}

fn read_bf16(bytes: &[u8]) -> Result<Vec<bf16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Error::InvalidArgument(
            "BF16 tensor byte length is not divisible by 2".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn read_f16(bytes: &[u8]) -> Result<Vec<f16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Error::InvalidArgument(
            "FP16 tensor byte length is not divisible by 2".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn read_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::InvalidArgument(
            "FP32 tensor byte length is not divisible by 4".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::InvalidArgument(
            "INT32 tensor byte length is not divisible by 4".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_i64(bytes: &[u8]) -> Result<Vec<i64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::InvalidArgument(
            "INT64 tensor byte length is not divisible by 8".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_ne_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

pub(crate) fn data_type_from_executor_i32(data_type: i32) -> Result<DataType> {
    match data_type {
        0 => Ok(DataType::Bool),
        1 => Ok(DataType::Uint8),
        2 => Ok(DataType::Int8),
        3 => Ok(DataType::Int32),
        4 => Ok(DataType::Int64),
        5 => Ok(DataType::Bf16),
        6 => Ok(DataType::Fp8),
        7 => Ok(DataType::Half),
        8 => Ok(DataType::Float),
        dtype => Err(Error::InvalidArgument(format!(
            "unsupported TensorRT-LLM Executor dtype id {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn output_offsets_use_sequence_offsets() {
        let offsets = [0usize, 2, 5];
        let offsets = output_offsets(offsets.as_ptr(), 2, 5).unwrap();
        assert_eq!(offsets, vec![0, 2, 5]);
    }

    #[test]
    fn output_offsets_reject_bad_offsets() {
        let offsets = [0usize, 4];
        let error = output_offsets(offsets.as_ptr(), 1, 3).unwrap_err();
        assert!(error.to_string().contains("offsets are invalid"));
    }

    #[test]
    fn finish_reasons_map_executor_values() {
        let raw = [0, 1, 2, 3, 4, 5, 99];
        let reasons = finish_reasons(raw.as_ptr(), raw.len()).unwrap();
        assert_eq!(
            reasons,
            vec![
                FinishReason::NotFinished,
                FinishReason::EndId,
                FinishReason::StopWords,
                FinishReason::Length,
                FinishReason::TimedOut,
                FinishReason::Cancelled,
                FinishReason::Unknown(99),
            ]
        );
    }

    #[test]
    fn context_phase_params_copy_owned_values() {
        let first_gen_tokens = [10, 11];
        let serialized_state = [1u8, 2, 3];
        let draft_tokens = [20, 21];
        let endpoint = CString::new("localhost:1234").unwrap();
        let raw = sys::OwnedContextPhaseParams {
            first_gen_tokens: first_gen_tokens.as_ptr().cast_mut(),
            first_gen_token_count: first_gen_tokens.len(),
            request_id: 7,
            serialized_state: serialized_state.as_ptr().cast_mut(),
            serialized_state_len: serialized_state.len(),
            draft_tokens: draft_tokens.as_ptr().cast_mut(),
            draft_token_count: draft_tokens.len(),
            has_ctx_dp_rank: 1,
            ctx_dp_rank: 3,
            disagg_info_endpoint: endpoint.as_ptr().cast_mut(),
        };

        let params = context_phase_params(&raw).unwrap();

        assert_eq!(params.first_gen_tokens, vec![10, 11]);
        assert_eq!(params.request_id, 7);
        assert_eq!(params.serialized_state, vec![1, 2, 3]);
        assert_eq!(params.draft_tokens, Some(vec![20, 21]));
        assert_eq!(params.context_data_parallel_rank, Some(3));
        assert_eq!(
            params.disaggregated_info_endpoint.as_deref(),
            Some("localhost:1234")
        );
    }

    #[test]
    fn from_sys_exposes_executor_result_metadata() {
        let mut finish_reasons = [1];
        let raw = sys::Output {
            is_final: 1,
            has_spec_dec_fast_logits_info: 1,
            spec_dec_fast_logits_info: sys::SpeculativeDecodingFastLogitsInfo {
                draft_request_id: 42,
                draft_participant_id: 9,
            },
            finish_reasons: finish_reasons.as_mut_ptr(),
            finish_reason_count: finish_reasons.len(),
            sequence_index: 2,
            is_sequence_final: 1,
            ..sys::Output::default()
        };

        let output = from_sys(&raw, None).unwrap();

        assert!(output.is_final());
        assert_eq!(output.finish_reasons(), &[FinishReason::EndId]);
        assert_eq!(
            output.speculative_decoding_fast_logits_info(),
            Some(SpeculativeDecodingFastLogitsInfo {
                draft_request_id: 42,
                draft_participant_id: 9,
            })
        );
        assert_eq!(output.sequence_index(), 2);
        assert!(output.is_sequence_final());
    }

    #[test]
    fn get_and_index_include_additional_outputs() {
        let output = ExecutorOutput {
            tensors: vec![
                TensorOutput::ready(
                    "token_ids".to_owned(),
                    Tensor::from_vec(vec![1i32], 1, &Device::Cpu).unwrap(),
                ),
                TensorOutput::ready(
                    "marked_model_output".to_owned(),
                    Tensor::from_vec(vec![1f32], 1, &Device::Cpu).unwrap(),
                ),
                TensorOutput::ready(
                    "generation_score".to_owned(),
                    Tensor::from_vec(vec![2f32], 1, &Device::Cpu).unwrap(),
                ),
            ],
            is_final: true,
            finish_reasons: Vec::new(),
            context_phase_params: None,
            spec_dec_fast_logits_info: None,
            sequence_index: 0,
            is_sequence_final: true,
        };

        assert_eq!(output["token_ids"].dims(), &[1]);
        assert_eq!(output["marked_model_output"].dims(), &[1]);
        assert_eq!(output.get("generation_score").unwrap().dims(), &[1]);
    }

    #[test]
    fn iteration_includes_every_tensor_output() {
        let output = ExecutorOutput {
            tensors: vec![
                TensorOutput::ready(
                    "token_ids".to_owned(),
                    Tensor::from_vec(vec![1i32], 1, &Device::Cpu).unwrap(),
                ),
                TensorOutput::ready(
                    "marked_model_output".to_owned(),
                    Tensor::from_vec(vec![1f32], 1, &Device::Cpu).unwrap(),
                ),
                TensorOutput::ready(
                    "generation_score".to_owned(),
                    Tensor::from_vec(vec![2f32], 1, &Device::Cpu).unwrap(),
                ),
            ],
            is_final: true,
            finish_reasons: Vec::new(),
            context_phase_params: None,
            spec_dec_fast_logits_info: None,
            sequence_index: 0,
            is_sequence_final: true,
        };

        let names = output.iter().map(|(name, _)| name).collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["token_ids", "marked_model_output", "generation_score"]
        );

        let names = output.into_iter().map(|(name, _)| name).collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["token_ids", "marked_model_output", "generation_score"]
        );
    }

    #[test]
    fn consuming_iterator_reports_exact_remaining_length() {
        let output = ExecutorOutput {
            tensors: vec![
                TensorOutput::ready(
                    "token_ids".to_owned(),
                    Tensor::from_vec(vec![1i32], 1, &Device::Cpu).unwrap(),
                ),
                TensorOutput::ready(
                    "marked_model_output".to_owned(),
                    Tensor::from_vec(vec![1f32], 1, &Device::Cpu).unwrap(),
                ),
            ],
            is_final: true,
            finish_reasons: Vec::new(),
            context_phase_params: None,
            spec_dec_fast_logits_info: None,
            sequence_index: 0,
            is_sequence_final: true,
        };

        let mut iter = output.into_iter();
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next().unwrap().0, "token_ids");
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next().unwrap().0, "marked_model_output");
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn named_tensors_normalize_context_outputs_without_hiding_generation_outputs() {
        let context_name = CString::new("context_marked_model_output").unwrap();
        let generation_name = CString::new("generation_marked_model_output").unwrap();
        let fallback_name = CString::new("custom_output").unwrap();
        let context_value = 1f32;
        let generation_value = 2f32;
        let fallback_value = 3f32;
        let mut context_shape = [1i64];
        let mut generation_shape = [1i64];
        let mut fallback_shape = [1i64];
        let raw = [
            sys::NamedTensor {
                name: context_name.as_ptr().cast_mut(),
                tensor: sys::OwnedTensor {
                    data: (&context_value as *const f32).cast_mut().cast(),
                    byte_len: std::mem::size_of::<f32>(),
                    shape: context_shape.as_mut_ptr(),
                    rank: context_shape.len(),
                    data_type: 8,
                },
            },
            sys::NamedTensor {
                name: generation_name.as_ptr().cast_mut(),
                tensor: sys::OwnedTensor {
                    data: (&generation_value as *const f32).cast_mut().cast(),
                    byte_len: std::mem::size_of::<f32>(),
                    shape: generation_shape.as_mut_ptr(),
                    rank: generation_shape.len(),
                    data_type: 8,
                },
            },
            sys::NamedTensor {
                name: fallback_name.as_ptr().cast_mut(),
                tensor: sys::OwnedTensor {
                    data: (&fallback_value as *const f32).cast_mut().cast(),
                    byte_len: std::mem::size_of::<f32>(),
                    shape: fallback_shape.as_mut_ptr(),
                    rank: fallback_shape.len(),
                    data_type: 8,
                },
            },
        ];

        let outputs = named_tensors(raw.as_ptr(), raw.len(), None).unwrap();

        assert_eq!(outputs[0].name, "marked_model_output");
        assert_eq!(outputs[1].name, "generation_marked_model_output");
        assert_eq!(outputs[2].name, "custom_output");
        assert!(!outputs[0].is_materialized());
        assert_eq!(outputs[0].tensor().dims(), &[1]);
        assert!(outputs[0].is_materialized());
    }

    #[test]
    fn from_sys_defers_standard_tensor_materialization() {
        let mut token_ids = [1, 2, 3];
        let raw = sys::Output {
            token_ids: token_ids.as_mut_ptr(),
            token_count: token_ids.len(),
            ..sys::Output::default()
        };

        let output = from_sys(&raw, None).unwrap();

        assert_eq!(output.tensors.len(), 2);
        assert_eq!(output.tensors[0].name, "token_ids");
        assert_eq!(output.tensors[1].name, "token_offsets");
        assert!(!output.tensors[0].is_materialized());
        assert_eq!(output["token_ids"].dims(), &[3]);
        assert!(output.tensors[0].is_materialized());
        assert!(!output.tensors[1].is_materialized());
    }

    #[test]
    fn from_sys_with_guard_defers_raw_copy_until_access() {
        let mut token_ids = [1, 2, 3];
        let raw = sys::Output {
            token_ids: token_ids.as_mut_ptr(),
            token_count: token_ids.len(),
            ..sys::Output::default()
        };
        let output = from_sys(&raw, Some(RawOutputGuard::new(Arc::new(())))).unwrap();

        assert!(!output.tensors[0].is_materialized());
        unsafe { token_ids.as_mut_ptr().write(9) };

        assert_eq!(output["token_ids"].to_vec1::<i32>().unwrap(), vec![9, 2, 3]);
        assert!(output.tensors[0].is_materialized());
    }

    #[test]
    fn tensor_conversion_rejects_invalid_executor_buffers() {
        let mut shape = [1i64];
        let bytes = [0u8; 3];
        let raw = sys::OwnedTensor {
            data: bytes.as_ptr().cast_mut().cast(),
            byte_len: bytes.len(),
            shape: shape.as_mut_ptr(),
            rank: shape.len(),
            data_type: 8,
        };
        assert!(tensor_from_owned(&raw).is_err());

        let mut bad_shape = [-1i64];
        let bytes = [0u8; 4];
        let raw = sys::OwnedTensor {
            data: bytes.as_ptr().cast_mut().cast(),
            byte_len: bytes.len(),
            shape: bad_shape.as_mut_ptr(),
            rank: bad_shape.len(),
            data_type: 8,
        };
        assert!(tensor_from_owned(&raw).is_err());

        let raw = sys::OwnedTensor {
            data: bytes.as_ptr().cast_mut().cast(),
            byte_len: bytes.len(),
            shape: shape.as_mut_ptr(),
            rank: shape.len(),
            data_type: 0,
        };
        assert!(tensor_from_owned(&raw).is_err());

        let raw = sys::OwnedTensor {
            data: bytes.as_ptr().cast_mut().cast(),
            byte_len: bytes.len(),
            shape: std::ptr::null_mut(),
            rank: 1,
            data_type: 8,
        };
        assert!(tensor_from_owned(&raw).is_err());

        assert!(data_type_from_executor_i32(99).is_err());
        assert!(read_bf16(&[0]).is_err());
        assert!(read_f16(&[0]).is_err());
        assert!(read_i32(&[0, 1, 2]).is_err());
        assert!(read_i64(&[0, 1, 2, 3]).is_err());
    }

    #[test]
    fn output_helpers_reject_null_pointers_with_data() {
        assert!(finish_reasons(std::ptr::null(), 1).is_err());
        assert!(i32_vec(std::ptr::null(), 1).is_err());
        assert!(u8_vec(std::ptr::null(), 1).is_err());
        assert!(named_tensors(std::ptr::null(), 1, None).is_err());
        assert!(output_offsets(std::ptr::null(), 1, 1).is_err());

        let mut raw = [sys::NamedTensor {
            name: std::ptr::null_mut(),
            tensor: sys::OwnedTensor::default(),
        }];
        assert!(named_tensors(raw.as_mut_ptr(), raw.len(), None).is_err());
    }

    #[test]
    fn empty_output_accessors_are_stable() {
        let output = ExecutorOutput {
            tensors: Vec::new(),
            is_final: false,
            finish_reasons: Vec::new(),
            context_phase_params: None,
            spec_dec_fast_logits_info: None,
            sequence_index: -1,
            is_sequence_final: false,
        };

        assert!(output.is_empty());
        assert!(output.iter().next().is_none());
        assert!(output.context_phase().is_none());
        assert!(!output.is_final());
        assert!(!output.is_sequence_final());
        assert_eq!(output.sequence_index(), -1);
    }
}
