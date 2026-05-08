//! Candle tensor adapters for TensorRT sessions.
//!
//! This module intentionally lives above `tensorrt-sys`: it converts Candle
//! tensors into the existing named TensorRT binding API, then delegates all
//! execution to [`Session`](crate::Session). The input and output tensors are
//! `candle_core::Tensor`; this module does not wrap or replace Candle's tensor
//! type.

#[cfg(feature = "cuda")]
use crate::session::DeviceOutputPlan;
use crate::{CudaStream, DataType, Dims, Error, Result, Session, TensorShape};
#[cfg(feature = "cuda")]
use crate::{DeviceInputTensor, DeviceOutputTensor, tensor_byte_len};
#[cfg(feature = "cuda")]
use candle_core::{CudaStorage, Device, Storage};
use candle_core::{DType, Tensor};
#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::ops::Index;

/// Named Candle inputs for a TensorRT session run.
#[derive(Clone, Debug, Default)]
pub struct InputTensors<'a> {
    inputs: Vec<(String, &'a Tensor)>,
}

impl<'a> InputTensors<'a> {
    pub fn new<N, I>(inputs: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, &'a Tensor)>,
    {
        Self {
            inputs: inputs
                .into_iter()
                .map(|(name, tensor)| (name.into(), tensor))
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&'a Tensor> {
        self.inputs
            .iter()
            .find(|(input_name, _)| input_name == name)
            .map(|(_, tensor)| *tensor)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &'a Tensor)> + '_ {
        self.inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), *tensor))
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn first(&self) -> Option<(&str, &'a Tensor)> {
        self.inputs
            .first()
            .map(|(name, tensor)| (name.as_str(), *tensor))
    }

    pub(crate) fn shape_infos(&self) -> Result<Vec<TensorShape<'_>>> {
        self.inputs
            .iter()
            .map(|(name, tensor)| Ok(TensorShape::new(name, candle_dims(name, tensor)?)))
            .collect()
    }
}

/// Named Candle outputs written by a TensorRT session run.
#[derive(Clone, Debug, Default)]
pub struct OutputTensors {
    outputs: Vec<(String, Tensor)>,
}

impl OutputTensors {
    pub fn new<N, I>(outputs: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, Tensor)>,
    {
        Self {
            outputs: outputs
                .into_iter()
                .map(|(name, tensor)| (name.into(), tensor))
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.outputs
            .iter()
            .find(|(output_name, _)| output_name == name)
            .map(|(_, tensor)| tensor)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.outputs
            .iter_mut()
            .find(|(output_name, _)| output_name == name)
            .map(|(_, tensor)| tensor)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.outputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub fn into_vec(self) -> Vec<(String, Tensor)> {
        self.outputs
    }
}

impl Index<&str> for OutputTensors {
    type Output = Tensor;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("TensorRT output {name:?} was not returned"))
    }
}

impl IntoIterator for OutputTensors {
    type Item = (String, Tensor);
    type IntoIter = std::vec::IntoIter<(String, Tensor)>;

    fn into_iter(self) -> Self::IntoIter {
        self.outputs.into_iter()
    }
}

pub(crate) fn run_session(
    session: &Session,
    inputs: &InputTensors<'_>,
    outputs: &mut OutputTensors,
    stream: CudaStream<'_>,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        run_session_with_cuda_outputs(session, inputs, outputs, stream)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (session, outputs, stream);
        Err(Error::InvalidShape {
            tensor: inputs
                .first()
                .map(|(name, _)| name.to_owned())
                .unwrap_or_else(|| "<inputs>".to_owned()),
            reason: "Candle session run requires the `cuda` feature and CUDA input tensors"
                .to_owned(),
        })
    }
}

#[cfg(feature = "cuda")]
fn run_session_with_cuda_outputs(
    session: &Session,
    inputs: &InputTensors<'_>,
    outputs: &mut OutputTensors,
    stream: CudaStream<'_>,
) -> Result<()> {
    let output_device = cuda_output_device(inputs)?;
    validate_stream_device(&output_device, stream)?;
    if outputs.outputs.is_empty() {
        return Err(Error::InvalidShape {
            tensor: "<outputs>".to_owned(),
            reason: "Candle session run requires at least one preallocated output tensor"
                .to_owned(),
        });
    }
    let keepalive = run_keepalive(inputs, outputs);

    let device_inputs = inputs
        .iter()
        .map(|(name, tensor)| {
            let prepared = prepare_cuda_input(name, tensor)?;
            Ok(DeviceInputTensor::new(
                name,
                prepared.data_type,
                prepared.shape,
                prepared.ptr,
                prepared.bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut context = session.pooled_execution_context()?;
    let output_plan = session.cached_device_output_plan(
        &mut context,
        &device_inputs,
        outputs.outputs.iter().map(|(name, _)| name.as_str()),
    )?;

    let mut device_outputs = output_plan
        .iter()
        .zip(outputs.outputs.iter_mut())
        .map(|(plan, (name, tensor))| {
            let output = prepare_cuda_output(plan, name, tensor, &output_device)?;
            Ok(DeviceOutputTensor::new(
                name,
                output.data_type,
                output.shape,
                output.ptr,
                output.bytes,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    validate_cuda_output_aliases(&device_inputs, &device_outputs)?;

    let event_device_location = output_device.location();
    let completion_event = session.completion_event(event_device_location)?;
    unsafe {
        session.enqueue_device_with_context(
            &mut context,
            &device_inputs,
            &mut device_outputs,
            stream,
        )?;
    }
    session.defer_context_until_stream_complete(
        context,
        event_device_location,
        completion_event,
        stream,
        keepalive,
    )?;

    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_stream_device(device: &Device, stream: CudaStream<'_>) -> Result<()> {
    let device_location = device.location();
    let Some(stream_location) = stream.device_location() else {
        return Err(Error::InvalidShape {
            tensor: "<cuda-stream>".to_owned(),
            reason: "stream device is unknown; use CudaStream::from_device".to_owned(),
        });
    };

    if stream_location != device_location {
        return Err(Error::InvalidShape {
            tensor: "<cuda-stream>".to_owned(),
            reason: format!(
                "stream device {stream_location:?} does not match input device {device_location:?}"
            ),
        });
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_keepalive(inputs: &InputTensors<'_>, outputs: &OutputTensors) -> Vec<Tensor> {
    inputs
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .chain(outputs.outputs.iter().map(|(_, tensor)| tensor.clone()))
        .collect()
}

#[cfg(feature = "cuda")]
struct CudaInputData {
    data_type: DataType,
    shape: Dims,
    ptr: *const c_void,
    bytes: usize,
}

#[cfg(feature = "cuda")]
struct CudaOutputData {
    data_type: DataType,
    shape: Dims,
    ptr: *mut c_void,
    bytes: usize,
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
    fn new(name: &'a str, ptr: *const c_void, bytes: usize) -> Result<Self> {
        let start = ptr as usize;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::InvalidShape {
                tensor: name.to_owned(),
                reason: "CUDA tensor pointer range overflow".to_owned(),
            })?;
        Ok(Self { name, start, end })
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[cfg(feature = "cuda")]
fn cuda_output_device(inputs: &InputTensors<'_>) -> Result<Device> {
    let Some((first_name, first_tensor)) = inputs.first() else {
        return Err(Error::InvalidShape {
            tensor: "<inputs>".to_owned(),
            reason: "Candle session run requires at least one CUDA input tensor".to_owned(),
        });
    };

    let device = first_tensor.device().clone();
    if !device.is_cuda() {
        return Err(Error::InvalidShape {
            tensor: first_name.to_owned(),
            reason: "Candle session run requires CUDA input tensors".to_owned(),
        });
    }

    for (name, tensor) in inputs.iter() {
        if !tensor.device().is_cuda() {
            return Err(Error::InvalidShape {
                tensor: name.to_owned(),
                reason: "Candle session run requires CUDA input tensors".to_owned(),
            });
        }
        if !tensor.device().same_device(&device) {
            return Err(Error::InvalidShape {
                tensor: name.to_owned(),
                reason: "all Candle input tensors must be on the same CUDA device".to_owned(),
            });
        }
    }

    Ok(device)
}

#[cfg(feature = "cuda")]
fn prepare_cuda_output(
    plan: &DeviceOutputPlan,
    output_name: &str,
    tensor: &mut Tensor,
    output_device: &Device,
) -> Result<CudaOutputData> {
    if output_name != plan.name {
        return Err(Error::InvalidShape {
            tensor: output_name.to_owned(),
            reason: format!("expected output binding {:?}", plan.name),
        });
    }

    if !tensor.device().is_cuda() {
        return Err(Error::InvalidShape {
            tensor: output_name.to_owned(),
            reason: "Candle session run requires CUDA output tensors".to_owned(),
        });
    }
    if !tensor.device().same_device(output_device) {
        return Err(Error::InvalidShape {
            tensor: output_name.to_owned(),
            reason: "output tensor must be on the same CUDA device as the inputs and stream"
                .to_owned(),
        });
    }

    let data_type = data_type_from_candle(tensor.dtype())?;
    if data_type != plan.data_type {
        return Err(Error::TypeMismatch {
            tensor: output_name.to_owned(),
            expected: plan.data_type,
            actual: data_type,
        });
    }

    let shape = candle_dims(output_name, tensor)?;
    if shape != plan.shape {
        return Err(Error::InvalidShape {
            tensor: output_name.to_owned(),
            reason: format!(
                "expected output shape {:?}, got {:?}",
                plan.shape.as_slice(),
                shape.as_slice()
            ),
        });
    }

    let ptr = cuda_tensor_mut_ptr(output_name, tensor, data_type)?;

    Ok(CudaOutputData {
        data_type,
        shape,
        ptr,
        bytes: plan.bytes,
    })
}

#[cfg(feature = "cuda")]
fn validate_cuda_output_aliases(
    inputs: &[DeviceInputTensor<'_>],
    outputs: &[DeviceOutputTensor<'_>],
) -> Result<()> {
    for output in outputs {
        let output_range =
            CudaMemoryRange::new(output.name, output.ptr.cast_const(), output.bytes)?;
        for input in inputs {
            let input_range = CudaMemoryRange::new(input.name, input.ptr, input.bytes)?;
            if output_range.overlaps(input_range) {
                return Err(Error::InvalidShape {
                    tensor: output_range.name.to_owned(),
                    reason: format!(
                        "output CUDA memory overlaps input tensor {:?}; preallocate a distinct output tensor",
                        input_range.name
                    ),
                });
            }
        }
    }

    for (index, output) in outputs.iter().enumerate() {
        let output_range =
            CudaMemoryRange::new(output.name, output.ptr.cast_const(), output.bytes)?;
        for other in outputs.iter().skip(index + 1) {
            let other_range =
                CudaMemoryRange::new(other.name, other.ptr.cast_const(), other.bytes)?;
            if output_range.overlaps(other_range) {
                return Err(Error::InvalidShape {
                    tensor: output_range.name.to_owned(),
                    reason: format!(
                        "output CUDA memory overlaps output tensor {:?}; each output must use distinct storage",
                        other_range.name
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn prepare_cuda_input(name: &str, input: &Tensor) -> Result<CudaInputData> {
    let data_type = data_type_from_candle(input.dtype())?;
    let shape = candle_dims(name, input)?;
    let bytes = tensor_byte_len(name, &shape, data_type)?;
    let ptr = cuda_tensor_ptr(name, input, data_type)?;

    Ok(CudaInputData {
        data_type,
        shape,
        ptr,
        bytes,
    })
}

#[cfg(feature = "cuda")]
fn cuda_tensor_ptr(
    tensor_name: &str,
    tensor: &Tensor,
    data_type: DataType,
) -> Result<*const c_void> {
    if !tensor.is_contiguous() {
        return Err(Error::InvalidShape {
            tensor: tensor_name.to_owned(),
            reason: "CUDA zero-copy input must be contiguous".to_owned(),
        });
    }

    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(Error::InvalidShape {
            tensor: tensor_name.to_owned(),
            reason: "CUDA zero-copy input is not backed by CUDA storage".to_owned(),
        });
    };
    let len = tensor.elem_count();
    let start = layout.start_offset();
    let end = start.checked_add(len).ok_or_else(|| Error::InvalidShape {
        tensor: tensor_name.to_owned(),
        reason: "CUDA zero-copy input offset overflow".to_owned(),
    })?;

    match data_type {
        DataType::Float => cuda_storage_ptr::<f32>(storage, start, end),
        DataType::Half => cuda_storage_ptr::<f16>(storage, start, end),
        DataType::Bf16 => cuda_storage_ptr::<bf16>(storage, start, end),
        DataType::Int32 => cuda_storage_ptr::<i32>(storage, start, end),
        DataType::Int64 => cuda_storage_ptr::<i64>(storage, start, end),
        DataType::Uint8 => cuda_storage_ptr::<u8>(storage, start, end),
        data_type => Err(Error::UnsupportedDataType {
            data_type,
            context: "Candle CUDA input pointer conversion",
        }),
    }
}

#[cfg(feature = "cuda")]
fn cuda_tensor_mut_ptr(
    tensor_name: &str,
    tensor: &mut Tensor,
    data_type: DataType,
) -> Result<*mut c_void> {
    if !tensor.is_contiguous() {
        return Err(Error::InvalidShape {
            tensor: tensor_name.to_owned(),
            reason: "CUDA zero-copy output must be contiguous".to_owned(),
        });
    }

    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(Error::InvalidShape {
            tensor: tensor_name.to_owned(),
            reason: "CUDA zero-copy output is not backed by CUDA storage".to_owned(),
        });
    };
    let len = tensor.elem_count();
    let start = layout.start_offset();
    let end = start.checked_add(len).ok_or_else(|| Error::InvalidShape {
        tensor: tensor_name.to_owned(),
        reason: "CUDA zero-copy output offset overflow".to_owned(),
    })?;

    // Candle exposes CUDA storage through a read guard. The caller only binds
    // this pointer after validating output metadata and rejecting aliasing with
    // the other tensors participating in the run.
    match data_type {
        DataType::Float => cuda_storage_ptr::<f32>(storage, start, end).map(|ptr| ptr.cast_mut()),
        DataType::Half => cuda_storage_ptr::<f16>(storage, start, end).map(|ptr| ptr.cast_mut()),
        DataType::Bf16 => cuda_storage_ptr::<bf16>(storage, start, end).map(|ptr| ptr.cast_mut()),
        DataType::Int32 => cuda_storage_ptr::<i32>(storage, start, end).map(|ptr| ptr.cast_mut()),
        DataType::Int64 => cuda_storage_ptr::<i64>(storage, start, end).map(|ptr| ptr.cast_mut()),
        DataType::Uint8 => cuda_storage_ptr::<u8>(storage, start, end).map(|ptr| ptr.cast_mut()),
        data_type => Err(Error::UnsupportedDataType {
            data_type,
            context: "Candle CUDA output pointer conversion",
        }),
    }
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

/// Converts a Candle dtype into a TensorRT data type for host/staging bindings.
pub fn data_type_from_candle(dtype: DType) -> Result<DataType> {
    match dtype {
        DType::F32 => Ok(DataType::Float),
        DType::F16 => Ok(DataType::Half),
        DType::BF16 => Ok(DataType::Bf16),
        DType::I32 => Ok(DataType::Int32),
        DType::I64 => Ok(DataType::Int64),
        DType::U8 => Ok(DataType::Uint8),
        _ => Err(Error::UnsupportedCandleDType {
            dtype,
            context: "TensorRT input conversion",
        }),
    }
}

/// Converts a TensorRT data type into the closest Candle dtype supported by the
/// host bridge.
pub fn candle_dtype_from_data_type(data_type: DataType) -> Result<DType> {
    match data_type {
        DataType::Float => Ok(DType::F32),
        DataType::Half => Ok(DType::F16),
        DataType::Bf16 => Ok(DType::BF16),
        DataType::Int32 => Ok(DType::I32),
        DataType::Int64 => Ok(DType::I64),
        DataType::Uint8 => Ok(DType::U8),
        _ => Err(Error::UnsupportedDataType {
            data_type,
            context: "Candle dtype conversion",
        }),
    }
}

fn candle_dims(tensor_name: &str, tensor: &Tensor) -> Result<Dims> {
    let dims = tensor
        .dims()
        .iter()
        .map(|dim| {
            i64::try_from(*dim).map_err(|_| Error::InvalidShape {
                tensor: tensor_name.to_owned(),
                reason: format!("dimension {dim} cannot be represented as i64"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Dims::new(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn maps_candle_dtypes_to_tensorrt_types() {
        assert_eq!(data_type_from_candle(DType::F32).unwrap(), DataType::Float);
        assert_eq!(data_type_from_candle(DType::F16).unwrap(), DataType::Half);
        assert_eq!(data_type_from_candle(DType::BF16).unwrap(), DataType::Bf16);
        assert_eq!(data_type_from_candle(DType::I32).unwrap(), DataType::Int32);
        assert_eq!(data_type_from_candle(DType::I64).unwrap(), DataType::Int64);
        assert_eq!(data_type_from_candle(DType::U8).unwrap(), DataType::Uint8);
        assert!(data_type_from_candle(DType::F64).is_err());
    }

    #[test]
    fn candle_outputs_can_be_indexed_by_name() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], 2, &Device::Cpu).unwrap();
        let mut outputs = OutputTensors::new([("logits", tensor)]);

        assert_eq!(outputs.iter().count(), 1);
        assert_eq!(outputs["logits"].dims(), &[2]);
        assert!(outputs.get("missing").is_none());
        assert_eq!(outputs.get("logits").unwrap().dtype(), DType::F32);
        *outputs.get_mut("logits").unwrap() =
            Tensor::from_vec(vec![3.0f32, 4.0], 2, &Device::Cpu).unwrap();
        assert_eq!(outputs["logits"].to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn candle_inputs_preserve_names_and_order() {
        let input_ids = Tensor::from_vec(vec![1i32, 2, 3], 3, &Device::Cpu).unwrap();
        let position_ids = Tensor::from_vec(vec![0i32, 1, 2], 3, &Device::Cpu).unwrap();
        let inputs =
            InputTensors::new([("input_ids", &input_ids), ("position_ids", &position_ids)]);

        assert_eq!(inputs.len(), 2);
        assert!(!inputs.is_empty());
        assert_eq!(inputs.get("input_ids").unwrap().dims(), &[3]);
        assert_eq!(
            inputs.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["input_ids", "position_ids"]
        );
    }

    #[test]
    fn candle_outputs_can_be_consumed_as_vec() {
        let logits = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        let hidden = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        let outputs = OutputTensors::new([("logits", logits), ("hidden", hidden)]);

        let names = outputs
            .into_vec()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["logits", "hidden"]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn run_keepalive_clones_inputs_and_outputs() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0], 2, &Device::Cpu).unwrap();
        let output = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
        let inputs = InputTensors::new([("input", &input)]);
        let outputs = OutputTensors::new([("output", output)]);

        let keepalive = run_keepalive(&inputs, &outputs);

        assert_eq!(keepalive.len(), 2);
        assert_eq!(keepalive[0].dims(), &[2]);
        assert_eq!(keepalive[1].dtype(), DType::F32);
    }
}
