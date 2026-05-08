#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(unsafe_op_in_unsafe_fn)]

//! Safe TensorRT runtime bindings.

pub mod candle;
mod cuda;
mod data_type;
mod engine;
mod error;
mod ffi {
    pub(crate) use tensorrt_sys::*;
}
mod prepared;
mod session;
mod tensor;

#[cfg(test)]
mod tests;

/// Maximum TensorRT tensor rank supported by the C ABI shim.
pub const MAX_DIMS: usize = ffi::MAX_DIMS;

pub use candle::{InputTensors, OutputTensors};
pub use cuda::{
    CudaError, CudaEvent, CudaEventHandle, CudaStream, record_event, stream_wait_event,
    synchronize_stream,
};
pub use data_type::{DataType, bf16_bits_to_f32, f32_to_bf16_bits};
pub use engine::{Engine, ExecutionContext, LogSeverity, Runtime, Version, version};
pub use error::{Error, Result};
pub use prepared::{PreparedDeviceSession, PreparedEnqueuedInference, PreparedMixedDeviceSession};
pub use session::{EnqueuedInference, Session};
pub use tensor::{
    DeviceInputTensor, DeviceOutputTensor, Dims, HostInputTensor, HostOutputTensor, HostTensor,
    InputBinding, TensorIOMode, TensorInfo, TensorLocation, TensorShape, as_bytes,
    dims_to_usize_shape, element_count, tensor_byte_len,
};
