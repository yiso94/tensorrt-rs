#![deny(unsafe_op_in_unsafe_fn)]

//! Low-level TensorRT runtime FFI.
//!
//! This crate exposes the small C ABI shim compiled by `build.rs`. Safe Rust
//! APIs live in the `tensorrt` crate.

use std::os::raw::{c_char, c_int, c_void};

pub const MAX_DIMS: usize = 8;

#[repr(C)]
pub struct RawRuntime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RawEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RawExecutionContext {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    AllocationFailed = 2,
    NullPointer = 3,
    TensorRtError = 4,
    Exception = 5,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub nb_dims: i32,
    pub d: [i64; MAX_DIMS],
}

unsafe extern "C" {
    pub fn safe_trt_version_major() -> c_int;
    pub fn safe_trt_version_minor() -> c_int;
    pub fn safe_trt_version_patch() -> c_int;
    pub fn safe_trt_version_build() -> c_int;

    pub fn safe_trt_runtime_create(
        min_severity: c_int,
        out_runtime: *mut *mut RawRuntime,
    ) -> Status;
    pub fn safe_trt_runtime_destroy(runtime: *mut RawRuntime);
    pub fn safe_trt_runtime_deserialize_engine(
        runtime: *mut RawRuntime,
        data: *const c_void,
        size: usize,
        out_engine: *mut *mut RawEngine,
    ) -> Status;

    pub fn safe_trt_engine_destroy(engine: *mut RawEngine);
    pub fn safe_trt_engine_num_io_tensors(
        engine: *const RawEngine,
        out_count: *mut c_int,
    ) -> Status;
    pub fn safe_trt_engine_io_tensor_name(
        engine: *const RawEngine,
        index: c_int,
        out_name: *mut *const c_char,
    ) -> Status;
    pub fn safe_trt_engine_tensor_shape(
        engine: *const RawEngine,
        name: *const c_char,
        out_dims: *mut Dims,
    ) -> Status;
    pub fn safe_trt_engine_tensor_data_type(
        engine: *const RawEngine,
        name: *const c_char,
        out_data_type: *mut c_int,
    ) -> Status;
    pub fn safe_trt_engine_tensor_io_mode(
        engine: *const RawEngine,
        name: *const c_char,
        out_io_mode: *mut c_int,
    ) -> Status;
    pub fn safe_trt_engine_create_execution_context(
        engine: *const RawEngine,
        out_context: *mut *mut RawExecutionContext,
    ) -> Status;

    pub fn safe_trt_execution_context_destroy(context: *mut RawExecutionContext);
    pub fn safe_trt_context_set_input_shape(
        context: *mut RawExecutionContext,
        name: *const c_char,
        dims: Dims,
    ) -> Status;
    pub fn safe_trt_context_tensor_shape(
        context: *mut RawExecutionContext,
        name: *const c_char,
        out_dims: *mut Dims,
    ) -> Status;
    pub fn safe_trt_context_set_tensor_address(
        context: *mut RawExecutionContext,
        name: *const c_char,
        data: *mut c_void,
    ) -> Status;
    pub fn safe_trt_context_enqueue_v3(
        context: *mut RawExecutionContext,
        stream: *mut c_void,
    ) -> Status;
}
