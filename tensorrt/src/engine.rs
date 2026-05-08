use super::cuda::CudaStream;
use super::data_type::DataType;
use super::error::{Error, Result};
#[cfg(feature = "cuda")]
use super::tensor::DeviceInputTensor;
use super::tensor::{Dims, TensorIOMode, TensorInfo, TensorShape};
use super::{MAX_DIMS, ffi};
use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::ptr::{self, NonNull};
use std::rc::Rc;

/// TensorRT version compiled into the binding shim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Version {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
    pub build: i32,
}

/// Returns the TensorRT version used to compile this crate.
pub fn version() -> Version {
    Version {
        major: unsafe { ffi::safe_trt_version_major() },
        minor: unsafe { ffi::safe_trt_version_minor() },
        patch: unsafe { ffi::safe_trt_version_patch() },
        build: unsafe { ffi::safe_trt_version_build() },
    }
}

/// Minimum severity emitted by the TensorRT logger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum LogSeverity {
    InternalError = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

/// TensorRT runtime used to deserialize plan files.
pub struct Runtime {
    raw: NonNull<ffi::RawRuntime>,
}

impl Runtime {
    pub fn new(min_severity: LogSeverity) -> Result<Self> {
        let mut raw = ptr::null_mut();
        let status = unsafe { ffi::safe_trt_runtime_create(min_severity as c_int, &mut raw) };
        Error::from_status(status)?;

        let raw = NonNull::new(raw).ok_or(Error::NullPointer)?;
        Ok(Self { raw })
    }

    pub fn deserialize_engine(&self, plan: impl AsRef<[u8]>) -> Result<Engine> {
        let plan = plan.as_ref();
        let mut raw = ptr::null_mut();
        let status = unsafe {
            ffi::safe_trt_runtime_deserialize_engine(
                self.raw.as_ptr(),
                plan.as_ptr().cast(),
                plan.len(),
                &mut raw,
            )
        };
        Error::from_status(status)?;

        let raw = NonNull::new(raw).ok_or(Error::NullPointer)?;
        Ok(Engine {
            inner: Rc::new(EngineInner { raw }),
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { ffi::safe_trt_runtime_destroy(self.raw.as_ptr()) };
    }
}

/// Deserialized TensorRT CUDA engine.
#[derive(Clone)]
pub struct Engine {
    inner: Rc<EngineInner>,
}

struct EngineInner {
    raw: NonNull<ffi::RawEngine>,
}

impl Engine {
    fn raw(&self) -> *mut ffi::RawEngine {
        self.inner.raw.as_ptr()
    }

    pub fn num_io_tensors(&self) -> Result<usize> {
        let mut count = 0;
        let status = unsafe { ffi::safe_trt_engine_num_io_tensors(self.raw(), &mut count) };
        Error::from_status(status)?;
        usize::try_from(count).map_err(|_| Error::InvalidArgument)
    }

    pub fn io_tensor_name(&self, index: usize) -> Result<String> {
        let index = c_int::try_from(index).map_err(|_| Error::InvalidArgument)?;
        let mut name = ptr::null();
        let status = unsafe { ffi::safe_trt_engine_io_tensor_name(self.raw(), index, &mut name) };
        Error::from_status(status)?;

        if name.is_null() {
            return Err(Error::NullPointer);
        }

        let name = unsafe { CStr::from_ptr(name) };
        Ok(name.to_string_lossy().into_owned())
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let name = tensor_name(name)?;
        let mut dims = ffi::Dims {
            nb_dims: 0,
            d: [0; MAX_DIMS],
        };
        let status =
            unsafe { ffi::safe_trt_engine_tensor_shape(self.raw(), name.as_ptr(), &mut dims) };
        Error::from_status(status)?;
        Dims::from_ffi(dims)
    }

    pub fn tensor_data_type(&self, name: &str) -> Result<DataType> {
        let name = tensor_name(name)?;
        let mut data_type = 0;
        let status = unsafe {
            ffi::safe_trt_engine_tensor_data_type(self.raw(), name.as_ptr(), &mut data_type)
        };
        Error::from_status(status)?;
        Ok(DataType::from_raw(data_type))
    }

    pub fn tensor_io_mode(&self, name: &str) -> Result<TensorIOMode> {
        let name = tensor_name(name)?;
        let mut io_mode = 0;
        let status =
            unsafe { ffi::safe_trt_engine_tensor_io_mode(self.raw(), name.as_ptr(), &mut io_mode) };
        Error::from_status(status)?;
        Ok(TensorIOMode::from_raw(io_mode))
    }

    pub fn io_tensors(&self) -> Result<Vec<TensorInfo>> {
        let count = self.num_io_tensors()?;
        let mut tensors = Vec::with_capacity(count);
        for index in 0..count {
            let name = self.io_tensor_name(index)?;
            tensors.push(TensorInfo::with_io_mode(
                name.clone(),
                self.tensor_data_type(&name)?,
                self.tensor_shape(&name)?,
                self.tensor_io_mode(&name)?,
            ));
        }
        Ok(tensors)
    }

    pub fn create_execution_context(&self) -> Result<ExecutionContext> {
        let mut raw = ptr::null_mut();
        let status = unsafe { ffi::safe_trt_engine_create_execution_context(self.raw(), &mut raw) };
        Error::from_status(status)?;

        let raw = NonNull::new(raw).ok_or(Error::NullPointer)?;
        Ok(ExecutionContext {
            raw,
            input_shapes: None,
            _engine: self.clone(),
        })
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        unsafe { ffi::safe_trt_engine_destroy(self.raw.as_ptr()) };
    }
}

/// TensorRT execution context for one in-flight inference stream.
pub struct ExecutionContext {
    raw: NonNull<ffi::RawExecutionContext>,
    input_shapes: Option<Vec<(String, Dims)>>,
    _engine: Engine,
}

impl ExecutionContext {
    pub fn set_input_shape(&mut self, name: &str, dims: &Dims) -> Result<()> {
        let name = tensor_name(name)?;
        let status = unsafe {
            ffi::safe_trt_context_set_input_shape(self.raw.as_ptr(), name.as_ptr(), dims.to_ffi())
        };
        Error::from_status(status)
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Dims> {
        let name = tensor_name(name)?;
        let mut dims = ffi::Dims {
            nb_dims: 0,
            d: [0; MAX_DIMS],
        };
        let status = unsafe {
            ffi::safe_trt_context_tensor_shape(self.raw.as_ptr(), name.as_ptr(), &mut dims)
        };
        Error::from_status(status)?;
        Dims::from_ffi(dims)
    }

    /// Sets a TensorRT input or output tensor address.
    ///
    /// Passing null resets the tensor address to TensorRT's default state.
    ///
    /// # Safety
    ///
    /// `data` must point to a host/device allocation appropriate for the tensor
    /// location, data type, format, shape, and active CUDA context. The buffer
    /// must remain valid until the queued inference work has completed.
    pub unsafe fn set_tensor_address(&mut self, name: &str, data: *mut c_void) -> Result<()> {
        let name = tensor_name(name)?;
        let status = unsafe {
            ffi::safe_trt_context_set_tensor_address(self.raw.as_ptr(), name.as_ptr(), data)
        };
        Error::from_status(status)
    }

    /// Enqueues inference on a CUDA stream.
    ///
    /// # Safety
    ///
    /// All required tensor addresses and dynamic input shapes must be set before
    /// enqueueing. `stream` must be valid for the active CUDA context, or
    /// `CudaStream::DEFAULT`.
    pub unsafe fn enqueue_v3(&mut self, stream: CudaStream<'_>) -> Result<()> {
        let status =
            unsafe { ffi::safe_trt_context_enqueue_v3(self.raw.as_ptr(), stream.as_raw()) };
        Error::from_status(status)
    }

    pub(crate) fn input_shapes_match(&self, inputs: &[TensorShape<'_>]) -> bool {
        let Some(cached) = &self.input_shapes else {
            return false;
        };
        cached.len() == inputs.len()
            && cached
                .iter()
                .zip(inputs.iter())
                .all(|((name, shape), input)| name == input.name && shape == &input.shape)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn device_input_shapes_match(&self, inputs: &[DeviceInputTensor<'_>]) -> bool {
        let Some(cached) = &self.input_shapes else {
            return false;
        };
        cached.len() == inputs.len()
            && cached
                .iter()
                .zip(inputs.iter())
                .all(|((name, shape), input)| name == input.name && shape == &input.shape)
    }

    pub(crate) fn remember_input_shapes(&mut self, inputs: &[TensorShape<'_>]) {
        self.input_shapes = Some(
            inputs
                .iter()
                .map(|input| (input.name.to_owned(), input.shape.clone()))
                .collect(),
        );
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn remember_device_input_shapes(&mut self, inputs: &[DeviceInputTensor<'_>]) {
        self.input_shapes = Some(
            inputs
                .iter()
                .map(|input| (input.name.to_owned(), input.shape.clone()))
                .collect(),
        );
    }
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        unsafe { ffi::safe_trt_execution_context_destroy(self.raw.as_ptr()) };
    }
}

pub(crate) fn tensor_name(name: &str) -> Result<CString> {
    CString::new(name).map_err(Error::InvalidTensorName)
}
