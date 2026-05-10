use super::error::{Error, Result};
#[cfg(feature = "cuda")]
use candle_core::Device;
use candle_core::DeviceLocation;
use std::ffi::CStr;
use std::fmt;
use std::marker::PhantomData;
#[cfg(feature = "cuda")]
use std::os::raw::c_ulonglong;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::{self, NonNull};

/// Opaque CUDA stream pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaStream<'a> {
    raw: *mut c_void,
    device_location: Option<DeviceLocation>,
    _owner: PhantomData<&'a ()>,
}

impl CudaStream<'static> {
    /// The CUDA default stream.
    ///
    /// This stream has no device metadata, so the high-level Candle `run` API
    /// rejects it. Use it only with lower-level APIs where the caller upholds
    /// the CUDA context requirements.
    pub const DEFAULT: Self = Self {
        raw: ptr::null_mut(),
        device_location: None,
        _owner: PhantomData,
    };
}

impl<'a> CudaStream<'a> {
    /// Borrows the CUDA stream owned by a Candle CUDA device.
    ///
    /// The returned stream is not owned by this crate. It must not be used after
    /// `device` is dropped.
    #[cfg(feature = "cuda")]
    pub fn from_device(device: &'a Device) -> Result<Self> {
        let stream = device.as_cuda_device()?.cuda_stream();
        Ok(Self {
            raw: stream.cu_stream().cast(),
            device_location: Some(device.location()),
            _owner: PhantomData,
        })
    }

    /// Creates a stream wrapper from a raw `cudaStream_t` pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be either null for the default stream or a valid
    /// `cudaStream_t` for the active CUDA context.
    pub unsafe fn from_raw(raw: *mut c_void) -> CudaStream<'static> {
        CudaStream {
            raw,
            device_location: None,
            _owner: PhantomData,
        }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn device_location(&self) -> Option<DeviceLocation> {
        self.device_location
    }

    pub fn synchronize(&self) -> Result<()> {
        synchronize_stream(*self)
    }

    pub fn wait_event(&self, event: CudaEvent) -> Result<()> {
        stream_wait_event(*self, event)
    }
}

/// Opaque CUDA event pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct CudaEvent(*mut c_void);

impl CudaEvent {
    /// Creates an event wrapper from a raw `cudaEvent_t` pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid `cudaEvent_t` for the active CUDA context.
    pub unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.0
    }

    pub fn record(&self, stream: CudaStream<'_>) -> Result<()> {
        record_event(*self, stream)
    }
}

#[derive(Clone, Debug)]
pub struct CudaError {
    pub code: i32,
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for CudaError {}

/// Owned CUDA event used for explicit stream dependencies.
#[derive(Debug)]
pub struct CudaEventHandle {
    raw: NonNull<c_void>,
}

unsafe impl Send for CudaEventHandle {}
unsafe impl Sync for CudaEventHandle {}

impl CudaEventHandle {
    #[cfg(feature = "cuda")]
    fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        unsafe {
            check_cuda(ffi::cudaEventCreateWithFlags(
                &mut raw,
                ffi::CUDA_EVENT_DISABLE_TIMING,
            ))?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            Error::Cuda(CudaError {
                code: 0,
                message: "cudaEventCreateWithFlags returned null".to_owned(),
            })
        })?;
        Ok(Self { raw })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn new_for_device_location(location: DeviceLocation) -> Result<Self> {
        let DeviceLocation::Cuda { gpu_id } = location else {
            return Err(Error::InvalidShape {
                tensor: "<cuda-event>".to_owned(),
                reason: format!("CUDA event requires a CUDA device location, got {location:?}"),
            });
        };
        let gpu_id = c_int::try_from(gpu_id).map_err(|_| Error::InvalidShape {
            tensor: "<cuda-event>".to_owned(),
            reason: format!("CUDA device id {gpu_id} cannot be represented as c_int"),
        })?;

        let mut previous_device = 0;
        unsafe {
            check_cuda(ffi::cudaGetDevice(&mut previous_device))?;
            check_cuda(ffi::cudaSetDevice(gpu_id))?;
        }

        let event = Self::new();
        let restore = unsafe { check_cuda(ffi::cudaSetDevice(previous_device)) };

        match (event, restore) {
            (Ok(event), Ok(())) => Ok(event),
            (Err(error), _) => Err(error),
            (Ok(_event), Err(error)) => Err(error),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_for_stream(stream: CudaStream<'_>) -> Result<Self> {
        let Some(location) = stream.device_location() else {
            return Err(Error::InvalidShape {
                tensor: "<cuda-event>".to_owned(),
                reason: "CUDA event requires a stream with device metadata".to_owned(),
            });
        };
        Self::new_for_device_location(location)
    }

    pub fn as_event(&self) -> CudaEvent {
        CudaEvent(self.raw.as_ptr())
    }

    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        synchronize_event(self.as_event())
    }

    #[cfg(feature = "cuda")]
    pub fn is_complete(&self) -> Result<bool> {
        query_event(self.as_event())
    }
}

impl Drop for CudaEventHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaEventDestroy(self.raw.as_ptr());
        }
    }
}

pub(crate) fn synchronize_stream(stream: CudaStream<'_>) -> Result<()> {
    unsafe { check_cuda(ffi::cudaStreamSynchronize(stream.as_raw())) }
}

pub(crate) fn record_event(event: CudaEvent, stream: CudaStream<'_>) -> Result<()> {
    unsafe { check_cuda(ffi::cudaEventRecord(event.as_raw(), stream.as_raw())) }
}

#[cfg(feature = "cuda")]
pub(crate) fn query_event(event: CudaEvent) -> Result<bool> {
    let code = unsafe { ffi::cudaEventQuery(event.as_raw()) };
    if code == ffi::CUDA_SUCCESS {
        Ok(true)
    } else if code == ffi::CUDA_ERROR_NOT_READY {
        Ok(false)
    } else {
        check_cuda(code).map(|_| true)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn synchronize_event(event: CudaEvent) -> Result<()> {
    unsafe { check_cuda(ffi::cudaEventSynchronize(event.as_raw())) }
}

pub(crate) fn stream_wait_event(stream: CudaStream<'_>, event: CudaEvent) -> Result<()> {
    unsafe { check_cuda(ffi::cudaStreamWaitEvent(stream.as_raw(), event.as_raw(), 0)) }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaGraphExec {
    raw: NonNull<c_void>,
}

#[cfg(feature = "cuda")]
impl CudaGraphExec {
    pub(crate) fn capture(
        stream: CudaStream<'_>,
        enqueue: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        unsafe {
            check_cuda(ffi::cudaStreamBeginCapture(
                stream.as_raw(),
                ffi::CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ))?;
        }

        let enqueue_result = enqueue();
        let mut graph_raw = ptr::null_mut();
        let end_capture = unsafe { ffi::cudaStreamEndCapture(stream.as_raw(), &mut graph_raw) };

        if let Err(error) = enqueue_result {
            if end_capture == ffi::CUDA_SUCCESS
                && let Some(graph_raw) = NonNull::new(graph_raw)
            {
                unsafe {
                    let _ = ffi::cudaGraphDestroy(graph_raw.as_ptr());
                }
            }
            return Err(error);
        }

        check_cuda(end_capture)?;
        let graph = CudaGraph {
            raw: NonNull::new(graph_raw).ok_or_else(|| {
                Error::Cuda(CudaError {
                    code: 0,
                    message: "cudaStreamEndCapture returned null graph".to_owned(),
                })
            })?,
        };

        let mut exec_raw = ptr::null_mut();
        unsafe {
            check_cuda(ffi::cudaGraphInstantiate(
                &mut exec_raw,
                graph.raw.as_ptr(),
                0,
            ))?;
        }
        let raw = NonNull::new(exec_raw).ok_or_else(|| {
            Error::Cuda(CudaError {
                code: 0,
                message: "cudaGraphInstantiate returned null graph executable".to_owned(),
            })
        })?;
        Ok(Self { raw })
    }

    pub(crate) fn launch(&self, stream: CudaStream<'_>) -> Result<()> {
        unsafe { check_cuda(ffi::cudaGraphLaunch(self.raw.as_ptr(), stream.as_raw())) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaGraphExecDestroy(self.raw.as_ptr());
        }
    }
}

#[cfg(feature = "cuda")]
struct CudaGraph {
    raw: NonNull<c_void>,
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaGraphDestroy(self.raw.as_ptr());
        }
    }
}

pub(crate) fn check_cuda(code: c_int) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        return Ok(());
    }

    let message = unsafe {
        let ptr = ffi::cudaGetErrorString(code);
        if ptr.is_null() {
            "unknown CUDA error".to_owned()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Err(Error::Cuda(CudaError { code, message }))
}

pub(crate) struct DeviceBuffer {
    ptr: NonNull<c_void>,
    bytes: usize,
}

impl DeviceBuffer {
    pub(crate) fn new(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::InvalidShape {
                tensor: "<device-buffer>".to_owned(),
                reason: "cannot allocate zero bytes".to_owned(),
            });
        }

        let mut ptr = ptr::null_mut();
        unsafe {
            check_cuda(ffi::cudaMalloc(&mut ptr, bytes))?;
        }
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::Cuda(CudaError {
                code: 0,
                message: "cudaMalloc returned null".to_owned(),
            })
        })?;
        Ok(Self { ptr, bytes })
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub(crate) fn copy_from_host(&self, src: &[u8], stream: CudaStream<'_>) -> Result<()> {
        if src.len() != self.bytes {
            return Err(Error::SizeMismatch {
                tensor: "<host-to-device>".to_owned(),
                expected_bytes: self.bytes,
                actual_bytes: src.len(),
            });
        }

        unsafe {
            check_cuda(ffi::cudaMemcpyAsync(
                self.ptr.as_ptr(),
                src.as_ptr().cast(),
                src.len(),
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            ))
        }
    }

    pub(crate) fn copy_to_host(&self, dst: &mut [u8], stream: CudaStream<'_>) -> Result<()> {
        if dst.len() != self.bytes {
            return Err(Error::SizeMismatch {
                tensor: "<device-to-host>".to_owned(),
                expected_bytes: self.bytes,
                actual_bytes: dst.len(),
            });
        }

        unsafe {
            check_cuda(ffi::cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                self.ptr.as_ptr(),
                dst.len(),
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                stream.as_raw(),
            ))
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaFree(self.ptr.as_ptr());
        }
    }
}

mod ffi {
    #[cfg(feature = "cuda")]
    use super::c_ulonglong;
    use super::{c_char, c_int, c_void};

    pub const CUDA_SUCCESS: c_int = 0;
    #[cfg(feature = "cuda")]
    pub const CUDA_ERROR_NOT_READY: c_int = 600;
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
    #[cfg(feature = "cuda")]
    pub const CUDA_EVENT_DISABLE_TIMING: c_int = 2;
    #[cfg(feature = "cuda")]
    pub const CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn cudaFree(dev_ptr: *mut c_void) -> c_int;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
            stream: *mut c_void,
        ) -> c_int;
        pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_int) -> c_int;
        pub fn cudaEventDestroy(event: *mut c_void) -> c_int;
        pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaGetDevice(device: *mut c_int) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaSetDevice(device: c_int) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaEventQuery(event: *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaEventSynchronize(event: *mut c_void) -> c_int;
        pub fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: c_int) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaStreamBeginCapture(stream: *mut c_void, mode: c_int) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaGraphInstantiate(
            graph_exec: *mut *mut c_void,
            graph: *mut c_void,
            flags: c_ulonglong,
        ) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaGraphLaunch(graph_exec: *mut c_void, stream: *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaGraphDestroy(graph: *mut c_void) -> c_int;
        #[cfg(feature = "cuda")]
        pub fn cudaGraphExecDestroy(graph_exec: *mut c_void) -> c_int;
        pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    }
}
