use super::cuda::{CudaStream, DeviceBuffer, synchronize_stream};
use super::data_type::DataType;
use super::engine::ExecutionContext;
use super::error::{Error, Result};
use super::tensor::{
    DeviceInputTensor, DeviceOutputTensor, Dims, InputBinding, TensorInfo, tensor_byte_len,
};
use std::ffi::c_void;

/// TensorRT device-pointer runner with a reusable execution context.
///
/// Dynamic input shapes and output shapes are resolved once when this value is
/// created. Each enqueue only validates the concrete CUDA bindings, updates
/// tensor addresses, and launches TensorRT on the supplied stream.
pub struct PreparedDeviceSession {
    context: ExecutionContext,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
}

/// TensorRT runner with a reusable execution context for mixed input bindings.
///
/// This is the prepared counterpart to
/// [`Session::run_mixed_to_device`](super::Session::run_mixed_to_device).
/// Input shapes, dtypes, binding order, and output shapes are fixed when this
/// value is created. Each enqueue updates tensor addresses, reuses any
/// host-to-device staging buffers, and launches TensorRT on the supplied stream.
pub struct PreparedMixedDeviceSession {
    context: ExecutionContext,
    input_infos: Vec<TensorInfo>,
    output_infos: Vec<TensorInfo>,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
    input_kinds: Vec<PreparedInputKind>,
    staged_inputs: Vec<Option<DeviceBuffer>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedInputKind {
    Host,
    HostToDevice,
    Device,
}

impl PreparedDeviceSession {
    pub(crate) fn new(
        context: ExecutionContext,
        input_infos: Vec<TensorInfo>,
        output_infos: Vec<TensorInfo>,
    ) -> Result<Self> {
        let input_bytes = prepared_tensor_bytes(&input_infos)?;
        let output_bytes = prepared_tensor_bytes(&output_infos)?;

        Ok(Self {
            context,
            input_infos,
            output_infos,
            input_bytes,
            output_bytes,
        })
    }

    pub fn input_infos(&self) -> &[TensorInfo] {
        &self.input_infos
    }

    pub fn output_infos(&self) -> &[TensorInfo] {
        &self.output_infos
    }

    /// Enqueues TensorRT inference using the prepared execution context.
    ///
    /// This avoids per-run execution-context creation and shape inference. The
    /// returned guard mutably borrows this prepared session, so the same context
    /// cannot be reused until the in-flight work is synchronized or dropped.
    ///
    /// # Safety
    ///
    /// Same pointer and stream requirements as
    /// [`Session::enqueue_device`](super::Session::enqueue_device).
    pub unsafe fn enqueue_device<'run, 'stream>(
        &'run mut self,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'stream>,
    ) -> Result<PreparedEnqueuedInference<'run, 'stream>> {
        validate_prepared_inputs(&self.input_infos, &self.input_bytes, inputs)?;
        validate_prepared_outputs(&self.output_infos, &self.output_bytes, outputs)?;

        let context = &mut self.context;
        for input in inputs {
            unsafe {
                context.set_tensor_address(input.name, input.ptr.cast_mut())?;
            }
        }
        for output in outputs.iter() {
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }
        Ok(PreparedEnqueuedInference {
            _context: context,
            stream,
            synchronized: false,
        })
    }

    /// Runs prepared device-pointer inference and synchronizes `stream`.
    ///
    /// Use [`enqueue_device`](Self::enqueue_device) to chain work on the CUDA
    /// stream and synchronize later.
    ///
    /// # Safety
    ///
    /// Same requirements as [`enqueue_device`](Self::enqueue_device).
    pub unsafe fn run_device(
        &mut self,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let inference = unsafe { self.enqueue_device(inputs, outputs, stream)? };
        inference.synchronize()
    }
}

impl PreparedMixedDeviceSession {
    pub(crate) fn new(
        context: ExecutionContext,
        input_infos: Vec<TensorInfo>,
        output_infos: Vec<TensorInfo>,
        input_kinds: Vec<PreparedInputKind>,
    ) -> Result<Self> {
        if input_infos.len() != input_kinds.len() {
            return Err(Error::InvalidShape {
                tensor: "<prepared-mixed-inputs>".to_owned(),
                reason: format!(
                    "{} input infos for {} input kinds",
                    input_infos.len(),
                    input_kinds.len()
                ),
            });
        }

        let input_bytes = prepared_tensor_bytes(&input_infos)?;
        let output_bytes = prepared_tensor_bytes(&output_infos)?;
        let staged_inputs = input_kinds
            .iter()
            .zip(input_bytes.iter())
            .map(|(kind, bytes)| match kind {
                PreparedInputKind::HostToDevice => DeviceBuffer::new(*bytes).map(Some),
                PreparedInputKind::Host | PreparedInputKind::Device => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            context,
            input_infos,
            output_infos,
            input_bytes,
            output_bytes,
            input_kinds,
            staged_inputs,
        })
    }

    pub fn input_infos(&self) -> &[TensorInfo] {
        &self.input_infos
    }

    pub fn output_infos(&self) -> &[TensorInfo] {
        &self.output_infos
    }

    /// Enqueues TensorRT inference using prepared mixed bindings.
    ///
    /// Host-to-device inputs are copied into reusable staging buffers on
    /// `stream`. Host inputs are bound directly as host pointers. Device inputs
    /// and outputs are bound directly without copies.
    ///
    /// # Safety
    ///
    /// Same pointer and stream requirements as
    /// [`Session::run_mixed_to_device`](super::Session::run_mixed_to_device).
    pub unsafe fn enqueue_mixed_to_device<'run, 'stream>(
        &'run mut self,
        inputs: &[InputBinding<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'stream>,
    ) -> Result<PreparedEnqueuedInference<'run, 'stream>> {
        validate_prepared_mixed_inputs(
            &self.input_infos,
            &self.input_bytes,
            &self.input_kinds,
            inputs,
        )?;
        validate_prepared_outputs(&self.output_infos, &self.output_bytes, outputs)?;

        let input_ptrs = self.prepare_mixed_input_pointers(inputs, stream)?;
        let context = &mut self.context;
        for (input, ptr) in inputs.iter().zip(input_ptrs.iter()) {
            unsafe {
                context.set_tensor_address(input.name(), *ptr)?;
            }
        }
        for output in outputs.iter() {
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }
        Ok(PreparedEnqueuedInference {
            _context: context,
            stream,
            synchronized: false,
        })
    }

    /// Runs prepared mixed inference and synchronizes `stream`.
    ///
    /// Use [`enqueue_mixed_to_device`](Self::enqueue_mixed_to_device) when the
    /// caller can keep the in-flight guard alive and synchronize later.
    ///
    /// # Safety
    ///
    /// Same requirements as
    /// [`enqueue_mixed_to_device`](Self::enqueue_mixed_to_device).
    pub unsafe fn run_mixed_to_device(
        &mut self,
        inputs: &[InputBinding<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let inference = unsafe { self.enqueue_mixed_to_device(inputs, outputs, stream)? };
        inference.synchronize()
    }

    fn prepare_mixed_input_pointers(
        &mut self,
        inputs: &[InputBinding<'_>],
        stream: CudaStream<'_>,
    ) -> Result<Vec<*mut c_void>> {
        inputs
            .iter()
            .zip(self.input_kinds.iter())
            .zip(self.staged_inputs.iter())
            .map(|((input, kind), staged)| match (input, kind, staged) {
                (InputBinding::Host(input), PreparedInputKind::Host, _) => {
                    Ok(input.bytes.as_ptr().cast_mut().cast())
                }
                (
                    InputBinding::HostToDevice(input),
                    PreparedInputKind::HostToDevice,
                    Some(buffer),
                ) => {
                    buffer.copy_from_host(input.bytes, stream)?;
                    Ok(buffer.as_mut_ptr())
                }
                (InputBinding::Device(input), PreparedInputKind::Device, _) => {
                    Ok(input.ptr.cast_mut())
                }
                _ => Err(Error::InvalidShape {
                    tensor: input.name().to_owned(),
                    reason: "prepared mixed binding kind changed".to_owned(),
                }),
            })
            .collect()
    }
}

/// In-flight TensorRT work launched from a [`PreparedDeviceSession`].
pub struct PreparedEnqueuedInference<'run, 'stream> {
    _context: &'run mut ExecutionContext,
    stream: CudaStream<'stream>,
    synchronized: bool,
}

impl<'stream> PreparedEnqueuedInference<'_, 'stream> {
    pub fn stream(&self) -> CudaStream<'stream> {
        self.stream
    }

    pub fn synchronize(mut self) -> Result<()> {
        synchronize_stream(self.stream)?;
        self.synchronized = true;
        Ok(())
    }

    /// Marks this in-flight launch as already synchronized by another stream
    /// boundary.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all work enqueued before this inference
    /// on `stream` has completed. This is useful when a downstream launch on
    /// the same stream has already been synchronized.
    pub unsafe fn assume_synchronized(mut self) {
        self.synchronized = true;
    }
}

impl Drop for PreparedEnqueuedInference<'_, '_> {
    fn drop(&mut self) {
        if !self.synchronized {
            let _ = synchronize_stream(self.stream);
        }
    }
}

fn validate_prepared_inputs(
    expected_infos: &[TensorInfo],
    expected_bytes: &[usize],
    inputs: &[DeviceInputTensor<'_>],
) -> Result<()> {
    if inputs.len() != expected_infos.len() {
        return Err(Error::InvalidShape {
            tensor: "<prepared-inputs>".to_owned(),
            reason: format!(
                "expected {} input bindings, got {}",
                expected_infos.len(),
                inputs.len()
            ),
        });
    }

    for ((expected, expected_bytes), input) in
        expected_infos.iter().zip(expected_bytes.iter()).zip(inputs)
    {
        validate_prepared_binding_name(expected, input.name, "input")?;
        validate_prepared_shape(expected, &input.shape)?;
        validate_prepared_device_bytes(expected, *expected_bytes, input.data_type, input.bytes)?;
        if input.ptr.is_null() {
            return Err(Error::InvalidShape {
                tensor: input.name.to_owned(),
                reason: "device input pointer is null".to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_prepared_mixed_inputs(
    expected_infos: &[TensorInfo],
    expected_bytes: &[usize],
    expected_kinds: &[PreparedInputKind],
    inputs: &[InputBinding<'_>],
) -> Result<()> {
    if inputs.len() != expected_infos.len() {
        return Err(Error::InvalidShape {
            tensor: "<prepared-mixed-inputs>".to_owned(),
            reason: format!(
                "expected {} input bindings, got {}",
                expected_infos.len(),
                inputs.len()
            ),
        });
    }

    for (((expected, expected_bytes), expected_kind), input) in expected_infos
        .iter()
        .zip(expected_bytes.iter())
        .zip(expected_kinds.iter())
        .zip(inputs.iter())
    {
        validate_prepared_binding_name(expected, input.name(), "mixed input")?;
        if input_kind(input) != *expected_kind {
            return Err(Error::InvalidShape {
                tensor: input.name().to_owned(),
                reason: "prepared mixed binding kind changed".to_owned(),
            });
        }

        match input {
            InputBinding::Host(input) | InputBinding::HostToDevice(input) => {
                validate_prepared_shape(expected, &input.shape)?;
                if let Some(data_type) = input.expected_data_type
                    && expected.data_type != data_type
                {
                    return Err(Error::TypeMismatch {
                        tensor: expected.name.clone(),
                        expected: expected.data_type,
                        actual: data_type,
                    });
                }
                if input.bytes.len() != *expected_bytes {
                    return Err(Error::SizeMismatch {
                        tensor: expected.name.clone(),
                        expected_bytes: *expected_bytes,
                        actual_bytes: input.bytes.len(),
                    });
                }
            }
            InputBinding::Device(input) => {
                validate_prepared_shape(expected, &input.shape)?;
                validate_prepared_device_bytes(
                    expected,
                    *expected_bytes,
                    input.data_type,
                    input.bytes,
                )?;
                if input.ptr.is_null() {
                    return Err(Error::InvalidShape {
                        tensor: input.name.to_owned(),
                        reason: "device input pointer is null".to_owned(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_prepared_outputs(
    expected_infos: &[TensorInfo],
    expected_bytes: &[usize],
    outputs: &[DeviceOutputTensor<'_>],
) -> Result<()> {
    if outputs.len() != expected_infos.len() {
        return Err(Error::InvalidShape {
            tensor: "<prepared-outputs>".to_owned(),
            reason: format!(
                "expected {} output bindings, got {}",
                expected_infos.len(),
                outputs.len()
            ),
        });
    }

    for ((expected, expected_bytes), output) in expected_infos
        .iter()
        .zip(expected_bytes.iter())
        .zip(outputs.iter())
    {
        validate_prepared_binding_name(expected, output.name, "output")?;
        validate_prepared_shape(expected, &output.shape)?;
        validate_prepared_device_bytes(expected, *expected_bytes, output.data_type, output.bytes)?;
        if output.ptr.is_null() {
            return Err(Error::InvalidShape {
                tensor: output.name.to_owned(),
                reason: "device output pointer is null".to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_prepared_binding_name(
    expected: &TensorInfo,
    actual: &str,
    binding_kind: &str,
) -> Result<()> {
    if expected.name == actual {
        Ok(())
    } else {
        Err(Error::InvalidShape {
            tensor: actual.to_owned(),
            reason: format!(
                "prepared {binding_kind} binding order expects {}, got {}",
                expected.name, actual
            ),
        })
    }
}

fn prepared_tensor_bytes(infos: &[TensorInfo]) -> Result<Vec<usize>> {
    infos
        .iter()
        .map(|info| tensor_byte_len(&info.name, &info.shape, info.data_type))
        .collect()
}

fn validate_prepared_shape(expected: &TensorInfo, actual: &Dims) -> Result<()> {
    if expected.shape == *actual {
        Ok(())
    } else {
        Err(Error::InvalidShape {
            tensor: expected.name.clone(),
            reason: format!(
                "prepared shape is {:?}, got {:?}",
                expected.shape.as_slice(),
                actual.as_slice()
            ),
        })
    }
}

fn validate_prepared_device_bytes(
    expected: &TensorInfo,
    expected_bytes: usize,
    data_type: DataType,
    actual_bytes: usize,
) -> Result<()> {
    if expected.data_type != data_type {
        return Err(Error::TypeMismatch {
            tensor: expected.name.clone(),
            expected: expected.data_type,
            actual: data_type,
        });
    }

    if actual_bytes == expected_bytes {
        Ok(())
    } else {
        Err(Error::SizeMismatch {
            tensor: expected.name.clone(),
            expected_bytes,
            actual_bytes,
        })
    }
}

pub(crate) fn input_kind(input: &InputBinding<'_>) -> PreparedInputKind {
    match input {
        InputBinding::Host(_) => PreparedInputKind::Host,
        InputBinding::HostToDevice(_) => PreparedInputKind::HostToDevice,
        InputBinding::Device(_) => PreparedInputKind::Device,
    }
}
