#[cfg(feature = "cuda")]
use super::cuda::{CudaEventHandle, CudaGraphExec, query_event, synchronize_event};
use super::cuda::{CudaStream, DeviceBuffer};
use super::data_type::DataType;
use super::engine::{Engine, ExecutionContext, LogSeverity, Runtime};
use super::error::{Error, Result};
use super::prepared::{PreparedDeviceSession, PreparedMixedDeviceSession, input_kind};
use super::tensor::{
    DeviceInputTensor, DeviceOutputTensor, Dims, HostInputTensor, HostOutputTensor, HostTensor,
    InputBinding, TensorIOMode, TensorInfo, TensorLocation, TensorShape, tensor_byte_len,
};
use crate::candle::{InputTensors, OutputTensors, run_session};
#[cfg(feature = "cuda")]
use candle_core::DeviceLocation;
#[cfg(feature = "cuda")]
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
#[cfg(feature = "cuda")]
use std::rc::Rc;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
mod plan;
mod select;

#[cfg(feature = "cuda")]
pub(crate) use plan::DeviceOutputPlan;
#[cfg(feature = "cuda")]
use plan::{RunPlanInputKey, RunPlanKey};
use select::select_outputs;

#[cfg(feature = "cuda")]
const CUDA_GRAPH_INITIAL_BACKOFF_RUNS: usize = 8;
#[cfg(feature = "cuda")]
const CUDA_GRAPH_MAX_BACKOFF_RUNS: usize = 64;
#[cfg(feature = "cuda")]
const CUDA_GRAPH_MAX_CACHE_ENTRIES: usize = 16;

/// TensorRT session wrapper for named inputs/outputs.
///
/// The high-level API uses `candle_core::Tensor` for named input and output
/// bindings. Lower-level host-buffer and device-pointer entry points remain
/// available for callers that need explicit memory control.
pub struct Session {
    context_pool: RefCell<Vec<ExecutionContext>>,
    #[cfg(feature = "cuda")]
    in_flight_contexts: RefCell<Vec<InFlightContext>>,
    #[cfg(feature = "cuda")]
    event_pool: RefCell<HashMap<DeviceLocation, Vec<CudaEventHandle>>>,
    #[cfg(feature = "cuda")]
    run_plan_cache: RefCell<HashMap<RunPlanKey, Arc<[DeviceOutputPlan]>>>,
    #[cfg(feature = "cuda")]
    cuda_graph_cache: RefCell<HashMap<CudaGraphKey, Rc<CudaGraphEntry>>>,
    #[cfg(feature = "cuda")]
    cuda_graph_backoff: RefCell<HashMap<CudaGraphKey, CudaGraphBackoff>>,
    #[cfg(feature = "cuda")]
    in_flight_graphs: RefCell<Vec<InFlightGraph>>,
    io_tensors: Vec<TensorInfo>,
    io_tensor_index: HashMap<String, usize>,
    engine: Engine,
    _runtime: Runtime,
}

/// In-flight TensorRT work enqueued on a CUDA stream.
///
/// This guard keeps the TensorRT execution context alive while the queued work
/// is in flight. Call [`synchronize`](Self::synchronize) to surface CUDA errors;
/// dropping the guard also synchronizes as a safety fallback.
pub struct EnqueuedInference<'stream> {
    _context: ExecutionContext,
    stream: CudaStream<'stream>,
    synchronized: bool,
}

impl<'stream> EnqueuedInference<'stream> {
    pub fn stream(&self) -> CudaStream<'stream> {
        self.stream
    }

    pub fn synchronize(mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.synchronized = true;
        Ok(())
    }
}

impl Drop for EnqueuedInference<'_> {
    fn drop(&mut self) {
        if !self.synchronized {
            let _ = self.stream.synchronize();
        }
    }
}

impl Session {
    pub fn new(engine_buffer: impl AsRef<[u8]>) -> Result<Self> {
        Self::with_log_severity(engine_buffer, LogSeverity::Warning)
    }

    pub fn with_log_severity(
        engine_buffer: impl AsRef<[u8]>,
        min_severity: LogSeverity,
    ) -> Result<Self> {
        let runtime = Runtime::new(min_severity)?;
        let engine = runtime.deserialize_engine(engine_buffer)?;
        let io_tensors = engine.io_tensors()?;
        let io_tensor_index = io_tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        Ok(Self {
            context_pool: RefCell::new(Vec::new()),
            #[cfg(feature = "cuda")]
            in_flight_contexts: RefCell::new(Vec::new()),
            #[cfg(feature = "cuda")]
            event_pool: RefCell::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            run_plan_cache: RefCell::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            cuda_graph_cache: RefCell::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            cuda_graph_backoff: RefCell::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            in_flight_graphs: RefCell::new(Vec::new()),
            io_tensors,
            io_tensor_index,
            engine,
            _runtime: runtime,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn io_tensors(&self) -> &[TensorInfo] {
        &self.io_tensors
    }

    pub fn tensor_byte_len(&self, name: &str, shape: &Dims) -> Result<usize> {
        tensor_byte_len(name, shape, self.tensor_data_type(name)?)
    }

    pub fn infer_shapes(&self, inputs: &[TensorShape<'_>]) -> Result<Vec<TensorInfo>> {
        let mut context = self.pooled_execution_context()?;
        self.infer_shapes_with_context(&mut context, inputs)
    }

    pub fn infer_outputs(&self, inputs: &InputTensors<'_>) -> Result<Vec<TensorInfo>> {
        let shapes = inputs.shape_infos()?;
        self.infer_shapes(&shapes)
    }

    pub(crate) fn pooled_execution_context(&self) -> Result<PooledExecutionContext<'_>> {
        #[cfg(feature = "cuda")]
        {
            self.reclaim_finished_contexts()?;
            self.reclaim_finished_graphs()?;
        }

        let (mut context, reused) = if let Some(context) = self.context_pool.borrow_mut().pop() {
            (context, true)
        } else {
            (self.engine.create_execution_context()?, false)
        };
        if reused {
            self.clear_tensor_addresses(&mut context)?;
        }
        Ok(PooledExecutionContext {
            session: self,
            context: Some(context),
        })
    }

    #[cfg(feature = "cuda")]
    fn reclaim_finished_contexts(&self) -> Result<()> {
        let mut completed_contexts = Vec::new();
        let mut completed_events = Vec::new();
        {
            let mut in_flight = self.in_flight_contexts.borrow_mut();
            let mut index = 0;
            while index < in_flight.len() {
                if query_event(in_flight[index].event.as_event())? {
                    let InFlightContext {
                        context,
                        device_location,
                        event,
                        _keepalive,
                    } = in_flight.swap_remove(index);
                    completed_contexts.push(context);
                    completed_events.push((device_location, event));
                } else {
                    index += 1;
                }
            }
        }

        self.context_pool.borrow_mut().extend(completed_contexts);
        let mut event_pool = self.event_pool.borrow_mut();
        for (device_location, event) in completed_events {
            event_pool.entry(device_location).or_default().push(event);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn reclaim_finished_graphs(&self) -> Result<()> {
        let mut completed_events = Vec::new();
        {
            let mut in_flight = self.in_flight_graphs.borrow_mut();
            let mut index = 0;
            while index < in_flight.len() {
                if query_event(in_flight[index].event.as_event())? {
                    let InFlightGraph {
                        device_location,
                        event,
                        _keepalive,
                        _graph_entry,
                    } = in_flight.swap_remove(index);
                    completed_events.push((device_location, event));
                } else {
                    index += 1;
                }
            }
        }

        let mut event_pool = self.event_pool.borrow_mut();
        for (device_location, event) in completed_events {
            event_pool.entry(device_location).or_default().push(event);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn completion_event(
        &self,
        device_location: DeviceLocation,
    ) -> Result<CudaEventHandle> {
        self.event_pool
            .borrow_mut()
            .get_mut(&device_location)
            .and_then(Vec::pop)
            .map(Ok)
            .unwrap_or_else(|| CudaEventHandle::new_for_device_location(device_location))
    }

    #[cfg(feature = "cuda")]
    fn recycle_completion_event(&self, device_location: DeviceLocation, event: CudaEventHandle) {
        self.event_pool
            .borrow_mut()
            .entry(device_location)
            .or_default()
            .push(event);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn defer_context_until_stream_complete(
        &self,
        context: PooledExecutionContext<'_>,
        device_location: DeviceLocation,
        completion_event: CudaEventHandle,
        stream: CudaStream<'_>,
        keepalive: SmallVec<[Tensor; 8]>,
    ) -> Result<()> {
        let mut in_flight = self.in_flight_contexts.borrow_mut();
        if in_flight.try_reserve(1).is_err() {
            drop(in_flight);
            let _ = stream.synchronize();
            self.recycle_completion_event(device_location, completion_event);
            return Err(Error::AllocationFailed);
        }
        if let Err(error) = completion_event.as_event().record(stream) {
            drop(in_flight);
            let _ = stream.synchronize();
            self.recycle_completion_event(device_location, completion_event);
            return Err(error);
        }

        let context = context.into_inner();
        in_flight.push(InFlightContext {
            context,
            device_location,
            event: completion_event,
            _keepalive: keepalive,
        });
        Ok(())
    }

    pub(crate) fn infer_shapes_with_context(
        &self,
        context: &mut ExecutionContext,
        inputs: &[TensorShape<'_>],
    ) -> Result<Vec<TensorInfo>> {
        set_input_shapes_if_needed(self, context, inputs)?;
        self.inferred_output_infos(context)
    }

    fn inferred_output_infos(&self, context: &ExecutionContext) -> Result<Vec<TensorInfo>> {
        self.io_tensors
            .iter()
            .filter(|tensor| tensor.io_mode == TensorIOMode::Output)
            .cloned()
            .map(|mut tensor| {
                tensor.shape = context.tensor_shape(&tensor.name)?;
                Ok(tensor)
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cached_device_output_plan<'name>(
        &self,
        context: &mut ExecutionContext,
        inputs: &[DeviceInputTensor<'_>],
        output_names: impl IntoIterator<Item = &'name str>,
    ) -> Result<Arc<[DeviceOutputPlan]>> {
        let key = self.device_run_plan_key(inputs, output_names)?;

        if let Some(output_infos) = self.run_plan_cache.borrow().get(&key).cloned() {
            set_device_input_shapes_if_needed(context, inputs)?;
            return Ok(output_infos);
        }

        self.validate_device_inputs(inputs)?;
        set_device_input_shapes_if_needed(context, inputs)?;
        let output_infos = self.inferred_output_infos_for_indices(context, &key.outputs)?;
        let output_infos = output_infos
            .into_iter()
            .map(|info| self.device_output_plan(info))
            .collect::<Result<Vec<_>>>()?;
        let output_infos: Arc<[DeviceOutputPlan]> = Arc::from(output_infos.into_boxed_slice());
        self.run_plan_cache
            .borrow_mut()
            .insert(key, output_infos.clone());
        Ok(output_infos)
    }

    #[cfg(feature = "cuda")]
    fn device_run_plan_key<'name>(
        &self,
        inputs: &[DeviceInputTensor<'_>],
        output_names: impl IntoIterator<Item = &'name str>,
    ) -> Result<RunPlanKey> {
        let input_indices: SmallVec<[usize; 4]> = inputs
            .iter()
            .map(|input| self.tensor_index(input.name))
            .collect::<Result<_>>()?;
        let output_indices: SmallVec<[usize; 4]> = output_names
            .into_iter()
            .map(|name| self.tensor_index(name))
            .collect::<Result<_>>()?;

        Ok(RunPlanKey::new(
            input_indices
                .iter()
                .copied()
                .zip(inputs.iter())
                .map(|(index, input)| RunPlanInputKey::new(index, input)),
            output_indices,
        ))
    }

    #[cfg(feature = "cuda")]
    fn inferred_output_infos_for_indices(
        &self,
        context: &ExecutionContext,
        output_indices: &[usize],
    ) -> Result<Vec<TensorInfo>> {
        if output_indices.is_empty() {
            return self.inferred_output_infos(context);
        }

        output_indices
            .iter()
            .map(|&index| {
                let mut tensor =
                    self.io_tensors
                        .get(index)
                        .cloned()
                        .ok_or_else(|| Error::InvalidShape {
                            tensor: format!("<tensor-index:{index}>"),
                            reason: "tensor index is not an engine I/O tensor".to_owned(),
                        })?;
                if tensor.io_mode != TensorIOMode::Output {
                    return Err(Error::TensorModeMismatch {
                        tensor: tensor.name,
                        expected: TensorIOMode::Output,
                        actual: tensor.io_mode,
                    });
                }
                tensor.shape = context.tensor_shape(&tensor.name)?;
                Ok(tensor)
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn validate_device_inputs(&self, inputs: &[DeviceInputTensor<'_>]) -> Result<()> {
        for input in inputs {
            self.validate_tensor_shape(input.name, &input.shape)?;
            self.ensure_tensor_mode(input.name, TensorIOMode::Input)?;
            self.ensure_tensor_type(input.name, input.data_type)?;
            self.validate_device_bytes(input.name, &input.shape, input.data_type, input.bytes)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn device_output_plan(&self, info: TensorInfo) -> Result<DeviceOutputPlan> {
        let bytes = tensor_byte_len(&info.name, &info.shape, info.data_type)?;
        Ok(DeviceOutputPlan {
            name: info.name,
            data_type: info.data_type,
            shape: info.shape,
            bytes,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) unsafe fn enqueue_device_auto_graph(
        &self,
        context: PooledExecutionContext<'_>,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
        completion: DeviceRunCompletion,
    ) -> Result<()> {
        let DeviceRunCompletion {
            device_location,
            event: completion_event,
            keepalive,
        } = completion;

        self.reclaim_finished_graphs()?;
        let key = self.cuda_graph_key(inputs, outputs, stream)?;

        // CUDA Graph replay is only valid for the exact stream and concrete
        // device addresses captured below. New tensors or a different stream
        // intentionally fall back to a normal TensorRT enqueue.
        if let Some(entry) = self.cuda_graph_cache.borrow().get(&key).cloned() {
            if let Err(error) = entry.exec.launch(stream) {
                self.cuda_graph_cache.borrow_mut().remove(&key);
                self.recycle_completion_event(device_location, completion_event);
                return Err(error);
            }
            return self.defer_graph_until_stream_complete(
                device_location,
                completion_event,
                stream,
                keepalive,
                entry,
            );
        }

        if self.in_flight_graphs.borrow().is_empty() {
            if self.should_skip_cuda_graph_capture(&key) {
                let mut context = context;
                unsafe {
                    self.enqueue_device_with_context(&mut context, inputs, outputs, stream)?;
                }
                return self.defer_context_until_stream_complete(
                    context,
                    device_location,
                    completion_event,
                    stream,
                    keepalive,
                );
            }

            match unsafe {
                self.capture_device_graph(context, inputs, outputs, stream, key.clone())
            } {
                Ok(()) => {
                    self.clear_cuda_graph_capture_backoff(&key);
                    let entry = self
                        .cuda_graph_cache
                        .borrow()
                        .get(&key)
                        .cloned()
                        .expect("captured graph was not cached");
                    if let Err(error) = entry.exec.launch(stream) {
                        self.cuda_graph_cache.borrow_mut().remove(&key);
                        self.recycle_completion_event(device_location, completion_event);
                        return Err(error);
                    }
                    return self.defer_graph_until_stream_complete(
                        device_location,
                        completion_event,
                        stream,
                        keepalive,
                        entry,
                    );
                }
                Err(error) => {
                    let GraphCaptureError {
                        mut context,
                        error: _capture_error,
                    } = *error;
                    self.record_cuda_graph_capture_failure(key);
                    unsafe {
                        self.enqueue_device_with_context(&mut context, inputs, outputs, stream)?;
                    }
                    return self.defer_context_until_stream_complete(
                        context,
                        device_location,
                        completion_event,
                        stream,
                        keepalive,
                    );
                }
            }
        }

        let mut context = context;
        unsafe {
            self.enqueue_device_with_context(&mut context, inputs, outputs, stream)?;
        }
        self.defer_context_until_stream_complete(
            context,
            device_location,
            completion_event,
            stream,
            keepalive,
        )
    }

    #[cfg(feature = "cuda")]
    fn should_skip_cuda_graph_capture(&self, key: &CudaGraphKey) -> bool {
        let mut backoff = self.cuda_graph_backoff.borrow_mut();
        let Some(backoff) = backoff.get_mut(key) else {
            return false;
        };
        if backoff.remaining_runs == 0 {
            return false;
        }

        backoff.remaining_runs -= 1;
        true
    }

    #[cfg(feature = "cuda")]
    fn record_cuda_graph_capture_failure(&self, key: CudaGraphKey) {
        self.cuda_graph_backoff
            .borrow_mut()
            .entry(key)
            .and_modify(CudaGraphBackoff::back_off_again)
            .or_insert_with(CudaGraphBackoff::new);
    }

    #[cfg(feature = "cuda")]
    fn clear_cuda_graph_capture_backoff(&self, key: &CudaGraphKey) {
        self.cuda_graph_backoff.borrow_mut().remove(key);
    }

    #[cfg(feature = "cuda")]
    fn cuda_graph_key(
        &self,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &[DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<CudaGraphKey> {
        let input_indices: SmallVec<[usize; 4]> = inputs
            .iter()
            .map(|input| self.tensor_index(input.name))
            .collect::<Result<_>>()?;
        let output_indices: SmallVec<[usize; 4]> = outputs
            .iter()
            .map(|output| self.tensor_index(output.name))
            .collect::<Result<_>>()?;

        Ok(CudaGraphKey::new(
            input_indices
                .iter()
                .copied()
                .zip(inputs.iter())
                .map(|(index, input)| CudaGraphTensorKey::from_input(index, input)),
            output_indices
                .iter()
                .copied()
                .zip(outputs.iter())
                .map(|(index, output)| CudaGraphTensorKey::from_output(index, output)),
            stream,
        ))
    }

    #[cfg(feature = "cuda")]
    unsafe fn capture_device_graph<'session>(
        &self,
        mut context: PooledExecutionContext<'session>,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
        key: CudaGraphKey,
    ) -> std::result::Result<(), Box<GraphCaptureError<'session>>> {
        if let Err(error) =
            unsafe { self.bind_device_tensors_on_context(&mut context, inputs, outputs) }
        {
            return Err(Box::new(GraphCaptureError { context, error }));
        }

        let exec = match CudaGraphExec::capture(stream, || unsafe { context.enqueue_v3(stream) }) {
            Ok(exec) => exec,
            Err(error) => return Err(Box::new(GraphCaptureError { context, error })),
        };

        let context = context.into_inner();
        self.cache_cuda_graph(
            key,
            CudaGraphEntry {
                _context: context,
                exec,
            },
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cache_cuda_graph(&self, key: CudaGraphKey, entry: CudaGraphEntry) {
        let mut cache = self.cuda_graph_cache.borrow_mut();
        if cache.len() >= CUDA_GRAPH_MAX_CACHE_ENTRIES
            && !cache.contains_key(&key)
            && let Some(oldest_key) = cache.keys().next().cloned()
        {
            cache.remove(&oldest_key);
        }
        cache.insert(key, Rc::new(entry));
    }

    #[cfg(feature = "cuda")]
    fn defer_graph_until_stream_complete(
        &self,
        device_location: DeviceLocation,
        completion_event: CudaEventHandle,
        stream: CudaStream<'_>,
        keepalive: SmallVec<[Tensor; 8]>,
        graph_entry: Rc<CudaGraphEntry>,
    ) -> Result<()> {
        let mut in_flight = self.in_flight_graphs.borrow_mut();
        if in_flight.try_reserve(1).is_err() {
            drop(in_flight);
            let _ = stream.synchronize();
            self.recycle_completion_event(device_location, completion_event);
            return Err(Error::AllocationFailed);
        }
        if let Err(error) = completion_event.as_event().record(stream) {
            drop(in_flight);
            let _ = stream.synchronize();
            self.recycle_completion_event(device_location, completion_event);
            return Err(error);
        }

        in_flight.push(InFlightGraph {
            device_location,
            event: completion_event,
            _keepalive: keepalive,
            _graph_entry: graph_entry,
        });
        Ok(())
    }

    fn clear_tensor_addresses(&self, context: &mut ExecutionContext) -> Result<()> {
        for tensor in &self.io_tensors {
            unsafe {
                context.set_tensor_address(&tensor.name, std::ptr::null_mut())?;
            }
        }
        Ok(())
    }

    fn tensor_info(&self, tensor: &str) -> Result<&TensorInfo> {
        let index = self.tensor_index(tensor)?;
        Ok(&self.io_tensors[index])
    }

    fn tensor_index(&self, tensor: &str) -> Result<usize> {
        self.io_tensor_index
            .get(tensor)
            .copied()
            .ok_or_else(|| Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: "tensor is not an engine I/O tensor".to_owned(),
            })
    }

    fn tensor_data_type(&self, tensor: &str) -> Result<DataType> {
        Ok(self.tensor_info(tensor)?.data_type)
    }

    fn input_info_from_shape(&self, input: &TensorShape<'_>) -> Result<TensorInfo> {
        let info = self.tensor_info(input.name)?;
        Ok(TensorInfo::with_io_mode(
            input.name,
            info.data_type,
            input.shape.clone(),
            TensorIOMode::Input,
        ))
    }

    fn validate_tensor_shape(&self, tensor: &str, shape: &Dims) -> Result<()> {
        let engine_shape = &self.tensor_info(tensor)?.shape;
        if engine_shape.rank() != shape.rank() {
            return Err(Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: format!(
                    "expected rank {}, got rank {}",
                    engine_shape.rank(),
                    shape.rank()
                ),
            });
        }
        for (axis, (engine_dim, dim)) in engine_shape
            .as_slice()
            .iter()
            .zip(shape.as_slice().iter())
            .enumerate()
        {
            if *engine_dim >= 0 && engine_dim != dim {
                return Err(Error::InvalidShape {
                    tensor: tensor.to_owned(),
                    reason: format!("axis {axis} is static at {engine_dim}, but requested {dim}"),
                });
            }
        }
        if shape.is_dynamic() {
            return Err(Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: format!(
                    "runtime bindings require concrete dimensions, got {:?}",
                    shape.as_slice()
                ),
            });
        }
        Ok(())
    }

    fn ensure_tensor_mode(&self, tensor: &str, expected: TensorIOMode) -> Result<()> {
        let actual = self.tensor_info(tensor)?.io_mode;
        if actual == expected {
            Ok(())
        } else {
            Err(Error::TensorModeMismatch {
                tensor: tensor.to_owned(),
                expected,
                actual,
            })
        }
    }

    fn ensure_tensor_type(&self, tensor: &str, expected: DataType) -> Result<()> {
        let actual = self.tensor_data_type(tensor)?;
        if actual == expected {
            Ok(())
        } else {
            Err(Error::TypeMismatch {
                tensor: tensor.to_owned(),
                expected,
                actual,
            })
        }
    }

    fn validate_device_bytes(
        &self,
        tensor: &str,
        shape: &Dims,
        data_type: DataType,
        actual_bytes: usize,
    ) -> Result<()> {
        let engine_data_type = self.tensor_data_type(tensor)?;
        if engine_data_type != data_type {
            return Err(Error::TypeMismatch {
                tensor: tensor.to_owned(),
                expected: engine_data_type,
                actual: data_type,
            });
        }

        let expected_bytes = tensor_byte_len(tensor, shape, data_type)?;
        if actual_bytes == expected_bytes {
            Ok(())
        } else {
            Err(Error::SizeMismatch {
                tensor: tensor.to_owned(),
                expected_bytes,
                actual_bytes,
            })
        }
    }

    /// Prepares a reusable TensorRT execution context for CUDA device bindings.
    ///
    /// This resolves dynamic input and output shapes once. The returned runner
    /// is intended for hot paths with stable shapes and caller-managed CUDA
    /// buffers.
    pub fn prepare_device(
        &self,
        inputs: &[TensorShape<'_>],
        output_names: &[&str],
    ) -> Result<PreparedDeviceSession> {
        let mut context = self.engine.create_execution_context()?;
        set_input_shapes(self, &mut context, inputs)?;

        let input_infos = inputs
            .iter()
            .map(|input| self.input_info_from_shape(input))
            .collect::<Result<Vec<_>>>()?;

        let output_infos = self.inferred_output_infos(&context)?;
        let output_infos = select_outputs(output_infos, output_names)?;

        PreparedDeviceSession::new(context, input_infos, output_infos)
    }

    /// Prepares a reusable TensorRT execution context for mixed input bindings.
    ///
    /// This is the low-level prepared variant of
    /// [`run_mixed_to_device`](Self::run_mixed_to_device). The concrete input
    /// shapes and binding kinds are fixed here; per-run calls may change host
    /// bytes and device pointers but not binding order, dtype, shape, or
    /// host/staged/device kind.
    pub fn prepare_mixed_device(
        &self,
        inputs: &[InputBinding<'_>],
        output_names: &[&str],
    ) -> Result<PreparedMixedDeviceSession> {
        let input_shapes = inputs
            .iter()
            .map(InputBinding::shape_info)
            .collect::<Vec<_>>();
        let mut context = self.engine.create_execution_context()?;
        set_input_shapes_if_needed(self, &mut context, &input_shapes)?;

        let input_infos = input_shapes
            .iter()
            .map(|input| self.input_info_from_shape(input))
            .collect::<Result<Vec<_>>>()?;
        let input_kinds = inputs.iter().map(input_kind).collect::<Vec<_>>();

        let output_infos = self.inferred_output_infos(&context)?;
        let output_infos = select_outputs(output_infos, output_names)?;

        PreparedMixedDeviceSession::new(context, input_infos, output_infos, input_kinds)
    }

    /// Enqueues TensorRT inference with Candle inputs and preallocated Candle outputs.
    ///
    /// Inputs must be Candle CUDA tensors on the same device as `stream`.
    /// Outputs must be preallocated Candle CUDA tensors on that same device.
    /// The inference is ordered on `stream`; synchronize the stream or device
    /// before reading outputs on the host or using them from another stream.
    /// `Session` keeps the bound Candle tensor storage alive internally until
    /// the enqueued work has completed.
    ///
    pub fn run(
        &self,
        inputs: &InputTensors<'_>,
        outputs: &mut OutputTensors,
        stream: CudaStream<'_>,
    ) -> Result<()> {
        run_session(self, inputs, outputs, stream)
    }

    /// Runs TensorRT inference and allocates host output buffers from inferred shapes.
    ///
    /// `output_names` controls which outputs are returned and in what order. An
    /// empty slice returns every output tensor reported by the engine.
    ///
    /// # Safety
    ///
    /// `stream` must be valid for the active CUDA context, or
    /// [`CudaStream::DEFAULT`].
    pub unsafe fn run_allocating(
        &self,
        inputs: &[HostInputTensor<'_>],
        output_names: &[&str],
        stream: CudaStream<'_>,
    ) -> Result<Vec<HostTensor>> {
        let input_shapes = inputs
            .iter()
            .map(HostInputTensor::shape_info)
            .collect::<Vec<_>>();
        let output_infos = self.infer_shapes(&input_shapes)?;
        let output_infos = select_outputs(output_infos, output_names)?;

        let mut output_storage = output_infos
            .into_iter()
            .map(|info| {
                let bytes = tensor_byte_len(&info.name, &info.shape, info.data_type)?;
                Ok((info, vec![0u8; bytes]))
            })
            .collect::<Result<Vec<_>>>()?;

        {
            let mut outputs = output_storage
                .iter_mut()
                .map(|(info, bytes)| {
                    HostOutputTensor::new(&info.name, info.shape.clone(), bytes.as_mut_slice())
                })
                .collect::<Vec<_>>();
            unsafe {
                self.run_host(inputs, &mut outputs, stream)?;
            }
        }

        Ok(output_storage
            .into_iter()
            .map(|(info, bytes)| HostTensor {
                name: info.name,
                shape: info.shape,
                data_type: info.data_type,
                bytes,
            })
            .collect())
    }

    /// Runs TensorRT inference with named host input/output buffers.
    ///
    /// Input and output buffers are host memory. The session allocates temporary
    /// CUDA buffers, copies inputs to device, enqueues inference, copies outputs
    /// back to host, and synchronizes `stream` before returning.
    ///
    /// # Safety
    ///
    /// `stream` must be valid for the active CUDA context, or
    /// [`CudaStream::DEFAULT`].
    pub unsafe fn run_host(
        &self,
        inputs: &[HostInputTensor<'_>],
        outputs: &mut [HostOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let input_shapes = inputs
            .iter()
            .map(HostInputTensor::shape_info)
            .collect::<Vec<_>>();
        let mut context = self.pooled_execution_context()?;
        set_input_shapes_if_needed(self, &mut context, &input_shapes)?;

        let input_bindings = self.stage_input_bindings(inputs, stream)?;

        for (input, binding) in inputs.iter().zip(input_bindings.iter()) {
            unsafe {
                context.set_tensor_address(input.name, binding.as_mut_ptr())?;
            }
        }

        let output_buffers = outputs
            .iter()
            .map(|output| {
                self.validate_tensor_shape(output.name, &output.shape)?;
                self.ensure_tensor_mode(output.name, TensorIOMode::Output)?;
                let actual_shape = context.tensor_shape(output.name)?;
                if actual_shape != output.shape {
                    return Err(Error::InvalidShape {
                        tensor: output.name.to_owned(),
                        reason: format!(
                            "expected output shape {:?}, got {:?}",
                            output.shape.as_slice(),
                            actual_shape.as_slice()
                        ),
                    });
                }

                let expected_bytes = self.tensor_byte_len(output.name, &output.shape)?;
                if output.bytes.len() != expected_bytes {
                    return Err(Error::SizeMismatch {
                        tensor: output.name.to_owned(),
                        expected_bytes,
                        actual_bytes: output.bytes.len(),
                    });
                }
                DeviceBuffer::new(expected_bytes)
            })
            .collect::<Result<Vec<_>>>()?;

        for (output, buffer) in outputs.iter().zip(output_buffers.iter()) {
            unsafe {
                context.set_tensor_address(output.name, buffer.as_mut_ptr())?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }

        for (output, buffer) in outputs.iter_mut().zip(output_buffers.iter()) {
            buffer.copy_to_host(output.bytes, stream)?;
        }
        stream.synchronize()
    }

    /// Runs TensorRT inference with host/device inputs and caller-owned CUDA
    /// device outputs.
    ///
    /// This is the mixed-binding counterpart to [`run_host`](Self::run_host): inputs use
    /// [`HostInputTensor::location`](HostInputTensor) to choose host pointers or
    /// temporary device staging buffers, while outputs are bound directly to
    /// CUDA device pointers supplied by the caller. The stream is synchronized
    /// before returning so temporary input staging buffers remain alive for the
    /// whole launch.
    ///
    /// # Safety
    ///
    /// `stream` must be valid for the active CUDA context, or
    /// [`CudaStream::DEFAULT`]. Every output device pointer must be non-null,
    /// valid for at least the supplied byte length, and compatible with the
    /// tensor dtype and shape.
    pub unsafe fn run_to_device(
        &self,
        inputs: &[HostInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let input_shapes = inputs
            .iter()
            .map(HostInputTensor::shape_info)
            .collect::<Vec<_>>();
        let mut context = self.pooled_execution_context()?;
        set_input_shapes_if_needed(self, &mut context, &input_shapes)?;

        let input_bindings = self.stage_input_bindings(inputs, stream)?;

        for (input, binding) in inputs.iter().zip(input_bindings.iter()) {
            unsafe {
                context.set_tensor_address(input.name, binding.as_mut_ptr())?;
            }
        }

        for output in outputs.iter() {
            self.validate_tensor_shape(output.name, &output.shape)?;
            self.ensure_tensor_mode(output.name, TensorIOMode::Output)?;
            self.ensure_tensor_type(output.name, output.data_type)?;

            let actual_shape = context.tensor_shape(output.name)?;
            if actual_shape != output.shape {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: format!(
                        "expected output shape {:?}, got {:?}",
                        output.shape.as_slice(),
                        actual_shape.as_slice()
                    ),
                });
            }

            self.validate_device_bytes(output.name, &output.shape, output.data_type, output.bytes)?;
            if output.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: "device output pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }
        stream.synchronize()
    }

    /// Runs TensorRT inference with mixed host/staged/device inputs and
    /// caller-owned CUDA device outputs.
    ///
    /// This is the direct mixed-input path for engines that need a few small
    /// host control tensors and one or more large CUDA tensors. Host-to-device
    /// inputs are staged into temporary CUDA buffers, host inputs are bound as
    /// host pointers, and device inputs are bound directly without a copy. The
    /// stream is synchronized before returning so temporary staging buffers stay
    /// alive for the whole launch.
    ///
    /// # Safety
    ///
    /// `stream` must be valid for the active CUDA context, or
    /// [`CudaStream::DEFAULT`]. Every device input/output pointer must be
    /// non-null, valid for at least the supplied byte length, and compatible
    /// with the tensor dtype and shape.
    pub unsafe fn run_mixed_to_device(
        &self,
        inputs: &[InputBinding<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let input_shapes = inputs
            .iter()
            .map(InputBinding::shape_info)
            .collect::<Vec<_>>();
        let mut context = self.pooled_execution_context()?;
        set_input_shapes_if_needed(self, &mut context, &input_shapes)?;

        let input_bindings = self.stage_mixed_input_bindings(inputs, stream)?;

        for (input, binding) in inputs.iter().zip(input_bindings.iter()) {
            unsafe {
                context.set_tensor_address(input.name(), binding.as_mut_ptr())?;
            }
        }

        for output in outputs.iter() {
            self.validate_tensor_shape(output.name, &output.shape)?;
            self.ensure_tensor_mode(output.name, TensorIOMode::Output)?;
            self.ensure_tensor_type(output.name, output.data_type)?;

            let actual_shape = context.tensor_shape(output.name)?;
            if actual_shape != output.shape {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: format!(
                        "expected output shape {:?}, got {:?}",
                        output.shape.as_slice(),
                        actual_shape.as_slice()
                    ),
                });
            }

            self.validate_device_bytes(output.name, &output.shape, output.data_type, output.bytes)?;
            if output.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: "device output pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }
        stream.synchronize()
    }

    /// Enqueues TensorRT inference with named CUDA device input/output buffers.
    ///
    /// This path does not allocate staging buffers or copy through host memory.
    /// Callers must pass pointers that are valid for the active CUDA context and
    /// remain alive until the returned [`EnqueuedInference`] is synchronized or
    /// the caller has otherwise established stream completion. Shapes and dtypes
    /// are validated against the engine before TensorRT is enqueued.
    ///
    /// # Safety
    ///
    /// `stream` must be valid for the active CUDA context, or
    /// [`CudaStream::DEFAULT`]. Every device pointer must be non-null, valid for
    /// at least the supplied byte length, and compatible with the tensor dtype
    /// and shape.
    pub unsafe fn enqueue_device<'stream>(
        &self,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'stream>,
    ) -> Result<EnqueuedInference<'stream>> {
        let input_shapes = inputs
            .iter()
            .map(DeviceInputTensor::shape_info)
            .collect::<Vec<_>>();
        let mut context = self.engine.create_execution_context()?;
        set_input_shapes(self, &mut context, &input_shapes)?;
        unsafe {
            self.enqueue_device_on_context(&mut context, inputs, outputs, stream)?;
        }
        Ok(EnqueuedInference {
            _context: context,
            stream,
            synchronized: false,
        })
    }

    /// Runs TensorRT inference with named CUDA device input/output buffers and
    /// synchronizes `stream` before returning.
    ///
    /// Use [`enqueue_device`](Self::enqueue_device) when the caller can keep the
    /// returned in-flight guard alive and express completion through CUDA stream
    /// dependencies instead of host synchronization.
    ///
    /// # Safety
    ///
    /// Same requirements as [`enqueue_device`](Self::enqueue_device).
    pub unsafe fn run_device(
        &self,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        let inference = unsafe { self.enqueue_device(inputs, outputs, stream)? };
        inference.synchronize()
    }

    #[cfg(feature = "cuda")]
    pub(crate) unsafe fn enqueue_device_with_context(
        &self,
        context: &mut ExecutionContext,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        unsafe {
            self.bind_device_tensors_on_context(context, inputs, outputs)?;
        }
        unsafe {
            context.enqueue_v3(stream)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    unsafe fn bind_device_tensors_on_context(
        &self,
        context: &mut ExecutionContext,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &[DeviceOutputTensor<'_>],
    ) -> Result<()> {
        for input in inputs {
            if input.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: input.name.to_owned(),
                    reason: "device input pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(input.name, input.ptr.cast_mut())?;
            }
        }

        for output in outputs.iter() {
            if output.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: "device output pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }
        Ok(())
    }

    unsafe fn enqueue_device_on_context(
        &self,
        context: &mut ExecutionContext,
        inputs: &[DeviceInputTensor<'_>],
        outputs: &mut [DeviceOutputTensor<'_>],
        stream: CudaStream<'_>,
    ) -> Result<()> {
        for input in inputs {
            self.validate_tensor_shape(input.name, &input.shape)?;
            self.ensure_tensor_mode(input.name, TensorIOMode::Input)?;
            self.ensure_tensor_type(input.name, input.data_type)?;
            self.validate_device_bytes(input.name, &input.shape, input.data_type, input.bytes)?;
            if input.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: input.name.to_owned(),
                    reason: "device input pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(input.name, input.ptr.cast_mut())?;
            }
        }

        for output in outputs.iter() {
            self.validate_tensor_shape(output.name, &output.shape)?;
            self.ensure_tensor_mode(output.name, TensorIOMode::Output)?;
            self.ensure_tensor_type(output.name, output.data_type)?;

            let actual_shape = context.tensor_shape(output.name)?;
            if actual_shape != output.shape {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: format!(
                        "expected output shape {:?}, got {:?}",
                        output.shape.as_slice(),
                        actual_shape.as_slice()
                    ),
                });
            }

            self.validate_device_bytes(output.name, &output.shape, output.data_type, output.bytes)?;
            if output.ptr.is_null() {
                return Err(Error::InvalidShape {
                    tensor: output.name.to_owned(),
                    reason: "device output pointer is null".to_owned(),
                });
            }
            unsafe {
                context.set_tensor_address(output.name, output.ptr)?;
            }
        }

        unsafe {
            context.enqueue_v3(stream)?;
        }
        Ok(())
    }

    fn stage_input_bindings<'input>(
        &self,
        inputs: &[HostInputTensor<'input>],
        stream: CudaStream<'_>,
    ) -> Result<Vec<BoundInputBinding<'input>>> {
        inputs
            .iter()
            .map(|input| self.stage_input_tensor(input, stream))
            .collect()
    }

    fn stage_mixed_input_bindings<'input>(
        &self,
        inputs: &[InputBinding<'input>],
        stream: CudaStream<'_>,
    ) -> Result<Vec<BoundInputBinding<'input>>> {
        inputs
            .iter()
            .map(|input| match input {
                InputBinding::Host(input) => {
                    let mut input = input.clone();
                    input.location = TensorLocation::Host;
                    self.stage_input_tensor(&input, stream)
                }
                InputBinding::HostToDevice(input) => {
                    let mut input = input.clone();
                    input.location = TensorLocation::Device;
                    self.stage_input_tensor(&input, stream)
                }
                InputBinding::Device(input) => {
                    self.validate_tensor_shape(input.name, &input.shape)?;
                    self.ensure_tensor_mode(input.name, TensorIOMode::Input)?;
                    self.ensure_tensor_type(input.name, input.data_type)?;
                    self.validate_device_bytes(
                        input.name,
                        &input.shape,
                        input.data_type,
                        input.bytes,
                    )?;
                    if input.ptr.is_null() {
                        return Err(Error::InvalidShape {
                            tensor: input.name.to_owned(),
                            reason: "device input pointer is null".to_owned(),
                        });
                    }
                    Ok(BoundInputBinding::DevicePtr(input.ptr.cast_mut()))
                }
            })
            .collect()
    }

    fn stage_input_tensor<'input>(
        &self,
        input: &HostInputTensor<'input>,
        stream: CudaStream<'_>,
    ) -> Result<BoundInputBinding<'input>> {
        self.validate_tensor_shape(input.name, &input.shape)?;
        self.ensure_tensor_mode(input.name, TensorIOMode::Input)?;
        if let Some(expected) = input.expected_data_type {
            self.ensure_tensor_type(input.name, expected)?;
        }
        let expected_bytes = self.tensor_byte_len(input.name, &input.shape)?;
        if input.bytes.len() != expected_bytes {
            return Err(Error::SizeMismatch {
                tensor: input.name.to_owned(),
                expected_bytes,
                actual_bytes: input.bytes.len(),
            });
        }
        match input.location {
            TensorLocation::Device => {
                let buffer = DeviceBuffer::new(expected_bytes)?;
                buffer.copy_from_host(input.bytes, stream)?;
                Ok(BoundInputBinding::StagedDevice(buffer))
            }
            TensorLocation::Host => Ok(BoundInputBinding::Host(input.bytes)),
        }
    }
}

impl AsRef<Session> for Session {
    fn as_ref(&self) -> &Session {
        self
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            for in_flight in self.in_flight_contexts.get_mut().drain(..) {
                let _ = synchronize_event(in_flight.event.as_event());
            }
            for in_flight in self.in_flight_graphs.get_mut().drain(..) {
                let _ = synchronize_event(in_flight.event.as_event());
            }
            self.cuda_graph_cache.get_mut().clear();
            self.cuda_graph_backoff.get_mut().clear();
            self.event_pool.get_mut().clear();
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DeviceRunCompletion {
    device_location: DeviceLocation,
    event: CudaEventHandle,
    keepalive: SmallVec<[Tensor; 8]>,
}

#[cfg(feature = "cuda")]
impl DeviceRunCompletion {
    pub(crate) fn new(
        device_location: DeviceLocation,
        event: CudaEventHandle,
        keepalive: SmallVec<[Tensor; 8]>,
    ) -> Self {
        Self {
            device_location,
            event,
            keepalive,
        }
    }
}

#[cfg(feature = "cuda")]
struct GraphCaptureError<'session> {
    context: PooledExecutionContext<'session>,
    error: Error,
}

#[cfg(feature = "cuda")]
pub(crate) struct InFlightContext {
    context: ExecutionContext,
    device_location: DeviceLocation,
    event: CudaEventHandle,
    _keepalive: SmallVec<[Tensor; 8]>,
}

#[cfg(feature = "cuda")]
struct InFlightGraph {
    device_location: DeviceLocation,
    event: CudaEventHandle,
    _keepalive: SmallVec<[Tensor; 8]>,
    _graph_entry: Rc<CudaGraphEntry>,
}

#[cfg(feature = "cuda")]
struct CudaGraphEntry {
    _context: ExecutionContext,
    exec: CudaGraphExec,
}

#[cfg(feature = "cuda")]
struct CudaGraphBackoff {
    remaining_runs: usize,
    next_runs: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraphBackoff {
    fn new() -> Self {
        Self {
            remaining_runs: CUDA_GRAPH_INITIAL_BACKOFF_RUNS,
            next_runs: (CUDA_GRAPH_INITIAL_BACKOFF_RUNS * 2).min(CUDA_GRAPH_MAX_BACKOFF_RUNS),
        }
    }

    fn back_off_again(&mut self) {
        self.remaining_runs = self.next_runs;
        self.next_runs = (self.next_runs * 2).min(CUDA_GRAPH_MAX_BACKOFF_RUNS);
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphKey {
    stream: usize,
    inputs: SmallVec<[CudaGraphTensorKey; 4]>,
    outputs: SmallVec<[CudaGraphTensorKey; 4]>,
}

#[cfg(feature = "cuda")]
impl CudaGraphKey {
    fn new<I, O>(inputs: I, outputs: O, stream: CudaStream<'_>) -> Self
    where
        I: IntoIterator<Item = CudaGraphTensorKey>,
        O: IntoIterator<Item = CudaGraphTensorKey>,
    {
        Self {
            stream: stream.as_raw() as usize,
            inputs: inputs.into_iter().collect(),
            outputs: outputs.into_iter().collect(),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphTensorKey {
    tensor_index: usize,
    data_type: DataType,
    shape: Dims,
    ptr: usize,
    bytes: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraphTensorKey {
    fn from_input(tensor_index: usize, input: &DeviceInputTensor<'_>) -> Self {
        Self {
            tensor_index,
            data_type: input.data_type,
            shape: input.shape.clone(),
            ptr: input.ptr as usize,
            bytes: input.bytes,
        }
    }

    fn from_output(tensor_index: usize, output: &DeviceOutputTensor<'_>) -> Self {
        Self {
            tensor_index,
            data_type: output.data_type,
            shape: output.shape.clone(),
            ptr: output.ptr as usize,
            bytes: output.bytes,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_graph_tests {
    use super::*;

    fn input(ptr: usize, shape: Dims) -> DeviceInputTensor<'static> {
        DeviceInputTensor::new("input", DataType::Float, shape, ptr as *const c_void, 16)
    }

    fn output(ptr: usize, shape: Dims) -> DeviceOutputTensor<'static> {
        DeviceOutputTensor::new("output", DataType::Float, shape, ptr as *mut c_void, 16)
    }

    fn stream(ptr: usize) -> CudaStream<'static> {
        unsafe { CudaStream::from_raw(ptr as *mut c_void) }
    }

    #[test]
    fn cuda_graph_key_requires_same_stream_and_binding_addresses() {
        let shape = Dims::new([1, 4]).unwrap();
        let left_input = [input(0x1000, shape.clone())];
        let left_output = [output(0x2000, shape.clone())];
        let left = CudaGraphKey::new(
            [CudaGraphTensorKey::from_input(0, &left_input[0])],
            [CudaGraphTensorKey::from_output(1, &left_output[0])],
            stream(0x3000),
        );

        assert_eq!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(
                    0,
                    &input(0x1000, shape.clone()),
                )],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2000, shape.clone()),
                )],
                stream(0x3000),
            )
        );
        assert_ne!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(
                    0,
                    &input(0x1100, shape.clone()),
                )],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2000, shape.clone()),
                )],
                stream(0x3000),
            )
        );
        assert_ne!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(
                    0,
                    &input(0x1000, shape.clone()),
                )],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2100, shape.clone()),
                )],
                stream(0x3000),
            )
        );
        assert_ne!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(0, &input(0x1000, shape))],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2000, Dims::new([1, 4]).unwrap()),
                )],
                stream(0x3100),
            )
        );
        assert_ne!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(
                    2,
                    &input(0x1000, Dims::new([1, 4]).unwrap()),
                )],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2000, Dims::new([1, 4]).unwrap()),
                )],
                stream(0x3000),
            )
        );
        assert_ne!(
            left,
            CudaGraphKey::new(
                [CudaGraphTensorKey::from_input(
                    0,
                    &DeviceInputTensor::new(
                        "input",
                        DataType::Float,
                        Dims::new([1, 4]).unwrap(),
                        0x1000usize as *const c_void,
                        12,
                    ),
                )],
                [CudaGraphTensorKey::from_output(
                    1,
                    &output(0x2000, Dims::new([1, 4]).unwrap()),
                )],
                stream(0x3000),
            )
        );
    }

    #[test]
    fn cuda_graph_capture_backoff_is_bounded() {
        let mut backoff = CudaGraphBackoff::new();

        assert_eq!(backoff.remaining_runs, CUDA_GRAPH_INITIAL_BACKOFF_RUNS);
        assert_eq!(
            backoff.next_runs,
            (CUDA_GRAPH_INITIAL_BACKOFF_RUNS * 2).min(CUDA_GRAPH_MAX_BACKOFF_RUNS)
        );

        for _ in 0..16 {
            backoff.back_off_again();
        }

        assert_eq!(backoff.remaining_runs, CUDA_GRAPH_MAX_BACKOFF_RUNS);
        assert_eq!(backoff.next_runs, CUDA_GRAPH_MAX_BACKOFF_RUNS);
    }
}

pub(crate) struct PooledExecutionContext<'session> {
    session: &'session Session,
    context: Option<ExecutionContext>,
}

impl Deref for PooledExecutionContext<'_> {
    type Target = ExecutionContext;

    fn deref(&self) -> &Self::Target {
        self.context
            .as_ref()
            .expect("pooled execution context was already returned")
    }
}

impl DerefMut for PooledExecutionContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context
            .as_mut()
            .expect("pooled execution context was already returned")
    }
}

impl Drop for PooledExecutionContext<'_> {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            self.session.context_pool.borrow_mut().push(context);
        }
    }
}

impl PooledExecutionContext<'_> {
    #[cfg(feature = "cuda")]
    fn into_inner(mut self) -> ExecutionContext {
        self.context
            .take()
            .expect("pooled execution context was already returned")
    }
}

enum BoundInputBinding<'a> {
    StagedDevice(DeviceBuffer),
    DevicePtr(*mut c_void),
    Host(&'a [u8]),
}

impl BoundInputBinding<'_> {
    fn as_mut_ptr(&self) -> *mut c_void {
        match self {
            Self::StagedDevice(buffer) => buffer.as_mut_ptr(),
            Self::DevicePtr(ptr) => *ptr,
            Self::Host(bytes) => bytes.as_ptr().cast_mut().cast(),
        }
    }
}

fn set_input_shapes(
    session: &Session,
    context: &mut ExecutionContext,
    inputs: &[TensorShape<'_>],
) -> Result<()> {
    for input in inputs {
        session.validate_tensor_shape(input.name, &input.shape)?;
        session.ensure_tensor_mode(input.name, TensorIOMode::Input)?;
        context.set_input_shape(input.name, &input.shape)?;
    }
    context.remember_input_shapes(inputs);
    Ok(())
}

fn set_input_shapes_if_needed(
    session: &Session,
    context: &mut ExecutionContext,
    inputs: &[TensorShape<'_>],
) -> Result<()> {
    if context.input_shapes_match(inputs) {
        return Ok(());
    }
    set_input_shapes(session, context, inputs)
}

#[cfg(feature = "cuda")]
fn set_device_input_shapes_if_needed(
    context: &mut ExecutionContext,
    inputs: &[DeviceInputTensor<'_>],
) -> Result<()> {
    if context.device_input_shapes_match(inputs) {
        return Ok(());
    }
    for input in inputs {
        context.set_input_shape(input.name, &input.shape)?;
    }
    context.remember_device_input_shapes(inputs);
    Ok(())
}
