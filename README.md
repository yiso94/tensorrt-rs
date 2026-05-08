# tensorrt-rs

Rust bindings for NVIDIA TensorRT.

This workspace provides `tensorrt`, a safe runtime/session API with Candle
tensor input/output, and `tensorrt-sys`, the low-level C ABI shim. It focuses on
general TensorRT runtime execution; TensorRT-LLM Executor and generation APIs
are intentionally out of scope.

## Install

```bash
cargo add tensorrt --features cuda
```

For a local checkout:

```bash
cargo add tensorrt --path /path/to/tensorrt-rs/tensorrt --features cuda
```

Use `tensorrt-sys` directly only when you need the low-level FFI crate.

## Build

TensorRT must be discoverable through standard system paths or:

```bash
TENSORRT_ROOT=/usr/local/tensorrt cargo build -p tensorrt --features cuda
```

You can also set `TENSORRT_INCLUDE_DIR`, `TENSORRT_LIB_DIR`, `CUDA_HOME`, or
`CUDA_PATH` for non-standard installations.

## Candle Runtime

Candle tensors are the high-level session API. The shared adapter lives in
`tensorrt::candle`, so dtype, shape, input staging, and output conversion stay
in one place.

```rust
use candle_core::{DType, Device, Tensor};
use tensorrt::{CudaStream, InputTensors, OutputTensors, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda_with_stream(0)?;
    let stream = CudaStream::from_device(&device)?;
    let engine_buffer = std::fs::read("model.plan")?;
    let session = Session::from_serialized_engine(&engine_buffer)?;
    let input = Tensor::from_vec(vec![0f32; 1 * 3 * 224 * 224], (1, 3, 224, 224), &device)?;
    let output = Tensor::zeros((1, 1000), DType::F32, &device)?;
    let inputs = InputTensors::new([("input", &input)]);
    let mut outputs = OutputTensors::new([("logits", output)]);

    session.run(&inputs, &mut outputs, stream)?;

    device.synchronize()?;
    let _values = outputs["logits"].to_vec2::<f32>()?;
    Ok(())
}
```

The high-level Candle bridge is CUDA-first and zero-copy for device bindings:
inputs must be contiguous Candle CUDA tensors on the same device as the stream,
and outputs must be preallocated contiguous Candle CUDA tensors on that same
device before calling `run`.

`CudaStream::from_device(&device)?` borrows Candle's CUDA stream for that
device. Use `Device::new_cuda_with_stream` for performance-sensitive runs: it
gives Candle a non-default CUDA stream, so Candle kernels and TensorRT enqueue
work are ordered on the same explicit stream. `Device::new_cuda` uses CUDA's
default stream, which is correct but can make TensorRT add extra synchronization
and reduce overlap. Passing a stream for a different device, or a raw/default
stream whose device is unknown, returns an error at the high-level `run`
boundary.

The high-level `run` API writes directly into the supplied output tensors by
enqueueing work on the supplied stream. It does not synchronize before
returning, so GPU work can continue without a host-side stall. `Session` keeps
the bound Candle tensor storage alive internally until the enqueued work has
completed. Synchronize the device or stream before reading outputs from the CPU
or using them from another stream.

`Session` keeps reusable TensorRT execution contexts internally, so repeated
runs avoid recreating an execution context on the hot path. For repeated runs
with the same input/output names, dtypes, and shapes, `Session` also reuses the
resolved output plan, including output byte sizes, and skips redundant TensorRT
shape setup on a pooled context when possible. Runtime binding validation uses
the engine metadata cached at session creation instead of repeatedly querying
TensorRT. Output tensors must not share CUDA storage with inputs or with each
other.

CPU/host-buffer execution remains available through the lower-level
`Session::run_host` and `Session::run_allocating` APIs.
