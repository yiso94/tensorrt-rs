# tensorrt-rs

Rust bindings for NVIDIA TensorRT and TensorRT-LLM.

The workspace contains:

- `tensorrt`: safe TensorRT runtime/session API with Candle tensor I/O.
- `tensorrt-llm`: TensorRT-LLM Executor API with Candle tensor support.
- `tensorrt-sys`, `tensorrt-llm-sys`: low-level C ABI shims.

## Install

```bash
cargo add tensorrt --features cuda
cargo add tensorrt-llm --features cuda
```

For a local checkout:

```bash
cargo add tensorrt --path /path/to/tensorrt-rs/tensorrt --features cuda
cargo add tensorrt-llm --path /path/to/tensorrt-rs/tensorrt_llm --features cuda
```

## Build

TensorRT and TensorRT-LLM must be installed and discoverable at build time.
`cargo add` only edits `Cargo.toml`; pass install paths to `cargo build`,
`cargo test`, or your application build.

```bash
TENSORRT_ROOT=/usr/local/tensorrt \
TENSORRT_LLM_ROOT=/usr/local/tensorrt_llm \
cargo build --workspace --features cuda
```

Useful environment variables:

- `TENSORRT_ROOT`, `TENSORRT_INCLUDE_DIR`, `TENSORRT_LIB_DIR`
- `TENSORRT_LLM_ROOT`, `TENSORRT_LLM_INCLUDE_DIR`, `TENSORRT_LLM_LIB_DIR`
- `CUDA_HOME`, `CUDA_PATH`

## TensorRT

```rust
use candle_core::{DType, Device, Tensor};
use tensorrt::{CudaStream, InputTensors, OutputTensors, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda_with_stream(0)?;
    let stream = CudaStream::from_device(&device)?;
    let session = Session::new(std::fs::read("model.plan")?)?;

    let input = Tensor::from_vec(vec![0f32; 1 * 3 * 224 * 224], (1, 3, 224, 224), &device)?;
    let output = Tensor::zeros((1, 1000), DType::F32, &device)?;
    let inputs = InputTensors::new([("input", &input)]);
    let mut outputs = OutputTensors::new([("logits", output)]);

    session.run(&inputs, &mut outputs, stream)?;

    stream.synchronize()?;
    let _logits = outputs["logits"].to_vec2::<f32>()?;
    Ok(())
}
```

Inputs and outputs must be contiguous Candle CUDA tensors on the same device as
the stream. `run` enqueues work and returns without synchronizing; synchronize
the stream before reading GPU outputs on the CPU.

## TensorRT-LLM

```rust
use candle_core::Device;
use tensorrt::CudaStream;
use tensorrt_llm::{Executor, Request, SamplingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda_with_stream(0)?;
    let stream = CudaStream::from_device(&device)?;
    let executor = Executor::builder("engine").build()?;

    let input_ids = [1, 42, 2];
    let request = Request::new(&input_ids)
        .max_new_tokens(1)
        .sampling(SamplingConfig::default().top_k(1));

    let response = executor.run(request)?;
    let _token_ids = &response.outputs()["token_ids"];
    Ok(())
}
```

Use `Request::additional_outputs(...)` when you need marked model outputs such
as hidden states in preallocated CUDA tensors.

## Status

This is an early `0.1` binding. The high-level APIs are intentionally small:
`Session::run` for TensorRT and `Executor::run` for TensorRT-LLM.
