use candle_core::{DType, Device, Tensor};
use tensorrt::{InputTensors, OutputTensors};

#[test]
fn candle_tensor_containers_are_public_api() {
    let input = Tensor::from_vec(vec![1f32, 2.0], 2, &Device::Cpu).unwrap();
    let inputs = InputTensors::new([("input", &input)]);
    assert_eq!(inputs.get("input").unwrap().dims(), &[2]);

    let output = Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap();
    let outputs = OutputTensors::new([("output", output)]);
    assert_eq!(outputs["output"].dtype(), DType::F32);
}
