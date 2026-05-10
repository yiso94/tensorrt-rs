use crate::{DataType, DeviceInputTensor, Dims};
use smallvec::SmallVec;

const INLINE_PLAN_TENSORS: usize = 4;

#[derive(Debug)]
pub(crate) struct DeviceOutputPlan {
    pub(crate) name: String,
    pub(crate) data_type: DataType,
    pub(crate) shape: Dims,
    pub(crate) bytes: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RunPlanKey {
    pub(crate) inputs: SmallVec<[RunPlanInputKey; INLINE_PLAN_TENSORS]>,
    pub(crate) outputs: SmallVec<[usize; INLINE_PLAN_TENSORS]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RunPlanInputKey {
    tensor_index: usize,
    data_type: DataType,
    shape: Dims,
    bytes: usize,
}

impl RunPlanKey {
    pub(crate) fn new<I, O>(inputs: I, output_indices: O) -> Self
    where
        I: IntoIterator<Item = RunPlanInputKey>,
        O: IntoIterator<Item = usize>,
    {
        Self {
            inputs: inputs.into_iter().collect(),
            outputs: output_indices.into_iter().collect(),
        }
    }
}

impl RunPlanInputKey {
    pub(crate) fn new(tensor_index: usize, input: &DeviceInputTensor<'_>) -> Self {
        Self {
            tensor_index,
            data_type: input.data_type,
            shape: input.shape.clone(),
            bytes: input.bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    #[test]
    fn run_plan_key_preserves_input_and_output_order() {
        let inputs = [
            DeviceInputTensor::new(
                "input_ids",
                DataType::Int32,
                Dims::new([1, 3]).unwrap(),
                0x10usize as *const c_void,
                12,
            ),
            DeviceInputTensor::new(
                "position_ids",
                DataType::Int32,
                Dims::new([1, 3]).unwrap(),
                0x20usize as *const c_void,
                12,
            ),
        ];

        let key = RunPlanKey::new(
            [
                RunPlanInputKey::new(0, &inputs[0]),
                RunPlanInputKey::new(1, &inputs[1]),
            ],
            [2, 3],
        );

        assert_eq!(
            key.inputs
                .iter()
                .map(|input| input.tensor_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(key.outputs.as_slice(), [2, 3]);
    }

    #[test]
    fn run_plan_key_ignores_pointer_identity() {
        let left = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([1, 4]).unwrap(),
            0x10usize as *const c_void,
            16,
        )];
        let right = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([1, 4]).unwrap(),
            0x20usize as *const c_void,
            16,
        )];

        assert_eq!(
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [1]),
            RunPlanKey::new([RunPlanInputKey::new(0, &right[0])], [1])
        );
    }

    #[test]
    fn run_plan_key_distinguishes_shapes_and_outputs() {
        let left = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([1, 4]).unwrap(),
            std::ptr::null(),
            16,
        )];
        let right = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([2, 4]).unwrap(),
            std::ptr::null(),
            32,
        )];

        assert_ne!(
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [1]),
            RunPlanKey::new([RunPlanInputKey::new(0, &right[0])], [1])
        );
        assert_ne!(
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [1]),
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [2])
        );
    }

    #[test]
    fn run_plan_key_distinguishes_tensor_indices_and_byte_lengths() {
        let left = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([1, 4]).unwrap(),
            std::ptr::null(),
            16,
        )];
        let wrong_bytes = [DeviceInputTensor::new(
            "input",
            DataType::Float,
            Dims::new([1, 4]).unwrap(),
            std::ptr::null(),
            12,
        )];

        assert_ne!(
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [1]),
            RunPlanKey::new([RunPlanInputKey::new(2, &left[0])], [1])
        );
        assert_ne!(
            RunPlanKey::new([RunPlanInputKey::new(0, &left[0])], [1]),
            RunPlanKey::new([RunPlanInputKey::new(0, &wrong_bytes[0])], [1])
        );
    }
}
