use crate::{DataType, DeviceInputTensor, Dims};

#[derive(Debug)]
pub(crate) struct DeviceOutputPlan {
    pub(crate) name: String,
    pub(crate) data_type: DataType,
    pub(crate) shape: Dims,
    pub(crate) bytes: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RunPlanKey {
    pub(crate) inputs: Vec<RunPlanInputKey>,
    pub(crate) outputs: Vec<String>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RunPlanInputKey {
    name: String,
    data_type: DataType,
    shape: Dims,
}

impl RunPlanKey {
    pub(crate) fn new<'name>(
        inputs: &[DeviceInputTensor<'_>],
        output_names: impl IntoIterator<Item = &'name str>,
    ) -> Self {
        Self {
            inputs: inputs
                .iter()
                .map(|input| RunPlanInputKey {
                    name: input.name.to_owned(),
                    data_type: input.data_type,
                    shape: input.shape.clone(),
                })
                .collect(),
            outputs: output_names.into_iter().map(str::to_owned).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    #[test]
    fn run_plan_key_preserves_input_and_output_order() {
        let inputs = vec![
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

        let key = RunPlanKey::new(&inputs, ["logits", "hidden"]);

        assert_eq!(
            key.inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["input_ids", "position_ids"]
        );
        assert_eq!(key.outputs, vec!["logits", "hidden"]);
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
            RunPlanKey::new(&left, ["output"]),
            RunPlanKey::new(&right, ["output"])
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
            RunPlanKey::new(&left, ["output"]),
            RunPlanKey::new(&right, ["output"])
        );
        assert_ne!(
            RunPlanKey::new(&left, ["output"]),
            RunPlanKey::new(&left, ["other"])
        );
    }
}
