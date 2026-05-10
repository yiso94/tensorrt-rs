use crate::{Error, Result, TensorInfo};

pub(crate) fn select_outputs(
    output_infos: Vec<TensorInfo>,
    output_names: &[&str],
) -> Result<Vec<TensorInfo>> {
    if output_names.is_empty() {
        return Ok(output_infos);
    }

    output_names
        .iter()
        .map(|name| {
            output_infos
                .iter()
                .find(|tensor| tensor.name == *name)
                .cloned()
                .ok_or_else(|| Error::InvalidShape {
                    tensor: (*name).to_owned(),
                    reason: "output tensor was not reported by TensorRT shape inference".to_owned(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataType, Dims, TensorIOMode};

    #[test]
    fn select_outputs_preserves_requested_order() {
        let outputs = test_outputs();
        let selected = select_outputs(outputs, &["hidden", "logits"]).unwrap();

        assert_eq!(
            selected
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            vec!["hidden", "logits"]
        );
    }

    #[test]
    fn select_outputs_returns_all_outputs_when_names_are_empty() {
        let outputs = test_outputs();
        let selected = select_outputs(outputs.clone(), &[]).unwrap();

        assert_eq!(selected, outputs);
    }

    #[test]
    fn select_outputs_rejects_missing_output() {
        let err = select_outputs(test_outputs(), &["missing"]).unwrap_err();

        match err {
            Error::InvalidShape { tensor, .. } => assert_eq!(tensor, "missing"),
            err => panic!("expected InvalidShape, got {err:?}"),
        }
    }

    fn test_outputs() -> Vec<TensorInfo> {
        vec![
            TensorInfo::with_io_mode(
                "logits",
                DataType::Float,
                Dims::new([1, 2]).unwrap(),
                TensorIOMode::Output,
            ),
            TensorInfo::with_io_mode(
                "hidden",
                DataType::Half,
                Dims::new([1, 2, 4]).unwrap(),
                TensorIOMode::Output,
            ),
        ]
    }
}
