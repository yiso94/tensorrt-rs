use crate::{Error, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use tensorrt::DataType;

#[derive(Clone, Debug)]
pub(crate) struct Config {
    data_type: DataType,
    vocab_size: usize,
    hidden_size: usize,
    max_batch_size: usize,
    max_num_tokens: usize,
    max_prompt_embedding_table_size: usize,
}

impl Config {
    pub(crate) fn from_engine_dir(engine_dir: impl AsRef<Path>) -> Result<Self> {
        let config_path = engine_dir.as_ref().join("config.json");
        let raw = fs::read_to_string(&config_path)?;
        Self::from_json(&raw)
    }

    pub(crate) fn from_json(raw: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(raw)?;
        let pretrained = raw.pretrained_config.ok_or_else(|| {
            Error::InvalidArgument("TensorRT-LLM config is missing pretrained_config".to_owned())
        })?;
        let dtype = pretrained.dtype.ok_or_else(|| {
            Error::InvalidArgument("TensorRT-LLM config is missing pretrained_config.dtype".into())
        })?;
        let data_type = DataType::from_binding_dtype(&dtype).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "TensorRT-LLM model dtype {dtype:?} is not supported"
            ))
        })?;
        let vocab_size = required_usize(pretrained.vocab_size, "pretrained_config.vocab_size")?;
        let hidden_size = required_usize(pretrained.hidden_size, "pretrained_config.hidden_size")?;
        let max_batch_size = positive_usize(
            raw.build_config.max_batch_size,
            "build_config.max_batch_size",
        )?;
        let max_num_tokens = positive_usize(
            raw.build_config.max_num_tokens,
            "build_config.max_num_tokens",
        )?;

        Ok(Self {
            data_type,
            vocab_size,
            hidden_size,
            max_batch_size,
            max_num_tokens,
            max_prompt_embedding_table_size: raw.build_config.max_prompt_embedding_table_size,
        })
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub(crate) fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub(crate) fn max_num_tokens(&self) -> usize {
        self.max_num_tokens
    }

    pub(crate) fn max_prompt_embedding_table_size(&self) -> usize {
        self.max_prompt_embedding_table_size
    }
}

fn required_usize(value: Option<usize>, field: &str) -> Result<usize> {
    value.ok_or_else(|| Error::InvalidArgument(format!("TensorRT-LLM config is missing {field}")))
}

fn positive_usize(value: usize, field: &str) -> Result<usize> {
    if value == 0 {
        return Err(Error::InvalidArgument(format!(
            "TensorRT-LLM config {field} must be greater than zero"
        )));
    }
    Ok(value)
}

#[derive(Deserialize)]
struct RawConfig {
    pretrained_config: Option<RawPretrainedConfig>,
    build_config: RawBuildConfig,
}

#[derive(Deserialize)]
struct RawPretrainedConfig {
    dtype: Option<String>,
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
}

#[derive(Deserialize)]
struct RawBuildConfig {
    max_batch_size: usize,
    max_num_tokens: usize,
    #[serde(default)]
    max_prompt_embedding_table_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let raw = r#"{
            "pretrained_config": {
                "dtype": "bfloat16",
                "vocab_size": 32000,
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048,
                "max_prompt_embedding_table_size": 512
            }
        }"#;
        let config = Config::from_json(raw).unwrap();
        assert_eq!(config.data_type(), DataType::Bf16);
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.hidden_size(), 4096);
        assert_eq!(config.max_batch_size(), 1);
        assert_eq!(config.max_num_tokens(), 2048);
        assert_eq!(config.max_prompt_embedding_table_size(), 512);
    }

    #[test]
    fn parses_config_from_engine_directory() {
        let dir =
            std::env::temp_dir().join(format!("tensorrt-llm-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "pretrained_config": {
                    "dtype": "float16",
                    "vocab_size": 10,
                    "hidden_size": 4
                },
                "build_config": {
                    "max_batch_size": 2,
                    "max_num_tokens": 16
                }
            }"#,
        )
        .unwrap();

        let config = Config::from_engine_dir(&dir).unwrap();
        assert_eq!(config.data_type(), DataType::Half);
        assert_eq!(config.max_prompt_embedding_table_size(), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_missing_pretrained_config() {
        let raw = r#"{
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048
            }
        }"#;

        let error = Config::from_json(raw).unwrap_err();
        assert!(error.to_string().contains("missing pretrained_config"));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let raw = r#"{
            "pretrained_config": {
                "dtype": "float64",
                "vocab_size": 32000,
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048
            }
        }"#;

        let error = Config::from_json(raw).unwrap_err();
        assert!(error.to_string().contains("dtype"));
    }

    #[test]
    fn rejects_missing_required_pretrained_fields() {
        let missing_dtype = r#"{
            "pretrained_config": {
                "vocab_size": 32000,
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048
            }
        }"#;
        assert!(
            Config::from_json(missing_dtype)
                .unwrap_err()
                .to_string()
                .contains("dtype")
        );

        let missing_vocab = r#"{
            "pretrained_config": {
                "dtype": "float16",
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048
            }
        }"#;
        assert!(
            Config::from_json(missing_vocab)
                .unwrap_err()
                .to_string()
                .contains("vocab_size")
        );

        let missing_hidden = r#"{
            "pretrained_config": {
                "dtype": "float16",
                "vocab_size": 32000
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 2048
            }
        }"#;
        assert!(
            Config::from_json(missing_hidden)
                .unwrap_err()
                .to_string()
                .contains("hidden_size")
        );
    }

    #[test]
    fn rejects_zero_build_limits() {
        let raw = r#"{
            "pretrained_config": {
                "dtype": "float16",
                "vocab_size": 32000,
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 0,
                "max_num_tokens": 2048
            }
        }"#;

        let error = Config::from_json(raw).unwrap_err();
        assert!(error.to_string().contains("max_batch_size"));

        let raw = r#"{
            "pretrained_config": {
                "dtype": "float16",
                "vocab_size": 32000,
                "hidden_size": 4096
            },
            "build_config": {
                "max_batch_size": 1,
                "max_num_tokens": 0
            }
        }"#;

        let error = Config::from_json(raw).unwrap_err();
        assert!(error.to_string().contains("max_num_tokens"));
    }
}
