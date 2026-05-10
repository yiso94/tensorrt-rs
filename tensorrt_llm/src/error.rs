use std::ffi::NulError;
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    PluginInit { status: i32 },
    InvalidArgument(String),
    Executor(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    TensorRt(tensorrt::Error),
    Candle(candle_core::Error),
    Nul(NulError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PluginInit { status } => {
                write!(
                    f,
                    "TensorRT-LLM plugin initialization failed with status {status}"
                )
            }
            Self::InvalidArgument(reason) => write!(f, "{reason}"),
            Self::Executor(reason) => write!(f, "TensorRT-LLM executor operation failed: {reason}"),
            Self::Io(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::TensorRt(error) => error.fmt(f),
            Self::Candle(error) => error.fmt(f),
            Self::Nul(error) => write!(f, "string contains interior NUL: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PluginInit { .. } | Self::InvalidArgument(_) | Self::Executor(_) => None,
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::TensorRt(error) => Some(error),
            Self::Candle(error) => Some(error),
            Self::Nul(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<tensorrt::Error> for Error {
    fn from(error: tensorrt::Error) -> Self {
        Self::TensorRt(error)
    }
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Candle(error)
    }
}

impl From<NulError> for Error {
    fn from(error: NulError) -> Self {
        Self::Nul(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;
    use std::ffi::CString;

    #[test]
    fn display_and_source_cover_error_variants() {
        let json_error = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let nul_error = CString::new(b"bad\0name".to_vec()).unwrap_err();
        let cases = [
            Error::PluginInit { status: -1 },
            Error::InvalidArgument("bad argument".to_owned()),
            Error::Executor("failed".to_owned()),
            Error::Io(std::io::Error::other("io")),
            Error::Json(json_error),
            Error::TensorRt(tensorrt::Error::InvalidArgument),
            Error::Candle(candle_core::Error::UnsupportedDTypeForOp(
                candle_core::DType::U8,
                "unit-test",
            )),
            Error::Nul(nul_error),
        ];

        for error in cases {
            assert!(!error.to_string().is_empty());
            match error {
                Error::PluginInit { .. } | Error::InvalidArgument(_) | Error::Executor(_) => {
                    assert!(error.source().is_none());
                }
                _ => assert!(error.source().is_some()),
            }
        }
    }

    #[test]
    fn from_conversions_preserve_error_sources() {
        let io: Error = std::io::Error::other("io").into();
        assert!(matches!(io, Error::Io(_)));
        assert!(io.source().is_some());

        let json: Error = serde_json::from_str::<serde_json::Value>("not json")
            .unwrap_err()
            .into();
        assert!(matches!(json, Error::Json(_)));
        assert!(json.source().is_some());

        let trt: Error = tensorrt::Error::InvalidArgument.into();
        assert!(matches!(trt, Error::TensorRt(_)));
        assert!(trt.source().is_some());

        let candle: Error =
            candle_core::Error::UnsupportedDTypeForOp(candle_core::DType::U8, "unit-test").into();
        assert!(matches!(candle, Error::Candle(_)));
        assert!(candle.source().is_some());

        let nul: Error = CString::new(b"bad\0name".to_vec()).unwrap_err().into();
        assert!(matches!(nul, Error::Nul(_)));
        assert!(nul.source().is_some());
    }
}
