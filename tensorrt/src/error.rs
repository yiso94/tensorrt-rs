use super::cuda::CudaError;
use super::data_type::DataType;
use super::ffi;
use super::tensor::TensorIOMode;
use std::ffi::NulError;
use std::fmt;

/// Result type returned by the TensorRT runtime bindings.
pub type Result<T> = std::result::Result<T, Error>;

/// TensorRT runtime binding errors.
#[derive(Debug)]
pub enum Error {
    InvalidArgument,
    AllocationFailed,
    NullPointer,
    TensorRtError,
    Exception,
    InvalidTensorName(NulError),
    InvalidDimensions {
        len: usize,
        max: usize,
    },
    InvalidDimensionCount(i32),
    InvalidShape {
        tensor: String,
        reason: String,
    },
    SizeMismatch {
        tensor: String,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    TypeMismatch {
        tensor: String,
        expected: DataType,
        actual: DataType,
    },
    TensorModeMismatch {
        tensor: String,
        expected: TensorIOMode,
        actual: TensorIOMode,
    },
    UnsupportedDataType {
        data_type: DataType,
        context: &'static str,
    },
    Cuda(CudaError),
    Candle(candle_core::Error),
    UnsupportedCandleDType {
        dtype: candle_core::DType,
        context: &'static str,
    },
}

impl Error {
    pub(crate) fn from_status(status: ffi::Status) -> Result<()> {
        match status {
            ffi::Status::Ok => Ok(()),
            ffi::Status::InvalidArgument => Err(Self::InvalidArgument),
            ffi::Status::AllocationFailed => Err(Self::AllocationFailed),
            ffi::Status::NullPointer => Err(Self::NullPointer),
            ffi::Status::TensorRtError => Err(Self::TensorRtError),
            ffi::Status::Exception => Err(Self::Exception),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument => write!(f, "invalid argument passed to TensorRT"),
            Self::AllocationFailed => write!(f, "allocation failed in TensorRT binding"),
            Self::NullPointer => write!(f, "TensorRT returned a null pointer"),
            Self::TensorRtError => write!(f, "TensorRT operation failed"),
            Self::Exception => write!(f, "C++ exception escaped from TensorRT binding"),
            Self::InvalidTensorName(err) => write!(f, "tensor name contains interior NUL: {err}"),
            Self::InvalidDimensions { len, max } => {
                write!(
                    f,
                    "invalid dimension count {len}; TensorRT binding supports at most {max}"
                )
            }
            Self::InvalidDimensionCount(count) => {
                write!(f, "TensorRT returned invalid dimension count {count}")
            }
            Self::InvalidShape { tensor, reason } => {
                write!(f, "invalid tensor shape for {tensor}: {reason}")
            }
            Self::SizeMismatch {
                tensor,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "size mismatch for {tensor}: expected {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::TypeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "data type mismatch for {tensor}: expected {expected:?}, got {actual:?}"
            ),
            Self::TensorModeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "I/O mode mismatch for {tensor}: expected {expected:?}, got {actual:?}"
            ),
            Self::UnsupportedDataType { data_type, context } => {
                write!(
                    f,
                    "unsupported TensorRT data type for {context}: {data_type:?}"
                )
            }
            Self::Cuda(error) => write!(f, "{error}"),
            Self::Candle(error) => write!(f, "Candle tensor error: {error}"),
            Self::UnsupportedCandleDType { dtype, context } => {
                write!(f, "unsupported Candle dtype for {context}: {dtype:?}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidTensorName(error) => Some(error),
            Self::Cuda(error) => Some(error),
            Self::Candle(error) => Some(error),
            _ => None,
        }
    }
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Candle(error)
    }
}
