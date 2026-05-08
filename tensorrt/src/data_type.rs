use super::error::{Error, Result};
use std::fmt;
use std::str::FromStr;

/// TensorRT tensor data type.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DataType {
    Float,
    Half,
    Int8,
    Int32,
    Bool,
    Uint8,
    Fp8,
    Bf16,
    Int64,
    Int4,
    Fp4,
    E8m0,
    Unknown(i32),
}

impl DataType {
    pub(crate) fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::Float,
            1 => Self::Half,
            2 => Self::Int8,
            3 => Self::Int32,
            4 => Self::Bool,
            5 => Self::Uint8,
            6 => Self::Fp8,
            7 => Self::Bf16,
            8 => Self::Int64,
            9 => Self::Int4,
            10 => Self::Fp4,
            11 => Self::E8m0,
            other => Self::Unknown(other),
        }
    }

    /// Converts a TensorRT binding dtype string to a TensorRT data type.
    pub fn from_binding_dtype(dtype: &str) -> Option<Self> {
        match dtype {
            "bfloat16" => Some(Self::Bf16),
            "float16" => Some(Self::Half),
            "float32" => Some(Self::Float),
            "int64" => Some(Self::Int64),
            "int32" => Some(Self::Int32),
            "fp8" => Some(Self::Fp8),
            _ => None,
        }
    }

    /// Returns the TensorRT binding dtype string for this data type.
    pub fn as_binding_dtype(self) -> Option<&'static str> {
        match self {
            Self::Bf16 => Some("bfloat16"),
            Self::Half => Some("float16"),
            Self::Float => Some("float32"),
            Self::Int64 => Some("int64"),
            Self::Int32 => Some("int32"),
            Self::Fp8 => Some("fp8"),
            _ => None,
        }
    }

    pub fn bytes_per_element(self) -> Option<usize> {
        match self {
            Self::Float | Self::Int32 => Some(4),
            Self::Half | Self::Bf16 => Some(2),
            Self::Int8 | Self::Bool | Self::Uint8 | Self::Fp8 => Some(1),
            Self::Int64 => Some(8),
            Self::Int4 | Self::Fp4 | Self::E8m0 | Self::Unknown(_) => None,
        }
    }
}

impl FromStr for DataType {
    type Err = Error;

    fn from_str(dtype: &str) -> Result<Self> {
        Self::from_binding_dtype(dtype).ok_or(Error::UnsupportedDataType {
            data_type: Self::Unknown(-1),
            context: "TensorRT binding dtype parsing",
        })
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_binding_dtype() {
            Some(dtype) => write!(f, "{dtype}"),
            None => write!(f, "{self:?}"),
        }
    }
}

/// Converts an `f32` value to a BF16 bit pattern with round-to-nearest-even.
pub fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Converts a BF16 bit pattern to `f32`.
pub fn bf16_bits_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}
