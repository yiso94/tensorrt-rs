use super::data_type::DataType;
use super::error::{Error, Result};
use super::{MAX_DIMS, ffi};
use smallvec::SmallVec;
use std::{ffi::c_void, mem, ptr, slice};

/// Shape dimensions for a TensorRT tensor.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Dims {
    values: SmallVec<[i64; MAX_DIMS]>,
}

impl Dims {
    pub fn new(values: impl IntoIterator<Item = i64>) -> Result<Self> {
        let values: SmallVec<[i64; MAX_DIMS]> = values.into_iter().collect();
        if values.len() > MAX_DIMS {
            return Err(Error::InvalidDimensions {
                len: values.len(),
                max: MAX_DIMS,
            });
        }
        Ok(Self { values })
    }

    pub fn as_slice(&self) -> &[i64] {
        &self.values
    }

    pub fn rank(&self) -> usize {
        self.values.len()
    }

    pub fn is_dynamic(&self) -> bool {
        self.values.iter().any(|&dim| dim < 0)
    }

    pub(crate) fn from_ffi(dims: ffi::Dims) -> Result<Self> {
        let len = usize::try_from(dims.nb_dims)
            .map_err(|_| Error::InvalidDimensionCount(dims.nb_dims))?;
        if len > MAX_DIMS {
            return Err(Error::InvalidDimensionCount(dims.nb_dims));
        }
        Self::new(dims.d[..len].iter().copied())
    }

    pub(crate) fn to_ffi(&self) -> ffi::Dims {
        let mut dims = ffi::Dims {
            nb_dims: self.values.len() as i32,
            d: [0; MAX_DIMS],
        };
        dims.d[..self.values.len()].copy_from_slice(&self.values);
        dims
    }
}

impl TryFrom<&[i64]> for Dims {
    type Error = Error;

    fn try_from(value: &[i64]) -> Result<Self> {
        Self::new(value.iter().copied())
    }
}

impl<const N: usize> TryFrom<[i64; N]> for Dims {
    type Error = Error;

    fn try_from(value: [i64; N]) -> Result<Self> {
        Self::new(value)
    }
}

impl From<Dims> for Vec<i64> {
    fn from(value: Dims) -> Self {
        value.values.into_vec()
    }
}

/// Whether a TensorRT tensor is an input, output, or not an I/O tensor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TensorIOMode {
    None,
    Input,
    Output,
    Unknown(i32),
}

impl TensorIOMode {
    pub(crate) fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Input,
            2 => Self::Output,
            other => Self::Unknown(other),
        }
    }
}

/// Summary of a TensorRT engine I/O tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Dims,
    pub data_type: DataType,
    pub io_mode: TensorIOMode,
}

impl TensorInfo {
    pub fn new(name: impl Into<String>, shape: Dims, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            shape,
            data_type,
            io_mode: TensorIOMode::Unknown(-1),
        }
    }

    pub fn with_io_mode(
        name: impl Into<String>,
        data_type: DataType,
        shape: Dims,
        io_mode: TensorIOMode,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            data_type,
            io_mode,
        }
    }

    pub fn shape_info(&self) -> TensorShape<'_> {
        TensorShape::new(&self.name, self.shape.clone())
    }
}

/// Memory location expected for a TensorRT host input binding supplied to
/// [`Session::run_host`](crate::Session::run_host).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorLocation {
    Device,
    Host,
}

/// Named tensor shape used for TensorRT shape inference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorShape<'a> {
    pub name: &'a str,
    pub shape: Dims,
}

impl<'a> TensorShape<'a> {
    pub fn new(name: &'a str, shape: Dims) -> Self {
        Self { name, shape }
    }
}

/// Named host input buffer for [`Session::run_host`](crate::Session::run_host).
#[derive(Clone, Debug)]
pub struct HostInputTensor<'a> {
    pub name: &'a str,
    pub expected_data_type: Option<DataType>,
    pub shape: Dims,
    pub bytes: &'a [u8],
    pub location: TensorLocation,
}

impl<'a> HostInputTensor<'a> {
    pub fn new(name: &'a str, shape: Dims, bytes: &'a [u8]) -> Self {
        Self {
            name,
            expected_data_type: None,
            shape,
            bytes,
            location: TensorLocation::Device,
        }
    }

    pub fn with_data_type(
        name: &'a str,
        data_type: DataType,
        shape: Dims,
        bytes: &'a [u8],
    ) -> Self {
        Self {
            name,
            expected_data_type: Some(data_type),
            shape,
            bytes,
            location: TensorLocation::Device,
        }
    }

    pub fn with_location(mut self, location: TensorLocation) -> Self {
        self.location = location;
        self
    }

    pub fn from_f32(name: &'a str, shape: Dims, values: &'a [f32]) -> Self {
        Self::with_data_type(name, DataType::Float, shape, as_bytes(values))
    }

    pub fn from_bf16_bits(name: &'a str, shape: Dims, values: &'a [u16]) -> Self {
        Self::with_data_type(name, DataType::Bf16, shape, as_bytes(values))
    }

    pub fn shape_info(&self) -> TensorShape<'a> {
        TensorShape::new(self.name, self.shape.clone())
    }
}

/// Named device input buffer for TensorRT device-pointer inference.
#[derive(Clone, Debug)]
pub struct DeviceInputTensor<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub shape: Dims,
    pub ptr: *const c_void,
    pub bytes: usize,
}

impl<'a> DeviceInputTensor<'a> {
    pub fn new(
        name: &'a str,
        data_type: DataType,
        shape: Dims,
        ptr: *const c_void,
        bytes: usize,
    ) -> Self {
        Self {
            name,
            data_type,
            shape,
            ptr,
            bytes,
        }
    }

    pub fn shape_info(&self) -> TensorShape<'a> {
        TensorShape::new(self.name, self.shape.clone())
    }
}

/// Named TensorRT input binding that can be backed by host memory, staged host
/// memory, or a caller-owned CUDA device pointer.
#[derive(Clone, Debug)]
pub enum InputBinding<'a> {
    /// Host pointer bound directly to TensorRT.
    Host(HostInputTensor<'a>),
    /// Host bytes copied to a temporary CUDA buffer before TensorRT enqueue.
    HostToDevice(HostInputTensor<'a>),
    /// Caller-owned CUDA device pointer bound directly to TensorRT.
    Device(DeviceInputTensor<'a>),
}

impl<'a> InputBinding<'a> {
    pub fn host(name: &'a str, data_type: DataType, shape: Dims, bytes: &'a [u8]) -> Self {
        Self::Host(
            HostInputTensor::with_data_type(name, data_type, shape, bytes)
                .with_location(TensorLocation::Host),
        )
    }

    pub fn host_to_device(
        name: &'a str,
        data_type: DataType,
        shape: Dims,
        bytes: &'a [u8],
    ) -> Self {
        Self::HostToDevice(
            HostInputTensor::with_data_type(name, data_type, shape, bytes)
                .with_location(TensorLocation::Device),
        )
    }

    pub fn device(
        name: &'a str,
        data_type: DataType,
        shape: Dims,
        ptr: *const c_void,
        bytes: usize,
    ) -> Self {
        Self::Device(DeviceInputTensor::new(name, data_type, shape, ptr, bytes))
    }

    pub fn name(&self) -> &'a str {
        match self {
            Self::Host(input) | Self::HostToDevice(input) => input.name,
            Self::Device(input) => input.name,
        }
    }

    pub fn shape_info(&self) -> TensorShape<'a> {
        match self {
            Self::Host(input) | Self::HostToDevice(input) => input.shape_info(),
            Self::Device(input) => input.shape_info(),
        }
    }
}

impl<'a> From<HostInputTensor<'a>> for InputBinding<'a> {
    fn from(input: HostInputTensor<'a>) -> Self {
        match input.location {
            TensorLocation::Host => Self::Host(input),
            TensorLocation::Device => Self::HostToDevice(input),
        }
    }
}

impl<'a> From<DeviceInputTensor<'a>> for InputBinding<'a> {
    fn from(input: DeviceInputTensor<'a>) -> Self {
        Self::Device(input)
    }
}

/// Named host output buffer for [`Session::run_host`](crate::Session::run_host).
#[derive(Debug)]
pub struct HostOutputTensor<'a> {
    pub name: &'a str,
    pub shape: Dims,
    pub bytes: &'a mut [u8],
}

impl<'a> HostOutputTensor<'a> {
    pub fn new(name: &'a str, shape: Dims, bytes: &'a mut [u8]) -> Self {
        Self { name, shape, bytes }
    }
}

/// Named device output buffer for TensorRT device-pointer inference.
#[derive(Clone, Debug)]
pub struct DeviceOutputTensor<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub shape: Dims,
    pub ptr: *mut c_void,
    pub bytes: usize,
}

impl<'a> DeviceOutputTensor<'a> {
    pub fn new(
        name: &'a str,
        data_type: DataType,
        shape: Dims,
        ptr: *mut c_void,
        bytes: usize,
    ) -> Self {
        Self {
            name,
            data_type,
            shape,
            ptr,
            bytes,
        }
    }
}

/// Host tensor bytes returned by a TensorRT session.
#[derive(Clone, Debug)]
pub struct HostTensor {
    pub name: String,
    pub shape: Dims,
    pub data_type: DataType,
    pub bytes: Vec<u8>,
}

impl HostTensor {
    pub fn element_count(&self) -> Result<usize> {
        element_count(&self.name, &self.shape)
    }

    pub fn shape_as_usize(&self) -> Result<Vec<usize>> {
        dims_as_usize(&self.name, &self.shape)
    }

    pub fn into_f32_vec(self) -> Result<Vec<f32>> {
        self.into_vec(DataType::Float)
    }

    pub fn into_bf16_bits_vec(self) -> Result<Vec<u16>> {
        self.into_vec(DataType::Bf16)
    }

    pub fn into_f16_bits_vec(self) -> Result<Vec<u16>> {
        self.into_vec(DataType::Half)
    }

    pub fn into_i32_vec(self) -> Result<Vec<i32>> {
        self.into_vec(DataType::Int32)
    }

    pub fn into_i64_vec(self) -> Result<Vec<i64>> {
        self.into_vec(DataType::Int64)
    }

    pub fn into_u8_vec(self) -> Result<Vec<u8>> {
        self.into_vec(DataType::Uint8)
    }

    fn into_vec<T: Copy>(self, expected: DataType) -> Result<Vec<T>> {
        if self.data_type != expected {
            return Err(Error::TypeMismatch {
                tensor: self.name,
                expected,
                actual: self.data_type,
            });
        }
        bytes_into_vec(self.bytes)
    }
}

pub fn tensor_byte_len(tensor: &str, shape: &Dims, data_type: DataType) -> Result<usize> {
    let elements = element_count(tensor, shape)?;
    let Some(bytes_per_element) = data_type.bytes_per_element() else {
        return Err(Error::InvalidShape {
            tensor: tensor.to_owned(),
            reason: format!("unsupported data type {data_type:?}"),
        });
    };
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| Error::InvalidShape {
            tensor: tensor.to_owned(),
            reason: "byte size overflow".to_owned(),
        })
}

pub fn element_count(tensor: &str, shape: &Dims) -> Result<usize> {
    shape.as_slice().iter().try_fold(1usize, |acc, dim| {
        if *dim < 0 {
            return Err(Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: format!("shape still has dynamic dimension: {:?}", shape.as_slice()),
            });
        }
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: "element count overflow".to_owned(),
            })
    })
}

pub fn dims_as_usize(tensor: &str, shape: &Dims) -> Result<Vec<usize>> {
    shape
        .as_slice()
        .iter()
        .map(|dim| {
            if *dim < 0 {
                return Err(Error::InvalidShape {
                    tensor: tensor.to_owned(),
                    reason: format!("shape still has dynamic dimension: {:?}", shape.as_slice()),
                });
            }
            usize::try_from(*dim).map_err(|_| Error::InvalidShape {
                tensor: tensor.to_owned(),
                reason: format!("dimension {dim} cannot be represented as usize"),
            })
        })
        .collect()
}

pub fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), mem::size_of_val(values)) }
}

fn bytes_into_vec<T: Copy>(bytes: Vec<u8>) -> Result<Vec<T>> {
    let element_size = mem::size_of::<T>();
    if !bytes.len().is_multiple_of(element_size) {
        return Err(Error::SizeMismatch {
            tensor: "<typed-output>".to_owned(),
            expected_bytes: bytes.len() / element_size * element_size,
            actual_bytes: bytes.len(),
        });
    }

    let len = bytes.len() / element_size;
    let mut values = Vec::<T>::with_capacity(len);
    unsafe {
        let dst = values.as_mut_ptr().cast::<u8>();
        ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        values.set_len(len);
    }
    Ok(values)
}
