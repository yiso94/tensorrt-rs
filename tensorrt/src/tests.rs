use super::*;

#[test]
fn binding_dtype_parses_python_mapping() {
    let pairs = [
        ("bfloat16", DataType::Bf16),
        ("float16", DataType::Half),
        ("float32", DataType::Float),
        ("int64", DataType::Int64),
        ("int32", DataType::Int32),
        ("fp8", DataType::Fp8),
    ];

    for (dtype, data_type) in pairs {
        assert_eq!(dtype.parse::<DataType>().unwrap(), data_type);
        assert_eq!(data_type.to_string(), dtype);
        assert_eq!(DataType::from_binding_dtype(dtype), Some(data_type));
        assert_eq!(data_type.as_binding_dtype(), Some(dtype));
    }
}

#[test]
fn binding_dtype_conversion_rejects_non_python_mapping_entries() {
    assert!("int8".parse::<DataType>().is_err());
    assert!("bool".parse::<DataType>().is_err());
    assert!("uint8".parse::<DataType>().is_err());
    assert_eq!(DataType::Int8.as_binding_dtype(), None);
    assert_eq!(DataType::Bool.as_binding_dtype(), None);
    assert_eq!(DataType::Uint8.as_binding_dtype(), None);
    assert_eq!(DataType::Int4.as_binding_dtype(), None);
    assert_eq!(DataType::Unknown(-1).as_binding_dtype(), None);
}

#[test]
fn raw_dtype_and_io_mode_mappings_cover_known_and_unknown_values() {
    let dtypes = [
        (0, DataType::Float, Some(4)),
        (1, DataType::Half, Some(2)),
        (2, DataType::Int8, Some(1)),
        (3, DataType::Int32, Some(4)),
        (4, DataType::Bool, Some(1)),
        (5, DataType::Uint8, Some(1)),
        (6, DataType::Fp8, Some(1)),
        (7, DataType::Bf16, Some(2)),
        (8, DataType::Int64, Some(8)),
        (9, DataType::Int4, None),
        (10, DataType::Fp4, None),
        (11, DataType::E8m0, None),
        (99, DataType::Unknown(99), None),
    ];
    for (raw, dtype, bytes) in dtypes {
        assert_eq!(DataType::from_raw(raw), dtype);
        assert_eq!(dtype.bytes_per_element(), bytes);
    }

    assert_eq!(TensorIOMode::from_raw(0), TensorIOMode::None);
    assert_eq!(TensorIOMode::from_raw(1), TensorIOMode::Input);
    assert_eq!(TensorIOMode::from_raw(2), TensorIOMode::Output);
    assert_eq!(TensorIOMode::from_raw(99), TensorIOMode::Unknown(99));
}

#[test]
fn host_tensor_typed_vec_helpers_keep_raw_bits() {
    let bf16_bits = [0x3f80u16, 0x4000];
    let bf16_tensor = HostTensor {
        name: "bf16_output".to_owned(),
        shape: Dims::new(vec![2]).unwrap(),
        data_type: DataType::Bf16,
        bytes: as_bytes(&bf16_bits).to_vec(),
    };
    assert_eq!(bf16_tensor.into_bf16_bits_vec().unwrap(), bf16_bits);

    let f16_bits = [0x3c00u16, 0x4000];
    let f16_tensor = HostTensor {
        name: "f16_output".to_owned(),
        shape: Dims::new(vec![2]).unwrap(),
        data_type: DataType::Half,
        bytes: as_bytes(&f16_bits).to_vec(),
    };
    assert_eq!(f16_tensor.into_f16_bits_vec().unwrap(), f16_bits);
}

#[test]
fn bf16_conversion_uses_expected_bit_pattern() {
    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(f32_to_bf16_bits(0.0), 0x0000);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(f32_to_bf16_bits(1.003_906_3), 0x3f80);
}

#[test]
fn dims_reject_rank_above_ffi_limit() {
    let err = Dims::new(vec![1; MAX_DIMS + 1]).unwrap_err();
    match err {
        Error::InvalidDimensions { len, max } => {
            assert_eq!(len, MAX_DIMS + 1);
            assert_eq!(max, MAX_DIMS);
        }
        err => panic!("expected InvalidDimensions, got {err:?}"),
    }
}

#[test]
fn dims_and_tensor_metadata_roundtrip_through_public_helpers() {
    let dims = Dims::new([2, 3, 4]).unwrap();
    assert_eq!(dims.rank(), 3);
    assert!(!dims.is_dynamic());
    assert_eq!(Vec::<i64>::from(dims.clone()), vec![2, 3, 4]);

    let ffi = dims.to_ffi();
    assert_eq!(Dims::from_ffi(ffi).unwrap(), dims);
    assert!(
        Dims::from_ffi(crate::ffi::Dims {
            nb_dims: -1,
            d: [0; MAX_DIMS]
        })
        .is_err()
    );

    let info = TensorInfo::new("input", dims.clone(), DataType::Float);
    assert_eq!(info.shape_info(), TensorShape::new("input", dims.clone()));
    assert_eq!(info.io_mode, TensorIOMode::Unknown(-1));

    let output =
        TensorInfo::with_io_mode("output", DataType::Half, dims.clone(), TensorIOMode::Output);
    assert_eq!(output.shape_info().name, "output");
    assert_eq!(output.io_mode, TensorIOMode::Output);
}

#[test]
fn tensor_size_helpers_reject_dynamic_shapes_and_overflow() {
    let dynamic = Dims::new([-1, 4]).unwrap();
    assert!(element_count("input", &dynamic).is_err());
    assert!(dims_as_usize("input", &dynamic).is_err());

    let overflowing = Dims::new([i64::MAX, 3]).unwrap();
    assert!(element_count("input", &overflowing).is_err());
    assert!(tensor_byte_len("input", &overflowing, DataType::Float).is_err());
}

#[test]
fn tensor_byte_len_rejects_unknown_dtype() {
    let shape = Dims::new([2, 3]).unwrap();
    assert!(tensor_byte_len("output", &shape, DataType::Unknown(-1)).is_err());
}

#[test]
fn host_tensor_typed_conversions_reject_dtype_and_size_mismatches() {
    let tensor = HostTensor {
        name: "logits".to_owned(),
        shape: Dims::new([1]).unwrap(),
        data_type: DataType::Float,
        bytes: vec![0, 0, 0, 0],
    };
    assert!(tensor.into_bf16_bits_vec().is_err());

    let tensor = HostTensor {
        name: "bad".to_owned(),
        shape: Dims::new([1]).unwrap(),
        data_type: DataType::Float,
        bytes: vec![0, 1, 2],
    };
    assert!(tensor.into_f32_vec().is_err());
}

#[test]
fn host_tensor_typed_conversions_cover_integer_outputs() {
    let tensor = HostTensor {
        name: "logits".to_owned(),
        shape: Dims::new([2]).unwrap(),
        data_type: DataType::Float,
        bytes: as_bytes(&[1.0f32, 2.0]).to_vec(),
    };
    assert_eq!(tensor.element_count().unwrap(), 2);
    assert_eq!(tensor.shape_as_usize().unwrap(), vec![2]);
    assert_eq!(tensor.into_f32_vec().unwrap(), vec![1.0, 2.0]);

    let tensor = HostTensor {
        name: "ids".to_owned(),
        shape: Dims::new([2]).unwrap(),
        data_type: DataType::Int32,
        bytes: as_bytes(&[1i32, 2]).to_vec(),
    };
    assert_eq!(tensor.into_i32_vec().unwrap(), vec![1, 2]);

    let tensor = HostTensor {
        name: "lengths".to_owned(),
        shape: Dims::new([1]).unwrap(),
        data_type: DataType::Int64,
        bytes: as_bytes(&[3i64]).to_vec(),
    };
    assert_eq!(tensor.into_i64_vec().unwrap(), vec![3]);

    let tensor = HostTensor {
        name: "mask".to_owned(),
        shape: Dims::new([3]).unwrap(),
        data_type: DataType::Uint8,
        bytes: vec![1, 0, 1],
    };
    assert_eq!(tensor.into_u8_vec().unwrap(), vec![1, 0, 1]);
}

#[test]
fn input_binding_constructors_preserve_kind_and_shape() {
    let bytes = [0_u8; 4];
    let shape = Dims::new([1]).unwrap();

    let host = InputBinding::host("host", DataType::Float, shape.clone(), &bytes);
    let staged = InputBinding::host_to_device("staged", DataType::Float, shape.clone(), &bytes);
    let device = InputBinding::device(
        "device",
        DataType::Float,
        shape.clone(),
        bytes.as_ptr().cast(),
        bytes.len(),
    );

    assert_eq!(host.name(), "host");
    assert_eq!(host.shape_info().shape, shape);
    assert!(matches!(host, InputBinding::Host(_)));
    assert!(matches!(staged, InputBinding::HostToDevice(_)));
    assert!(matches!(device, InputBinding::Device(_)));

    let host_input = HostInputTensor::new("raw", shape.clone(), &bytes);
    assert_eq!(host_input.location, TensorLocation::Device);
    assert_eq!(
        host_input.shape_info(),
        TensorShape::new("raw", shape.clone())
    );

    let mut output_bytes = [0_u8; 4];
    let host_output = HostOutputTensor::new("out", shape.clone(), &mut output_bytes);
    assert_eq!(host_output.name, "out");
    assert_eq!(host_output.shape, shape);

    let device_output = DeviceOutputTensor::new(
        "device_out",
        DataType::Float,
        Dims::new([1]).unwrap(),
        output_bytes.as_mut_ptr().cast(),
        output_bytes.len(),
    );
    assert_eq!(device_output.name, "device_out");
    assert_eq!(device_output.data_type, DataType::Float);
    assert_eq!(device_output.bytes, output_bytes.len());
}
