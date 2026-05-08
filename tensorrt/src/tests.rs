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
fn tensor_size_helpers_reject_dynamic_shapes_and_overflow() {
    let dynamic = Dims::new([-1, 4]).unwrap();
    assert!(element_count("input", &dynamic).is_err());
    assert!(dims_to_usize_shape("input", &dynamic).is_err());

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
fn host_tensor_into_vec_rejects_dtype_and_size_mismatches() {
    let tensor = HostTensor {
        name: "logits".to_owned(),
        shape: Dims::new([1]).unwrap(),
        data_type: DataType::Float,
        bytes: vec![0, 0, 0, 0],
    };
    assert!(tensor.into_vec::<u16>(DataType::Half).is_err());

    let tensor = HostTensor {
        name: "bad".to_owned(),
        shape: Dims::new([1]).unwrap(),
        data_type: DataType::Float,
        bytes: vec![0, 1, 2],
    };
    assert!(tensor.into_vec::<f32>(DataType::Float).is_err());
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
}
