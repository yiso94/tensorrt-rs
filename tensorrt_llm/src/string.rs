use std::ffi::CString;

pub(crate) fn cstring(value: impl AsRef<str>, field: &str) -> CString {
    let value = value.as_ref();
    assert!(
        !value.as_bytes().contains(&0),
        "TensorRT-LLM {field} cannot contain NUL bytes"
    );
    CString::new(value).expect("NUL bytes were checked")
}
