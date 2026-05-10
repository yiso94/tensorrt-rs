use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/tensorrt_llm_executor.cpp");
    println!("cargo:rerun-if-changed=native/tensorrt_llm_executor.h");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LLM_ROOT");
    println!("cargo:rerun-if-env-changed=TENSORRT_LLM_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LLM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let tensorrt_root = env::var_os("TENSORRT_ROOT").map(PathBuf::from);
    let tensorrt_llm_root = env::var_os("TENSORRT_LLM_ROOT").map(PathBuf::from);
    let cuda_root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));

    let mut include_dirs = Vec::new();
    if let Some(include_dir) = env::var_os("TENSORRT_INCLUDE_DIR") {
        include_dirs.push(PathBuf::from(include_dir));
    }
    if let Some(root) = tensorrt_root.as_deref() {
        include_dirs.push(root.join("include"));
    }
    include_dirs.push(cuda_root.join("include"));
    include_dirs.push(PathBuf::from("/usr/include"));
    include_dirs.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));
    include_dirs.push(PathBuf::from("/usr/local/include"));
    include_dirs.push(PathBuf::from("/usr/local/tensorrt/include"));
    include_dirs.push(PathBuf::from("/usr/local/TensorRT/include"));
    include_dirs.push(PathBuf::from("/opt/tensorrt/include"));
    if let Some(include_dir) = env::var_os("TENSORRT_LLM_INCLUDE_DIR") {
        include_dirs.push(PathBuf::from(include_dir));
    }
    if let Some(root) = tensorrt_llm_root.as_deref() {
        include_dirs.push(root.join("include"));
    }
    include_dirs.push(PathBuf::from("/app/tensorrt_llm/include"));
    include_dirs.push(PathBuf::from(
        "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/include",
    ));
    extend_split_paths(&mut include_dirs, "CPATH");
    extend_split_paths(&mut include_dirs, "CPLUS_INCLUDE_PATH");

    if !include_dirs
        .iter()
        .any(|dir| dir.join("NvInfer.h").is_file())
    {
        panic!(
            "TensorRT header NvInfer.h was not found. Install TensorRT, or set \
             TENSORRT_ROOT=/path/to/TensorRT or TENSORRT_INCLUDE_DIR=/path/to/include."
        );
    }
    if !include_dirs
        .iter()
        .any(|dir| dir.join("tensorrt_llm/executor/executor.h").is_file())
    {
        panic!(
            "TensorRT-LLM header tensorrt_llm/executor/executor.h was not found. Install \
             TensorRT-LLM, or set TENSORRT_LLM_ROOT=/path/to/TensorRT-LLM or \
             TENSORRT_LLM_INCLUDE_DIR=/path/to/include."
        );
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("native/tensorrt_llm_executor.cpp")
        .include("native")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter");

    for dir in include_dirs.iter().filter(|dir| dir.is_dir()) {
        build.include(dir);
    }

    build.compile("tensorrt_llm_rs_executor");

    if let Some(lib_dir) = env::var_os("TENSORRT_LIB_DIR") {
        let lib_dir = PathBuf::from(lib_dir);
        emit_link_search(&lib_dir);
        emit_rpath(&lib_dir);
    }
    if let Some(root) = tensorrt_root.as_deref() {
        emit_link_search(root.join("lib"));
        emit_link_search(root.join("lib64"));
        emit_rpath(root.join("lib"));
        emit_rpath(root.join("lib64"));
    }
    emit_link_search("/usr/local/tensorrt/lib");
    emit_link_search("/usr/local/tensorrt/lib64");
    emit_link_search("/opt/tensorrt/lib");
    emit_link_search("/opt/tensorrt/lib64");
    emit_link_search(cuda_root.join("lib64"));
    emit_link_search(cuda_root.join("targets/x86_64-linux/lib"));
    emit_rpath(cuda_root.join("lib64"));
    emit_rpath(cuda_root.join("targets/x86_64-linux/lib"));

    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Some(lib_dir) = env::var_os("TENSORRT_LLM_LIB_DIR") {
        let lib_dir = PathBuf::from(lib_dir);
        emit_link_search(&lib_dir);
        emit_rpath(&lib_dir);
    }
    if let Some(root) = tensorrt_llm_root.as_deref() {
        emit_link_search(root.join("lib"));
        emit_link_search(root.join("lib64"));
        emit_link_search(root.join("libs"));
        emit_rpath(root.join("lib"));
        emit_rpath(root.join("lib64"));
        emit_rpath(root.join("libs"));
    }
    let python_trtllm_libs =
        PathBuf::from("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs");
    emit_link_search(&python_trtllm_libs);
    emit_rpath(&python_trtllm_libs);

    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
}

fn emit_link_search(path: impl AsRef<Path>) {
    let path = path.as_ref();
    if path.is_dir() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}

fn emit_rpath(path: impl AsRef<Path>) {
    let path = path.as_ref();
    if path.is_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}

fn extend_split_paths(paths: &mut Vec<PathBuf>, var: &str) {
    if let Some(value) = env::var_os(var) {
        paths.extend(env::split_paths(&value));
    }
}
