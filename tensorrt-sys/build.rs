use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/tensorrt_runtime.cpp");
    println!("cargo:rerun-if-changed=native/tensorrt_runtime.h");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let tensorrt_root = env::var_os("TENSORRT_ROOT").map(PathBuf::from);
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

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("native/tensorrt_runtime.cpp")
        .include("native")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter");

    for dir in include_dirs.iter().filter(|dir| dir.is_dir()) {
        build.include(dir);
    }

    build.compile("tensorrt_rs_runtime");

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
    emit_rpath("/usr/local/tensorrt/lib");
    emit_rpath("/usr/local/tensorrt/lib64");
    emit_rpath("/opt/tensorrt/lib");
    emit_rpath("/opt/tensorrt/lib64");
    emit_link_search(cuda_root.join("lib64"));
    emit_link_search(cuda_root.join("targets/x86_64-linux/lib"));
    emit_rpath(cuda_root.join("lib64"));
    emit_rpath(cuda_root.join("targets/x86_64-linux/lib"));

    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=cudart");
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
