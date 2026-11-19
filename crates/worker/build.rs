//! Hands the crate's proto directory to a dependent's build script. The
//! file is copied under `OUT_DIR` and that directory is printed: cargo passes
//! it to each crate that depends on this one directly, as
//! `DEP_SMG_WORKER_PROTO_DIR`. A path under `OUT_DIR` follows the target
//! directory when a checkout moves and is made again when the file changes;
//! a path into the source tree would do neither.
//!
//! The `proto` feature also generates the file's messages here, for the
//! crate's `common_proto` module. Without it nothing is compiled and the
//! build needs no `protoc`.

use std::{env, fs, io, path::PathBuf};

const PROTO: &str = "proto/common.proto";

fn main() -> io::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={PROTO}");
    let out_dir = env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("OUT_DIR is not set"))?;
    let proto_dir = PathBuf::from(out_dir).join("proto");
    fs::create_dir_all(&proto_dir)?;
    let copy = proto_dir.join("common.proto");
    fs::copy(PROTO, &copy)
        .map_err(|e| io::Error::new(e.kind(), format!("{PROTO} to {}: {e}", copy.display())))?;
    println!("cargo::metadata=dir={}", proto_dir.display());

    // The messages alone: the file holds no service. proto3 `optional` needs
    // the flag below protoc 3.15, the oldest the repository's build scripts
    // run under. The generator finds its compiler by `PROTOC`, so another
    // compiler there makes the messages again.
    #[cfg(feature = "proto")]
    {
        println!("cargo::rerun-if-env-changed=PROTOC");
        prost_build::Config::new()
            .protoc_arg("--experimental_allow_proto3_optional")
            .compile_protos(&[PROTO], &["proto"])?;
    }
    Ok(())
}
