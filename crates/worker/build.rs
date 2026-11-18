//! Hands the crate's proto directory to a dependent's build script. The
//! file is copied under `OUT_DIR` and that directory is printed: cargo passes
//! it to each crate that depends on this one directly, as
//! `DEP_SMG_WORKER_PROTO_DIR`. A path under `OUT_DIR` follows the target
//! directory when a checkout moves and is made again when the file changes;
//! a path into the source tree would do neither. Nothing is compiled here.

use std::{env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=proto/common.proto");
    let out_dir = env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("OUT_DIR is not set"))?;
    let proto_dir = PathBuf::from(out_dir).join("proto");
    fs::create_dir_all(&proto_dir)?;
    fs::copy("proto/common.proto", proto_dir.join("common.proto"))?;
    println!("cargo::metadata=dir={}", proto_dir.display());
    Ok(())
}
