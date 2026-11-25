//! What the crate's manifest cases share: the read of the manifests cargo
//! sees from the crate's directory.

use std::{error::Error, process::Command};

use serde_json::Value;

pub type Outcome<T> = Result<T, Box<dyn Error>>;

/// The manifests cargo reads from the crate's directory, as
/// `cargo metadata --no-deps` prints them: every member of the workspace in
/// the checkout, this package alone in a published copy.
pub fn cargo_metadata() -> Outcome<Value> {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// This package's entry among the manifests of `metadata`.
pub fn this_package(metadata: &Value) -> Outcome<&Value> {
    const NAME: &str = env!("CARGO_PKG_NAME");
    metadata["packages"]
        .as_array()
        .and_then(|packages| packages.iter().find(|package| package["name"] == NAME))
        .ok_or_else(|| format!("cargo metadata lists no {NAME}").into())
}
