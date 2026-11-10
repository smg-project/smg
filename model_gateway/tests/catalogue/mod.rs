//! Catalogue checks: the metric families, the dashboard and the marker tokens agree with the sources.
//! The checks that walk the sources share the repository root and the walk.

use std::{
    fs,
    path::{Path, PathBuf},
};

mod grafana_dashboard_test;
mod marker_tokens_test;
mod metrics_aggregator_test;
mod metrics_catalogue_test;

/// The workspace root, one level above the gateway crate.
#[expect(clippy::expect_used, reason = "test helper")]
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the gateway crate lives one level under the workspace root")
        .to_path_buf()
}

/// Every `.rs` file under `dir`, build directories skipped.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if name != "target" && !name.to_string_lossy().starts_with('.') {
                rust_files(&path, out);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}
