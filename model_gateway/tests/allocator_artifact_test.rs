//! Verifies that the shipped SMG executable owns the expected jemalloc.

#![cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use std::process::Command;

#[test]
fn smg_binary_uses_global_jemalloc() {
    let output = Command::new(env!("CARGO_BIN_EXE_smg"))
        .arg("--version")
        .env("_RJEM_MALLOC_CONF", "stats_print:true,stats_print_opts:J")
        .output()
        .expect("run the SMG executable");
    assert!(
        output.status.success(),
        "SMG executable failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stats = String::from_utf8_lossy(&output.stderr);
    assert!(
        stats.contains("\"jemalloc\""),
        "SMG executable did not emit jemalloc statistics: {stats}"
    );

    #[cfg(all(target_arch = "aarch64", target_env = "gnu"))]
    assert!(
        stats.contains("\"page\":65536"),
        "aarch64 SMG executable was not built for 64 KiB page compatibility: {stats}"
    );
}
