//! Verifies that the shipped SMG executable owns the expected jemalloc and
//! runs it with the gateway's long-running-server options.

#![cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use std::process::Command;

/// Runs the executable with jemalloc's JSON statistics printed at exit under
/// the given `_RJEM_MALLOC_CONF` and returns the statistics text (stderr).
fn jemalloc_stats(malloc_conf: &str) -> Result<String, String> {
    let output = Command::new(env!("CARGO_BIN_EXE_smg"))
        .arg("--version")
        .env("_RJEM_MALLOC_CONF", malloc_conf)
        .output()
        .map_err(|e| format!("run the SMG executable: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "SMG executable failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

/// The `"opt"` object of jemalloc's JSON statistics: the options in effect.
/// It holds scalars only, so it ends at the first closing brace.
fn opt_section(stats: &str) -> Option<&str> {
    let start = stats.find("\"opt\":{")?;
    let end = start + stats[start..].find('}')?;
    Some(&stats[start..end])
}

#[test]
fn smg_binary_uses_global_jemalloc() {
    let stats = jemalloc_stats("stats_print:true,stats_print_opts:J")
        .expect("jemalloc statistics from the SMG executable");
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

#[test]
fn smg_binary_applies_the_server_malloc_conf() {
    // The executable's `_rjem_malloc_conf` symbol: purge on a background
    // thread, dirty pages back to the OS after 10 s, muzzy pages at once.
    let stats = jemalloc_stats("stats_print:true,stats_print_opts:J")
        .expect("jemalloc statistics from the SMG executable");
    let opt = opt_section(&stats).expect("an \"opt\" section in the jemalloc statistics");
    for expected in [
        "\"background_thread\":true",
        "\"dirty_decay_ms\":10000",
        "\"muzzy_decay_ms\":0",
    ] {
        assert!(
            opt.contains(expected),
            "jemalloc did not take the executable's malloc_conf ({expected} missing): {opt}"
        );
    }
}

#[test]
fn environment_overrides_the_server_malloc_conf_entry_by_entry() {
    let stats = jemalloc_stats(
        "stats_print:true,stats_print_opts:J,background_thread:false,dirty_decay_ms:2000",
    )
    .expect("jemalloc statistics from the SMG executable");
    let opt = opt_section(&stats).expect("an \"opt\" section in the jemalloc statistics");
    for expected in [
        "\"background_thread\":false",
        "\"dirty_decay_ms\":2000",
        "\"muzzy_decay_ms\":0",
    ] {
        assert!(
            opt.contains(expected),
            "_RJEM_MALLOC_CONF did not override the executable's malloc_conf ({expected} missing): {opt}"
        );
    }
}
