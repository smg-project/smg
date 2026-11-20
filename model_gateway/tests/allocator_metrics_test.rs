//! Verifies the final-artifact contract for jemalloc metrics.
//!
//! This integration test is a separate executable. It takes the stock
//! allocator by the library's macro, as a host's own binary does, and makes
//! the registration the shipped SMG binary makes.

#![cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use smg::observability::metrics::{
    register_jemalloc_as_global_allocator, start_prometheus, PrometheusConfig,
};

smg::stock_allocator!();

#[test]
fn registered_global_jemalloc_exposes_allocator_gauges() {
    use tikv_jemalloc_ctl::thread;

    // A cumulative, thread-local counter proves an ordinary Rust allocation
    // went through the declared global jemalloc without races from other tests.
    let allocated = thread::allocatedp::read().expect("thread allocation counter");
    let before = allocated.get();
    let allocation = std::hint::black_box(vec![0_u8; 1024 * 1024]);
    let after = allocated.get();
    assert!(after > before, "Rust allocation bypassed jemalloc");
    std::hint::black_box(&allocation);

    // The macro exports the stock options too. The allocator reads them at
    // its start, and sixteen arenas is one of them.
    let arenas = tikv_jemalloc_ctl::opt::narenas::read().expect("the arenas option");
    assert_eq!(arenas, 16, "the stock options were not exported");

    register_jemalloc_as_global_allocator();
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let body = handle.render();

    for metric in [
        "smg_allocator_allocated_bytes",
        "smg_allocator_active_bytes",
        "smg_allocator_resident_bytes",
        "smg_allocator_metadata_bytes",
    ] {
        assert!(
            body.lines().any(|line| line.starts_with(metric)),
            "registered allocator metric {metric} missing:\n{body}"
        );
    }
}
