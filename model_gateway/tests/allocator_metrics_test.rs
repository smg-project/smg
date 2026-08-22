//! Verifies the final-artifact contract for jemalloc metrics.
//!
//! This integration test is a separate executable, so it can declare the same
//! global allocator and registration sequence used by the shipped SMG binary.

#![cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]

use smg::observability::metrics::{
    register_jemalloc_as_global_allocator, start_prometheus, PrometheusConfig,
};

#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
