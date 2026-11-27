use metrics::{describe_gauge, gauge};

use super::jemalloc_is_global;

pub(crate) fn describe() {
    if !jemalloc_is_global() {
        return;
    }
    describe_gauge!(
        "smg_allocator_allocated_bytes",
        "Bytes in live Rust allocations managed by SMG's jemalloc instance"
    );
    describe_gauge!(
        "smg_allocator_active_bytes",
        "Bytes in active pages for SMG's Rust jemalloc heap"
    );
    describe_gauge!(
        "smg_allocator_resident_bytes",
        "Upper bound on resident bytes for SMG's Rust jemalloc heap"
    );
    describe_gauge!(
        "smg_allocator_metadata_bytes",
        "Metadata bytes for SMG's Rust jemalloc instance"
    );
}

fn record() {
    use tikv_jemalloc_ctl::{epoch, stats};
    if epoch::advance().is_err() {
        return;
    }
    if let Ok(v) = stats::allocated::read() {
        gauge!("smg_allocator_allocated_bytes").set(v as f64);
    }
    if let Ok(v) = stats::active::read() {
        gauge!("smg_allocator_active_bytes").set(v as f64);
    }
    if let Ok(v) = stats::resident::read() {
        gauge!("smg_allocator_resident_bytes").set(v as f64);
    }
    if let Ok(v) = stats::metadata::read() {
        gauge!("smg_allocator_metadata_bytes").set(v as f64);
    }
}

/// The gauges are published where this jemalloc serves the process's Rust
/// allocations, and nowhere else.
pub(crate) fn start_reporting() {
    if !jemalloc_is_global() {
        return;
    }
    record();
    // Plain thread: metrics must not depend on a runtime being alive.
    let _ = std::thread::Builder::new()
        .name("smg-allocator-stats".into())
        .spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            record();
        });
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_allocator_of_the_lib_tests_is_found() {
        // The lib's tests declare jemalloc as their allocator (lib.rs),
        // so the one allocation moves its count.
        assert!(super::jemalloc_is_global());
    }

    #[test]
    fn jemalloc_stats_interface_readable() {
        use tikv_jemalloc_ctl::{epoch, stats};
        epoch::advance().expect("epoch advance");
        stats::allocated::read().expect("stats.allocated readable");
        stats::active::read().expect("stats.active readable");
        stats::resident::read().expect("stats.resident readable");
        stats::metadata::read().expect("stats.metadata readable");
    }
}
