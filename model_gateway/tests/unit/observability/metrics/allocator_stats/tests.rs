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
