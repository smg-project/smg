use super::{parse_worker_threads, DEFAULT_WORKER_THREADS};

/// One worker by default: every stream's per-token send then takes the
/// connection's lock from the same core, and no engine output batch
/// wakes tasks across threads.
#[test]
fn one_worker_by_default() {
    assert_eq!(parse_worker_threads(None), 1);
}

#[test]
fn worker_threads_default_and_override() {
    assert_eq!(parse_worker_threads(None), DEFAULT_WORKER_THREADS);
    assert_eq!(parse_worker_threads(Some(" 8 ")), 8);
    // Zero and garbage fall back to the default rather than panicking.
    assert_eq!(parse_worker_threads(Some("0")), DEFAULT_WORKER_THREADS);
    assert_eq!(parse_worker_threads(Some("many")), DEFAULT_WORKER_THREADS);
}
