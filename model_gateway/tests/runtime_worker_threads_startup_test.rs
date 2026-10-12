//! `--runtime-worker-threads 0` is a configuration error: the gateway exits
//! non-zero with the field in the error, without the async runtime builder's
//! panic, and its port is never bound; a count of 1 serves its health route,
//! and the runtime's size reaches the log with the pinned count. The gateway
//! runs as a process on purpose: the runtime is built in `main`, before
//! `startup`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use gateway_launch::{launch, Until, EXIT_TIMEOUT};

/// Zero is refused before anything listens, with the field in the error and
/// without a panic.
#[test]
fn a_zero_runtime_worker_thread_count_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--runtime-worker-threads", "0"],
        Until::Bound,
        &dir.path().join("zero.log"),
    );
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "the gateway bound port {:?} with a zero worker-thread count\n{log}",
        launch.bound
    );
    let status = launch.status.unwrap_or_else(|| {
        panic!("the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
    });
    assert!(!status.success(), "the gateway exited successfully\n{log}");
    assert!(
        !log.contains("panicked"),
        "a panic is not a configuration error\n{log}"
    );
    assert!(
        log.contains("runtime_worker_threads"),
        "the error does not name the field\n{log}"
    );
}

/// The smallest count the runtime accepts serves, and the runtime's size is
/// in the log with the pinned count: the line is written once the log
/// subscriber is up, so the positive case can read it (emitted before the
/// subscriber existed, it reached nothing). The log is JSON so the field is
/// read exactly.
#[test]
fn a_runtime_worker_thread_count_of_one_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--runtime-worker-threads", "1", "--log-json"],
        Until::Served,
        &dir.path().join("one.log"),
    );
    assert!(
        launch.served,
        "the gateway did not serve its health route with one worker thread (exit {:?})\n{}",
        launch.status, launch.log
    );
    // The line is flat JSON: read it exactly, so a default-sized runtime on
    // a host whose core count starts with 1 cannot pass for the pinned one.
    let sized = launch
        .log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|line| line["message"] == "Tokio runtime sized")
        .unwrap_or_else(|| panic!("the runtime's size did not reach the log\n{}", launch.log));
    assert_eq!(sized["worker_threads"], 1, "{sized}");
    assert_eq!(sized["pinned"], true, "{sized}");
}
