//! `--runtime-worker-threads 0`, and a count above the ceiling, are
//! configuration errors: the gateway exits non-zero with the field in the
//! error, without the async runtime builder's panic and without building a
//! thread, and its port is never bound; a count of 1 serves its health route,
//! and the runtime's size reaches the log with the pinned count. The gateway
//! runs as a process on purpose: the runtime is built in `main`, before
//! `startup`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use std::time::{Duration, Instant};

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

/// A count above the ceiling (the same on every host) is refused before
/// anything listens, with the field and the ceiling in the error, and at
/// once: 100,000 used to pass, build its threads for over a minute and hold
/// gigabytes before the first bind.
#[test]
fn a_runtime_worker_thread_count_above_the_ceiling_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let started = Instant::now();
    let launch = launch(
        &["--runtime-worker-threads", "100000"],
        Until::Bound,
        &dir.path().join("huge.log"),
    );
    let elapsed = started.elapsed();
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "the gateway bound port {:?} with 100,000 worker threads\n{log}",
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
        log.contains("runtime_worker_threads") && log.contains("Must be at most"),
        "the error does not name the field and the bound\n{log}"
    );
    // Refused at validation, not after the threads were built: a fraction of
    // a second, where the threads took over a minute. The allowance is for a
    // loaded host, not for the threads.
    assert!(
        elapsed < Duration::from_secs(20),
        "the refusal took {elapsed:?}\n{log}"
    );
}

/// A count above four per CPU available to the process serves, with a
/// warning in the log that names the count and the advised one: the ceiling
/// is the refusal, the host's CPUs are advice (a runtime deliberately
/// oversubscribed is the operator's choice, which the host's quota must not
/// veto). The child reads the same CPUs as this process: it inherits the
/// affinity mask and the cgroup.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "a host the case cannot run on is told so, instead of a silent pass"
)]
fn a_runtime_worker_thread_count_above_four_per_cpu_serves_with_a_warning() {
    let cpus = std::thread::available_parallelism().map_or(1, usize::from);
    let advised = cpus * 4;
    let threads = advised + 1;
    if threads > smg::config::RUNTIME_WORKER_THREADS_CEILING {
        // On a host with more CPUs than the ceiling allows for, no count above
        // the advised one is accepted; there is nothing to warn about short of
        // a refusal, which the ceiling test covers.
        eprintln!(
            "skipped: {cpus} CPUs advise {advised} worker threads, and one more is above the \
             ceiling of {}",
            smg::config::RUNTIME_WORKER_THREADS_CEILING
        );
        return;
    }
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &[
            "--runtime-worker-threads",
            &threads.to_string(),
            "--log-json",
        ],
        Until::Served,
        &dir.path().join("over.log"),
    );
    assert!(
        launch.served,
        "the gateway did not serve its health route with {threads} worker threads (exit {:?})\n{}",
        launch.status, launch.log
    );
    let expected = format!("runtime_worker_threads {threads} is above {advised}");
    let warned = launch
        .log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|line| {
            line["message"]
                .as_str()
                .is_some_and(|message| message.starts_with(&expected))
        })
        .unwrap_or_else(|| panic!("the warning did not reach the log\n{}", launch.log));
    assert_eq!(warned["level"], "WARN", "{warned}");
    assert!(
        warned["message"]
            .as_str()
            .is_some_and(|message| message.contains(&format!("{cpus} CPUs available"))),
        "{warned}"
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
