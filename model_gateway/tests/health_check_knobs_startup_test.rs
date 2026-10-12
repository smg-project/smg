//! The gateway refuses to start on a health-check knob set to 0 (the probe
//! interval, the probe timeout, the failure and the success threshold): the
//! configuration error names the field, the process exits non-zero without
//! a panic, and its port is never bound. A zero interval used to start the
//! gateway with a busy probe loop. The gateway runs as a process on purpose:
//! the check belongs before the runtime and the listener exist.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use gateway_launch::{launch, Until, EXIT_TIMEOUT};

/// Each knob at 0 is refused before anything listens, with the field in the
/// error and no panic.
#[test]
fn a_zero_health_check_knob_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let knobs = [
        ("--health-check-interval-secs", "check_interval_secs"),
        ("--health-check-timeout-secs", "timeout_secs"),
        ("--health-failure-threshold", "failure_threshold"),
        ("--health-success-threshold", "success_threshold"),
    ];
    for (flag, field) in knobs {
        let launch = launch(
            &[flag, "0"],
            Until::Bound,
            &dir.path()
                .join(format!("{}.log", flag.trim_start_matches('-'))),
        );
        let log = &launch.log;
        assert!(
            launch.bound.is_none(),
            "{flag} 0: the gateway bound port {:?} with a health-check knob at 0\n{log}",
            launch.bound
        );
        let status = launch.status.unwrap_or_else(|| {
            panic!(
                "{flag} 0: the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}"
            )
        });
        assert!(
            !status.success(),
            "{flag} 0: the gateway exited successfully\n{log}"
        );
        assert!(
            !log.contains("panicked"),
            "{flag} 0: a panic is not a configuration error\n{log}"
        );
        assert!(
            log.contains(field) && log.contains(flag),
            "{flag} 0: the error names neither the field nor the flag\n{log}"
        );
    }
}

/// The smallest interval is a configuration like any other: the gateway
/// serves its health route.
#[test]
fn a_one_second_health_check_interval_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--health-check-interval-secs", "1"],
        Until::Served,
        &dir.path().join("interval-1.log"),
    );
    assert!(
        launch.served,
        "the gateway did not serve its health route with a one-second probe interval (status {:?})\n{}",
        launch.status, launch.log
    );
}
