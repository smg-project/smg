//! The gateway refuses to start on a priority-scheduler configuration file
//! it cannot use: the configuration error names the field and the file, the
//! process exits non-zero without a panic, and its port is never bound. The
//! gateway runs as a process on purpose: the check belongs before the
//! runtime and the listener exist.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use std::fs;

use gateway_launch::{launch, Until, EXIT_TIMEOUT};

/// A path that does not exist and a file that is not YAML: both are refused
/// before anything listens, with the field and the file in the error.
#[test]
fn an_unusable_priority_scheduler_config_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let missing = dir.path().join("missing.yaml");
    let garbage = dir.path().join("garbage.yaml");
    fs::write(&garbage, "not: [valid: yaml\n").expect("write the file");
    for (name, path) in [("missing", &missing), ("garbage", &garbage)] {
        let path = path.to_str().expect("a UTF-8 path");
        let launch = launch(
            &[
                "--priority-scheduler-enabled",
                "--priority-scheduler-config",
                path,
            ],
            Until::Bound,
            &dir.path().join(format!("{name}.log")),
        );
        let log = &launch.log;
        assert!(
            launch.bound.is_none(),
            "{name}: the gateway bound port {:?} with a scheduler config it cannot use\n{log}",
            launch.bound
        );
        let status = launch.status.unwrap_or_else(|| {
            panic!("{name}: the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
        });
        assert!(
            !status.success(),
            "{name}: the gateway exited successfully\n{log}"
        );
        assert!(
            !log.contains("panicked"),
            "{name}: a panic is not a configuration error\n{log}"
        );
        assert!(
            log.contains("priority_scheduler_config") && log.contains(path),
            "{name}: the error names neither the field nor the file\n{log}"
        );
    }
}
