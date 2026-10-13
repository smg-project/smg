//! The gateway refuses to start on a priority-scheduler configuration file
//! it cannot use, on a file with a key it does not know and on a default max
//! class that is none of the four: the configuration error names the field
//! and the value, the process exits non-zero without a panic, and its port
//! is never bound. The gateway runs as a process on purpose: the check
//! belongs before the runtime and the listener exist.
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

/// A key the scheduler's YAML does not know and a default max class that is
/// none of the four names are refused the same way: dropping the key, or
/// mapping the name to `default` as the request header parser does, would
/// hide an operator's typo behind a scheduler serving without the policy
/// they wrote.
#[test]
fn an_unknown_key_or_default_max_class_name_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let unknown_key = dir.path().join("unknown-key.yaml");
    fs::write(
        &unknown_key,
        "classes: {}\ntenant_policies: {}\nextra_field: 1\n",
    )
    .expect("write the file");
    let unknown_key = unknown_key.to_str().expect("a UTF-8 path");
    let cases = [
        (
            "unknown-key",
            ["--priority-scheduler-config", unknown_key],
            ["priority_scheduler_config", "extra_field"],
        ),
        (
            "unknown-class",
            ["--priority-scheduler-default-max-class", "gold"],
            ["priority_scheduler_default_max_class", "gold"],
        ),
    ];
    for (name, flags, named) in cases {
        let mut args = vec!["--priority-scheduler-enabled"];
        args.extend(flags);
        let launch = launch(&args, Until::Bound, &dir.path().join(format!("{name}.log")));
        let log = &launch.log;
        assert!(
            launch.bound.is_none(),
            "{name}: the gateway bound port {:?} with a scheduler configuration it should refuse\n{log}",
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
        for word in named {
            assert!(
                log.contains(word),
                "{name}: the error does not name `{word}`\n{log}"
            );
        }
    }
}

/// A known class name is accepted in any case: the gateway comes up.
#[test]
fn a_known_default_max_class_name_in_any_case_starts() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &[
            "--priority-scheduler-enabled",
            "--priority-scheduler-default-max-class",
            "Interactive",
        ],
        Until::Served,
        &dir.path().join("known-class.log"),
    );
    assert!(
        launch.served,
        "the gateway did not serve its health route with a known class name (status {:?})\n{}",
        launch.status, launch.log
    );
}
