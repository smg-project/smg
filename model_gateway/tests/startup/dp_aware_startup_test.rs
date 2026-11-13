//! The retired `--dp-aware` flag: data-parallel ranks are discovered at
//! every registration, so the flag is accepted for one release and ignored,
//! with one start-up line that says so; a launch without it says nothing.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

use gateway_launch::{launch, Until};

use crate::gateway_launch;

const DEPRECATION: &str = "--dp-aware is deprecated and ignored";

/// The flag serves and draws the deprecation line, which names the release
/// the flag leaves in.
#[test]
fn the_retired_dp_aware_flag_serves_with_one_deprecation_line() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--dp-aware"],
        Until::Served,
        &dir.path().join("dp-aware.log"),
    );
    let log = &launch.log;
    assert!(
        launch.served,
        "the gateway did not serve with --dp-aware (exit {:?})\n{log}",
        launch.status
    );
    assert_eq!(
        log.matches(DEPRECATION).count(),
        1,
        "one deprecation line\n{log}"
    );
    assert!(
        log.contains("removed in release 1.17"),
        "the line names the release the flag leaves in\n{log}"
    );
}

/// Without the flag nothing is said about it.
#[test]
fn a_launch_without_the_flag_says_nothing_about_it() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(&[], Until::Served, &dir.path().join("no-flag.log"));
    assert!(
        launch.served,
        "the gateway did not serve (exit {:?})\n{}",
        launch.status, launch.log
    );
    assert!(
        !launch.log.contains("dp-aware"),
        "a launch without the flag names it\n{}",
        launch.log
    );
}
