//! The gateway reads `--policy-spec` at start: a document the engine
//! accepts starts it, a document the engine refuses stops the start naming
//! the field it refused, and a document under the legacy decider is refused
//! by name, as is a document beside `--enable-igw`; no refusal binds a port.
//! The gateway runs as a process on
//! purpose: the checks belong before the runtime and the listener exist.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{fs, path::Path};

use gateway_launch::{
    base_args, free_ports, launch, launch_with, serial, Launch, Until, EXIT_TIMEOUT,
};
use smg_policy_controller::spec::PolicySpec;

use crate::gateway_launch;

/// A launch that must end in a refusal: no port answered, a non-zero exit,
/// no panic, and every text of `named` in the error.
fn assert_refused(name: &str, launch: &Launch, named: &[&str]) {
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "{name}: port {:?} answered although the configuration is unusable\n{log}",
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
    for text in named {
        assert!(
            log.contains(text),
            "{name}: the error does not name {text:?}\n{log}"
        );
    }
}

/// A launch of the common flags without `--enable-igw` (its routers pair, so
/// a document is refused beside it on this base) plus `args`, until a bind.
/// It picks its own ports, so it holds the harness's serial guard from the
/// pick to the verdict, as [`launch`] does.
fn launch_regular(args: &[&str], log_path: &Path) -> Launch {
    let _serial = serial();
    let [port, metrics_port] = free_ports::<2>();
    let mut all = base_args(port, metrics_port);
    all.retain(|flag| flag != "--enable-igw");
    all.extend(args.iter().map(|flag| (*flag).to_string()));
    launch_with(&all, &[port, metrics_port], Until::Bound, log_path)
}

/// The round-robin preset as a document, written to `path`.
fn write_preset(path: &Path) {
    let spec = PolicySpec::preset("round_robin").expect("a preset document");
    fs::write(
        path,
        serde_yaml::to_string(&spec).expect("the document as YAML"),
    )
    .unwrap();
}

/// A document the engine accepts, under the engine: the gateway starts,
/// says so in its start-up lines and binds its port.
#[test]
fn a_policy_spec_the_engine_accepts_starts_the_gateway() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let spec = dir.path().join("spec.yaml");
    write_preset(&spec);
    let launch = launch_regular(
        &[
            "--policy-engine",
            "engine",
            "--policy-spec",
            spec.to_str().unwrap(),
        ],
        &dir.path().join("accepted.log"),
    );
    assert!(
        launch.bound.is_some(),
        "the gateway did not bind under an accepted document\n{}",
        launch.log
    );
    assert!(
        launch.log.contains("policy document applied"),
        "the start does not say the document was applied\n{}",
        launch.log
    );
}

/// A document the engine refuses stops the start, the error naming the flag
/// and the field the engine refused.
#[test]
fn a_policy_spec_the_engine_refuses_stops_the_start_naming_the_field() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let spec = dir.path().join("spec.yaml");
    fs::write(
        &spec,
        "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: []\n    score: [nope]\n    pick: best\n",
    )
    .unwrap();
    let launch = launch_regular(
        &[
            "--policy-engine",
            "engine",
            "--policy-spec",
            spec.to_str().unwrap(),
        ],
        &dir.path().join("refused.log"),
    );
    // The error leaves main in its Debug form, the quotes of the field escaped.
    assert_refused(
        "an unknown scorer",
        &launch,
        &["--policy-spec", "profile", "unknown scorer", "nope"],
    );
}

/// A document under the legacy decider (the default) is refused by name
/// before the file is read: the legacy policies read no specification.
#[test]
fn a_policy_spec_under_the_legacy_decider_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let spec = dir.path().join("spec.yaml");
    write_preset(&spec);
    let launch = launch_regular(
        &["--policy-spec", spec.to_str().unwrap()],
        &dir.path().join("legacy.log"),
    );
    assert_refused(
        "the legacy decider",
        &launch,
        &["policy_spec", "--policy-engine engine"],
    );
}

/// A document beside `--enable-igw` (the common flags' launch) is refused by
/// name before the file is read: the IGW routers' pair sides read the flags
/// on this base.
#[test]
fn a_policy_spec_beside_igw_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let spec = dir.path().join("spec.yaml");
    write_preset(&spec);
    let launch = launch(
        &[
            "--policy-engine",
            "engine",
            "--policy-spec",
            spec.to_str().unwrap(),
        ],
        Until::Bound,
        &dir.path().join("igw.log"),
    );
    assert_refused(
        "the IGW routers",
        &launch,
        &["policy_spec", "--enable-igw", "pair decider"],
    );
}
