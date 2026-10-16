//! `--rdma-listen-ip` takes an IP address: a value that is not one is a
//! configuration error (the gateway exits non-zero with the field in the
//! error, and its port is never bound) where it used to start and fail only
//! when the RDMA lane came up; an address serves. The lane's switch is on in
//! both cases, as in a deployment that sets the flag.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use gateway_launch::{launch, Until, EXIT_TIMEOUT};

/// A listener address that is not an IP address is refused before anything
/// listens, with the field and the value in the error and without a panic.
#[test]
fn a_listen_ip_that_is_not_an_address_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--rdma-listen-ip", "999.1.1.1", "--mm-pixel-rdma"],
        Until::Bound,
        &dir.path().join("non-address.log"),
    );
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "the gateway bound port {:?} with a listen IP that is not an address\n{log}",
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
        log.contains("rdma_listen_ip") && log.contains("999.1.1.1"),
        "the error does not name the field and the value\n{log}"
    );
}

/// A plain address serves: the lane's exporter is built on the first export,
/// not at start-up, so the health route answers on any host.
#[test]
fn a_listen_ip_that_is_an_address_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--rdma-listen-ip", "127.0.0.1", "--mm-pixel-rdma"],
        Until::Served,
        &dir.path().join("address.log"),
    );
    assert!(
        launch.served,
        "the gateway did not serve its health route with a listen IP (exit {:?})\n{}",
        launch.status, launch.log
    );
}
