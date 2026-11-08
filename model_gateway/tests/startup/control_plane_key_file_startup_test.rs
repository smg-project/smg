//! `--control-plane-api-keys-file`: a gateway started with the control-plane
//! keys in a file alone (a mounted secret's shape) authenticates the file's
//! admin key on the control plane, names the file as the keys' source in its
//! CREDENTIAL SOURCES record, and carries the key in neither its arguments
//! nor its environment. The gateway runs as a process on purpose: the
//! record and the listener are the binary's.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{fs, net::SocketAddr};

use gateway_launch::{
    base_args, free_ports, log_with_line, request, serial, Gateway, EXIT_TIMEOUT, IP,
};

use crate::gateway_launch;

const ADMIN_KEY: &str = "file-only-admin-key-7f3a";

/// The status code `GET path` gets on `addr`, with `key` as the bearer when
/// given; `None` when nothing answered.
fn status_of(addr: SocketAddr, path: &str, key: Option<&str>) -> Option<u16> {
    request(addr, "GET", path, key, None).map(|(status, _)| status)
}

#[test]
fn the_control_plane_keys_are_read_from_a_file_and_the_record_names_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let keys = dir.path().join("control-plane-keys");
    fs::write(
        &keys,
        format!("# the operators\n\nops:Operations:admin:{ADMIN_KEY}\n"),
    )
    .unwrap();
    let [port, metrics_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(["--control-plane-api-keys-file", keys.to_str().unwrap()].map(str::to_string));
    assert!(
        args.iter().all(|arg| !arg.contains(ADMIN_KEY)),
        "the key is in a file, never in the arguments"
    );
    let addr = SocketAddr::from((IP, port));
    let gateway = Gateway::start(&args, addr, &dir.path().join("gateway.log"));

    // The file's key opens the control plane; no key and an unknown key do not.
    assert_eq!(status_of(addr, "/workers", Some(ADMIN_KEY)), Some(200));
    assert_eq!(status_of(addr, "/workers", None), Some(401));
    assert_eq!(status_of(addr, "/workers", Some("not-the-key")), Some(401));

    // The record names the file as the source, with the entry count, and the
    // key itself reaches neither the record nor any other line.
    let line = "control_plane_api_keys=\"file (1 entries)\"";
    if let Err(log) = log_with_line(&gateway.log, line, EXIT_TIMEOUT) {
        panic!("the CREDENTIAL SOURCES record does not name the file:\n{log}");
    }
    let log = gateway.log();
    assert!(log.contains("CREDENTIAL SOURCES"), "{log}");
    assert!(!log.contains(ADMIN_KEY), "the key reached the log:\n{log}");
    // Keyed control plane: the posture record does not call it open.
    assert!(
        !log.contains("control_plane=\"OPEN"),
        "the control plane is keyed, not open:\n{log}"
    );
}
