//! `--secrets-auto-reload`: a gateway started with its serving key and its
//! control-plane keys in files puts a rotated file's keys in force on its own,
//! as a secret manager rotates it (a staged file renamed over the key file),
//! without a call to the reload route. The start-up record says the files are
//! watched; a file that cannot be used leaves every key in force and is
//! logged; no key reaches the log. The gateway runs as a process on purpose:
//! the flag, the watch and the record are the binary's.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    fs,
    net::SocketAddr,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{
    base_args, free_ports, log_with_line, request, serial, serving_accepts, Gateway, EXIT_TIMEOUT,
    IP,
};

use crate::gateway_launch;

const SERVING_KEY: &str = "serving-key-one-2a7e";
const ROTATED_SERVING_KEY: &str = "serving-key-two-6d1f";
const ADMIN_KEY: &str = "admin-key-one-c90b";
const ROTATED_ADMIN_KEY: &str = "admin-key-two-e44a";

/// Rotates `file` as a secret manager does: the new content staged beside it
/// and renamed over it.
fn rotate(file: &Path, content: &str) {
    let staged = file.with_extension("staged");
    fs::write(&staged, content).unwrap();
    fs::rename(&staged, file).unwrap();
}

/// Polls `accepted` until it holds, within ten seconds.
fn within_ten_seconds(what: &str, mut accepted: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !accepted() {
        assert!(Instant::now() < deadline, "{what} did not happen in time");
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_rotated_key_file_is_in_force_without_a_call() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let serving = dir.path().join("api-key");
    let admins = dir.path().join("control-plane-keys");
    fs::write(&serving, format!("{SERVING_KEY}\n")).unwrap();
    fs::write(&admins, format!("ops:Operations:admin:{ADMIN_KEY}\n")).unwrap();
    let [port, metrics_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(
        [
            "--api-key-file",
            serving.to_str().unwrap(),
            "--control-plane-api-keys-file",
            admins.to_str().unwrap(),
            "--secrets-auto-reload",
        ]
        .map(str::to_string),
    );
    let addr = SocketAddr::from((IP, port));
    let gateway = Gateway::start(&args, addr, &dir.path().join("gateway.log"));

    // The record says the two files are watched.
    if let Err(log) = log_with_line(
        &gateway.log,
        "secrets_auto_reload=\"on (2 files)\"",
        EXIT_TIMEOUT,
    ) {
        panic!("the CREDENTIAL SOURCES record does not say the files are watched:\n{log}");
    }
    assert!(serving_accepts(addr, Some(SERVING_KEY)));
    assert!(!serving_accepts(addr, Some(ROTATED_SERVING_KEY)));

    // The serving key rotated in its file: in force without a call.
    rotate(&serving, &format!("{ROTATED_SERVING_KEY}\n"));
    within_ten_seconds("the serving key's reload", || {
        serving_accepts(addr, Some(ROTATED_SERVING_KEY))
    });
    assert!(
        !serving_accepts(addr, Some(SERVING_KEY)),
        "the old key is out"
    );
    // The key is in force at the swap; the line follows it.
    let log = match log_with_line(
        &gateway.log,
        "the keys were reloaded from their files",
        EXIT_TIMEOUT,
    ) {
        Ok(log) => log,
        Err(log) => panic!("the watch's reload is not logged:\n{log}"),
    };
    assert!(
        log.contains("trigger=\"watch\""),
        "the reload is not logged as the watch's:\n{log}"
    );

    // A control-plane file that cannot be used (an unknown role): logged,
    // every key in force standing; fixed, the new key is in force.
    rotate(
        &admins,
        &format!("ops:Operations:owner:{ROTATED_ADMIN_KEY}\n"),
    );
    if let Err(log) = log_with_line(&gateway.log, "the keys were not reloaded", EXIT_TIMEOUT) {
        panic!("the bad file is not logged:\n{log}");
    }
    assert_eq!(
        request(addr, "GET", "/workers", Some(ADMIN_KEY), None).map(|(status, _)| status),
        Some(200),
        "the admin key stands"
    );
    rotate(
        &admins,
        &format!("ops:Operations:admin:{ROTATED_ADMIN_KEY}\n"),
    );
    within_ten_seconds("the control-plane key's reload", || {
        request(addr, "GET", "/workers", Some(ROTATED_ADMIN_KEY), None).map(|(status, _)| status)
            == Some(200)
    });
    assert_eq!(
        request(addr, "GET", "/workers", Some(ADMIN_KEY), None).map(|(status, _)| status),
        Some(401),
        "the old admin key is out"
    );

    let log = gateway.log();
    for key in [
        SERVING_KEY,
        ROTATED_SERVING_KEY,
        ADMIN_KEY,
        ROTATED_ADMIN_KEY,
    ] {
        assert!(!log.contains(key), "a key reached the log:\n{log}");
    }
}
