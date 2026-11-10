//! `POST /secrets/reload`: a gateway started with its serving key and its
//! control-plane keys in files (a mounted secret's shape) puts a rotated
//! file's keys in force on the route, without a restart. The old serving
//! key answers 401 and the new one 200 once the reload ran and not before;
//! a file that cannot be used answers 422 and leaves every key in force; a
//! rotated control-plane key opens the control plane after the reload. The
//! gateway runs as a process on purpose: the command line resolves the file
//! sources, and the audit line is the binary's.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{fs, net::SocketAddr};

use gateway_launch::{
    base_args, free_ports, log_with_line, request, serial, serving_accepts, Gateway, EXIT_TIMEOUT,
    IP,
};
use smg::middleware::key_fingerprint;

use crate::gateway_launch;

const SERVING_KEY: &str = "serving-key-one-4c1d";
const ROTATED_SERVING_KEY: &str = "serving-key-two-9e2b";
const ADMIN_KEY: &str = "admin-key-one-7f3a";
const ROTATED_ADMIN_KEY: &str = "admin-key-two-0b5c";

/// The status `method path` gets on `addr` with `key` as the bearer.
fn status(addr: SocketAddr, method: &str, path: &str, key: Option<&str>) -> Option<u16> {
    request(addr, method, path, key, None).map(|(status, _)| status)
}

/// The status and body of a reload under `key`.
fn reload(addr: SocketAddr, key: &str) -> (u16, String) {
    request(addr, "POST", "/secrets/reload", Some(key), None).expect("the gateway answers")
}

#[test]
fn a_reload_puts_the_rotated_files_keys_in_force_without_a_restart() {
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
        ]
        .map(str::to_string),
    );
    let addr = SocketAddr::from((IP, port));
    let gateway = Gateway::start(&args, addr, &dir.path().join("gateway.log"));

    // The start's keys: the serving key on a serving route, the admin key on
    // the control plane; the serving key is no control-plane credential.
    assert!(serving_accepts(addr, Some(SERVING_KEY)));
    assert!(!serving_accepts(addr, None));
    assert_eq!(status(addr, "GET", "/workers", Some(ADMIN_KEY)), Some(200));
    assert_eq!(
        status(addr, "POST", "/secrets/reload", Some(SERVING_KEY)),
        Some(401)
    );
    assert_eq!(status(addr, "POST", "/secrets/reload", None), Some(401));

    // Unchanged files: a reload reads both and changes nothing.
    let (code, body) = reload(addr, ADMIN_KEY);
    assert_eq!(code, 200, "{body}");
    assert!(
        body.contains("\"setting\":\"api-key-file\"")
            && body.contains("\"setting\":\"control-plane-api-keys-file\""),
        "{body}"
    );
    assert!(
        body.contains("\"serving_changed\":false")
            && body.contains("\"control_plane_changed\":false"),
        "{body}"
    );

    // The serving key rotated in its file: in force after the reload, not before.
    fs::write(&serving, format!("{ROTATED_SERVING_KEY}\n")).unwrap();
    assert!(
        !serving_accepts(addr, Some(ROTATED_SERVING_KEY)),
        "a file edit alone changes nothing"
    );
    let (code, body) = reload(addr, ADMIN_KEY);
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("\"serving_changed\":true"), "{body}");
    assert!(serving_accepts(addr, Some(ROTATED_SERVING_KEY)));
    assert!(
        !serving_accepts(addr, Some(SERVING_KEY)),
        "the old key is out"
    );
    // The configuration view names the key in force, the rotated one.
    let (code, view) =
        request(addr, "GET", "/config", Some(ADMIN_KEY), None).expect("the gateway answers");
    assert_eq!(code, 200, "{view}");
    assert!(
        view.contains(&key_fingerprint(ROTATED_SERVING_KEY)),
        "the view names the rotated key: {view}"
    );
    assert!(
        !view.contains(&key_fingerprint(SERVING_KEY)),
        "the start's key is gone from the view: {view}"
    );

    // A file that cannot be used (an unknown role): 422 naming the field,
    // never the content, and every key in force stands.
    fs::write(
        &admins,
        format!("ops:Operations:owner:{ROTATED_ADMIN_KEY}\n"),
    )
    .unwrap();
    let (code, body) = reload(addr, ADMIN_KEY);
    assert_eq!(code, 422, "{body}");
    assert!(body.contains("control-plane-api-keys"), "{body}");
    assert!(
        !body.contains(ROTATED_ADMIN_KEY),
        "the content reached the answer: {body}"
    );
    assert_eq!(
        status(addr, "GET", "/workers", Some(ADMIN_KEY)),
        Some(200),
        "the admin key stands"
    );
    assert!(serving_accepts(addr, Some(ROTATED_SERVING_KEY)));

    // The control-plane key rotated: the reload under the old key puts the
    // new one in force, and the old one is out.
    fs::write(
        &admins,
        format!("ops:Operations:admin:{ROTATED_ADMIN_KEY}\n"),
    )
    .unwrap();
    let (code, body) = reload(addr, ADMIN_KEY);
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("\"control_plane_changed\":true"), "{body}");
    assert_eq!(
        status(addr, "GET", "/workers", Some(ADMIN_KEY)),
        Some(401),
        "the old admin key is out"
    );
    assert_eq!(
        status(addr, "GET", "/workers", Some(ROTATED_ADMIN_KEY)),
        Some(200)
    );

    // The audit line records the mutation with its resource; no key reaches
    // the log, in force or rotated out.
    if let Err(log) = log_with_line(&gateway.log, "control_plane_mutation", EXIT_TIMEOUT) {
        panic!("no audit line for the reload:\n{log}");
    }
    let log = gateway.log();
    assert!(
        log.contains("resource=\"secrets\"") || log.contains("resource=secrets"),
        "the audit line does not name the resource:\n{log}"
    );
    for key in [
        SERVING_KEY,
        ROTATED_SERVING_KEY,
        ADMIN_KEY,
        ROTATED_ADMIN_KEY,
    ] {
        assert!(!log.contains(key), "a key reached the log:\n{log}");
    }
}
