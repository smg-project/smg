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

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, log_with_line, serial, EXIT_TIMEOUT, IP};

use crate::gateway_launch;

const ADMIN_KEY: &str = "file-only-admin-key-7f3a";

/// A running gateway, killed when dropped.
struct Gateway {
    child: Child,
    log: std::path::PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args` and a bare environment, its output in
    /// `log`, and waits until its health route answers on `addr`.
    fn start(args: &[String], addr: SocketAddr, log: &Path) -> Self {
        let file = File::create(log).expect("log file");
        let child = Command::new(env!("CARGO_BIN_EXE_smg"))
            .args(args)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .stdin(Stdio::null())
            .stdout(Stdio::from(file.try_clone().expect("log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("spawn the gateway");
        let mut gateway = Self {
            child,
            log: log.to_path_buf(),
        };
        let deadline = Instant::now() + EXIT_TIMEOUT;
        loop {
            if let Some(status) = gateway.child.try_wait().expect("poll the gateway") {
                panic!(
                    "the gateway exited ({status}) before serving\n{}",
                    gateway.log()
                );
            }
            if status_of(addr, "/health", None).is_some_and(|status| status == 200) {
                return gateway;
            }
            assert!(
                Instant::now() < deadline,
                "the gateway did not serve within {EXIT_TIMEOUT:?}\n{}",
                gateway.log()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The status code `GET path` gets on `addr`, with `key` as the bearer when
/// given, over one connection that is closed with the response; `None` when
/// nothing answered.
fn status_of(addr: SocketAddr, path: &str, key: Option<&str>) -> Option<u16> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let authorization = key.map_or(String::new(), |key| {
        format!("Authorization: Bearer {key}\r\n")
    });
    stream
        .write_all(
            format!(
                "GET {path} HTTP/1.1\r\nHost: localhost\r\n{authorization}Connection: close\r\n\r\n"
            )
            .as_bytes(),
        )
        .ok()?;
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    let head = String::from_utf8_lossy(&response);
    head.strip_prefix("HTTP/1.1 ")?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
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
