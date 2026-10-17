//! `--admin-port` gives the control plane a listener of its own: the admin
//! and worker-registry routes answer there, the data listener answers 404
//! for them, the serving routes stay off the admin listener, and the probe
//! routes answer on both; `--admin-host` without `--admin-port` is a
//! configuration error named before anything binds. The gateway runs as a
//! process on purpose: the listeners are the binary's.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, launch, serial, Until, EXIT_TIMEOUT, IP};

/// A running gateway, killed when dropped.
struct Gateway {
    child: Child,
    log: PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args`, its output in `log`, and waits until
    /// its health route answers on `addr`.
    fn start(args: &[String], addr: SocketAddr, log: &Path) -> Self {
        let file = File::create(log).expect("log file");
        let child = Command::new(env!("CARGO_BIN_EXE_smg"))
            .args(args)
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
            if status_of(addr, "/health").is_some_and(|status| status == 200) {
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

/// The status code `GET path` gets on `addr`, over one connection that is
/// closed with the response; `None` when nothing answered.
fn status_of(addr: SocketAddr, path: &str) -> Option<u16> {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
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
fn the_admin_listener_carries_the_control_plane_and_the_data_listener_answers_404() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let [port, metrics_port, admin_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(["--admin-port", &admin_port.to_string()].map(str::to_string));
    let data = SocketAddr::from((IP, port));
    let admin = SocketAddr::from((IP, admin_port));
    let gateway = Gateway::start(&args, data, &dir.path().join("gateway.log"));

    for path in ["/workers", "/v1/tokenizers"] {
        assert_eq!(
            status_of(admin, path),
            Some(200),
            "{path} on the admin listener"
        );
        assert_eq!(
            status_of(data, path),
            Some(404),
            "{path} on the data listener"
        );
    }
    // The serving routes stay on the data listener (503 without a worker,
    // the route's own answer, never the fallback's 404) and off the admin one.
    for path in ["/v1/models", "/get_server_info"] {
        let on_data = status_of(data, path);
        assert!(
            on_data.is_some() && on_data != Some(404),
            "{path} on the data listener: {on_data:?}"
        );
        assert_eq!(
            status_of(admin, path),
            Some(404),
            "{path} on the admin listener"
        );
    }
    for path in ["/health", "/liveness"] {
        assert_eq!(
            status_of(data, path),
            Some(200),
            "{path} on the data listener"
        );
        assert_eq!(
            status_of(admin, path),
            Some(200),
            "{path} on the admin listener"
        );
    }

    let log = gateway.log();
    assert!(
        log.contains(&format!(
            "Admin listener serving the control plane on {admin} (--admin-port)"
        )),
        "the start does not report the admin listener\n{log}"
    );
    drop(gateway);
}

/// `--admin-host` without `--admin-port` is refused before anything binds,
/// naming the field and the missing flag.
#[test]
fn an_admin_host_without_an_admin_port_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--admin-host", &IP.to_string()],
        Until::Bound,
        &dir.path().join("admin-host-alone.log"),
    );
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "the gateway bound port {:?} with an admin host and no admin port\n{log}",
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
        log.contains("admin_host") && log.contains("--admin-port"),
        "the error names neither the field nor the missing flag\n{log}"
    );
}
