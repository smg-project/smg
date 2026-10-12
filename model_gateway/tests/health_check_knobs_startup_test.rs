//! The gateway refuses to start on a health-check knob set to 0 (the probe
//! interval, the probe timeout, the failure and the success threshold): the
//! configuration error names the field, the process exits non-zero without
//! a panic, and its port is never bound. A zero interval used to start the
//! gateway with a busy probe loop. The gateway runs as a process on purpose:
//! the check belongs before the runtime and the listener exist.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

use std::{
    fs::{self, File},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const IP: Ipv4Addr = Ipv4Addr::LOCALHOST;
/// How long the gateway gets to refuse or to serve; one that serves is
/// killed as soon as its port answers, one that hangs at this deadline.
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// What became of a gateway launched with the configuration under test.
struct Launch {
    /// The exit status; `None` when the gateway had to be killed.
    status: Option<ExitStatus>,
    /// Whether its port accepted a connection at any point.
    listened: bool,
    /// Its stdout and stderr.
    log: String,
}

/// Two distinct free ports: bound at once so the kernel hands out different
/// ones, then released for the gateway to bind moments later.
fn free_ports() -> [u16; 2] {
    let listeners = [
        TcpListener::bind((IP, 0)).expect("a free port"),
        TcpListener::bind((IP, 0)).expect("a free port"),
    ];
    listeners.map(|listener| listener.local_addr().expect("the bound address").port())
}

/// Runs the gateway with `args` on fresh ports, its output in `log_path`,
/// until it exits, binds its port or `EXIT_TIMEOUT` passes.
fn launch(args: &[&str], log_path: &Path) -> Launch {
    let [port, metrics_port] = free_ports();
    let ip = IP.to_string();
    let log = File::create(log_path).expect("log file");
    let mut child = Command::new(env!("CARGO_BIN_EXE_smg"))
        .args(["--enable-igw", "--host", &ip, "--port", &port.to_string()])
        .args(["--prometheus-host", &ip])
        .args(["--prometheus-port", &metrics_port.to_string()])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().expect("log handle")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn the gateway");
    let deadline = Instant::now() + EXIT_TIMEOUT;
    let address = SocketAddr::from((IP, port));
    let mut listened = false;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the gateway") {
            break Some(status);
        }
        listened = TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok();
        if listened || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    Launch {
        status,
        listened,
        log: fs::read_to_string(log_path).unwrap_or_default(),
    }
}

/// Each knob at 0 is refused before anything listens, with the field in the
/// error and no panic.
#[test]
fn a_zero_health_check_knob_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let knobs = [
        ("--health-check-interval-secs", "check_interval_secs"),
        ("--health-check-timeout-secs", "timeout_secs"),
        ("--health-failure-threshold", "failure_threshold"),
        ("--health-success-threshold", "success_threshold"),
    ];
    for (flag, field) in knobs {
        let launch = launch(
            &[flag, "0"],
            &dir.path()
                .join(format!("{}.log", flag.trim_start_matches('-'))),
        );
        let log = &launch.log;
        assert!(
            !launch.listened,
            "{flag} 0: the gateway served with a health-check knob at 0\n{log}"
        );
        let status = launch.status.unwrap_or_else(|| {
            panic!(
                "{flag} 0: the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}"
            )
        });
        assert!(
            !status.success(),
            "{flag} 0: the gateway exited successfully\n{log}"
        );
        assert!(
            !log.contains("panicked"),
            "{flag} 0: a panic is not a configuration error\n{log}"
        );
        assert!(
            log.contains(field) && log.contains(flag),
            "{flag} 0: the error names neither the field nor the flag\n{log}"
        );
    }
}

/// The smallest interval is a configuration like any other: the gateway
/// serves.
#[test]
fn a_one_second_health_check_interval_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--health-check-interval-secs", "1"],
        &dir.path().join("interval-1.log"),
    );
    assert!(
        launch.listened,
        "the gateway did not serve with a one-second probe interval (status {:?})\n{}",
        launch.status, launch.log
    );
}
