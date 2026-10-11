//! The gateway refuses to start on a priority-scheduler configuration file
//! it cannot use: the configuration error names the field and the file, the
//! process exits non-zero without a panic, and its port is never bound. The
//! gateway runs as a process on purpose: the check belongs before the
//! runtime and the listener exist.
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
/// How long the gateway gets to refuse; one that serves instead is killed as
/// soon as its port answers, one that hangs at this deadline.
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// What became of a gateway launched with a configuration it should refuse.
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
            &dir.path().join(format!("{name}.log")),
        );
        let log = &launch.log;
        assert!(
            !launch.listened,
            "{name}: the gateway served with a scheduler config it cannot use\n{log}"
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
