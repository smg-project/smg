//! The gateway refuses to start on a `--log-dir` it cannot use as a log
//! directory (an existing file, a path under a file): the configuration
//! error names the field and the path, the process exits non-zero without a
//! panic, and its port is never bound. A missing directory is still created.
//! The gateway runs as a process on purpose: the check belongs before the
//! runtime, the log appender and the listener exist.
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

/// An existing file and a path under a file: both are refused before
/// anything listens, with the field and the path in the error, no panic.
#[test]
fn an_unusable_log_dir_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let file = dir.path().join("a-file");
    fs::write(&file, "not a directory").expect("write the file");
    let under_a_file = file.join("logs");
    for (name, path) in [("file", &file), ("under-a-file", &under_a_file)] {
        let path = path.to_str().expect("a UTF-8 path");
        let launch = launch(
            &["--log-dir", path],
            &dir.path().join(format!("{name}.log")),
        );
        let log = &launch.log;
        assert!(
            !launch.listened,
            "{name}: the gateway served with a log directory it cannot use\n{log}"
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
            log.contains("log_dir") && log.contains(path),
            "{name}: the error names neither the field nor the path\n{log}"
        );
    }
    assert!(!under_a_file.exists(), "nothing was created under the file");
}

/// A directory that does not exist yet is created and written to, as it
/// always was: the gateway serves.
#[test]
fn a_missing_log_dir_is_created_and_the_gateway_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let logs = dir.path().join("created").join("logs");
    let launch = launch(
        &["--log-dir", logs.to_str().expect("a UTF-8 path")],
        &dir.path().join("created.log"),
    );
    let log = &launch.log;
    assert!(
        launch.listened,
        "the gateway did not serve with a log directory it can create (status {:?})\n{log}",
        launch.status
    );
    let log_files: Vec<_> = fs::read_dir(&logs)
        .expect("the created log directory")
        .map(|entry| entry.expect("a directory entry").file_name())
        .filter(|name| name.to_string_lossy().starts_with("smg."))
        .collect();
    assert!(
        !log_files.is_empty(),
        "no log file in the created directory\n{log}"
    );
}
