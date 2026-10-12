//! The gateway refuses to start on a tokenizer, model or chat-template path
//! it cannot read (a path that does not exist, a directory without a
//! tokenizer file, an empty template): the configuration error names the
//! field and the path, the process exits non-zero without a panic, and its
//! port is never bound. They used to be one ERROR line and a gateway that
//! served without the configured tokenizer. A hub id and a usable local
//! layout still start. The gateway runs as a process on purpose: the check
//! belongs before the runtime and the listener exist.
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
/// until it exits, binds its port or `EXIT_TIMEOUT` passes. No network: a
/// hub id is never fetched by this test.
fn launch(args: &[&str], log_path: &Path) -> Launch {
    let [port, metrics_port] = free_ports();
    let ip = IP.to_string();
    let log = File::create(log_path).expect("log file");
    let mut child = Command::new(env!("CARGO_BIN_EXE_smg"))
        .args(["--enable-igw", "--host", &ip, "--port", &port.to_string()])
        .args(["--prometheus-host", &ip])
        .args(["--prometheus-port", &metrics_port.to_string()])
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
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

/// A directory with the one file the tokenizer loader looks for first.
fn tokenizer_dir(dir: &Path) -> String {
    let tok = dir.join("tok");
    fs::create_dir(&tok).expect("the tokenizer directory");
    fs::write(tok.join("tokenizer.json"), "{}").expect("the tokenizer file");
    tok.to_str().expect("a UTF-8 path").to_string()
}

/// A missing path, a directory without a tokenizer file, a missing and an
/// empty chat template: each refused before anything listens, with the
/// field and the path in the error, no panic.
#[test]
fn an_unreadable_tokenizer_source_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let tok = tokenizer_dir(dir.path());
    let missing = dir.path().join("missing");
    let empty_dir = dir.path().join("empty-dir");
    fs::create_dir(&empty_dir).expect("the empty directory");
    let missing_template = dir.path().join("missing.jinja");
    let empty_template = dir.path().join("empty.jinja");
    fs::write(&empty_template, "").expect("the empty template");
    let missing = missing.to_str().expect("a UTF-8 path");
    let empty_dir = empty_dir.to_str().expect("a UTF-8 path");
    let missing_template = missing_template.to_str().expect("a UTF-8 path");
    let empty_template = empty_template.to_str().expect("a UTF-8 path");

    let cases: [(&str, Vec<&str>, &str, &str); 5] = [
        (
            "tokenizer-missing",
            vec!["--tokenizer-path", missing],
            "tokenizer_path",
            missing,
        ),
        (
            "tokenizer-empty-dir",
            vec!["--tokenizer-path", empty_dir],
            "tokenizer_path",
            empty_dir,
        ),
        (
            "model-missing",
            vec!["--model-path", missing],
            "model_path",
            missing,
        ),
        (
            "chat-template-missing",
            vec![
                "--tokenizer-path",
                &tok,
                "--chat-template",
                missing_template,
            ],
            "chat_template",
            missing_template,
        ),
        (
            "chat-template-empty",
            vec!["--tokenizer-path", &tok, "--chat-template", empty_template],
            "chat_template",
            empty_template,
        ),
    ];
    for (name, args, field, path) in cases {
        let launch = launch(&args, &dir.path().join(format!("{name}.log")));
        let log = &launch.log;
        assert!(
            !launch.listened,
            "{name}: the gateway served with a tokenizer source it cannot read\n{log}"
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
            log.contains(field) && log.contains(path),
            "{name}: the error names neither the field nor the path\n{log}"
        );
    }
}

/// A usable local layout with a template, a hub id (fetched lazily, not a
/// path), and, in gRPC connection mode, a model path that is not on this
/// host (the tokenizer comes from the first worker that registers): the
/// gateway serves.
#[test]
fn a_present_tokenizer_source_or_a_hub_id_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let tok = tokenizer_dir(dir.path());
    let template = dir.path().join("template.jinja");
    fs::write(&template, "{{ messages }}").expect("the template");
    let template = template.to_str().expect("a UTF-8 path");
    let on_the_worker = dir.path().join("models").join("on-the-worker");
    let on_the_worker = on_the_worker.to_str().expect("a UTF-8 path");
    let cases: [(&str, Vec<&str>); 3] = [
        (
            "local-layout",
            vec!["--tokenizer-path", &tok, "--chat-template", template],
        ),
        ("hub-id", vec!["--tokenizer-path", "some-org/some-model"]),
        // The connection mode follows the worker URLs; a gRPC worker that is
        // not there yet is the shape of a deployment whose workers register
        // later and bring the tokenizer with them.
        (
            "grpc-model-on-the-worker",
            vec![
                "--worker-urls",
                "grpc://127.0.0.1:1",
                "--model-path",
                on_the_worker,
            ],
        ),
    ];
    for (name, args) in cases {
        let launch = launch(&args, &dir.path().join(format!("{name}.log")));
        assert!(
            launch.listened,
            "{name}: the gateway did not serve (status {:?})\n{}",
            launch.status, launch.log
        );
    }
}
