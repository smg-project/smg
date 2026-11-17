//! The example controller started as a user starts it: the built binary
//! names the stages it holds, binds its listeners, elects itself and reads
//! ready on its admin listener.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const BIN: &str = env!("CARGO_BIN_EXE_policy-controller-example");

/// How long a start, a line of its log or an answer may take.
const BOUND: Duration = Duration::from_secs(20);

/// The controller under test, killed when the case ends, whatever its
/// verdict.
struct Started {
    child: Child,
    /// Where its admin listener is bound.
    admin: String,
    /// The line of its log that names the stages it holds.
    stages: String,
}

impl Drop for Started {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts the binary on loopback at ports the system picks and reads the
/// admin listener's address from its log, or says what the start lacked.
fn start() -> Result<Started, String> {
    let mut child = Command::new(BIN)
        .args(["--controller-id", "example-0", "--log-level", "info"])
        .args(["--gateway-listen", "127.0.0.1:0"])
        .args(["--control-listen", "127.0.0.1:0"])
        .args(["--replication-listen", "127.0.0.1:0"])
        .args(["--admin-listen", "127.0.0.1:0"])
        .env_remove("RUST_LOG")
        .env_remove("FORCE_COLOR")
        .env_remove("CLICOLOR_FORCE")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("the example binary did not start: {error}"))?;
    let log = child.stdout.take();
    // The log is read to its end on a thread of its own, so the controller
    // never waits on a full pipe.
    let mut started = Started {
        child,
        admin: String::new(),
        stages: String::new(),
    };
    let log = log.ok_or("the binary's log is not piped")?;
    let (lines, read) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(log).lines().map_while(Result::ok) {
            let _ = lines.send(line);
        }
    });
    let deadline = Instant::now() + BOUND;
    started.admin = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let line = read
            .recv_timeout(left)
            .map_err(|_| "no admin listener line within the bound")?;
        if line.contains("stages held") {
            started.stages = line;
        } else if line.contains("admin listener bound") {
            break line
                .split_whitespace()
                .find_map(|field| field.strip_prefix("addr="))
                .ok_or_else(|| format!("the line names no address: {line}"))?
                .to_owned();
        }
    };
    Ok(started)
}

/// One `GET` on the admin listener: the status and the body.
fn get(admin: &str, path: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(admin)?;
    stream.set_read_timeout(Some(BOUND))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {admin}\r\nConnection: close\r\n\r\n"
    )?;
    let mut answer = String::new();
    stream.read_to_string(&mut answer)?;
    let status = answer
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = answer
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_owned())
        .unwrap_or_default();
    Ok((status, body))
}

/// The body of the first answer of `path` with the status 200, within the
/// bound.
fn ok_body(admin: &str, path: &str) -> Result<String, String> {
    let deadline = Instant::now() + BOUND;
    loop {
        match get(admin, path) {
            Ok((200, body)) => return Ok(body),
            answer if Instant::now() >= deadline => {
                return Err(format!("{path} did not answer 200: {answer:?}"));
            }
            _ => thread::sleep(Duration::from_millis(20)),
        }
    }
}

#[test]
fn the_built_binary_binds_leads_and_reads_ready() {
    let started = start().expect("the example controller starts");
    let ready = ok_body(&started.admin, "/readiness").expect("it reads ready");
    assert!(ready.contains(r#""status":"ready""#), "{ready}");
    assert!(ready.contains(r#""role":"leader""#), "{ready}");
    // The contract it speaks is the crate's.
    let version = ok_body(&started.admin, "/version").expect("it answers its version");
    assert!(
        version.contains(r#""protocol":"smg.policy.v1""#),
        "{version}"
    );
}

#[test]
fn the_built_binary_holds_the_stages_of_this_build() {
    let started = start().expect("the example controller starts");
    // The stock build's line names no `pack`.
    assert!(started.stages.contains(r#""pack""#), "{}", started.stages);
}

/// The example runs on the allocator the stock controller's binary
/// registers, with its settings. The allocator is asked for its statistics
/// at the process's end, an ask the system allocator does not hear, and the
/// statistics name the settings it runs on, and on Linux aarch64 the page
/// size it was built for.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[test]
fn the_built_binary_runs_on_the_stock_binarys_allocator() {
    let output = Command::new(BIN)
        .arg("--version")
        .env("_RJEM_MALLOC_CONF", "stats_print:true")
        .stdin(Stdio::null())
        .output()
        .expect("the example binary runs");
    let written = String::from_utf8_lossy(&output.stderr);
    assert!(
        written.contains("jemalloc statistics"),
        "no statistics of the allocator at the end: {written:?}"
    );
    for setting in [
        "opt.background_thread: true",
        "opt.dirty_decay_ms: 10000 (",
        "opt.muzzy_decay_ms: 0 (",
        "opt.narenas: 16\n",
        "opt.oversize_threshold: 33554432\n",
        // The cargo configuration builds the allocator of an aarch64 binary
        // for 64 KiB pages: the workspace's here, and the line
        // `own-repository.sh` writes in a repository of your own.
        #[cfg(all(target_arch = "aarch64", target_env = "gnu"))]
        "Page size: 65536\n",
    ] {
        assert!(
            written.contains(setting),
            "the allocator does not run on {setting:?}: {written:?}"
        );
    }
}
