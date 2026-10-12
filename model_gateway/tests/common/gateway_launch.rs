//! The launcher the gateway's start-up tests share: the built binary on
//! fresh loopback ports, run until it exits, a watched port is bound or its
//! health route answers, within a deadline; what became of it is read from
//! [`Launch`]. Included by path from the start-up tests, which need nothing
//! else of this directory, and as `common::gateway_launch` elsewhere.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Command, ExitStatus, Stdio},
    sync::{Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant},
};

/// The loopback address every listener of a launched gateway binds.
pub const IP: Ipv4Addr = Ipv4Addr::LOCALHOST;
/// How long a gateway gets to refuse or to serve; one that does neither is
/// killed at this deadline.
pub const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// What ends a launch's polling, short of the gateway's exit or the
/// deadline; the gateway is killed at that point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Until {
    /// A watched port accepts a connection: the signal a refusal case must
    /// never see, so the gateway is killed as soon as it shows.
    Bound,
    /// The gateway answers `GET /health` with 200 on the first watched port:
    /// the proof that the gateway itself serves there, where a bare connect
    /// would also pass for another process that took the port in the gap
    /// after [`free_ports`] released it.
    Served,
}

/// What became of a launched gateway.
pub struct Launch {
    /// The exit status; `None` when the gateway had to be killed.
    pub status: Option<ExitStatus>,
    /// The first watched port that accepted a connection, if any did:
    /// latched on a bare connect, so a bind that precedes an exit is not
    /// missed.
    pub bound: Option<u16>,
    /// Whether the gateway answered `GET /health` with 200 on the first
    /// watched port (read only until [`Until::Served`]).
    pub served: bool,
    /// Its stdout and stderr.
    pub log: String,
}

/// The launches of one test binary run one at a time: a port a test picked
/// with [`free_ports`] and released for its gateway to bind must not be taken
/// meanwhile by another test's gateway, or by a listener it holds, which a
/// poll of that port would read as the first gateway's bind. [`launch`] holds
/// the guard; a test that picks its own ports and runs [`launch_with`] holds
/// it itself, from the first to the last.
pub fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `N` distinct free ports: bound at once so the kernel hands out different
/// ones, then released for the gateway to bind moments later.
pub fn free_ports<const N: usize>() -> [u16; N] {
    let listeners: [TcpListener; N] =
        std::array::from_fn(|_| TcpListener::bind((IP, 0)).expect("a free port"));
    listeners.map(|listener| listener.local_addr().expect("the bound address").port())
}

/// The common flags: the HTTP listener on `port`, the metrics listener on
/// `metrics_port`, both on the loopback.
pub fn base_args(port: u16, metrics_port: u16) -> Vec<String> {
    let ip = IP.to_string();
    [
        "--enable-igw",
        "--host",
        &ip,
        "--port",
        &port.to_string(),
        "--prometheus-host",
        &ip,
        "--prometheus-port",
        &metrics_port.to_string(),
    ]
    .map(str::to_string)
    .to_vec()
}

/// Runs the gateway with the common flags on two fresh ports plus `args`,
/// its output in `log_path`, until it exits, `until` is met (both ports are
/// watched for a bind; the health probe goes to the HTTP port) or
/// [`EXIT_TIMEOUT`] passes.
pub fn launch(args: &[&str], until: Until, log_path: &Path) -> Launch {
    let _serial = serial();
    let [port, metrics_port] = free_ports();
    let mut all = base_args(port, metrics_port);
    all.extend(args.iter().map(|arg| (*arg).to_string()));
    launch_with(&all, &[port, metrics_port], until, log_path)
}

/// Runs the gateway with `args` as its complete flag list, its output in
/// `log_path`, until it exits, `until` is met on the `watch`ed ports (the
/// first of them is the HTTP port the health probe goes to) or
/// [`EXIT_TIMEOUT`] passes. The ports are polled tightly: a listener that is
/// up for a tenth of a second must not go unnoticed. The caller holds
/// [`serial`] from the moment it picked the ports.
pub fn launch_with(args: &[String], watch: &[u16], until: Until, log_path: &Path) -> Launch {
    let log = File::create(log_path).expect("log file");
    let mut child = Command::new(env!("CARGO_BIN_EXE_smg"))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().expect("log handle")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn the gateway");
    let deadline = Instant::now() + EXIT_TIMEOUT;
    let mut bound = None;
    let mut served = false;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the gateway") {
            break Some(status);
        }
        for (index, port) in watch.iter().copied().enumerate() {
            let address = SocketAddr::from((IP, port));
            if let Ok(stream) = TcpStream::connect_timeout(&address, Duration::from_millis(200)) {
                bound.get_or_insert(port);
                if index == 0 && until == Until::Served {
                    served = answers_health(stream);
                }
            }
        }
        let met = match until {
            Until::Bound => bound.is_some(),
            Until::Served => served,
        };
        if met || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(5));
    };
    Launch {
        status,
        bound,
        served,
        log: fs::read_to_string(log_path).unwrap_or_default(),
    }
}

/// Whether the gateway answers `GET /health` with 200 over `stream`, a
/// connection to its port.
fn answers_health(mut stream: TcpStream) -> bool {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut head = [0u8; 64];
    let read = stream.read(&mut head).unwrap_or(0);
    head[..read].starts_with(b"HTTP/1.1 200")
}
