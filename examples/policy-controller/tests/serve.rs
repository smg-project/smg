//! The example controller started as a user starts it: the built binary
//! names the stages it holds, binds its listeners, elects itself and reads
//! ready on its admin listener.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use smg_grpc_client::{vllm_engine::AbortOnDropStream, vllm_proto, VllmEngineClient};
use smg_policy_controller::{
    service::{
        client::{Answer, Client},
        proto::{
            self, decision, policy_controller_client::PolicyControllerClient,
            register_worker_request::Mode, worker_event,
        },
    },
    store::PUBLISH_WINDOW,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinSet};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Request};

const BIN: &str = env!("CARGO_BIN_EXE_policy-controller-example");

/// The example's document on disk, as `--spec-file` names it.
const POLICY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/policy.yaml");

/// How long a start, a line of its log or an answer may take.
const BOUND: Duration = Duration::from_secs(20);

/// The controller under test, killed when the case ends, whatever its
/// verdict.
struct Started {
    child: Child,
    /// Where its admin listener is bound.
    admin: String,
    /// Where its control listener is bound.
    control: String,
    /// Where its gateway listener is bound.
    gateway: String,
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
/// `flags` follow the listeners' own.
fn start(flags: &[&str]) -> Result<Started, String> {
    let mut child = Command::new(BIN)
        .args(["--controller-id", "example-0", "--log-level", "info"])
        .args(["--gateway-listen", "127.0.0.1:0"])
        .args(["--control-listen", "127.0.0.1:0"])
        .args(["--replication-listen", "127.0.0.1:0"])
        .args(["--admin-listen", "127.0.0.1:0"])
        .args(flags)
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
        control: String::new(),
        gateway: String::new(),
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
        } else if line.contains("listener=control") {
            started.control = address_of(&line)?;
        } else if line.contains("listener=gateway") {
            started.gateway = address_of(&line)?;
        } else if line.contains("admin listener bound") {
            break address_of(&line)?;
        }
    };
    Ok(started)
}

/// The address a listener's line of the log names.
fn address_of(line: &str) -> Result<String, String> {
    line.split_whitespace()
        .find_map(|field| field.strip_prefix("addr="))
        .map(str::to_owned)
        .ok_or_else(|| format!("the line names no address: {line}"))
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
    let started = start(&["--spec-file", POLICY]).expect("the example controller starts");
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
    let started = start(&["--spec-file", POLICY]).expect("the example controller starts");
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

#[test]
fn the_built_binary_runs_its_document_from_a_spec_file() {
    let started = start(&["--spec-file", POLICY]).expect("the example controller starts on it");
    // The gauge names the generation the controller runs: the file's.
    let metrics = ok_body(&started.admin, "/metrics").expect("it renders its metrics");
    assert!(
        metrics
            .lines()
            .any(|line| line == "smg_policy_spec_generation 1"),
        "{metrics}"
    );
}

/// The value of an unlabelled counter in a scrape.
fn counter(scrape: &str, name: &str) -> Option<u64> {
    scrape
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
}

/// A mock engine of one rank, served in the case's process until the
/// returned sender drops, or what its start lacked.
async fn engine() -> Result<(SocketAddr, oneshot::Sender<()>), String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|error| format!("no port for the mock engine: {error}"))?;
    let addr = listener
        .local_addr()
        .map_err(|error| format!("the mock engine's listener names no address: {error}"))?;
    let cfg = Arc::new(mock_worker::config::Config {
        grpc_base_port: addr.port(),
        grpc_count: 1,
        grpc_dp_size: 1,
        grpc_wire: mock_worker::config::SchedulerWire::Vllm,
        loads_like: mock_worker::engine::LoadsLike::Vllm,
        realistic: true,
        ..mock_worker::config::Config::default()
    });
    let (stop, stopped) = oneshot::channel();
    let mut served = JoinSet::new();
    served.spawn(mock_worker::grpc::serve_with_listener_until(
        cfg,
        listener,
        async move {
            let _ = stopped.await;
        },
    ));
    served.detach_all();
    Ok((addr, stop))
}

/// The name the cases state on the control listener.
const OPERATOR: &str = "operator-a";

/// The one model the cases' workers serve.
const MODEL: &str = "a-model";

/// `message` as a call that states `name`: a listener its operator named
/// takes the name a call states.
fn stating<T>(name: &'static str, message: T) -> Request<T> {
    let mut call = Request::new(message);
    call.metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(name));
    call
}

/// Registers the engine at `engine` through the door as a worker of the
/// model and returns the worker's id, or says what the door answered. Its
/// checks are off, so it is ready from its registration: its registrant
/// says "do not check". The door opens once the election's steps are told,
/// so a refusal is asked again within the bound.
async fn through_the_door(
    door: &mut PolicyControllerClient<Channel>,
    engine: SocketAddr,
) -> Result<u64, String> {
    let ask = proto::RegisterWorkerRequest {
        spec: Some(proto::WorkerSpec {
            url: format!("http://{engine}"),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            runtime: "vllm".to_owned(),
            models: vec![MODEL.to_owned()],
            health_policy: Some(proto::HealthPolicy {
                disabled: true,
                ..Default::default()
            }),
            ..Default::default()
        }),
        mode: Mode::Add as i32,
        wait_ready: false,
    };
    let deadline = Instant::now() + BOUND;
    loop {
        match door.register_worker(stating(OPERATOR, ask.clone())).await {
            Ok(event) => {
                return match event.into_inner().event {
                    Some(worker_event::Event::Registered(row)) => Ok(row.id),
                    other => Err(format!("the door answered a registration with {other:?}")),
                };
            }
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(status) => return Err(format!("the door took no registration: {status}")),
        }
    }
}

/// The started example takes a worker through its door, as an operator's
/// registration hands it one, and its own load driver polls the engine
/// behind the worker's address: what a decision reads reaches the store of
/// the binary a user starts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_started_binary_takes_a_worker_through_its_door_and_polls_its_engine() {
    // The control listener is one its operator named, and the driver polls
    // an engine once a second.
    let started = start(&[
        "--spec-file",
        POLICY,
        "--identity-as-told",
        "control",
        "--load-monitor-interval",
        "1",
    ])
    .expect("the example controller starts on it");
    let (engine, _served) = engine().await.expect("a mock engine");
    let channel = Channel::from_shared(format!("http://{}", started.control))
        .expect("a uri")
        .connect()
        .await
        .expect("the control listener accepts");
    let mut door = PolicyControllerClient::new(channel);
    through_the_door(&mut door, engine)
        .await
        .expect("the door takes the worker");
    // The leader's own driver polls the engine it was handed.
    let deadline = Instant::now() + BOUND;
    loop {
        let scrape = tokio::task::block_in_place(|| ok_body(&started.admin, "/metrics"))
            .expect("it renders its metrics");
        if counter(&scrape, "smg_policy_controller_load_rows_total").is_some_and(|rows| rows > 0) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "no load row of the engine within the bound: {scrape}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The tokens a held request asks for: it outlasts the case.
const HELD_TOKENS: u32 = 1 << 20;

/// Holds `count` requests in flight on the engine at `engine`, over the
/// engine's own wire, so its load reports say so by themselves. A held
/// stream is never read, and dropping it ends its request.
async fn hold(engine: SocketAddr, count: usize) -> Result<Vec<AbortOnDropStream>, String> {
    let client = VllmEngineClient::connect(&format!("http://{engine}"))
        .await
        .map_err(|error| format!("no connection to the mock engine: {error}"))?;
    let mut held = Vec::with_capacity(count);
    for request in 0..count {
        let asked = client.generate(vllm_proto::GenerateRequest {
            request_id: format!("held-{request}"),
            input: Some(vllm_proto::generate_request::Input::Tokenized(
                vllm_proto::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1; 64],
                },
            )),
            sampling_params: Some(vllm_proto::SamplingParams {
                max_tokens: Some(HELD_TOKENS),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        });
        let stream = asked
            .await
            .map_err(|status| format!("the mock engine took no held request: {status}"))?;
        held.push(stream);
    }
    Ok(held)
}

/// The name the case's gateway states and says Hello under.
const GATEWAY: &str = "gateway-a";

/// How many decisions the case asks for.
const ASKS: u8 = 8;

/// The started example decides by its document over the fleet its door was
/// handed. Of two workers of one model it names the one whose engine
/// reports requests running: `pack` fills a worker before it opens the
/// next, where the stock scorers open the idle one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_started_binary_decides_by_its_document_over_the_fleet_its_door_was_handed() {
    // The gateway and control listeners are ones their operator named, and
    // the driver polls an engine once a second.
    let started = start(&[
        "--spec-file",
        POLICY,
        "--identity-as-told",
        "gateway,control",
        "--load-monitor-interval",
        "1",
    ])
    .expect("the example controller starts on it");
    let (filled_engine, _filled_served) = engine().await.expect("a mock engine");
    let (idle_engine, _idle_served) = engine().await.expect("a mock engine");
    let _held = hold(filled_engine, 3)
        .await
        .expect("one engine holds three requests");
    let channel = Channel::from_shared(format!("http://{}", started.control))
        .expect("a uri")
        .connect()
        .await
        .expect("the control listener accepts");
    let mut door = PolicyControllerClient::new(channel);
    let filled = through_the_door(&mut door, filled_engine)
        .await
        .expect("the door takes the filled worker");
    let idle = through_the_door(&mut door, idle_engine)
        .await
        .expect("the door takes the idle worker");
    // The store holds an engine's report once the driver has asked it
    // twice: the first answer is written before the second ask.
    let deadline = Instant::now() + BOUND;
    while [filled_engine, idle_engine]
        .iter()
        .any(|engine| mock_worker::overrides::for_port(engine.port()).loads_polls() < 2)
    {
        assert!(
            Instant::now() < deadline,
            "the driver did not ask both engines twice within the bound"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // A record the store was handed steers within one publication window.
    tokio::time::sleep(PUBLISH_WINDOW).await;
    // The gateway's side is the crate's reference client: it says Hello on
    // a session and asks on a decide stream, one ask at a time.
    let mut gateway = Client::connect(&format!("http://{}", started.gateway), GATEWAY)
        .await
        .expect("the gateway's session opens");
    // Every answer names the filled worker first, whatever its seed.
    for ask in 0..ASKS {
        let answer = gateway
            .decide(proto::DecideRequest {
                request_id: vec![ask],
                facts: Some(proto::RequestFacts {
                    model: MODEL.to_owned(),
                    seed: u64::from(ask),
                    budget_micros: 5_000_000,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .expect("the controller answers the ask");
        let Answer::Decision(proto::Decision {
            outcome: Some(decision::Outcome::Pick(pick)),
            ..
        }) = &answer
        else {
            panic!("a pick answers an ask over a ready fleet: {answer:?}");
        };
        let first = pick.targets.first().map(|target| target.id);
        assert_eq!(first, Some(filled), "the idle worker is {idle}: {pick:?}");
    }
}
