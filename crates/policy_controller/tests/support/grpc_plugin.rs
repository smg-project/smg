//! A `PolicyPlugin` served in the test's process: one server per test over a
//! Unix socket in a directory of its own (or TCP under mutual TLS), on a
//! runtime of its own so the test's thread stays the deciding thread. A test
//! sets how the plugin answers ([`Script`]) between decisions and reads what
//! it was sent ([`Seen`]).

use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use rustls::crypto::ring;
use smg_policy_controller::service::proto::{
    self,
    decline::Reason,
    policy_plugin_server::{PolicyPlugin, PolicyPluginServer},
};
use tokio::{
    net::{TcpListener, UnixListener},
    sync::{mpsc, Semaphore},
    task::JoinSet,
};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream, UnixListenerStream};
use tonic::{
    transport::{Certificate, Identity, Server, ServerTlsConfig},
    Request, Response, Status, Streaming,
};

pub const FILTER: u32 = proto::Stage::Filter as u32;
pub const SCORE: u32 = proto::Stage::Score as u32;
pub const PICK: u32 = proto::Stage::Pick as u32;
pub const ALL: u32 = FILTER | SCORE | PICK;

/// What the plugin does with a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Behaviour {
    /// The filter keeps the healthy rows, a score falls with the row's
    /// in-flight count, the pick is the surviving row with the fewest in
    /// flight; asked for the pick alone, the surviving row (a combined score
    /// at or above zero) with the lowest combined score, which no native
    /// picker would choose.
    #[default]
    Honest,
    /// The filter keeps row 0 alone and the pick is row 0: the most loaded
    /// row of the tests' pools, which no built-in stage would leave.
    KeepFirst,
    /// A score that rises with the row's in-flight count: the most loaded
    /// row scores highest, the opposite of the built-in scorer.
    ScoreMostLoaded,
    /// Honest scores with a tie band in thousandths.
    ScoreBand(u16),
    /// Declines these stages for this reason and answers the rest.
    Decline { stages: u32, reason: i32 },
    /// Answers with no field at all.
    Unanswered,
    /// Never answers.
    Silent,
    /// A surviving position past the candidates.
    SurvivorOutOfRange,
    /// A surviving position named twice.
    SurvivorTwice,
    /// One score too few.
    ScoreCount,
    /// A score that is not a number.
    ScoreNan,
    /// A band that is not a number.
    BandNan,
    /// A band above one.
    BandAboveOne,
    /// A band below zero.
    BandBelowZero,
    /// A band with a named decline of the scorer.
    BandWithScoreDecline,
    /// A band without scores.
    BandWithoutScores,
    /// A pick outside the plugin's own survivors.
    PickOutsideOwnSurvivors,
    /// Asked for the pick alone: a row the frame marked as dropped.
    PickDroppedRow,
    /// Asked at the open: always position 0, whatever the filters do later.
    PickFirst,
    /// Declines the filter and fills its field too.
    DeclinedAndAnswered,
    /// Ends the response stream instead of answering.
    EndStream,
}

/// A gate a hello's ack waits at until a case opens it: an order two
/// streams settle in is the case's own, at any load.
#[derive(Clone, Debug)]
pub struct Gate(Arc<Semaphore>);

impl Default for Gate {
    fn default() -> Self {
        Self(Arc::new(Semaphore::new(0)))
    }
}

impl Gate {
    /// Every ack that waits here goes, and a later one does not wait.
    pub fn open(&self) {
        self.0.close();
    }

    async fn opened(&self) {
        // No permit is ever given: the wait ends when the gate is closed.
        let _ = self.0.acquire().await;
    }
}

/// How the plugin answers; a test changes it between decisions.
#[derive(Clone, Debug)]
pub struct Script {
    pub behaviour: Behaviour,
    /// Waits this long before an answer.
    pub delay: Duration,
    /// Waits this long before the hello's ack.
    pub hello_delay: Duration,
    /// When set: the hello's ack waits until the gate is opened.
    pub hello_gate: Option<Gate>,
    /// The ack: `ok`, its error, the version it names and the stages it
    /// claims.
    pub ack_ok: bool,
    pub ack_error: String,
    pub ack_protocol: u32,
    pub implements: u32,
    /// Counter increments every answer carries.
    pub counters: Vec<(u32, u64)>,
    /// Keeps nothing it is sent: a bench's frames and events are millions.
    pub forget: bool,
    /// With `EndStream`: the status's words a `Decide` stream ends with,
    /// when not empty; a clean end otherwise.
    pub end_status: String,
    /// When not empty: the `Observe` call is answered at its next message
    /// with an Ack that carries these words.
    pub observe_ack: String,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            behaviour: Behaviour::Honest,
            delay: Duration::ZERO,
            hello_delay: Duration::ZERO,
            hello_gate: None,
            ack_ok: true,
            ack_error: String::new(),
            ack_protocol: 1,
            implements: ALL,
            counters: Vec::new(),
            forget: false,
            end_status: String::new(),
            observe_ack: String::new(),
        }
    }
}

/// What the plugin was sent.
#[derive(Debug, Default)]
pub struct Seen {
    pub hellos: Vec<proto::PluginHello>,
    pub frames: Vec<proto::PluginDecide>,
    pub observe_hellos: Vec<proto::PluginHello>,
    pub events: Vec<proto::LifecycleEvent>,
    /// `Decide` streams whose request side ended.
    pub ended_streams: u32,
    /// `Observe` calls answered with their Ack.
    pub acked_observes: u32,
}

#[derive(Clone, Default)]
struct Shared {
    script: Arc<Mutex<Script>>,
    seen: Arc<Mutex<Seen>>,
    /// One task per `Decide` stream, gone with the served plugin.
    streams: Arc<Mutex<JoinSet<()>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn healthy(c: &proto::Candidate) -> bool {
    c.health & (proto::HealthFlag::Healthy as u32) != 0
}

/// The honest answer to `frame`: each asked stage's own field.
fn honest(frame: &proto::PluginDecide) -> proto::PluginDecision {
    let asked = frame.stages;
    let pick_only = !frame.scores_in.is_empty();
    let survivors: Vec<u32> = frame
        .candidates
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            if pick_only {
                frame.scores_in.get(*i).is_some_and(|s| *s >= 0.0)
            } else {
                asked & FILTER == 0 || healthy(c)
            }
        })
        .map(|(i, _)| i as u32)
        .collect();
    let pick = if pick_only {
        survivors
            .iter()
            .copied()
            .min_by(|a, b| frame.scores_in[*a as usize].total_cmp(&frame.scores_in[*b as usize]))
    } else {
        survivors
            .iter()
            .copied()
            .min_by_key(|i| frame.candidates[*i as usize].in_flight)
    };
    proto::PluginDecision {
        decision_id: frame.decision_id,
        survivors: (asked & FILTER != 0).then_some(proto::Survivors {
            positions: survivors,
        }),
        scores: if asked & SCORE != 0 {
            frame
                .candidates
                .iter()
                .map(|c| 1.0 / (1.0 + c.in_flight as f32))
                .collect()
        } else {
            Vec::new()
        },
        pick: if asked & PICK != 0 { pick } else { None },
        decline: None,
        counters: Vec::new(),
        tie_band: None,
    }
}

/// The scripted answer to `frame`; `None` for no answer at all.
fn answer(frame: &proto::PluginDecide, script: &Script) -> Option<proto::PluginDecision> {
    let count = frame.candidates.len() as u32;
    let mut out = honest(frame);
    match script.behaviour {
        Behaviour::Honest => {}
        Behaviour::KeepFirst => {
            if out.survivors.is_some() {
                out.survivors = Some(proto::Survivors { positions: vec![0] });
            }
            if out.pick.is_some() {
                out.pick = Some(0);
            }
        }
        Behaviour::ScoreMostLoaded => {
            let most = frame
                .candidates
                .iter()
                .map(|c| c.in_flight)
                .max()
                .unwrap_or(1)
                .max(1);
            for (score, c) in out.scores.iter_mut().zip(&frame.candidates) {
                *score = c.in_flight as f32 / most as f32;
            }
        }
        Behaviour::ScoreBand(band) => out.tie_band = Some(f32::from(band) / 1_000.0),
        Behaviour::Decline { stages, reason } => {
            if stages & FILTER != 0 {
                out.survivors = None;
            }
            if stages & SCORE != 0 {
                out.scores.clear();
            }
            if stages & PICK != 0 {
                out.pick = None;
            }
            out.decline = Some(proto::Decline {
                stage: stages,
                reason,
                detail: "scripted".to_owned(),
            });
        }
        Behaviour::Unanswered => {
            out.survivors = None;
            out.scores.clear();
            out.pick = None;
        }
        Behaviour::Silent | Behaviour::EndStream => return None,
        Behaviour::SurvivorOutOfRange => {
            out.survivors = Some(proto::Survivors {
                positions: vec![count],
            });
            out.pick = None;
        }
        Behaviour::SurvivorTwice => {
            out.survivors = Some(proto::Survivors {
                positions: vec![0, 0],
            });
            out.pick = None;
        }
        Behaviour::ScoreCount => {
            out.scores.pop();
        }
        Behaviour::ScoreNan => {
            if let Some(first) = out.scores.first_mut() {
                *first = f32::NAN;
            }
        }
        Behaviour::BandNan => out.tie_band = Some(f32::NAN),
        Behaviour::BandAboveOne => out.tie_band = Some(1.375),
        Behaviour::BandBelowZero => out.tie_band = Some(-0.375),
        Behaviour::BandWithScoreDecline => {
            out.scores.clear();
            out.tie_band = Some(0.0625);
            out.decline = Some(proto::Decline {
                stage: SCORE,
                reason: Reason::NotApplicable as i32,
                detail: String::new(),
            });
        }
        Behaviour::BandWithoutScores => {
            out.scores.clear();
            out.tie_band = Some(0.8125);
        }
        Behaviour::PickOutsideOwnSurvivors => {
            out.survivors = Some(proto::Survivors { positions: vec![0] });
            out.pick = Some(1);
        }
        Behaviour::PickDroppedRow => {
            out.pick = frame
                .scores_in
                .iter()
                .position(|s| *s < 0.0)
                .map(|i| i as u32);
        }
        Behaviour::PickFirst => {
            out.pick = Some(0);
        }
        Behaviour::DeclinedAndAnswered => {
            out.decline = Some(proto::Decline {
                stage: FILTER,
                reason: Reason::NotApplicable as i32,
                detail: String::new(),
            });
        }
    }
    out.counters = script
        .counters
        .iter()
        .map(|(id, by)| proto::CounterDelta { id: *id, by: *by })
        .collect();
    Some(out)
}

/// Waits `delay` when the script sets one. The timer counts in whole
/// milliseconds and rounds up, so a sleep of zero would hold every answer
/// for up to one.
async fn pause(delay: Duration) {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
}

#[tonic::async_trait]
impl PolicyPlugin for Shared {
    type DecideStream = ReceiverStream<Result<proto::PluginResponse, Status>>;

    async fn decide(
        &self,
        request: Request<Streaming<proto::PluginRequest>>,
    ) -> Result<Response<Self::DecideStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel(16);
        let shared = self.clone();
        lock(&self.streams).spawn(async move {
            while let Ok(Some(message)) = inbound.message().await {
                let script = lock(&shared.script).clone();
                match message.request {
                    Some(proto::plugin_request::Request::Hello(hello)) => {
                        lock(&shared.seen).hellos.push(hello);
                        pause(script.hello_delay).await;
                        if let Some(gate) = &script.hello_gate {
                            gate.opened().await;
                        }
                        let ack = proto::PluginHelloAck {
                            protocol: script.ack_protocol,
                            ok: script.ack_ok,
                            error: script.ack_error.clone(),
                            stages: script.implements,
                        };
                        let response = proto::PluginResponse {
                            response: Some(proto::plugin_response::Response::Hello(ack)),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Some(proto::plugin_request::Request::Decide(frame)) => {
                        let answered = answer(&frame, &script);
                        if !script.forget {
                            lock(&shared.seen).frames.push(frame);
                        }
                        if script.behaviour == Behaviour::EndStream {
                            // The response stream ends with its sender, after
                            // the delay an answer would wait: cleanly, or
                            // with the status the script names.
                            pause(script.delay).await;
                            if !script.end_status.is_empty() {
                                let status = Status::internal(script.end_status);
                                let _ = tx.send(Err(status)).await;
                            }
                            return;
                        }
                        let Some(decision) = answered else {
                            continue;
                        };
                        pause(script.delay).await;
                        let response = proto::PluginResponse {
                            response: Some(proto::plugin_response::Response::Decision(decision)),
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    None => {}
                }
            }
            lock(&shared.seen).ended_streams += 1;
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn observe(
        &self,
        request: Request<Streaming<proto::ObserveRequest>>,
    ) -> Result<Response<proto::Ack>, Status> {
        let mut inbound = request.into_inner();
        let mut frames = 0;
        while let Some(message) = inbound.message().await? {
            frames += 1;
            match message.request {
                Some(proto::observe_request::Request::Hello(hello)) => {
                    lock(&self.seen).observe_hellos.push(hello);
                }
                Some(proto::observe_request::Request::Event(event))
                    if !lock(&self.script).forget =>
                {
                    lock(&self.seen).events.push(event);
                }
                Some(proto::observe_request::Request::Event(_)) | None => {}
            }
            let words = lock(&self.script).observe_ack.clone();
            if !words.is_empty() {
                return Ok(Response::new(proto::Ack {
                    frames,
                    accepted: frames,
                    error: words,
                }));
            }
        }
        lock(&self.seen).acked_observes += 1;
        Ok(Response::new(proto::Ack {
            frames,
            accepted: frames,
            error: String::new(),
        }))
    }
}

/// The server's side of mutual TLS: its identity and the authority that
/// signs its clients, PEM.
pub struct ServerTls {
    pub cert_pem: String,
    pub key_pem: String,
    pub client_ca_pem: String,
}

/// One served plugin; dropped, its server and every stream go.
pub struct TestPlugin {
    /// The Unix socket's path (empty for the TLS server).
    pub socket: PathBuf,
    /// The TLS server's address.
    pub address: Option<SocketAddr>,
    shared: Shared,
    _dir: tempfile::TempDir,
    runtime: Option<tokio::runtime::Runtime>,
}

impl TestPlugin {
    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("the test plugin's runtime")
    }

    /// Serves a plugin on a Unix socket in a fresh directory.
    pub fn serve() -> Self {
        let dir = tempfile::tempdir().expect("a directory for the socket");
        let socket = dir.path().join("plugin.sock");
        let runtime = Self::runtime();
        let shared = Shared::default();
        let listener = {
            let _guard = runtime.enter();
            UnixListener::bind(&socket).expect("the socket binds")
        };
        let service = PolicyPluginServer::new(shared.clone());
        runtime.spawn(async move {
            let _ = Server::builder()
                .add_service(service)
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await;
        });
        Self {
            socket,
            address: None,
            shared,
            _dir: dir,
            runtime: Some(runtime),
        }
    }

    /// Serves a plugin on a loopback port under mutual TLS.
    pub fn serve_tls(tls: &ServerTls) -> Self {
        // The process TLS provider, as a served plugin installs its own.
        let _ = ring::default_provider().install_default();
        let dir = tempfile::tempdir().expect("a directory");
        let runtime = Self::runtime();
        let shared = Shared::default();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("a loopback port");
        let address = listener.local_addr().expect("the bound address");
        let config = ServerTlsConfig::new()
            .identity(Identity::from_pem(&tls.cert_pem, &tls.key_pem))
            .client_ca_root(Certificate::from_pem(&tls.client_ca_pem));
        let service = PolicyPluginServer::new(shared.clone());
        runtime.spawn(async move {
            let _ = Server::builder()
                .tls_config(config)
                .expect("the server's TLS configuration")
                .add_service(service)
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await;
        });
        Self {
            socket: PathBuf::new(),
            address: Some(address),
            shared,
            _dir: dir,
            runtime: Some(runtime),
        }
    }

    /// Changes how the plugin answers from the next frame on.
    pub fn script(&self, change: impl FnOnce(&mut Script)) {
        change(&mut lock(&self.shared.script));
    }

    pub fn behave(&self, behaviour: Behaviour) {
        self.script(|s| s.behaviour = behaviour);
    }

    /// What the plugin was sent so far.
    pub fn seen(&self) -> MutexGuard<'_, Seen> {
        lock(&self.shared.seen)
    }

    /// The frames the plugin was sent so far.
    pub fn frames(&self) -> usize {
        self.seen().frames.len()
    }

    /// Stops the server: every stream ends and the socket refuses.
    pub fn stop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
        if !self.socket.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.socket);
        }
    }
}

impl Drop for TestPlugin {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Waits (at most five seconds) until `ready` holds; whether it did.
pub fn eventually(mut ready: impl FnMut() -> bool) -> bool {
    for _ in 0..5_000 {
        if ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    ready()
}
