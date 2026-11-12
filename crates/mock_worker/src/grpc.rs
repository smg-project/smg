//! Mock gRPC worker: the scheduler service of one engine wire, chosen per
//! process (`--grpc-wire`). The gateway tokenizes and sends token ids; the
//! service streams back canned token ids. This module is what the wires
//! share: the server, the worker's state and its ranks, the generate paths (a
//! prefill side's answer, a scripted answer, the realistic engine, the canned
//! ids) as wire-neutral frames, the loads, the KV-event streams, and the
//! standard `grpc.health.v1` service, which answers as the scheduler's own
//! `HealthCheck` does. The wire modules read each proto's request and shape
//! its responses.

use std::{
    collections::VecDeque,
    future::{pending, Future},
    pin::{pin, Pin},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{common_proto as common, constraint_failed_status};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch},
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{server::NamedService, transport::Server, Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus,
    health_server::{Health, HealthServer},
    HealthCheckRequest, HealthCheckResponse,
};

use crate::{
    admin_ops::{AdminOps, ProfileOptions},
    config::{Config, Role, SchedulerWire},
    drain::{InFlight, WorkerInFlight},
    engine::{self, Engine, KvEventStream, LoadsLike, NewRequest},
    replay::Capture,
    script::{self, Answer, MatchedStop},
};

mod sglang;
mod tokenspeed;
mod vllm;

/// Serve the mock gRPC service of the configured wire on `port` until the
/// process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    serve_until(cfg, host, port, pending()).await;
}

/// Serve on `port` until `shutdown` resolves, then stop accepting connections
/// and let the requests in flight finish (the drain of a termination signal).
pub async fn serve_until(
    cfg: Arc<Config>,
    host: String,
    port: u16,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    // What the HTTP listener accepts: a bare IP of either family or a name.
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("grpc worker bind {host}:{port} failed: {e}");
            return;
        }
    };
    serve_with_listener_until(cfg, listener, shutdown).await;
}

/// Serve on an already-bound listener. Callers that need a free port bind
/// port 0 and read it back instead of picking one and binding it later.
pub async fn serve_with_listener(cfg: Arc<Config>, listener: TcpListener) {
    serve_with_listener_until(cfg, listener, pending()).await;
}

/// [`serve_with_listener`] with the shutdown of [`serve_until`].
pub async fn serve_with_listener_until(
    cfg: Arc<Config>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    let addr = listener.local_addr().ok();
    let capture = match &cfg.replay.capture {
        Some(path) => match Capture::open(path) {
            Ok(capture) => Some(Arc::new(capture)),
            Err(e) => {
                tracing::error!(
                    "grpc worker {addr:?} cannot open capture file {}: {e}",
                    path.display()
                );
                return;
            }
        },
        None => None,
    };
    if let Err(e) = script::ensure_loaded(&cfg) {
        tracing::error!("grpc worker {addr:?} cannot load script: {e}");
        return;
    }
    // A prefill worker's bootstrap port, numbered by the worker's port offset:
    // bound before the worker joins the fleet and serves, as an engine's
    // bootstrap server is up when its health answers; a port that cannot be
    // bound stops the worker as its own port would, and a worker that never
    // served is in no registry.
    let bootstrap_port = addr
        .map(|a| a.port())
        .and_then(|port| port.checked_sub(cfg.grpc_base_port))
        .and_then(|index| cfg.bootstrap_port_for(index));
    let bootstrap = match crate::bootstrap::bind(&cfg.host, bootstrap_port).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("grpc worker {addr:?} {e}");
            return;
        }
    };
    // One simulated engine per listener (i.e. per virtual worker), registered
    // in the process fleet under `grpc:<port>` for the oracle and admin API.
    let name = format!("grpc:{}", addr.map(|a| a.port()).unwrap_or(0));
    script::register(&name);
    let ops = AdminOps::register(&name);
    let ranks = Rank::spawn_all(&cfg, &name);
    // The worker's vLLM-wire KV-event publisher, numbered by its port offset:
    // its one rank's (the publisher ports are per worker, not per rank, so
    // `Config::parse` refuses the flag on a worker of several ranks).
    let publisher = ranks.first().and_then(|rank| {
        let port = addr.map(|a| a.port())?;
        let index = port.checked_sub(cfg.grpc_base_port)?;
        cfg.kv_zmq_for(index, 0)
            .map(|kv| crate::kv_zmq::serve(rank.engine.clone(), kv))
    });
    let max_message_bytes = cfg.grpc_max_message_bytes;
    // The stop reaches the service's streams that have no end of their own
    // (a KV-event subscription) before the server's graceful shutdown
    // begins, which waits for every open stream.
    let (stop, stopped) = watch::channel(false);
    let shutdown = async move {
        shutdown.await;
        let _ = stop.send(true);
    };
    let overrides = crate::overrides::for_port(addr.map(|a| a.port()).unwrap_or(0));
    let wire = cfg.grpc_wire;
    let health = MockHealth {
        overrides: Arc::clone(&overrides),
        stopped: stopped.clone(),
        service: health_service_name(wire, cfg.role),
    };
    let bootstrap =
        bootstrap.map(|listener| crate::bootstrap::accept_and_close(listener, stopped.clone()));
    let service = MockScheduler {
        cfg,
        name,
        ranks,
        scan_start: Arc::new(AtomicUsize::new(0)),
        capture,
        stopped,
        in_flight: WorkerInFlight::default(),
        overrides,
        bootstrap_port,
        ops,
    };
    let server = async {
        // tonic drains on incoming EOF when graceful shutdown is enabled,
        // but retains the stream while waiting for existing connections.
        // Drop the listener at EOF so the port closes before that wait.
        let incoming = stream::unfold(
            Box::pin(connections(listener).take_until(shutdown)),
            |mut incoming| async move {
                incoming
                    .next()
                    .await
                    .map(|connection| (connection, incoming))
            },
        );
        // tonic's 4 MiB default would refuse the Generate of a long prompt (a
        // million token ids are a few megabytes as varints, more with the
        // prompt text alongside); the limit follows the config: none by
        // default, as the engine servicers run.
        // The one scheduler service of the configured wire: the gateway's
        // detection asks each wire's health check in turn and takes the
        // first that answers, so another wire's scheduler here would name the
        // worker's runtime for it.
        let mut builder = Server::builder();
        let routes = match wire {
            // An encode worker on this wire mounts the engine's encoder service
            // beside the scheduler's, as the servicer does: the gateway's encode
            // request is that service's `Encode`.
            SchedulerWire::TokenSpeed => {
                let encoder = (service.cfg.role == Some(Role::Encode)).then(|| {
                    tokenspeed::EncoderServer::new(service.clone())
                        .max_decoding_message_size(max_message_bytes)
                        .max_encoding_message_size(max_message_bytes)
                });
                builder
                    .add_service(
                        tokenspeed::Server::new(service)
                            .max_decoding_message_size(max_message_bytes)
                            .max_encoding_message_size(max_message_bytes),
                    )
                    .add_optional_service(encoder)
            }
            SchedulerWire::Vllm => builder.add_service(
                vllm::Server::new(service)
                    .max_decoding_message_size(max_message_bytes)
                    .max_encoding_message_size(max_message_bytes),
            ),
            SchedulerWire::Sglang => builder.add_service(
                sglang::Server::new(service)
                    .max_decoding_message_size(max_message_bytes)
                    .max_encoding_message_size(max_message_bytes),
            ),
        };
        if let Err(e) = routes
            .add_service(HealthServer::new(health))
            .serve_with_incoming_shutdown(incoming, pending())
            .await
        {
            tracing::error!("grpc worker {addr:?} stopped: {e}");
        }
    };
    // The worker's side tasks run beside the server and go with it when it
    // stops (the shutdown, or an error): the KV-event publisher, which leaves
    // the server serving when it stops on its own, as before, and a prefill
    // worker's bootstrap listener, which ends at the stop. The two are
    // joined, so one ending early leaves the other polled.
    let mut server = pin!(server);
    let sides = async {
        let publisher = async {
            match publisher {
                Some(publisher) => publisher.await,
                None => pending().await,
            }
        };
        let bootstrap = async {
            match bootstrap {
                Some(bootstrap) => bootstrap.await,
                None => pending().await,
            }
        };
        tokio::join!(publisher, bootstrap);
    };
    tokio::select! {
        () = &mut server => {}
        () = sides => server.await,
    }
}

/// The listener's connections with `TCP_NODELAY` set, as the engine
/// servicers serve theirs. tonic ignores its own `tcp_nodelay` setting for a
/// caller-provided incoming stream, and with Nagle on the response HEADERS
/// frame goes out alone while the first DATA frame (the first token) waits
/// for its acknowledgement, which the gateway's delayed ACK holds for ~40 ms
/// on every other request.
fn connections(listener: TcpListener) -> impl Stream<Item = std::io::Result<TcpStream>> {
    TcpListenerStream::new(listener).map(|conn| {
        if let Ok(stream) = &conn {
            if let Err(e) = stream.set_nodelay(true) {
                tracing::warn!("grpc worker: set_nodelay failed: {e}");
            }
        }
        conn
    })
}

#[derive(Clone)]
struct MockScheduler {
    cfg: Arc<Config>,
    /// The worker's name in the process (`grpc:<port>`): what the admin API
    /// and the script store address it by.
    name: String,
    /// The worker's data-parallel ranks, one realistic engine each (empty for
    /// a canned worker). A pinned request runs on its rank, an unpinned one
    /// on the rank holding the fewest of this worker's requests.
    ranks: Vec<Rank>,
    /// Where the unpinned scan starts, one rank further each time, so
    /// all-zero ties do not always resolve to rank 0.
    scan_start: Arc<AtomicUsize>,
    /// Present iff `--capture` is set: every `Generate` request is recorded.
    capture: Option<Arc<Capture>>,
    /// Flips when the worker stops: ends the streams that have no end of
    /// their own.
    stopped: watch::Receiver<bool>,
    /// The requests this worker is serving: what its load report counts as
    /// running on a canned worker, which runs no engine simulation.
    in_flight: WorkerInFlight,
    /// The flush and profile operations the gateway fans out, and their counts.
    ops: Arc<AdminOps>,
    /// What the admin API made this worker report about itself (serving,
    /// server args, model capabilities).
    overrides: Arc<crate::overrides::WorkerOverrides>,
    /// The bootstrap port this worker bound and advertises: a prefill
    /// worker's, when it was given one.
    bootstrap_port: Option<u16>,
}

/// One data-parallel rank of a gRPC worker: its engine, the prefill work in
/// hand on it (prompt tokens handed to the engine with no token yet, as a
/// servicer reports them on the rank's load record), and the requests this
/// worker placed on it whose streams have not ended.
#[derive(Clone)]
struct Rank {
    engine: Engine,
    prefill_pending: Arc<AtomicU64>,
    in_flight: Arc<AtomicUsize>,
}

impl Rank {
    /// The worker's ranks: `--grpc-dp-size` realistic engines, rank `r` named
    /// `<worker>:r<r>` in the process fleet (a single rank keeps the worker's
    /// name) and stamping `r` on its KV-event batches; none for a canned
    /// worker.
    fn spawn_all(cfg: &Config, worker: &str) -> Vec<Rank> {
        if !cfg.realistic {
            return Vec::new();
        }
        let size = usize::from(cfg.grpc_dp_size.max(1));
        (0..size)
            .map(|rank| {
                let mut params = cfg.engine.clone();
                params.dp_rank = i32::try_from(rank).unwrap_or(i32::MAX);
                let name = if size > 1 {
                    format!("{worker}:r{rank}")
                } else {
                    worker.to_string()
                };
                Rank {
                    engine: Engine::spawn_named(params, name, true),
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                    in_flight: Arc::new(AtomicUsize::new(0)),
                }
            })
            .collect()
    }
}

/// A request placed on a rank: the rank's count comes down when this is
/// dropped, with the stream that carries it (however the stream ends).
struct Placed(Arc<AtomicUsize>);

impl Drop for Placed {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The line a worker of several ranks logs per request: the rank the request
/// came pinned to (`dp_rank=<n>`, or `dp_rank=None` for an unpinned one) and
/// the rank it ran on. An engine's servicer logs the pinned rank in this form
/// and the data-parallel e2e class reads a worker's log for it, so the mock
/// twin of that class reads the mock as it reads an engine. A worker of one
/// rank logs nothing: the line is about the placement, not the request.
fn rank_line(request_id: &str, pin: Option<i32>, rank: usize) -> String {
    let pin = pin.map_or_else(|| "None".to_string(), |pin| pin.to_string());
    format!("request {request_id}: dp_rank={pin}, ran on rank {rank}")
}

impl MockScheduler {
    /// The rank a request runs on: its `data_parallel_rank` when pinned (a
    /// rank past the worker's is the caller's error, as the servicers answer
    /// it), else the rank holding the fewest of this worker's requests, the
    /// scan starting one rank further each time. The rank's count is taken
    /// here, under no lock: two unpinned requests placed at once may read the
    /// same counts, as a servicer's own balancer may.
    fn place(&self, pin: Option<i32>) -> Result<(usize, Placed), Status> {
        let count = self.ranks.len();
        let rank = match pin {
            Some(rank) => usize::try_from(rank)
                .ok()
                .filter(|rank| *rank < count)
                .ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "data_parallel_rank {rank} out of range [0, {count})"
                    ))
                })?,
            None => {
                let start = self.scan_start.fetch_add(1, Ordering::Relaxed) % count.max(1);
                (0..count)
                    .map(|offset| (start + offset) % count)
                    .min_by_key(|&rank| self.ranks[rank].in_flight.load(Ordering::Relaxed))
                    .unwrap_or(0)
            }
        };
        let in_flight = Arc::clone(&self.ranks[rank].in_flight);
        in_flight.fetch_add(1, Ordering::Relaxed);
        Ok((rank, Placed(in_flight)))
    }
}

/// What a wire's `Generate` asks of the core, whatever its proto's spelling:
/// the gateway's token ids and the prompt it rendered them from, how many
/// choices the request samples, its token limit, whether it streams, the
/// logprobs it wants (the alternatives per token; `None` for none), the token
/// ids it asks a score for (a prefill-only scoring request's; empty on the
/// wires without the field), the rank it is pinned to, and the KV
/// rendezvous a disaggregated request carries ([`Rendezvous`]).
pub(crate) struct GenerateInput {
    pub(crate) request_id: String,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) rendered: String,
    pub(crate) choices: usize,
    pub(crate) max_new: Option<u32>,
    pub(crate) stream: bool,
    pub(crate) logprobs: Option<usize>,
    pub(crate) score_ids: Vec<u32>,
    pub(crate) pin: Option<i32>,
    pub(crate) rendezvous: Rendezvous,
}

/// The KV rendezvous a disaggregated request carries, read from each
/// wire's own field ([`SchedulerWire::rendezvous_field`]): the prefill
/// side's bootstrap server and the room both sides meet under on the
/// TokenSpeed and SGLang wires, or the connector's parameters the vLLM
/// wire carries, opaque to the mock. A request without one is a regular
/// request, or a disaggregated one the gateway sent without its rendezvous.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rendezvous {
    #[default]
    None,
    Named {
        host: String,
        port: i32,
        room: i64,
    },
    /// A rendezvous field the request carries emptied (no host, or no port),
    /// reported under that field's name rather than as the field's absence:
    /// the vLLM wire's typed pair, which the legacy connector path sends. The
    /// port is the wire's `uint32` as carried, so the refusal names it whole.
    Emptied {
        field: &'static str,
        host: String,
        port: u32,
    },
    Opaque,
}

/// One media item's placeholder spans against the prompt, as the engines
/// check them before they place its embedding: at least one span, none
/// empty, each inside the prompt when there is one (an encode request has
/// none), none over a token an earlier span of the request claimed
/// (`claimed` holds the request's spans so far, one list shared across its
/// items), and, when the item names its placeholder token id, every prompt
/// token under its spans is that id. Returns the spans' total length, which
/// the caller sums per token id for [`check_placeholder_tokens`]. `what`
/// names the item in the refusal.
pub(crate) fn check_placeholder_spans(
    prompt_ids: Option<&[u32]>,
    spans: impl IntoIterator<Item = (u32, u32)>,
    placeholder_token_id: Option<u32>,
    claimed: &mut Vec<(usize, usize)>,
    what: &str,
) -> Result<usize, Status> {
    let mut covered = 0usize;
    let mut any = false;
    for (offset, length) in spans {
        any = true;
        if length == 0 {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} carries a placeholder of length 0 at offset {offset}"
            )));
        }
        let start = usize::try_from(offset).unwrap_or(usize::MAX);
        let end = start.checked_add(usize::try_from(length).unwrap_or(usize::MAX));
        let end = if let Some(prompt_ids) = prompt_ids {
            let Some(end) = end.filter(|end| *end <= prompt_ids.len()) else {
                return Err(Status::invalid_argument(format!(
                    "mock-worker: {what}'s placeholder at offset {offset} of length {length} lies \
                     outside the prompt of {} tokens",
                    prompt_ids.len()
                )));
            };
            if let Some(id) = placeholder_token_id {
                if let Some(other) = prompt_ids[start..end].iter().find(|token| **token != id) {
                    return Err(Status::invalid_argument(format!(
                        "mock-worker: {what}'s placeholder at offset {offset} of length {length} \
                         covers token {other}, not its placeholder token {id}"
                    )));
                }
            }
            end
        } else {
            // An encode request has no prompt: the span is held against the
            // request's other spans alone.
            end.unwrap_or(usize::MAX)
        };
        if let Some(&(from, _)) = claimed.iter().find(|(from, to)| start < *to && *from < end) {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what}'s placeholder at offset {offset} of length {length} overlaps \
                 an earlier placeholder at token {}",
                start.max(from)
            )));
        }
        claimed.push((start, end));
        covered += end - start;
    }
    if !any {
        return Err(Status::invalid_argument(format!(
            "mock-worker: {what} carries no placeholders"
        )));
    }
    Ok(covered)
}

/// The prompt's placeholder tokens against the items' spans: every prompt
/// token equal to `id` belongs to an item's span (`covered` is the spans'
/// total under that id, a count of distinct tokens as the spans are
/// disjoint), so an item the gateway dropped while another item of that
/// id remains, or a placeholder it left unexpanded, is refused rather than
/// fed to the engine as text (the ids come from the items that remain: a
/// request that lost every item of an id is the engine's to refuse).
pub(crate) fn check_placeholder_tokens(
    prompt_ids: &[u32],
    id: u32,
    covered: usize,
) -> Result<(), Status> {
    let in_prompt = prompt_ids.iter().filter(|token| **token == id).count();
    if in_prompt != covered {
        return Err(Status::invalid_argument(format!(
            "mock-worker: the prompt holds {in_prompt} placeholder token(s) {id} and the media \
             items' spans cover {covered}"
        )));
    }
    Ok(())
}

/// The logprobs of a frame's tokens: the sampled token's per position, the
/// alternatives the request asked for, and on a completion the score of each
/// token id the request asked for (empty elsewhere).
#[derive(Clone, Debug)]
pub(crate) struct LogProbs {
    pub(crate) token_logprobs: Vec<f32>,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) top: Vec<script::TopAlternatives>,
    pub(crate) scores: Vec<(u32, f32)>,
}

/// A streamed frame: the tokens new since the frame before, and the running
/// counts. Every wire's chunk carries the frame's new ids; the gateway sums
/// them on the wire whose counts are deltas and reads the completion's on
/// the others, so nothing here differs per wire.
#[derive(Clone, Debug)]
pub(crate) struct Chunk {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) cached_tokens: u32,
    pub(crate) logprobs: Option<LogProbs>,
    pub(crate) index: u32,
}

/// A choice's completion: the whole output and its totals.
#[derive(Clone, Debug)]
pub(crate) struct Complete {
    pub(crate) output_ids: Vec<u32>,
    pub(crate) finish_reason: String,
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) cached_tokens: u32,
    pub(crate) logprobs: Option<LogProbs>,
    pub(crate) matched_stop: Option<MatchedStop>,
    pub(crate) index: u32,
}

/// One item of a generate stream before it takes a wire's shape.
#[derive(Clone, Debug)]
pub(crate) enum Frame {
    Chunk(Chunk),
    Complete(Complete),
}

pub(crate) type FrameStream = Pin<Box<dyn Stream<Item = Result<Frame, Status>> + Send>>;
pub(crate) type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

/// The KV transport a disaggregated mock worker advertises on every wire (as
/// a transfer backend, or as the connector that carries it): the mock moves
/// no KV cache, and the gateway pairs its workers on this one transport.
pub(crate) const KV_TRANSPORT: &str = "mooncake";

/// The name of the scheduler service a worker of `wire` serves: what the
/// standard health service answers for, and what a gateway's detection asks.
pub(crate) fn service_name(wire: SchedulerWire) -> &'static str {
    match wire {
        SchedulerWire::TokenSpeed => <tokenspeed::Server as NamedService>::NAME,
        SchedulerWire::Vllm => <vllm::Server as NamedService>::NAME,
        SchedulerWire::Sglang => <sglang::Server as NamedService>::NAME,
    }
}

/// The service name the standard health service knows beside the empty
/// name: the scheduler's, or the encoder's for an encode worker on the
/// TokenSpeed wire, where the engine servicer reports health under the
/// encoder's name (a probe of the scheduler's name reads NOT_FOUND there, as
/// on the engine).
pub(crate) fn health_service_name(wire: SchedulerWire, role: Option<Role>) -> &'static str {
    match (wire, role) {
        (SchedulerWire::TokenSpeed, Some(Role::Encode)) => {
            <tokenspeed::EncoderServer as NamedService>::NAME
        }
        _ => service_name(wire),
    }
}

impl MockScheduler {
    /// The realistic engine's fault and admission hooks for a side that
    /// serves no tokens (the prefill role's prompt-side completion, the
    /// encode role's encoder completion), as the regular path counts them
    /// ([`Engine::admit_without_tokens`]), then the canned delay. `Ok(true)`
    /// when the fault asks for a `finish=error` completion, counted as a cut.
    pub(crate) async fn admit_without_tokens(
        &self,
        request_id: &str,
        pin: Option<i32>,
    ) -> Result<bool, Status> {
        let mut finish_error = false;
        if !self.ranks.is_empty() {
            let (rank, _placed) = self.place(pin)?;
            if self.ranks.len() > 1 {
                let line = rank_line(request_id, pin, rank);
                tracing::info!("{line}");
            }
            finish_error = self.ranks[rank]
                .engine
                .admit_without_tokens()
                .await
                .map_err(injected_status)?;
        }
        if !self.cfg.gen_delay.is_zero() {
            tokio::time::sleep(self.cfg.gen_delay).await;
        }
        Ok(finish_error)
    }

    /// A prefill or decode worker refuses a request without its KV rendezvous,
    /// as the engines' disaggregated servers do and as the HTTP wire refuses a
    /// request without its `bootstrap_room`: a request without one would pair
    /// nowhere. The wire says which sides the gateway names it to: both on
    /// TokenSpeed and SGLang, the decode side on vLLM, whose prefill request
    /// carries the connector's parameters only when a connector relays them.
    /// A named rendezvous needs a host and a port (a field the request
    /// carries emptied is refused under its own name). Its room may be 0:
    /// the gateway draws SGLang rooms from 0 and a TokenSpeed room residue
    /// can land on 0, and the engines take it. A prefill worker that bound
    /// a bootstrap port refuses one naming another port, where its decode
    /// partner would come. A regular worker takes any request. The check
    /// comes before the realistic engine's hooks: a request sent to the wrong
    /// pool spends no armed fault.
    fn check_rendezvous(&self, rendezvous: &Rendezvous) -> Result<(), Status> {
        let role = match self.cfg.role {
            Some(role @ (Role::Prefill | Role::Decode)) => role,
            Some(Role::Encode) | None => return Ok(()),
        };
        let wire = self.cfg.grpc_wire;
        let field = wire.rendezvous_field();
        match rendezvous {
            Rendezvous::None if role == Role::Prefill && !wire.prefill_takes_rendezvous() => Ok(()),
            Rendezvous::None => Err(Status::invalid_argument(format!(
                "mock-worker: a {} worker takes a disaggregated request with its {field}",
                role.as_str()
            ))),
            Rendezvous::Named { host, port, room } => {
                if host.is_empty() || *port <= 0 {
                    return Err(Status::invalid_argument(format!(
                        "mock-worker: {field} names no rendezvous (host {host:?}, port {port}, \
                         room {room})"
                    )));
                }
                match self.bootstrap_port {
                    Some(own) if role == Role::Prefill && i32::from(own) != *port => {
                        Err(Status::invalid_argument(format!(
                            "mock-worker: {field} names bootstrap port {port}, this prefill \
                             worker bound {own}"
                        )))
                    }
                    _ => Ok(()),
                }
            }
            Rendezvous::Emptied { .. }
                if role == Role::Prefill && !wire.prefill_takes_rendezvous() =>
            {
                Ok(())
            }
            Rendezvous::Emptied { field, host, port } => Err(Status::invalid_argument(format!(
                "mock-worker: {field} names no rendezvous (host {host:?}, port {port})"
            ))),
            Rendezvous::Opaque => Ok(()),
        }
    }

    /// The generate stream of one request, as every wire's `Generate` answers
    /// it, in wire-neutral frames. In flight from here until the stream is
    /// done (or dropped by the gateway), so a shutdown knows what it waits
    /// for and this worker's load report counts it.
    pub(crate) async fn generate_frames(
        &self,
        input: GenerateInput,
    ) -> Result<FrameStream, Status> {
        self.check_rendezvous(&input.rendezvous)?;
        let in_flight = InFlight::enter_on(&self.in_flight);

        // A prefill worker's answer: the prompt side's completion alone,
        // ahead of the script and of either engine mode. The decode side of
        // the pair has the prompt and generates the output, so nothing is
        // scripted here and nothing counts as a miss; the canned delay paces
        // it as it paces a canned answer. The realistic engine's fault and
        // admission hooks apply as to any request (`admit_without_tokens`):
        // `finish=error` ends the prompt side with `error` whatever the
        // status; with no output to cut after, a status cut fires at once.
        if self.cfg.role == Some(Role::Prefill) {
            let finish_reason = if self
                .admit_without_tokens(&input.request_id, input.pin)
                .await?
            {
                "error"
            } else {
                "stop"
            };
            let complete = Frame::Complete(Complete {
                output_ids: Vec::new(),
                finish_reason: finish_reason.to_string(),
                prompt_tokens: u32::try_from(input.prompt_ids.len()).unwrap_or(u32::MAX),
                completion_tokens: 0,
                cached_tokens: 0,
                logprobs: None,
                matched_stop: None,
                index: 0,
            });
            return Ok(Box::pin(in_flight.until_done(stream::iter([Ok(complete)]))));
        }

        // A scripted answer, ahead of either engine mode: the worker's queue
        // or the prompt's key decides ([`crate::script`]), one answer per
        // choice the request samples; the rendered prompt the gateway sends
        // beside the ids serves the answers filed under a text (read without
        // a tokenizer).
        let scripted = script::take_ids_per_choice(
            &self.cfg,
            &self.name,
            &input.prompt_ids,
            &input.rendered,
            input.choices,
        )
        .map_err(Status::failed_precondition)?;
        if let Some(answers) = scripted {
            // The request's limit ends each answer as it would an engine's.
            let answers = answers
                .into_iter()
                .map(|answer| answer.cut_to(input.max_new))
                .collect();
            let stream = scripted_stream(
                answers,
                input.prompt_ids.len(),
                input.stream,
                input.logprobs,
                input.score_ids,
            );
            return Ok(Box::pin(in_flight.until_done(stream)));
        }

        // Realistic mode: place the request on a rank, submit it to that
        // rank's engine simulator and stream its output.
        if !self.ranks.is_empty() {
            // An `n` above the gateway's own bound is refused before the
            // request touches a rank or the fault hook.
            if input.choices > 128 {
                return Err(Status::invalid_argument(format!(
                    "n {} is above the gateway's bound of 128 choices",
                    input.choices
                )));
            }
            let (rank, placed) = self.place(input.pin)?;
            if self.ranks.len() > 1 {
                let line = rank_line(&input.request_id, input.pin, rank);
                tracing::info!("{line}");
            }
            let rank = &self.ranks[rank];
            let engine = &rank.engine;
            // The fault hook's answer for this request, when one is armed: a
            // stall, a status before admission, or a cut after some output.
            let injected = engine.inject();
            if let Some(fault) = injected {
                if !fault.stall.is_zero() {
                    tokio::time::sleep(fault.stall).await;
                }
                if fault.status != 0 && fault.after_tokens.is_none() {
                    engine.record_failure();
                    return Err(injected_status(fault.status));
                }
            }
            let cut = injected.and_then(|fault| {
                let end = if fault.finish_error {
                    CutEnd::EngineError
                } else {
                    CutEnd::Status(injected_status(fault.status))
                };
                fault
                    .after_tokens
                    .map(|after| (after, end, engine.cut_counter()))
            });
            // Omitted limit falls back to the worker default, matching the HTTP
            // path; `unwrap_or(0)` here would make unbounded requests generate
            // nothing (zero tokens), starving the routing signals.
            let max_new = input.max_new.unwrap_or(self.cfg.output_tokens);
            // One engine request per choice: the realistic branch samples `n`
            // outputs as the scripted path does, each under its own limit and
            // its own seed (the choice salts it), and streams them one choice
            // after another with the index on every frame. The admission hold
            // is paid once; each choice counts its prompt as pending prefill
            // from its submission until the engine's first token for it,
            // relayed as it arrives whatever the choice's place in the
            // streaming order: a queued choice keeps the count up while it
            // waits, a choice prefilled beside the one streaming does not
            // (the liveness bound reads that count).
            // The choices share the request id: an abort ends every one of
            // them, a dropped stream releases them all at the next pass. The
            // first cut of the fault hook, of either kind, ends the whole stream.
            let choices = u32::try_from(input.choices.max(1)).unwrap_or(128);
            let mut relays = Vec::with_capacity(choices as usize);
            let mut streams = VecDeque::with_capacity(choices as usize);
            // The admission hooks: the request is held here, between the
            // gateway and the engine, as a backlog in transit would hold it;
            // the load record does not see it until it lands.
            engine.admit().await;
            let prompt_tokens = input.prompt_ids.len() as u64;
            for index in 0..choices {
                let (to_engine, from_engine) = mpsc::unbounded_channel();
                let (to_frames, frames) = mpsc::unbounded_channel();
                rank.prefill_pending
                    .fetch_add(prompt_tokens, Ordering::Relaxed);
                engine.submit(NewRequest {
                    request_id: input.request_id.clone(),
                    prompt_token_ids: input.prompt_ids.clone(),
                    max_new,
                    choice: index,
                    events: to_engine,
                });
                relays.push(ChoiceRelay {
                    from_engine: Some(from_engine),
                    to_frames: Some(to_frames),
                    share: Some((Arc::clone(&rank.prefill_pending), prompt_tokens)),
                });
                streams.push_back(generate_stream(frames, input.stream, cut.clone(), index));
            }
            let stream = stream::unfold(
                (
                    FanOut {
                        relays,
                        streams,
                        current: None,
                    },
                    false,
                ),
                |(mut rest, cut)| async move {
                    // Read before the next choice is polled: polling it would
                    // run that choice to its own cut, counted once more.
                    if cut {
                        return None;
                    }
                    let item = rest.next().await?;
                    let cut = match &item {
                        Err(_) => true,
                        Ok(Frame::Complete(complete)) => complete.finish_reason == "error",
                        Ok(_) => false,
                    };
                    Some((item, (rest, cut)))
                },
            )
            // The rank's count rides with the stream: down when it is dropped.
            .map(move |item| {
                let _placed = &placed;
                item
            });
            return Ok(Box::pin(in_flight.until_done(stream)));
        }

        // Canned mode: a single up-front delay, then synthetic token ids.
        if !self.cfg.gen_delay.is_zero() {
            tokio::time::sleep(self.cfg.gen_delay).await;
        }
        // The configured prefix first, then synthetic ids.
        let ids: Vec<u32> = (0..self.cfg.output_tokens)
            .map(|i| {
                self.cfg
                    .engine
                    .output_prefix
                    .get(i as usize)
                    .copied()
                    .unwrap_or(100 + i)
            })
            .collect();
        let mut items: Vec<Result<Frame, Status>> = Vec::new();
        for id in &ids {
            items.push(Ok(Frame::Chunk(Chunk {
                token_ids: vec![*id],
                prompt_tokens: 1,
                completion_tokens: 1,
                cached_tokens: 0,
                logprobs: None,
                index: 0,
            })));
        }
        items.push(Ok(Frame::Complete(Complete {
            output_ids: ids,
            finish_reason: "stop".to_string(),
            prompt_tokens: 1,
            completion_tokens: self.cfg.output_tokens,
            cached_tokens: 0,
            logprobs: None,
            matched_stop: None,
            index: 0,
        })));
        Ok(Box::pin(in_flight.until_done(stream::iter(items))))
    }

    /// The admin fan-out's flush: every rank's cache cleared at once, whatever
    /// is in flight (the request's wait for an idle scheduler is not
    /// simulated), answered with success; a canned worker has no cache and
    /// answers the same. One answer for the three wires, which all carry the
    /// common proto's messages.
    pub(crate) fn flush_cache_response(&self) -> common::FlushCacheResponse {
        let message = self.ops.flush(self.ranks.iter().map(|rank| &rank.engine));
        common::FlushCacheResponse {
            success: true,
            message,
        }
    }

    /// A profiling run starts with the request's options; a second start
    /// while one runs answers `success: false`, as an engine's profiler does.
    pub(crate) fn start_profile_response(
        &self,
        request: common::StartProfileRequest,
    ) -> common::ProfileResponse {
        let options = ProfileOptions {
            output_dir: request.output_dir,
            start_step: request.start_step,
            num_steps: request.num_steps,
            activities: request.activities,
            with_stack: request.with_stack,
            record_shapes: request.record_shapes,
            profile_by_stage: request.profile_by_stage,
        };
        profile_response(self.ops.start_profile(options))
    }

    /// The run ends with one trace file in its output directory; no run in
    /// progress answers `success: false`.
    pub(crate) fn stop_profile_response(&self) -> common::ProfileResponse {
        profile_response(self.ops.stop_profile())
    }

    /// Every wire's `HealthCheck` answer: healthy unless the admin API set
    /// the worker not serving, with the message each proto carries.
    pub(crate) fn health(&self) -> (bool, &'static str) {
        let serving = self.overrides.serving();
        (serving, if serving { "ok" } else { "not serving" })
    }

    /// Abort `request_id` on every rank: the abort does not name the
    /// request's rank, and an unknown id is harmless.
    pub(crate) fn abort_everywhere(&self, request_id: &str) {
        for rank in &self.ranks {
            rank.engine.abort(request_id.to_string());
        }
    }

    /// The data-parallel width the worker advertises: its ranks, one for a
    /// canned worker.
    pub(crate) fn dp_size(&self) -> usize {
        self.ranks.len().max(1)
    }

    /// The port this worker serves on, read off its fleet name
    /// (`grpc:<port>`): what an encode worker names as its own address in
    /// the transfer handles it hands out.
    pub(crate) fn port(&self) -> u16 {
        self.name
            .rsplit(':')
            .next()
            .and_then(|port| port.parse().ok())
            .unwrap_or(0)
    }

    /// The `server_args` of the wires whose server info carries a struct of
    /// them: the configured args, with the admin API's overrides on top (what
    /// the gateway flattens into this worker's labels). A multi-rank worker
    /// advertises its width, as an engine's server args carry it; a
    /// disaggregated worker advertises its role and, on the prefill side, the
    /// port its partners rendezvous on, as the engines' server args carry
    /// them; the pairing protocol as the engine servicers advertise theirs.
    pub(crate) fn server_args(&self) -> Option<prost_types::Struct> {
        let string = |value: String| prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(value)),
        };
        let mut fields: std::collections::BTreeMap<String, prost_types::Value> = self
            .cfg
            .server_args
            .iter()
            .map(|(k, v)| (k.clone(), string(v.clone())))
            .collect();
        // The admin API's info patch may still override the width.
        if self.ranks.len() > 1 {
            fields.insert("dp_size".to_string(), string(self.ranks.len().to_string()));
        }
        if let Some(role) = self.cfg.role {
            fields.insert(
                "disaggregation_mode".to_string(),
                string(role.as_str().to_string()),
            );
            // The mock moves no KV cache; it names the one transport the
            // gateway pairs its workers on, as the engines name theirs.
            fields.insert(
                "disaggregation_transfer_backend".to_string(),
                string(KV_TRANSPORT.to_string()),
            );
            if let Some(port) = self.bootstrap_port {
                fields.insert(
                    "disaggregation_bootstrap_port".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::NumberValue(f64::from(port))),
                    },
                );
            }
        }
        if let Some(protocol) = &self.cfg.pairing_protocol {
            fields.insert("pairing_protocol".to_string(), string(protocol.clone()));
        }
        fields.extend(crate::overrides::server_args(&self.overrides.info()).fields);
        (!fields.is_empty()).then_some(prost_types::Struct { fields })
    }

    /// One load row per rank, the request's `dp_rank` keeping that rank's
    /// alone, as the servicers answer; each wire shapes the rows into its
    /// `SchedulerLoad`. A canned worker is one rank whose running count is
    /// the canned and scripted answers with a stream still open on this
    /// worker (not the process's other workers); nothing waits there.
    pub(crate) fn load_rows(&self, wanted: Option<i32>) -> Vec<(i32, engine::LoadSnapshot)> {
        self.overrides.note_loads_poll();
        let mut rows: Vec<(i32, engine::LoadSnapshot)> = if self.ranks.is_empty() {
            let running = i32::try_from(self.in_flight.get()).unwrap_or(i32::MAX);
            vec![(
                0,
                engine::LoadSnapshot {
                    num_running_reqs: running,
                    num_waiting_reqs: 0,
                    num_waiting_uncached_tokens: 0,
                    num_used_tokens: 0,
                    max_total_num_tokens: 1_000_000,
                    max_running_requests: 0,
                    token_usage: 0.0,
                    gen_throughput: 0.0,
                    prefill_throughput: None,
                    cache_hit_rate: 0.0,
                    num_cached_blocks: 0,
                    num_preemptions: 0,
                    num_kv_batches: 0,
                },
            )]
        } else {
            self.ranks
                .iter()
                .enumerate()
                .map(|(index, rank)| {
                    (
                        i32::try_from(index).unwrap_or(i32::MAX),
                        rank.engine.load().as_reported_by(self.cfg.loads_like),
                    )
                })
                .collect()
        };
        if let Some(wanted) = wanted {
            rows.retain(|(rank, _)| *rank == wanted);
        }
        rows
    }

    /// The KV-event subscription every wire serves alike (the batches are the
    /// shared proto's). Realistic mode with prefix caching: the ranks' KV
    /// events on one stream, each batch stamped with its rank and carrying
    /// the load record a servicer attaches for that rank; one record state
    /// over the merged stream, as the Rust relay keeps one per subscription
    /// (its heartbeat names the last batch's rank).
    pub(crate) fn kv_events(
        &self,
        request: common::SubscribeKvEventsRequest,
    ) -> Result<KvEventStream, Status> {
        if request.loads_only {
            // The load records alone, every rank's at the record's cadence,
            // from the ranks' engines whether or not they publish KV events:
            // a servicer's own figures need no relay.
            let sources: Vec<RankLoadSource> = self.ranks.iter().map(RankLoadSource::of).collect();
            if sources.is_empty() {
                return Err(Status::unimplemented(
                    "mock-worker (load records require --engine realistic)",
                ));
            }
            let records = load_records_only(sources, self.cfg.loads_like, self.cfg.step_reports);
            let mut stopped = self.stopped.clone();
            let stop = async move {
                let _ = stopped.wait_for(|stop| *stop).await;
            };
            return Ok(Box::pin(records.take_until(stop)));
        }
        let sources: Vec<RankLoadSource> = self
            .ranks
            .iter()
            .filter(|rank| rank.engine.kv_enabled())
            .map(|rank| RankLoadSource {
                engine: rank.engine.clone(),
                prefill_pending: Arc::clone(&rank.prefill_pending),
            })
            .collect();
        if sources.is_empty() {
            // Unimplemented, on which the gateway's KvEventMonitor gives up
            // cleanly instead of keeping an idle per-worker task.
            return Err(Status::unimplemented(
                "mock-worker (KV events require --engine realistic with --prefix-cache true)",
            ));
        }
        if sources.len() > 1 && request.start_sequence_number != 0 {
            // One cursor names one rank's sequence (the gateway resumes with
            // rank 0's). The relay refuses it over several ranks and the
            // gateway comes back from zero; resuming every rank at that
            // cursor would skip the other ranks' batches below it in silence.
            return Err(Status::out_of_range(format!(
                "mock-worker: {} ranks and a cursor names one rank's sequence; \
                 resubscribe from zero",
                sources.len()
            )));
        }
        let streams: Vec<KvEventStream> = sources
            .iter()
            .map(|source| source.engine.subscribe_kv(request.start_sequence_number))
            .collect();
        let events = with_load_records(
            sources,
            self.cfg.loads_like,
            self.cfg.step_reports,
            request.load_records_before_first_batch,
            Box::pin(stream::select_all(streams)),
        );
        // A subscription ends with the worker's stop, as a serving engine's
        // ends with its process; nothing else ends it, and the server's
        // shutdown waits for it.
        let mut stopped = self.stopped.clone();
        let stop = async move {
            let _ = stopped.wait_for(|stop| *stop).await;
        };
        Ok(Box::pin(events.take_until(stop)))
    }
}

#[cfg(test)]
impl MockScheduler {
    /// A scheduler as `serve_with_listener_until` builds one for `cfg` on
    /// `port`, without the server: its ranks spawned, its name registered for
    /// scripted answers, no capture, the configured bootstrap base port as its
    /// own. The wire modules' tests drive it through their trait.
    pub(super) fn for_tests(cfg: Config, port: u16) -> (Self, watch::Sender<bool>) {
        let cfg = Arc::new(cfg);
        let name = format!("grpc:{port}");
        script::register(&name);
        let (stop, stopped) = watch::channel(false);
        let service = MockScheduler {
            ranks: Rank::spawn_all(&cfg, &name),
            scan_start: Arc::new(AtomicUsize::new(0)),
            capture: None,
            stopped,
            in_flight: WorkerInFlight::default(),
            overrides: crate::overrides::for_port(port),
            bootstrap_port: cfg.bootstrap_base_port,
            ops: AdminOps::register(&name),
            name,
            cfg,
        };
        (service, stop)
    }
}

/// `Health.Watch` poll interval: the override is read, not signalled.
const HEALTH_WATCH_INTERVAL: Duration = Duration::from_millis(250);

type HealthWatchStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

/// The standard `grpc.health.v1` service of a gRPC worker: the whole server
/// (the empty name) and the one service it names (the scheduler's, or the
/// encoder's in the TokenSpeed encode role, as the engine servicer names it)
/// are SERVING exactly when the scheduler's own `HealthCheck` answers
/// healthy, i.e. unless the admin API set the worker not serving
/// (`/admin/serving/{port}`); any other name is unknown. It is the service a
/// harness or a probe asks before it asks the engine's own protocol, as every
/// engine servicer hosts it.
#[derive(Clone)]
struct MockHealth {
    overrides: Arc<crate::overrides::WorkerOverrides>,
    /// Flips when the worker stops: ends the watch streams, which have no
    /// end of their own, so the server's drain can finish.
    stopped: watch::Receiver<bool>,
    /// The one service name known beside the empty one
    /// ([`health_service_name`]).
    service: &'static str,
}

impl MockHealth {
    /// The serving status of `service`, read from the override per probe.
    fn status(&self, service: &str) -> ServingStatus {
        if !service.is_empty() && service != self.service {
            ServingStatus::ServiceUnknown
        } else if self.overrides.serving() {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        }
    }
}

#[tonic::async_trait]
impl Health for MockHealth {
    type WatchStream = HealthWatchStream;

    /// An unknown service is NOT_FOUND, as the health protocol specifies.
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let service = request.into_inner().service;
        match self.status(&service) {
            ServingStatus::ServiceUnknown => {
                Err(Status::not_found(format!("unknown service {service:?}")))
            }
            status => Ok(Response::new(HealthCheckResponse {
                status: status as i32,
            })),
        }
    }

    /// The current status first, then every change; an unknown service is
    /// SERVICE_UNKNOWN on the stream rather than an error, as the health
    /// protocol specifies. The stream ends when the worker stops (a known
    /// name reads NOT_SERVING once more first): the server's drain waits for
    /// every open stream, and a client keeps a watch open for as long as it
    /// lives.
    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let service = request.into_inner().service;
        let known = self.status(&service) != ServingStatus::ServiceUnknown;
        let stream = stream::unfold(
            (self.clone(), service, None, self.stopped.clone(), false),
            move |(health, service, last, mut stopped, ended)| async move {
                if ended {
                    return None;
                }
                loop {
                    let status = health.status(&service) as i32;
                    if last != Some(status) {
                        return Some((
                            Ok(HealthCheckResponse { status }),
                            (health, service, Some(status), stopped, false),
                        ));
                    }
                    let stopping = tokio::select! {
                        _ = tokio::time::sleep(HEALTH_WATCH_INTERVAL) => false,
                        _ = stopped.wait_for(|stop| *stop) => true,
                    };
                    if stopping {
                        let not_serving = ServingStatus::NotServing as i32;
                        if known && last != Some(not_serving) {
                            return Some((
                                Ok(HealthCheckResponse {
                                    status: not_serving,
                                }),
                                (health, service, Some(not_serving), stopped, true),
                            ));
                        }
                        return None;
                    }
                }
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}

/// The gRPC status code closest to the HTTP status a request fault answers,
/// chosen so the gateway's mapping of codes to HTTP statuses gives that status
/// back where it can: 408 reads back as 504 and 502 as 503 (both still
/// retryable), 499 as 502 (`cancelled`: the worker's side ending the call,
/// retryable and charged to the worker), 422 as the engine's report that it
/// could not continue the request under its constraint (a 422
/// `constraint_failed` at the gateway, never replayed), any other 4xx as 400
/// (not retryable, as on the HTTP worker), any other 5xx as 500.
fn injected_status(status: u16) -> Status {
    const MESSAGE: &str = "injected";
    match status {
        400 => Status::invalid_argument(MESSAGE),
        401 => Status::unauthenticated(MESSAGE),
        403 => Status::permission_denied(MESSAGE),
        404 => Status::not_found(MESSAGE),
        408 | 504 => Status::deadline_exceeded(MESSAGE),
        409 => Status::aborted(MESSAGE),
        422 => constraint_failed_status(MESSAGE),
        429 => Status::resource_exhausted(MESSAGE),
        // The call the worker's side ended: what a connection that closed
        // under it produces at the gateway.
        499 => Status::cancelled(MESSAGE),
        501 => Status::unimplemented(MESSAGE),
        502 | 503 => Status::unavailable(MESSAGE),
        other if (400..=499).contains(&other) => Status::invalid_argument(MESSAGE),
        _ => Status::internal(MESSAGE),
    }
}

/// How the fault hook's cut ends a stream.
#[derive(Clone, Debug)]
enum CutEnd {
    /// The gRPC status the stream ends with.
    Status(Status),
    /// A `Complete` whose finish reason is `error`, as an engine that failed
    /// the request sends it: the output so far, no status.
    EngineError,
}

/// The generate stream's state between items.
struct Generating {
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    output_ids: Vec<u32>,
    stream_chunks: bool,
    /// The fault hook's cut: the stream ends with the given end once the
    /// engine has produced more than this many tokens; the counter records
    /// the cut.
    cut: Option<(u32, CutEnd, Arc<AtomicU64>)>,
    /// The choice this stream is, on every frame it emits.
    index: u32,
    ended: bool,
}

/// One choice of a fanned-out request between the engine and its frame
/// stream. The engine's events are relayed as they arrive, whatever the
/// choice's place in the streaming order, so the choice's share of the rank's
/// pending prefill count settles at the engine's first token for it (or at
/// the engine's end, or at the drop), not when its turn to stream comes.
struct ChoiceRelay {
    /// The engine's events for this choice; `None` once the engine is done.
    from_engine: Option<mpsc::UnboundedReceiver<engine::GenEvent>>,
    /// The frame stream's side, dropped with the engine's end so the stream
    /// sees it.
    to_frames: Option<mpsc::UnboundedSender<engine::GenEvent>>,
    /// The rank's pending-prefill counter and this choice's share of it (its
    /// own prompt), settled once.
    share: Option<(Arc<AtomicU64>, u64)>,
}

impl ChoiceRelay {
    /// Relays what the engine has produced so far: a first token settles the
    /// share, the engine's end closes the frame stream's channel.
    fn pump(&mut self, cx: &mut Context<'_>) {
        loop {
            let polled = match self.from_engine.as_mut() {
                Some(from_engine) => from_engine.poll_recv(cx),
                None => return,
            };
            match polled {
                Poll::Ready(Some(event)) => {
                    if matches!(event, engine::GenEvent::Token { .. }) {
                        self.settle();
                    }
                    if let Some(to_frames) = &self.to_frames {
                        // A frame stream that ended early (a cut) reads
                        // nothing more; its events have nowhere to go.
                        let _ = to_frames.send(event);
                    }
                }
                Poll::Ready(None) => {
                    self.settle();
                    self.from_engine = None;
                    self.to_frames = None;
                    return;
                }
                Poll::Pending => return,
            }
        }
    }

    /// The prompt is prefilled (the engine's first token, or its end): it no
    /// longer counts as pending prefill.
    fn settle(&mut self) {
        if let Some((counter, tokens)) = self.share.take() {
            counter.fetch_sub(tokens, Ordering::Relaxed);
        }
    }
}

impl Drop for ChoiceRelay {
    fn drop(&mut self) {
        self.settle();
    }
}

/// The choices of one request, streamed one after another in index order;
/// every poll relays every choice's engine events first, so a later choice's
/// prefill settles as the engine produces it while an earlier one streams.
struct FanOut {
    relays: Vec<ChoiceRelay>,
    /// The frame streams still to be read, in index order.
    streams: VecDeque<FrameStream>,
    /// The frame stream being read.
    current: Option<FrameStream>,
}

impl Stream for FanOut {
    type Item = <FrameStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        for relay in &mut this.relays {
            relay.pump(cx);
        }
        loop {
            if this.current.is_none() {
                this.current = this.streams.pop_front();
            }
            let Some(current) = this.current.as_mut() else {
                return Poll::Ready(None);
            };
            match current.poll_next_unpin(cx) {
                Poll::Ready(Some(item)) => return Poll::Ready(Some(item)),
                Poll::Ready(None) => this.current = None,
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Map the engine's [`engine::GenEvent`] channel to the generate stream's
/// frames. In streaming mode each token becomes a chunk; otherwise tokens
/// are accumulated and only the final completion is sent. After the engine's
/// `Done` it has dropped the sender, so the next `recv()` yields `None` and
/// the stream ends. With `cut`, the stream ends with the given status or
/// engine-failure completion instead once the engine has produced more than
/// the given number of tokens (the fault hook's `after_tokens`); an output no
/// longer than that completes as usual.
fn generate_stream(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    stream_chunks: bool,
    cut: Option<(u32, CutEnd, Arc<AtomicU64>)>,
    index: u32,
) -> FrameStream {
    let init = Generating {
        rx,
        output_ids: Vec::new(),
        stream_chunks,
        cut,
        index,
        ended: false,
    };
    Box::pin(stream::unfold(init, |mut st| async move {
        if st.ended {
            return None;
        }
        loop {
            match st.rx.recv().await {
                Some(engine::GenEvent::Token {
                    token_id,
                    prompt_tokens,
                    cached_tokens,
                }) => {
                    st.output_ids.push(token_id);
                    if let Some((after, end, cut_total)) = &st.cut {
                        if st.output_ids.len() > *after as usize {
                            let item = match end {
                                CutEnd::Status(status) => Err(status.clone()),
                                CutEnd::EngineError => {
                                    // The output the engine had produced
                                    // before it failed: the tokens it sent.
                                    st.output_ids.truncate(*after as usize);
                                    Ok(Frame::Complete(Complete {
                                        output_ids: std::mem::take(&mut st.output_ids),
                                        finish_reason: "error".to_string(),
                                        prompt_tokens,
                                        completion_tokens: *after,
                                        cached_tokens,
                                        logprobs: None,
                                        matched_stop: None,
                                        index: st.index,
                                    }))
                                }
                            };
                            cut_total.fetch_add(1, Ordering::Relaxed);
                            st.ended = true;
                            return Some((item, st));
                        }
                    }
                    if st.stream_chunks {
                        let chunk = Frame::Chunk(Chunk {
                            token_ids: vec![token_id],
                            prompt_tokens,
                            completion_tokens: st.output_ids.len() as u32,
                            cached_tokens,
                            logprobs: None,
                            index: st.index,
                        });
                        return Some((Ok(chunk), st));
                    }
                    // Non-streaming: keep accumulating until Done.
                }
                Some(engine::GenEvent::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                }) => {
                    let complete = Frame::Complete(Complete {
                        output_ids: std::mem::take(&mut st.output_ids),
                        finish_reason: finish_reason.to_string(),
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        logprobs: None,
                        matched_stop: None,
                        index: st.index,
                    });
                    return Some((Ok(complete), st));
                }
                None => return None,
            }
        }
    }))
}

/// The generate stream of the scripted answers of a request, one per choice
/// in order: a chunk per frame when the request streams, then the choice's
/// completion with its index, with the answer's pause before every frame but
/// the first; the logprobs ride along when the request asked for them
/// (`logprobs`, the alternatives per token). The completions that follow no
/// chunk are what a non-streaming request reads, as it reads the engine's.
fn scripted_stream(
    answers: Vec<Answer>,
    prompt_len: usize,
    stream_chunks: bool,
    logprobs: Option<usize>,
    score_ids: Vec<u32>,
) -> FrameStream {
    struct Choice {
        index: u32,
        answer: Answer,
        frames: std::collections::VecDeque<script::Frame>,
        /// Output tokens sent so far (the chunks' running `completion_tokens`).
        sent: u32,
        started: bool,
    }
    struct Scripted {
        choices: std::collections::VecDeque<Choice>,
        prompt_len: usize,
        stream_chunks: bool,
        logprobs: Option<usize>,
        score_ids: Vec<u32>,
    }
    let choices = answers
        .into_iter()
        .enumerate()
        .map(|(index, answer)| Choice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            frames: answer.token_frames().into(),
            answer,
            sent: 0,
            started: false,
        })
        .collect();
    let init = Scripted {
        choices,
        prompt_len,
        stream_chunks,
        logprobs,
        score_ids,
    };
    Box::pin(stream::unfold(init, |mut st| async move {
        loop {
            if st
                .choices
                .front()
                .is_some_and(|choice| choice.frames.is_empty())
            {
                // Every frame of the choice went out: its completion, then
                // the next choice from its first frame.
                let choice = st.choices.pop_front()?;
                let answer = &choice.answer;
                let mut logprobs = st.logprobs.and_then(|top| {
                    answer
                        .logprobs
                        .as_ref()
                        .filter(|logprobs| logprobs.len() == answer.output_ids.len())
                        .map(|logprobs| LogProbs {
                            token_logprobs: logprobs.clone(),
                            token_ids: answer.output_ids.clone(),
                            top: answer.top_logprobs(&answer.output_ids, logprobs, top),
                            scores: Vec::new(),
                        })
                });
                // A scoring answer: the score of each token id the request asked
                // for, in the request's order, on the completion. The engine
                // computes the row only for a request that asks for logprobs,
                // so one that does not is refused, as a wrong count is.
                if let Some(values) = &answer.token_id_logprobs {
                    if st.logprobs.is_none() {
                        let status = Status::failed_precondition(format!(
                            "the scripted answer scores {} token ids, the request asks for no logprobs",
                            values.len()
                        ));
                        return Some((Err(status), st));
                    }
                    if values.len() != st.score_ids.len() {
                        let status = Status::failed_precondition(format!(
                            "the scripted answer scores {} token ids, the request asks for {}",
                            values.len(),
                            st.score_ids.len()
                        ));
                        return Some((Err(status), st));
                    }
                    let scores = st
                        .score_ids
                        .iter()
                        .copied()
                        .zip(values.iter().copied())
                        .collect();
                    logprobs = Some(match logprobs {
                        Some(logprobs) => LogProbs { scores, ..logprobs },
                        None => LogProbs {
                            token_logprobs: Vec::new(),
                            token_ids: Vec::new(),
                            top: Vec::new(),
                            scores,
                        },
                    });
                }
                let complete = Frame::Complete(Complete {
                    output_ids: answer.output_ids.clone(),
                    finish_reason: answer.finish_reason.clone(),
                    prompt_tokens: answer.prompt_tokens(st.prompt_len),
                    completion_tokens: answer.completion_tokens(),
                    cached_tokens: answer.cached_tokens,
                    logprobs,
                    matched_stop: answer.matched_stop.clone(),
                    index: choice.index,
                });
                return Some((Ok(complete), st));
            }
            let choice = st.choices.front_mut()?;
            let frame = choice.frames.pop_front()?;
            if choice.started && !choice.answer.delay.is_zero() {
                tokio::time::sleep(choice.answer.delay).await;
            }
            choice.started = true;
            choice.sent += u32::try_from(frame.ids.len()).unwrap_or(u32::MAX);
            if !st.stream_chunks {
                continue;
            }
            let logprobs = st
                .logprobs
                .filter(|_| frame.logprobs.len() == frame.ids.len())
                .map(|top| LogProbs {
                    token_logprobs: frame.logprobs.clone(),
                    token_ids: frame.ids.clone(),
                    top: choice.answer.top_logprobs(&frame.ids, &frame.logprobs, top),
                    scores: Vec::new(),
                });
            let chunk = Frame::Chunk(Chunk {
                token_ids: frame.ids,
                prompt_tokens: choice.answer.prompt_tokens(st.prompt_len),
                completion_tokens: choice.sent,
                cached_tokens: choice.answer.cached_tokens,
                logprobs,
                index: choice.index,
            });
            return Some((Ok(chunk), st));
        }
    }))
}

/// The load record the engine's `GetLoads` figures make, as reported for
/// `loads_like` (the vLLM shape has no queued token-work).
fn load_record(
    snapshot: &engine::LoadSnapshot,
    like: LoadsLike,
    sample: u64,
) -> common::EngineLoad {
    let unsigned = |value: i32| u32::try_from(value).unwrap_or(0);
    common::EngineLoad {
        running_requests: unsigned(snapshot.num_running_reqs),
        waiting_requests: unsigned(snapshot.num_waiting_reqs),
        waiting_uncached_tokens: matches!(like, LoadsLike::Mock)
            .then(|| unsigned(snapshot.num_waiting_uncached_tokens)),
        token_usage: snapshot.token_usage,
        gen_throughput: snapshot.gen_throughput,
        prefill_throughput: snapshot.prefill_throughput,
        max_running_requests: unsigned(snapshot.max_running_requests),
        age_ms: 0,
        sample,
        load_only: false,
        // What both reports carry beyond the core (the mock has no
        // memory, queue, speculative, LoRA or disaggregation sections).
        cache_hit_rate: Some(snapshot.cache_hit_rate),
        num_used_tokens: Some(snapshot.num_used_tokens),
        max_total_num_tokens: Some(snapshot.max_total_num_tokens),
        ..Default::default()
    }
}

/// The worker's pending prefill as the record carries it.
fn pending_u32(counter: &AtomicU64) -> u32 {
    u32::try_from(counter.load(Ordering::Relaxed)).unwrap_or(u32::MAX)
}

/// The engine's silence on its in-flight requests, as the Rust servicer's
/// record carries it (`EngineLoad.engine_silence_ms`).
fn silence_ms(engine: &engine::Engine) -> Option<u32> {
    engine
        .output_silence()
        .map(|silence| u32::try_from(silence.as_millis()).unwrap_or(u32::MAX))
}

/// Whether a record moved enough from `last` to be worth a `load_only`
/// batch: the Rust relay's rule (any queue, running or window change, KV
/// usage by half a percent, either rate by 5 % or 50 tokens per second, the
/// measured input capacity appearing or expiring, the engine's silence by a
/// whole second).
fn load_changed(last: &common::EngineLoad, current: &common::EngineLoad) -> bool {
    let prefill_changed = match (last.prefill_throughput, current.prefill_throughput) {
        (Some(last), Some(current)) => throughput_changed(last, current),
        (None, None) => false,
        _ => true,
    };
    prefill_changed
        || last.running_requests != current.running_requests
        || last.waiting_requests != current.waiting_requests
        || last.waiting_uncached_tokens != current.waiting_uncached_tokens
        || last.max_running_requests != current.max_running_requests
        || (last.token_usage - current.token_usage).abs() > 0.005
        || throughput_changed(last.gen_throughput, current.gen_throughput)
        || last.engine_silence_ms.map(|ms| ms / 1000)
            != current.engine_silence_ms.map(|ms| ms / 1000)
        || last.prefill_pending_tokens != current.prefill_pending_tokens
}

/// Rate noise is no change; a move of 5 % or 50 tokens per second is.
fn throughput_changed(last: f64, current: f64) -> bool {
    let delta = (last - current).abs();
    delta > 50.0 || delta > 0.05 * last.max(current)
}

/// How often the stream checks the record while no batch flows, the silence
/// after which a heartbeat goes out, and the heartbeat interval once the
/// engine has been idle for two of them: the Rust relay's figures.
const LOAD_TICK: Duration = Duration::from_millis(100);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
const HEARTBEAT_BACKOFF: Duration = Duration::from_secs(5);

/// Where a rank's load record comes from: its engine and the prompt tokens
/// handed to it with no token yet.
struct RankLoadSource {
    engine: engine::Engine,
    prefill_pending: Arc<AtomicU64>,
}

impl RankLoadSource {
    fn of(rank: &Rank) -> Self {
        Self {
            engine: rank.engine.clone(),
            prefill_pending: Arc::clone(&rank.prefill_pending),
        }
    }

    /// The rank's current record, numbered `sample`: its engine's load as
    /// `GetLoads` reports it plus the liveness fields.
    fn record(&self, like: LoadsLike, step_reports: bool, sample: u64) -> common::EngineLoad {
        let snapshot = self.engine.load().as_reported_by(like);
        let mut record = load_record(&snapshot, like, sample);
        record.engine_silence_ms = silence_ms(&self.engine);
        record.engine_reports_steps = Some(step_reports);
        record.prefill_pending_tokens = Some(pending_u32(&self.prefill_pending));
        record
    }
}

/// One rank's clock on a records-only stream: what it last sent and when.
struct RankClock {
    last_record: Option<common::EngineLoad>,
    last_sent_at: Option<Instant>,
    idle_heartbeats: u32,
}

/// The records-only stream a `loads_only` subscriber gets, as the Rust
/// servicer's (`crates/engine_servicer/src/load_stream.rs`): every rank's
/// record at the first tick, then a rank's when its record moved or its
/// heartbeat is due, each rank on its own clock; `load_only`, numbered 0,
/// never an event.
struct LoadRecordsOnly {
    ranks: Vec<RankLoadSource>,
    clocks: Vec<RankClock>,
    like: LoadsLike,
    step_reports: bool,
    pending: std::collections::VecDeque<common::KvEventBatch>,
    next_tick: Instant,
    sample: u64,
}

impl LoadRecordsOnly {
    fn collect_due(&mut self, now: Instant) {
        for (index, (source, clock)) in self.ranks.iter().zip(&mut self.clocks).enumerate() {
            let current = source.record(self.like, self.step_reports, 0);
            let changed = clock
                .last_record
                .as_ref()
                .is_none_or(|last| load_changed(last, &current));
            let due_after = if clock.idle_heartbeats >= 2 {
                HEARTBEAT_BACKOFF
            } else {
                HEARTBEAT_INTERVAL
            };
            let silent_for = clock
                .last_sent_at
                .map_or(Duration::MAX, |at| now.saturating_duration_since(at));
            if !changed && silent_for < due_after {
                continue;
            }
            self.sample += 1;
            let mut record = current;
            record.sample = self.sample;
            record.load_only = true;
            clock.idle_heartbeats = if changed {
                0
            } else {
                clock.idle_heartbeats.saturating_add(1)
            };
            clock.last_sent_at = Some(now);
            clock.last_record = Some(record.clone());
            self.pending.push_back(common::KvEventBatch {
                sequence_number: 0,
                timestamp: engine::unix_seconds(),
                events: Vec::new(),
                dp_rank: Some(i32::try_from(index).unwrap_or(i32::MAX)),
                snapshot: None,
                load: Some(record),
            });
        }
    }

    async fn next(&mut self) -> common::KvEventBatch {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return batch;
            }
            tokio::time::sleep_until(self.next_tick.into()).await;
            let now = Instant::now();
            self.next_tick = now + LOAD_TICK;
            self.collect_due(now);
        }
    }
}

fn load_records_only(
    ranks: Vec<RankLoadSource>,
    like: LoadsLike,
    step_reports: bool,
) -> KvEventStream {
    let records = LoadRecordsOnly {
        clocks: ranks
            .iter()
            .map(|_| RankClock {
                last_record: None,
                last_sent_at: None,
                idle_heartbeats: 0,
            })
            .collect(),
        ranks,
        like,
        step_reports,
        pending: std::collections::VecDeque::new(),
        next_tick: Instant::now(),
        sample: 0,
    };
    Box::pin(stream::unfold(records, |mut records| async move {
        let batch = records.next().await;
        Some((Ok(batch), records))
    }))
}

struct LoadRecords {
    stream: KvEventStream,
    /// The ranks' sources, by rank; a batch's record is its rank's.
    ranks: Vec<RankLoadSource>,
    like: LoadsLike,
    /// What the record says of the wire (`--step-reports`).
    step_reports: bool,
    /// The subscriber takes `load_only` batches before the stream's first
    /// batch, numbered 0 (`SubscribeKvEventsRequest.
    /// load_records_before_first_batch`); unset, it waits for that batch.
    records_before_first_batch: bool,
    /// The last sequence sent, which a `load_only` batch repeats; `None`
    /// before the stream's first batch.
    last_seq: Option<u64>,
    last_rank: Option<i32>,
    last_record: Option<common::EngineLoad>,
    last_sent_at: Instant,
    idle_heartbeats: u32,
    sample: u64,
}

#[expect(
    clippy::large_enum_variant,
    reason = "the item is matched and moved out at once; boxing it would allocate per batch"
)]
enum Waited {
    Item(Option<Result<common::KvEventBatch, Status>>),
    Tick,
}

impl LoadRecords {
    /// The source of `rank`'s record: rank 0's for a batch that names none
    /// or one past the ranks.
    fn source(&self, rank: Option<i32>) -> &RankLoadSource {
        rank.and_then(|rank| usize::try_from(rank).ok())
            .and_then(|rank| self.ranks.get(rank))
            .unwrap_or(&self.ranks[0])
    }

    /// The current record of `rank`, numbered `sample`.
    fn current(&self, rank: Option<i32>, sample: u64) -> common::EngineLoad {
        self.source(rank)
            .record(self.like, self.step_reports, sample)
    }

    fn record(&mut self, rank: Option<i32>, load_only: bool) -> common::EngineLoad {
        self.sample += 1;
        let mut record = self.current(rank, self.sample);
        record.load_only = load_only;
        // As the Rust relay: telemetry on heartbeats and the first record.
        if !load_only && self.sample > 1 {
            smg_grpc_client::engine_load::core_only(&mut record);
        }
        self.last_record = Some(record.clone());
        record
    }

    /// A `load_only` batch when the record moved since the last one sent, or
    /// after the heartbeat interval of silence (the backoff interval after two
    /// unchanged heartbeats); it repeats the last sequence sent, as the
    /// gateway expects. Before the stream's first batch there is none to
    /// repeat: as the Rust relay, the mock sends one numbered 0 only to a
    /// subscriber that asked for records before the first batch, and keeps
    /// an older subscriber waiting for that batch.
    fn load_only_batch(&mut self) -> Option<common::KvEventBatch> {
        let sequence_number = match self.last_seq {
            Some(sequence) => sequence,
            None if self.records_before_first_batch => 0,
            None => return None,
        };
        // The heartbeat is the last batch's rank's, as the relay's.
        let current = self.current(self.last_rank, 0);
        let changed = self
            .last_record
            .as_ref()
            .is_none_or(|last| load_changed(last, &current));
        let due_after = if self.idle_heartbeats >= 2 {
            HEARTBEAT_BACKOFF
        } else {
            HEARTBEAT_INTERVAL
        };
        if !changed && self.last_sent_at.elapsed() < due_after {
            return None;
        }
        let record = self.record(self.last_rank, true);
        self.last_sent_at = Instant::now();
        self.idle_heartbeats = if changed {
            0
        } else {
            self.idle_heartbeats.saturating_add(1)
        };
        Some(common::KvEventBatch {
            sequence_number,
            timestamp: engine::unix_seconds(),
            events: Vec::new(),
            dp_rank: self.last_rank,
            snapshot: None,
            load: Some(record),
        })
    }
}

/// `stream` (the ranks' batches, each stamped with its rank) with the
/// rank's load record on every batch and `load_only` batches while the
/// stream is quiet, as the Rust servicer's relay sends them
/// (`crates/engine_servicer/src/kv_events.rs`): one record state per
/// subscription, the heartbeat naming the last batch's rank (so a quiet
/// rank's records stop while another rank is active, as on the servicer);
/// from the subscription on for a subscriber that takes records before the
/// first batch (`records_before_first_batch`, rank 0's record then), from
/// the stream's first batch otherwise.
fn with_load_records(
    ranks: Vec<RankLoadSource>,
    like: LoadsLike,
    step_reports: bool,
    records_before_first_batch: bool,
    stream: KvEventStream,
) -> KvEventStream {
    assert!(!ranks.is_empty(), "a load record needs a rank's source");
    let records = LoadRecords {
        stream,
        ranks,
        like,
        step_reports,
        records_before_first_batch,
        last_seq: None,
        last_rank: Some(0),
        last_record: None,
        last_sent_at: Instant::now(),
        idle_heartbeats: 0,
        sample: 0,
    };
    Box::pin(stream::unfold(records, |mut records| async move {
        loop {
            let waited = tokio::select! {
                item = records.stream.next() => Waited::Item(item),
                () = tokio::time::sleep(LOAD_TICK) => Waited::Tick,
            };
            match waited {
                Waited::Item(None) => return None,
                Waited::Item(Some(Err(status))) => return Some((Err(status), records)),
                Waited::Item(Some(Ok(mut batch))) => {
                    records.last_seq = Some(batch.sequence_number);
                    records.last_rank = batch.dp_rank;
                    records.last_sent_at = Instant::now();
                    records.idle_heartbeats = 0;
                    batch.load = Some(records.record(batch.dp_rank, false));
                    return Some((Ok(batch), records));
                }
                Waited::Tick => {
                    if let Some(batch) = records.load_only_batch() {
                        return Some((Ok(batch), records));
                    }
                }
            }
        }
    }))
}

/// A profile call's answer: the engines report a refused call as
/// `success: false` with the reason, not as a status.
fn profile_response(result: Result<String, String>) -> common::ProfileResponse {
    match result {
        Ok(message) => common::ProfileResponse {
            success: true,
            message,
        },
        Err(message) => common::ProfileResponse {
            success: false,
            message,
        },
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
    use tonic::{transport::Channel, Code};
    use tonic_health::pb::health_client::HealthClient;
    use ts::{
        token_speed_scheduler_client::TokenSpeedSchedulerClient,
        token_speed_scheduler_server::TokenSpeedScheduler,
    };

    use super::*;

    /// The next item of a health watch, within the test's patience.
    async fn next(watch: &mut tonic::Streaming<HealthCheckResponse>) -> ServingStatus {
        tokio::time::timeout(Duration::from_secs(5), watch.message())
            .await
            .expect("a watch item in time")
            .expect("watch item")
            .expect("a stream item")
            .status()
    }

    /// A watch has no end of its own, so the worker's stop ends it: the
    /// server's drain finishes with a watch open, and the watcher reads
    /// NOT_SERVING once more, then the stream's end.
    #[tokio::test]
    async fn an_open_watch_ends_when_the_worker_stops() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins the server's task: its drain is what it checks"
        )]
        let server = tokio::spawn(serve_with_listener_until(
            Arc::new(Config::default()),
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let mut health = HealthClient::new(channel);
        let mut watch = health
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut watch).await, ServingStatus::Serving);

        stop.send(()).expect("the server waits for the stop");
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the drain finishes with a watch open")
            .expect("the server task");
        assert_eq!(next(&mut watch).await, ServingStatus::NotServing);
        let end = tokio::time::timeout(Duration::from_secs(5), watch.message())
            .await
            .expect("the watch ends in time")
            .expect("a clean end of the stream");
        assert!(end.is_none(), "the watch did not end: {end:?}");
    }

    /// The standard health service answers as the scheduler's own
    /// `HealthCheck`: SERVING for the whole server and for the scheduler
    /// service, NOT_FOUND for another name, NOT_SERVING on both once the admin
    /// override flips, and `Watch` reports the flip.
    #[tokio::test]
    async fn the_standard_health_service_follows_the_serving_override() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "the server serves until the test process exits"
        )]
        let _server = tokio::spawn(serve_with_listener(Arc::new(Config::default()), listener));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let mut health = HealthClient::new(channel.clone());
        let mut scheduler = TokenSpeedSchedulerClient::new(channel);
        let scheduler_service = service_name(SchedulerWire::TokenSpeed);

        // One `Check` on a clone of the client, so `health` stays free to watch.
        let probe = health.clone();
        let check = |service: &str| {
            let request = HealthCheckRequest {
                service: service.to_string(),
            };
            let mut probe = probe.clone();
            async move {
                probe
                    .check(request)
                    .await
                    .map(|response| response.into_inner().status())
            }
        };
        assert_eq!(check("").await.expect("check"), ServingStatus::Serving);
        assert_eq!(
            check(scheduler_service).await.expect("check"),
            ServingStatus::Serving
        );
        assert_eq!(
            check("not.a.Service").await.expect_err("unknown").code(),
            Code::NotFound
        );

        let mut watch = health
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut watch).await, ServingStatus::Serving);

        // The override map is process-wide and keyed by port: put the entry
        // back on the way out (a failed assertion included), or a later test
        // that draws the same ephemeral port would start out not serving.
        struct Restore(Arc<crate::overrides::WorkerOverrides>);
        impl Drop for Restore {
            fn drop(&mut self) {
                self.0.set_serving(true);
            }
        }
        let overrides = crate::overrides::for_port(addr.port());
        let _restore = Restore(Arc::clone(&overrides));
        overrides.set_serving(false);
        assert_eq!(check("").await.expect("check"), ServingStatus::NotServing);
        assert_eq!(
            check(scheduler_service).await.expect("check"),
            ServingStatus::NotServing
        );
        assert_eq!(next(&mut watch).await, ServingStatus::NotServing);
        let own = scheduler
            .health_check(ts::HealthCheckRequest::default())
            .await
            .expect("the scheduler's own health check")
            .into_inner();
        assert!(!own.healthy, "the two health answers disagree: {own:?}");

        let mut unknown = health
            .watch(HealthCheckRequest {
                service: "not.a.Service".to_string(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut unknown).await, ServingStatus::ServiceUnknown);
    }

    /// An encode worker on the TokenSpeed wire reports health under the
    /// encoder service's name, as the engine servicer does in that role: the
    /// scheduler's name is unknown there, and the encoder's name is unknown
    /// on a worker without the role.
    #[tokio::test]
    async fn the_tokenspeed_encode_role_reports_health_under_the_encoder_name() {
        let check = |addr: std::net::SocketAddr, service: &str| {
            let request = HealthCheckRequest {
                service: service.to_string(),
            };
            async move {
                let channel = Channel::from_shared(format!("http://{addr}"))
                    .expect("endpoint")
                    .connect()
                    .await
                    .expect("connect");
                HealthClient::new(channel)
                    .check(request)
                    .await
                    .map(|response| response.into_inner().status())
            }
        };
        let encoder_service = <tokenspeed::EncoderServer as NamedService>::NAME;
        let scheduler_service = service_name(SchedulerWire::TokenSpeed);
        let serve = |role: Option<Role>| async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let addr = listener.local_addr().expect("local addr");
            let cfg = Config {
                grpc_wire: SchedulerWire::TokenSpeed,
                role,
                ..Config::default()
            };
            #[expect(
                clippy::disallowed_methods,
                reason = "the server serves until the test process exits"
            )]
            let _server = tokio::spawn(serve_with_listener(Arc::new(cfg), listener));
            addr
        };

        let encode = serve(Some(Role::Encode)).await;
        assert_eq!(
            check(encode, "").await.expect("check"),
            ServingStatus::Serving
        );
        assert_eq!(
            check(encode, encoder_service).await.expect("check"),
            ServingStatus::Serving
        );
        assert_eq!(
            check(encode, scheduler_service)
                .await
                .expect_err("unknown")
                .code(),
            Code::NotFound
        );

        let plain = serve(None).await;
        assert_eq!(
            check(plain, scheduler_service).await.expect("check"),
            ServingStatus::Serving
        );
        assert_eq!(
            check(plain, encoder_service)
                .await
                .expect_err("unknown")
                .code(),
            Code::NotFound
        );
    }

    /// The sockets the gRPC worker serves carry `TCP_NODELAY`, so a token
    /// frame is never held back for the acknowledgement of the frame before.
    #[tokio::test]
    async fn served_connections_have_nodelay_set() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let mut incoming = std::pin::pin!(connections(listener));
        let client = TcpStream::connect(addr).await.expect("connect");
        let accepted = incoming
            .next()
            .await
            .expect("a connection")
            .expect("accepted");
        assert!(
            accepted.nodelay().expect("nodelay"),
            "the accepted socket runs with Nagle on"
        );
        drop(client);
    }

    fn token(id: u32) -> engine::GenEvent {
        engine::GenEvent::Token {
            token_id: id,
            prompt_tokens: 4,
            cached_tokens: 0,
        }
    }

    fn done(tokens: u32) -> engine::GenEvent {
        engine::GenEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 4,
            completion_tokens: tokens,
            cached_tokens: 0,
        }
    }

    fn chunk_ids(item: Option<Result<Frame, Status>>) -> Vec<u32> {
        match item.expect("an item").expect("ok") {
            Frame::Chunk(chunk) => chunk.token_ids,
            other => panic!("not a chunk: {other:?}"),
        }
    }

    /// A realistic scheduler of `ranks` ranks, as `serve_with_listener_until`
    /// builds one, with no listener behind it; the stop sender keeps the
    /// worker running.
    fn scheduler_with_ranks(ranks: u16, port: u16) -> (MockScheduler, watch::Sender<bool>) {
        MockScheduler::for_tests(
            Config {
                realistic: true,
                grpc_dp_size: ranks,
                engine: engine::EngineParams {
                    timing: engine::TimingModel::linear(),
                    ..Default::default()
                },
                ..Default::default()
            },
            port,
        )
    }

    fn generate_request(id: &str, pin: Option<i32>, max_new: u32) -> ts::GenerateRequest {
        ts::GenerateRequest {
            request_id: id.to_string(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: vec![11, 12, 13, 14],
                ..Default::default()
            }),
            sampling_params: Some(ts::SamplingParams {
                max_new_tokens: Some(max_new),
                ..Default::default()
            }),
            stream: true,
            data_parallel_rank: pin,
            ..Default::default()
        }
    }

    /// The engine the request was admitted on, from the process records.
    fn admitted_on(request_id: &str) -> String {
        engine::records_since(0, usize::MAX)
            .into_iter()
            .find(|record| record.request_id == request_id)
            .unwrap_or_else(|| panic!("{request_id} was admitted"))
            .worker
    }

    /// The admin fan-out's three RPCs on a canned worker, over the wire: the
    /// flush answers success, a profiling run ends with one trace file in the
    /// requested directory, a stop with no run in progress is `success:
    /// false`, and the admin operations count them.
    #[tokio::test]
    async fn the_admin_fan_out_rpcs_answer_on_a_canned_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "the server serves until the test process exits"
        )]
        let _server = tokio::spawn(serve_with_listener(Arc::new(Config::default()), listener));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let mut scheduler = TokenSpeedSchedulerClient::new(channel);

        let flushed = scheduler
            .flush_cache(common::FlushCacheRequest { timeout_s: 0.0 })
            .await
            .expect("flush")
            .into_inner();
        assert!(flushed.success, "{}", flushed.message);
        assert!(
            flushed.message.ends_with("cache cleared on 0 engine(s)"),
            "{}",
            flushed.message
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let traces = dir.path().join("traces");
        let started = scheduler
            .start_profile(common::StartProfileRequest {
                output_dir: Some(traces.display().to_string()),
                num_steps: Some(2),
                ..Default::default()
            })
            .await
            .expect("start")
            .into_inner();
        assert!(started.success, "{}", started.message);
        let stopped = scheduler
            .stop_profile(common::StopProfileRequest {})
            .await
            .expect("stop")
            .into_inner();
        assert!(stopped.success, "{}", stopped.message);
        let files: Vec<_> = std::fs::read_dir(&traces)
            .expect("the output directory")
            .map(|entry| entry.expect("entry").path())
            .collect();
        assert_eq!(files.len(), 1, "{}", stopped.message);
        assert!(stopped.message.ends_with(&files[0].display().to_string()));

        let again = scheduler
            .stop_profile(common::StopProfileRequest {})
            .await
            .expect("stop")
            .into_inner();
        assert!(!again.success, "a stop with no run in progress is refused");
        let status = AdminOps::find(&format!("grpc:{}", addr.port()))
            .expect("the worker's operations")
            .status();
        assert_eq!(status["flushes"], 1);
        assert_eq!(status["profiles_started"], 1);
        assert_eq!(status["profiles_stopped"], 1);
    }

    /// On a realistic worker the flush clears every rank's cache: the blocks
    /// a request stored are gone once the engines have run the reset pass.
    #[tokio::test]
    async fn a_flush_clears_every_rank_s_cache_on_a_realistic_worker() {
        let (service, _stop) = scheduler_with_ranks(2, 65_108);
        for (id, rank) in [("flush-r0", 0), ("flush-r1", 1)] {
            // A prompt of four blocks: a shorter one stores nothing to clear.
            let mut request = generate_request(id, Some(rank), 2);
            request.tokenized = Some(ts::TokenizedInput {
                input_ids: vec![3; 64],
                ..Default::default()
            });
            let mut stream = service
                .generate(Request::new(request))
                .await
                .expect("accepted")
                .into_inner();
            while stream.next().await.is_some() {}
        }
        let cached = |service: &MockScheduler| -> Vec<usize> {
            service
                .ranks
                .iter()
                .map(|rank| rank.engine.cache_keys().len())
                .collect()
        };
        assert!(
            cached(&service).iter().all(|blocks| *blocks > 0),
            "both ranks cached the prompt: {:?}",
            cached(&service)
        );
        let flushed = service
            .flush_cache(Request::new(common::FlushCacheRequest { timeout_s: 0.0 }))
            .await
            .expect("flush")
            .into_inner();
        assert!(flushed.success, "{}", flushed.message);
        assert!(
            flushed.message.ends_with("cache cleared on 2 engine(s)"),
            "{}",
            flushed.message
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while cached(&service).iter().any(|blocks| *blocks > 0) {
            assert!(
                Instant::now() < deadline,
                "the caches did not clear: {:?}",
                cached(&service)
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// A zero limit is a prefill-only request on the wire too: one complete
    /// frame, no output token, `length`, a zero completion count; not the
    /// worker's default length, which belongs to an omitted limit.
    #[tokio::test]
    async fn a_zero_limit_completes_with_no_output_token_on_a_realistic_worker() {
        let (service, _stop) = scheduler_with_ranks(1, 65_109);
        let mut stream = service
            .generate(Request::new(generate_request("score", None, 0)))
            .await
            .expect("accepted")
            .into_inner();
        let mut frames = Vec::new();
        while let Some(frame) = stream.next().await {
            frames.push(frame.expect("a frame"));
        }
        let [frame] = frames.as_slice() else {
            panic!("one complete frame, got {frames:?}");
        };
        let Some(ts::generate_response::Response::Complete(complete)) = &frame.response else {
            panic!("a complete frame, got {frame:?}");
        };
        assert!(complete.output_ids.is_empty(), "{complete:?}");
        assert_eq!(complete.finish_reason, "length");
        assert_eq!(complete.completion_tokens, 0);
        assert_eq!(complete.prompt_tokens, 4);
    }

    /// A pinned request runs on its rank's engine; a pin past the worker's
    /// ranks is the caller's error, as the servicers answer it.
    #[tokio::test]
    async fn a_pinned_request_runs_on_its_rank() {
        let (service, _stop) = scheduler_with_ranks(4, 65_101);
        let mut stream = service
            .generate(Request::new(generate_request("dp-pin-2", Some(2), 2)))
            .await
            .expect("accepted")
            .into_inner();
        while stream.next().await.is_some() {}
        assert_eq!(admitted_on("dp-pin-2"), "grpc:65101:r2");
        let refused = match service
            .generate(Request::new(generate_request("dp-pin-4", Some(4), 2)))
            .await
        {
            Err(status) => status,
            Ok(_) => panic!("a pin past the ranks was accepted"),
        };
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(
            refused.message().contains("out of range [0, 4)"),
            "{refused}"
        );
    }

    /// `n` samples on the realistic branch: one engine request per choice,
    /// streamed one choice after another with its index on every frame,
    /// each cut by its own limit and each with its own output, the shape
    /// the scripted path has.
    #[tokio::test]
    async fn two_choices_stream_one_after_the_other_with_their_indexes_on_a_realistic_worker() {
        let (service, _stop) = scheduler_with_ranks(1, 65_110);
        let mut request = generate_request("two-choices", None, 2);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(2),
            n: 2,
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        let mut frames = Vec::new();
        while let Some(frame) = stream.next().await {
            frames.push(frame.expect("a frame"));
        }
        let mut indexes = Vec::new();
        let mut completes = Vec::new();
        let mut outputs = Vec::new();
        for frame in &frames {
            match &frame.response {
                Some(ts::generate_response::Response::Chunk(chunk)) => indexes.push(chunk.index),
                Some(ts::generate_response::Response::Complete(complete)) => {
                    indexes.push(complete.index);
                    completes.push((
                        complete.index,
                        complete.completion_tokens,
                        complete.output_ids.len(),
                    ));
                    outputs.push(complete.output_ids.clone());
                }
                other => panic!("a chunk or a complete frame, got {other:?}"),
            }
        }
        assert_eq!(completes, vec![(0, 2, 2), (1, 2, 2)], "{frames:?}");
        assert_eq!(
            indexes,
            vec![0, 0, 0, 1, 1, 1],
            "two chunks then the complete per choice, in index order: {frames:?}"
        );
        assert_ne!(
            outputs[0], outputs[1],
            "the two choices differ by their seed: {outputs:?}"
        );
    }

    /// An abort of an `n = 2` request ends both choices: the stream closes
    /// with each choice's `abort` completion and the engine holds nothing of
    /// the request afterwards.
    #[tokio::test]
    async fn an_abort_ends_every_choice_of_a_fanned_out_request_on_a_realistic_worker() {
        let (service, _stop) = scheduler_with_ranks(1, 65_111);
        let mut request = generate_request("two-choices-abort", None, 1_000);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(1_000),
            n: 2,
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        let first = stream
            .next()
            .await
            .expect("a first frame")
            .expect("a frame");
        assert!(
            matches!(
                &first.response,
                Some(ts::generate_response::Response::Chunk(_))
            ),
            "{first:?}"
        );
        service
            .abort(Request::new(ts::AbortRequest {
                request_id: "two-choices-abort".to_string(),
                ..Default::default()
            }))
            .await
            .expect("aborted");
        let mut completes = Vec::new();
        while let Some(frame) = stream.next().await {
            if let Some(ts::generate_response::Response::Complete(complete)) =
                frame.expect("a frame").response
            {
                completes.push((complete.index, complete.finish_reason));
            }
        }
        assert_eq!(
            completes,
            vec![(0, "abort".to_string()), (1, "abort".to_string())],
            "each choice ends with its abort completion"
        );
        let load = service.ranks[0].engine.load();
        assert_eq!(
            (
                load.num_running_reqs,
                load.num_waiting_reqs,
                load.num_used_tokens
            ),
            (0, 0, 0),
            "the engine holds nothing of the request: {load:?}"
        );
    }

    /// A later choice's prompt settles as pending prefill at the engine's
    /// first token for it, not when its turn to stream comes: with two
    /// choices on the default engine both are prefilled together, so by the
    /// time the first choice's own complete frame is read, no frame of the
    /// second read yet, the count is down; it stays down through the rest.
    #[tokio::test]
    async fn a_later_choice_settles_its_prompt_at_the_engines_first_token_not_at_its_turn() {
        let (service, _stop) = scheduler_with_ranks(1, 65_112);
        let mut request = generate_request("two-choices-pending", None, 2);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(2),
            n: 2,
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        // The first choice's two chunks and its complete: nothing of the second.
        let mut first_choice = Vec::new();
        while first_choice.len() < 3 {
            let frame = stream.next().await.expect("a frame").expect("a frame");
            first_choice.push(frame);
        }
        assert!(
            matches!(
                &first_choice[2].response,
                Some(ts::generate_response::Response::Complete(complete)) if complete.index == 0
            ),
            "{:?}",
            first_choice[2]
        );
        assert_eq!(
            service.ranks[0].prefill_pending.load(Ordering::Relaxed),
            0,
            "the second choice's prompt settled at the engine's first token, before its turn"
        );
        while stream.next().await.is_some() {}
        assert_eq!(
            service.ranks[0].prefill_pending.load(Ordering::Relaxed),
            0,
            "nothing left pending"
        );
    }

    /// One `finish=error` cut on a non-streaming `n = 2` request ends the
    /// stream at the first choice's `error` complete and counts once: the
    /// second choice is never read, so it cannot run to a cut of its own.
    #[tokio::test]
    async fn the_first_cut_ends_a_fanned_out_request_before_the_next_choice_is_read() {
        let (service, _stop) = scheduler_with_ranks(1, 65_113);
        service.ranks[0]
            .engine
            .fault_fail(Some(crate::engine::RequestFault {
                status: 503,
                stall: std::time::Duration::ZERO,
                after_tokens: Some(1),
                finish_error: true,
                scope: crate::engine::FaultScope::Requests(1),
            }));
        let mut request = generate_request("two-choices-cut", None, 4);
        request.stream = false;
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(4),
            n: 2,
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        let mut completes = Vec::new();
        while let Some(frame) = stream.next().await {
            match frame.expect("a frame").response {
                Some(ts::generate_response::Response::Complete(complete)) => {
                    completes.push((complete.index, complete.finish_reason));
                }
                other => panic!("a complete frame alone on a non-streaming request, got {other:?}"),
            }
        }
        assert_eq!(
            completes,
            vec![(0, "error".to_string())],
            "the first choice's cut ends the stream"
        );
        assert_eq!(
            service.ranks[0].engine.fault_status().cut_total,
            1,
            "one cut, counted once"
        );
    }

    /// Unpinned requests go to the rank holding the fewest of the worker's
    /// requests, the scan start rotating, so three in a row on an idle worker
    /// cover its three ranks; the counts come down when the streams end.
    #[tokio::test]
    async fn an_unpinned_request_goes_to_the_rank_with_the_fewest_requests() {
        let (service, _stop) = scheduler_with_ranks(3, 65_102);
        let mut streams = Vec::new();
        for id in ["dp-free-a", "dp-free-b", "dp-free-c"] {
            streams.push(
                service
                    .generate(Request::new(generate_request(id, None, 2)))
                    .await
                    .expect("accepted")
                    .into_inner(),
            );
        }
        let counts: Vec<usize> = service
            .ranks
            .iter()
            .map(|rank| rank.in_flight.load(Ordering::Relaxed))
            .collect();
        assert_eq!(counts, vec![1, 1, 1]);
        for stream in &mut streams {
            while stream.next().await.is_some() {}
        }
        let mut placed: Vec<String> = ["dp-free-a", "dp-free-b", "dp-free-c"]
            .iter()
            .map(|id| admitted_on(id))
            .collect();
        placed.sort();
        assert_eq!(
            placed,
            vec!["grpc:65102:r0", "grpc:65102:r1", "grpc:65102:r2"]
        );
        drop(streams);
        let counts: Vec<usize> = service
            .ranks
            .iter()
            .map(|rank| rank.in_flight.load(Ordering::Relaxed))
            .collect();
        assert_eq!(counts, vec![0, 0, 0]);
    }

    /// `GetLoads` lists one entry per rank, the request's `dp_rank` keeping
    /// one; `GetServerInfo` advertises the width.
    #[tokio::test]
    async fn loads_and_server_info_report_the_ranks() {
        let (service, _stop) = scheduler_with_ranks(4, 65_103);
        let loads = service
            .get_loads(Request::new(ts::GetLoadsRequest::default()))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!(loads.dp_rank_count, 4);
        let ranks: Vec<i32> = loads.loads.iter().map(|load| load.dp_rank).collect();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
        assert!(
            loads
                .loads
                .iter()
                .all(|load| load.prefill_throughput.is_none()),
            "no rank has prefilled: no measured capacity yet"
        );
        let one = service
            .get_loads(Request::new(ts::GetLoadsRequest {
                dp_rank: Some(3),
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!(
            (one.dp_rank_count, one.loads.len(), one.loads[0].dp_rank),
            (1, 1, 3)
        );
        let info = service
            .get_server_info(Request::new(ts::GetServerInfoRequest::default()))
            .await
            .expect("info")
            .into_inner();
        let width = info
            .server_args
            .expect("server args")
            .fields
            .remove("dp_size")
            .expect("dp_size");
        assert_eq!(
            width.kind,
            Some(prost_types::value::Kind::StringValue("4".to_string()))
        );
    }

    /// A `loads_only` subscription carries every rank's record and never an
    /// event: one batch per rank at once, then the rank whose record moved.
    #[tokio::test]
    async fn a_loads_only_subscription_streams_each_ranks_record_and_no_events() {
        let (service, _stop) = scheduler_with_ranks(3, 65_105);
        let mut records = service
            .subscribe_kv_events(Request::new(common::SubscribeKvEventsRequest {
                loads_only: true,
                ..Default::default()
            }))
            .await
            .expect("subscribed")
            .into_inner();
        let mut ranks = Vec::new();
        for _ in 0..3 {
            let batch = tokio::time::timeout(Duration::from_secs(2), records.next())
                .await
                .expect("a record at the first tick")
                .expect("an item")
                .expect("a batch");
            assert!(batch.load.as_ref().is_some_and(|load| load.load_only));
            assert!(batch.events.is_empty());
            assert_eq!(batch.sequence_number, 0);
            ranks.push(batch.dp_rank.expect("a rank"));
        }
        ranks.sort_unstable();
        assert_eq!(ranks, vec![0, 1, 2]);

        // A request on rank 1 fills blocks the engine would publish as
        // events: this stream carries rank 1's moved record and no event.
        let mut request = generate_request("dp-loads-1", Some(1), 4);
        request.tokenized = Some(ts::TokenizedInput {
            input_ids: (100..180).collect(),
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        while stream.next().await.is_some() {}
        // Whatever comes next (rank 1's moved record, or the heartbeats) is
        // a record without events; rank 1's arrives within its heartbeat.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let batch = tokio::time::timeout_at(deadline, records.next())
                .await
                .expect("rank 1's record in time")
                .expect("an item")
                .expect("a batch");
            assert!(batch.events.is_empty(), "{batch:?}");
            assert!(batch.load.as_ref().is_some_and(|load| load.load_only));
            if batch.dp_rank == Some(1) {
                // The rank prefilled the request: its record carries the
                // measured prefill capacity, as the servicers' records do.
                assert!(
                    batch
                        .load
                        .as_ref()
                        .and_then(|load| load.prefill_throughput)
                        .is_some_and(|rate| rate > 0.0),
                    "{batch:?}"
                );
                break;
            }
        }
        let rank1 = service
            .get_loads(Request::new(ts::GetLoadsRequest {
                dp_rank: Some(1),
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert!(
            rank1.loads[0]
                .prefill_throughput
                .is_some_and(|rate| rate > 0.0),
            "the poll reports it too: {rank1:?}"
        );
    }

    /// One subscription carries every rank's stream with one record state,
    /// as the relay's: the first record (before any batch) is rank 0's, a
    /// batch carries its own rank's record, and the heartbeat that follows
    /// names the last batch's rank.
    #[tokio::test]
    async fn one_subscription_carries_every_ranks_batches_with_their_ranks_records() {
        let (service, _stop) = scheduler_with_ranks(3, 65_104);
        let mut events = service
            .subscribe_kv_events(Request::new(common::SubscribeKvEventsRequest {
                start_sequence_number: 0,
                load_records_before_first_batch: true,
                loads_only: false,
            }))
            .await
            .expect("subscribed")
            .into_inner();
        let first = tokio::time::timeout(Duration::from_secs(2), events.next())
            .await
            .expect("a record before the first batch")
            .expect("an item")
            .expect("a batch");
        assert!(first.load.as_ref().is_some_and(|load| load.load_only));
        assert_eq!((first.sequence_number, first.dp_rank), (0, Some(0)));

        // A request on rank 2 long enough to fill blocks (the engine
        // publishes a batch per completed block): its batches name rank 2
        // and carry a record.
        let mut request = generate_request("dp-sub-2", Some(2), 4);
        request.tokenized = Some(ts::TokenizedInput {
            input_ids: (100..180).collect(),
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        while stream.next().await.is_some() {}
        let batch = loop {
            let batch = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await
                .expect("rank 2's batch in time")
                .expect("an item")
                .expect("a batch");
            if !batch.events.is_empty() {
                break batch;
            }
        };
        assert_eq!(batch.dp_rank, Some(2));
        assert!(batch.load.as_ref().is_some_and(|load| !load.load_only));
        let beat = loop {
            let batch = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await
                .expect("a heartbeat in time")
                .expect("an item")
                .expect("a batch");
            if batch.events.is_empty() {
                break batch;
            }
        };
        assert_eq!(
            beat.dp_rank,
            Some(2),
            "the heartbeat names the last batch's rank"
        );
    }

    /// As the relays: a resume cursor names one rank's sequence, so a
    /// worker of several ranks refuses it (`OutOfRange`, on which the
    /// gateway resubscribes from zero) instead of resuming every rank at
    /// it and skipping the other ranks' earlier batches; a one-rank worker
    /// resumes at the cursor as before.
    #[tokio::test]
    async fn a_resume_cursor_over_several_ranks_is_out_of_range() {
        let (ranks, _stop) = scheduler_with_ranks(2, 65_106);
        let subscribed = ranks
            .subscribe_kv_events(Request::new(common::SubscribeKvEventsRequest {
                start_sequence_number: 5,
                ..Default::default()
            }))
            .await;
        let status = match subscribed {
            Err(status) => status,
            Ok(_) => panic!("a cursor over two ranks is refused"),
        };
        assert_eq!(status.code(), Code::OutOfRange);
        assert!(status.message().contains("resubscribe from zero"));

        let (one, _stop) = scheduler_with_ranks(1, 65_107);
        assert!(
            one.subscribe_kv_events(Request::new(common::SubscribeKvEventsRequest {
                start_sequence_number: 5,
                ..Default::default()
            }))
            .await
            .is_ok(),
            "one rank resumes at its cursor"
        );
    }

    /// As the Rust relay: a subscriber that asks for records before the
    /// stream's first batch gets a `load_only` one numbered 0 within a tick;
    /// one that does not waits for the first batch, whose sequence the
    /// heartbeats repeat from there.
    #[tokio::test]
    async fn load_records_before_the_first_batch_go_to_the_subscriber_that_asked() {
        let engine = Engine::spawn(engine::EngineParams::default());
        let records = |asked: bool, kv: KvEventStream| {
            with_load_records(
                vec![RankLoadSource {
                    engine: engine.clone(),
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                }],
                LoadsLike::Mock,
                false,
                asked,
                kv,
            )
        };
        let quiet = || Box::pin(stream::pending()) as KvEventStream;
        let load_only =
            |batch: &common::KvEventBatch| batch.load.as_ref().is_some_and(|load| load.load_only);

        let mut asked = records(true, quiet());
        let first = tokio::time::timeout(Duration::from_millis(500), asked.next())
            .await
            .expect("a record before the stream's first batch")
            .expect("an item")
            .expect("a batch");
        assert!(load_only(&first), "{first:?}");
        assert_eq!((first.sequence_number, first.events.len()), (0, 0));

        let mut waiting = records(false, quiet());
        assert!(
            tokio::time::timeout(Duration::from_millis(400), waiting.next())
                .await
                .is_err(),
            "no batch may precede the stream's first for a subscriber that did not ask"
        );

        let seventh = common::KvEventBatch {
            sequence_number: 7,
            ..Default::default()
        };
        let mut after = records(
            false,
            Box::pin(stream::iter([Ok::<_, Status>(seventh)]).chain(stream::pending())),
        );
        let live = after
            .next()
            .await
            .expect("the first batch")
            .expect("a batch");
        assert_eq!(live.sequence_number, 7);
        assert!(!load_only(&live));
        let beat = tokio::time::timeout(
            HEARTBEAT_INTERVAL + Duration::from_millis(500),
            after.next(),
        )
        .await
        .expect("a heartbeat after the interval")
        .expect("an item")
        .expect("a batch");
        assert!(load_only(&beat));
        assert_eq!(beat.sequence_number, 7, "repeats the last sequence sent");
    }

    #[tokio::test]
    async fn a_cut_stream_ends_with_the_status_after_the_given_tokens() {
        let (tx, rx) = mpsc::unbounded_channel();
        for id in 1..=4 {
            tx.send(token(id)).unwrap();
        }
        tx.send(done(4)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((
            2,
            CutEnd::Status(Status::unavailable("injected")),
            Arc::clone(&cuts),
        ));
        let mut stream = generate_stream(rx, true, cut, 0);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        assert_eq!(chunk_ids(stream.next().await), vec![2]);
        let cut = stream.next().await.expect("the cut");
        assert!(matches!(cut, Err(ref status) if status.code() == tonic::Code::Unavailable));
        assert!(stream.next().await.is_none(), "nothing after the cut");
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
    }

    #[tokio::test]
    async fn an_engine_failure_cut_ends_with_an_error_complete_after_the_given_tokens() {
        let (tx, rx) = mpsc::unbounded_channel();
        for id in 1..=4 {
            tx.send(token(id)).unwrap();
        }
        tx.send(done(4)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((2, CutEnd::EngineError, Arc::clone(&cuts)));
        let mut stream = generate_stream(rx, true, cut, 0);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        assert_eq!(chunk_ids(stream.next().await), vec![2]);
        let item = stream
            .next()
            .await
            .expect("the cut")
            .expect("a response, not a status");
        match item {
            Frame::Complete(complete) => {
                assert_eq!(complete.finish_reason, "error");
                assert_eq!(
                    complete.output_ids,
                    vec![1, 2],
                    "the output the engine had sent"
                );
                assert_eq!(complete.completion_tokens, 2);
            }
            other => panic!("not an error completion: {other:?}"),
        }
        assert!(stream.next().await.is_none(), "nothing after the cut");
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
    }

    #[tokio::test]
    async fn an_output_no_longer_than_the_cut_completes_as_usual() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(token(1)).unwrap();
        tx.send(done(1)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((
            1,
            CutEnd::Status(Status::internal("injected")),
            Arc::clone(&cuts),
        ));
        let mut stream = generate_stream(rx, true, cut, 0);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        let complete = stream.next().await.expect("complete").expect("ok");
        assert!(matches!(complete, Frame::Complete(_)));
        assert!(stream.next().await.is_none());
        assert_eq!(
            cuts.load(Ordering::Relaxed),
            0,
            "an output no longer than the cut is not counted as cut"
        );
    }

    #[test]
    fn injected_statuses_map_to_the_closest_grpc_code() {
        assert_eq!(injected_status(503).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(502).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(500).code(), tonic::Code::Internal);
        assert_eq!(injected_status(429).code(), tonic::Code::ResourceExhausted);
        assert_eq!(injected_status(504).code(), tonic::Code::DeadlineExceeded);
        assert_eq!(injected_status(400).code(), tonic::Code::InvalidArgument);
        assert_eq!(injected_status(499).code(), tonic::Code::Cancelled);
        // The three non-inverse cases (408 and 502 read back as 504 and 503,
        // 499 as 502) and a 4xx without its own code (read back as 400).
        assert_eq!(injected_status(408).code(), tonic::Code::DeadlineExceeded);
        assert_eq!(injected_status(502).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(418).code(), tonic::Code::InvalidArgument);
        // 422 is the engine's report that it could not continue the request
        // under its constraint, which the gateway reads back as its 422.
        let constraint = injected_status(422);
        assert_eq!(constraint.code(), tonic::Code::FailedPrecondition);
        assert!(smg_grpc_client::is_constraint_failure(&constraint));
    }

    /// The measured input capacity alone moves a record: appearing, expiring
    /// or moving by the relay's margin is a change worth a push, noise within
    /// the margin is not.
    #[test]
    fn the_measured_input_capacity_alone_changes_a_record() {
        let last = common::EngineLoad {
            prefill_throughput: Some(20_000.0),
            ..Default::default()
        };
        let with = |rate| common::EngineLoad {
            prefill_throughput: rate,
            ..last.clone()
        };
        assert!(!load_changed(&last, &with(Some(20_000.0))), "unmoved");
        assert!(!load_changed(&last, &with(Some(20_040.0))), "noise");
        assert!(load_changed(&last, &with(Some(22_000.0))), "5 % more");
        assert!(load_changed(&last, &with(None)), "expired");
        assert!(load_changed(&with(None), &last), "appeared");
    }

    /// The per-request line of a worker of several ranks carries the pin as
    /// an engine's servicer writes it, and the rank the request ran on.
    #[test]
    fn the_rank_line_names_the_pin_and_the_rank() {
        assert_eq!(
            rank_line("r-7", Some(1), 1),
            "request r-7: dp_rank=1, ran on rank 1"
        );
        assert_eq!(
            rank_line("r-8", None, 0),
            "request r-8: dp_rank=None, ran on rank 0"
        );
    }

    /// An `n` above the gateway's bound of 128 is refused as an invalid
    /// argument before the request reaches a rank or the fault hook: the
    /// armed fault is neither consumed nor answered.
    #[tokio::test]
    async fn an_n_above_the_gateways_bound_is_refused_before_the_fault_hook() {
        let (service, _stop) = scheduler_with_ranks(1, 65_114);
        service.ranks[0]
            .engine
            .fault_fail(Some(crate::engine::RequestFault {
                status: 503,
                stall: std::time::Duration::ZERO,
                after_tokens: None,
                finish_error: false,
                scope: crate::engine::FaultScope::Requests(1),
            }));
        let mut request = generate_request("too-many-choices", None, 2);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(2),
            n: 129,
            ..Default::default()
        });
        let error = service
            .generate(Request::new(request))
            .await
            .err()
            .expect("refused");
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "{error:?}");
        let status = service.ranks[0].engine.fault_status();
        assert_eq!(status.failed_total, 0, "the fault hook was not consulted");
        assert!(
            status.fail.is_some(),
            "the armed fault waits for the next request"
        );
    }

    /// One rank whose engine runs one request at a time, prefix caching off:
    /// the shape under which the pending count has to cover a choice still
    /// queued behind the one that streams.
    fn scheduler_with_serialized_admission(port: u16) -> (MockScheduler, watch::Sender<bool>) {
        MockScheduler::for_tests(
            Config {
                realistic: true,
                grpc_dp_size: 1,
                engine: engine::EngineParams {
                    timing: engine::TimingModel::linear(),
                    max_running: 1,
                    prefix_cache: false,
                    ..Default::default()
                },
                ..Default::default()
            },
            port,
        )
    }

    /// Under serialized admission the second choice of an `n = 2` request is
    /// queued while the first streams; its prompt stays pending prefill
    /// after the first choice's first token (the liveness bound reads that
    /// count while the queued choice prefills) and settles when it streams.
    #[tokio::test]
    async fn a_queued_choice_keeps_its_prompt_pending_under_serialized_admission() {
        let (service, _stop) = scheduler_with_serialized_admission(65_115);
        let mut request = generate_request("two-choices-serialized", None, 2);
        request.sampling_params = Some(ts::SamplingParams {
            max_new_tokens: Some(2),
            n: 2,
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        let first = stream
            .next()
            .await
            .expect("a first frame")
            .expect("a frame");
        assert!(
            matches!(
                &first.response,
                Some(ts::generate_response::Response::Chunk(_))
            ),
            "{first:?}"
        );
        assert_eq!(
            service.ranks[0].prefill_pending.load(Ordering::Relaxed),
            4,
            "the queued choice's prompt is still pending after the first token"
        );
        while stream.next().await.is_some() {}
        assert_eq!(
            service.ranks[0].prefill_pending.load(Ordering::Relaxed),
            0,
            "settled once the queued choice streamed"
        );
    }
}
