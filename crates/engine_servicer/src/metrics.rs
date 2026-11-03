//! Prometheus metrics of a servicer, whichever engine it serves: the
//! `smg_servicer_*` families, observed at the points the request path
//! already passes, and the `GET /metrics` listener that renders them.
//!
//! The recorder is this module's own instance, never the process-global one:
//! a process that hosts a servicer beside another metrics user (the Python
//! extension links the gateway too) never fights over the global recorder,
//! and two servicers in one process keep two registries. Every handle is
//! resolved once, per rank and per label value, when the servicer starts;
//! the request path never builds a key, hashes a label or takes a lock.
//!
//! What the request path does, and nothing else: one clock read at arrival,
//! one clock read plus one histogram push and two atomic adds at the first
//! output, one atomic add (and, with the inter-token histogram on, one clock
//! read and one push) per output chunk, and one clock read plus a handful of
//! pushes when the stream ends. Counters are plain atomics the scrape maps
//! onto Prometheus counters; the one per-chunk histogram sits behind a
//! pressure sampler that drops observations above a bound and counts the
//! drops, so a scraper that never comes cannot grow the process. Without a
//! listener nothing is recorded beyond the token totals the stats line reads.
//!
//! The listener runs on a dedicated thread with its own runtime: a slow or
//! concurrent scraper never shares the request runtime. It binds beside the
//! gRPC listener, before the engine connects, so a scrape during the engine's
//! start answers; `stop` closes it at once (a scrape in flight gets its bytes
//! or a reset), inside the lifecycle owner's stop budget. A scrape that fails
//! answers 500 for that scrape alone and is counted; the next one renders
//! again.

use std::{
    future::IntoFuture,
    net::SocketAddr,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State as AxumState,
    http::{header::CONTENT_TYPE, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::FutureExt;
use metrics::{Counter, Gauge, Histogram, Key, KeyName, Label, Level, Metadata, Recorder};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use smg_grpc_client::{vllm_proto as vllm, WorkerMediaFault};
use tokio::net::TcpListener;
use tonic::{Code, Status};
use tracing::{info, warn};

use crate::{ServerThread, ServicerError, Shutdown};

/// The exposition content type, as the gateway answers it.
const CONTENT_TYPE_PROMETHEUS: &str = "text/plain; version=0.0.4; charset=utf-8";
/// How often the listener's thread drains pending histogram samples into
/// the rendered distributions (and resets the pressure sampler) between
/// scrapes.
pub(crate) const UPKEEP_INTERVAL: Duration = Duration::from_secs(5);
/// Pending inter-token observations above which the sampler drops the next
/// one and counts it: about 8 MB at the measured cost per pending sample,
/// reached only by a servicer nobody scrapes for minutes at tens of thousands
/// of tokens per second, or whose listener thread died.
pub(crate) const ITL_PENDING_BOUND: u32 = 512 * 1024;
/// The least a stop gives the listener to close, however spent the budget.
const METRICS_STOP_FLOOR: Duration = Duration::from_millis(500);

/// vLLM's time-to-first-token buckets (seconds): the same edges, so a
/// dashboard built for the engine's own server reads this family by `le`.
pub(crate) const TTFT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
/// vLLM's inter-token latency buckets (seconds), for the per-chunk latency
/// and the per-request mean.
pub(crate) const INTER_TOKEN_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
/// vLLM's request latency buckets (seconds), for the end-to-end latency.
pub(crate) const REQUEST_LATENCY_BUCKETS: &[f64] = &[
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
/// Buckets of the `n` parameter.
pub(crate) const PARAMS_N_BUCKETS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 20.0];
/// Buckets of the media references a request carries, per modality.
pub(crate) const MEDIA_ITEMS_BUCKETS: &[f64] =
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0];
/// The token-count buckets' ceiling when the model's context length is not
/// known.
const DEFAULT_MAX_MODEL_LEN: u32 = 32_768;

/// A 1-2-5 series of bucket edges up to `max_value` (vLLM's
/// `build_1_2_5_buckets`): the token-count histograms' edges.
pub(crate) fn one_two_five_buckets(max_value: u32) -> Vec<f64> {
    let max_value = f64::from(max_value.max(1));
    let mut buckets = Vec::new();
    let mut exponent = 0;
    loop {
        for mantissa in [1.0, 2.0, 5.0] {
            let value = mantissa * 10f64.powi(exponent);
            if value > max_value {
                return buckets;
            }
            buckets.push(value);
        }
        exponent += 1;
    }
}

/// How a request ended, as `smg_servicer_requests_total` splits it. A
/// fan-out (`n > 1`) is one request: its reason is that of the last choice
/// to end, with an abort winning over the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum FinishReason {
    /// The engine's stop (EOS, a stop id, or a stop string this servicer
    /// matched itself).
    Stop = 0,
    /// The token budget ran out.
    Length = 1,
    /// The caller's abort (the `Abort` RPC or a dropped stream), or the
    /// engine's own.
    Abort = 2,
    /// The engine or the wire failed the request.
    Error = 3,
    /// A shutdown cut the request while it was still open.
    Cut = 4,
}

impl FinishReason {
    const ALL: [Self; 5] = [
        Self::Stop,
        Self::Length,
        Self::Abort,
        Self::Error,
        Self::Cut,
    ];
    /// No choice has ended yet.
    const NONE: u8 = u8::MAX;

    const fn label(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Error => "error",
            Self::Cut => "cut",
        }
    }

    /// The reason a terminal `Complete` carries on every wire (`stop`,
    /// `length`, `abort`, `error`); anything else counts as a stop, the way
    /// the adapters normalize an unknown wire value.
    pub(crate) fn from_wire(finish_reason: &str) -> Self {
        match finish_reason {
            "length" => Self::Length,
            "abort" => Self::Abort,
            "error" => Self::Error,
            _ => Self::Stop,
        }
    }

    fn from_u8(value: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|reason| *reason as u8 == value)
    }
}

/// Why a request was refused before it reached the engine
/// (`smg_servicer_requests_refused_total`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refusal {
    /// The id names a request still in flight.
    DuplicateId = 0,
    /// The servicer is draining.
    Draining = 1,
    /// The request itself was invalid.
    InvalidArgument = 2,
    /// The request's media could not be processed.
    Media = 3,
}

impl Refusal {
    const ALL: [Self; 4] = [
        Self::DuplicateId,
        Self::Draining,
        Self::InvalidArgument,
        Self::Media,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::DuplicateId => "duplicate_id",
            Self::Draining => "draining",
            Self::InvalidArgument => "invalid_argument",
            Self::Media => "media",
        }
    }
}

/// The message of the `UNAVAILABLE` status a draining servicer refuses new
/// requests with (the registry's), the one refusal told apart by its text.
pub(crate) const DRAINING_MESSAGE: &str = "servicer is draining";

/// The lifecycle state a scrape reports as `smg_servicer_state`, one-hot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServicerState {
    /// The engine has not completed its handshake.
    Starting = 0,
    /// The engine is connected and alive, admission is open and readiness
    /// is reported.
    Serving = 1,
    /// The lifecycle owner announced a drain: admission is closed.
    Draining = 2,
    /// The engine connection failed or the engine is no longer alive.
    EngineDown = 3,
    /// The engine is connected and alive and admission is open, but the
    /// lifecycle owner holds readiness back (a planned warm-up before
    /// SERVING).
    WarmingUp = 4,
}

impl ServicerState {
    const ALL: [Self; 5] = [
        Self::Starting,
        Self::Serving,
        Self::Draining,
        Self::EngineDown,
        Self::WarmingUp,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Serving => "serving",
            Self::Draining => "draining",
            Self::EngineDown => "engine_down",
            Self::WarmingUp => "warming_up",
        }
    }
}

/// What a scrape reads off the servicer's state: the gauges that are
/// derived rather than counted.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Snapshot {
    /// Admitted requests still preprocessing, streaming, or unwinding.
    pub(crate) active: usize,
    pub(crate) state: Option<ServicerState>,
}

/// The facts a servicer's metrics carry on every series: fixed for the life
/// of the process.
pub(crate) struct Identity {
    /// The served model name (`model` label).
    pub(crate) model: String,
    /// The wire's name (`wire` label): the per-engine module's constant.
    pub(crate) wire: &'static str,
    /// The engines that dial in: one handle set per rank (`rank` label).
    pub(crate) ranks: usize,
    /// The model's context length, the token-count histograms' ceiling.
    pub(crate) max_model_len: u32,
    /// Whether this servicer matches string stops itself (vLLM and
    /// TokenSpeed; SGLang's scheduler matches its own): the string-stop
    /// family is registered only then.
    pub(crate) matches_string_stops: bool,
    /// Whether this servicer processes media itself (the worker-side media
    /// path): the three media families are registered only then, so a wire
    /// that measures no media exports no media series.
    pub(crate) processes_media: bool,
}

/// The handles of one data-parallel rank.
struct RankHandles {
    requests_total: [Counter; FinishReason::ALL.len()],
    prompt_tokens: Counter,
    prompt_tokens_cached: Counter,
    generation_tokens: Counter,
    string_stops: Option<Counter>,
    time_to_first_token: Histogram,
    e2e_latency: Histogram,
    time_per_output_token: Histogram,
    inter_token_latency: Histogram,
    request_prompt_tokens: Histogram,
    request_generation_tokens: Histogram,
    request_params_max_tokens: Histogram,
    request_params_n: Histogram,
}

/// The plain atomics the request path adds to; mapped onto the rank's
/// counters at scrape (`Counter::absolute`). They exist whether or not a
/// listener does: the periodic stats line reads the token totals.
#[derive(Default)]
struct RankTotals {
    prompt_tokens: AtomicU64,
    prompt_tokens_cached: AtomicU64,
    generation_tokens: AtomicU64,
    string_stops: AtomicU64,
}

/// What exists only with a listener: the recorder and every handle on it.
struct Recorded {
    handle: PrometheusHandle,
    ranks: Vec<RankHandles>,
    /// The handles of a rank beyond the configured width (an engine that
    /// dialed in under an index the launcher did not announce), built on
    /// first use under `rank="other"`.
    other_rank: OnceLock<RankHandles>,
    registrar: Registrar,
    refused: [Counter; Refusal::ALL.len()],
    /// One counter per gRPC code, registered on the first error of that
    /// code (an error is a cold path; a code that never happens renders no
    /// series).
    errors: Vec<OnceLock<Counter>>,
    active: Gauge,
    state: [Gauge; ServicerState::ALL.len()],
    dropped_inter_token: Counter,
    scrape_seconds: Gauge,
    scrapes_ok: Counter,
    scrapes_failed: Counter,
    /// Worker-side media: the references per request by modality (image,
    /// video, audio), the time per stage, the requests by outcome; `None` on
    /// a wire that processes no media itself.
    media_items: Option<[Histogram; MEDIA_MODALITIES.len()]>,
    media_stages: Option<[Histogram; MediaStage::ALL.len()]>,
    media_outcomes: Option<[Counter; MediaOutcome::ALL.len()]>,
}

/// Registers handles on this servicer's own recorder with its fixed labels.
struct Registrar {
    recorder: metrics_exporter_prometheus::PrometheusRecorder,
    model: String,
    wire: &'static str,
}

impl Registrar {
    fn metadata() -> Metadata<'static> {
        Metadata::new(module_path!(), Level::INFO, Some(module_path!()))
    }

    fn labels(&self, extra: &[(&'static str, &str)]) -> Vec<Label> {
        let mut labels = Vec::with_capacity(2 + extra.len());
        labels.push(Label::new("model", self.model.clone()));
        labels.push(Label::new("wire", self.wire));
        for (name, value) in extra {
            labels.push(Label::new(*name, value.to_string()));
        }
        labels
    }

    fn counter(&self, name: &'static str, extra: &[(&'static str, &str)]) -> Counter {
        let key = Key::from_parts(name, self.labels(extra));
        self.recorder.register_counter(&key, &Self::metadata())
    }

    fn gauge(&self, name: &'static str, extra: &[(&'static str, &str)]) -> Gauge {
        let key = Key::from_parts(name, self.labels(extra));
        self.recorder.register_gauge(&key, &Self::metadata())
    }

    fn histogram(&self, name: &'static str, extra: &[(&'static str, &str)]) -> Histogram {
        let key = Key::from_parts(name, self.labels(extra));
        self.recorder.register_histogram(&key, &Self::metadata())
    }

    fn describe_counter(&self, name: &'static str, help: &'static str) {
        self.recorder
            .describe_counter(KeyName::from_const_str(name), None, help.into());
    }

    fn describe_gauge(&self, name: &'static str, help: &'static str) {
        self.recorder
            .describe_gauge(KeyName::from_const_str(name), None, help.into());
    }

    fn describe_histogram(&self, name: &'static str, help: &'static str) {
        self.recorder
            .describe_histogram(KeyName::from_const_str(name), None, help.into());
    }

    fn rank(&self, rank: &str, string_stops: bool) -> RankHandles {
        let labels: &[(&'static str, &str)] = &[("rank", rank)];
        RankHandles {
            requests_total: FinishReason::ALL.map(|reason| {
                self.counter(
                    names::REQUESTS_TOTAL,
                    &[("rank", rank), ("finish_reason", reason.label())],
                )
            }),
            prompt_tokens: self.counter(names::PROMPT_TOKENS_TOTAL, labels),
            prompt_tokens_cached: self.counter(names::PROMPT_TOKENS_CACHED_TOTAL, labels),
            generation_tokens: self.counter(names::GENERATION_TOKENS_TOTAL, labels),
            string_stops: string_stops.then(|| self.counter(names::STRING_STOPS_TOTAL, labels)),
            time_to_first_token: self.histogram(names::TIME_TO_FIRST_TOKEN_SECONDS, labels),
            e2e_latency: self.histogram(names::E2E_REQUEST_LATENCY_SECONDS, labels),
            time_per_output_token: self
                .histogram(names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS, labels),
            inter_token_latency: self.histogram(names::INTER_TOKEN_LATENCY_SECONDS, labels),
            request_prompt_tokens: self.histogram(names::REQUEST_PROMPT_TOKENS, labels),
            request_generation_tokens: self.histogram(names::REQUEST_GENERATION_TOKENS, labels),
            request_params_max_tokens: self.histogram(names::REQUEST_PARAMS_MAX_TOKENS, labels),
            request_params_n: self.histogram(names::REQUEST_PARAMS_N, labels),
        }
    }
}

/// The family names, each with the prefix.
pub(crate) mod names {
    pub(crate) const REQUESTS_TOTAL: &str = "smg_servicer_requests_total";
    pub(crate) const REQUESTS_REFUSED_TOTAL: &str = "smg_servicer_requests_refused_total";
    pub(crate) const REQUESTS_ACTIVE: &str = "smg_servicer_requests_active";
    pub(crate) const REQUEST_ERRORS_TOTAL: &str = "smg_servicer_request_errors_total";
    pub(crate) const PROMPT_TOKENS_TOTAL: &str = "smg_servicer_prompt_tokens_total";
    pub(crate) const PROMPT_TOKENS_CACHED_TOTAL: &str = "smg_servicer_prompt_tokens_cached_total";
    pub(crate) const GENERATION_TOKENS_TOTAL: &str = "smg_servicer_generation_tokens_total";
    pub(crate) const STRING_STOPS_TOTAL: &str = "smg_servicer_string_stops_total";
    pub(crate) const TIME_TO_FIRST_TOKEN_SECONDS: &str = "smg_servicer_time_to_first_token_seconds";
    pub(crate) const E2E_REQUEST_LATENCY_SECONDS: &str = "smg_servicer_e2e_request_latency_seconds";
    pub(crate) const REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS: &str =
        "smg_servicer_request_time_per_output_token_seconds";
    pub(crate) const INTER_TOKEN_LATENCY_SECONDS: &str = "smg_servicer_inter_token_latency_seconds";
    pub(crate) const REQUEST_PROMPT_TOKENS: &str = "smg_servicer_request_prompt_tokens";
    pub(crate) const REQUEST_GENERATION_TOKENS: &str = "smg_servicer_request_generation_tokens";
    pub(crate) const REQUEST_PARAMS_MAX_TOKENS: &str = "smg_servicer_request_params_max_tokens";
    pub(crate) const REQUEST_PARAMS_N: &str = "smg_servicer_request_params_n";
    pub(crate) const STATE: &str = "smg_servicer_state";
    pub(crate) const START_TIME_SECONDS: &str = "smg_servicer_start_time_seconds";
    pub(crate) const METRICS_DROPPED_TOTAL: &str = "smg_servicer_metrics_dropped_total";
    pub(crate) const METRICS_SCRAPE_SECONDS: &str = "smg_servicer_metrics_scrape_seconds";
    pub(crate) const METRICS_SCRAPES_TOTAL: &str = "smg_servicer_metrics_scrapes_total";
    pub(crate) const MEDIA_ITEMS_PER_REQUEST: &str = "smg_servicer_media_items_per_request";
    pub(crate) const MEDIA_PROCESSING_SECONDS: &str = "smg_servicer_media_processing_seconds";
    pub(crate) const MEDIA_REQUESTS_TOTAL: &str = "smg_servicer_media_requests_total";
}

/// A media request's stage, for its time: the wait for a processing slot
/// and the processing itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaStage {
    Wait = 0,
    Process = 1,
}

impl MediaStage {
    const ALL: [Self; 2] = [Self::Wait, Self::Process];

    const fn label(self) -> &'static str {
        match self {
            Self::Wait => "wait",
            Self::Process => "process",
        }
    }
}

/// How a request's worker-side media processing ended: processed, refused
/// before processing (references this worker does not take, a missing
/// processor, no slot), or the processor's verdict (the caller's media, a
/// transient failure, an invalid request, an unavailable or a failed
/// processor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MediaOutcome {
    Ok = 0,
    Refused = 1,
    Client = 2,
    Transient = 3,
    Invalid = 4,
    Unavailable = 5,
    Internal = 6,
}

impl MediaOutcome {
    const ALL: [Self; 7] = [
        Self::Ok,
        Self::Refused,
        Self::Client,
        Self::Transient,
        Self::Invalid,
        Self::Unavailable,
        Self::Internal,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Refused => "refused",
            Self::Client => "client",
            Self::Transient => "transient",
            Self::Invalid => "invalid",
            Self::Unavailable => "unavailable",
            Self::Internal => "internal",
        }
    }
}

/// The media modalities a request's references are counted under.
const MEDIA_MODALITIES: [&str; 3] = ["image", "video", "audio"];

/// A servicer's metrics: the totals the request path adds to, and, with a
/// listener, the recorder and the handles the scrape renders.
pub(crate) struct ServicerMetrics {
    totals: Vec<RankTotals>,
    other_totals: RankTotals,
    recorded: Option<Recorded>,
    /// Pending inter-token observations since the last drain (the pressure
    /// sampler's count) and the observations it dropped.
    inter_token_pending: AtomicU32,
    inter_token_dropped: AtomicU64,
}

impl ServicerMetrics {
    /// The metrics of a servicer that exposes them (`enabled`), or the bare
    /// token totals of one that does not: without a listener nothing is
    /// recorded, because nothing would ever drain it.
    pub(crate) fn new(identity: &Identity, enabled: bool) -> Result<Arc<Self>, ServicerError> {
        let ranks = identity.ranks.max(1);
        let recorded = enabled
            .then(|| Recorded::new(identity, ranks))
            .transpose()?;
        Ok(Arc::new(Self {
            totals: (0..ranks).map(|_| RankTotals::default()).collect(),
            other_totals: RankTotals::default(),
            recorded,
            inter_token_pending: AtomicU32::new(0),
            inter_token_dropped: AtomicU64::new(0),
        }))
    }

    /// Whether observations are recorded (a listener exists).
    pub(crate) fn enabled(&self) -> bool {
        self.recorded.is_some()
    }

    /// Prompt tokens counted so far, over every rank: the stats line's figure.
    pub(crate) fn prompt_tokens_total(&self) -> u64 {
        self.totals()
            .map(|totals| totals.prompt_tokens.load(Ordering::Relaxed))
            .sum()
    }

    /// Generation tokens counted so far, over every rank.
    pub(crate) fn generation_tokens_total(&self) -> u64 {
        self.totals()
            .map(|totals| totals.generation_tokens.load(Ordering::Relaxed))
            .sum()
    }

    fn totals(&self) -> impl Iterator<Item = &RankTotals> {
        self.totals
            .iter()
            .chain(std::iter::once(&self.other_totals))
    }

    /// The slot of a rank: its index, or the catch-all past the configured
    /// width for an index the launcher did not announce (or none).
    fn slot(&self, rank: Option<u32>) -> usize {
        rank.and_then(|rank| usize::try_from(rank).ok())
            .filter(|rank| *rank < self.totals.len())
            .unwrap_or(self.totals.len())
    }

    fn rank_totals(&self, slot: usize) -> &RankTotals {
        self.totals.get(slot).unwrap_or(&self.other_totals)
    }

    fn rank_handles(&self, slot: usize) -> Option<&RankHandles> {
        let recorded = self.recorded.as_ref()?;
        Some(recorded.ranks.get(slot).unwrap_or_else(|| {
            recorded.other_rank.get_or_init(|| {
                recorded
                    .registrar
                    .rank("other", recorded.ranks[0].string_stops.is_some())
            })
        }))
    }

    /// A request was admitted and its choices submitted: the observation
    /// its stream and choices record into, ending when the last of them
    /// drops. `prompt_tokens` is the servicer's own count of the prompt
    /// (the engine's report replaces it at the first output).
    pub(crate) fn request(
        self: &Arc<Self>,
        arrival: Instant,
        rank: Option<u32>,
        prompt_tokens: u32,
        max_tokens: Option<u32>,
        n: u32,
    ) -> Arc<RequestMetrics> {
        let slot = self.slot(rank);
        if let Some(handles) = self.rank_handles(slot) {
            if let Some(max_tokens) = max_tokens {
                handles.request_params_max_tokens.record(max_tokens);
            }
            handles.request_params_n.record(n.max(1));
        }
        Arc::new(RequestMetrics {
            metrics: Arc::clone(self),
            slot,
            arrival,
            first_output: OnceLock::new(),
            last_output_ns: AtomicU64::new(0),
            prompt_counted: AtomicBool::new(false),
            prompt_tokens: AtomicU32::new(prompt_tokens),
            generation_tokens: AtomicU32::new(0),
            last_finish: AtomicU8::new(FinishReason::NONE),
            aborted: AtomicBool::new(false),
            cut: AtomicBool::new(false),
            errored: AtomicBool::new(false),
        })
    }

    /// The media references a request carried, per modality, counted before
    /// they are checked (a refused request's included); a modality with none
    /// is not an observation.
    pub(crate) fn media_items(&self, image: usize, video: usize, audio: usize) {
        let Some(handles) = self
            .recorded
            .as_ref()
            .and_then(|recorded| recorded.media_items.as_ref())
        else {
            return;
        };
        for (handle, count) in handles.iter().zip([image, video, audio]) {
            if count > 0 {
                handle.record(count as f64);
            }
        }
    }

    /// A media request's time in `stage`.
    pub(crate) fn media_stage(&self, stage: MediaStage, took: Duration) {
        if let Some(handles) = self
            .recorded
            .as_ref()
            .and_then(|recorded| recorded.media_stages.as_ref())
        {
            handles[stage as usize].record(took);
        }
    }

    /// How a request's worker-side media processing ended.
    pub(crate) fn media_outcome(&self, outcome: MediaOutcome) {
        if let Some(handles) = self
            .recorded
            .as_ref()
            .and_then(|recorded| recorded.media_outcomes.as_ref())
        {
            handles[outcome as usize].increment(1);
        }
    }

    /// SGLang PD media preprocessing/reuse attempts, on this servicer's
    /// recorder (never the gateway's process-global recorder).
    pub(crate) fn media_identity(&self, outcome: &'static str) {
        if let Some(recorded) = &self.recorded {
            recorded
                .registrar
                .counter("smg_servicer_media_identity_total", &[("outcome", outcome)])
                .increment(1);
        }
    }

    /// A request ended before its stream existed: the status `Generate`
    /// answered with. A refusal of the servicer's own (a duplicate id, a
    /// drain, an invalid request, media that could not be processed) counts
    /// under its reason; anything else is an error under its gRPC code.
    pub(crate) fn request_not_started(&self, status: &Status) {
        let Some(recorded) = &self.recorded else {
            return;
        };
        let refusal = match status.code() {
            Code::AlreadyExists => Some(Refusal::DuplicateId),
            Code::Unavailable if status.message() == DRAINING_MESSAGE => Some(Refusal::Draining),
            _ if status
                .metadata()
                .contains_key(WorkerMediaFault::METADATA_KEY) =>
            {
                Some(Refusal::Media)
            }
            Code::InvalidArgument => Some(Refusal::InvalidArgument),
            _ => None,
        };
        match refusal {
            Some(refusal) => recorded.refused[refusal as usize].increment(1),
            None => recorded.error(status.code()).increment(1),
        }
    }

    /// One inter-token observation, under the pressure sampler: above the
    /// bound of pending samples the observation is dropped and counted.
    fn inter_token(&self, slot: usize, latency: Duration) {
        let Some(handles) = self.rank_handles(slot) else {
            return;
        };
        let pending = self.inter_token_pending.fetch_add(1, Ordering::Relaxed);
        if pending >= ITL_PENDING_BOUND {
            self.inter_token_pending.fetch_sub(1, Ordering::Relaxed);
            self.inter_token_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        handles.inter_token_latency.record(latency);
    }

    /// Drain pending histogram samples into the rendered distributions and
    /// reset the pressure sampler: the listener's periodic upkeep, and the
    /// first step of every scrape.
    fn upkeep(&self) {
        if let Some(recorded) = &self.recorded {
            recorded.handle.run_upkeep();
        }
        self.inter_token_pending.store(0, Ordering::Relaxed);
    }

    /// Set the derived gauges and map the totals onto their counters, then
    /// render the exposition. Only a listener calls it; a test may too.
    pub(crate) fn render(&self, snapshot: Snapshot) -> String {
        let Some(recorded) = &self.recorded else {
            return String::new();
        };
        self.upkeep();
        recorded
            .active
            .set(u32::try_from(snapshot.active).unwrap_or(u32::MAX));
        for state in ServicerState::ALL {
            recorded.state[state as usize].set(u8::from(snapshot.state == Some(state)));
        }
        for (slot, totals) in self.totals.iter().enumerate() {
            if let Some(handles) = recorded.ranks.get(slot) {
                handles.absolute(totals);
            }
        }
        if let Some(handles) = recorded.other_rank.get() {
            handles.absolute(&self.other_totals);
        }
        recorded
            .dropped_inter_token
            .absolute(self.inter_token_dropped.load(Ordering::Relaxed));
        recorded.handle.render()
    }

    /// The scrape as the listener serves it: the body and its status; a
    /// failure inside the render (a panic) is one 500, counted, and the
    /// next scrape renders again.
    fn scrape(&self, snapshot: impl FnOnce() -> Snapshot) -> (StatusCode, String) {
        let Some(recorded) = &self.recorded else {
            return (StatusCode::NOT_FOUND, String::new());
        };
        let started = Instant::now();
        match catch_unwind(AssertUnwindSafe(|| self.render(snapshot()))) {
            Ok(body) => {
                recorded.scrape_seconds.set(started.elapsed());
                recorded.scrapes_ok.increment(1);
                (StatusCode::OK, body)
            }
            Err(_) => {
                recorded.scrapes_failed.increment(1);
                warn!("the metrics scrape failed; the next one renders again");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "metrics scrape failed; the next scrape renders again\n".to_string(),
                )
            }
        }
    }
}

impl RankHandles {
    fn absolute(&self, totals: &RankTotals) {
        self.prompt_tokens
            .absolute(totals.prompt_tokens.load(Ordering::Relaxed));
        self.prompt_tokens_cached
            .absolute(totals.prompt_tokens_cached.load(Ordering::Relaxed));
        self.generation_tokens
            .absolute(totals.generation_tokens.load(Ordering::Relaxed));
        if let Some(string_stops) = &self.string_stops {
            string_stops.absolute(totals.string_stops.load(Ordering::Relaxed));
        }
    }
}

impl Recorded {
    fn new(identity: &Identity, ranks: usize) -> Result<Self, ServicerError> {
        let max_model_len = if identity.max_model_len == 0 {
            DEFAULT_MAX_MODEL_LEN
        } else {
            identity.max_model_len
        };
        let token_buckets = one_two_five_buckets(max_model_len);
        let buckets: [(&str, &[f64]); 10] = [
            (names::MEDIA_ITEMS_PER_REQUEST, MEDIA_ITEMS_BUCKETS),
            (names::MEDIA_PROCESSING_SECONDS, TTFT_BUCKETS),
            (names::TIME_TO_FIRST_TOKEN_SECONDS, TTFT_BUCKETS),
            (names::E2E_REQUEST_LATENCY_SECONDS, REQUEST_LATENCY_BUCKETS),
            (
                names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
                INTER_TOKEN_BUCKETS,
            ),
            (names::INTER_TOKEN_LATENCY_SECONDS, INTER_TOKEN_BUCKETS),
            (names::REQUEST_PROMPT_TOKENS, &token_buckets),
            (names::REQUEST_GENERATION_TOKENS, &token_buckets),
            (names::REQUEST_PARAMS_MAX_TOKENS, &token_buckets),
            (names::REQUEST_PARAMS_N, PARAMS_N_BUCKETS),
        ];
        let mut builder = PrometheusBuilder::new().upkeep_timeout(UPKEEP_INTERVAL);
        for (name, edges) in buckets {
            builder = builder
                .set_buckets_for_metric(Matcher::Full(name.to_string()), edges)
                .map_err(|error| {
                    ServicerError::Startup(format!("metrics buckets of {name}: {error}"))
                })?;
        }
        let recorder = builder.build_recorder();
        let handle = recorder.handle();
        let registrar = Registrar {
            recorder,
            model: identity.model.clone(),
            wire: identity.wire,
        };
        describe(&registrar);
        let ranks = (0..ranks)
            .map(|rank| registrar.rank(&rank.to_string(), identity.matches_string_stops))
            .collect();
        let refused = Refusal::ALL.map(|refusal| {
            registrar.counter(
                names::REQUESTS_REFUSED_TOTAL,
                &[("reason", refusal.label())],
            )
        });
        let state = ServicerState::ALL
            .map(|state| registrar.gauge(names::STATE, &[("state", state.label())]));
        let start_time = registrar.gauge(names::START_TIME_SECONDS, &[]);
        start_time.set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO),
        );
        Ok(Self {
            handle,
            ranks,
            other_rank: OnceLock::new(),
            refused,
            errors: (0..=GRPC_CODES).map(|_| OnceLock::new()).collect(),
            active: registrar.gauge(names::REQUESTS_ACTIVE, &[]),
            state,
            dropped_inter_token: registrar.counter(
                names::METRICS_DROPPED_TOTAL,
                &[("family", "inter_token_latency")],
            ),
            scrape_seconds: registrar.gauge(names::METRICS_SCRAPE_SECONDS, &[]),
            scrapes_ok: registrar.counter(names::METRICS_SCRAPES_TOTAL, &[("code", "200")]),
            scrapes_failed: registrar.counter(names::METRICS_SCRAPES_TOTAL, &[("code", "500")]),
            media_items: identity.processes_media.then(|| {
                MEDIA_MODALITIES.map(|modality| {
                    registrar.histogram(names::MEDIA_ITEMS_PER_REQUEST, &[("modality", modality)])
                })
            }),
            media_stages: identity.processes_media.then(|| {
                MediaStage::ALL.map(|stage| {
                    registrar
                        .histogram(names::MEDIA_PROCESSING_SECONDS, &[("stage", stage.label())])
                })
            }),
            media_outcomes: identity.processes_media.then(|| {
                MediaOutcome::ALL.map(|outcome| {
                    registrar.counter(names::MEDIA_REQUESTS_TOTAL, &[("outcome", outcome.label())])
                })
            }),
            registrar,
        })
    }

    /// The error counter of a gRPC code, registered on first use.
    fn error(&self, code: Code) -> &Counter {
        let index = usize::try_from(i32::from(code))
            .ok()
            .filter(|index| *index <= GRPC_CODES)
            .unwrap_or(GRPC_CODES);
        self.errors[index].get_or_init(|| {
            self.registrar
                .counter(names::REQUEST_ERRORS_TOTAL, &[("code", code_label(code))])
        })
    }
}

/// The largest gRPC status code (`UNAUTHENTICATED`).
const GRPC_CODES: usize = 16;

/// A gRPC code as the `code` label spells it.
fn code_label(code: Code) -> &'static str {
    match code {
        Code::Ok => "ok",
        Code::Cancelled => "cancelled",
        Code::Unknown => "unknown",
        Code::InvalidArgument => "invalid_argument",
        Code::DeadlineExceeded => "deadline_exceeded",
        Code::NotFound => "not_found",
        Code::AlreadyExists => "already_exists",
        Code::PermissionDenied => "permission_denied",
        Code::ResourceExhausted => "resource_exhausted",
        Code::FailedPrecondition => "failed_precondition",
        Code::Aborted => "aborted",
        Code::OutOfRange => "out_of_range",
        Code::Unimplemented => "unimplemented",
        Code::Internal => "internal",
        Code::Unavailable => "unavailable",
        Code::DataLoss => "data_loss",
        Code::Unauthenticated => "unauthenticated",
    }
}

fn describe(registrar: &Registrar) {
    registrar.describe_counter(
        names::REQUESTS_TOTAL,
        "Requests whose stream ended, by how they ended; a fan-out (n > 1) is one request with \
         the reason of its last choice to end, an abort winning.",
    );
    registrar.describe_counter(
        names::REQUESTS_REFUSED_TOTAL,
        "Requests refused before they reached the engine, by reason.",
    );
    registrar.describe_gauge(
        names::REQUESTS_ACTIVE,
        "Admitted requests still preprocessing, streaming, or unwinding an abort.",
    );
    registrar.describe_counter(
        names::REQUEST_ERRORS_TOTAL,
        "Requests that failed with a gRPC status, by code.",
    );
    registrar.describe_counter(
        names::PROMPT_TOKENS_TOTAL,
        "Prompt tokens as the engine reported them at each request's first output.",
    );
    registrar.describe_counter(
        names::PROMPT_TOKENS_CACHED_TOTAL,
        "Prompt tokens the engine served from its prefix cache.",
    );
    registrar.describe_counter(
        names::GENERATION_TOKENS_TOTAL,
        "Generated tokens streamed to callers.",
    );
    registrar.describe_counter(
        names::STRING_STOPS_TOTAL,
        "Requests this servicer ended on a stop string it matched itself.",
    );
    registrar.describe_histogram(
        names::TIME_TO_FIRST_TOKEN_SECONDS,
        "Seconds from a request's arrival at the servicer to its first output.",
    );
    registrar.describe_histogram(
        names::E2E_REQUEST_LATENCY_SECONDS,
        "Seconds from a request's arrival at the servicer to the end of its stream.",
    );
    registrar.describe_histogram(
        names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
        "Mean seconds per generated token after the first, per request (servicer clock).",
    );
    registrar.describe_histogram(
        names::INTER_TOKEN_LATENCY_SECONDS,
        "Seconds between consecutive output chunks of a choice, per token when a chunk carries \
         several (servicer clock; sampled above a pending-observation bound).",
    );
    registrar.describe_histogram(names::REQUEST_PROMPT_TOKENS, "Prompt tokens per request.");
    registrar.describe_histogram(
        names::REQUEST_GENERATION_TOKENS,
        "Generated tokens per request.",
    );
    registrar.describe_histogram(
        names::REQUEST_PARAMS_MAX_TOKENS,
        "The max_tokens parameter of requests that set it.",
    );
    registrar.describe_histogram(names::REQUEST_PARAMS_N, "The n parameter of requests.");
    registrar.describe_gauge(
        names::STATE,
        "The servicer's lifecycle state, one-hot: starting, serving, warming_up (readiness held, \
         admitting the warm-up's own requests only), draining (admission closed) or engine_down.",
    );
    registrar.describe_gauge(
        names::START_TIME_SECONDS,
        "When the servicer started, in seconds since the Unix epoch.",
    );
    registrar.describe_counter(
        names::METRICS_DROPPED_TOTAL,
        "Observations the pressure sampler dropped, by family.",
    );
    registrar.describe_histogram(
        names::MEDIA_ITEMS_PER_REQUEST,
        "Media references a request carried, by modality, counted before they are checked (a \
         refused request counts).",
    );
    registrar.describe_histogram(
        names::MEDIA_PROCESSING_SECONDS,
        "Seconds a request's worker-side media spent in a stage: waiting for a processing slot, \
         or being processed.",
    );
    registrar.describe_counter(
        names::MEDIA_REQUESTS_TOTAL,
        "Requests with worker-side media, by outcome: ok, refused before processing, or the \
         processor's verdict (client, transient, invalid, unavailable, internal).",
    );
    registrar.describe_gauge(
        names::METRICS_SCRAPE_SECONDS,
        "How long the last scrape took to render.",
    );
    registrar.describe_counter(
        names::METRICS_SCRAPES_TOTAL,
        "Scrapes answered, by HTTP status code.",
    );
}

/// One request's observation, shared by its stream and its choices; the
/// request-level families are recorded when the last holder drops.
pub(crate) struct RequestMetrics {
    metrics: Arc<ServicerMetrics>,
    slot: usize,
    arrival: Instant,
    first_output: OnceLock<Instant>,
    /// Nanoseconds from arrival to the last output chunk; 0 before any.
    last_output_ns: AtomicU64,
    /// Whether the engine's prompt figures went into the totals (at the
    /// first output that reports a non-zero prompt, once per request).
    prompt_counted: AtomicBool,
    prompt_tokens: AtomicU32,
    generation_tokens: AtomicU32,
    /// The reason of the last choice to end ([`FinishReason`] as `u8`;
    /// [`FinishReason::NONE`] while none has).
    last_finish: AtomicU8,
    aborted: AtomicBool,
    cut: AtomicBool,
    errored: AtomicBool,
}

impl RequestMetrics {
    /// The observation of one choice of this request.
    pub(crate) fn choice(self: &Arc<Self>) -> ChoiceMetrics {
        ChoiceMetrics {
            request: Arc::clone(self),
            last_chunk: None,
        }
    }

    /// A shutdown cut the request while a choice was still open.
    pub(crate) fn cut(&self) {
        self.cut.store(true, Ordering::Relaxed);
    }

    /// The stream ended with a gRPC status.
    pub(crate) fn failed(&self, code: Code) {
        if !self.errored.swap(true, Ordering::Relaxed) {
            if let Some(recorded) = &self.metrics.recorded {
                recorded.error(code).increment(1);
            }
        }
        self.last_finish
            .store(FinishReason::Error as u8, Ordering::Relaxed);
    }

    /// An output (a chunk or the terminal) reached the servicer, with the
    /// engine's prompt figures: the first non-zero prompt report goes into
    /// the totals once per request, whether or not a listener exists (the
    /// stats line reads them); `now` is the instant a listener read, and the
    /// first one is the request's first output.
    fn output(&self, now: Option<Instant>, prompt_tokens: u32, cached_tokens: u32) {
        if prompt_tokens > 0 && !self.prompt_counted.swap(true, Ordering::Relaxed) {
            let totals = self.metrics.rank_totals(self.slot);
            totals
                .prompt_tokens
                .fetch_add(u64::from(prompt_tokens), Ordering::Relaxed);
            totals
                .prompt_tokens_cached
                .fetch_add(u64::from(cached_tokens), Ordering::Relaxed);
            self.prompt_tokens.store(prompt_tokens, Ordering::Relaxed);
        }
        let Some(now) = now else {
            return;
        };
        if self.first_output.set(now).is_ok() {
            if let Some(handles) = self.metrics.rank_handles(self.slot) {
                handles
                    .time_to_first_token
                    .record(now.saturating_duration_since(self.arrival));
            }
        }
    }

    fn choice_ended(&self, reason: FinishReason) {
        if reason == FinishReason::Abort {
            self.aborted.store(true, Ordering::Relaxed);
        }
        self.last_finish.store(reason as u8, Ordering::Relaxed);
    }

    fn finish_reason(&self) -> FinishReason {
        if self.cut.load(Ordering::Relaxed) {
            return FinishReason::Cut;
        }
        if self.aborted.load(Ordering::Relaxed) {
            return FinishReason::Abort;
        }
        // No choice ended: the caller went away with the stream open, which
        // aborts the engine side, so it is counted as an abort.
        FinishReason::from_u8(self.last_finish.load(Ordering::Relaxed))
            .unwrap_or(FinishReason::Abort)
    }
}

impl Drop for RequestMetrics {
    /// The stream and every choice are gone: the request's end.
    fn drop(&mut self) {
        let Some(handles) = self.metrics.rank_handles(self.slot) else {
            return;
        };
        let now = Instant::now();
        let reason = self.finish_reason();
        handles.requests_total[reason as usize].increment(1);
        handles
            .e2e_latency
            .record(now.saturating_duration_since(self.arrival));
        let generation_tokens = self.generation_tokens.load(Ordering::Relaxed);
        if generation_tokens > 1 {
            if let Some(first) = self.first_output.get() {
                let first_ns = first.saturating_duration_since(self.arrival).as_nanos();
                let last_ns = u128::from(self.last_output_ns.load(Ordering::Relaxed));
                let span = Duration::from_nanos(
                    u64::try_from(last_ns.saturating_sub(first_ns)).unwrap_or(u64::MAX),
                );
                handles
                    .time_per_output_token
                    .record(span / (generation_tokens - 1));
            }
        }
        handles
            .request_prompt_tokens
            .record(self.prompt_tokens.load(Ordering::Relaxed));
        handles.request_generation_tokens.record(generation_tokens);
    }
}

/// The observation of one choice: the per-choice inter-token clock over the
/// request's shared observation.
pub(crate) struct ChoiceMetrics {
    request: Arc<RequestMetrics>,
    last_chunk: Option<Instant>,
}

impl ChoiceMetrics {
    /// An output chunk of `tokens` new tokens reached the servicer, with
    /// the engine's prompt figures on it. Returns the instant it read, for
    /// a caller that needs one too; `None` without a listener (nothing read).
    pub(crate) fn chunk(
        &mut self,
        tokens: u32,
        prompt_tokens: u32,
        cached_tokens: u32,
    ) -> Option<Instant> {
        let metrics = &self.request.metrics;
        if tokens > 0 {
            metrics
                .rank_totals(self.request.slot)
                .generation_tokens
                .fetch_add(u64::from(tokens), Ordering::Relaxed);
            self.request
                .generation_tokens
                .fetch_add(tokens, Ordering::Relaxed);
        }
        let now = metrics.enabled().then(Instant::now);
        self.request.output(now, prompt_tokens, cached_tokens);
        let now = now?;
        if tokens > 0 {
            if let Some(last) = self.last_chunk {
                metrics.inter_token(
                    self.request.slot,
                    now.saturating_duration_since(last) / tokens,
                );
            }
            self.last_chunk = Some(now);
            let since_arrival = now.saturating_duration_since(self.request.arrival);
            self.request.last_output_ns.store(
                u64::try_from(since_arrival.as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        Some(now)
    }

    /// The engine's terminal for this choice, with the reason it carries.
    pub(crate) fn complete(&mut self, finish_reason: &str, prompt_tokens: u32, cached_tokens: u32) {
        self.request.output(
            self.request.metrics.enabled().then(Instant::now),
            prompt_tokens,
            cached_tokens,
        );
        self.request
            .choice_ended(FinishReason::from_wire(finish_reason));
    }

    /// This servicer ended the choice on a stop string it matched.
    pub(crate) fn string_stop(&mut self) {
        self.request
            .metrics
            .rank_totals(self.request.slot)
            .string_stops
            .fetch_add(1, Ordering::Relaxed);
        self.request.choice_ended(FinishReason::Stop);
    }

    /// The choice ended on a completion this servicer synthesized (an
    /// abort), or the engine's held one, with `finish_reason`.
    pub(crate) fn ended(&mut self, finish_reason: &str) {
        self.request
            .choice_ended(FinishReason::from_wire(finish_reason));
    }

    /// The choice's stream failed with a gRPC status.
    pub(crate) fn failed(&mut self, code: Code) {
        self.request.failed(code);
    }
}

/// The finish reason of a terminal response in the vLLM proto shape every
/// wire's intermediate carries, for the per-wire modules' bookkeeping.
pub(crate) fn complete_finish_reason(response: &vllm::GenerateResponse) -> Option<&str> {
    match response.response.as_ref()? {
        vllm::generate_response::Response::Complete(complete) => {
            Some(complete.finish_reason.as_str())
        }
        vllm::generate_response::Response::Chunk(_) => None,
    }
}

/// What a scrape reads off the servicer at that moment.
pub(crate) type Probe = Arc<dyn Fn() -> Snapshot + Send + Sync>;

struct ListenerState {
    metrics: Arc<ServicerMetrics>,
    probe: Probe,
}

async fn serve_scrape(AxumState(state): AxumState<Arc<ListenerState>>) -> impl IntoResponse {
    let probe = Arc::clone(&state.probe);
    let (status, body) = state.metrics.scrape(move || probe());
    (status, [(CONTENT_TYPE, CONTENT_TYPE_PROMETHEUS)], body)
}

/// The `GET /metrics` listener on its own thread and runtime.
pub(crate) struct MetricsListener {
    thread: ServerThread,
}

impl MetricsListener {
    /// Bind `bind_address` (`host:port`; port 0 for an ephemeral one) and
    /// serve `metrics` from a dedicated thread. A bind failure is a startup
    /// error: the operator asked for the port.
    pub(crate) fn start(
        bind_address: &str,
        metrics: Arc<ServicerMetrics>,
        probe: Probe,
    ) -> Result<Self, ServicerError> {
        let state = Arc::new(ListenerState { metrics, probe });
        let thread = ServerThread::start_current_thread(
            "smg-servicer-metrics",
            bind_address,
            move |listener: TcpListener, shutdown: Shutdown, _last_error| async move {
                let upkeep = Arc::clone(&state);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the upkeep tick lives as long as the listener's runtime; the runtime drop ends it"
                )]
                let _upkeep = tokio::spawn(async move {
                    let mut ticker = tokio::time::interval(UPKEEP_INTERVAL);
                    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    loop {
                        ticker.tick().await;
                        upkeep.metrics.upkeep();
                    }
                });
                info!(address = %listener.local_addr().map(|a| a.to_string()).unwrap_or_default(), "servicer metrics listening (/metrics)");
                let app = Router::new()
                    .route("/metrics", get(serve_scrape))
                    .with_state(state);
                // No drain on stop: the runtime goes with the thread, which
                // closes every connection, in flight or idle, at once.
                tokio::select! {
                    served = axum::serve(listener, app).into_future() => served.map_err(|error| error.to_string()),
                    () = shutdown.map(|_| ()) => Ok(()),
                }
            },
        )
        .map_err(|error| match error {
            ServicerError::Startup(message) => {
                ServicerError::Startup(format!("metrics listener: {message}"))
            }
            other => other,
        })?;
        Ok(Self { thread })
    }

    /// The bound address.
    pub(crate) fn address(&self) -> SocketAddr {
        self.thread.address()
    }

    /// Close the listener and join its thread, within what is left of a
    /// stop budget (never less than a floor: closing takes no drain).
    pub(crate) fn stop(&self, remaining: Duration) -> Result<(), ServicerError> {
        self.thread.stop(remaining.max(METRICS_STOP_FLOOR))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpStream,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::testing::http_get;

    fn identity(ranks: usize) -> Identity {
        Identity {
            model: "served".to_string(),
            wire: "test",
            ranks,
            max_model_len: 100,
            matches_string_stops: true,
            processes_media: true,
        }
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            active: 0,
            state: Some(ServicerState::Serving),
        }
    }

    fn sample(body: &str, name: &str, labels: &str) -> Option<f64> {
        let prefix = format!("{name}{{");
        body.lines()
            .find(|line| {
                line.starts_with(&prefix) && labels.split(',').all(|label| line.contains(label))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
    }

    #[test]
    fn the_token_count_buckets_follow_the_1_2_5_series_up_to_the_context_length() {
        assert_eq!(
            one_two_five_buckets(100),
            [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        );
        assert_eq!(one_two_five_buckets(0), [1.0]);
        assert_eq!(one_two_five_buckets(4096).len(), 11);
    }

    #[test]
    fn a_request_records_its_families_when_the_last_holder_drops() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let arrival = Instant::now();
        let request = metrics.request(arrival, Some(0), 3, Some(8), 1);
        let mut choice = request.choice();
        assert!(choice.chunk(1, 3, 1).is_some());
        choice.chunk(2, 3, 1);
        choice.complete("length", 3, 1);
        drop(choice);
        drop(request);
        let body = metrics.render(snapshot());
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"length\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::PROMPT_TOKENS_TOTAL, "rank=\"0\""),
            Some(3.0)
        );
        assert_eq!(
            sample(&body, names::PROMPT_TOKENS_CACHED_TOTAL, "rank=\"0\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::GENERATION_TOKENS_TOTAL, "rank=\"0\""),
            Some(3.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::TIME_TO_FIRST_TOKEN_SECONDS),
                "rank=\"0\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::INTER_TOKEN_LATENCY_SECONDS),
                "rank=\"0\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS),
                "rank=\"0\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::E2E_REQUEST_LATENCY_SECONDS),
                "rank=\"0\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_sum", names::REQUEST_PROMPT_TOKENS),
                "rank=\"0\""
            ),
            Some(3.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_sum", names::REQUEST_GENERATION_TOKENS),
                "rank=\"0\""
            ),
            Some(3.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_sum", names::REQUEST_PARAMS_MAX_TOKENS),
                "rank=\"0\""
            ),
            Some(8.0)
        );
        assert_eq!(sample(&body, names::STATE, "state=\"serving\""), Some(1.0));
        assert_eq!(sample(&body, names::STATE, "state=\"starting\""), Some(0.0));
        assert_eq!(
            sample(&body, names::STATE, "state=\"warming_up\""),
            Some(0.0)
        );
        assert!(body.contains("# HELP smg_servicer_requests_total "));
    }

    /// Without a listener nothing is recorded or read beyond the token
    /// totals; the per-token path reads no clock.
    #[test]
    fn without_a_listener_only_the_token_totals_count() {
        let metrics = ServicerMetrics::new(&identity(1), false).unwrap();
        let request = metrics.request(Instant::now(), Some(0), 3, Some(8), 1);
        let mut choice = request.choice();
        assert!(choice.chunk(4, 3, 1).is_none());
        choice.complete("stop", 3, 1);
        drop(choice);
        drop(request);
        assert_eq!(metrics.generation_tokens_total(), 4);
        assert_eq!(metrics.prompt_tokens_total(), 3, "the stats line's figure");
        assert_eq!(metrics.render(snapshot()), "");
    }

    /// The prompt goes into the totals at the first output that reports it
    /// (a chunk may precede the engine's prefill figures), once per request,
    /// with or without a listener.
    #[test]
    fn the_prompt_counts_once_at_the_first_non_zero_report() {
        for enabled in [false, true] {
            let metrics = ServicerMetrics::new(&identity(1), enabled).unwrap();
            let request = metrics.request(Instant::now(), Some(0), 3, None, 1);
            let mut choice = request.choice();
            choice.chunk(1, 0, 0);
            choice.chunk(1, 5, 2);
            choice.chunk(1, 5, 2);
            choice.complete("stop", 5, 2);
            drop(choice);
            drop(request);
            assert_eq!(metrics.prompt_tokens_total(), 5, "enabled={enabled}");
            assert_eq!(
                metrics
                    .rank_totals(0)
                    .prompt_tokens_cached
                    .load(Ordering::Relaxed),
                2
            );
        }
    }

    /// A fan-out is one request: its reason is the last choice's, an abort
    /// wins, and the first output of any choice is the request's.
    #[test]
    fn a_fan_out_counts_one_request_with_the_last_reason_and_an_abort_winning() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let request = metrics.request(Instant::now(), Some(0), 3, None, 2);
        let mut first = request.choice();
        let mut second = request.choice();
        first.chunk(1, 3, 0);
        second.chunk(1, 3, 0);
        first.complete("stop", 3, 0);
        second.complete("length", 3, 0);
        drop((first, second, request));
        let request = metrics.request(Instant::now(), Some(0), 3, None, 2);
        let mut first = request.choice();
        let mut second = request.choice();
        first.ended("abort");
        second.complete("stop", 3, 0);
        drop((first, second, request));
        let body = metrics.render(snapshot());
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"length\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"abort\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"stop\""),
            Some(0.0)
        );
        assert_eq!(
            sample(&body, names::PROMPT_TOKENS_TOTAL, "rank=\"0\""),
            Some(6.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::TIME_TO_FIRST_TOKEN_SECONDS),
                "rank=\"0\""
            ),
            Some(2.0)
        );
    }

    /// A stream dropped with no choice ended, a cut and a failure each
    /// have their reason; a failure also counts under its code.
    #[test]
    fn a_dropped_cut_or_failed_request_counts_under_its_reason() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        drop(metrics.request(Instant::now(), Some(0), 3, None, 1));
        let cut = metrics.request(Instant::now(), Some(0), 3, None, 1);
        cut.cut();
        drop(cut);
        let failed = metrics.request(Instant::now(), Some(0), 3, None, 1);
        failed.failed(Code::Internal);
        failed.failed(Code::Internal);
        drop(failed);
        metrics.request_not_started(&Status::already_exists("dup"));
        metrics.request_not_started(&Status::unavailable(DRAINING_MESSAGE));
        metrics.request_not_started(&Status::invalid_argument("bad"));
        metrics.request_not_started(&Status::unavailable("engine is still connecting"));
        let body = metrics.render(snapshot());
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"abort\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"cut\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"error\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUEST_ERRORS_TOTAL, "code=\"internal\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUEST_ERRORS_TOTAL, "code=\"unavailable\""),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                names::REQUESTS_REFUSED_TOTAL,
                "reason=\"duplicate_id\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_REFUSED_TOTAL, "reason=\"draining\""),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &body,
                names::REQUESTS_REFUSED_TOTAL,
                "reason=\"invalid_argument\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::REQUESTS_REFUSED_TOTAL, "reason=\"media\""),
            Some(0.0)
        );
    }

    /// Every rank has its own series; an index past the configured width
    /// lands under `other`, which exists only once used.
    #[test]
    fn ranks_have_their_own_series_and_an_unannounced_one_lands_under_other() {
        let metrics = ServicerMetrics::new(&identity(2), true).unwrap();
        let before = metrics.render(snapshot());
        assert!(!before.contains("rank=\"other\""));
        for rank in [Some(0), Some(1), Some(1), Some(7), None] {
            let request = metrics.request(Instant::now(), rank, 3, None, 1);
            let mut choice = request.choice();
            choice.chunk(1, 3, 0);
            choice.complete("stop", 3, 0);
        }
        let body = metrics.render(snapshot());
        let stops = |rank: &str| {
            sample(
                &body,
                names::REQUESTS_TOTAL,
                &format!("finish_reason=\"stop\",model=\"served\",rank=\"{rank}\""),
            )
        };
        assert_eq!(stops("0"), Some(1.0));
        assert_eq!(stops("1"), Some(2.0));
        assert_eq!(stops("other"), Some(2.0));
        assert_eq!(metrics.generation_tokens_total(), 5);
    }

    #[test]
    fn the_inter_token_sampler_drops_and_counts_above_the_bound() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        for _ in 0..(ITL_PENDING_BOUND + 10) {
            metrics.inter_token(0, Duration::from_millis(1));
        }
        assert_eq!(metrics.inter_token_dropped.load(Ordering::Relaxed), 10);
        assert_eq!(
            metrics.inter_token_pending.load(Ordering::Relaxed),
            ITL_PENDING_BOUND
        );
        let body = metrics.render(snapshot());
        assert_eq!(
            sample(
                &body,
                names::METRICS_DROPPED_TOTAL,
                "family=\"inter_token_latency\""
            ),
            Some(10.0)
        );
        assert_eq!(
            sample(
                &body,
                &format!("{}_count", names::INTER_TOKEN_LATENCY_SECONDS),
                "rank=\"0\""
            ),
            Some(f64::from(ITL_PENDING_BOUND))
        );
        // The scrape drained the samples: the sampler starts over.
        assert_eq!(metrics.inter_token_pending.load(Ordering::Relaxed), 0);
        metrics.inter_token(0, Duration::from_millis(1));
        assert_eq!(metrics.inter_token_dropped.load(Ordering::Relaxed), 10);
    }

    /// The render time of a scrape at one rank and at eight (the handle
    /// sets of a data-parallel group), printed for the Numbers table: a
    /// scrape runs on the listener's thread, never on the request runtime,
    /// so this is a cost in milliseconds per scrape, not per request.
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "the render times are the test's report"
    )]
    fn a_scrape_renders_in_milliseconds_at_one_and_eight_ranks() {
        for ranks in [1usize, 8] {
            let metrics = ServicerMetrics::new(&identity(ranks), true).unwrap();
            for rank in 0..ranks {
                for _ in 0..16 {
                    let request = metrics.request(Instant::now(), Some(rank as u32), 3, Some(8), 1);
                    let mut choice = request.choice();
                    choice.chunk(1, 3, 1);
                    choice.chunk(1, 3, 1);
                    choice.complete("stop", 3, 1);
                }
            }
            let mut renders = Vec::new();
            let mut bytes = 0;
            for _ in 0..20 {
                let started = Instant::now();
                bytes = metrics.render(snapshot()).len();
                renders.push(started.elapsed());
            }
            renders.sort();
            eprintln!(
                "scrape at {ranks} rank(s): median {:.3} ms, max {:.3} ms, {bytes} bytes",
                renders[renders.len() / 2].as_secs_f64() * 1e3,
                renders[renders.len() - 1].as_secs_f64() * 1e3
            );
            assert!(renders[renders.len() / 2] < Duration::from_millis(100));
        }
    }

    /// Two servicers in one process keep two registries: nothing is
    /// process-global.
    #[test]
    fn a_process_hosting_two_servicers_keeps_two_registries() {
        let first = ServicerMetrics::new(&identity(1), true).unwrap();
        let second = ServicerMetrics::new(
            &Identity {
                model: "other".to_string(),
                ..identity(1)
            },
            true,
        )
        .unwrap();
        drop(first.request(Instant::now(), Some(0), 1, None, 1));
        let first_body = first.render(snapshot());
        let second_body = second.render(snapshot());
        assert_eq!(
            sample(
                &first_body,
                names::REQUESTS_TOTAL,
                "finish_reason=\"abort\",model=\"served\""
            ),
            Some(1.0)
        );
        assert_eq!(
            sample(
                &second_body,
                names::REQUESTS_TOTAL,
                "finish_reason=\"abort\",model=\"other\""
            ),
            Some(0.0)
        );
        assert!(!second_body.contains("model=\"served\""));
    }

    #[test]
    fn the_listener_serves_a_scrape_and_an_ipv6_host_binds_with_or_without_brackets() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let probe: Probe = Arc::new(snapshot);
        let listener =
            MetricsListener::start("127.0.0.1:0", Arc::clone(&metrics), Arc::clone(&probe))
                .unwrap();
        let (status, body) = http_get(listener.address(), "/metrics");
        assert_eq!(status, 200);
        assert!(body.contains("smg_servicer_state{"));
        assert_eq!(http_get(listener.address(), "/other").0, 404);
        listener.stop(Duration::from_secs(5)).unwrap();
        // Nothing to assert on a host without an IPv6 loopback.
        if std::net::TcpListener::bind("[::1]:0").is_err() {
            return;
        }
        for bind in ["[::1]:0", "::1:0"] {
            let listener =
                MetricsListener::start(bind, Arc::clone(&metrics), Arc::clone(&probe)).unwrap();
            assert!(
                listener.address().is_ipv6(),
                "{bind} bound {}",
                listener.address()
            );
            assert_eq!(http_get(listener.address(), "/metrics").0, 200);
            listener.stop(Duration::from_secs(5)).unwrap();
        }
    }

    #[test]
    fn a_busy_metrics_port_is_a_startup_error() {
        let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let error = MetricsListener::start(
            &busy.local_addr().unwrap().to_string(),
            metrics,
            Arc::new(snapshot),
        )
        .err()
        .expect("a busy port is refused");
        let text = error.to_string();
        assert!(text.contains("metrics listener: failed to bind"), "{text}");
    }

    /// A scraper holding a connection does not hold the stop: the listener
    /// closes at once, inside the budget, and the port refuses after.
    #[test]
    fn stop_closes_the_listener_inside_the_budget_with_a_scraper_connected() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let listener = MetricsListener::start("127.0.0.1:0", metrics, Arc::new(snapshot)).unwrap();
        let address = listener.address();
        let mut held = TcpStream::connect(address).unwrap();
        write!(held, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut first = [0u8; 16];
        held.read_exact(&mut first).unwrap();
        assert!(first.starts_with(b"HTTP/1.1 200"));
        let started = Instant::now();
        listener.stop(Duration::from_secs(5)).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        held.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut rest = Vec::new();
        let _ = held.read_to_end(&mut rest);
        assert!(TcpStream::connect(address).is_err(), "the port is closed");
    }

    /// A probe that panics fails that scrape alone with a 500; the next
    /// scrape renders again and the failure is counted.
    #[test]
    fn a_failed_scrape_answers_500_once_and_the_next_renders_again() {
        let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let probe_calls = Arc::clone(&calls);
        let probe: Probe = Arc::new(move || {
            assert_ne!(
                probe_calls.fetch_add(1, Ordering::Relaxed),
                0,
                "the first scrape fails"
            );
            snapshot()
        });
        let listener = MetricsListener::start("127.0.0.1:0", metrics, probe).unwrap();
        let (status, body) = http_get(listener.address(), "/metrics");
        assert_eq!(status, 500);
        assert!(body.contains("the next scrape renders again"));
        let (status, body) = http_get(listener.address(), "/metrics");
        assert_eq!(status, 200);
        assert_eq!(
            sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"500\""),
            Some(1.0)
        );
        assert_eq!(
            sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"200\""),
            Some(0.0),
            "the current scrape counts once it is rendered"
        );
        let (_, body) = http_get(listener.address(), "/metrics");
        assert_eq!(
            sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"200\""),
            Some(1.0)
        );
        assert!(sample(&body, names::METRICS_SCRAPE_SECONDS, "wire=\"test\"").is_some());
        listener.stop(Duration::from_secs(5)).unwrap();
    }
}
