//! The metrics a router emits. They live here so the gateway and every
//! external router record the same names with the same labels.

use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use dashmap::{mapref::entry::Entry, DashMap};
use http::StatusCode;
use metrics::{counter, histogram};
use once_cell::sync::Lazy;

const BOUNDED_LABEL_SENTINEL: &str = metrics_labels::MODEL_OTHER;

/// Max distinct client-supplied model labels retained before collapsing to the
/// sentinel. A gateway fronts far fewer real models than this; the cap only bites
/// on adversarial or unvalidated input.
const MAX_MODEL_LABELS: usize = 1024;

/// Max distinct client/model-controlled MCP tool-name labels.
const MAX_TOOL_LABELS: usize = 1024;

static MODEL_LABELS: Lazy<BoundedLabels> = Lazy::new(|| BoundedLabels::new(MAX_MODEL_LABELS));
static TOOL_LABELS: Lazy<BoundedLabels> = Lazy::new(|| BoundedLabels::new(MAX_TOOL_LABELS));

/// Max distinct worker-group labels (the operator's `traffic_group` label
/// values) retained before collapsing to the sentinel.
const MAX_GROUP_LABELS: usize = 64;

static GROUP_LABELS: Lazy<BoundedLabels> = Lazy::new(|| BoundedLabels::new(MAX_GROUP_LABELS));
static BOUNDED_LABEL_SENTINEL_ARC: Lazy<Arc<str>> = Lazy::new(|| Arc::from(BOUNDED_LABEL_SENTINEL));

/// Intern a client-controlled label with a hard cardinality cap.
///
/// Distinct values beyond `cap` collapse to a shared sentinel, so untrusted input
/// (client-supplied model names, model-generated tool names) cannot grow the
/// interner — or the metric's Prometheus series set — without bound. Unlike an LRU,
/// admitted values are never evicted and re-admitted, which would keep minting new
/// series in the recorder even as the map churned.
/// A label set that admits at most `cap` distinct values; everything past
/// the cap maps to the sentinel. Admission reserves a slot before touching
/// the map, so a burst of distinct labels cannot overshoot the cap.
struct BoundedLabels {
    map: DashMap<String, Arc<str>>,
    admitted: AtomicUsize,
    cap: usize,
}

impl BoundedLabels {
    fn new(cap: usize) -> Self {
        Self {
            map: DashMap::new(),
            admitted: AtomicUsize::new(0),
            cap,
        }
    }

    fn intern(&self, s: &str) -> Arc<str> {
        if let Some(entry) = self.map.get(s) {
            return Arc::clone(entry.value());
        }
        let mut admitted = self.admitted.load(Ordering::Relaxed);
        loop {
            if admitted >= self.cap {
                return Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
            }
            match self.admitted.compare_exchange_weak(
                admitted,
                admitted + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => admitted = now,
            }
        }
        match self.map.entry(s.to_string()) {
            Entry::Occupied(entry) => {
                // Someone else admitted this label first; give the slot back.
                self.admitted.fetch_sub(1, Ordering::AcqRel);
                Arc::clone(entry.get())
            }
            Entry::Vacant(entry) => Arc::clone(&*entry.insert(Arc::from(s))),
        }
    }
}

/// Intern a client-supplied model label, bounded by `MAX_MODEL_LABELS`.
pub fn intern_model_label(model_id: &str) -> Arc<str> {
    MODEL_LABELS.intern(model_id)
}

/// Intern a client/model-controlled MCP tool-name label, bounded by
/// `MAX_TOOL_LABELS`.
pub fn intern_tool_label(tool_name: &str) -> Arc<str> {
    TOOL_LABELS.intern(tool_name)
}

/// Intern an operator-controlled worker-group label, bounded by
/// [`MAX_GROUP_LABELS`].
pub fn intern_group_label(group: &str) -> Arc<str> {
    GROUP_LABELS.intern(group)
}

pub const STREAMING_TRUE: &str = "true";
pub const STREAMING_FALSE: &str = "false";

pub const fn bool_to_static_str(b: bool) -> &'static str {
    if b {
        STREAMING_TRUE
    } else {
        STREAMING_FALSE
    }
}

pub mod metrics_labels {
    /// The `model` label of a request that names no served model: an unknown
    /// name refused with 404, or a value past the model-label cap. One fixed
    /// value, as the `path` label is `other` for an unmatched route, so a
    /// client cannot mint label values.
    pub const MODEL_OTHER: &str = "other";

    // Router types
    pub const ROUTER_OPENAI: &str = "openai";
    pub const ROUTER_HTTP: &str = "http";
    pub const ROUTER_GRPC: &str = "grpc";

    // Backend types
    pub const BACKEND_REGULAR: &str = "regular";
    pub const BACKEND_PD: &str = "pd";
    pub const BACKEND_EXTERNAL: &str = "external";
    pub const BACKEND_HARMONY: &str = "harmony";

    // Connection modes
    pub const CONNECTION_HTTP: &str = "http";
    pub const CONNECTION_GRPC: &str = "grpc";
    pub const CONNECTION_ZMQ: &str = "zmq";

    // Endpoints
    pub const ENDPOINT_CHAT: &str = "chat";
    pub const ENDPOINT_GENERATE: &str = "generate";
    pub const ENDPOINT_RESPONSES: &str = "responses";
    pub const ENDPOINT_DECISIONS: &str = "decisions";
    pub const ENDPOINT_SYSTEMONE: &str = "systemone";
    pub const ENDPOINT_COMPLETIONS: &str = "completions";
    pub const ENDPOINT_RERANK: &str = "rerank";
    pub const ENDPOINT_EMBEDDINGS: &str = "embeddings";
    pub const ENDPOINT_CLASSIFY: &str = "classify";
    pub const ENDPOINT_MESSAGES: &str = "messages";
    pub const ENDPOINT_REALTIME: &str = "realtime";
    pub const ENDPOINT_REALTIME_SESSIONS: &str = "realtime_sessions";
    pub const ENDPOINT_REALTIME_CLIENT_SECRETS: &str = "realtime_client_secrets";
    pub const ENDPOINT_REALTIME_TRANSCRIPTION: &str = "realtime_transcription";
    pub const ENDPOINT_AUDIO_TRANSCRIPTIONS: &str = "audio_transcriptions";

    // Connection modes
    pub const CONNECTION_WEBSOCKET: &str = "websocket";
    pub const CONNECTION_WEBRTC: &str = "webrtc";

    // Worker types
    pub const WORKER_REGULAR: &str = "regular";
    pub const WORKER_PREFILL: &str = "prefill";
    pub const WORKER_DECODE: &str = "decode";
    pub const WORKER_ENCODE: &str = "encode";
    pub const WORKER_HTTP: &str = "http";
    pub const WORKER_GRPC: &str = "grpc";

    // Token types
    pub const TOKEN_INPUT: &str = "input";
    pub const TOKEN_OUTPUT: &str = "output";
    /// The prompt tokens the engine served from its cache: a share of
    /// `input`, counted again under its own type.
    pub const TOKEN_CACHED: &str = "cached";

    // PD KV connector modes (smg_pd_kv_connector_mode_total)
    pub const KV_CONNECTOR_MOONCAKE: &str = "mooncake";
    pub const KV_CONNECTOR_MORIIO: &str = "moriio";
    pub const KV_CONNECTOR_NIXL: &str = "nixl";
    pub const KV_CONNECTOR_PASSTHROUGH: &str = "passthrough";
    pub const KV_CONNECTOR_STORE: &str = "store";

    // Storage types
    pub const STORAGE_RESPONSE: &str = "response";
    pub const STORAGE_CONVERSATION: &str = "conversation";
    pub const STORAGE_CONVERSATION_ITEM: &str = "conversation_item";

    // Database operations
    pub const DB_OP_GET: &str = "get";
    pub const DB_OP_PUT: &str = "put";
    pub const DB_OP_DELETE: &str = "delete";
    pub const DB_OP_LIST: &str = "list";

    // Result types
    pub const RESULT_SUCCESS: &str = "success";
    pub const RESULT_ERROR: &str = "error";
    pub const RESULT_TIMEOUT: &str = "timeout";
    pub const RESULT_NOT_FOUND: &str = "not_found";

    // Discovery sources
    pub const DISCOVERY_STATIC: &str = "static";
    pub const DISCOVERY_KUBERNETES: &str = "kubernetes";
    pub const DISCOVERY_FILE: &str = "file";
    pub const DISCOVERY_SLURM: &str = "slurm";
    pub const DISCOVERY_CONSUL: &str = "consul";
    pub const DISCOVERY_MANUAL: &str = "manual";

    // Discovery registration results
    pub const REGISTRATION_SUCCESS: &str = "success";
    pub const REGISTRATION_FAILED: &str = "failed";
    pub const DEREGISTRATION_RECONCILED: &str = "reconciled";

    // Discovery snapshot errors: the source could not be read, or was read
    // and rejected as invalid
    pub const SNAPSHOT_ERROR_UNREADABLE: &str = "unreadable";
    pub const SNAPSHOT_ERROR_INVALID: &str = "invalid";

    // Discovered workers a provider left out of an otherwise good snapshot:
    // their record could not be used, or one started earlier at their address
    pub const DISCOVERY_SKIP_INVALID: &str = "invalid";
    pub const DISCOVERY_SKIP_CONFLICT: &str = "conflict";

    // Rate limit results
    pub const RATE_LIMIT_ALLOWED: &str = "allowed";
    pub const RATE_LIMIT_REJECTED: &str = "rejected";

    // Admission rejection reasons
    pub const ADMISSION_REJECTED_FULL: &str = "full";
    pub const ADMISSION_REJECTED_TIMEOUT: &str = "timeout";
    /// Admitted, then held inside the gateway ahead of worker selection for
    /// longer than the queue timeout.
    pub const ADMISSION_REJECTED_SELECTION_TIMEOUT: &str = "selection_timeout";

    // Circuit breaker states
    pub const CB_CLOSED: &str = "closed";
    pub const CB_OPEN: &str = "open";
    pub const CB_HALF_OPEN: &str = "half_open";

    // Circuit breaker outcomes
    pub const CB_SUCCESS: &str = "success";
    pub const CB_FAILURE: &str = "failure";

    // Router error types
    pub const ERROR_NO_WORKERS: &str = "no_workers";
    pub const ERROR_TIMEOUT: &str = "timeout";
    pub const ERROR_BACKEND: &str = "backend_error";
    pub const ERROR_VALIDATION: &str = "validation_error";
    /// Capacity pushback (a 429): the worker, or the gateway's own admission,
    /// turned the request away for now. Named as the client reads it in the
    /// error body's `type`.
    pub const ERROR_RATE_LIMIT: &str = "rate_limit_error";
    /// A refusal the gateway answered on its own, before any worker took the
    /// request: every worker of the model unavailable (unhealthy or behind an
    /// open circuit breaker), the overload shed, a wait past the selection
    /// queue's bound. No backend answered, so it is not a backend's error.
    pub const ERROR_UNAVAILABLE: &str = "unavailable";
    pub const ERROR_INTERNAL: &str = "internal_error";
    /// The engine itself reported that the request failed (its finish
    /// reason was `error`), as opposed to a failed call to the worker.
    pub const ERROR_ENGINE: &str = "engine_error";
    /// The engine could not continue the request under the request's own
    /// constraint (a grammar it could not keep to): the request's outcome,
    /// not the worker's failure; the client reads a 422 `constraint_failed`.
    pub const ERROR_CONSTRAINT: &str = "constraint_error";

    // Router error reasons: the cause behind an `error_type`, as far as the
    // router that counted the error could tell it. The set is fixed; a
    // failure outside it counts as `other`.
    /// The call or the stream ran past its time bound: the worker answered
    /// DEADLINE_EXCEEDED, a side was silent past its liveness bound, an
    /// upstream call timed out (a 504 or 408).
    pub const REASON_DEADLINE: &str = "deadline";
    /// The worker's connection failed under the call: an HTTP/2 or socket
    /// fault below the status, the worker's side cancelling the call, a
    /// connect failure, a relayed body cut short (a 502).
    pub const REASON_TRANSPORT_DISCONNECT: &str = "transport_disconnect";
    /// The engine finished the request with an error of its own.
    pub const REASON_ENGINE_ERROR: &str = "engine_error";
    /// The worker answered INTERNAL (not the engine's own report), UNKNOWN or
    /// DATA_LOSS, or a 500 of its own.
    pub const REASON_ENGINE_INTERNAL: &str = "engine_internal";
    /// The worker answered UNAVAILABLE or a 503 of its own.
    pub const REASON_ENGINE_UNAVAILABLE: &str = "engine_unavailable";
    /// The worker pushed the request back for capacity (RESOURCE_EXHAUSTED,
    /// a 429 of its own).
    pub const REASON_OVERLOADED: &str = "overloaded";
    /// The gateway's own tenant rate limit refused the request.
    pub const REASON_RATE_LIMITED: &str = "rate_limited";
    /// The gateway shed the request on a capacity bound of its own: its
    /// overload protection, a full prefill admission queue, or another
    /// budget or cap it holds (a 429 under a code of the gateway's).
    pub const REASON_OVERLOAD_SHED: &str = "overload_shed";
    /// The request waited longer than an admission bound inside the gateway:
    /// the selection queue's timeout, or the prefill admission queue's.
    pub const REASON_SELECTION_QUEUE_TIMEOUT: &str = "selection_queue_timeout";
    /// Every worker of the model was unavailable (unhealthy, behind an open
    /// circuit breaker, or declined by the policy).
    pub const REASON_NO_AVAILABLE_WORKERS: &str = "no_available_workers";
    /// No worker of the model has what the request needs (media
    /// processing, a compatible prefill/decode pair): a configuration
    /// verdict, not a pool state.
    pub const REASON_NO_CAPABLE_WORKER: &str = "no_capable_worker";
    /// No worker serves the model the request named.
    pub const REASON_MODEL_NOT_FOUND: &str = "model_not_found";
    /// The request's own fault: a 400, INVALID_ARGUMENT, FAILED_PRECONDITION,
    /// OUT_OF_RANGE, missing or malformed credentials.
    pub const REASON_INVALID_REQUEST: &str = "invalid_request";
    /// The request's media failed (a reference the worker could not fetch
    /// within its budget, an item over the size limit).
    pub const REASON_MEDIA_FAULT: &str = "media_fault";
    /// The engine could not continue the request under the request's own
    /// constraint.
    pub const REASON_CONSTRAINT_FAILED: &str = "constraint_failed";
    /// The worker answered, but its answer could not be read.
    pub const REASON_BAD_RESPONSE: &str = "bad_response";
    /// The gateway's own processing failed.
    pub const REASON_INTERNAL: &str = "internal";
    /// A status outside the classes above.
    pub const REASON_OTHER: &str = "other";
}

/// The `reason` label of a router error known by its HTTP status alone: a
/// worker's answer relayed as it came, or a relayed body cut short. The
/// status is the worker's, so a 5xx is read as the worker's failure (the
/// gateway's own failures carry their code, and the gateway's routers read
/// the reason off the code first).
pub fn error_reason_from_status(status: StatusCode) -> &'static str {
    use metrics_labels::*;
    match status.as_u16() {
        400 => REASON_INVALID_REQUEST,
        404 => REASON_MODEL_NOT_FOUND,
        408 | 504 => REASON_DEADLINE,
        429 => REASON_OVERLOADED,
        500 => REASON_ENGINE_INTERNAL,
        502 => REASON_TRANSPORT_DISCONNECT,
        503 => REASON_ENGINE_UNAVAILABLE,
        _ => REASON_OTHER,
    }
}

/// The `reason` label of a request the router could not place: the pool's
/// refusal is a 404 for a model no worker serves and a 503 for a model whose
/// workers are all unavailable.
pub fn error_reason_for_refused_selection(status: StatusCode) -> &'static str {
    if status == StatusCode::NOT_FOUND {
        metrics_labels::REASON_MODEL_NOT_FOUND
    } else {
        metrics_labels::REASON_NO_AVAILABLE_WORKERS
    }
}

/// Per-thread caches of the handles the per-request records go through.
///
/// The `metrics` macros build a key on every call (the labels in a `Vec`,
/// the hash) and search the recorder's registry for the series under one of
/// its shard locks, after the dynamic labels were interned through shared
/// maps. A request records a handful of metrics that way from whichever
/// runtime worker is polling it, against the same few label sets, so at a
/// thousand requests per second the interners' and the registry's locks are
/// taken thousands of times per second from every worker. A handle the
/// recorder hands out refers to the series' shared value, so each worker
/// keeps the handles it has used in a map of its own and a record is a local
/// lookup plus the atomic update of the value. The label sets are bounded
/// (static values, the route templates, the capped model labels), and so is
/// every map. Nothing is kept before the recorder is installed: a handle
/// from the no-op recorder would be a dead end for the series.
pub mod handles {
    use std::{
        cell::{Cell, RefCell},
        collections::HashMap,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use metrics::{Counter, Histogram};

    use super::{intern_model_label, BOUNDED_LABEL_SENTINEL_ARC, MAX_MODEL_LABELS};

    static RECORDER_INSTALLED: AtomicBool = AtomicBool::new(false);

    /// The most handles one map keeps: the model labels the interner admits
    /// and its sentinel. The other dynamic labels (route templates, status
    /// codes, worker addresses) are far fewer. Past the cap a record builds
    /// its handle and uses it without keeping it, so a flood of distinct
    /// labels, which the interner collapses to its sentinel, cannot grow a
    /// map.
    pub const MAX_KEPT_PER_MAP: usize = MAX_MODEL_LABELS + 1;

    /// The process's Prometheus recorder is installed: records keep their
    /// handles from here on.
    pub fn recorder_installed() {
        RECORDER_INSTALLED.store(true, Ordering::Release);
    }

    /// Whether a record on this thread keeps the handle it builds.
    pub fn keep_handles() -> bool {
        FORCED
            .get()
            .unwrap_or_else(|| RECORDER_INSTALLED.load(Ordering::Acquire))
    }

    /// For tests: have the records of this thread keep their handles (or
    /// build keys), whatever the installed flag says; `None` gives the flag
    /// its say back. A handle kept under a test's local recorder belongs to
    /// that recorder, so a test that forces keeping stays on its own thread
    /// (the harness gives every test one) and its handles die with it.
    pub fn force_on_this_thread(keep: Option<bool>) {
        FORCED.set(keep);
    }

    /// Handles by a static label set, then by the model label as the
    /// interner gave it: the key is the interner's own allocation, shared
    /// with every thread that kept the handle.
    pub type ByLabels<L, H> = HashMap<L, HashMap<Arc<str>, H>>;

    /// The static labels of `smg_router_requests_total`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub(super) struct RequestLabels {
        pub(super) router_type: &'static str,
        pub(super) backend_type: &'static str,
        pub(super) connection_mode: &'static str,
        pub(super) endpoint: &'static str,
        pub(super) streaming: &'static str,
    }

    /// The static labels of `smg_router_request_duration_seconds`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub(super) struct DurationLabels {
        pub(super) router_type: &'static str,
        pub(super) backend_type: &'static str,
        pub(super) connection_mode: &'static str,
        pub(super) endpoint: &'static str,
    }

    thread_local! {
        /// A test's say for this thread, over the installed flag.
        static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
        /// `smg_router_requests_total`.
        pub(super) static ROUTER_REQUESTS: RefCell<ByLabels<RequestLabels, Counter>> = RefCell::new(HashMap::new());
        /// `smg_router_request_duration_seconds`.
        pub(super) static ROUTER_DURATIONS: RefCell<ByLabels<DurationLabels, Histogram>> = RefCell::new(HashMap::new());
    }

    /// Uses the handle under `key`, building it on first sight and keeping
    /// it while the map is under [`MAX_KEPT_PER_MAP`]. The key is only
    /// allocated for a new entry.
    pub fn with_handle<H>(
        map: &mut HashMap<String, H>,
        key: &str,
        build: impl FnOnce() -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if let Some(handle) = map.get(key) {
            use_handle(handle);
            return;
        }
        let handle = build();
        use_handle(&handle);
        if map.len() < MAX_KEPT_PER_MAP {
            map.insert(key.to_owned(), handle);
        }
    }

    /// Whether `label` is the sentinel the interner gives a name past its
    /// cap (or refused in a race at the cap).
    pub fn is_sentinel(label: &Arc<str>) -> bool {
        Arc::ptr_eq(label, &BOUNDED_LABEL_SENTINEL_ARC)
    }

    /// Whether a handle built under `label` is kept in a map of `len`
    /// entries: under the cap, and never for the sentinel. A name the
    /// interner refused is used and dropped, so a flood of distinct names
    /// grows no map, and a name refused in a race does not pin the
    /// sentinel's series under its own spelling for the thread's life.
    pub fn kept(len: usize, label: &Arc<str>) -> bool {
        len < MAX_KEPT_PER_MAP && !is_sentinel(label)
    }

    /// Uses the handle under the model label of `model_id`, building it on
    /// first sight with the label the interner gives and keeping it, when
    /// [`kept`] says so, under that same `Arc<str>`: the map holds the
    /// interner's allocation, not a copy of a client-controlled name.
    pub fn with_model_handle<H>(
        map: &mut HashMap<Arc<str>, H>,
        model_id: &str,
        build: impl FnOnce(Arc<str>) -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if let Some(handle) = map.get(model_id) {
            use_handle(handle);
            return;
        }
        let label = intern_model_label(model_id);
        let keep = kept(map.len(), &label);
        let handle = build(Arc::clone(&label));
        use_handle(&handle);
        if keep {
            map.insert(label, handle);
        }
    }

    /// The inner map under `key`, created on first sight; the key is only
    /// allocated for a new entry.
    #[expect(
        clippy::expect_used,
        reason = "the key is inserted just above when missing"
    )]
    pub fn nested<'m, V: Default>(map: &'m mut HashMap<String, V>, key: &str) -> &'m mut V {
        if !map.contains_key(key) {
            map.insert(key.to_owned(), V::default());
        }
        map.get_mut(key).expect("inserted above")
    }

    #[cfg(test)]
    mod tests {
        use std::cell::Cell;

        use super::*;

        #[test]
        fn a_handle_is_built_once_per_key_and_used_every_time() {
            let mut map: HashMap<String, usize> = HashMap::new();
            let builds = Cell::new(0);
            let mut used = Vec::new();
            for key in ["/a", "/a", "/b", "/a"] {
                with_handle(
                    &mut map,
                    key,
                    || {
                        builds.set(builds.get() + 1);
                        key.len() * 10
                    },
                    |handle| used.push(*handle),
                );
            }
            assert_eq!(builds.get(), 2);
            assert_eq!(map.len(), 2);
            assert_eq!(used, [20, 20, 20, 20]);
        }

        #[test]
        fn a_full_map_uses_a_new_handle_without_keeping_it() {
            let mut map: HashMap<String, usize> = HashMap::new();
            for i in 0..MAX_KEPT_PER_MAP {
                with_handle(&mut map, &format!("m{i}"), || i, |_| {});
            }
            assert_eq!(map.len(), MAX_KEPT_PER_MAP);
            let mut used = None;
            with_handle(
                &mut map,
                "one-too-many",
                || 7,
                |handle| used = Some(*handle),
            );
            assert_eq!(used, Some(7));
            assert_eq!(map.len(), MAX_KEPT_PER_MAP);
            assert!(!map.contains_key("one-too-many"));
            // A kept key is still found when the map is full.
            with_handle(&mut map, "m3", || 99, |handle| used = Some(*handle));
            assert_eq!(used, Some(3));
        }

        #[test]
        fn a_nested_map_is_created_once() {
            let mut map: HashMap<String, Vec<u8>> = HashMap::new();
            nested(&mut map, "/a").push(1);
            nested(&mut map, "/a").push(2);
            assert_eq!(map.len(), 1);
            assert_eq!(map["/a"], [1, 2]);
        }

        #[test]
        fn a_kept_model_handle_is_keyed_by_the_interners_label() {
            let mut map: HashMap<Arc<str>, usize> = HashMap::new();
            let mut built = None;
            with_model_handle(
                &mut map,
                "kept-model",
                |label| {
                    built = Some(Arc::clone(&label));
                    label.len()
                },
                |_| {},
            );
            let key = map.keys().next().expect("kept");
            assert!(Arc::ptr_eq(key, &intern_model_label("kept-model")));
            assert!(Arc::ptr_eq(key, built.as_ref().expect("built once")));
            let mut used = None;
            with_model_handle(&mut map, "kept-model", |_| 0, |handle| used = Some(*handle));
            assert_eq!(used, Some("kept-model".len()), "found by its text");
            assert_eq!(map.len(), 1);
        }

        #[test]
        fn the_sentinel_is_used_but_never_kept() {
            let sentinel = Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
            assert!(is_sentinel(&sentinel));
            assert!(!kept(0, &sentinel));
            let admitted = intern_model_label("an-admitted-model");
            assert!(!is_sentinel(&admitted));
            assert!(kept(0, &admitted));
            assert!(!kept(MAX_KEPT_PER_MAP, &admitted));
        }

        #[test]
        fn the_forced_say_of_a_thread_outranks_the_installed_flag() {
            assert!(!keep_handles());
            force_on_this_thread(Some(true));
            assert!(keep_handles());
            force_on_this_thread(Some(false));
            assert!(!keep_handles());
            force_on_this_thread(None);
            assert!(!keep_handles());
        }
    }
}

/// Recorders for the metrics a router emits.
pub struct Metrics;

impl Metrics {
    /// Record a routed request.
    ///
    /// Uses string interning for model_id to avoid repeated allocations.
    ///
    /// # Arguments
    /// * `streaming` - Use `bool_to_static_str(request.stream)` or the constants
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: &'static str,
    ) {
        let build = |model: Arc<str>| {
            counter!(
                "smg_router_requests_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "connection_mode" => connection_mode,
                "model" => model,
                "endpoint" => endpoint,
                "streaming" => streaming
            )
        };
        if !handles::keep_handles() {
            build(intern_model_label(model_id)).increment(1);
            return;
        }
        let labels = handles::RequestLabels {
            router_type,
            backend_type,
            connection_mode,
            endpoint,
            streaming,
        };
        handles::ROUTER_REQUESTS.with_borrow_mut(|by_labels| {
            handles::with_model_handle(
                by_labels.entry(labels).or_default(),
                model_id,
                build,
                |counter| counter.increment(1),
            );
        });
    }
    /// Record router request duration.
    /// Uses string interning for model_id.
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let build = |model: Arc<str>| {
            histogram!(
                "smg_router_request_duration_seconds",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "connection_mode" => connection_mode,
                "model" => model,
                "endpoint" => endpoint
            )
        };
        let seconds = duration.as_secs_f64();
        if !handles::keep_handles() {
            build(intern_model_label(model_id)).record(seconds);
            return;
        }
        let labels = handles::DurationLabels {
            router_type,
            backend_type,
            connection_mode,
            endpoint,
        };
        handles::ROUTER_DURATIONS.with_borrow_mut(|by_labels| {
            handles::with_model_handle(
                by_labels.entry(labels).or_default(),
                model_id,
                build,
                |histogram| histogram.record(seconds),
            );
        });
    }
    /// Record a router error: its class (`error_type`) and the cause behind
    /// it (`reason`, one of the `metrics_labels::REASON_*` values).
    /// Uses string interning for model_id.
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
        reason: &'static str,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_request_errors_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "error_type" => error_type,
            "reason" => reason
        )
        .increment(1);
    }
    /// Record tokens processed
    pub fn record_router_tokens(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        token_type: &'static str,
        count: u64,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => token_type
        )
        .increment(count);
    }
    /// Record retry attempt
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }
    /// Record retries exhausted
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_exhausted_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }
    /// Record retry backoff duration.
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        let attempt_str: Cow<'static, str> = match attempt {
            1 => Cow::Borrowed("1"),
            2 => Cow::Borrowed("2"),
            3 => Cow::Borrowed("3"),
            4 => Cow::Borrowed("4"),
            5 => Cow::Borrowed("5"),
            _ => Cow::Owned(attempt.to_string()),
        };
        histogram!(
            "smg_worker_retry_backoff_seconds",
            "attempt" => attempt_str
        )
        .record(duration.as_secs_f64());
    }
    /// Record MCP tool call
    pub fn record_mcp_tool_call(model_id: &str, tool_name: &str, result: &'static str) {
        let model = intern_model_label(model_id);
        let tool = intern_tool_label(tool_name);
        counter!(
            "smg_mcp_tool_calls_total",
            "model" => model,
            "tool_name" => tool,
            "result" => result
        )
        .increment(1);
    }
    /// Record MCP tool execution duration
    pub fn record_mcp_tool_duration(model_id: &str, tool_name: &str, duration: Duration) {
        let model = intern_model_label(model_id);
        let tool = intern_tool_label(tool_name);
        histogram!(
            "smg_mcp_tool_duration_seconds",
            "model" => model,
            "tool_name" => tool
        )
        .record(duration.as_secs_f64());
    }
    /// Record MCP tool loop iteration
    pub fn record_mcp_tool_iteration(model_id: &str) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_mcp_tool_iterations_total",
            "model" => model
        )
        .increment(1);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::*;

    /// The scrape of a recorder of its own after `records`, as series lines
    /// (no `# HELP` or `# TYPE`), sorted.
    fn series_after(records: &dyn Fn()) -> Vec<String> {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, records);
        let mut series: Vec<String> = handle
            .render()
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(str::to_owned)
            .collect();
        series.sort_unstable();
        series
    }

    /// The request counter and the duration histogram recorded through
    /// kept handles render exactly as with a key built per record, and one
    /// handle is kept per label set and model.
    #[test]
    fn router_records_render_the_same_through_kept_handles() {
        let records = || {
            for streaming in [STREAMING_TRUE, STREAMING_FALSE, STREAMING_TRUE] {
                Metrics::record_router_request(
                    metrics_labels::ROUTER_GRPC,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_GRPC,
                    "m1",
                    metrics_labels::ENDPOINT_CHAT,
                    streaming,
                );
            }
            Metrics::record_router_request(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                "m2",
                metrics_labels::ENDPOINT_GENERATE,
                STREAMING_FALSE,
            );
            for millis in [5, 7] {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_GRPC,
                    "m1",
                    metrics_labels::ENDPOINT_CHAT,
                    Duration::from_millis(millis),
                );
            }
        };
        let built = series_after(&records);

        handles::force_on_this_thread(Some(true));
        let kept = series_after(&records);
        let kept_counters = handles::ROUTER_REQUESTS
            .with_borrow(|by_labels| by_labels.values().map(HashMap::len).sum::<usize>());
        let kept_histograms = handles::ROUTER_DURATIONS
            .with_borrow(|by_labels| by_labels.values().map(HashMap::len).sum::<usize>());
        handles::force_on_this_thread(None);

        assert_eq!(kept, built);
        assert_eq!(kept_counters, 3, "a counter per label set and model");
        assert_eq!(kept_histograms, 1, "a histogram per label set and model");
        for expected in [
            "smg_router_requests_total{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\",streaming=\"true\"} 2",
            "smg_router_requests_total{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\",streaming=\"false\"} 1",
            "smg_router_requests_total{router_type=\"http\",backend_type=\"regular\",connection_mode=\"http\",model=\"m2\",endpoint=\"generate\",streaming=\"false\"} 1",
            "smg_router_request_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",endpoint=\"chat\"} 2",
        ] {
            assert!(
                kept.iter().any(|line| line == expected),
                "missing {expected} in:\n{}",
                kept.join("\n")
            );
        }
    }

    /// A router error known by its status alone reads as the worker's: the
    /// status classes an operator asks about first (a deadline, a dropped
    /// connection, capacity pushback, the worker's own failure) each get
    /// their reason, and a status outside them is `other`.
    #[test]
    fn a_status_alone_names_the_reason_of_the_workers_answer() {
        use metrics_labels::*;
        for (status, reason) in [
            (StatusCode::BAD_REQUEST, REASON_INVALID_REQUEST),
            (StatusCode::NOT_FOUND, REASON_MODEL_NOT_FOUND),
            (StatusCode::REQUEST_TIMEOUT, REASON_DEADLINE),
            (StatusCode::GATEWAY_TIMEOUT, REASON_DEADLINE),
            (StatusCode::TOO_MANY_REQUESTS, REASON_OVERLOADED),
            (StatusCode::INTERNAL_SERVER_ERROR, REASON_ENGINE_INTERNAL),
            (StatusCode::BAD_GATEWAY, REASON_TRANSPORT_DISCONNECT),
            (StatusCode::SERVICE_UNAVAILABLE, REASON_ENGINE_UNAVAILABLE),
            (StatusCode::UNAUTHORIZED, REASON_OTHER),
            (StatusCode::CONFLICT, REASON_OTHER),
            (StatusCode::NOT_IMPLEMENTED, REASON_OTHER),
        ] {
            assert_eq!(error_reason_from_status(status), reason, "{status}");
        }
        // The pool's two refusals of a request it could not place.
        assert_eq!(
            error_reason_for_refused_selection(StatusCode::NOT_FOUND),
            REASON_MODEL_NOT_FOUND
        );
        assert_eq!(
            error_reason_for_refused_selection(StatusCode::SERVICE_UNAVAILABLE),
            REASON_NO_AVAILABLE_WORKERS
        );
    }

    /// Every sample of the router error family carries its `reason` next to
    /// its `error_type`, so a sum over `reason` is the count the family had
    /// before the label existed, and two causes under one class are two
    /// series.
    #[test]
    fn a_router_error_is_counted_under_its_class_and_its_reason() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            for reason in [
                metrics_labels::REASON_TRANSPORT_DISCONNECT,
                metrics_labels::REASON_ENGINE_INTERNAL,
                metrics_labels::REASON_ENGINE_INTERNAL,
            ] {
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    metrics_labels::BACKEND_REGULAR,
                    metrics_labels::CONNECTION_GRPC,
                    "reason-test",
                    metrics_labels::ENDPOINT_CHAT,
                    metrics_labels::ERROR_BACKEND,
                    reason,
                );
            }
        });
        let rendered = handle.render();
        let samples: Vec<&str> = rendered
            .lines()
            .filter(|line| {
                line.starts_with("smg_router_request_errors_total{")
                    && line.contains(r#"model="reason-test""#)
            })
            .collect();
        assert_eq!(samples.len(), 2, "{rendered}");
        let value = |reason: &str| -> u64 {
            let line = samples
                .iter()
                .find(|line| line.contains(&format!(r#"reason="{reason}""#)))
                .unwrap_or_else(|| panic!("a {reason} series:\n{rendered}"));
            assert!(line.contains(r#"error_type="backend_error""#), "{line}");
            line.rsplit(' ').next().unwrap().parse().unwrap()
        };
        assert_eq!(value("transport_disconnect"), 1);
        assert_eq!(value("engine_internal"), 2);
    }

    #[test]
    fn bounded_labels_cap_cardinality_with_a_sentinel() {
        let labels = BoundedLabels::new(2);

        let a = labels.intern("m1");
        let b = labels.intern("m2");
        assert_eq!(labels.map.len(), 2);

        // Repeats return the same interned Arc without growing the map.
        let a2 = labels.intern("m1");
        assert!(Arc::ptr_eq(&a, &a2));
        assert_eq!(labels.map.len(), 2);

        // A distinct value past the cap collapses to the sentinel and does not
        // grow the map, so no new Prometheus series is minted for it.
        let c = labels.intern("m3");
        assert_eq!(&*c, BOUNDED_LABEL_SENTINEL);
        assert_eq!(labels.map.len(), 2);

        // Already-admitted values still resolve normally after the cap is hit.
        let b2 = labels.intern("m2");
        assert!(Arc::ptr_eq(&b, &b2));
        assert_ne!(&*a, BOUNDED_LABEL_SENTINEL);
    }
}
