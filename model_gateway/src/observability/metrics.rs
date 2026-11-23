#[cfg(test)]
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::{borrow::Cow, collections::HashMap, sync::Arc, time::Duration};

use llm_tokenizer::{
    cache::{cache_activity_stats, CacheActivityStats},
    decode_activity_stats, encode_fallbacks, encode_path_loads,
};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{
    DistributionBuilder, Matcher, PrometheusBuilder, PrometheusHandle,
};
use metrics_util::parse_quantiles;
use smg_external_router::metrics::Metrics as RouterMetrics;
pub use smg_external_router::metrics::{
    bool_to_static_str, error_reason_from_status, intern_model_label, intern_tool_label,
    metrics_labels, STREAMING_FALSE, STREAMING_TRUE,
};
use smg_worker::metrics::{
    series as worker_series,
    series::{WorkerSeries, WorkerSeriesRecorder},
    writers::{
        describe_series as describe_worker_series, intern_string, KV_EVENT_APPLY_SECONDS,
        KV_EVENT_LAG_SECONDS,
    },
};

use crate::{
    config::{ConfigError, ConfigResult},
    routers::grpc::regular::symphony::telemetry as symphony_telemetry,
};

/// Per-thread caches of the handles the gateway's per-request records go
/// through: the HTTP layer's, the stream-timing set's, the selection's, the
/// cache-aware decision's, the index lookup's, the admission gauges' and the
/// body path's. The worker's own caches are the worker module's
/// (`smg_worker::metrics::writers`); the helpers and the installed flag are the
/// worker crate's, so every writer records the same way.
mod handles {
    use std::{
        cell::{OnceCell, RefCell},
        collections::HashMap,
        hash::Hash,
        thread::LocalKey,
    };

    use metrics::{Counter, Gauge, Histogram};
    use rustc_hash::FxHashMap;
    #[cfg(test)]
    pub(super) use smg_external_router::metrics::handles::force_on_this_thread;
    pub(super) use smg_external_router::metrics::handles::{
        keep_handles, nested, recorder_installed, with_handle, with_model_handle, ByLabels,
    };

    /// Handles by the matched path (the only label a request computes: the
    /// route template, the gateway's own, so these maps hash with FxHash).
    pub(super) type ByPath<H> = FxHashMap<String, H>;
    /// Handles by method, then matched path.
    pub(super) type ByMethodAndPath<H> = FxHashMap<&'static str, ByPath<H>>;
    /// Response counters by matched path, then status code, then error code.
    /// The error-code level keeps the standard hasher: a worker's own error
    /// code can reach it through the response middleware, so that key is not
    /// one the gateway controls.
    pub(super) type ByPathStatusAndError =
        FxHashMap<String, FxHashMap<u16, HashMap<String, Counter>>>;

    /// The static labels of the stream-timing set.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub(super) struct StreamLabels {
        pub(super) router_type: &'static str,
        pub(super) backend_type: &'static str,
        pub(super) endpoint: &'static str,
    }

    /// The static labels of `smg_worker_selection_total`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub(super) struct SelectionLabels {
        pub(super) worker_type: &'static str,
        pub(super) connection_mode: &'static str,
        pub(super) policy: &'static str,
    }

    /// The handles of one label set of the stream-timing set, each built
    /// the first time a stream has something for it, so the series that
    /// exist are exactly the ones recorded to.
    #[derive(Default)]
    pub(super) struct StreamHandles {
        /// `smg_router_ttft_seconds`.
        pub(super) ttft: OnceCell<Histogram>,
        /// `smg_router_dispatch_wait_seconds`.
        pub(super) dispatch_wait: OnceCell<Histogram>,
        /// `smg_router_tpot_seconds`.
        pub(super) tpot: OnceCell<Histogram>,
        /// `smg_router_time_to_first_token_seconds`.
        pub(super) first_token: OnceCell<Histogram>,
        /// `smg_router_request_time_per_output_token_seconds`.
        pub(super) per_output_token: OnceCell<Histogram>,
        /// `smg_router_generation_duration_seconds`.
        pub(super) generation: OnceCell<Histogram>,
        /// `smg_router_tokens_total{token_type="input"}`.
        pub(super) input_tokens: OnceCell<Counter>,
        /// `smg_router_tokens_total{token_type="output"}`.
        pub(super) output_tokens: OnceCell<Counter>,
        /// `smg_router_tokens_total{token_type="cached"}`.
        pub(super) cached_tokens: OnceCell<Counter>,
    }

    /// Uses the one handle of a series with no dynamic label: this thread's
    /// kept handle, built on first use, or a key built per call when handles
    /// are not kept.
    pub(super) fn with_single<H>(
        kept: &'static LocalKey<RefCell<Option<H>>>,
        build: impl FnOnce() -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if !keep_handles() {
            use_handle(&build());
            return;
        }
        kept.with_borrow_mut(|slot| use_handle(slot.get_or_insert_with(build)));
    }

    /// Uses the handle under a static label (or a tuple of them): this
    /// thread's kept handle, built on the label's first sight, or a key
    /// built per call when handles are not kept. A static label set is
    /// finite, so the map needs no cap; its labels are the gateway's own
    /// constants, so the map hashes with FxHash.
    pub(super) fn with_static<K: Copy + Eq + Hash, H>(
        kept: &'static LocalKey<RefCell<FxHashMap<K, H>>>,
        label: K,
        build: impl FnOnce() -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if !keep_handles() {
            use_handle(&build());
            return;
        }
        kept.with_borrow_mut(|map| use_handle(map.entry(label).or_insert_with(build)));
    }

    thread_local! {
        /// `smg_http_requests_total`.
        pub(super) static HTTP_REQUESTS: RefCell<ByMethodAndPath<Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_http_request_duration_seconds`.
        pub(super) static HTTP_DURATIONS: RefCell<ByMethodAndPath<Histogram>> = RefCell::new(FxHashMap::default());
        /// `smg_http_responses_total`.
        pub(super) static HTTP_RESPONSES: RefCell<ByPathStatusAndError> = RefCell::new(FxHashMap::default());
        /// `smg_http_rate_limit_total` by result.
        pub(super) static HTTP_RATE_LIMITS: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_http_connections_active`: one series, a handle to it per worker.
        pub(super) static HTTP_CONNECTIONS_ACTIVE: RefCell<Option<Gauge>> = const { RefCell::new(None) };
        /// The stream-timing set by its static labels, then the model.
        pub(super) static STREAMS: RefCell<ByLabels<StreamLabels, StreamHandles>> = RefCell::new(FxHashMap::default());
        /// `smg_worker_selection_total` by its static labels, then the model.
        pub(super) static WORKER_SELECTIONS: RefCell<ByLabels<SelectionLabels, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_cache_aware_policy_branch_total` by branch.
        pub(super) static CACHE_AWARE_BRANCHES: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_manual_policy_branch_total` by branch.
        pub(super) static MANUAL_BRANCHES: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_consistent_hashing_policy_branch_total` by branch.
        pub(super) static CONSISTENT_HASHING_BRANCHES: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_prefix_hash_policy_branch_total` by branch.
        pub(super) static PREFIX_HASH_BRANCHES: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_cache_aware_match_ratio`: one series.
        pub(super) static CACHE_AWARE_MATCH_RATIO: RefCell<Option<Histogram>> = const { RefCell::new(None) };
        /// `smg_kv_index_lookup_seconds` by index.
        pub(super) static KV_INDEX_LOOKUPS: RefCell<FxHashMap<&'static str, Histogram>> = RefCell::new(FxHashMap::default());
        /// `smg_routing_key_source_total` by source.
        pub(super) static ROUTING_KEY_SOURCES: RefCell<FxHashMap<&'static str, Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_admission_queue_depth`: one series.
        pub(super) static ADMISSION_QUEUE_DEPTH: RefCell<Option<Gauge>> = const { RefCell::new(None) };
        /// `smg_admission_inflight`: one series.
        pub(super) static ADMISSION_INFLIGHT: RefCell<Option<Gauge>> = const { RefCell::new(None) };
        /// `smg_router_request_body_path_total` by path, then reason.
        pub(super) static REQUEST_BODY_PATHS: RefCell<FxHashMap<(&'static str, &'static str), Counter>> = RefCell::new(FxHashMap::default());
        /// `smg_router_request_buffers_released_early_bytes_total`: one series.
        pub(super) static REQUEST_BUFFERS_RELEASED_EARLY: RefCell<Option<Counter>> = const { RefCell::new(None) };
    }

    /// For tests: this thread's kept handles of the gateway's families,
    /// dropped, and the installed flag given its say back. The worker's
    /// module keeps its own handles and has its own reset.
    #[cfg(test)]
    pub(super) fn reset() {
        force_on_this_thread(None);
        HTTP_REQUESTS.with_borrow_mut(HashMap::clear);
        HTTP_DURATIONS.with_borrow_mut(HashMap::clear);
        HTTP_RESPONSES.with_borrow_mut(HashMap::clear);
        HTTP_RATE_LIMITS.with_borrow_mut(HashMap::clear);
        HTTP_CONNECTIONS_ACTIVE.set(None);
        STREAMS.with_borrow_mut(HashMap::clear);
        WORKER_SELECTIONS.with_borrow_mut(HashMap::clear);
        CACHE_AWARE_BRANCHES.with_borrow_mut(HashMap::clear);
        MANUAL_BRANCHES.with_borrow_mut(HashMap::clear);
        CONSISTENT_HASHING_BRANCHES.with_borrow_mut(HashMap::clear);
        PREFIX_HASH_BRANCHES.with_borrow_mut(HashMap::clear);
        CACHE_AWARE_MATCH_RATIO.set(None);
        KV_INDEX_LOOKUPS.with_borrow_mut(HashMap::clear);
        ROUTING_KEY_SOURCES.with_borrow_mut(HashMap::clear);
        ADMISSION_QUEUE_DEPTH.set(None);
        ADMISSION_INFLIGHT.set(None);
        REQUEST_BODY_PATHS.with_borrow_mut(HashMap::clear);
        REQUEST_BUFFERS_RELEASED_EARLY.set(None);
    }
}

// =============================================================================
// STATIC STRING CONSTANTS
// =============================================================================

/// Static lookup table for common HTTP status codes to avoid allocations.
/// Returns a static string for known codes, or None for unknown codes.
#[inline]
pub fn status_code_to_static_str(code: u16) -> Option<&'static str> {
    // Using a match with explicit arms is faster than a lookup table for this size
    match code {
        200 => Some("200"),
        201 => Some("201"),
        204 => Some("204"),
        400 => Some("400"),
        401 => Some("401"),
        403 => Some("403"),
        404 => Some("404"),
        408 => Some("408"),
        422 => Some("422"),
        429 => Some("429"),
        500 => Some("500"),
        502 => Some("502"),
        503 => Some("503"),
        504 => Some("504"),
        _ => None,
    }
}

/// Static HTTP method strings to avoid allocations on every request.
pub(crate) mod http_methods {
    pub const GET: &str = "GET";
    pub const POST: &str = "POST";
    pub const PUT: &str = "PUT";
    pub const DELETE: &str = "DELETE";
    pub const PATCH: &str = "PATCH";
    pub const HEAD: &str = "HEAD";
    pub const OPTIONS: &str = "OPTIONS";
}

/// Convert HTTP method to static string. Returns the method as-is for unknown methods.
#[inline]
pub fn method_to_static_str(method: &str) -> &'static str {
    match method {
        "GET" => http_methods::GET,
        "POST" => http_methods::POST,
        "PUT" => http_methods::PUT,
        "DELETE" => http_methods::DELETE,
        "PATCH" => http_methods::PATCH,
        "HEAD" => http_methods::HEAD,
        "OPTIONS" => http_methods::OPTIONS,
        _ => "OTHER",
    }
}

/// Get status code as Cow - static for common codes, allocated for rare ones.
#[inline]
pub fn status_code_to_cow(code: u16) -> Cow<'static, str> {
    match status_code_to_static_str(code) {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(code.to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
    pub duration_buckets: Option<Vec<f64>>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
            duration_buckets: None,
        }
    }
}

impl PrometheusConfig {
    /// Refuses a duration bucket list the exporter would render wrongly: a
    /// bound that is not a finite number above 0, or bounds that do not
    /// ascend strictly (the `le` labels of every duration histogram would
    /// come out of order). The one check for every construction path, so
    /// the CLI and the Python binding refuse the same list with the same
    /// words.
    pub fn validate_duration_buckets(buckets: &[f64]) -> ConfigResult<()> {
        if let Some(bad) = buckets.iter().find(|b| !b.is_finite() || **b <= 0.0) {
            return Err(ConfigError::invalid_value(
                "prometheus_duration_buckets",
                bad,
                "every bucket bound must be a finite number above 0",
            ));
        }
        if let Some(pair) = buckets.windows(2).find(|w| w[1] <= w[0]) {
            return Err(ConfigError::invalid_value(
                "prometheus_duration_buckets",
                format!("{} {}", pair[0], pair[1]),
                "bucket bounds must ascend strictly",
            ));
        }
        Ok(())
    }
}

/// Upkeep interval for histogram maintenance. Must match the value passed to
/// `PrometheusBuilder::upkeep_timeout()` in `start_prometheus`.
pub(crate) const UPKEEP_INTERVAL_SECS: u64 = 5 * 60;

/// Histogram buckets for `smg_cache_aware_match_ratio`. The ratio is
/// dimensionless (matched/input, 0..1), so it takes deciles instead of the
/// duration buckets; `le="0"` isolates requests with no cached prefix at all.
pub(crate) const CACHE_AWARE_MATCH_RATIO_BUCKETS: &[f64] =
    &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Time to first token on edges that resolve it. The same observation as
/// `smg_router_ttft_seconds`, which keeps the request-duration buckets (and
/// so the `le` values recording rules were written against).
pub(crate) const TIME_TO_FIRST_TOKEN_SECONDS: &str = "smg_router_time_to_first_token_seconds";
/// Per-request mean time per output token on edges that resolve it. The
/// same observation as `smg_router_tpot_seconds`, which keeps the
/// request-duration buckets; named like the servicer's per-request mean
/// (`request_time_per_output_token`), as the per-interval gap is another
/// family's quantity.
pub(crate) const REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS: &str =
    "smg_router_request_time_per_output_token_seconds";

/// The wait ahead of the dispatch the first-token clock starts at: from the
/// request's entry at the edge to the answering attempt's RPC dispatch.
/// Observed with `smg_router_ttft_seconds`, for the same requests and under
/// the same labels, so the two add up to the time to first token from the
/// request's receipt. A streamed Responses tool loop observes one per turn,
/// a later turn's from that turn's start.
pub(crate) const DISPATCH_WAIT_SECONDS: &str = "smg_router_dispatch_wait_seconds";

/// Histogram buckets for `smg_router_time_to_first_token_seconds`: six per
/// decade (1, 1.5, 2, 3, 5, 7.5) from 10 ms to 10 s, where a first-token
/// objective lives, then coarser to two minutes; a first token later than that
/// is a timeout class and lands in `+Inf`. The request-duration list has four
/// edges between 50 ms and 500 ms; this one has seven.
pub(crate) const TIME_TO_FIRST_TOKEN_BUCKETS: &[f64] = &[
    0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
    7.5, 10.0, 20.0, 30.0, 60.0, 120.0,
];

/// Histogram buckets for `smg_router_request_time_per_output_token_seconds`: six per
/// decade (1, 1.5, 2, 3, 5, 7.5) from 1 ms to 1 s, where a per-token objective
/// lives, then two edges for stalled decodes. The request-duration list has
/// four edges between 5 ms and 50 ms; this one has seven.
pub(crate) const REQUEST_TIME_PER_OUTPUT_TOKEN_BUCKETS: &[f64] = &[
    0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,
    0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 10.0,
];

/// Histogram buckets for `smg_kv_index_lookup_seconds` and
/// `smg_kv_event_apply_seconds`: a lookup takes microseconds and a batch
/// apply tens of microseconds, below the request-latency buckets' first
/// edge, so these double from half a microsecond to a quarter of a second.
pub(crate) const KV_INDEX_MICRO_BUCKETS: &[f64] = &[
    0.000_000_5,
    0.000_001,
    0.000_002,
    0.000_004,
    0.000_008,
    0.000_016,
    0.000_032,
    0.000_064,
    0.000_128,
    0.000_256,
    0.000_512,
    0.001_024,
    0.002_048,
    0.004_096,
    0.008_192,
    0.016_384,
    0.065_536,
    0.262_144,
];

/// Histogram buckets for `smg_kv_event_lag_seconds`: the age of a KV event
/// batch when it is applied, publisher clock to router clock. A batch is
/// normally a few milliseconds old (publish interval plus transport), and
/// seconds or more when a relay replays history or the router falls behind,
/// so the edges run from a tenth of a millisecond to a minute. Without
/// explicit buckets the recorder renders the family as a summary, whose
/// per-worker quantiles cannot be aggregated across workers or windows.
pub(crate) const KV_EVENT_LAG_BUCKETS: &[f64] = &[
    0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0, 60.0,
];

/// Histogram buckets for `smg_worker_retry_backoff_seconds`: the delay slept
/// before a retried upstream call. The schedule is exponential with jitter,
/// from the configured initial backoff (50 ms by default) to the configured
/// maximum (30 s by default), so the edges run from ten milliseconds to a
/// minute. Without explicit buckets the recorder renders the family as a
/// summary, whose per-attempt quantiles cannot be aggregated across gateways
/// or time windows.
pub(crate) const WORKER_RETRY_BACKOFF_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Buckets of the media pipeline's times (seconds): a processing stage or
/// one fetch runs from well under a millisecond (a token expansion, inline
/// bytes) to seconds (a video decode, a slow host).
pub(crate) const MM_SECONDS_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Buckets of the media items a request carries, per modality.
pub(crate) const MM_MEDIA_PER_REQUEST_BUCKETS: &[f64] =
    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0];

/// Buckets of the images a request carries: dense where most requests land,
/// and starting at 0, the bucket a request without media is counted in.
pub(crate) const MM_REQUEST_IMAGES_BUCKETS: &[f64] = &[
    0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 100.0, 200.0,
];

/// The `modality` label of a media series.
fn mm_modality_label(modality: llm_multimodal::Modality) -> &'static str {
    match modality {
        llm_multimodal::Modality::Image => "image",
        llm_multimodal::Modality::ImageEmbeds => "image_embeds",
        llm_multimodal::Modality::Audio => "audio",
        llm_multimodal::Modality::Video => "video",
    }
}

/// Jemalloc's run-time options for a long-running gateway. Binaries that use
/// [`crate::stock_allocator!`] and the Python and Go bindings export them as
/// `_rjem_malloc_conf` next to their `#[global_allocator]`. The
/// multimodal crate's baseline harness (`crates/multimodal/examples/mm_baseline.rs`)
/// sits below this crate and repeats the bytes by hand; the test
/// `the_multimodal_harness_repeats_the_malloc_conf` keeps that copy equal. With
/// the allocator's defaults, a thread's freed pages decay back only when it
/// allocates again, so after a traffic burst an idle gateway kept ~1.5 GB
/// resident over ~270 MB of live objects (soak s2, ten hours). A background
/// thread purges on schedule instead; dirty pages are returned after 10 s and
/// muzzy pages at once.
///
/// jemalloc runs 16 arenas. Its default is four per CPU of the affinity mask
/// (one on a single CPU) with every thread on an arena of its own, and each
/// (arena, size class) pair keeps at least one partly used slab, a 64 KiB
/// page on the aarch64 build: with ~240 threads on a many-core host a gateway
/// holding 235 MB of live objects kept 1.6 GB of slabs (1.17 GB resident)
/// under a steady multi-turn mix, and every new blocking-pool thread added a
/// fresh set. With 16 arenas the same load and live heap took 0.4 GB of slabs
/// (0.46 GB resident) at the same CPU per request; 4 arenas saved little more
/// and cost 12 % CPU in arena contention. On a host with fewer than four CPUs
/// the fixed count is more arenas than the default, a few mostly empty slabs.
///
/// Allocations of `oversize_threshold` bytes or more bypass the arenas'
/// dirty-page cache: jemalloc serves them from a dedicated arena and returns
/// their pages to the OS the moment they are freed, so the next one is
/// faulted in page by page and zeroed by the kernel. Its default is 8 MiB.
/// An image request's pixel tensor rides in one gRPC message, which the
/// codec encodes into one contiguous buffer per call: 9 MiB for a 1024x768
/// image in bfloat16, 24 MiB for 1920x1080, twice that in float32. Under the
/// default every such request faulted the whole encode buffer in again
/// (~2,300 minor faults for 1024x768); with the threshold at 32 MiB those
/// buffers recycle through the dirty cache like everything else. On its own
/// the setting takes a 1024x768 request from ~5,200 minor faults to ~3,000
/// and 15 % less CPU: the request's zero-filled decoded frame and wire
/// buffers still fault, because jemalloc purges a recycled dirty extent
/// before it hands it out zeroed, so only a buffer that is not allocated per
/// request stops faulting. With those buffers pooled across requests the
/// same request takes 77 faults, a further 25 % less CPU than the pool alone
/// (40 % less than with neither), for ~30 MB more peak resident memory at
/// 8 threads. Messages of 32 MiB and more (float32 tensors of 1080p-class
/// grids and larger) keep the default behaviour.
///
/// The threshold keeps a tokenizer's load-time transients in the dirty cache
/// as well, and the vocabulary's hash tables land inside them: for a
/// vocabulary of 115k-229k tokens each table is 8.25 MiB in the 10 MiB class,
/// its last 1.75 MiB a tail nothing writes (a 256k vocabulary's 16.5 MiB
/// table sits in the 20 MiB class with a 3.5 MiB tail), and a tail that lands
/// on a recycled extent an earlier transient wrote end to end stays resident
/// for the life of the process. That is about 5 MB per loaded tokenizer on
/// every text path, present before the first request and invisible in
/// jemalloc's own resident count, which counts a live table's whole class
/// whether its tail was written or not (RSS at idle after 10,000 chat
/// requests on one binary: 129 MB under the default threshold, 134 MB at
/// 32 MiB). No entry of this string separates the tables from the image
/// buffers: a 1024x768 encode buffer is the same 10 MiB class in the same
/// kind of arena. A shorter `dirty_decay_ms` leaves a live table's pages
/// alone and purges an image buffer between requests; a threshold between
/// the two values gives up the classes above it (a 1920x1080 image in
/// bfloat16 is a 28 MiB-class message); an exact-class fit of dirty extents
/// clears the tails and hands the image path fresh pages again (jemalloc
/// merges freed large extents eagerly, and the merged run no longer fits its
/// class). A deployment that serves text alone takes the default back with
/// `_RJEM_MALLOC_CONF=oversize_threshold:8388608` and loses nothing.
///
/// This is jemalloc's application-provided `malloc_conf` string under the
/// vendored build's `_rjem_` prefix; the `_RJEM_MALLOC_CONF` environment
/// variable is read after it and overrides it entry by entry.
pub const SERVER_MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

/// The bytes of the one allocation that tells which allocator serves the
/// process.
#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
const ALLOCATOR_PROBE_BYTES: usize = 4096;

/// Whether this process's Rust allocations go through the library's
/// jemalloc.
///
/// The gateway finds out and is not told: it makes one allocation and reads
/// the allocator's own count of this thread's bytes before and after. A
/// count that did not move means another allocator serves the process, and
/// the statistics of the linked jemalloc are then not published. The answer
/// is found once and kept.
#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
pub(crate) fn jemalloc_is_global() -> bool {
    static FOUND: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let Ok(allocated) = tikv_jemalloc_ctl::thread::allocatedp::read() else {
            return false;
        };
        let before = allocated.get();
        let probe = std::hint::black_box(Vec::<u8>::with_capacity(ALLOCATOR_PROBE_BYTES));
        let after = allocated.get();
        drop(probe);
        after > before
    })
}

pub(crate) fn init_metrics() {
    // What the Symphony tables read that was not content, reasoning or a call
    // (`routers::grpc::regular::symphony::telemetry`).
    describe_counter!(
        symphony_telemetry::OUTPUTS,
        "Outputs a Symphony table read to their end, by table and outcome: clean, malformed (a call region the table gave up on, or a marker of the table where no row leads, kept as content; not such a marker in an output the request ran past the model's own end with ignore_eos, which counts as nothing), failed (the table refused the output and the request failed), empty (no text, reasoning or call: whitespace and framing alone, or nothing), unknown_tool (read whole, with a call to a function the request did not declare, delivered as the model named it), panicked (the gateway's reading of the output panicked, caught at the request's boundary; the request failed under reader_panicked)"
    );
    describe_counter!(
        symphony_telemetry::MALFORMED,
        "Malformed runs in the outputs a Symphony table read, by table and reason: the crate's own name of the reason (unterminated_region, invalid_arguments, unknown_tool, tag_out_of_place, text_where_the_name_should_be, key_without_a_value, ..., marker_out_of_place, open_at_the_stop, open_at_the_marker), other for a reason this build does not name"
    );
    describe_histogram!(
        symphony_telemetry::MALFORMED_BYTES,
        "Bytes of the malformed runs of one output a Symphony table read, by table; observed for every output counted malformed"
    );
    describe_counter!(
        symphony_telemetry::CALLS_SUPPRESSED,
        "Tool calls after the first that a Symphony table read and the gateway dropped under parallel_tool_calls: false, by table"
    );
    describe_counter!(
        symphony_telemetry::CALLS_FORBIDDEN,
        "Tool calls a Symphony table read and the gateway dropped under a tool_choice that forbids them (every call under none; a call outside the callable tools under a named function, an allowed list or required), by table"
    );
    describe_counter!(
        symphony_telemetry::FALLBACK,
        "Requests Symphony was configured for (the --parser flag or the request's X-SMG-Parser header) that the legacy parsers read instead of a table, by reason (no_table: the registry has no table for the model and nothing names one; reasoning_name_only: a reasoning parser name alone that no table answers to; legacy_tool_name: a legacy tool-call parser name no table answers to; passthrough), under the table none"
    );
    describe_counter!(
        symphony_telemetry::REFUSED,
        "Whole Messages responses a Symphony table read that the gateway refused (HTTP 500), by table and reason (tool_use_input: a tool_use block cannot carry the arguments the model wrote); the output stays counted under smg_symphony_outputs_total by its own outcome"
    );
    describe_counter!("smg_tokenizer_cache_lookups_total", "Tokenizer cache lookups by layer and result; L1 includes inputs without cacheable boundaries and excludes L0 hits");
    describe_counter!(
        "smg_tokenizer_cache_evictions_total",
        "Tokenizer cache entries removed for capacity; excludes clear, drop and replacement"
    );
    describe_counter!("smg_tokenizer_cache_reused_bytes_total", "UTF-8 input bytes served by tokenizer cache hits: whole inputs for L0, matched prefixes for L1; not memory usage");
    describe_counter!("smg_detokenize_prefix_mismatch_total", "Incremental decode steps and end-of-input finishes whose output no longer started with the text already decoded and went on with the remainder (several per stream are possible; unary responses count too), by kind: byte_run (an invalid byte collapsed a ByteFallback run whose characters were already decoded) or decoder (another decoder step rewrote the decoded prefix)");
    describe_counter!("smg_tokenizer_encode_path_loads_total", "Hugging Face tokenizers loaded, by the encode path chosen at load: v1 (the opt-in tokenizers v1 pipeline), native (one of the crate's direct encoders: byte-level BPE or WordPiece) or tokenizers (the reference implementation); a tokenizer loaded twice counts twice");
    describe_counter!("smg_tokenizer_encode_fallbacks_total", "Inputs the direct byte-level encoder gave up to the reference implementation after the load's choice, by reason: piece_declined (the model declined a piece)");

    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    allocator_stats::describe();

    // Layer 1: HTTP metrics
    describe_counter!(
        "smg_http_requests_total",
        "Total HTTP requests by method and path"
    );
    describe_histogram!(
        "smg_http_request_duration_seconds",
        "HTTP request duration by method and path"
    );
    describe_gauge!(
        "smg_http_inflight_request_age_count",
        "In-flight HTTP requests per age bucket (gt < age <= le, non-cumulative)"
    );
    describe_counter!(
        "smg_http_responses_total",
        "Total HTTP responses by path, status_code and error_code, counted once the \
         response has gone out in full (its last frame, or the length it declared); a \
         request whose client closed the connection first counts as status_code 499 \
         with error_code client_closed_request, as does the rare request the server \
         dropped unanswered (a handler panic, a forced shutdown); a response whose body \
         failed after its head (an upstream stream that broke mid-body, or a streamed \
         body its producer ended with an error event, whatever the failure's cause) \
         keeps its head's status_code with error_code stream_error, unless the head \
         carried a gateway error code"
    );
    describe_gauge!(
        "smg_http_connections_active",
        "HTTP requests in flight at the edge layer, not open connections: an idle keep-alive connection carries no request and one HTTP/2 connection carries many"
    );
    describe_counter!(
        "smg_http_connection_timeouts_total",
        "HTTP connections closed for carrying no request, by the timeout that closed them (header_read, idle)"
    );
    describe_counter!(
        "smg_http_connections_rejected_total",
        "HTTP connections closed at accept for arriving beyond a listener's connection limit (max_connections on the data listener, admin_max_connections on the admin listener): nothing was read from or written to them"
    );
    describe_counter!(
        "smg_http_rate_limit_total",
        "Rate limiting decisions by result (allowed/rejected)"
    );
    describe_gauge!(
        "smg_admission_queue_depth",
        "Requests currently parked in the admission queue"
    );
    describe_counter!(
        "smg_admission_queue_rejected_total",
        "Requests rejected at admission by reason (full/timeout), or refused ahead of worker selection after waiting longer than the queue timeout inside the gateway (selection_timeout)"
    );
    describe_gauge!(
        "smg_admission_inflight",
        "Requests currently holding an admission token"
    );

    // Layer 2: Router metrics
    describe_counter!(
        "smg_router_requests_total",
        "Total routed requests by router_type, backend_type, connection_mode, model, endpoint, streaming; model is the served model's id, unknown for a request that names no model, or other for a model the gateway does not serve"
    );
    describe_histogram!(
        "smg_router_request_duration_seconds",
        "Router request duration by router_type, backend_type, connection_mode, model, endpoint"
    );
    describe_counter!(
        "smg_router_request_errors_total",
        "Router errors by router_type, backend_type, connection_mode, model, endpoint, error_type, reason; model is the served model's id, unknown for a request that names no model, or other for a model the gateway does not serve; reason is the cause behind the error_type as far as the router could tell it (deadline, transport_disconnect, engine_error, engine_internal, engine_unavailable, overloaded, rate_limited, overload_shed, selection_queue_timeout, no_available_workers, no_capable_worker, model_not_found, invalid_request, media_fault, constraint_failed, bad_response, internal, other)"
    );
    describe_counter!(
        "smg_router_upstream_responses_total",
        "Upstream backend HTTP responses by router_type, status_code, error_code"
    );
    describe_counter!(
        "smg_router_upstream_send_retries_total",
        "Single-shot resends after a pre-response transport failure, by router_type"
    );
    describe_counter!(
        "smg_router_request_buffers_released_early_bytes_total",
        "Serialized size of request buffers freed at dispatch instead of response completion (retries disabled)"
    );
    describe_counter!(
        "smg_router_request_body_path_total",
        "Per-request body-path decisions by path (streamed/buffered) and dominant reason"
    );

    // Layer 2: Router inference metrics (gRPC only). The first-token and
    // per-token families hold streamed answers alone: a collected answer is
    // built from the drained stream and sees no first token, and a streamed
    // Responses answer on the Harmony pipeline records only the prefill-decode
    // first-token pair, none of these four. The dispatch wait is observed with
    // the first token: it ends where the first-token clock starts.
    describe_histogram!(
        "smg_router_ttft_seconds",
        "Time to first token of streamed answers by router_type, backend_type, model, endpoint (gRPC only; a collected answer records no first token, nor does a streamed Responses answer on the Harmony pipeline); the clock starts at the answering attempt's RPC dispatch (the prefill dispatch of a prefill-decode pair, after the encode call when encode pushes), so a refused attempt's wait and its backoff are in smg_router_request_duration_seconds, the backoff alone in smg_worker_retry_backoff_seconds, and the refused attempt counted by smg_worker_retries_total, not here"
    );
    describe_histogram!(
        DISPATCH_WAIT_SECONDS,
        "Dispatch wait of streamed answers by router_type, backend_type, model, endpoint (gRPC only; a collected answer records no first token, nor does a streamed Responses answer on the Harmony pipeline, so no wait): from the request's entry at the edge to the answering attempt's RPC dispatch (the prefill dispatch of a prefill-decode pair, after the encode call when encode pushes), where the clock of smg_router_ttft_seconds starts, so the body read, admission, tokenizing, templating, worker selection, a refused attempt's wait and its backoff are here; observed with smg_router_ttft_seconds, once per observation of it and under the same labels, so the two add up to the time to first token from the request's receipt; a streamed Responses tool loop observes one per turn, as it observes smg_router_ttft_seconds: the first turn's from the request's entry, a later turn's from that turn's start, after the generations and tool calls before it, so a later turn's two add up to the time to first token from the turn's start"
    );
    describe_histogram!(
        "smg_router_tpot_seconds",
        "Time per output token of streamed answers by router_type, backend_type, model, endpoint (gRPC only; a collected answer records no first token, nor does a streamed Responses answer on the Harmony pipeline, so no time per token)"
    );
    describe_histogram!(
        TIME_TO_FIRST_TOKEN_SECONDS,
        "Time to first token of streamed answers by router_type, backend_type, model, endpoint (gRPC only; a collected answer records no first token, nor does a streamed Responses answer on the Harmony pipeline): the observations of smg_router_ttft_seconds on buckets that resolve them (six per decade from 10 ms to 10 s; the request-duration bucket flag does not apply), on the same clock from the answering attempt's RPC dispatch"
    );
    describe_histogram!(
        REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
        "Per-request mean time per output token of streamed answers by router_type, backend_type, model, endpoint (gRPC only; a collected answer records no first token, nor does a streamed Responses answer on the Harmony pipeline, so no time per token): the observations of smg_router_tpot_seconds on buckets that resolve them (six per decade from 1 ms to 1 s; the request-duration bucket flag does not apply); a per-request mean, as the servicer's request_time_per_output_token family, not the per-interval gap"
    );
    describe_counter!(
        "smg_router_tokens_total",
        "Total tokens processed by router_type, backend_type, model, endpoint, token_type (gRPC only); token_type is input, output, or cached (the prompt tokens the engine reports it served from its cache, a share of input counted again, so cached / input is the cache hit share the router's overlap prediction is judged by)"
    );
    describe_histogram!(
        "smg_router_generation_duration_seconds",
        "Total generation time by router_type, backend_type, model, endpoint (gRPC only): from the answering attempt's RPC dispatch (the prefill dispatch of a prefill-decode pair, after the encode call when encode pushes) to the answer's end, so a refused attempt's wait and its backoff are in smg_router_request_duration_seconds, the backoff alone in smg_worker_retry_backoff_seconds, and the refused attempt counted by smg_worker_retries_total, not here"
    );
    describe_counter!(
        "smg_router_local_stop_aborts_total",
        "Streamed requests the gateway aborted at the engine because its own stop (a stop string or stop token id matched in the gateway, not by the engine) had ended every choice for the client while the engine was still generating, by router_type, backend_type, model, endpoint (gRPC only)"
    );
    describe_counter!(
        "smg_router_local_stop_tail_tokens_total",
        "Output tokens the engine streamed for a choice after the gateway's own stop had ended that choice, read before the request was aborted (the tail generated for nobody; what the abort cuts on the engine side is never read, so never counted), by router_type, backend_type, model, endpoint (gRPC only)"
    );

    // Layer 2: PD disaggregation metrics (signals only SMG can measure — it is the
    // only component that observes both the prefill and decode sides of a request).
    //
    // The PD time to first token is split into phases the gateway bounds itself.
    // Under sequential dispatch (the gRPC router relays the sides) the three
    // phase families add up to `smg_pd_ttft_seconds` exactly:
    //   prefill  smg_pd_prefill_duration_seconds    prefill dispatch -> prefill response drained
    //   relay    smg_pd_kv_transfer_duration_seconds prefill drained -> decode dispatch (gateway work only)
    //   decode   smg_pd_decode_ttft_seconds          decode dispatch -> first decode token
    // The engines' own KV transfer runs inside the decode phase. Parallel
    // dispatch sends both sides together: the decode phase is the whole TTFT and
    // there is no relay phase.
    describe_histogram!(
        "smg_pd_prefill_duration_seconds",
        "PD prefill phase: prefill dispatch to the end of the prefill response, the prefill engine's queueing and compute included, by backend_type, model, runtime"
    );
    describe_histogram!(
        "smg_pd_kv_transfer_duration_seconds",
        "PD relay phase (sequential dispatch): end of the prefill response to the decode dispatch, gateway work only; not the engines' KV transfer, which runs inside smg_pd_decode_ttft_seconds; by backend_type, model, runtime"
    );
    describe_histogram!(
        "smg_pd_decode_ttft_seconds",
        "PD decode phase: decode dispatch to the first decode token, the engines' KV transfer and the decode scheduling included (the whole TTFT under parallel dispatch), by backend_type, model, runtime"
    );
    describe_histogram!(
        "smg_pd_ttft_seconds",
        "PD time to first token: prefill dispatch to the first decode token (the decode response head on the HTTP router); under sequential dispatch the prefill, relay and decode phases add up to it; by backend_type, model, runtime"
    );
    describe_counter!(
        "smg_pd_kv_connector_mode_total",
        "KV connector mode decisions by mode (mooncake/nixl/moriio/store/passthrough)"
    );
    describe_counter!(
        "smg_pd_bootstrap_failures_total",
        "PD bootstrap injection failures"
    );
    describe_counter!(
        "smg_pd_kv_transfer_failures_total",
        "PD KV-transfer failures (missing connector params at decode handoff)"
    );
    describe_gauge!(
        "smg_pd_prefill_admission_inflight",
        "Prefill requests admitted by SMG per worker"
    );
    describe_gauge!(
        "smg_pd_prefill_admission_queued",
        "Requests waiting in the SMG Prefill admission queue"
    );
    describe_histogram!(
        "smg_pd_prefill_admission_wait_seconds",
        "Time spent waiting in the SMG Prefill admission queue"
    );
    describe_counter!(
        "smg_pd_prefill_admission_rejections_total",
        "Prefill admission rejections by reason"
    );
    describe_counter!(
        "smg_pd_admission_waits_total",
        "PD dispatches that waited for a decode admission slot"
    );
    describe_counter!(
        "smg_pd_admission_sheds_total",
        "PD dispatches shed because no decode admission slot freed in time"
    );
    describe_gauge!(
        "smg_pd_admission_window",
        "Running window the PD admission gate bounds a decode worker by; 0 when the engine reports none and admission is off for that worker"
    );

    // Layer 3: Worker metrics
    // The worker's own series, described by the worker's module.
    describe_worker_series();
    describe_counter!(
        "smg_worker_selection_total",
        "Worker selection events by worker_type, connection_mode, model, policy"
    );
    describe_counter!(
        "smg_worker_errors_total",
        "Worker-level errors by worker_type, connection_mode, error_type"
    );
    describe_histogram!(
        "smg_kv_index_lookup_seconds",
        "Time of one KV index lookup (overlap scoring of a request's block hashes) \
         in cache-aware routing, by index kind"
    );
    describe_gauge!(
        "smg_log_dropped_lines",
        "Log lines the lossy writer queue dropped since the process started, by sink \
         (stdout, file)"
    );
    describe_counter!(
        "smg_worker_overload_shed_total",
        "Requests shed because every worker for the model is overloaded, by stage \
         (selection, dispatch)"
    );
    describe_counter!(
        "smg_worker_overload_fallback_total",
        "Requests placed although every worker they could use is overloaded and shedding is \
         off, by stage: routed to the least-loaded worker, or, for a cache-aware prefill pool \
         of a prefill/decode pair or a regular request that consistent_hashing places by \
         its routing key, placed by that policy"
    );
    describe_counter!(
        "smg_worker_liveness_fallback_total",
        "Requests routed to the least-loaded ready worker because every worker for the model \
         is vetoed and the ready ones are vetoed by the liveness tracker, by stage"
    );
    describe_gauge!(
        "smg_manual_policy_cache_entries",
        "Number of routing entries in manual policy cache"
    );
    describe_counter!(
        "smg_manual_policy_branch_total",
        "Manual policy selection branch (no_healthy_workers, occupied_hit, occupied_miss, \
         vacant, no_routing_id, cap_respill)"
    );
    describe_counter!(
        "smg_consistent_hashing_policy_branch_total",
        "Consistent hashing policy selection branch (no_healthy_workers, target_worker_hit, \
         target_worker_miss, routing_key_hit, random_fallback)"
    );
    describe_counter!(
        "smg_prefix_hash_policy_branch_total",
        "Prefix hash policy selection branch (no_healthy_workers, no_routing_key, ring_hit, \
         load_balance_walk, fallback_least_load)"
    );
    describe_counter!(
        "smg_routing_key_source_total",
        "Keyed requests by the source that supplied the sticky routing key"
    );
    describe_gauge!(
        "smg_cache_tree_chars",
        "Cache-aware string tree cached characters by model (summed across tenants)"
    );
    describe_gauge!(
        "smg_cache_tree_tokens",
        "Cache-aware token tree cached tokens by model (summed across tenants)"
    );
    describe_gauge!(
        "smg_cache_tree_tenants",
        "Cache-aware tree tenant count by model and tree (string/token)"
    );
    describe_gauge!(
        "smg_cache_placement_entries",
        "Cache-aware hash-index placement entries by model (keys with a live holder)"
    );
    describe_counter!(
        "smg_cache_aware_policy_branch_total",
        "Cache-aware tree-mode selection branch (tree_match, spill, expected_wait_fallback, \
         first_healthy_fallback)"
    );
    describe_counter!(
        "smg_cache_aware_prefix_margin_held_total",
        "Cache-aware affinity picks the spill gate's swing-sized margin alone would have spilled, \
         kept on their holder by the price of the prefix it holds (--balance-prefix-margin-tokens)"
    );
    describe_counter!(
        "smg_policy_inflight_reconciled_total",
        "Policy bookings released at the per-poll in-flight reconciliation because their \
         completion never arrived, by policy"
    );
    describe_histogram!(
        "smg_cache_aware_match_ratio",
        "The predicted overlap of a cache-aware routing decision as a ratio (0..1), one observation per decision: in tree mode the best prefix match in the tree (matched/input units, whatever branch the decision took); in event mode the chosen worker's overlap in blocks over the request's blocks (0 where the chosen worker holds none: a miss, a spill, a warm-up placement)"
    );

    // Layer 3: Worker resilience metrics (retry)
    describe_counter!(
        "smg_worker_retries_total",
        "Total retry attempts by worker_type and endpoint"
    );
    describe_counter!(
        "smg_worker_retries_exhausted_total",
        "Requests that exhausted all retries by worker_type and endpoint"
    );
    describe_histogram!(
        "smg_worker_retry_backoff_seconds",
        "Retry backoff duration by attempt number"
    );

    // Layer 4: Discovery metrics
    smg_service_discovery::describe_series();

    // Layer 5: MCP metrics
    describe_counter!(
        "smg_mcp_tool_calls_total",
        "Total MCP tool invocations by model, tool_name, result"
    );
    describe_histogram!(
        "smg_mcp_tool_duration_seconds",
        "MCP tool execution duration by model, tool_name"
    );
    describe_gauge!("smg_mcp_servers_active", "Active MCP server connections");
    describe_counter!(
        "smg_mcp_tool_iterations_total",
        "Tool loop iterations in Responses API by model"
    );

    // Multimodal tensor transport (shm vs inline), labeled by runtime.
    describe_counter!(
        "smg_mm_tensors_total",
        "Multimodal tensors sent, by runtime and transport path (shm/inline)"
    );
    describe_counter!(
        "smg_mm_tensor_bytes_total",
        "Multimodal tensor bytes sent, by runtime and transport path (shm/inline)"
    );
    describe_counter!(
        "smg_mm_shm_write_failures_total",
        "SHM tensor write attempts that failed and fell back to inline, by runtime"
    );
    describe_counter!(
        "smg_mm_processing_total",
        "Multimodal requests by processing location (router/worker) and resolution reason"
    );
    describe_gauge!(
        "smg_mm_turbojpeg_available",
        "1 when JPEGs decode through libjpeg-turbo (PIL's pixels), 0 through the pure-Rust fallback"
    );
    describe_gauge!(
        "smg_mm_webp_available",
        "1 when WebP files decode through libwebp, 0 through the pure-Rust fallback"
    );
    // The router-side pixel cache (--mm-pixel-cache-mb): one per process,
    // shared by every model, keyed by image and model configuration.
    describe_counter!(
        "smg_mm_pixel_cache_hits_total",
        "Pixel cache lookups that found the image's preprocessed encoder input, by model"
    );
    describe_counter!(
        "smg_mm_pixel_cache_misses_total",
        "Pixel cache lookups that had to preprocess the image, by model"
    );
    describe_counter!(
        "smg_mm_pixel_cache_budget_bypasses_total",
        "Image requests that bypassed the pixel cache because their working set exceeds its byte budget, by model"
    );
    describe_counter!(
        "smg_mm_pixel_cache_evictions_total",
        "Entries the pixel cache dropped to stay within its byte budget"
    );
    describe_gauge!(
        "smg_mm_pixel_cache_bytes",
        "Bytes of preprocessed encoder inputs the pixel cache holds"
    );
    describe_gauge!("smg_mm_pixel_cache_entries", "Images the pixel cache holds");
    describe_gauge!(
        "smg_mm_inflight_bytes",
        "Bytes of preprocessed media the gateway holds in flight for engines, summed over every router of the process (a gateway in IGW mode builds three, each with the configured budget, so the gauge may read up to three budgets)"
    );
    // The media pipeline per request, from the clock reads and counts it
    // already makes: its stages' time, the items a request carries and its
    // fetches.
    describe_histogram!(
        "smg_mm_processing_seconds",
        "Seconds a multimodal request spent in a processing stage (config_lookup, media_fetch_decode, preprocess, token_expand, total), by model; pool_queue is one observation per blocking task, the wait for a pool thread, so its count is tasks, not requests"
    );
    describe_histogram!(
        "smg_mm_media_per_request",
        "Media items a generation request carried, by model and modality, counted before any refusal or limit (a refused request counts; a token count does not)"
    );
    describe_histogram!(
        "smg_mm_request_images",
        "Images a generation request carried, by model and the source class of its media (none, inline, remote, hybrid: inline is a data: URL or carried bytes, remote a fetched URL, read over every modality), observed once per request that reaches media resolution, a request without media included (0 under none)"
    );
    describe_counter!(
        "smg_mm_media_items_total",
        "Media items generation requests carried, by model, modality and source (url, data_url, inline, read as the fetch reads them), counted before any refusal or limit"
    );
    describe_histogram!(
        "smg_mm_media_fetch_seconds",
        "Seconds one media fetch and decode took, by model, modality, source and outcome (ok, too_large, client_fault, host_fault)"
    );
    describe_counter!(
        "smg_mm_media_fetch_bytes_total",
        "Bytes the media fetches brought in, by model, modality and source"
    );
    describe_counter!(
        "smg_responses_stream_failures_total",
        "Responses streams that ended with a response.failed terminal, by model and reason"
    );
    describe_counter!(
        "smg_responses_not_stored_total",
        "Responses with store: true whose record or conversation linking was not written, by mode (foreground: the request was answered with the failure; background: the record keeps what it had) and step (record: nothing is stored; linking: the record is stored, its items are not in the conversation)"
    );
    describe_counter!(
        "smg_messages_thread_turns_total",
        "Message-thread turns on the Messages route, by op (create, continue) and outcome (stored: the answer is kept under its id; reissued: kept under a fresh id, the answer's id being a stored turn's; unstored: answered, not kept, a stream cut before its message_stop among them; not_found, mismatch: a continue refused before the model ran)"
    );

    // Layer 0: Tokio runtime self-observability (event-loop canary + sampler).
    super::runtime_metrics::describe();
    super::history_metrics::describe();

    // Initialize mesh metrics
    smg_mesh::init_mesh_metrics();

    // RL control plane metrics (emit only when the plane is enabled).
    smg_rl::init_rl_metrics();

    // Priority scheduler metrics (no-op at scrape time unless the scheduler
    // is enabled and recording).
    use crate::middleware::scheduler::metrics as scheduler_metrics;
    scheduler_metrics::describe();
}

/// Publish the protection and retry families at zero.
///
/// These counters are otherwise created by their first event, so a fresh
/// gateway exposes no series for them: dashboards cannot tell "never
/// happened" from "not exported", `absent()` alerts fire on a healthy
/// process, and `increase()` misses the first event after a restart. Every
/// label value the recording paths can emit is enumerated here. Call once the
/// recorder is installed (see [`start_prometheus`]); `absolute(0)` never
/// lowers a counter, so calling it again is harmless.
pub fn init_startup_series() {
    use metrics_labels::{
        ENDPOINT_AUDIO_TRANSCRIPTIONS, ENDPOINT_CHAT, ENDPOINT_COMPLETIONS, ENDPOINT_DECISIONS,
        ENDPOINT_GENERATE, ENDPOINT_MESSAGES, ENDPOINT_RERANK, ENDPOINT_RESPONSES,
        ENDPOINT_SYSTEMONE, WORKER_DECODE, WORKER_PREFILL, WORKER_REGULAR,
    };
    use smg_worker::health::overload::{STAGE_DISPATCH, STAGE_PD_ADMISSION, STAGE_SELECTION};

    // Shedding happens at every stage; the two fallbacks only while the
    // candidate pool is assembled.
    for stage in [STAGE_SELECTION, STAGE_DISPATCH, STAGE_PD_ADMISSION] {
        counter!("smg_worker_overload_shed_total", "stage" => stage).absolute(0);
    }
    counter!("smg_worker_overload_fallback_total", "stage" => STAGE_SELECTION).absolute(0);
    counter!("smg_worker_liveness_fallback_total", "stage" => STAGE_SELECTION).absolute(0);
    // The connection limit's refusals: a zero series from the first scrape,
    // bound or no bound, so a dashboard reads nothing refused as a zero.
    counter!("smg_http_connections_rejected_total").absolute(0);

    // The retrying dispatch loops label their endpoint through
    // `route_to_endpoint`, whose values these are, for regular and
    // prefill/decode workers.
    for worker_type in [WORKER_REGULAR, WORKER_PREFILL, WORKER_DECODE] {
        for endpoint in [
            ENDPOINT_CHAT,
            ENDPOINT_GENERATE,
            ENDPOINT_COMPLETIONS,
            ENDPOINT_RERANK,
            ENDPOINT_RESPONSES,
            ENDPOINT_DECISIONS,
            ENDPOINT_SYSTEMONE,
            ENDPOINT_MESSAGES,
            ENDPOINT_AUDIO_TRANSCRIPTIONS,
            "other",
        ] {
            counter!(
                "smg_worker_retries_total",
                "worker_type" => worker_type,
                "endpoint" => endpoint
            )
            .absolute(0);
            counter!(
                "smg_worker_retries_exhausted_total",
                "worker_type" => worker_type,
                "endpoint" => endpoint
            )
            .absolute(0);
        }
    }
    // A histogram has no zero to set: registering the first attempt's series
    // publishes the family with an empty distribution.
    let _ = histogram!("smg_worker_retry_backoff_seconds", "attempt" => "1");
    super::heap_profile::init_series();
}

/// Publish process-lifetime totals without scanning or retaining tokenizer instances.
pub(super) fn record_tokenizer_cache_activity() {
    for stats in cache_activity_stats() {
        record_tokenizer_cache_activity_snapshot(stats);
    }
}

pub(super) fn record_detokenize_activity() {
    for stats in decode_activity_stats() {
        counter!("smg_detokenize_prefix_mismatch_total", "kind" => stats.kind)
            .absolute(stats.prefix_mismatches);
    }
}

/// Publish the encode path chosen per tokenizer load and the inputs that left
/// the direct encoder, process-lifetime totals like the cache activity.
pub(super) fn record_tokenizer_encode_activity() {
    for stats in encode_path_loads() {
        counter!("smg_tokenizer_encode_path_loads_total", "path" => stats.path.as_str())
            .absolute(stats.loads);
    }
    for stats in encode_fallbacks() {
        counter!("smg_tokenizer_encode_fallbacks_total", "reason" => stats.reason)
            .absolute(stats.encodes);
    }
}

fn record_tokenizer_cache_activity_snapshot(stats: CacheActivityStats) {
    counter!("smg_tokenizer_cache_lookups_total", "layer" => stats.layer, "result" => "hit")
        .absolute(stats.hits);
    counter!("smg_tokenizer_cache_lookups_total", "layer" => stats.layer, "result" => "miss")
        .absolute(stats.misses);
    counter!("smg_tokenizer_cache_evictions_total", "layer" => stats.layer)
        .absolute(stats.evictions);
    counter!("smg_tokenizer_cache_reused_bytes_total", "layer" => stats.layer)
        .absolute(stats.reused_bytes);
}

#[expect(
    clippy::expect_used,
    reason = "startup initialization — metrics exporter must be installed or the process cannot serve metrics"
)]
pub fn start_prometheus(config: PrometheusConfig) -> PrometheusHandle {
    let bucket_overrides = histogram_bucket_overrides(config.duration_buckets);
    let mut builder =
        PrometheusBuilder::new().upkeep_timeout(Duration::from_secs(UPKEEP_INTERVAL_SECS));
    for (matcher, buckets) in &bucket_overrides {
        builder = builder
            .set_buckets_for_metric(matcher.clone(), buckets)
            .expect("histogram bucket overrides are not empty");
    }
    let recorder = builder.build_recorder();
    let handle = recorder.handle();

    // Series labelled `worker` live in a store of their own so they can be
    // dropped with their worker (see `smg_worker::metrics::series`); it renders histograms
    // with the exporter's buckets and quantiles.
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        Some(bucket_overrides),
        None,
    )));
    worker_series::install(Arc::clone(&workers));
    metrics::set_global_recorder(WorkerSeriesRecorder::new(recorder, workers))
        .expect("failed to install Prometheus recorder");
    handles::recorder_installed();
    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    allocator_stats::start_reporting();

    // Descriptions are kept by whichever recorder is current when they are
    // registered. Before `set_global_recorder()` that is the no-op recorder, so
    // the `describe_*!` calls must run after it or `/metrics` has `# TYPE`
    // lines but no `# HELP` text.
    init_metrics();

    handle
}

/// The exporter's default summary quantiles, for the summaries of the
/// per-worker store (`PrometheusBuilder` keeps its own copy private).
const DEFAULT_SUMMARY_QUANTILES: &[f64] = &[0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0];

/// Histogram buckets by metric name. The exporter renders a histogram with
/// these buckets for a matching name and a summary for any other name.
fn histogram_bucket_overrides(duration_buckets: Option<Vec<f64>>) -> HashMap<Matcher, Vec<f64>> {
    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket: Vec<f64> = duration_buckets.unwrap_or_else(|| {
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 480.0, 900.0, 1200.0, 1800.0, 2700.0, 3600.0,
            5400.0, 7200.0,
        ]
    });

    // The event-loop canary needs its own buckets: its name does not end in
    // `duration_seconds`, and the request-latency buckets above are far too
    // coarse for 0-1s wake drift. Without explicit buckets the recorder would
    // render it as a summary.
    let canary_matcher = Matcher::Full(super::runtime_metrics::EVENT_LOOP_DELAY_SECONDS.into());

    // TTFT and TPOT (per-request mean inter-token latency) end in `_seconds`
    // but NOT `duration_seconds`, so without explicit buckets the recorder
    // renders them as summaries (quantile lines only) — not heatmap-able. They
    // keep the request-latency buckets (0.001-7200s, an operator's list with
    // them), so recording rules on their `le` values hold; the same
    // observations on edges that resolve them are the two families below.
    let ttft_matcher = Matcher::Suffix(String::from("ttft_seconds"));
    let tpot_matcher = Matcher::Suffix(String::from("tpot_seconds"));

    // The dispatch wait (edge entry to the answering attempt's dispatch) ends
    // in `_wait_seconds`, not `duration_seconds`: matched by its full name, it
    // keeps the request-latency buckets (an operator's list with them), as
    // the TTFT it adds up with does.
    let dispatch_wait_matcher = Matcher::Full(String::from(DISPATCH_WAIT_SECONDS));

    // The first-token and per-output-token families: their own log-spaced
    // edges, which `--prometheus-duration-buckets` does not replace.
    let first_token_matcher = Matcher::Full(String::from(TIME_TO_FIRST_TOKEN_SECONDS));
    let per_output_token_matcher =
        Matcher::Full(String::from(REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS));

    // The cache-aware match ratio is a dimensionless 0..1 value: no `_seconds`
    // matcher applies, so it also needs its own buckets or it renders as a
    // summary.
    let match_ratio_matcher = Matcher::Full(String::from("smg_cache_aware_match_ratio"));

    // The KV event lag is milliseconds to seconds, between the request
    // buckets and the microsecond ones below: its own buckets, or the
    // recorder renders it as a summary.
    let kv_lag_matcher = Matcher::Full(String::from(KV_EVENT_LAG_SECONDS));

    // The KV index's lookup and apply times are microseconds: their own
    // buckets, or the recorder renders them as summaries.
    let kv_lookup_matcher = Matcher::Full(String::from("smg_kv_index_lookup_seconds"));
    let kv_apply_matcher = Matcher::Full(String::from(KV_EVENT_APPLY_SECONDS));

    // The retry backoff is tens of milliseconds to the configured maximum
    // (seconds): its own buckets, or the recorder renders it as a summary.
    let retry_backoff_matcher = Matcher::Full(String::from("smg_worker_retry_backoff_seconds"));

    // The malformed bytes of an output run from a marker's worth to a whole
    // answer: their own buckets, or the recorder renders them as a summary.
    let malformed_bytes_matcher = Matcher::Full(String::from(symphony_telemetry::MALFORMED_BYTES));
    // The media pipeline's histograms (seconds, and items per request): their
    // own buckets, or the recorder renders them as summaries.
    let mm_processing_matcher = Matcher::Full(String::from("smg_mm_processing_seconds"));
    let mm_fetch_matcher = Matcher::Full(String::from("smg_mm_media_fetch_seconds"));
    let mm_per_request_matcher = Matcher::Full(String::from("smg_mm_media_per_request"));
    let mm_request_images_matcher = Matcher::Full(String::from("smg_mm_request_images"));

    HashMap::from([
        (duration_matcher, duration_bucket.clone()),
        (ttft_matcher, duration_bucket.clone()),
        (tpot_matcher, duration_bucket.clone()),
        (dispatch_wait_matcher, duration_bucket),
        (first_token_matcher, TIME_TO_FIRST_TOKEN_BUCKETS.to_vec()),
        (
            per_output_token_matcher,
            REQUEST_TIME_PER_OUTPUT_TOKEN_BUCKETS.to_vec(),
        ),
        (
            canary_matcher,
            super::runtime_metrics::EVENT_LOOP_DELAY_BUCKETS.to_vec(),
        ),
        (
            match_ratio_matcher,
            CACHE_AWARE_MATCH_RATIO_BUCKETS.to_vec(),
        ),
        (kv_lookup_matcher, KV_INDEX_MICRO_BUCKETS.to_vec()),
        (kv_apply_matcher, KV_INDEX_MICRO_BUCKETS.to_vec()),
        (kv_lag_matcher, KV_EVENT_LAG_BUCKETS.to_vec()),
        (retry_backoff_matcher, WORKER_RETRY_BACKOFF_BUCKETS.to_vec()),
        (
            malformed_bytes_matcher,
            symphony_telemetry::MALFORMED_BYTES_BUCKETS.to_vec(),
        ),
        (mm_processing_matcher, MM_SECONDS_BUCKETS.to_vec()),
        (mm_fetch_matcher, MM_SECONDS_BUCKETS.to_vec()),
        (
            mm_per_request_matcher,
            MM_MEDIA_PER_REQUEST_BUCKETS.to_vec(),
        ),
        (
            mm_request_images_matcher,
            MM_REQUEST_IMAGES_BUCKETS.to_vec(),
        ),
    ])
}

#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
pub(crate) mod allocator_stats {
    use metrics::{describe_gauge, gauge};

    use super::jemalloc_is_global;

    pub(crate) fn describe() {
        if !jemalloc_is_global() {
            return;
        }
        describe_gauge!(
            "smg_allocator_allocated_bytes",
            "Bytes in live Rust allocations managed by SMG's jemalloc instance"
        );
        describe_gauge!(
            "smg_allocator_active_bytes",
            "Bytes in active pages for SMG's Rust jemalloc heap"
        );
        describe_gauge!(
            "smg_allocator_resident_bytes",
            "Upper bound on resident bytes for SMG's Rust jemalloc heap"
        );
        describe_gauge!(
            "smg_allocator_metadata_bytes",
            "Metadata bytes for SMG's Rust jemalloc instance"
        );
    }

    fn record() {
        use tikv_jemalloc_ctl::{epoch, stats};
        if epoch::advance().is_err() {
            return;
        }
        if let Ok(v) = stats::allocated::read() {
            gauge!("smg_allocator_allocated_bytes").set(v as f64);
        }
        if let Ok(v) = stats::active::read() {
            gauge!("smg_allocator_active_bytes").set(v as f64);
        }
        if let Ok(v) = stats::resident::read() {
            gauge!("smg_allocator_resident_bytes").set(v as f64);
        }
        if let Ok(v) = stats::metadata::read() {
            gauge!("smg_allocator_metadata_bytes").set(v as f64);
        }
    }

    /// The gauges are published where this jemalloc serves the process's Rust
    /// allocations, and nowhere else.
    pub(crate) fn start_reporting() {
        if !jemalloc_is_global() {
            return;
        }
        record();
        // Plain thread: metrics must not depend on a runtime being alive.
        let _ = std::thread::Builder::new()
            .name("smg-allocator-stats".into())
            .spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                record();
            });
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn the_allocator_of_the_lib_tests_is_found() {
            // The lib's tests declare jemalloc as their allocator (lib.rs),
            // so the one allocation moves its count.
            assert!(super::jemalloc_is_global());
        }

        #[test]
        fn jemalloc_stats_interface_readable() {
            use tikv_jemalloc_ctl::{epoch, stats};
            epoch::advance().expect("epoch advance");
            stats::allocated::read().expect("stats.allocated readable");
            stats::active::read().expect("stats.active readable");
            stats::resident::read().expect("stats.resident readable");
            stats::metadata::read().expect("stats.metadata readable");
        }
    }
}

/// SMG Metrics helper struct for the new layered metrics architecture.
///
/// Design principles for low overhead:
/// - Dynamic labels use string interning (single allocation per unique value)
/// - Static labels use the metrics crate's internal caching
pub struct Metrics;

/// A streamed request the gateway aborted at the engine because its own
/// stop had ended every choice for the client while the engine was still
/// generating (see `smg_router_local_stop_aborts_total`).
pub struct LocalStopAbort<'a> {
    pub router_type: &'static str,
    pub backend_type: &'static str,
    pub model_id: &'a str,
    pub endpoint: &'static str,
    /// Output tokens the engine streamed for the stopped choices after their
    /// stop, read before the abort.
    pub tail_tokens: u64,
}

/// Parameters for recording streaming metrics.
pub struct StreamingMetricsParams<'a> {
    /// Router type label (e.g., "grpc", "http")
    pub router_type: &'static str,
    /// Backend type label (e.g., "regular", "pd")
    pub backend_type: &'static str,
    /// Model identifier (will be converted to owned String for metrics)
    pub model_id: &'a str,
    /// Endpoint label (e.g., "chat", "generate")
    pub endpoint: &'static str,
    /// Time to first token (None if no tokens were generated)
    pub ttft: Option<Duration>,
    /// The request's wait ahead of the dispatch `ttft` runs from: its entry
    /// at the edge to that dispatch (for a later turn of a streamed tool
    /// loop, that turn's start to its dispatch). Recorded only with `ttft`.
    /// None for a generation dispatched under no entry stamp, and for a
    /// collected answer.
    pub dispatch_wait: Option<Duration>,
    /// Total generation time
    pub generation_duration: Duration,
    /// Input token count (None for endpoints that don't track this)
    pub input_tokens: Option<u64>,
    /// Prompt tokens the engine reported it served from its cache (None for
    /// endpoints that don't track this)
    pub cached_tokens: Option<u64>,
    /// Output token count
    pub output_tokens: u64,
}

impl Metrics {
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: &'static str,
    ) {
        RouterMetrics::record_router_request(
            router_type,
            backend_type,
            connection_mode,
            model_id,
            endpoint,
            streaming,
        );
    }
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        RouterMetrics::record_router_duration(
            router_type,
            backend_type,
            connection_mode,
            model_id,
            endpoint,
            duration,
        );
    }
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
        reason: &'static str,
    ) {
        RouterMetrics::record_router_error(
            router_type,
            backend_type,
            connection_mode,
            model_id,
            endpoint,
            error_type,
            reason,
        );
    }
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        RouterMetrics::record_worker_retry(worker_type, endpoint);
    }
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        RouterMetrics::record_worker_retries_exhausted(worker_type, endpoint);
    }
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        RouterMetrics::record_worker_retry_backoff(attempt, duration);
    }
    pub fn record_mcp_tool_call(model_id: &str, tool_name: &str, result: &'static str) {
        RouterMetrics::record_mcp_tool_call(model_id, tool_name, result);
    }
    pub fn record_mcp_tool_duration(model_id: &str, tool_name: &str, duration: Duration) {
        RouterMetrics::record_mcp_tool_duration(model_id, tool_name, duration);
    }
    pub fn record_mcp_tool_iteration(model_id: &str) {
        RouterMetrics::record_mcp_tool_iteration(model_id);
    }
    /// Record an HTTP request.
    /// Here we want a metric to directly reflect user's experience ("I am sending a request")
    /// when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    pub fn record_http_request(method: &'static str, path: &str) {
        let build = || {
            let path_interned = intern_string(path);
            counter!(
                "smg_http_requests_total",
                "method" => method,
                "path" => path_interned,
            )
        };
        if !handles::keep_handles() {
            build().increment(1);
            return;
        }
        handles::HTTP_REQUESTS.with_borrow_mut(|by_method| {
            handles::with_handle(
                by_method.entry(method).or_default(),
                path,
                build,
                |counter| {
                    counter.increment(1);
                },
            );
        });
    }

    /// Record HTTP request duration.
    /// For best performance, pass static strings for method.
    pub fn record_http_duration(method: &'static str, path: &str, duration: Duration) {
        let build = || {
            let path_interned = intern_string(path);
            histogram!(
                "smg_http_request_duration_seconds",
                "method" => method,
                "path" => path_interned
            )
        };
        let seconds = duration.as_secs_f64();
        if !handles::keep_handles() {
            build().record(seconds);
            return;
        }
        handles::HTTP_DURATIONS.with_borrow_mut(|by_method| {
            handles::with_handle(
                by_method.entry(method).or_default(),
                path,
                build,
                |histogram| {
                    histogram.record(seconds);
                },
            );
        });
    }

    /// Set `smg_http_connections_active`: the count of HTTP requests in flight
    /// at the edge layer (the callers pass the in-flight request tracker's
    /// count), not of open connections.
    pub fn set_http_connections_active(count: usize) {
        handles::with_single(
            &handles::HTTP_CONNECTIONS_ACTIVE,
            || gauge!("smg_http_connections_active"),
            |active| active.set(count as f64),
        );
    }

    /// Record HTTP response.
    pub fn record_http_response(path: &str, status_code: u16, error_code: &str) {
        let build = || {
            let path_interned = intern_string(path);
            let status_str: Cow<'static, str> = status_code_to_cow(status_code);
            let error_interned = intern_string(error_code);
            counter!(
                "smg_http_responses_total",
                "path" => path_interned,
                "status_code" => status_str,
                "error_code" => error_interned
            )
        };
        if !handles::keep_handles() {
            build().increment(1);
            return;
        }
        handles::HTTP_RESPONSES.with_borrow_mut(|by_path| {
            handles::with_handle(
                handles::nested(by_path, path)
                    .entry(status_code)
                    .or_default(),
                error_code,
                build,
                |counter| counter.increment(1),
            );
        });
    }

    /// Record a connection closed by one of the HTTP edge's connection
    /// timeouts (`header_read` or `idle`).
    pub fn record_http_connection_timeout(timeout: &'static str) {
        counter!(
            "smg_http_connection_timeouts_total",
            "timeout" => timeout
        )
        .increment(1);
    }

    /// Record a connection closed at accept for arriving beyond the
    /// connection limit.
    pub fn record_http_connection_rejected() {
        counter!("smg_http_connections_rejected_total").increment(1);
    }

    /// Record rate limit decision.
    pub fn record_http_rate_limit(result: &'static str) {
        handles::with_static(
            &handles::HTTP_RATE_LIMITS,
            result,
            || counter!("smg_http_rate_limit_total", "result" => result),
            |count| count.increment(1),
        );
    }

    /// Track a request entering the admission queue.
    pub fn record_admission_queue_entered() {
        handles::with_single(
            &handles::ADMISSION_QUEUE_DEPTH,
            || gauge!("smg_admission_queue_depth"),
            |depth| depth.increment(1.0),
        );
    }

    /// Track a request leaving the admission queue (admitted, rejected, or cancelled).
    pub fn record_admission_queue_exited() {
        handles::with_single(
            &handles::ADMISSION_QUEUE_DEPTH,
            || gauge!("smg_admission_queue_depth"),
            |depth| depth.decrement(1.0),
        );
    }

    /// Record a request rejected at admission.
    pub fn record_admission_rejected(reason: &'static str) {
        counter!(
            "smg_admission_queue_rejected_total",
            "reason" => reason
        )
        .increment(1);
    }

    /// Track acquisition of an admission token.
    pub fn record_admission_inflight_acquired() {
        handles::with_single(
            &handles::ADMISSION_INFLIGHT,
            || gauge!("smg_admission_inflight"),
            |inflight| inflight.increment(1.0),
        );
    }

    /// Track release of an admission token.
    pub fn record_admission_inflight_released() {
        handles::with_single(
            &handles::ADMISSION_INFLIGHT,
            || gauge!("smg_admission_inflight"),
            |inflight| inflight.decrement(1.0),
        );
    }

    /// Record one multimodal tensor sent over `path` ("inline"|"shm"|"remote") for `runtime`.
    pub fn record_mm_tensor(runtime: &'static str, path: &'static str, nbytes: usize) {
        counter!("smg_mm_tensors_total", "runtime" => runtime, "path" => path).increment(1);
        counter!("smg_mm_tensor_bytes_total", "runtime" => runtime, "path" => path)
            .increment(nbytes as u64);
    }

    /// Record a SHM tensor write that failed and fell back to inline, for `runtime`.
    pub fn record_mm_shm_write_failure(runtime: &'static str) {
        counter!("smg_mm_shm_write_failures_total", "runtime" => runtime).increment(1);
    }

    pub fn set_mm_turbojpeg_available(available: bool) {
        gauge!("smg_mm_turbojpeg_available").set(if available { 1.0 } else { 0.0 });
    }

    pub fn set_mm_webp_available(available: bool) {
        gauge!("smg_mm_webp_available").set(if available { 1.0 } else { 0.0 });
    }

    /// The pixel cache's occupancy and eviction series, at zero: called when
    /// the cache is built, so they are on the first scrape.
    pub fn init_mm_pixel_cache_series() {
        counter!("smg_mm_pixel_cache_evictions_total").absolute(0);
        gauge!("smg_mm_pixel_cache_bytes").set(0.0);
        gauge!("smg_mm_pixel_cache_entries").set(0.0);
    }

    /// The in-flight bytes gauge at zero: called when the media pipeline is
    /// built, so it is on the first scrape.
    pub fn init_mm_inflight_series() {
        gauge!("smg_mm_inflight_bytes").set(0.0);
    }

    /// A request's preprocessed media bytes from their reservation to their
    /// release: with a budget the bytes the budget counts, without one the
    /// count alone. The gauge moves by the bytes, so concurrent requests add
    /// up, across every router of the process: a gateway in IGW mode builds
    /// three gRPC routers, each with its own budget of the configured size, so
    /// the one series may read up to three budgets while no router refuses.
    pub fn record_mm_inflight_reserved(bytes: usize) {
        gauge!("smg_mm_inflight_bytes").increment(bytes as f64);
    }

    pub fn record_mm_inflight_released(bytes: usize) {
        gauge!("smg_mm_inflight_bytes").decrement(bytes as f64);
    }

    /// One pixel cache lookup for `model_id`: a hit served the image's
    /// preprocessed encoder input, a miss preprocessed it. Both series exist
    /// from the model's first lookup, so a hit rate reads from the start.
    pub fn record_mm_pixel_cache_lookup(model_id: &str, hit: bool) {
        let model = intern_model_label(model_id);
        counter!("smg_mm_pixel_cache_hits_total", "model" => model.clone())
            .increment(u64::from(hit));
        counter!("smg_mm_pixel_cache_misses_total", "model" => model).increment(u64::from(!hit));
    }

    /// A request bypassed pixel cache lookup because its working set is too large.
    pub fn record_mm_pixel_cache_budget_bypass(model_id: &str) {
        counter!(
            "smg_mm_pixel_cache_budget_bypasses_total",
            "model" => intern_model_label(model_id)
        )
        .increment(1);
    }

    /// The pixel cache after an insert: the entries it dropped to stay within
    /// its budget, and what it holds now.
    pub fn record_mm_pixel_cache_insert(evicted: u64, bytes: usize, entries: usize) {
        counter!("smg_mm_pixel_cache_evictions_total").increment(evicted);
        gauge!("smg_mm_pixel_cache_bytes").set(bytes as f64);
        gauge!("smg_mm_pixel_cache_entries").set(entries as f64);
    }

    /// Count a Responses stream whose terminal event was `response.failed`.
    ///
    /// `reason` is a bounded label (`stream_error`, `server_error`, `other`).
    pub fn record_responses_stream_failure(model_id: &str, reason: &'static str) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_responses_stream_failures_total",
            "model" => model,
            "reason" => reason
        )
        .increment(1);
    }

    /// A `store: true` response whose record (`step` record: nothing is
    /// stored) or conversation linking (`step` linking) was not written, by
    /// `mode` (foreground, background).
    pub fn record_response_not_stored(mode: &'static str, step: &'static str) {
        counter!(
            "smg_responses_not_stored_total",
            "mode" => mode,
            "step" => step
        )
        .increment(1);
    }

    /// A message-thread turn, by `op` (create, continue) and `outcome`:
    /// `stored` (the answer is kept under its id), `reissued` (the answer's
    /// id was a stored turn's: kept under a fresh id, in the answer too),
    /// `unstored` (answered and not kept: a streamed answer, an error, an
    /// answer the gateway could not read, a write the store refused),
    /// `not_found` and `mismatch` (a continue refused before the model ran).
    pub fn record_message_thread_turn(op: &'static str, outcome: &'static str) {
        counter!(
            "smg_messages_thread_turns_total",
            "op" => op,
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Record where a multimodal request's media is processed and why.
    pub fn record_mm_processing(model_id: &str, mode: &'static str, reason: &'static str) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_mm_processing_total",
            "model" => model,
            "mode" => mode,
            "reason" => reason
        )
        .increment(1);
    }

    /// The interned label of `model_id`, for a handle that reports under it
    /// (the media tracker's fetch observer).
    pub fn interned_model_label(model_id: &str) -> Arc<str> {
        intern_model_label(model_id)
    }

    /// One processing stage of a multimodal request: `seconds` spent in
    /// `stage` (`config_lookup`, `media_fetch_decode`, `preprocess`,
    /// `token_expand`, `total`), the clock reads the timing log makes; and
    /// `pool_queue`, one observation per blocking task, the time it waited
    /// in the pool's queue before it ran.
    pub fn record_mm_processing_stage(model_id: &str, stage: &'static str, seconds: f64) {
        let model = intern_model_label(model_id);
        histogram!("smg_mm_processing_seconds", "model" => model, "stage" => stage).record(seconds);
    }

    /// The media a request carries of `modality`, counted once per request
    /// before its limits are checked: `count` items, `by_source` of them per
    /// source (`url`, `data_url`, `inline`; a zero is skipped).
    pub fn record_mm_request_media(
        model_id: &str,
        modality: llm_multimodal::Modality,
        count: usize,
        by_source: &[(&'static str, usize)],
    ) {
        let model = intern_model_label(model_id);
        let modality = mm_modality_label(modality);
        histogram!(
            "smg_mm_media_per_request",
            "model" => model.clone(),
            "modality" => modality
        )
        .record(count as f64);
        for (source, items) in by_source {
            if *items > 0 {
                counter!(
                    "smg_mm_media_items_total",
                    "model" => model.clone(),
                    "modality" => modality,
                    "source" => *source
                )
                .increment(*items as u64);
            }
        }
    }

    /// The images one generation request carried, by the source class of its
    /// media (`none`, `inline`, `remote`, `hybrid`, read over every
    /// modality). Observed once per request that reaches media resolution, a
    /// request without media included (`images` 0, class `none`), so the
    /// family's count is the requests and its 0 bucket the ones without an
    /// image.
    pub fn record_mm_request_images(model_id: &str, source_class: &'static str, images: usize) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_mm_request_images",
            "model" => model,
            "source_class" => source_class
        )
        .record(images as f64);
    }

    /// One media fetch, as the tracker reports it: its time by outcome, and
    /// the bytes it brought in.
    pub fn record_mm_fetch(model_id: &str, report: &llm_multimodal::FetchReport) {
        let model = intern_model_label(model_id);
        let modality = mm_modality_label(report.modality);
        let source = report.source.as_str();
        histogram!(
            "smg_mm_media_fetch_seconds",
            "model" => model.clone(),
            "modality" => modality,
            "source" => source,
            "outcome" => report.outcome.as_str()
        )
        .record(report.duration.as_secs_f64());
        if report.bytes > 0 {
            counter!(
                "smg_mm_media_fetch_bytes_total",
                "model" => model,
                "modality" => modality,
                "source" => source
            )
            .increment(report.bytes as u64);
        }
    }

    // ========================================================================
    // Layer 2: Router metrics
    // ========================================================================

    /// Record a single-shot resend after a pre-response transport failure.
    pub fn record_upstream_send_retry(router_type: &'static str) {
        counter!(
            "smg_router_upstream_send_retries_total",
            "router_type" => router_type
        )
        .increment(1);
    }

    /// Record one per-request body-path decision with its dominant reason.
    pub fn record_request_body_path(path: &'static str, reason: &'static str) {
        handles::with_static(
            &handles::REQUEST_BODY_PATHS,
            (path, reason),
            || {
                counter!(
                    "smg_router_request_body_path_total",
                    "path" => path,
                    "reason" => reason
                )
            },
            |count| count.increment(1),
        );
    }

    /// Record request buffers freed at dispatch (retries disabled), sized by
    /// the serialized upstream body.
    pub fn record_request_buffers_released_early(bytes: usize) {
        handles::with_single(
            &handles::REQUEST_BUFFERS_RELEASED_EARLY,
            || counter!("smg_router_request_buffers_released_early_bytes_total"),
            |total| total.increment(bytes as u64),
        );
    }

    /// Record upstream backend response.
    /// Uses static strings for common status codes and interning for error_code.
    pub fn record_router_upstream_response(
        router_type: &'static str,
        status_code: u16,
        error_code: &str,
    ) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "smg_router_upstream_responses_total",
            "router_type" => router_type,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router inference metrics (gRPC only)
    // ========================================================================

    /// A request the gateway's own stop ended for the client while the
    /// engine was still generating: aborted at the engine, with the tail it
    /// had streamed past the stop. At most once per stream, so the handles
    /// are not kept.
    pub fn record_local_stop_abort(abort: LocalStopAbort<'_>) {
        let LocalStopAbort {
            router_type,
            backend_type,
            model_id,
            endpoint,
            tail_tokens,
        } = abort;
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_local_stop_aborts_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model.clone(),
            "endpoint" => endpoint
        )
        .increment(1);
        counter!(
            "smg_router_local_stop_tail_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .increment(tail_tokens);
    }

    /// Record an answer's token and latency families in a single batch call:
    /// TTFT, TPOT, generation duration and the token counters, with the TPOT
    /// computed here. A streamed answer records at its end; a collected one
    /// when its response is built, with no TTFT (no first token is seen on
    /// that path) and so no TPOT. The dispatch wait is recorded where the
    /// TTFT is and only with it, so the two families count the same requests.
    pub fn record_streaming_metrics(params: StreamingMetricsParams<'_>) {
        let StreamingMetricsParams {
            router_type,
            backend_type,
            model_id,
            endpoint,
            ttft,
            dispatch_wait,
            generation_duration,
            input_tokens,
            cached_tokens,
            output_tokens,
        } = params;

        // TPOT is only meaningful past the first token.
        let tpot = ttft.and_then(|ttft| {
            (output_tokens > 1)
                .then(|| generation_duration.saturating_sub(ttft) / (output_tokens as u32 - 1))
        });

        // The handles, built the first time a stream has something for
        // each of them; the model label is interned once per build.
        let model = || intern_model_label(model_id);
        let duration_histogram = |name: &'static str| {
            histogram!(
                name,
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => model(),
                "endpoint" => endpoint
            )
        };
        let ttft_histogram = || duration_histogram("smg_router_ttft_seconds");
        let dispatch_wait_histogram = || duration_histogram(DISPATCH_WAIT_SECONDS);
        let tpot_histogram = || duration_histogram("smg_router_tpot_seconds");
        let first_token_histogram = || duration_histogram(TIME_TO_FIRST_TOKEN_SECONDS);
        let per_output_token_histogram =
            || duration_histogram(REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS);
        let generation_histogram = || duration_histogram("smg_router_generation_duration_seconds");
        let tokens_counter = |token_type: &'static str| {
            counter!(
                "smg_router_tokens_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => model(),
                "endpoint" => endpoint,
                "token_type" => token_type
            )
        };
        let input_counter = || tokens_counter(metrics_labels::TOKEN_INPUT);
        let output_counter = || tokens_counter(metrics_labels::TOKEN_OUTPUT);
        let cached_counter = || tokens_counter(metrics_labels::TOKEN_CACHED);

        if !handles::keep_handles() {
            if let Some(ttft) = ttft {
                let seconds = ttft.as_secs_f64();
                ttft_histogram().record(seconds);
                first_token_histogram().record(seconds);
                if let Some(wait) = dispatch_wait {
                    dispatch_wait_histogram().record(wait.as_secs_f64());
                }
            }
            if let Some(tpot) = tpot {
                let seconds = tpot.as_secs_f64();
                tpot_histogram().record(seconds);
                per_output_token_histogram().record(seconds);
            }
            generation_histogram().record(generation_duration.as_secs_f64());
            if let Some(input) = input_tokens {
                input_counter().increment(input);
            }
            if let Some(cached) = cached_tokens {
                cached_counter().increment(cached);
            }
            output_counter().increment(output_tokens);
            return;
        }

        let labels = handles::StreamLabels {
            router_type,
            backend_type,
            endpoint,
        };
        handles::STREAMS.with_borrow_mut(|by_labels| {
            handles::with_model_handle(
                by_labels.entry(labels).or_default(),
                model_id,
                |_label| handles::StreamHandles::default(),
                |kept| {
                    if let Some(ttft) = ttft {
                        let seconds = ttft.as_secs_f64();
                        kept.ttft.get_or_init(ttft_histogram).record(seconds);
                        kept.first_token
                            .get_or_init(first_token_histogram)
                            .record(seconds);
                        if let Some(wait) = dispatch_wait {
                            kept.dispatch_wait
                                .get_or_init(dispatch_wait_histogram)
                                .record(wait.as_secs_f64());
                        }
                    }
                    if let Some(tpot) = tpot {
                        let seconds = tpot.as_secs_f64();
                        kept.tpot.get_or_init(tpot_histogram).record(seconds);
                        kept.per_output_token
                            .get_or_init(per_output_token_histogram)
                            .record(seconds);
                    }
                    kept.generation
                        .get_or_init(generation_histogram)
                        .record(generation_duration.as_secs_f64());
                    if let Some(input) = input_tokens {
                        kept.input_tokens
                            .get_or_init(input_counter)
                            .increment(input);
                    }
                    if let Some(cached) = cached_tokens {
                        kept.cached_tokens
                            .get_or_init(cached_counter)
                            .increment(cached);
                    }
                    kept.output_tokens
                        .get_or_init(output_counter)
                        .increment(output_tokens);
                },
            );
        });
    }

    // ========================================================================
    // Layer 2: PD disaggregation metrics
    //
    // Per-request, engine-agnostic signals that no backend can self-report: SMG
    // is the only component that sees both the prefill and decode sides. All
    // durations come from a monotonic clock and are recorded once per request
    // (never per retry attempt).
    // ========================================================================

    /// Record the PD prefill phase: prefill dispatch to the end of the prefill
    /// response.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_prefill_duration(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_prefill_duration_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record the PD relay phase of sequential dispatch: the gateway's own gap
    /// between the drained prefill response and the decode dispatch. The family
    /// name predates the phase model and stays for compatibility; this is not
    /// the engines' KV transfer, which runs inside the decode phase.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_kv_transfer_duration(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_kv_transfer_duration_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record the PD decode phase: decode dispatch to the first decode token.
    /// Under sequential dispatch this is where the engines' KV transfer and the
    /// decode scheduling show up; under parallel dispatch both sides go out
    /// together and the value is the end-to-end TTFT.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_decode_ttft(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_decode_ttft_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record the PD time to first token: prefill dispatch to the first decode
    /// token.
    ///
    /// `smg_router_ttft_seconds{backend_type="pd"}` shares the prefill-dispatch
    /// anchor on the gRPC streaming paths (their stream base is the prefill
    /// dispatch). This family adds the runtime label and is the total that the
    /// phase families (`record_pd_prefill_duration`,
    /// `record_pd_kv_transfer_duration`, `record_pd_decode_ttft`) add up to
    /// under sequential dispatch.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_ttft(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_ttft_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record a KV connector mode decision (mooncake/nixl/moriio/store/passthrough).
    pub fn record_pd_kv_connector_mode(mode: &'static str) {
        counter!(
            "smg_pd_kv_connector_mode_total",
            "mode" => mode
        )
        .increment(1);
    }

    /// Record a PD bootstrap injection failure.
    pub fn record_pd_bootstrap_failure() {
        counter!("smg_pd_bootstrap_failures_total").increment(1);
    }

    /// Record a PD KV-transfer failure (missing connector params at handoff).
    pub fn record_pd_kv_transfer_failure() {
        counter!("smg_pd_kv_transfer_failures_total").increment(1);
    }

    /// Record a PD dispatch that had to wait for a decode admission slot.
    pub fn record_pd_admission_wait() {
        counter!("smg_pd_admission_waits_total").increment(1);
    }

    /// Record a PD dispatch shed because no decode slot freed in time.
    pub fn record_pd_admission_shed() {
        counter!("smg_pd_admission_sheds_total").increment(1);
    }

    /// Publish a decode worker's PD admission window: the running window the
    /// gate claims rooms against, 0 when the engine reports none and the gate
    /// abstains for that worker.
    pub fn set_pd_admission_window(worker_url: &str, window: Option<u16>) {
        let worker = intern_string(worker_url);
        gauge!("smg_pd_admission_window", "worker" => worker).set(window.map_or(0.0, f64::from));
    }

    /// Set the number of requests SMG has admitted to a prefill worker.
    pub fn set_pd_prefill_admission_inflight(worker_url: &str, count: usize) {
        let worker = intern_string(worker_url);
        gauge!(
            "smg_pd_prefill_admission_inflight",
            "worker" => worker
        )
        .set(count as f64);
    }

    /// Set the depth of the Router-wide Prefill admission queue.
    pub fn set_pd_prefill_admission_queued(depth: usize) {
        gauge!("smg_pd_prefill_admission_queued").set(depth as f64);
    }

    /// Record how long a request waited in the Prefill admission queue.
    pub fn record_pd_prefill_admission_wait(duration: Duration) {
        histogram!("smg_pd_prefill_admission_wait_seconds").record(duration.as_secs_f64());
    }

    /// Record a Prefill admission rejection (queue full or timeout).
    pub fn record_pd_prefill_admission_rejection(reason: &'static str) {
        counter!(
            "smg_pd_prefill_admission_rejections_total",
            "reason" => reason
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 3: Worker metrics
    // ========================================================================

    /// Record worker selection
    pub fn record_worker_selection(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        policy: &'static str,
    ) {
        let build = |model: Arc<str>| {
            counter!(
                "smg_worker_selection_total",
                "worker_type" => worker_type,
                "connection_mode" => connection_mode,
                "model" => model,
                "policy" => policy
            )
        };
        if !handles::keep_handles() {
            build(intern_model_label(model_id)).increment(1);
            return;
        }
        let labels = handles::SelectionLabels {
            worker_type,
            connection_mode,
            policy,
        };
        handles::WORKER_SELECTIONS.with_borrow_mut(|by_labels| {
            handles::with_model_handle(
                by_labels.entry(labels).or_default(),
                model_id,
                build,
                |count| count.increment(1),
            );
        });
    }

    /// Record worker error
    pub fn record_worker_error(
        worker_type: &'static str,
        connection_mode: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            "smg_worker_errors_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record a request shed because every worker for the model is overloaded.
    /// `stage` is "selection" or "dispatch".
    pub fn record_worker_overload_shed(stage: &'static str) {
        counter!(
            "smg_worker_overload_shed_total",
            "stage" => stage
        )
        .increment(1);
    }

    /// Record a request placed although every worker it could use is
    /// overloaded and shedding is off: steered to the least-loaded worker,
    /// or, for a cache-aware prefill pool of a prefill/decode pair or a
    /// regular request that consistent_hashing places by its routing key,
    /// placed by that policy (which counts the pick as a selection too).
    pub fn record_worker_overload_fallback(stage: &'static str) {
        counter!(
            "smg_worker_overload_fallback_total",
            "stage" => stage
        )
        .increment(1);
    }

    /// Record a request steered to the least-loaded ready worker because
    /// every worker for the model is vetoed and the ready ones are vetoed by
    /// the liveness tracker.
    pub fn record_worker_liveness_fallback(stage: &'static str) {
        counter!(
            "smg_worker_liveness_fallback_total",
            "stage" => stage
        )
        .increment(1);
    }

    /// Record manual policy execution branch for routing decisions
    pub fn record_worker_manual_policy_branch(branch: &'static str) {
        handles::with_static(
            &handles::MANUAL_BRANCHES,
            branch,
            || counter!("smg_manual_policy_branch_total", "branch" => branch),
            |count| count.increment(1),
        );
    }

    /// A key the manual policy's map did not hold was inserted: the map's
    /// size gauge moves up by one.
    pub fn manual_policy_cache_entry_added() {
        gauge!("smg_manual_policy_cache_entries").increment(1.0);
    }

    /// `count` entries left the manual policy's map (an eviction pass, or the
    /// policy dropped with its map): the size gauge moves down by that many.
    pub fn manual_policy_cache_entries_evicted(count: usize) {
        gauge!("smg_manual_policy_cache_entries").decrement(count as f64);
    }

    /// Record which source supplied the sticky routing key for a keyed request
    pub fn record_routing_key_source(source: &'static str) {
        handles::with_static(
            &handles::ROUTING_KEY_SOURCES,
            source,
            || counter!("smg_routing_key_source_total", "source" => source),
            |count| count.increment(1),
        );
    }

    /// Set cache-aware string-tree cached characters for a model
    pub fn set_cache_tree_chars(model_id: &str, chars: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_chars", "model" => model).set(chars as f64);
    }

    /// Set cache-aware token-tree cached tokens for a model
    pub fn set_cache_tree_tokens(model_id: &str, tokens: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_tokens", "model" => model).set(tokens as f64);
    }

    /// Set cache-aware tree tenant count for a model and tree kind ("string"/"token")
    pub fn set_cache_tree_tenants(model_id: &str, tree: &'static str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_tenants", "model" => model, "tree" => tree).set(count as f64);
    }

    /// Set cache-aware hash-index placement entry count for a model
    pub fn set_cache_placement_entries(model_id: &str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_placement_entries", "model" => model).set(count as f64);
    }

    /// Record cache-aware policy execution branch for tree-mode routing decisions
    pub fn record_worker_cache_aware_policy_branch(branch: &'static str) {
        handles::with_static(
            &handles::CACHE_AWARE_BRANCHES,
            branch,
            || counter!("smg_cache_aware_policy_branch_total", "branch" => branch),
            |count| count.increment(1),
        );
    }

    /// Record a cache-aware affinity pick that the spill gate's swing-sized
    /// margin alone would have spilled, kept on its holder by the held
    /// prefix's price
    pub fn record_cache_aware_prefix_margin_held() {
        counter!("smg_cache_aware_prefix_margin_held_total").increment(1);
    }

    /// Record bookings a policy released at the in-flight reconciliation
    /// tick: dispatches whose completion never reached it.
    pub fn record_policy_inflight_reconciled(policy: &'static str, released: usize) {
        counter!(
            "smg_policy_inflight_reconciled_total",
            "policy" => policy
        )
        .increment(released as u64);
    }

    /// Record the predicted overlap ratio (0..1) of a cache-aware routing
    /// decision: the tree's best prefix match in tree mode, the chosen
    /// worker's overlap over the request's blocks in event mode.
    pub fn record_cache_aware_match_ratio(ratio: f64) {
        handles::with_single(
            &handles::CACHE_AWARE_MATCH_RATIO,
            || histogram!("smg_cache_aware_match_ratio"),
            |ratios| ratios.record(ratio),
        );
    }

    /// Record consistent hashing policy execution branch for routing decisions
    pub fn record_worker_consistent_hashing_policy_branch(branch: &'static str) {
        handles::with_static(
            &handles::CONSISTENT_HASHING_BRANCHES,
            branch,
            || counter!("smg_consistent_hashing_policy_branch_total", "branch" => branch),
            |count| count.increment(1),
        );
    }

    /// Record prefix hash policy execution branch for routing decisions
    pub fn record_worker_prefix_hash_policy_branch(branch: &'static str) {
        handles::with_static(
            &handles::PREFIX_HASH_BRANCHES,
            branch,
            || counter!("smg_prefix_hash_policy_branch_total", "branch" => branch),
            |count| count.increment(1),
        );
    }

    /// Time of one index lookup on the routing path, recorded after the
    /// lookup returned and outside any lock.
    pub fn record_kv_index_lookup(index: &'static str, seconds: f64) {
        handles::with_static(
            &handles::KV_INDEX_LOOKUPS,
            index,
            || histogram!("smg_kv_index_lookup_seconds", "index" => index),
            |lookups| lookups.record(seconds),
        );
    }

    /// Publish the lines a log writer's queue has dropped so far, by sink.
    pub fn set_log_dropped_lines(sink: &'static str, dropped: usize) {
        gauge!("smg_log_dropped_lines", "sink" => sink).set(dropped as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (retry)
    // ========================================================================

    // ========================================================================
    // Layer 5: MCP metrics
    // ========================================================================

    /// Set active MCP servers count
    pub fn set_mcp_servers_active(count: usize) {
        gauge!("smg_mcp_servers_active").set(count as f64);
    }
}

/// Metrics helpers for tests elsewhere in the crate.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::OnceLock;

    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

    /// The lib test binary's one process-wide recorder, for records made on
    /// threads no local recorder covers (the blocking pool). Installed once;
    /// every other test keeps its own local recorder, which takes precedence
    /// on its thread.
    pub(crate) fn global_recorder() -> &'static PrometheusHandle {
        static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
        HANDLE.get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("the lib test binary's one global recorder")
        })
    }

    /// The scrape a test renders under its own recorder, with the two
    /// readers of it: the worker's test support owns them.
    pub(crate) use smg_worker::test_support::assert_metric;
    pub(crate) use smg_worker::test_support::{render_with_recorder, sorted_series};
}

#[cfg(test)]
mod tests;
