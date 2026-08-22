use std::collections::HashMap;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};

// Jemalloc as the global allocator: glibc malloc retains freed pages badly
// under the gateway's allocation churn. Prefixed symbols only — vendored C
// libraries keep their own malloc.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use openai_protocol::worker::TransportMode;
use rand::{distr::Alphanumeric, RngExt};
use smg::{
    config::{
        validate_mesh_server_name, CacheIndexKind, CircuitBreakerConfig, ConfigError, ConfigResult,
        DiscoveryConfig, HealthCheckConfig, HistoryBackend, ManualAssignmentMode, MetricsConfig,
        OracleConfig, PolicyConfig, PostgresConfig, RedisConfig, RetryConfig, RouterConfig,
        RoutingKeyOverrideConfig, RoutingMode, SchemaConfig, TenantApiKeyEntry,
        TokenizerCacheConfig, TraceConfig,
    },
    observability::{
        metrics::{register_jemalloc_as_global_allocator, PrometheusConfig},
        otel_trace::{is_otel_enabled, shutdown_otel},
    },
    server::{self, ServerConfig},
    service_discovery::{ModelIdSource, ServiceDiscoveryConfig},
    version,
    worker::{ConnectionMode, RuntimeType},
};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role};
use smg_mesh::MeshServerConfig;
use tracing::info;

/// Parse repeated `<flag> <url> [bootstrap_port|none]` occurrences into
/// (url, optional bootstrap port) pairs. The trailing port is optional, so
/// these flags are hand-parsed (clap cannot express the optional positional)
/// and stripped from argv before `Cli::parse_from`.
fn parse_url_port_args(flag: &str) -> Vec<(String, Option<u16>)> {
    let args: Vec<String> = std::env::args().collect();
    let mut entries = Vec::new();
    let mut i = 0;

    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let bootstrap_port = if i + 2 < args.len() && !args[i + 2].starts_with("--") {
                if let Ok(port) = args[i + 2].parse::<u16>() {
                    i += 1;
                    Some(port)
                } else if args[i + 2].to_lowercase() == "none" {
                    i += 1;
                    None
                } else {
                    None
                }
            } else {
                None
            };
            entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }

    entries
}

fn parse_prefill_args() -> Vec<(String, Option<u16>)> {
    parse_url_port_args("--prefill")
}

fn parse_encode_args() -> Vec<(String, Option<u16>)> {
    parse_url_port_args("--encode")
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    #[value(name = "sglang")]
    Sglang,
    #[value(name = "vllm")]
    Vllm,
    #[value(name = "trtllm")]
    Trtllm,
    #[value(name = "tokenspeed")]
    Tokenspeed,
    #[value(name = "openai")]
    Openai,
    #[value(name = "anthropic")]
    Anthropic,
    #[value(name = "gemini")]
    Gemini,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Backend::Sglang => "sglang",
            Backend::Vllm => "vllm",
            Backend::Trtllm => "trtllm",
            Backend::Tokenspeed => "tokenspeed",
            Backend::Openai => "openai",
            Backend::Anthropic => "anthropic",
            Backend::Gemini => "gemini",
        };
        write!(f, "{s}")
    }
}

#[derive(Parser, Debug)]
#[command(name = "shepherd-model-gateway", alias = "smg", alias = "amg")]
#[command(about = "Shepherd Model Gateway - High-performance inference gateway")]
#[command(args_conflicts_with_subcommands = true)]
#[command(long_about = r#"
Shepherd Model Gateway - Rust-based inference gateway

Usage:
  smg launch [OPTIONS]             Launch gateway (short command)
  amg launch [OPTIONS]             Launch gateway (alternative)
  shepherd-model-gateway launch [OPTIONS] Launch gateway (full name)

Examples:
  # Regular mode
  smg launch --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 9002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --policy cache_aware

  # With different policies
  smg launch --pd-disaggregation \
    --prefill http://127.0.0.1:30001 9001 \
    --prefill http://127.0.0.2:30002 \
    --decode http://127.0.0.3:30003 \
    --decode http://127.0.0.4:30004 \
    --prefill-policy cache_aware --decode-policy power_of_two

"#)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    router_args: CliArgs,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the router (same as running without subcommand)
    #[command(visible_alias = "start")]
    Launch {
        #[command(flatten)]
        args: CliArgs,
    },
}

/// Parse the `--multimodal-tensor-transport` value into a `TransportMode`.
fn parse_transport_mode(value: &str) -> Result<TransportMode, String> {
    TransportMode::parse(value)
        .ok_or_else(|| format!("invalid value '{value}'; expected inline, shm, auto, or rdma"))
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        Ok(_) => Err(format!("invalid value '{value}'; expected a count >= 1")),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

fn parse_usize_in_range(value: &str, max: usize) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if (1..=max).contains(&n) => Ok(n),
        Ok(n) => Err(format!("invalid value '{n}'; expected 1..={max}")),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

/// Parse a ratio in `(0.0, 1.0]`. Zero is excluded because a `>=` threshold of
/// 0.0 would mark every worker overloaded unconditionally.
fn parse_unit_fraction(value: &str) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(v) if v > 0.0 && v <= 1.0 => Ok(v),
        Ok(_) => Err(format!(
            "invalid value '{value}'; expected a fraction in (0.0, 1.0]"
        )),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

fn parse_job_queue_capacity(value: &str) -> Result<usize, String> {
    parse_usize_in_range(value, 1_000_000)
}

fn parse_job_queue_concurrency(value: &str) -> Result<usize, String> {
    parse_usize_in_range(value, 100_000)
}

#[derive(Parser, Debug)]
struct CliArgs {
    // ==================== Worker Configuration ====================
    /// Host address to bind the router server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Worker Configuration")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000, help_heading = "Worker Configuration")]
    port: u16,

    /// Dedicated port for liveness/readiness/health probes (Kubernetes,
    /// load balancers, uptime monitors, etc.).
    ///
    /// When set, `/liveness`, `/readiness`, and `/health` are additionally
    /// served on this port by a middleware-free router running on its own
    /// single-worker runtime and OS thread, isolated from the request
    /// runtime so a saturated gateway cannot starve probes (and trigger the
    /// failed-probe restarts or depooling that follow) under load. The same
    /// probe routes always remain available on the main `--port` too.
    /// Unset = dedicated probe listener off.
    #[arg(long, help_heading = "Worker Configuration")]
    health_check_port: Option<u16>,

    /// List of worker URLs (supports IPv4 and IPv6)
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    worker_urls: Vec<String>,

    // ==================== Routing Policy ====================
    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "passthrough", "cache_aware", "cache_aware_length", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "Routing Policy")]
    policy: String,

    /// Minimum matched-prefix share (0.0-1.0) before cache-aware routing
    /// pins a request to a worker already holding that prefix; below it the
    /// request is load-balanced instead
    #[arg(
        long,
        visible_alias = "cache-match-threshold",
        default_value_t = 0.3,
        help_heading = "Routing Policy"
    )]
    cache_threshold: f32,

    /// Spill gate, absolute part: a matched worker is skipped for the
    /// least-loaded one when its load exceeds the healthy-fleet mean by this
    /// many requests AND by --balance-rel-threshold
    #[arg(
        long,
        visible_alias = "spill-abs-threshold",
        default_value_t = 64,
        help_heading = "Routing Policy"
    )]
    balance_abs_threshold: usize,

    /// Spill gate, relative part (multiple of the healthy-fleet mean); fires
    /// only together with --balance-abs-threshold
    #[arg(
        long,
        visible_alias = "spill-rel-threshold",
        default_value_t = 1.5,
        help_heading = "Routing Policy"
    )]
    balance_rel_threshold: f32,

    /// Abandon cache affinity for shortest-queue when the KV-usage spread
    /// (hottest minus coldest backend, 0.0-1.0) exceeds this — catches
    /// long-context KV imbalance that request counts miss. Backend must
    /// report token_usage. >= 1.0 disables it.
    #[arg(long, default_value_t = 1.0, help_heading = "Routing Policy")]
    balance_token_usage_threshold: f32,

    /// Safety valve for critically-saturated engines: when the hottest
    /// backend's KV utilization (0.0-1.0) exceeds this, shed load off it
    /// regardless of spread. Best set high (e.g. 0.9). >= 1.0 disables it.
    #[arg(long, default_value_t = 1.0, help_heading = "Routing Policy")]
    overload_token_usage_threshold: f32,

    /// Enable worker overload protection with the gateway default thresholds.
    ///
    /// A worker whose load signal crosses a threshold is considered overloaded
    /// and excluded from routing until the signal recovers; when every worker
    /// is overloaded, requests are shed immediately rather than queued.
    ///
    /// This flag alone applies --worker-overload-token-usage 0.9 and leaves
    /// --worker-overload-waiting-requests unset: KV token usage means the same
    /// thing on every engine, while a sensible waiting-requests ceiling is
    /// workload-dependent, so it has no universal default. Explicit thresholds
    /// override the default, and either threshold set on its own enables
    /// protection without this flag — exactly as before it existed. Per-worker
    /// `overload` blocks on a WorkerSpec override the gateway values per
    /// signal, and enable protection for that worker even with everything here
    /// unset.
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    worker_overload_protection: bool,

    /// Queued-request count at or above which a worker is considered
    /// overloaded and excluded from routing until the signal recovers; when
    /// every worker is overloaded, requests are shed immediately rather than
    /// queued. Unset disables overload protection.
    ///
    /// Queued (waiting) requests, summed across DP ranks. Must be >= 1: the
    /// comparison is inclusive, so 0 would veto every worker unconditionally.
    #[arg(long, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    worker_overload_waiting_requests: Option<usize>,

    /// KV-cache token usage at or above which a worker is considered
    /// overloaded and excluded from routing until the signal recovers; when
    /// every worker is overloaded, requests are shed immediately rather than
    /// queued. Unset disables overload protection.
    ///
    /// Mean KV-cache token usage across DP ranks, the same signal
    /// `--balance-token-usage-threshold` reads, applied as an absolute
    /// per-worker ceiling rather than a fleet-relative spread. Backend must
    /// report token_usage. Must be in (0.0, 1.0]: the comparison is inclusive,
    /// so 0.0 would veto every worker unconditionally.
    ///
    /// Distinct from `--overload-token-usage-threshold`, which only de-ranks
    /// the hottest backend within cache-aware affinity; this flag removes the
    /// worker from routing entirely and sheds when every worker crosses it.
    #[arg(long, value_parser = parse_unit_fraction, help_heading = "Routing Policy")]
    worker_overload_token_usage: Option<f64>,

    /// Anti-hotspot decay: de-rank cache-affine candidates by their
    /// waiting-prefill backlog (overlap score divided by 1 + overlap_decay
    /// * backlog blocks per request block). Requires backend load
    /// reporting. 0.0 disables.
    #[arg(long, default_value_t = 0.0, help_heading = "Routing Policy")]
    overlap_decay: f32,

    /// Spread event-driven cache-aware picks across near-equal candidates:
    /// softmax temperature over min-max normalized scores. 0.0 is exact
    /// argmax.
    #[arg(long, default_value_t = 0.0, help_heading = "Routing Policy")]
    selection_temperature: f32,

    /// Interval in seconds between cache-tree eviction cycles
    #[arg(long, default_value_t = 120, help_heading = "Routing Policy")]
    eviction_interval: u64,

    /// Maximum total size of each model's approximation tree for cache-aware
    /// routing (chars for HTTP, tokens for gRPC), shared across all workers;
    /// eviction keeps every tree at or under this bound
    #[arg(long, default_value_t = 67108864, help_heading = "Routing Policy")]
    max_tree_size: usize,

    /// Match granularity for cache-aware token routing: the token-tree page
    /// size, and the KV block size assumed for event-driven selection
    #[arg(long, default_value_t = 16, help_heading = "Routing Policy")]
    block_size: usize,

    /// Token positions at which serving engines retain reusable prefix
    /// state; cache-affinity policies hash request heads at the deepest
    /// applicable boundary.
    #[arg(long, value_delimiter = ',', help_heading = "Routing Policy")]
    cache_boundaries: Vec<usize>,

    /// Index under-layer for cache_aware: "tree" (radix prefix trees) or
    /// "hash" (TTL'd exact-match placement map keyed on request heads at
    /// --cache-boundaries; token-bearing requests only — untokenized
    /// requests stay load-balanced)
    #[arg(long, default_value = "tree", value_parser = ["tree", "hash"], help_heading = "Routing Policy")]
    cache_index: String,

    /// Seconds a cache-affinity placement stays routable; should
    /// approximate serving-engine cache retention
    #[arg(long, default_value_t = 180, value_parser = clap::value_parser!(u64).range(1..), help_heading = "Routing Policy")]
    cache_ttl_secs: u64,

    // ---- cache_aware_length policy ----
    /// Divisor for char-level token estimation when X-Prompt-Tokens is absent
    /// (cache_aware_length policy). Default 4.
    #[arg(long, default_value_t = 4, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    chars_per_token: usize,

    /// Uncached-prefill-token boundary between long and short requests
    /// (cache_aware_length policy). Default 100000.
    #[arg(long, default_value_t = 100_000, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    long_prefill_threshold: usize,

    /// Load ceiling for the long pool (pool=long workers) in the
    /// cache_aware_length policy. Default 4.
    #[arg(long, default_value_t = 4, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    long_pool_max_load: usize,

    /// Load ceiling for the short pool (remaining workers) in the
    /// cache_aware_length policy. Default 32.
    #[arg(long, default_value_t = 32, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    short_pool_max_load: usize,

    /// Comma-separated 0-based indices of --prefill URLs that belong to the
    /// long pool (get pool=long label for cache_aware_length). E.g. "3,4"
    /// marks the 4th and 5th prefill workers as long pool.
    #[arg(long, value_delimiter = ',', help_heading = "PD Disaggregation")]
    long_prefill_indices: Vec<usize>,

    /// How long an unused sticky routing key stays pinned: keys idle beyond
    /// this many seconds are evicted from the manual-policy / sticky-session
    /// map
    #[arg(
        long,
        visible_alias = "sticky-key-idle-secs",
        default_value_t = 14400,
        help_heading = "Routing Policy"
    )]
    max_idle_secs: u64,

    /// How a first-seen routing key picks its worker: random, min_load
    /// (fewest requests), min_group (fewest keys), or delegate (route via
    /// the underlying policy, then pin). Defaults to random for the manual
    /// policy and delegate for the sticky-session map
    #[arg(long, value_parser = ["random", "min_load", "min_group", "delegate"], help_heading = "Routing Policy")]
    assignment_mode: Option<String>,

    /// Number of prefix tokens to use for prefix_hash policy, or four times
    /// as many characters of the prompt when the request is untokenized
    #[arg(long, default_value_t = 256, help_heading = "Routing Policy")]
    prefix_token_count: usize,

    /// Load factor threshold for prefix_hash policy
    #[arg(long, default_value_t = 1.25, help_heading = "Routing Policy")]
    prefix_hash_load_factor: f64,

    /// Absolute load difference over average a worker must also exceed before
    /// the prefix_hash policy treats it as overloaded
    #[arg(long, default_value_t = 10, help_heading = "Routing Policy")]
    prefix_hash_balance_abs_threshold: usize,

    /// KV-pressure weight (seconds) for the least_load policy
    #[arg(long, default_value_t = 0.15, help_heading = "Routing Policy")]
    least_load_kv_pressure_weight: f64,

    /// Fallback generation throughput (tokens/s) for least_load when a backend
    /// reports no live throughput
    #[arg(long, default_value_t = 2000.0, help_heading = "Routing Policy")]
    least_load_default_throughput: f64,

    /// Mean prefill tokens for least_load's in-flight estimate when a request's
    /// token count is unknown at routing
    #[arg(long, default_value_t = 1024, help_heading = "Routing Policy")]
    least_load_mean_prefill_tokens: u32,

    /// Per-worker waiting-queue cap for least_load: skip workers whose reported
    /// waiting requests (plus dispatches since their last poll) have reached
    /// this count; 0 disables. Set below the engine's max batch size
    #[arg(long, default_value_t = 0, help_heading = "Routing Policy")]
    least_load_max_waiting_requests: u32,

    /// Enable data parallelism aware scheduling
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_aware: bool,

    /// Sticky sessions: route every request of a conversation to the same
    /// worker, on any policy. The key is derived from the request body's rid
    /// with per-turn/per-retry suffixes stripped (conv_t2_r1 -> conv),
    /// falling back to the routing-key headers when no rid is present;
    /// raw-streamed requests carry no readable rid and use the headers only.
    /// Reuses the manual eviction/idle/assignment knobs for the sticky map
    #[arg(
        long,
        visible_alias = "sticky-sessions",
        default_value_t = false,
        help_heading = "Routing Policy"
    )]
    routing_key_override: bool,

    /// Ordered header names checked for the routing key; the first header
    /// present with a valid value wins. Header keys get the same
    /// per-turn/per-retry suffix stripping as rid-derived keys when the
    /// override is enabled
    #[arg(long, num_args = 0.., default_value = "x-smg-routing-key", value_parser = parse_routing_key_header, help_heading = "Routing Policy")]
    routing_key_headers: Vec<String>,

    /// Enable IGW (Inference Gateway) mode for multi-model support
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    enable_igw: bool,

    /// Enable minimum tokens scheduler for data parallel group
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_minimum_tokens_scheduler: bool,

    // ==================== PD Disaggregation ====================
    /// Enable PD (Prefill-Decode) disaggregated mode
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    pd_disaggregation: bool,

    /// Enable EPD (Encode-Prefill-Decode) disaggregated mode (gRPC + TokenSpeed only).
    /// Encode workers run the vision tower and ship embeddings to prefill over Mooncake;
    /// prefill/decode reuse the PD path. Encode urls are given via `--encode <url> [bootstrap_port]`.
    #[arg(long, default_value_t = false, help_heading = "PD Disaggregation")]
    epd_disaggregation: bool,

    /// Decode server URLs (can be specified multiple times)
    #[arg(long, action = ArgAction::Append, help_heading = "PD Disaggregation")]
    decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "cache_aware_length", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "PD Disaggregation")]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "cache_aware_length", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "PD Disaggregation")]
    decode_policy: Option<String>,

    /// Specific policy for encode nodes in EPD mode. Defaults to consistent_hashing.
    #[arg(long, value_parser = ["random", "round_robin", "consistent_hashing"], help_heading = "PD Disaggregation")]
    encode_policy: Option<String>,

    /// Timeout in seconds for worker startup and registration
    #[arg(long, default_value_t = 1800, help_heading = "PD Disaggregation")]
    worker_startup_timeout_secs: u64,

    /// Grace period in seconds before the first worker startup check
    #[arg(long, default_value_t = 0, help_heading = "PD Disaggregation")]
    worker_startup_delay: u64,

    /// Interval in seconds between worker startup checks
    #[arg(long, default_value_t = 30, help_heading = "PD Disaggregation")]
    worker_startup_check_interval: u64,

    /// Max pending control-plane jobs (worker add/remove, tokenizer, MCP,
    /// WASM). Size to fleet scale so a service-discovery reconcile pass can
    /// enqueue every worker without blocking.
    #[arg(long, default_value_t = 1000, value_parser = parse_job_queue_capacity, help_heading = "Worker Configuration")]
    job_queue_capacity: usize,

    /// Max control-plane jobs dispatched concurrently
    #[arg(long, default_value_t = 200, value_parser = parse_job_queue_concurrency, help_heading = "Worker Configuration")]
    job_queue_concurrency: usize,

    /// DP engines per startup ZMQ worker: each ipc:// worker becomes a grouped
    /// worker whose handshake awaits this many engines on one socket set.
    #[arg(long, value_parser = parse_positive_usize, help_heading = "Worker Configuration")]
    zmq_engine_count: Option<usize>,

    /// Speak HTTP/2 to workers via prior knowledge (h2c on cleartext) on all
    /// engine-directed connections — request dispatch and health/probe traffic
    /// alike — multiplexing every request to a worker over one connection.
    /// Requires every HTTP worker to serve HTTP/2 without an upgrade handshake.
    #[arg(long, default_value_t = false, help_heading = "Worker Configuration")]
    upstream_http2: bool,

    /// Interval in seconds between load monitor checks for PowerOfTwo routing
    #[arg(long, default_value_t = 10, help_heading = "Load Monitoring")]
    load_monitor_interval: u64,

    /// Only poll worker loads when a load-aware routing policy,
    /// --engine-metrics, or worker overload protection needs the data. By
    /// default every worker group is polled from registration onward; this
    /// restores the old conditional gate (a load-aware policy is always fed
    /// regardless).
    #[arg(long, default_value_t = false, help_heading = "Load Monitoring")]
    disable_load_monitoring: bool,

    /// Re-export engine GetLoads signals (incl. PD) as smg_engine_* Prometheus
    /// gauges, polling even without a load-aware routing policy.
    #[arg(long, default_value_t = false, help_heading = "Load Monitoring")]
    engine_metrics: bool,

    /// TTL in seconds for event-driven cache-aware indexer entries: entries
    /// neither stored nor read by a query within this window are pruned.
    /// Bounds index growth when a backend stops emitting removal events.
    /// Unset or 0 disables the TTL pass.
    #[arg(long, help_heading = "Routing Policy")]
    kv_indexer_ttl_secs: Option<u64>,

    /// Capacity ceiling per model for the event-driven cache-aware indexer;
    /// beyond it, oldest-touched entries are pruned down to 90% of the
    /// ceiling. Unset or 0 disables the ceiling.
    #[arg(long, help_heading = "Routing Policy")]
    kv_indexer_max_entries: Option<usize>,

    /// Multimodal tensor transport mode: `inline` (default), `shm` (same-host
    /// /dev/shm), or `auto` (shm only when the worker shares /dev/shm). A
    /// per-worker `WorkerSpec.multimodal_tensor_transport` overrides this.
    #[arg(long, value_parser = parse_transport_mode, help_heading = "Multimodal")]
    multimodal_tensor_transport: Option<TransportMode>,

    /// Minimum multimodal tensor size (bytes) before the SHM transport is used.
    /// Overridable per worker via `WorkerSpec.multimodal_shm_min_bytes`.
    #[arg(long, help_heading = "Multimodal")]
    multimodal_shm_min_bytes: Option<usize>,

    // ==================== Service Discovery (Kubernetes) ====================
    /// Enable Kubernetes service discovery
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key=value)
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    selector: Vec<String>,

    /// Port to use for discovered worker pods without a `smg.ai/worker-ports`
    /// annotation (pods running multiple servers list their ports there)
    #[arg(
        long,
        default_value_t = 80,
        help_heading = "Service Discovery (Kubernetes)"
    )]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long, help_heading = "Service Discovery (Kubernetes)")]
    service_discovery_namespace: Option<String>,

    /// Label selector for encode server pods in EPD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    encode_selector: Vec<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    decode_selector: Vec<String>,

    /// Label selector for router pod discovery in HA mesh mode (format: key=value)
    #[arg(long, num_args = 0.., help_heading = "Service Discovery (Kubernetes)")]
    router_selector: Vec<String>,

    /// Override each worker's model_id from pod metadata.
    /// Accepted values: "namespace", "label:<key>", or "annotation:<key>"
    #[arg(long, help_heading = "Service Discovery (Kubernetes)", value_parser = parse_model_id_from)]
    model_id_from: Option<String>,

    /// Accept an extra client-facing model name for a served model
    /// (format: alias=canonical, repeatable). Applied to every locally
    /// registered worker whose model ID equals the canonical side,
    /// including workers registered by Kubernetes service discovery.
    /// Matching is case-sensitive.
    #[arg(long = "model-alias", action = ArgAction::Append, value_parser = parse_model_alias, help_heading = "Service Discovery (Kubernetes)")]
    model_alias: Vec<String>,

    // ==================== Logging ====================
    /// Directory to store log files
    #[arg(long, help_heading = "Logging")]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Output logs as JSON
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    log_json: bool,

    // ==================== Prometheus Metrics ====================
    /// Port to expose Prometheus metrics; 0 binds an OS-assigned ephemeral port
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "0.0.0.0", help_heading = "Prometheus Metrics")]
    prometheus_host: String,

    /// Custom buckets for Prometheus duration metrics
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    prometheus_duration_buckets: Vec<f64>,

    // ==================== Request Handling ====================
    /// Custom HTTP headers to check for request IDs
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    request_id_headers: Vec<String>,

    /// Map HTTP headers into storage hook request context (format: header=context_key)
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    storage_context_headers: Vec<String>,

    /// Trust an upstream-provided tenant header for canonical tenant resolution.
    #[arg(long, default_value_t = false, help_heading = "Request Handling")]
    trust_tenant_header: bool,

    /// Header name to use when --trust-tenant-header is enabled.
    #[arg(
        long,
        default_value = "x-smg-tenant-id",
        help_heading = "Request Handling"
    )]
    tenant_header_name: String,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 1800, help_heading = "Request Handling")]
    request_timeout_secs: u64,

    /// Idle timeout in seconds for pooled upstream connections. Must stay
    /// below the backend HTTP server's keep-alive timeout (vLLM and SGLang
    /// default to 5), or reused connections the server already closed fail
    /// non-idempotent sends. 0 keeps idle connections forever.
    #[arg(long, default_value_t = 3, help_heading = "Request Handling")]
    upstream_pool_idle_timeout_secs: u64,

    /// Grace period in seconds to wait for in-flight requests during shutdown
    #[arg(long, default_value_t = 180, help_heading = "Request Handling")]
    shutdown_grace_period_secs: u64,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 536870912, help_heading = "Request Handling")]
    max_payload_size: usize,

    /// Forward request bodies larger than this many bytes to the worker as a
    /// raw stream instead of buffering, when the route's policy needs no
    /// request text — or the request carries a valid x-smg-routing-tokens
    /// hint, or a valid x-smg-routing-key under --routing-key-override — and
    /// the worker applies no body mutation. Streamed bodies are forwarded
    /// verbatim, so JSON validation and normalization defer to the worker,
    /// and they cannot be replayed, so those requests bypass router-level
    /// retries; bodies without a Content-Length header always buffer. WASM
    /// OnRequest modules inspect buffered bodies, so deployments running
    /// them keep buffering (and their own body-size cap) ahead of this.
    /// Must be below max-payload-size. 0 disables
    #[arg(long, default_value_t = 0, help_heading = "Request Handling")]
    stream_request_bodies_over: u64,

    /// Abort a streamed request body once the upstream sender has waited on
    /// the client for this many seconds (408, request_body_stalled). The
    /// clock pauses while the worker applies backpressure, so a slow worker
    /// read never trips it. Applies only to bodies streamed via
    /// --stream-request-bodies-over. 0 disables
    #[arg(long, default_value_t = 300, help_heading = "Request Handling")]
    stream_body_stall_timeout_secs: u64,

    /// CORS allowed origins
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    cors_allowed_origins: Vec<String>,

    // ==================== Rate Limiting ====================
    /// Maximum standing concurrent requests (-1 to disable). Each admission
    /// permit is held for the full response, including streaming bodies.
    #[arg(long, default_value_t = -1, help_heading = "Rate Limiting")]
    max_concurrent_requests: i32,

    /// Queue size for pending requests when limit reached
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    queue_size: usize,

    /// Maximum time in seconds a request can wait in queue
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    queue_timeout_secs: u64,

    // ==================== Priority Scheduler ====================
    /// Enable the priority-aware admission scheduler. When unset (default),
    /// the legacy concurrency-limit middleware stays wired.
    #[arg(long, help_heading = "Priority Scheduler")]
    priority_scheduler_enabled: bool,

    /// Max priority class for tenants not listed in the scheduler YAML
    /// (system | interactive | default | bulk).
    #[arg(long, default_value = "default", help_heading = "Priority Scheduler")]
    priority_scheduler_default_max_class: String,

    /// Optional path to the priority-scheduler YAML config.
    #[arg(long, help_heading = "Priority Scheduler")]
    priority_scheduler_config: Option<String>,

    /// Cap on per-tenant scheduler metric label cardinality (top-N + "other").
    #[arg(long, default_value_t = 32, help_heading = "Priority Scheduler")]
    priority_scheduler_tenant_metric_top_n: u32,

    // ==================== Tenant Rate Limit ====================
    /// Enable per-tenant LLM token/request rate limiting. When unset
    /// (default), no rate limiter is constructed.
    #[arg(long, help_heading = "Tenant Rate Limit")]
    tenant_rate_limit_enabled: bool,

    /// Path to the tenant-rate-limit YAML. Required when
    /// `--tenant-rate-limit-enabled` is set.
    #[arg(long, help_heading = "Tenant Rate Limit")]
    tenant_rate_limit_config: Option<String>,

    /// Token bucket refill rate (tokens per second). Unset or 0 = no refill:
    /// --max-concurrent-requests bounds standing concurrency alone.
    #[arg(long, help_heading = "Rate Limiting")]
    rate_limit_tokens_per_second: Option<i32>,

    // ==================== Retry Configuration ====================
    /// Maximum number of retry attempts
    #[arg(long, default_value_t = 5, help_heading = "Retry Configuration")]
    retry_max_retries: u32,

    /// Initial backoff delay in milliseconds
    #[arg(long, default_value_t = 50, help_heading = "Retry Configuration")]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds
    #[arg(long, default_value_t = 30000, help_heading = "Retry Configuration")]
    retry_max_backoff_ms: u64,

    /// Multiplier for exponential backoff
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    retry_backoff_multiplier: f32,

    /// Jitter factor (0.0-1.0) for retry delays
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    retry_jitter_factor: f32,

    /// Disable automatic retries
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    disable_retries: bool,

    // ==================== Circuit Breaker ====================
    /// Number of failures before circuit opens
    #[arg(long, default_value_t = 10, help_heading = "Circuit Breaker")]
    cb_failure_threshold: u32,

    /// Successes needed in half-open state to close
    #[arg(long, default_value_t = 3, help_heading = "Circuit Breaker")]
    cb_success_threshold: u32,

    /// Seconds before attempting to close open circuit
    #[arg(long, default_value_t = 60, help_heading = "Circuit Breaker")]
    cb_timeout_duration_secs: u64,

    /// Sliding window duration for tracking failures
    #[arg(long, default_value_t = 120, help_heading = "Circuit Breaker")]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false, help_heading = "Circuit Breaker")]
    disable_circuit_breaker: bool,

    // ==================== Health Checks ====================
    /// Consecutive probe failures before a worker is taken out of rotation
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    health_failure_threshold: u32,

    /// Consecutive probe successes before a worker returns to rotation
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    health_success_threshold: u32,

    /// Timeout in seconds for a single health probe
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    health_check_timeout_secs: u64,

    /// Interval in seconds between health probes of each worker
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    health_check_interval_secs: u64,

    /// HTTP path probed on each worker
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    health_check_endpoint: String,

    /// Disable all worker health probing
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    disable_health_check: bool,

    /// Let workers recover after prolonged failure: a worker that stays
    /// unhealthy long enough is removed from the registry so service
    /// discovery re-registers and re-probes it once its engine returns
    /// (without this, a worker unreachable for ~12 minutes reaches a
    /// terminal Failed state and is never probed again)
    #[arg(
        long,
        visible_alias = "worker-auto-recovery",
        default_value_t = false,
        help_heading = "Health Checks"
    )]
    remove_unhealthy_workers: bool,

    /// Seconds to keep a Ready worker in `Draining` before removing it from
    /// the registry. Applies to all RemoveWorker submissions (K8s deletion,
    /// `--remove-unhealthy-workers`, manual API). Per-worker overrides are
    /// supported via `WorkerSpec::health.drain_settle_secs`. Set to `0` to
    /// remove immediately without draining.
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    drain_settle_secs: u64,

    // ==================== Tokenizer ====================
    /// Model path for loading tokenizer (HuggingFace ID or local path)
    #[arg(long, alias = "model", help_heading = "Tokenizer")]
    model_path: Option<String>,

    /// Explicit tokenizer path (overrides model_path)
    #[arg(long, help_heading = "Tokenizer")]
    tokenizer_path: Option<String>,

    /// Chat template path
    #[arg(long, help_heading = "Tokenizer")]
    chat_template: Option<String>,

    /// Disable automatic tokenizer loading at startup and worker registration
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    disable_tokenizer_autoload: bool,

    /// Enable L0 (exact match) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l0: bool,

    /// Maximum entries in L0 tokenizer cache
    #[arg(long, default_value_t = 10000, help_heading = "Tokenizer")]
    tokenizer_cache_l0_max_entries: usize,

    /// Enable L1 (prefix matching) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l1: bool,

    /// Maximum memory for L1 tokenizer cache in bytes
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    tokenizer_cache_l1_max_memory: usize,

    // ==================== Parsers ====================
    /// Parser for reasoning models (e.g., deepseek-r1, qwen3)
    #[arg(long, help_heading = "Parsers")]
    reasoning_parser: Option<String>,

    /// Parser for tool-call interactions
    #[arg(long, help_heading = "Parsers")]
    tool_call_parser: Option<String>,

    /// Path to MCP server configuration file
    #[arg(long, help_heading = "Parsers")]
    mcp_config_path: Option<String>,

    // ==================== Backend ====================
    /// Backend runtime to use (auto-detected if not specified)
    #[arg(long, value_enum, alias = "runtime", help_heading = "Backend")]
    backend: Option<Backend>,

    /// History storage backend
    #[arg(long, default_value = "memory", value_parser = ["memory", "none", "oracle", "postgres", "redis"], help_heading = "Backend")]
    history_backend: String,

    /// Enable WebAssembly support
    #[arg(long, default_value_t = false, help_heading = "Backend")]
    enable_wasm: bool,

    /// Path to a WASM component implementing storage hooks
    #[arg(long, help_heading = "Backend")]
    storage_hook_wasm_path: Option<String>,

    /// Path to a YAML schema config file for storage table/column remapping
    #[arg(long, help_heading = "Backend")]
    schema_config: Option<String>,

    // ==================== Oracle Database ====================
    /// Path to Oracle ATP wallet directory
    #[arg(long, env = "ATP_WALLET_PATH", help_heading = "Oracle Database")]
    oracle_wallet_path: Option<String>,

    /// Oracle TNS alias from tnsnames.ora
    #[arg(long, env = "ATP_TNS_ALIAS", help_heading = "Oracle Database")]
    oracle_tns_alias: Option<String>,

    /// Oracle connection descriptor/DSN
    #[arg(long, env = "ATP_DSN", help_heading = "Oracle Database")]
    oracle_dsn: Option<String>,

    /// Oracle database username
    #[arg(long, env = "ATP_USER", help_heading = "Oracle Database")]
    oracle_user: Option<String>,

    /// Oracle database password
    #[arg(long, env = "ATP_PASSWORD", help_heading = "Oracle Database")]
    oracle_password: Option<String>,

    /// Enable Oracle external authentication
    #[arg(
        long,
        env = "ATP_EXTERNAL_AUTH",
        default_value_t = false,
        help_heading = "Oracle Database"
    )]
    oracle_external_auth: bool,

    /// Minimum Oracle connection pool size
    #[arg(long, env = "ATP_POOL_MIN", help_heading = "Oracle Database")]
    oracle_pool_min: Option<usize>,

    /// Maximum Oracle connection pool size
    #[arg(long, env = "ATP_POOL_MAX", help_heading = "Oracle Database")]
    oracle_pool_max: Option<usize>,

    /// Oracle connection pool timeout in seconds
    #[arg(long, env = "ATP_POOL_TIMEOUT_SECS", help_heading = "Oracle Database")]
    oracle_pool_timeout_secs: Option<u64>,

    // ==================== PostgreSQL Database ====================
    /// PostgreSQL database connection URL
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_db_url: Option<String>,

    /// Maximum PostgreSQL connection pool size
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_pool_max_size: Option<usize>,

    // ==================== Redis Database ====================
    /// Redis connection URL
    #[arg(long, help_heading = "Redis Database")]
    redis_url: Option<String>,

    /// Maximum Redis connection pool size
    #[arg(long, help_heading = "Redis Database")]
    redis_pool_max_size: Option<usize>,

    /// Redis data retention in days (-1 for persistent, default 30)
    #[arg(long, help_heading = "Redis Database")]
    redis_retention_days: Option<i64>,

    // ==================== TLS/mTLS Security ====================
    /// Path to server TLS certificate (PEM format)
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_cert_path: Option<String>,

    /// Path to server TLS private key (PEM format)
    #[arg(long, help_heading = "TLS/mTLS Security")]
    tls_key_path: Option<String>,

    // ==================== Tracing (OpenTelemetry) ====================
    /// Enable OpenTelemetry tracing
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Tracing (OpenTelemetry)"
    )]
    enable_trace: bool,

    /// OTLP collector endpoint (format: host:port)
    #[arg(
        long,
        default_value = "localhost:4317",
        help_heading = "Tracing (OpenTelemetry)"
    )]
    otlp_traces_endpoint: String,

    // ==================== Control Plane Authentication ====================
    /// API key for worker authorization
    #[arg(long, help_heading = "Control Plane Authentication")]
    api_key: Option<String>,

    /// Per-tenant API keys for serving-path auth (format: tenant_id:key,
    /// repeatable). Layers on top of `--api-key`, each resolving to its own
    /// tenant identity.
    #[arg(long = "tenant-api-key", action = ArgAction::Append, help_heading = "Data Plane Authentication")]
    tenant_api_keys: Vec<String>,

    /// JWT issuer URL for OIDC authentication
    #[arg(
        long,
        env = "JWT_ISSUER",
        help_heading = "Control Plane Authentication"
    )]
    jwt_issuer: Option<String>,

    /// Expected JWT audience claim
    #[arg(
        long,
        env = "JWT_AUDIENCE",
        help_heading = "Control Plane Authentication"
    )]
    jwt_audience: Option<String>,

    /// Explicit JWKS URI (discovered from issuer if not set)
    #[arg(
        long,
        env = "JWT_JWKS_URI",
        help_heading = "Control Plane Authentication"
    )]
    jwt_jwks_uri: Option<String>,

    /// JWT claim name containing the role
    #[arg(
        long,
        default_value = "roles",
        help_heading = "Control Plane Authentication"
    )]
    jwt_role_claim: String,

    /// Role mapping from IDP to gateway role (format: idp_role=gateway_role)
    #[arg(long, action = ArgAction::Append, help_heading = "Control Plane Authentication")]
    jwt_role_mapping: Vec<String>,

    /// API keys for control plane access (format: id:name:role:key)
    #[arg(long = "control-plane-api-keys", action = ArgAction::Append, env = "CONTROL_PLANE_API_KEYS", help_heading = "Control Plane Authentication")]
    control_plane_api_keys: Vec<String>,

    /// Disable audit logging for control plane operations
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Control Plane Authentication"
    )]
    disable_audit_logging: bool,

    // ==================== Mesh Server ====================
    #[arg(long, default_value_t = false)]
    enable_mesh: bool,

    #[arg(long)]
    mesh_server_name: Option<String>,

    /// Bind address for the mesh listener.
    #[arg(long, default_value = "0.0.0.0")]
    mesh_host: String,

    /// Advertised address for this mesh node.
    /// Required when `--mesh-host` is an unspecified bind address such as `0.0.0.0`.
    #[arg(long)]
    mesh_advertise_host: Option<String>,

    #[arg(long, default_value_t = 39527)]
    mesh_port: u16,

    #[arg(long, num_args = 0..)]
    mesh_peer_urls: Vec<String>,

    // ==================== WebRTC ====================
    /// Bind address for WebRTC UDP sockets (client-facing ICE candidate IP).
    /// Default: 0.0.0.0 (auto-detect via routing table).
    /// Set to 127.0.0.1 for local development on the same machine.
    #[arg(long, help_heading = "WebRTC")]
    webrtc_bind_addr: Option<std::net::IpAddr>,

    /// STUN server for ICE candidate gathering (host:port).
    /// Set to your own STUN server for enterprise deployments that
    /// restrict outbound traffic to external STUN servers.
    /// Defaults to `stun.l.google.com:19302`. Set to "none" to disable.
    #[arg(long, help_heading = "WebRTC")]
    webrtc_stun_server: Option<String>,

    // ==================== Runtime ====================
    /// Explicit async runtime worker-thread count. Leave unset to use tokio's
    /// default (`available_parallelism()`), which already honors the cgroup CPU
    /// quota on Rust 1.95+ and is therefore container-aware.
    #[arg(long, help_heading = "Runtime")]
    runtime_worker_threads: Option<usize>,
}

enum OracleConnectSource {
    Dsn { descriptor: String },
    Wallet { path: String, alias: String },
}

/// Validate `--model-id-from` value at CLI parse time.
fn parse_model_id_from(s: &str) -> Result<String, String> {
    ModelIdSource::parse(s)?;
    Ok(s.to_string())
}

/// Validate `--routing-key-headers` values at CLI parse time; names are
/// normalized to lowercase.
fn parse_routing_key_header(s: &str) -> Result<String, String> {
    http::header::HeaderName::try_from(s)
        .map(|name| name.as_str().to_string())
        .map_err(|e| format!("Invalid header name '{s}': {e}"))
}

/// Validate `--model-alias` value at CLI parse time (format: alias=canonical).
fn parse_model_alias(s: &str) -> Result<String, String> {
    let Some((alias, canonical)) = s.split_once('=') else {
        return Err(format!(
            "Invalid model-alias value '{s}'. Expected: <alias>=<canonical>"
        ));
    };
    if alias.is_empty() || canonical.is_empty() {
        return Err(format!(
            "Invalid model-alias value '{s}'. Alias and canonical model ID must be non-empty"
        ));
    }
    if alias == canonical {
        return Err(format!(
            "Invalid model-alias value '{s}'. Alias must differ from the canonical model ID"
        ));
    }
    Ok(s.to_string())
}

/// Parse role mapping from CLI format "idp_role=gateway_role"
#[expect(
    clippy::print_stderr,
    reason = "pre-logger CLI argument parsing warnings"
)]
fn parse_role_mapping(mapping: &str) -> Option<(String, Role)> {
    let parts: Vec<&str> = mapping.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!(
            "WARNING: Invalid role mapping format '{mapping}'. Expected 'idp_role=gateway_role'"
        );
        return None;
    }
    let idp_role = parts[0].to_string();
    let gateway_role = match parts[1].to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid gateway role '{other}' in mapping. Valid roles: admin, user"
            );
            return None;
        }
    };
    Some((idp_role, gateway_role))
}

/// Parse control plane API key from CLI format "id:name:role:key"
#[expect(
    clippy::print_stderr,
    reason = "pre-logger CLI argument parsing warnings"
)]
fn parse_control_plane_api_key(key_str: &str) -> Option<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        eprintln!(
            "WARNING: Invalid control-plane-api-key format '{key_str}'. Expected 'id:name:role:key'"
        );
        return None;
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        other => {
            eprintln!(
                "WARNING: Invalid role '{other}' in control-plane-api-key. Valid roles: admin, user"
            );
            return None;
        }
    };

    Some(ApiKeyEntry::new(id, name, key, role))
}

/// Parse a tenant-scoped data-plane API key from CLI format "tenant_id:key".
/// Only checks for the ':' separator; non-empty/duplicate checks live in
/// `ConfigValidator::validate_tenant_api_keys` so they also cover
/// `TenantApiKeyEntry` values from a config file or language binding. Fails
/// hard (an empty `AuthConfig` disables auth entirely, not just narrows it)
/// and never echoes `key_str`, which may be the plaintext credential.
fn parse_tenant_api_key(key_str: &str) -> ConfigResult<TenantApiKeyEntry> {
    let Some((tenant_id, key)) = key_str.split_once(':') else {
        return Err(ConfigError::InvalidValue {
            field: "tenant-api-key".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'tenant_id:key' (missing ':' separator)".to_string(),
        });
    };

    Ok(TenantApiKeyEntry {
        tenant_id: tenant_id.trim().to_string(),
        key: key.trim().to_string(),
    })
}

impl CliArgs {
    /// Build control plane authentication configuration from CLI args.
    #[expect(clippy::print_stderr, reason = "pre-logger CLI configuration warnings")]
    fn build_control_plane_auth_config(&self) -> ControlPlaneAuthConfig {
        // Build JWT config if issuer and audience are provided
        let jwt = match (&self.jwt_issuer, &self.jwt_audience) {
            (Some(issuer), Some(audience)) => {
                let role_mapping: HashMap<String, Role> = self
                    .jwt_role_mapping
                    .iter()
                    .filter_map(|m| parse_role_mapping(m))
                    .collect();

                let mut jwt_config = JwtConfig::new(issuer.clone(), audience.clone());
                jwt_config.role_claim.clone_from(&self.jwt_role_claim);
                jwt_config.role_mapping = role_mapping;
                if let Some(jwks_uri) = &self.jwt_jwks_uri {
                    jwt_config.jwks_uri = Some(jwks_uri.clone());
                }
                Some(jwt_config)
            }
            (Some(_), None) => {
                eprintln!("WARNING: --jwt-issuer provided but --jwt-audience is missing. JWT auth disabled.");
                None
            }
            (None, Some(_)) => {
                eprintln!("WARNING: --jwt-audience provided but --jwt-issuer is missing. JWT auth disabled.");
                None
            }
            (None, None) => None,
        };

        // Build API keys from CLI args
        let api_keys: Vec<ApiKeyEntry> = self
            .control_plane_api_keys
            .iter()
            .filter_map(|k| parse_control_plane_api_key(k))
            .collect();

        ControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled: !self.disable_audit_logging,
        }
    }

    fn determine_connection_mode(worker_urls: &[String]) -> ConnectionMode {
        // First worker URL that declares ipc:// or grpc:// wins; http:// and bare
        // host:port fall through to the HTTP default. See ConnectionMode::from_url.
        worker_urls
            .iter()
            .find_map(|url| match ConnectionMode::from_url(url) {
                mode @ (Some(ConnectionMode::Zmq) | Some(ConnectionMode::Grpc)) => mode,
                _ => None,
            })
            .unwrap_or(ConnectionMode::Http)
    }

    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    fn parse_mesh_socket_addr(
        host: &str,
        port: u16,
        field: &str,
    ) -> ConfigResult<std::net::SocketAddr> {
        let addr = format!("{host}:{port}");
        addr.parse::<std::net::SocketAddr>()
            .map_err(|e| ConfigError::InvalidValue {
                field: field.to_string(),
                value: host.to_string(),
                reason: format!("invalid mesh socket address '{addr}': {e}"),
            })
    }

    fn build_mesh_server_config(&self) -> ConfigResult<Option<MeshServerConfig>> {
        if !self.enable_mesh {
            return Ok(None);
        }

        let self_name = if let Some(name) = &self.mesh_server_name {
            validate_mesh_server_name(name)?;
            name.to_string()
        } else {
            let mut rng = rand::rng();
            let random_string: String = (0..4).map(|_| rng.sample(Alphanumeric) as char).collect();
            format!("Mesh_{random_string}")
        };

        let peer = self
            .mesh_peer_urls
            .first()
            .map(|url| {
                url.parse::<std::net::SocketAddr>()
                    .map_err(|e| ConfigError::InvalidValue {
                        field: "mesh_peer_urls".to_string(),
                        value: url.clone(),
                        reason: format!("invalid socket address: {e}"),
                    })
            })
            .transpose()?;

        let bind_addr = Self::parse_mesh_socket_addr(&self.mesh_host, self.mesh_port, "mesh_host")?;
        let (advertise_host, advertise_field) =
            if let Some(host) = self.mesh_advertise_host.as_deref() {
                (host, "mesh_advertise_host")
            } else {
                (self.mesh_host.as_str(), "mesh_host")
            };
        let advertise_addr =
            Self::parse_mesh_socket_addr(advertise_host, self.mesh_port, advertise_field)?;

        if advertise_addr.ip().is_unspecified() {
            return Err(ConfigError::InvalidValue {
                field: advertise_field.to_string(),
                value: advertise_host.to_string(),
                reason:
                    "mesh advertise address cannot be unspecified; set --mesh-advertise-host to a routable node IP".to_string(),
            });
        }

        Ok(Some(MeshServerConfig {
            self_name,
            bind_addr,
            advertise_addr,
            init_peer: peer,
            mtls_config: None,
        }))
    }

    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "passthrough" => PolicyConfig::Passthrough,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
                block_size: self.block_size,
                balance_token_usage_threshold: self.balance_token_usage_threshold,
                overload_token_usage_threshold: self.overload_token_usage_threshold,
                overlap_decay: self.overlap_decay,
                selection_temperature: self.selection_temperature,
                cache_index: Self::parse_cache_index(&self.cache_index),
                cache_ttl_secs: self.cache_ttl_secs,
                cache_boundaries: self.cache_boundaries.clone(),
            },
            "cache_aware_length" => PolicyConfig::CacheAwareLength {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
                block_size: self.block_size,
                balance_token_usage_threshold: self.balance_token_usage_threshold,
                overload_token_usage_threshold: self.overload_token_usage_threshold,
                overlap_decay: self.overlap_decay,
                selection_temperature: self.selection_temperature,
                cache_index: Self::parse_cache_index(&self.cache_index),
                cache_ttl_secs: self.cache_ttl_secs,
                cache_boundaries: self.cache_boundaries.clone(),
                chars_per_token: self.chars_per_token,
                long_prefill_threshold: self.long_prefill_threshold,
                long_pool_max_load: self.long_pool_max_load,
                short_pool_max_load: self.short_pool_max_load,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            "least_load" => PolicyConfig::LeastLoad {
                load_check_interval_secs: 5,
                kv_pressure_weight: self.least_load_kv_pressure_weight,
                mean_prefill_tokens: self.least_load_mean_prefill_tokens,
                default_throughput: self.least_load_default_throughput,
                max_waiting_requests: self.least_load_max_waiting_requests,
            },
            "bucket" => PolicyConfig::Bucket {
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                bucket_adjust_interval_secs: 5,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: self.prefix_token_count,
                load_factor: self.prefix_hash_load_factor,
                balance_abs_threshold: self.prefix_hash_balance_abs_threshold,
                cache_boundaries: self.cache_boundaries.clone(),
            },
            "consistent_hashing" => PolicyConfig::ConsistentHashing,
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs: self.eviction_interval,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.assignment_mode.as_deref(),
                    ManualAssignmentMode::Random,
                ),
            },
            _ => PolicyConfig::RoundRobin,
        }
    }

    #[expect(
        clippy::panic,
        reason = "unreachable: clap value_parser restricts valid cache index kinds"
    )]
    fn parse_cache_index(kind: &str) -> CacheIndexKind {
        match kind {
            "tree" => CacheIndexKind::Tree,
            "hash" => CacheIndexKind::Hash,
            other => panic!("Unknown cache index: {other}"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "unreachable: clap value_parser restricts valid assignment modes"
    )]
    fn parse_assignment_mode(
        mode: Option<&str>,
        default: ManualAssignmentMode,
    ) -> ManualAssignmentMode {
        let Some(mode) = mode else { return default };
        match mode {
            "random" => ManualAssignmentMode::Random,
            "min_load" => ManualAssignmentMode::MinLoad,
            "min_group" => ManualAssignmentMode::MinGroup,
            "delegate" => ManualAssignmentMode::Delegate,
            other => panic!("Unknown assignment mode: {other}"),
        }
    }

    fn load_schema_config(&self) -> ConfigResult<Option<SchemaConfig>> {
        match &self.schema_config {
            Some(path) => {
                let content =
                    std::fs::read_to_string(path).map_err(|e| ConfigError::ValidationFailed {
                        reason: format!("Failed to read schema config file '{path}': {e}"),
                    })?;
                let schema: SchemaConfig =
                    serde_yaml::from_str(&content).map_err(|e| ConfigError::ValidationFailed {
                        reason: format!("Failed to parse schema config file '{path}': {e}"),
                    })?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    fn resolve_oracle_connect_details(&self) -> ConfigResult<OracleConnectSource> {
        if let Some(dsn) = self.oracle_dsn.clone() {
            return Ok(OracleConnectSource::Dsn { descriptor: dsn });
        }

        let wallet_path =
            self.oracle_wallet_path
                .clone()
                .ok_or_else(|| ConfigError::MissingRequired {
                    field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
                })?;

        let tns_alias =
            self.oracle_tns_alias
                .clone()
                .ok_or_else(|| ConfigError::MissingRequired {
                    field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
                })?;

        Ok(OracleConnectSource::Wallet {
            path: wallet_path,
            alias: tns_alias,
        })
    }

    fn build_oracle_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match self.resolve_oracle_connect_details()? {
            OracleConnectSource::Dsn { descriptor } => (None, descriptor),
            OracleConnectSource::Wallet { path, alias } => (Some(path), alias),
        };
        let (username, password) = if self.oracle_external_auth {
            (
                self.oracle_user.clone().unwrap_or_default(),
                self.oracle_password.clone().unwrap_or_default(),
            )
        } else {
            (
                self.oracle_user
                    .clone()
                    .ok_or_else(|| ConfigError::MissingRequired {
                        field: "oracle_user or ATP_USER".to_string(),
                    })?,
                self.oracle_password
                    .clone()
                    .ok_or_else(|| ConfigError::MissingRequired {
                        field: "oracle_password or ATP_PASSWORD".to_string(),
                    })?,
            )
        };

        let pool_min = self
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .oracle_pool_max
            .unwrap_or_else(OracleConfig::default_pool_max);

        if pool_min == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_min".to_string(),
                value: pool_min.to_string(),
                reason: "pool minimum must be at least 1".to_string(),
            });
        }

        if pool_max < pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle_pool_max".to_string(),
                value: pool_max.to_string(),
                reason: "pool maximum must be greater than or equal to minimum".to_string(),
            });
        }

        let pool_timeout_secs = self
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            external_auth: self.oracle_external_auth,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
            schema,
        })
    }

    fn build_postgres_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<PostgresConfig> {
        let db_url = self.postgres_db_url.clone().unwrap_or_default();
        let pool_max = self
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);
        let pcf = PostgresConfig {
            db_url,
            pool_max,
            schema,
        };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(pcf)
    }

    fn build_redis_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<RedisConfig> {
        let url = self.redis_url.clone().unwrap_or_default();
        let pool_max = self.redis_pool_max_size.unwrap_or(16);

        let retention_days = match self.redis_retention_days {
            Some(d) if d < 0 => None, // Persistent
            Some(d) => Some(d as u64),
            None => Some(30), // Default 30 days
        };

        let rcf = RedisConfig {
            url,
            pool_max,
            retention_days,
            schema,
        };
        rcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(rcf)
    }

    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        // Determine routing mode based on backend type and PD disaggregation flag
        // IGW mode doesn't change routing mode, only affects router initialization
        let mode = if matches!(self.backend, Some(Backend::Openai)) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if matches!(self.backend, Some(Backend::Anthropic)) {
            RoutingMode::Anthropic {
                worker_urls: self.worker_urls.clone(),
            }
        } else if matches!(self.backend, Some(Backend::Gemini)) {
            RoutingMode::Gemini {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.epd_disaggregation {
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls: self.decode.clone(),
                encode_policy: self.encode_policy.as_ref().map(|p| self.parse_policy(p)),
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.decode.clone(),
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = self.parse_policy(&self.policy);

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                encode_selector: Self::parse_selector(&self.encode_selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                worker_ports_annotation: "smg.ai/worker-ports".to_string(),
                router_selector: Self::parse_selector(&self.router_selector),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                model_id_source: self.model_id_from.clone(),
            })
        } else {
            None
        };

        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        let trace_config = Some(TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let mut all_urls = Vec::new();
        match &mode {
            RoutingMode::Regular { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                for (url, _) in prefill_urls {
                    all_urls.push(url.clone());
                }
                all_urls.extend(decode_urls.clone());
            }
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                ..
            } => {
                for (url, _) in encode_urls.iter().chain(prefill_urls.iter()) {
                    all_urls.push(url.clone());
                }
                all_urls.extend(decode_urls.clone());
            }
            RoutingMode::OpenAI { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::Anthropic { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
            RoutingMode::Gemini { worker_urls } => {
                all_urls.extend(worker_urls.clone());
            }
        }
        let connection_mode = Self::determine_connection_mode(&all_urls);

        // `--backend` normally only steers the routing mode. Over ZMQ it
        // additionally pins the startup workers' runtime: the shared EngineCore
        // handshake carries no engine identity, so the wire protocol cannot be
        // probed and must be declared. HTTP/gRPC keep auto-detection (None).
        let startup_worker_runtime_type = if connection_mode == ConnectionMode::Zmq {
            match self.backend {
                Some(Backend::Vllm) => Some(RuntimeType::Vllm),
                Some(Backend::Tokenspeed) => Some(RuntimeType::TokenSpeed),
                _ => None,
            }
        } else {
            None
        };

        let history_backend = match self.history_backend.as_str() {
            "none" => HistoryBackend::None,
            "oracle" => HistoryBackend::Oracle,
            "postgres" => HistoryBackend::Postgres,
            "redis" => HistoryBackend::Redis,
            _ => HistoryBackend::Memory,
        };

        let schema = self.load_schema_config()?;

        let tenant_api_keys = self
            .tenant_api_keys
            .iter()
            .map(|k| parse_tenant_api_key(k))
            .collect::<ConfigResult<Vec<_>>>()?;

        let (oracle, postgres, redis) = match history_backend {
            HistoryBackend::Oracle => (Some(self.build_oracle_config(schema)?), None, None),
            HistoryBackend::Postgres => (None, Some(self.build_postgres_config(schema)?), None),
            HistoryBackend::Redis => (None, None, Some(self.build_redis_config(schema)?)),
            _ => (None, None, None),
        };

        // clap validated each entry's shape; here we only reject the same
        // alias naming two different canonical models, which would make the
        // routing outcome depend on argument order.
        let mut model_aliases: HashMap<String, String> = HashMap::new();
        for entry in &self.model_alias {
            let Some((alias, canonical)) = entry.split_once('=') else {
                return Err(ConfigError::InvalidValue {
                    field: "model_alias".to_string(),
                    value: entry.clone(),
                    reason: "Expected: <alias>=<canonical>".to_string(),
                });
            };
            if let Some(previous) = model_aliases.insert(alias.to_string(), canonical.to_string()) {
                if previous != canonical {
                    return Err(ConfigError::InvalidValue {
                        field: "model_alias".to_string(),
                        value: alias.to_string(),
                        reason: format!("Alias maps to both '{previous}' and '{canonical}'"),
                    });
                }
            }
        }

        let builder = RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .cache_boundaries(self.cache_boundaries.clone())
            .long_prefill_indices(self.long_prefill_indices.clone())
            .connection_mode(connection_mode)
            .startup_worker_runtime_type(startup_worker_runtime_type)
            .zmq_engine_count(self.zmq_engine_count)
            .host(&self.host)
            .port(self.port)
            .health_check_port(self.health_check_port)
            .runtime_worker_threads(self.runtime_worker_threads)
            .max_payload_size(self.max_payload_size)
            .stream_request_bodies_over(self.stream_request_bodies_over)
            .stream_body_stall_timeout_secs(self.stream_body_stall_timeout_secs)
            .request_timeout_secs(self.request_timeout_secs)
            .upstream_pool_idle_timeout_secs(self.upstream_pool_idle_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_delay_secs(self.worker_startup_delay)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .job_queue_capacity(self.job_queue_capacity)
            .job_queue_concurrency(self.job_queue_concurrency)
            .load_monitor_interval_secs(self.load_monitor_interval)
            .disable_load_monitoring(self.disable_load_monitoring)
            .worker_overload_protection(self.worker_overload_protection)
            .worker_overload_waiting_requests(self.worker_overload_waiting_requests)
            .worker_overload_token_usage(self.worker_overload_token_usage)
            .kv_indexer_ttl_secs(self.kv_indexer_ttl_secs)
            .kv_indexer_max_entries(self.kv_indexer_max_entries)
            .engine_metrics(self.engine_metrics)
            .multimodal_tensor_transport(self.multimodal_tensor_transport)
            .multimodal_shm_min_bytes(self.multimodal_shm_min_bytes)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .priority_scheduler_enabled(self.priority_scheduler_enabled)
            .priority_scheduler_default_max_class(self.priority_scheduler_default_max_class.clone())
            .priority_scheduler_config(self.priority_scheduler_config.clone())
            .priority_scheduler_tenant_metric_top_n(self.priority_scheduler_tenant_metric_top_n)
            .tenant_rate_limit_enabled(self.tenant_rate_limit_enabled)
            .tenant_rate_limit_config(self.tenant_rate_limit_config.clone())
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
                disable_health_check: self.disable_health_check,
                remove_unhealthy_workers: self.remove_unhealthy_workers,
                drain_settle_secs: self.drain_settle_secs,
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .disable_tokenizer_autoload(self.disable_tokenizer_autoload)
            .history_backend(history_backend)
            .log_level(&self.log_level)
            .maybe_api_key(self.api_key.as_ref())
            .tenant_api_keys(tenant_api_keys)
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_request_id_headers(
                (!self.request_id_headers.is_empty()).then(|| self.request_id_headers.clone()),
            )
            .maybe_storage_context_headers(
                (!self.storage_context_headers.is_empty())
                    .then(|| Self::parse_selector(&self.storage_context_headers)),
            )
            .trust_tenant_header(self.trust_tenant_header)
            .tenant_header_name(&self.tenant_header_name)
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .model_aliases(model_aliases)
            .maybe_oracle(oracle)
            .maybe_postgres(postgres)
            .maybe_redis(redis)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .dp_aware(self.dp_aware)
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: self.routing_key_override,
                eviction_interval_secs: self.eviction_interval,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.assignment_mode.as_deref(),
                    ManualAssignmentMode::Delegate,
                ),
                headers: self.routing_key_headers.clone(),
            })
            .retries(!self.disable_retries)
            .upstream_http2(self.upstream_http2)
            .circuit_breaker(!self.disable_circuit_breaker)
            .enable_wasm(self.enable_wasm)
            .maybe_storage_hook_wasm_path(self.storage_hook_wasm_path.as_deref())
            .igw(self.enable_igw)
            .dp_minimum_tokens_scheduler(self.dp_minimum_tokens_scheduler)
            .maybe_server_cert_and_key(self.tls_cert_path.as_ref(), self.tls_key_path.as_ref());

        builder.build()
    }

    fn to_server_config(&self, router_config: RouterConfig) -> ConfigResult<ServerConfig> {
        let service_discovery_config = if self.service_discovery {
            // Get router discovery config from router_config.discovery if available
            let (router_selector, router_mesh_port_annotation) = router_config
                .discovery
                .as_ref()
                .map(|d| {
                    (
                        d.router_selector.clone(),
                        d.router_mesh_port_annotation.clone(),
                    )
                })
                .unwrap_or_else(|| (HashMap::new(), "sglang.ai/mesh-port".to_string()));

            let model_id_source = self
                .model_id_from
                .as_deref()
                .or_else(|| {
                    router_config
                        .discovery
                        .as_ref()
                        .and_then(|d| d.model_id_source.as_deref())
                })
                .map(|s| {
                    ModelIdSource::parse(s).map_err(|e| ConfigError::InvalidValue {
                        field: "model_id_source".to_string(),
                        value: s.to_string(),
                        reason: e,
                    })
                })
                .transpose()?;

            Some(ServiceDiscoveryConfig {
                enabled: true,
                selector: Self::parse_selector(&self.selector),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                disaggregated_mode: self.pd_disaggregation || self.epd_disaggregation,
                encode_selector: Self::parse_selector(&self.encode_selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                worker_ports_annotation: "smg.ai/worker-ports".to_string(),
                router_selector,
                router_mesh_port_annotation,
                model_id_source,
            })
        } else {
            None
        };

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
            duration_buckets: if self.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.prometheus_duration_buckets.clone())
            },
        });

        // Build control plane auth config
        let control_plane_auth = {
            let config = self.build_control_plane_auth_config();
            if config.is_enabled() {
                Some(config)
            } else {
                None
            }
        };

        // ==================== Mesh Server ====================
        let mesh_server_config = self.build_mesh_server_config()?;

        Ok(ServerConfig {
            host: self.host.clone(),
            port: self.port,
            health_check_port: self.health_check_port,
            runtime_worker_threads: self.runtime_worker_threads,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            log_json: self.log_json,
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.shutdown_grace_period_secs,
            control_plane_auth,
            mesh_server_config,
            webrtc_bind_addr: self.webrtc_bind_addr,
            webrtc_stun_server: self.webrtc_stun_server.clone(),
        })
    }
}

#[expect(
    clippy::print_stdout,
    reason = "pre-logger startup output and version display"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_jemalloc_as_global_allocator();

    // Check for version flags before parsing other args to avoid errors
    let args: Vec<String> = std::env::args().collect();
    for arg in &args {
        if arg == "--version" || arg == "-V" {
            println!("{}", version::get_version_string());
            return Ok(());
        }
        if arg == "--version-verbose" {
            println!("{}", version::get_verbose_version_string());
            return Ok(());
        }
    }

    let prefill_urls = parse_prefill_args();
    let encode_urls = parse_encode_args();

    let mut filtered_args: Vec<String> = Vec::new();
    let raw_args: Vec<String> = std::env::args().collect();
    let mut i = 0;

    while i < raw_args.len() {
        if (raw_args[i] == "--prefill" || raw_args[i] == "--encode") && i + 1 < raw_args.len() {
            i += 2;
            if i < raw_args.len()
                && !raw_args[i].starts_with("--")
                && (raw_args[i].parse::<u16>().is_ok() || raw_args[i].to_lowercase() == "none")
            {
                i += 1;
            }
        } else {
            filtered_args.push(raw_args[i].clone());
            i += 1;
        }
    }

    let cli = Cli::parse_from(filtered_args);

    // Handle subcommands or use direct args
    let mut cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };

    // Automatically enable IGW mode when service discovery is turned on
    if cli_args.service_discovery && !cli_args.enable_igw {
        println!("INFO: IGW mode automatically enabled because service discovery is turned on");
        cli_args.enable_igw = true;
    }

    let mode_str = if cli_args.enable_igw {
        "IGW (Inference Gateway)".to_string()
    } else if matches!(cli_args.backend, Some(Backend::Openai)) {
        "OpenAI Backend".to_string()
    } else if matches!(cli_args.backend, Some(Backend::Anthropic)) {
        "Anthropic Backend".to_string()
    } else if cli_args.epd_disaggregation {
        "EPD Disaggregated".to_string()
    } else if cli_args.pd_disaggregation {
        "PD Disaggregated".to_string()
    } else if let Some(backend) = &cli_args.backend {
        format!("Regular ({backend})")
    } else {
        "Regular".to_string()
    };

    version::print_banner(&cli_args.host, cli_args.port, &mode_str);

    if !cli_args.enable_igw {
        println!("Policy: {}", cli_args.policy);

        if cli_args.pd_disaggregation && !prefill_urls.is_empty() {
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", cli_args.decode);
        }

        if cli_args.epd_disaggregation {
            println!("Encode nodes: {encode_urls:?}");
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", cli_args.decode);
        }
    }

    let router_config = cli_args.to_router_config(prefill_urls, encode_urls)?;
    router_config.validate()?;

    let server_config = cli_args.to_server_config(router_config)?;
    // tokio's default worker-thread count is `available_parallelism()`, which on
    // Rust 1.95+ already honors the cgroup CPU quota, so the default is
    // container-aware. Only build the runtime explicitly when an operator pins a
    // worker-thread count.
    let runtime = match server_config.runtime_worker_threads {
        Some(n) => {
            info!(
                worker_threads = n,
                "Sizing tokio runtime (explicit override)"
            );
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(n)
                .enable_all()
                .build()?
        }
        None => {
            info!("Sizing tokio runtime (default, container-aware)");
            tokio::runtime::Runtime::new()?
        }
    };
    runtime.block_on(Box::pin(server::startup(server_config)))?;
    if is_otel_enabled() {
        shutdown_otel();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse top-level CLI args into the flattened `CliArgs` the binary uses.
    fn cli_args_from(args: &[&str]) -> CliArgs {
        let argv: Vec<String> = std::iter::once("smg".to_string())
            .chain(args.iter().map(|s| (*s).to_string()))
            .collect();
        Cli::parse_from(argv).router_args
    }

    /// A grouped ZMQ handshake needs at least one engine, so `0` (and any
    /// non-positive value) must be rejected at parse time rather than silently
    /// degrading to an ungrouped worker.
    #[test]
    fn zmq_engine_count_rejects_non_positive() {
        assert_eq!(
            cli_args_from(&["--zmq-engine-count", "2"]).zmq_engine_count,
            Some(2)
        );
        assert_eq!(cli_args_from(&[]).zmq_engine_count, None);

        for bad in ["0", "-1"] {
            let argv = ["smg", "--zmq-engine-count", bad];
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "--zmq-engine-count {bad} should be rejected"
            );
        }
    }

    /// The indexer prune flags are router-only settings and must flow into
    /// `RouterConfig` (unset by default so the indexer stays unbounded).
    #[test]
    fn kv_indexer_prune_flags_flow_into_router_config() {
        let cli = cli_args_from(&[
            "--kv-indexer-ttl-secs",
            "600",
            "--kv-indexer-max-entries",
            "500000",
        ]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.kv_indexer_ttl_secs, Some(600));
        assert_eq!(router_config.kv_indexer_max_entries, Some(500_000));

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.kv_indexer_ttl_secs, None);
        assert_eq!(defaults.kv_indexer_max_entries, None);
    }

    /// Job queue sizing must flow through both conversion paths: into
    /// `RouterConfig` and through the `RouterConfig` carried by
    /// `ServerConfig`, which is what constructs the queue at startup.
    #[test]
    fn job_queue_flags_flow_into_both_config_paths() {
        let cli = cli_args_from(&[
            "--job-queue-capacity",
            "20000",
            "--job-queue-concurrency",
            "500",
        ]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.job_queue_capacity, 20_000);
        assert_eq!(router_config.job_queue_concurrency, 500);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.router_config.job_queue_capacity, 20_000);
        assert_eq!(server_config.router_config.job_queue_concurrency, 500);

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.job_queue_capacity, 1000);
        assert_eq!(defaults.job_queue_concurrency, 200);
    }

    /// A zero-capacity channel panics at construction and a zero-permit
    /// dispatcher never dequeues; both bounds are enforced at parse time.
    #[test]
    fn job_queue_flags_reject_out_of_range_values() {
        for (flag, bad) in [
            ("--job-queue-capacity", "0"),
            ("--job-queue-capacity", "1000001"),
            ("--job-queue-concurrency", "0"),
            ("--job-queue-concurrency", "100001"),
        ] {
            let argv = ["smg", flag, bad];
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "{flag} {bad} should be rejected"
            );
        }
    }

    /// The streamed-body stall timeout is a router-only setting and must
    /// flow into `RouterConfig`.
    #[test]
    fn stream_stall_timeout_flag_flows_into_router_config() {
        let cli = cli_args_from(&["--stream-body-stall-timeout-secs", "120"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.stream_body_stall_timeout_secs, 120);

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.stream_body_stall_timeout_secs, 300);
    }

    #[test]
    fn routing_key_flags_flow_into_router_config() {
        // The override enables with no other configuration; the sticky map
        // defaults to delegate while the manual policy default stays random.
        let cli = cli_args_from(&["--routing-key-override"]);
        let config = cli.to_router_config(vec![], vec![]).unwrap();
        let override_cfg = &config.routing_key_override;
        assert!(override_cfg.enabled);
        assert_eq!(override_cfg.assignment_mode, ManualAssignmentMode::Delegate);

        // An explicit --assignment-mode overrides the sticky-map default.
        let explicit = cli_args_from(&["--routing-key-override", "--assignment-mode", "min_load"])
            .to_router_config(vec![], vec![])
            .unwrap();
        assert_eq!(
            explicit.routing_key_override.assignment_mode,
            ManualAssignmentMode::MinLoad
        );

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert!(!defaults.routing_key_override.enabled);
        assert_eq!(
            defaults.routing_key_override.headers,
            vec!["x-smg-routing-key".to_string()]
        );
    }

    #[test]
    fn routing_key_headers_flow_into_router_config() {
        // Space-separated list, order preserved; names normalize to lowercase.
        let config = cli_args_from(&[
            "--routing-key-headers",
            "X-Routing-Key",
            "x-smg-routing-key",
        ])
        .to_router_config(vec![], vec![])
        .unwrap();
        assert_eq!(
            config.routing_key_override.headers,
            vec!["x-routing-key".to_string(), "x-smg-routing-key".to_string()]
        );

        for bad in ["has space", "bad\u{7f}name", ""] {
            let argv = ["smg", "--routing-key-headers", bad];
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "--routing-key-headers {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn manual_policy_assignment_default_stays_random() {
        // One invocation, both contexts: the manual policy keeps its random
        // default while the override sticky map defaults to delegate.
        let config = cli_args_from(&["--policy", "manual", "--routing-key-override"])
            .to_router_config(vec![], vec![])
            .unwrap();
        match &config.policy {
            PolicyConfig::Manual {
                assignment_mode, ..
            } => assert_eq!(*assignment_mode, ManualAssignmentMode::Random),
            other => panic!("expected manual policy, got {other:?}"),
        }
        assert_eq!(
            config.routing_key_override.assignment_mode,
            ManualAssignmentMode::Delegate
        );
    }

    #[test]
    fn alias_flags_parse_identically_to_canonical() {
        let canonical = cli_args_from(&[
            "--policy",
            "cache_aware",
            "--cache-threshold",
            "0.6",
            "--balance-abs-threshold",
            "8",
            "--balance-rel-threshold",
            "1.2",
            "--max-idle-secs",
            "300",
            "--routing-key-override",
            "--remove-unhealthy-workers",
        ])
        .to_router_config(vec![], vec![])
        .unwrap();
        let aliased = cli_args_from(&[
            "--policy",
            "cache_aware",
            "--cache-match-threshold",
            "0.6",
            "--spill-abs-threshold",
            "8",
            "--spill-rel-threshold",
            "1.2",
            "--sticky-key-idle-secs",
            "300",
            "--sticky-sessions",
            "--worker-auto-recovery",
        ])
        .to_router_config(vec![], vec![])
        .unwrap();
        assert_eq!(format!("{canonical:?}"), format!("{aliased:?}"));
    }

    /// `--health-check-port` must flow into BOTH conversion paths
    /// (`to_router_config` and `to_server_config`), mirroring the main
    /// listener `--port` field exactly. This is the two-path config-plumbing
    /// guard: wiring only one path would let the flag be silently ignored on
    /// the other.
    #[test]
    fn health_check_port_flows_into_both_configs() {
        let cli = cli_args_from(&["--health-check-port", "8081"]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router_config.health_check_port,
            Some(8081),
            "health_check_port must reach RouterConfig via to_router_config"
        );

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.health_check_port,
            Some(8081),
            "health_check_port must reach ServerConfig via to_server_config"
        );
    }

    /// Unset `--health-check-port` means the dedicated probe listener is off:
    /// `None` propagates through both conversions (backward-compatible default).
    #[test]
    fn health_check_port_defaults_to_none_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.health_check_port, None);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.health_check_port, None);
    }

    /// `--engine-metrics` must flow into `RouterConfig` and survive nesting
    /// into `ServerConfig.router_config` — the consumer (load monitor) reads it
    /// off `RouterConfig`. Two-path config-plumbing guard.
    #[test]
    fn engine_metrics_flows_into_both_configs() {
        let cli = cli_args_from(&["--engine-metrics"]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(
            router_config.engine_metrics,
            "engine_metrics must reach RouterConfig via to_router_config"
        );

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(
            server_config.router_config.engine_metrics,
            "engine_metrics must survive into ServerConfig via to_server_config"
        );
    }

    /// The overload thresholds must reach `RouterConfig` and survive nesting
    /// into `ServerConfig.router_config` — the consumer (load monitor) reads
    /// them off `RouterConfig`. Two-path config-plumbing guard.
    #[test]
    fn worker_overload_thresholds_flow_into_both_configs() {
        let cli = cli_args_from(&[
            "--worker-overload-waiting-requests",
            "64",
            "--worker-overload-token-usage",
            "0.9",
        ]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router_config.worker_overload_waiting_requests,
            Some(64),
            "worker_overload_waiting_requests must reach RouterConfig via to_router_config"
        );
        assert_eq!(
            router_config.worker_overload_token_usage,
            Some(0.9),
            "worker_overload_token_usage must reach RouterConfig via to_router_config"
        );

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.router_config.worker_overload_waiting_requests,
            Some(64),
            "worker_overload_waiting_requests must survive into ServerConfig"
        );
        assert_eq!(
            server_config.router_config.worker_overload_token_usage,
            Some(0.9),
            "worker_overload_token_usage must survive into ServerConfig"
        );
    }

    /// Unset means off on both paths: the feature must be byte-identical to
    /// pre-feature behavior until an operator opts in.
    #[test]
    fn worker_overload_thresholds_default_to_unset_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.worker_overload_waiting_requests, None);
        assert_eq!(router_config.worker_overload_token_usage, None);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.router_config.worker_overload_waiting_requests,
            None
        );
        assert_eq!(
            server_config.router_config.worker_overload_token_usage,
            None
        );
    }

    /// Both thresholds are `>=` comparisons, so the excluded ends of their
    /// ranges would veto every worker unconditionally. Reject them at parse
    /// time rather than letting a typo shed all traffic.
    #[test]
    fn degenerate_worker_overload_thresholds_are_rejected_at_parse_time() {
        for argv in [
            vec!["smg", "--worker-overload-waiting-requests", "0"],
            vec!["smg", "--worker-overload-token-usage", "0"],
            vec!["smg", "--worker-overload-token-usage", "1.5"],
            vec!["smg", "--worker-overload-token-usage", "-0.5"],
        ] {
            assert!(
                Cli::try_parse_from(&argv).is_err(),
                "{argv:?} must be rejected at parse time"
            );
        }

        // The inclusive ends of the accepted ranges must still parse.
        assert!(Cli::try_parse_from(["smg", "--worker-overload-waiting-requests", "1"]).is_ok());
        assert!(Cli::try_parse_from(["smg", "--worker-overload-token-usage", "1.0"]).is_ok());
    }

    /// `--worker-overload-protection` and `--disable-load-monitoring` must
    /// reach `RouterConfig` and survive nesting into
    /// `ServerConfig.router_config`. Two-path config-plumbing guard.
    #[test]
    fn overload_protection_and_monitoring_flags_flow_into_both_configs() {
        let cli = cli_args_from(&["--worker-overload-protection", "--disable-load-monitoring"]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(
            router_config.worker_overload_protection,
            "worker_overload_protection must reach RouterConfig via to_router_config"
        );
        assert!(
            router_config.disable_load_monitoring,
            "disable_load_monitoring must reach RouterConfig via to_router_config"
        );
        // The flag alone carries no thresholds; the token default is applied
        // at resolution, not stored in config.
        assert_eq!(router_config.worker_overload_waiting_requests, None);
        assert_eq!(router_config.worker_overload_token_usage, None);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(
            server_config.router_config.worker_overload_protection,
            "worker_overload_protection must survive into ServerConfig"
        );
        assert!(
            server_config.router_config.disable_load_monitoring,
            "disable_load_monitoring must survive into ServerConfig"
        );
    }

    /// Defaults: protection off, monitoring default-on (opt-out false) — the
    /// behavior change is monitoring, and it is carried by the default here.
    #[test]
    fn overload_protection_and_monitoring_flags_default_off_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(!router_config.worker_overload_protection);
        assert!(!router_config.disable_load_monitoring);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(!server_config.router_config.worker_overload_protection);
        assert!(!server_config.router_config.disable_load_monitoring);
    }

    /// Cache-index flags must reach the cache_aware policy variant and the
    /// shared `RouterConfig.cache_boundaries`, and survive into
    /// `ServerConfig.router_config`. Two-path config-plumbing guard.
    #[test]
    fn cache_index_flags_flow_into_both_configs() {
        let cli = cli_args_from(&[
            "--cache-boundaries",
            "2048,8192",
            "--cache-index",
            "hash",
            "--cache-ttl-secs",
            "60",
        ]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.cache_boundaries, vec![2048, 8192]);
        match &router_config.policy {
            PolicyConfig::CacheAware {
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                ..
            } => {
                assert_eq!(*cache_index, CacheIndexKind::Hash);
                assert_eq!(*cache_ttl_secs, 60);
                assert_eq!(*cache_boundaries, vec![2048, 8192]);
            }
            other => panic!("expected cache_aware policy, got {other:?}"),
        }

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.router_config.cache_boundaries,
            vec![2048, 8192],
            "cache_boundaries must survive into ServerConfig via to_server_config"
        );

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert!(defaults.cache_boundaries.is_empty());
        match &defaults.policy {
            PolicyConfig::CacheAware {
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                ..
            } => {
                assert_eq!(*cache_index, CacheIndexKind::Tree);
                assert_eq!(*cache_ttl_secs, 180);
                assert!(cache_boundaries.is_empty());
            }
            other => panic!("expected cache_aware policy, got {other:?}"),
        }
    }

    #[test]
    fn cache_ttl_secs_zero_rejected_at_parse() {
        assert!(Cli::try_parse_from(["smg", "--cache-ttl-secs", "0"]).is_err());
    }

    /// The multimodal transport flags must reach both `RouterConfig` and the
    /// wrapped `ServerConfig.router_config`. Two-path config-plumbing guard.
    #[test]
    fn multimodal_transport_flows_into_both_configs() {
        let cli = cli_args_from(&[
            "--multimodal-tensor-transport",
            "shm",
            "--multimodal-shm-min-bytes",
            "1024",
        ]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router_config.multimodal_tensor_transport,
            Some(TransportMode::Shm),
            "transport mode must reach RouterConfig via to_router_config"
        );
        assert_eq!(router_config.multimodal_shm_min_bytes, Some(1024));

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.router_config.multimodal_tensor_transport,
            Some(TransportMode::Shm),
            "transport mode must survive into ServerConfig via to_server_config"
        );
        assert_eq!(
            server_config.router_config.multimodal_shm_min_bytes,
            Some(1024)
        );
    }

    /// Default is off: the flag stays false through both conversions so
    /// existing deployments keep the routing-gated polling behavior.
    #[test]
    fn engine_metrics_defaults_to_false_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(!router_config.engine_metrics);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(!server_config.router_config.engine_metrics);
    }

    /// clap rejects out-of-range probe ports at parse time (the `u16`
    /// value_parser), matching `--port` validation — no runtime crash.
    #[test]
    fn health_check_port_out_of_range_is_rejected_at_parse_time() {
        let argv = ["smg", "--health-check-port", "70000"];
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "a port above u16::MAX must fail clap parsing"
        );
    }

    /// The `--runtime-worker-threads` override must flow into BOTH conversion
    /// paths (`to_router_config` and `to_server_config`); wiring only one path
    /// would let the flag be silently ignored on the other (the two-path footgun).
    #[test]
    fn runtime_worker_threads_flows_into_both_configs() {
        let cli = cli_args_from(&["--runtime-worker-threads", "3"]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.runtime_worker_threads, Some(3));

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.runtime_worker_threads,
            Some(3),
            "runtime_worker_threads must reach ServerConfig via to_server_config"
        );
    }

    /// Unset, the flag propagates as `None` through both conversions, so the
    /// runtime uses tokio's container-aware default.
    #[test]
    fn runtime_worker_threads_default_to_none_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.runtime_worker_threads, None);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.runtime_worker_threads, None);
    }

    /// Over ZMQ, `--backend` pins the startup workers' runtime (the shared
    /// EngineCore handshake cannot be probed for a wire protocol). The pin must
    /// reach `RouterConfig` and survive nesting into
    /// `ServerConfig.router_config` — two-path config-plumbing guard.
    #[test]
    fn zmq_backend_pins_startup_worker_runtime_in_both_configs() {
        let cli = cli_args_from(&[
            "--backend",
            "tokenspeed",
            "--worker-urls",
            "ipc:///tmp/smg-zmq/engine-0",
        ]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.connection_mode, ConnectionMode::Zmq);
        assert_eq!(
            router_config.startup_worker_runtime_type,
            Some(RuntimeType::TokenSpeed),
            "--backend tokenspeed must pin the ZMQ startup worker runtime"
        );

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(
            server_config.router_config.startup_worker_runtime_type,
            Some(RuntimeType::TokenSpeed),
            "the runtime pin must survive into ServerConfig via to_server_config"
        );
    }

    /// The runtime pin only disambiguates the ZMQ wire protocol: gRPC (and
    /// HTTP) workers keep backend auto-detection even when `--backend` names an
    /// engine, and a ZMQ deployment without `--backend` stays unpinned
    /// (detect_backend's vLLM default applies).
    #[test]
    fn startup_worker_runtime_stays_unpinned_off_the_zmq_path() {
        let grpc = cli_args_from(&[
            "--backend",
            "tokenspeed",
            "--worker-urls",
            "grpc://localhost:30001",
        ]);
        let router_config = grpc.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.connection_mode, ConnectionMode::Grpc);
        assert_eq!(router_config.startup_worker_runtime_type, None);

        let zmq_no_backend = cli_args_from(&["--worker-urls", "ipc:///tmp/smg-zmq/engine-0"]);
        let router_config = zmq_no_backend.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.connection_mode, ConnectionMode::Zmq);
        assert_eq!(router_config.startup_worker_runtime_type, None);

        let zmq_vllm = cli_args_from(&[
            "--backend",
            "vllm",
            "--worker-urls",
            "ipc:///tmp/smg-zmq/engine-0",
        ]);
        let router_config = zmq_vllm.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router_config.startup_worker_runtime_type,
            Some(RuntimeType::Vllm)
        );
    }

    /// The shared `--cache-boundaries` flag must also reach the prefix_hash
    /// policy's resolved copy (the shared-field flow is covered by
    /// `cache_index_flags_flow_into_both_configs`).
    #[test]
    fn cache_boundaries_flow_into_prefix_hash_policy() {
        let cli = cli_args_from(&[
            "--policy",
            "prefix_hash",
            "--cache-boundaries",
            "2048,8192,32768",
        ]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        match &router_config.policy {
            PolicyConfig::PrefixHash {
                cache_boundaries, ..
            } => assert_eq!(cache_boundaries, &vec![2048, 8192, 32768]),
            other => panic!("expected prefix_hash policy, got {other:?}"),
        }

        let defaults = cli_args_from(&["--policy", "prefix_hash"])
            .to_router_config(vec![], vec![])
            .unwrap();
        match &defaults.policy {
            PolicyConfig::PrefixHash {
                cache_boundaries, ..
            } => assert!(cache_boundaries.is_empty()),
            other => panic!("expected prefix_hash policy, got {other:?}"),
        }
    }

    #[test]
    fn non_ascending_cache_boundaries_are_rejected() {
        for bad in ["8192,2048", "0,2048", "2048,2048"] {
            let cli = cli_args_from(&["--cache-boundaries", bad]);
            assert!(
                matches!(
                    cli.to_router_config(vec![], vec![]),
                    Err(ConfigError::InvalidValue { ref field, .. })
                        if field == "cache_boundaries"
                ),
                "--cache-boundaries {bad} should be rejected"
            );
        }
    }

    #[test]
    fn conflicting_model_aliases_are_rejected() {
        let cli = cli_args_from(&["--model-alias", "x=a", "--model-alias", "x=b"]);

        assert!(matches!(
            cli.to_router_config(vec![], vec![]),
            Err(ConfigError::InvalidValue { field, reason, .. })
                if field == "model_alias" && reason == "Alias maps to both 'a' and 'b'"
        ));
    }
}
