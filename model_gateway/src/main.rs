use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use clap::{ArgAction, Parser, Subcommand, ValueEnum};

// Jemalloc as the global allocator: glibc malloc retains freed pages badly
// under the gateway's allocation churn. Prefixed symbols only — vendored C
// libraries keep their own malloc.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Jemalloc's run-time options for a long-running server (see
// `SERVER_MALLOC_CONF`): the executable exports them as jemalloc's
// application-provided `malloc_conf`, as the Python extension does for the
// router launched from Python and the Go FFI library for a Go host.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
use smg::observability::metrics::SERVER_MALLOC_CONF;
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[expect(
    unsafe_code,
    reason = "jemalloc reads its options from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 72] = SERVER_MALLOC_CONF;

use openai_protocol::worker::{MmProcessingMode, TransportMode};
use rand::{distr::Alphanumeric, RngExt};
use smg::{
    config::{
        bind_socket_addr, resolve_worker_auto_recovery, validate_mesh_server_name, CacheIndexKind,
        CircuitBreakerConfig, ConfigError, ConfigResult, DiscoveryConfig, FileDiscoveryConfig,
        HealthCheckConfig, HistoryBackend, KubernetesDiscoveryConfig, ManualAssignmentMode,
        MemoryConfig, MetricsConfig, OracleConfig, ParserChoice, PdPairingMode, PolicyConfig,
        PostgresConfig, PowerOfTwoLoadMetric, RedisConfig, RetryConfig, RouterConfig,
        RoutingKeyOverrideConfig, RoutingMode, SchemaConfig, TenantApiKeyEntry,
        TokenizerCacheConfig, TraceConfig, DEFAULT_FILE_DISCOVERY_CHECK_INTERVAL_SECS,
    },
    mesh_discovery::MeshDiscoveryConfig,
    observability::metrics::{register_jemalloc_as_global_allocator, PrometheusConfig},
    policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY,
    posture::{SecretSource, SecretSources},
    server::{self, ServerConfig},
    service_discovery::{ModelIdSource, RuntimeDiscoveryConfig},
    version,
    worker::{ConnectionMode, RuntimeType},
};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, JwtConfig, Role};
use smg_mesh::{MTLSConfig, MeshServerConfig};

/// The command line as the gateway reads it, before clap sees it: the
/// `--prefill <url> [bootstrap_port|none]` and `--encode <url>
/// [bootstrap_port|none]` occurrences are hand-parsed (clap cannot express
/// the optional trailing port) and stripped from the arguments clap parses.
struct CommandLine {
    /// argv without the `--prefill` / `--encode` occurrences and their values.
    clap_args: Vec<String>,
    prefill_urls: Vec<(String, Option<u16>)>,
    encode_urls: Vec<(String, Option<u16>)>,
}

impl CommandLine {
    /// Split `args` (argv, the program name first). Each url list is what
    /// [`parse_url_port_args`] reads for its flag over the whole of `args`,
    /// independently of the other flag; `clap_args` is `args` with every
    /// `<flag> <url> [port|none]` run removed by the same token rule, so a
    /// flag given without a url (last on the line) reaches clap, which
    /// refuses it as unexpected.
    fn split(args: &[String]) -> Self {
        Self {
            clap_args: strip_url_port_args(args),
            prefill_urls: parse_url_port_args(args, "--prefill"),
            encode_urls: parse_url_port_args(args, "--encode"),
        }
    }
}

/// Whether the token after a `--prefill` / `--encode` url is its optional
/// bootstrap port: a port number or `none` (any case). Another flag, or a
/// url, is not consumed.
fn is_bootstrap_port_token(token: &str) -> bool {
    !token.starts_with("--") && (token.parse::<u16>().is_ok() || token.to_lowercase() == "none")
}

/// Parse the repeated `<flag> <url> [bootstrap_port|none]` occurrences of
/// `args` into (url, optional bootstrap port) pairs; `none` and an absent
/// port both read as `None`.
fn parse_url_port_args(args: &[String], flag: &str) -> Vec<(String, Option<u16>)> {
    let mut entries = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            let url = args[i + 1].clone();
            let mut bootstrap_port = None;
            if args
                .get(i + 2)
                .is_some_and(|token| is_bootstrap_port_token(token))
            {
                bootstrap_port = args[i + 2].parse::<u16>().ok();
                i += 1;
            }
            entries.push((url, bootstrap_port));
            i += 2;
        } else {
            i += 1;
        }
    }
    entries
}

/// `args` without the `--prefill` / `--encode` occurrences and their url and
/// optional port, by the token rule of [`parse_url_port_args`].
fn strip_url_port_args(args: &[String]) -> Vec<String> {
    let mut kept = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if (args[i] == "--prefill" || args[i] == "--encode") && i + 1 < args.len() {
            i += 2;
            if args
                .get(i)
                .is_some_and(|token| is_bootstrap_port_token(token))
            {
                i += 1;
            }
        } else {
            kept.push(args[i].clone());
            i += 1;
        }
    }
    kept
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

/// A worker discovery provider, as `--discovery-provider` names it.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum DiscoveryProvider {
    #[value(name = "kubernetes")]
    Kubernetes,
    #[value(name = "file")]
    File,
}

impl DiscoveryProvider {
    /// The spelling `--discovery-provider` takes for this provider.
    fn flag_value(self) -> &'static str {
        match self {
            Self::Kubernetes => "kubernetes",
            Self::File => "file",
        }
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

/// Parse the deprecated `--kv-index` value. The chain index is the only KV
/// index, so `chain` is the one value accepted; `positional` names the index
/// that went with kv-index 2.0.0 and is refused with the reason.
fn parse_kv_index(value: &str) -> Result<String, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "chain" => Ok("chain".to_string()),
        "positional" => Err(
            "the positional KV index was removed (kv-index 2.0.0): the chain index is the \
             only KV index, drop --kv-index"
                .to_string(),
        ),
        _ => Err(format!(
            "invalid value '{value}'; the chain index is the only KV index, so --kv-index \
             takes chain only (and is deprecated)"
        )),
    }
}

/// Parse the `--power-of-two-load-metric` value into a `PowerOfTwoLoadMetric`.
fn parse_power_of_two_load_metric(value: &str) -> Result<PowerOfTwoLoadMetric, String> {
    PowerOfTwoLoadMetric::parse(value).ok_or_else(|| {
        format!("invalid value '{value}'; expected expected_wait, requests, or least_requests")
    })
}

/// Parse the `--mm-processing` value into an `MmProcessingMode`.
fn parse_mm_processing(value: &str) -> Result<MmProcessingMode, String> {
    MmProcessingMode::parse(value)
        .ok_or_else(|| format!("invalid value '{value}'; expected auto, router, or worker"))
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

// clap renders `--help` by `help_heading`: the headings in first-use order,
// the fields of a heading in declaration order. The sections below follow
// that grouping, one header per heading, so a field is added under its
// heading's header; the Mesh Server section is the exception, its flags
// carry no heading and `--help` lists them under Options. A field's doc
// comment is its help text. The second
// spellings, each a `visible_alias` of its field and mirrored by the Python
// launcher's `RouterArgs`: --cache-match-threshold (--cache-threshold),
// --spill-abs-threshold (--balance-abs-threshold), --spill-rel-threshold
// (--balance-rel-threshold), --sticky-key-idle-secs (--max-idle-secs),
// --sticky-sessions (--routing-key-override) and --worker-auto-recovery
// (--remove-unhealthy-workers); the hidden ones: --model (--model-path),
// --runtime (--backend), and `start` for the `launch` subcommand.
// Keep clap's generated construction functions on sibling stack frames. In
// debug builds, deriving all flags together exceeds a 2 MiB thread stack.
// Add flags to their responsibility's group so construction stays bounded;
// flatten and declaration order preserve the existing CLI and help layout.
#[derive(Parser, Debug)]
#[group(skip)]
struct CliArgs {
    #[command(flatten)]
    routing: RoutingArgs,
    #[command(flatten)]
    workers: WorkerArgs,
    #[command(flatten)]
    discovery: DiscoveryArgs,
    #[command(flatten)]
    serving: ServingArgs,
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    integrations: IntegrationArgs,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct RoutingArgs {
    // ==================== Worker Configuration ====================
    /// Host address to bind the router server: an IP literal, IPv6 with or
    /// without brackets (`0.0.0.0`, `::`, `[::]`, `[fd00::1]`)
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

    /// Port of the admin listener. When set, the control-plane routes (the
    /// admin routes /flush_cache, /start_profile, /stop_profile,
    /// /heap_profile, /get_loads, /parse/*, /wasm and /v1/tokenizers, the
    /// worker-registry routes /workers and the RL routes /v1/rl) are served
    /// there, with /health, /liveness and /readiness, and the data listener
    /// (--port) answers 404 for them: the control plane can sit on a private
    /// address, behind its own network policy, off the data port's ingress.
    /// Control-plane authentication applies on it as it does on the data
    /// listener. The public /loads stays on the data listener (a gateway
    /// registered as a worker is polled there) and names the backends as
    /// before. Each listener needs its own port. Unset = the control plane
    /// stays on the data listener
    #[arg(long, help_heading = "Worker Configuration")]
    admin_port: Option<u16>,

    /// Bind host of the admin listener (needs --admin-port). Defaults to
    /// --host
    #[arg(long, help_heading = "Worker Configuration")]
    admin_host: Option<String>,

    /// List of worker URLs (supports IPv4 and IPv6)
    #[arg(long, num_args = 0.., help_heading = "Worker Configuration")]
    worker_urls: Vec<String>,

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
    /// Negotiated per worker at registration: a worker that does not answer
    /// HTTP/2 stays on HTTP/1.1, so mixed fleets roll out in any order.
    /// `http_pool.http2` on a worker spec pins the version instead.
    #[arg(long, default_value_t = false, help_heading = "Worker Configuration")]
    upstream_http2: bool,

    // ==================== Routing Policy ====================
    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "passthrough", "cache_aware", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "Routing Policy")]
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
    /// many requests AND by --balance-rel-threshold. The margin is capped at
    /// the in-flight count's natural swing at the pool's current mean (two
    /// standard deviations, never under four requests unless this value is
    /// lower), so a small pool can spill at all. With the bucket policy the
    /// pair keeps its own rule: balancing when (max_load - min_load) exceeds
    /// this value and max_load exceeds min_load times --balance-rel-threshold
    #[arg(
        long,
        visible_alias = "spill-abs-threshold",
        default_value_t = 64,
        help_heading = "Routing Policy"
    )]
    balance_abs_threshold: usize,

    /// Spill gate, relative part (multiple of the healthy-fleet mean; finite,
    /// at least 1.0); fires only together with --balance-abs-threshold
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
    /// report token_usage. Finite and > 0.0; >= 1.0 disables it.
    #[arg(long, default_value_t = 1.0, help_heading = "Routing Policy")]
    balance_token_usage_threshold: f32,

    /// Safety valve for critically-saturated engines: when the hottest
    /// backend's KV utilization (0.0-1.0) exceeds this, shed load off it
    /// regardless of spread. Best set high (e.g. 0.9). Finite and > 0.0;
    /// >= 1.0 disables it.
    #[arg(long, default_value_t = 1.0, help_heading = "Routing Policy")]
    overload_token_usage_threshold: f32,

    /// Worker overload protection (the default; kept so existing command
    /// lines still parse). A worker whose load report is at or above
    /// --worker-overload-waiting-requests or --worker-overload-token-usage is
    /// left out of routing while another worker is under them; when every
    /// worker is over them the request goes to the least-loaded one, unless
    /// --worker-overload-shed asks for a 503 instead. Evaluated once per load
    /// report, never per request. Per-worker `overload` blocks on a WorkerSpec
    /// override the gateway values per signal
    #[arg(long, default_value_t = true, help_heading = "Routing Policy")]
    worker_overload_protection: bool,

    /// Switch worker overload protection off: no worker is ever left out of
    /// routing for its waiting queue or KV usage (per-worker `overload` blocks
    /// on a WorkerSpec still apply)
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    disable_worker_overload_protection: bool,

    /// Queued (waiting) requests, summed across DP ranks, at or above which a
    /// worker counts as overloaded. Must be >= 1: the comparison is inclusive,
    /// so 0 would veto every worker unconditionally
    #[arg(long, default_value_t = 8, value_parser = parse_positive_usize, help_heading = "Routing Policy")]
    worker_overload_waiting_requests: usize,

    /// Mean KV-cache token usage across DP ranks at or above which a worker
    /// counts as overloaded: the same signal --balance-token-usage-threshold
    /// reads, applied as an absolute per-worker ceiling rather than a
    /// fleet-relative spread. Backend must report token_usage. Must be in
    /// (0.0, 1.0]: the comparison is inclusive, so 0.0 would veto every worker
    /// unconditionally. Distinct from --overload-token-usage-threshold, which
    /// only de-ranks the hottest backend within cache-aware affinity
    #[arg(long, default_value_t = 0.8, value_parser = parse_unit_fraction, help_heading = "Routing Policy")]
    worker_overload_token_usage: f64,

    /// Refuse a request with a 503 (worker_overload_protection_shed,
    /// Retry-After the load poll interval) when every worker it could use is
    /// overloaded, instead of routing it to the least-loaded one; also sheds a
    /// worker that crossed a threshold between selection and dispatch. Off by
    /// default: steering never turns a load signal into an outage
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    worker_overload_shed: bool,

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

    /// Worker selection policy for cache_aware, run over the per-worker
    /// inputs the router gathers (prefix overlap, in-flight requests,
    /// backend load reports); cache-aware-default, the affinity-group
    /// decision, is the one policy
    #[arg(
        long,
        default_value = "cache-aware-default",
        help_heading = "Routing Policy"
    )]
    selection_policy: String,

    /// Lifetime in milliseconds of optimistic dispatch bookings for
    /// cache_aware: predicted prefill and prefix placement are charged to the
    /// chosen worker until the engine reports them or the booking expires.
    /// 0 disables; set a little above the engine's KV-event lag
    #[arg(long, default_value_t = 0, help_heading = "Routing Policy")]
    selection_accounting_ttl_ms: u64,

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

    /// Load factor threshold for prefix_hash policy (finite, at least 1.0)
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

    /// What power_of_two compares: expected_wait (default), requests or least_requests
    #[arg(long, default_value = "expected_wait", value_parser = parse_power_of_two_load_metric, help_heading = "Routing Policy")]
    power_of_two_load_metric: PowerOfTwoLoadMetric,

    /// Enable data parallelism aware scheduling
    #[arg(long, default_value_t = false, help_heading = "Routing Policy")]
    dp_aware: bool,

    /// Sticky sessions: route every request of a conversation to the same
    /// worker, on any policy. Valid routing-key headers take priority over
    /// a key derived from the request body's rid, with per-turn/per-retry
    /// suffixes stripped (conv_t2_r1 -> conv). Enabling this keeps automatic
    /// body forwarding buffered so the body rid remains available as fallback.
    /// Reuses the manual eviction/idle/assignment knobs for the sticky map
    #[arg(
        long,
        visible_alias = "sticky-sessions",
        default_value_t = false,
        help_heading = "Routing Policy"
    )]
    routing_key_override: bool,

    /// How strictly PD placement pairs a prefill with a decode on their KV
    /// transfer protocol. `lenient` refuses only a known difference in
    /// runtime, transport or KV layout (unknown components and engine
    /// versions pair with anything); `strict` also refuses unknown
    /// components and version differences; `off` pairs on nothing.
    #[arg(
        long,
        value_parser = ["off", "lenient", "strict"],
        default_value = "lenient",
        help_heading = "Routing Policy"
    )]
    pd_pairing_mode: String,

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

    /// Warm-up slice for cache-aware routing: for this many seconds after a
    /// worker becomes routable, until its index has grown by
    /// --worker-warmup-blocks blocks, one cache miss in 1/--worker-warmup-share
    /// is routed to it so it builds a cache instead of idling. 0 disables.
    #[arg(long, default_value_t = 60, help_heading = "Routing Policy")]
    worker_warmup_secs: u64,

    /// Share of cache misses offered to warming workers (a finite value from
    /// 0.0 to 1.0).
    #[arg(long, default_value_t = 0.25, help_heading = "Routing Policy")]
    worker_warmup_share: f32,

    /// A worker whose index has grown by this many blocks since it became
    /// thin is warm.
    #[arg(long, default_value_t = 1024, help_heading = "Routing Policy")]
    worker_warmup_blocks: usize,

    /// A worker whose index holds less than this share of the fleet's median
    /// (or nothing) is thin and receives the warm-up slice until it has grown
    /// by --worker-warmup-blocks, whatever emptied it (a resync after a
    /// publisher restart, an out-of-range or data-loss resubscription, an
    /// engine that came back empty). A finite value from 0.0 to 1.0; 0 keeps
    /// the age rule alone.
    #[arg(long, default_value_t = 0.5, help_heading = "Routing Policy")]
    worker_warmup_thin_ratio: f32,

    /// One cache hit in this many is diverted to a thin worker although
    /// another worker holds its prefix (shallow overlaps first, one in flight
    /// per thin worker), so an index emptied by a resync refills on a
    /// workload where every request has a holder. 0 disables.
    #[arg(long, default_value_t = 8, help_heading = "Routing Policy")]
    worker_warmup_divert_every: u64,

    /// Deprecated and ignored: the TTL prune of the positional indexer, which
    /// the chain index replaced. The chain index holds what the engines report
    /// and shrinks with their removals, so it has no prune; the flag is still
    /// accepted so that existing launch lines keep working, and goes in a
    /// later release.
    #[arg(long, help_heading = "Routing Policy")]
    kv_indexer_ttl_secs: Option<u64>,

    /// Deprecated and ignored: the capacity ceiling of the positional
    /// indexer's prune; see --kv-indexer-ttl-secs.
    #[arg(long, help_heading = "Routing Policy")]
    kv_indexer_max_entries: Option<usize>,

    /// Deprecated: the chain index is the only KV index behind cache-aware
    /// routing, so this takes `chain` only and does nothing; the flag goes in
    /// a later release. `positional` names the index that was removed and is
    /// refused.
    #[arg(long, value_parser = parse_kv_index, help_heading = "Routing Policy")]
    kv_index: Option<String>,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct WorkerArgs {
    // ==================== RL Control Plane ====================
    /// Mount the RL control plane under /v1/rl (worker discovery,
    /// engine-route passthrough, fan-out). Off by default; when off, no RL
    /// code path is reachable.
    #[arg(long, default_value_t = false, help_heading = "RL Control Plane")]
    enable_rl: bool,

    /// Total timeout for one proxied engine control call (weight refits can
    /// take minutes)
    #[arg(long, default_value_t = 600, help_heading = "RL Control Plane")]
    rl_control_timeout_secs: u64,

    /// Maximum concurrent engine calls in one fan-out
    #[arg(long, default_value_t = 32, help_heading = "RL Control Plane")]
    rl_fanout_concurrency: usize,

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
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "PD Disaggregation")]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two", "least_load", "prefix_hash", "consistent_hashing", "manual", "bucket"], help_heading = "PD Disaggregation")]
    decode_policy: Option<String>,

    /// Maximum in-flight Prefill requests per worker in PD or EPD mode (non-positive to disable)
    #[arg(long, default_value_t = -1, allow_negative_numbers = true, help_heading = "PD Disaggregation")]
    prefill_max_inflight_requests_per_worker: i32,

    /// Maximum number of requests waiting for Prefill admission (default: 100 when enabled)
    #[arg(long, help_heading = "PD Disaggregation")]
    prefill_queue_size: Option<usize>,

    /// Maximum time in seconds a request may wait for Prefill admission (default: 60 when enabled);
    /// in the gRPC router the wait is also held to --queue-timeout-secs, counted from admission
    #[arg(long, help_heading = "PD Disaggregation")]
    prefill_queue_timeout_secs: Option<u64>,

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

    // ==================== Load Monitoring ====================
    /// Interval in seconds between load monitor checks for PowerOfTwo routing
    #[arg(long, default_value_t = 10, help_heading = "Load Monitoring")]
    load_monitor_interval: u64,

    /// Seconds without any contact from a worker (a load poll, a health probe,
    /// a KV event, a response) after which a transport failure excludes it
    /// from routing; the first successful contact re-admits it.
    #[arg(long, default_value_t = 2, help_heading = "Load Monitoring")]
    worker_stall_secs: u64,

    /// Seconds without a token or a completion from a worker with requests
    /// in flight whose waiting queue grows, or whose in-flight pile grows or
    /// is four deep, after which new requests stop being routed to it until
    /// it makes progress. The bound stretches to the time its in-flight
    /// prompts may still need in prefill, up to 120 seconds. The pile is the
    /// streaming gRPC generations in flight; HTTP workers, PD legs and
    /// non-streaming generations give no signal and never form one. The
    /// same bound, read per request, is how long a prefill leg of a
    /// disaggregated dispatch may produce nothing, with its worker answering
    /// no other leg either, before its request fails over to another
    /// prefill worker (or fails, when there is none); a leg queued on a
    /// worker that keeps answering waits on. On the parallel dispatch shape
    /// a prefill leg's first response also waits for the decode side (its
    /// bootstrap, and KV pre-allocation under decode pressure), so the
    /// threshold must cover that wait for a prefill worker with nothing
    /// else to answer meanwhile. 0 disables both.
    #[arg(long, default_value_t = 3, help_heading = "Load Monitoring")]
    worker_wedge_secs: u64,

    /// Seconds without any contact (a load record, a poll answer, a probe, a
    /// token) from a worker whose KV-event stream pushes load records, after
    /// which it is excluded from routing as unreachable until it is heard
    /// from again. A reachable servicer pushes a record at least every few
    /// seconds, so a longer silence is a dead or one-way link the transport
    /// has not reported yet. Workers that never pushed a record (HTTP
    /// workers, older servicers) are not judged by it. Keep it above the
    /// load-monitor interval. 0 disables the rule.
    #[arg(long, default_value_t = 15, help_heading = "Load Monitoring")]
    worker_stale_secs: u64,

    /// Only poll worker loads when a load-aware routing policy,
    /// --engine-metrics, or worker overload protection needs the data. By
    /// default every worker group is polled from registration onward; this
    /// restores the old conditional gate (a load-aware policy is always fed
    /// regardless).
    #[arg(long, default_value_t = false, help_heading = "Load Monitoring")]
    disable_load_monitoring: bool,

    /// Force GetLoads polling for smg_engine_* Prometheus gauges even without
    /// a load-aware routing policy. Routing-owned polls are always re-exported.
    /// A worker whose KV-event stream pushes its load feeds the gauges from
    /// those records and is not polled while they flow (GetLoads is the
    /// fallback).
    #[arg(long, default_value_t = false, help_heading = "Load Monitoring")]
    engine_metrics: bool,

    /// Seconds a prefill/decode dispatch waits for a free slot in the decode
    /// engine's running window (--max-num-seqs / --max-running-requests)
    /// before shedding with 503 worker_overload_protection_shed. Keep it well
    /// under the engine's bootstrap deadline (120s on TokenSpeed) so a
    /// request that waits still dispatches with the deadline ahead of it. 0
    /// sheds immediately. Engines that report no running window are never
    /// gated
    #[arg(long, default_value_t = 30, help_heading = "Load Monitoring")]
    pd_admission_wait_secs: u64,

    // ==================== Multimodal ====================
    /// Multimodal tensor transport mode: `inline` (default), `shm` (same-host
    /// /dev/shm), or `auto` (shm only when the worker shares /dev/shm). A
    /// per-worker `WorkerSpec.multimodal_tensor_transport` overrides this.
    #[arg(long, value_parser = parse_transport_mode, help_heading = "Multimodal")]
    multimodal_tensor_transport: Option<TransportMode>,

    /// Minimum multimodal tensor size (bytes) before the SHM transport is used.
    /// Overridable per worker via `WorkerSpec.multimodal_shm_min_bytes`.
    #[arg(long, help_heading = "Multimodal")]
    multimodal_shm_min_bytes: Option<usize>,

    /// Most bytes of preprocessed media the gateway holds in flight for engines
    /// at once; a request that fits waits briefly, then gets 429, and one
    /// larger than the whole budget gets 413 straight away. A waiting request
    /// still holds its media, so size memory for about twice this value.
    /// Unset: unbounded. Counted in whole KiB: a value under 1024 (one 1 KiB
    /// unit, zero included) is refused rather than read as unset.
    #[arg(long, help_heading = "Multimodal")]
    multimodal_max_inflight_bytes: Option<usize>,

    /// Per-request image-count limit applied to every model, replacing each
    /// spec's built-in limit. It tightens, never loosens, the limit a model's
    /// workers advertise for their engine (`mm_item_limits`, e.g. vLLM's
    /// `--limit-mm-per-prompt`), which the router holds requests to by itself.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..), help_heading = "Multimodal")]
    mm_per_request_image_limit: Option<u64>,

    /// Where media for vLLM gRPC workers is fetched and preprocessed:
    /// `auto` (worker when the model spec and the worker both allow it,
    /// else router), `router` (always the gateway), `worker` (the engine,
    /// except that a request whose items carry per-item hints
    /// (max_long_side_pixel, fps) is preprocessed by the gateway, which
    /// applies them; models without worker expansion are rejected). Falls
    /// back to `SMG_MM_PROCESSING`, then `auto`.
    #[arg(long, value_parser = parse_mm_processing, help_heading = "Multimodal")]
    mm_processing: Option<MmProcessingMode>,

    /// Pixel cache budget in MiB for router-side preprocessed media; 0 keeps
    /// the cache off. Falls back to `SMG_MM_PIXEL_CACHE_MB`.
    #[arg(long, help_heading = "Multimodal")]
    mm_pixel_cache_mb: Option<usize>,

    /// Serve cached pixels over RDMA instead of inline bytes (the legacy
    /// switch; `--multimodal-tensor-transport rdma` is the first-class one).
    /// Falls back to `SMG_MM_PIXEL_RDMA`.
    #[arg(long, action = ArgAction::SetTrue, help_heading = "Multimodal")]
    mm_pixel_rdma: bool,

    /// Hosts the gateway fetches client-supplied media URLs from when it
    /// processes media itself, by name, comma-separated; a redirect to any
    /// other host is refused. Unset allows any host the address policy
    /// passes (public addresses, or private ones with
    /// `--mm-allow-private-media-hosts`).
    #[arg(long, value_delimiter = ',', help_heading = "Multimodal")]
    mm_allowed_media_domains: Option<Vec<String>>,

    /// Allow client-supplied media URLs that name or resolve to a loopback,
    /// link-local or private address, for a deployment whose media origin is
    /// internal; by default such URLs are refused before anything is
    /// connected. Falls back to `SMG_MM_ALLOW_PRIVATE_MEDIA_HOSTS`.
    #[arg(long, action = ArgAction::SetTrue, help_heading = "Multimodal")]
    mm_allow_private_media_hosts: bool,

    /// Listener IP for the RDMA pixel lane's metadata exchange, a plain IPv4
    /// or IPv6 address (no hostname, port or brackets: the workers connect to
    /// it as written); without one the lane stays on the inline path. Falls
    /// back to `SMG_RDMA_LISTEN_IP`.
    #[arg(long, help_heading = "Multimodal")]
    rdma_listen_ip: Option<String>,

    /// Seconds a leased RDMA pixel slot lives without a free notification (at
    /// least 1); must exceed the worker's hold or it is ignored for the derived
    /// TTL.
    /// Falls back to `SMG_RDMA_SLOT_TTL_S`.
    #[arg(long, help_heading = "Multimodal")]
    rdma_slot_ttl_s: Option<u64>,

    /// Emit per-request multimodal timing at INFO. Falls back to
    /// `SMG_LOG_MM_TIMING`.
    #[arg(long, action = ArgAction::SetTrue, help_heading = "Multimodal")]
    log_mm_timing: bool,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct DiscoveryArgs {
    // ==================== Service Discovery (Kubernetes) ====================
    /// Enable Kubernetes service discovery (the legacy spelling of
    /// `--discovery-provider kubernetes`)
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

    /// Pod annotation containing the vLLM KV connector name
    #[arg(
        long,
        default_value = "smg.ai/kv-connector",
        help_heading = "Service Discovery (Kubernetes)"
    )]
    kv_connector_annotation: String,

    /// Pod annotation containing per-worker KV engine IDs
    #[arg(
        long,
        default_value = "smg.ai/kv-engine-id",
        help_heading = "Service Discovery (Kubernetes)"
    )]
    kv_engine_id_annotation: String,

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
    /// Matching is case-sensitive. A vendor-named alias (kimi-*, minimax-*,
    /// ...) also gives the served model that vendor's contract profile under
    /// both names.
    #[arg(long = "model-alias", action = ArgAction::Append, value_parser = parse_model_alias, help_heading = "Service Discovery (Kubernetes)")]
    model_alias: Vec<String>,

    // ==================== Service Discovery ====================
    /// Worker discovery provider. Give this or `--service-discovery`, not both
    #[arg(
        long,
        value_enum,
        conflicts_with = "service_discovery",
        help_heading = "Service Discovery"
    )]
    discovery_provider: Option<DiscoveryProvider>,

    // ==================== Service Discovery (File) ====================
    /// JSON manifest listing the workers, for `--discovery-provider file`. It
    /// may be created after startup
    #[arg(long, help_heading = "Service Discovery (File)")]
    discovery_file: Option<PathBuf>,

    /// Seconds between periodic rereads of the `--discovery-file` manifest,
    /// which is also reread as soon as it changes (default: 30)
    #[arg(long, help_heading = "Service Discovery (File)")]
    discovery_check_interval_secs: Option<u64>,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct ServingArgs {
    // ==================== Logging ====================
    /// Directory for the log files, created at start-up when missing (under
    /// a writable directory); an existing path must be a writable directory
    /// (unset: console only)
    #[arg(long, help_heading = "Logging")]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"], help_heading = "Logging")]
    log_level: String,

    /// Output logs as JSON
    #[arg(long, default_value_t = false, help_heading = "Logging")]
    log_json: bool,

    // ==================== Prometheus Metrics ====================
    /// Port to expose Prometheus metrics (at least 1). Each listener (--port,
    /// --prometheus-port, --health-check-port, --mesh-port) needs its own
    /// port.
    #[arg(long, default_value_t = 29000, help_heading = "Prometheus Metrics")]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server. Defaults to the
    /// unspecified address of `--host`'s family: `::` for an IPv6 host,
    /// `0.0.0.0` otherwise.
    #[arg(long, help_heading = "Prometheus Metrics")]
    prometheus_host: Option<String>,

    /// Custom buckets for Prometheus duration metrics
    #[arg(long, num_args = 0.., help_heading = "Prometheus Metrics")]
    prometheus_duration_buckets: Vec<f64>,

    // ==================== Profiling ====================
    /// Directory where `POST /heap_profile` (an admin route) writes a jemalloc
    /// heap profile of this gateway. Needs a build with the `jemalloc-profiling`
    /// feature, started with `_RJEM_MALLOC_CONF=prof:true,prof_active:true`;
    /// unset, the route answers 404
    #[arg(long, help_heading = "Profiling")]
    jemalloc_prof_dir: Option<String>,

    // ==================== Request Handling ====================
    /// Custom HTTP headers to check for request IDs. The first one present
    /// is honoured when its value is 1 to 128 visible-ASCII characters;
    /// otherwise a fresh id is generated
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    request_id_headers: Vec<String>,

    /// Map HTTP headers into storage hook request context (format: header=context_key)
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    storage_context_headers: Vec<String>,

    /// Trust an upstream-provided tenant header for canonical tenant resolution.
    /// Stored responses and conversations are scoped to the tenant that created
    /// them: another tenant's id answers 404; records stored without a tenant
    /// identity stay readable by every caller.
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

    /// Seconds of upstream silence after which an open SSE stream carries a
    /// keep-alive frame (an SSE comment on the OpenAI-shaped endpoints, the
    /// `ping` event on /v1/messages), so an idle-timeout proxy does not cut
    /// a stream that is waiting on a long prefill. Nothing is sent while
    /// frames flow. 0 sends none.
    #[arg(long, default_value_t = 15, help_heading = "Request Handling")]
    sse_keepalive_secs: u64,

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

    /// Most bytes the router may hold for a request it buffers only to keep
    /// it retryable. The router decides per request: a request it must parse
    /// (text-routing policy without a routing hint, body-mutating worker,
    /// WASM request hooks, missing Content-Length, PD/batch backends) always
    /// buffers, bounded by max-payload-size; an eligible request buffers up
    /// to this many bytes when router retries are enabled, and otherwise
    /// streams to the worker verbatim — forfeiting router-level retries,
    /// with JSON validation deferring to the worker. 0 never buffers for
    /// retries
    #[arg(long, default_value_t = 1_048_576, help_heading = "Request Handling")]
    max_buffered_request_bytes: u64,

    /// Abort a streamed request body once the upstream sender has waited on
    /// the client for this many seconds (408, request_body_stalled). The
    /// clock pauses while the worker applies backpressure, so a slow worker
    /// read never trips it. Applies only to streamed request bodies. 0
    /// disables
    #[arg(long, default_value_t = 300, help_heading = "Request Handling")]
    stream_body_stall_timeout_secs: u64,

    /// Close a connection that has not sent a complete request head within
    /// this many seconds of being accepted (and a keep-alive connection whose
    /// next request head stalls for this long after its first byte). 0
    /// disables
    #[arg(long, default_value_t = 30, help_heading = "Request Handling")]
    header_read_timeout_secs: u64,

    /// Close a keep-alive connection that has had no request in flight for
    /// this many seconds; an open streaming response counts as in flight. The
    /// bound starts with the connection's first request (a connection that
    /// never sends one is bounded by --header-read-timeout-secs alone). Keep
    /// it above the idle timeout of any proxy or load balancer in front of the
    /// gateway, so that side closes an idle connection first; otherwise a
    /// request can be sent on a connection the gateway is closing and fail
    /// with a reset. A connection the proxy opened ahead of time and never
    /// used is bounded by --header-read-timeout-secs instead: keep the
    /// proxy's idle timeout under that too, or do not pre-open. 0 keeps idle
    /// connections open until the client closes them
    #[arg(long, default_value_t = 90, help_heading = "Request Handling")]
    idle_connection_timeout_secs: u64,

    /// Origins a browser may call the gateway from (exact origins,
    /// space-separated), or `*` alone for any origin. Without this flag the
    /// gateway makes no cross-origin allowance
    #[arg(long, num_args = 0.., help_heading = "Request Handling")]
    cors_allowed_origins: Vec<String>,

    // ==================== Rate Limiting ====================
    /// Maximum standing concurrent requests. Each admission permit is held
    /// for the full response, including streaming bodies. -1 (default)
    /// derives the bound from the host (1024 per available core, at least
    /// 4096) so an overloaded router sheds with 429/503 instead of queueing
    /// without bound; 0 disables the bound.
    #[arg(long, default_value_t = -1, allow_negative_numbers = true, help_heading = "Rate Limiting")]
    max_concurrent_requests: i32,

    /// Queue size for pending requests when limit reached
    #[arg(long, default_value_t = 100, help_heading = "Rate Limiting")]
    queue_size: usize,

    /// Maximum time in seconds a request can wait in queue, and the longest a
    /// request may sit inside the gRPC router ahead of worker selection (counted
    /// from its admission, or its acceptance without an admission layer) before
    /// it is refused with a 503, the prefill admission wait of a PD or EPD request
    /// included; the HTTP relay and the HTTP PD router are not bounded
    #[arg(long, default_value_t = 60, help_heading = "Rate Limiting")]
    queue_timeout_secs: u64,

    /// Sustained admission rate in requests per second, bursting up to
    /// --max-concurrent-requests, which keeps bounding standing concurrency
    /// (a completed request frees its slot, not rate budget). Unset or 0 =
    /// no rate limit: --max-concurrent-requests bounds concurrency alone.
    #[arg(long, allow_negative_numbers = true, help_heading = "Rate Limiting")]
    rate_limit_tokens_per_second: Option<i32>,

    /// Most Responses API requests with background: true in flight at once;
    /// one beyond answers 429. 0 switches background mode off: such requests
    /// then run in the foreground.
    #[arg(long, default_value_t = 1024, help_heading = "Rate Limiting")]
    max_background_responses: usize,

    // ==================== Priority Scheduler ====================
    /// Enable the priority-aware admission scheduler. When unset (default),
    /// the legacy concurrency-limit middleware stays wired.
    #[arg(long, help_heading = "Priority Scheduler")]
    priority_scheduler_enabled: bool,

    /// Max priority class for tenants not listed in the scheduler YAML
    /// (system | interactive | default | bulk).
    #[arg(long, default_value = "default", help_heading = "Priority Scheduler")]
    priority_scheduler_default_max_class: String,

    /// Optional path to the priority-scheduler YAML config. With
    /// --priority-scheduler-enabled, a file that cannot be read, parsed or
    /// validated is a start-up error; without a path the built-in class
    /// defaults apply.
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

    /// Multiplier for exponential backoff (finite, at least 1.0)
    #[arg(long, default_value_t = 1.5, help_heading = "Retry Configuration")]
    retry_backoff_multiplier: f32,

    /// Jitter factor (0.0-1.0) for retry delays
    #[arg(long, default_value_t = 0.2, help_heading = "Retry Configuration")]
    retry_jitter_factor: f32,

    /// Disable automatic retries
    #[arg(long, default_value_t = false, help_heading = "Retry Configuration")]
    disable_retries: bool,

    // ==================== Circuit Breaker ====================
    /// Number of failed requests before the circuit opens (a request's retries
    /// on a worker count once)
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
    /// (at least 1)
    #[arg(long, default_value_t = 3, help_heading = "Health Checks")]
    health_failure_threshold: u32,

    /// Consecutive probe successes before a worker returns to rotation (at
    /// least 1)
    #[arg(long, default_value_t = 2, help_heading = "Health Checks")]
    health_success_threshold: u32,

    /// Timeout in seconds for a single health probe (at least 1)
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    health_check_timeout_secs: u64,

    /// Interval in seconds between health probes of each worker (at least
    /// 1; --disable-health-check turns probing off)
    #[arg(long, default_value_t = 60, help_heading = "Health Checks")]
    health_check_interval_secs: u64,

    /// HTTP path probed on each worker
    #[arg(long, default_value = "/health", help_heading = "Health Checks")]
    health_check_endpoint: String,

    /// Disable all worker health probing
    #[arg(long, default_value_t = false, help_heading = "Health Checks")]
    disable_health_check: bool,

    /// Recover failed workers by removal: a worker that stays unhealthy
    /// long enough (Failed, ~12 minutes at the default thresholds) is
    /// removed from the registry so service discovery re-registers and
    /// re-probes it once its engine returns. Without this a Failed worker
    /// stays registered, out of rotation, and keeps being probed, so it
    /// rejoins in place as soon as it answers again. Defaults to the
    /// --service-discovery setting: discovery-managed fleets recover by
    /// removal plus re-registration, while a static fleet has nothing to
    /// re-add a removed worker and recovers in place instead. Pass =false
    /// to keep it off under discovery.
    #[arg(
        long,
        visible_alias = "worker-auto-recovery",
        num_args = 0..=1,
        default_missing_value = "true",
        help_heading = "Health Checks"
    )]
    remove_unhealthy_workers: Option<bool>,

    /// Seconds to keep a Ready worker in `Draining` before removing it from
    /// the registry. Applies to all RemoveWorker submissions (K8s deletion,
    /// `--remove-unhealthy-workers`, manual API). Per-worker overrides are
    /// supported via `WorkerSpec::health.drain_settle_secs`. Set to `0` to
    /// remove immediately without draining.
    #[arg(long, default_value_t = 5, help_heading = "Health Checks")]
    drain_settle_secs: u64,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct ModelArgs {
    // ==================== Tokenizer ====================
    /// Model path for loading tokenizer (HuggingFace ID, fetched lazily, or a
    /// local path: one that exists must hold a tokenizer layout; one that
    /// does not is left to the worker's tokenizer where a gRPC worker can
    /// still register, refused otherwise); not read as a tokenizer source
    /// beside --tokenizer-path
    #[arg(long, alias = "model", help_heading = "Tokenizer")]
    model_path: Option<String>,

    /// Explicit tokenizer path (overrides model_path); a local path follows
    /// the same rule as --model-path
    #[arg(long, help_heading = "Tokenizer")]
    tokenizer_path: Option<String>,

    /// Chat template path: an existing, non-empty template file
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

    /// Maximum memory for L0 tokenizer cache in bytes (texts, ids and per-entry overhead)
    #[arg(long, default_value_t = 268435456, help_heading = "Tokenizer")]
    tokenizer_cache_l0_max_memory: usize,

    /// Enable L1 (prefix matching) tokenizer cache
    #[arg(long, default_value_t = false, help_heading = "Tokenizer")]
    tokenizer_cache_enable_l1: bool,

    /// Maximum memory for L1 tokenizer cache in bytes
    #[arg(long, default_value_t = 52428800, help_heading = "Tokenizer")]
    tokenizer_cache_l1_max_memory: usize,

    // ==================== Parsers ====================
    /// Parser for reasoning models (e.g., deepseek-r1, qwen3). Under --parser
    /// symphony, a Symphony table's name here (and in --tool-call-parser, or
    /// there alone) reads every model with that table; `hf` reads each model
    /// with the table its tokenizer's response_template compiles into; a
    /// legacy parser's name keeps the legacy parsers. A worker's labels set
    /// the names per model.
    #[arg(long, help_heading = "Parsers")]
    reasoning_parser: Option<String>,

    /// Parser for tool-call interactions (e.g., json, qwen). Under --parser
    /// symphony, a Symphony table's name here (and in --reasoning-parser, or
    /// here alone) reads every model with that table; `hf` reads each model
    /// with the table its tokenizer's response_template compiles into; a
    /// legacy parser's name keeps the legacy parsers. A worker's labels set
    /// the names per model.
    #[arg(long, help_heading = "Parsers")]
    tool_call_parser: Option<String>,

    /// Which parser reads a chat completion's reasoning and tool calls: the
    /// per-model parsers (legacy) or Symphony. With Symphony, the parser names
    /// set for a model decide (its worker labels, else --tool-call-parser and
    /// --reasoning-parser): a Symphony table's name reads with that table, a
    /// legacy parser's keeps the legacy parsers, and no name takes Symphony's
    /// registry for the models it knows. A request's X-SMG-Parser header
    /// overrides the flag for that request.
    #[arg(long, default_value = "legacy", value_parser = ["legacy", "symphony"], help_heading = "Parsers")]
    parser: String,

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

    /// Most stored responses the memory history backend keeps; the oldest go
    /// first, on every write (memory backend only; a stored response takes
    /// about 12 KB).
    #[arg(long, default_value_t = 10_000, help_heading = "Backend")]
    history_memory_max_responses: usize,

    /// Most conversation items the memory history backend keeps; the oldest
    /// go first and leave every conversation that listed them (memory backend
    /// only; an item takes about 1.5 KB).
    #[arg(long, default_value_t = 100_000, help_heading = "Backend")]
    history_memory_max_items: usize,

    /// How long the memory history backend keeps a record, in seconds, checked
    /// on every write; 0 keeps a record until a count bound drops it. A
    /// conversation is kept that long after its last use (created, read or
    /// updated); a background response still running is not dropped by a
    /// count bound.
    #[arg(long, default_value_t = 86_400, help_heading = "Backend")]
    history_memory_retention_secs: u64,

    /// Enable WebAssembly support
    #[arg(long, default_value_t = false, help_heading = "Backend")]
    enable_wasm: bool,

    /// Path to a WASM component implementing storage hooks
    #[arg(long, help_heading = "Backend")]
    storage_hook_wasm_path: Option<String>,

    /// Path to a YAML schema config file for storage table/column remapping; `auto_migrate: true` in it applies pending schema migrations at start-up (the environment variable DB_AUTO_MIGRATE=true does the same without a file)
    #[arg(long, help_heading = "Backend")]
    schema_config: Option<String>,
}

#[derive(clap::Args, Debug)]
#[group(skip)]
struct IntegrationArgs {
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
    #[arg(
        long,
        env = "ATP_PASSWORD",
        hide_env_values = true,
        help_heading = "Oracle Database"
    )]
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
    #[arg(
        long,
        env = "ATP_POOL_MIN",
        allow_negative_numbers = true,
        help_heading = "Oracle Database"
    )]
    oracle_pool_min: Option<usize>,

    /// Maximum Oracle connection pool size
    #[arg(
        long,
        env = "ATP_POOL_MAX",
        allow_negative_numbers = true,
        help_heading = "Oracle Database"
    )]
    oracle_pool_max: Option<usize>,

    /// Oracle connection pool timeout in seconds
    #[arg(long, env = "ATP_POOL_TIMEOUT_SECS", help_heading = "Oracle Database")]
    oracle_pool_timeout_secs: Option<u64>,

    // ==================== PostgreSQL Database ====================
    /// PostgreSQL database connection URL (env: SMG_POSTGRES_DB_URL; or a
    /// file holding it, --postgres-db-url-file). The flag beats the
    /// environment; a URL and a URL file together are refused
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_db_url: Option<String>,

    /// File holding the PostgreSQL connection URL alone, such as a mounted
    /// secret; the surrounding whitespace is dropped (env:
    /// SMG_POSTGRES_DB_URL_FILE). Read when the history backend is postgres
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_db_url_file: Option<PathBuf>,

    /// Maximum PostgreSQL connection pool size
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_pool_max_size: Option<usize>,

    /// Bound in seconds on each call to the PostgreSQL store (pool wait, connect and the statement); a call past it is answered 503 storage_unavailable (default: 5)
    #[arg(long, help_heading = "PostgreSQL Database")]
    postgres_op_timeout_secs: Option<u64>,

    // ==================== Redis Database ====================
    /// Redis connection URL (env: SMG_REDIS_URL; or a file holding it,
    /// --redis-url-file). The flag beats the environment; a URL and a URL
    /// file together are refused
    #[arg(long, help_heading = "Redis Database")]
    redis_url: Option<String>,

    /// File holding the Redis connection URL alone, such as a mounted
    /// secret; the surrounding whitespace is dropped (env:
    /// SMG_REDIS_URL_FILE). Read when the history backend is redis
    #[arg(long, help_heading = "Redis Database")]
    redis_url_file: Option<PathBuf>,

    /// Maximum Redis connection pool size
    #[arg(long, help_heading = "Redis Database")]
    redis_pool_max_size: Option<usize>,

    /// Redis data retention in days (at least 1; -1 for persistent; default 30)
    #[arg(long, allow_negative_numbers = true, help_heading = "Redis Database")]
    redis_retention_days: Option<i64>,

    /// Bound in seconds on each call to the Redis store (pool wait, connect and the command; replaces the client library's 500 ms / 1 s defaults); a call past it is answered 503 storage_unavailable (default: 5)
    #[arg(long, help_heading = "Redis Database")]
    redis_op_timeout_secs: Option<u64>,

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
    /// Shared API key the serving routes require and the admin routes fall
    /// back to (env: SMG_API_KEY; or a file holding it, --api-key-file). The
    /// flag beats the environment; a key and a key file together are refused
    #[arg(long, help_heading = "Control Plane Authentication")]
    api_key: Option<String>,

    /// File holding the shared API key alone, such as a mounted secret; the
    /// surrounding whitespace is dropped (env: SMG_API_KEY_FILE)
    #[arg(long, help_heading = "Control Plane Authentication")]
    api_key_file: Option<PathBuf>,

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

    /// API keys for control plane access (format: id:name:role:key; role `admin` reaches every
    /// control plane route, `user` the read-only GET/HEAD routes)
    #[arg(long = "control-plane-api-keys", action = ArgAction::Append, env = "CONTROL_PLANE_API_KEYS", hide_env_values = true, help_heading = "Control Plane Authentication")]
    control_plane_api_keys: Vec<String>,

    /// Disable audit logging for control plane operations
    #[arg(
        long,
        default_value_t = false,
        help_heading = "Control Plane Authentication"
    )]
    disable_audit_logging: bool,

    // ==================== Data Plane Authentication ====================
    /// Per-tenant API keys for serving-path auth (format: tenant_id:key,
    /// repeatable). Layers on top of `--api-key`, each resolving to its own
    /// tenant identity. Stored responses and conversations are scoped to the
    /// tenant that created them: another tenant's id answers 404; records
    /// stored without a tenant identity stay readable by every caller.
    /// Also read from the environment (SMG_TENANT_API_KEYS, entries
    /// separated by ',' or a newline) or a file (--tenant-api-keys-file);
    /// the flag beats the environment, and entries together with a file are
    /// refused.
    #[arg(long = "tenant-api-key", action = ArgAction::Append, help_heading = "Data Plane Authentication")]
    tenant_api_keys: Vec<String>,

    /// File holding the per-tenant API keys, one tenant_id:key per line
    /// (blank lines and lines starting with # are skipped), such as a
    /// mounted secret (env: SMG_TENANT_API_KEYS_FILE)
    #[arg(long, help_heading = "Data Plane Authentication")]
    tenant_api_keys_file: Option<PathBuf>,

    /// Where the secrets above were read from, filled by
    /// `resolve_secret_sources` for the start-up record; not a flag.
    #[arg(skip)]
    secret_sources: SecretSources,

    // ==================== Mesh Server ====================
    #[arg(long, default_value_t = false)]
    enable_mesh: bool,

    #[arg(long)]
    mesh_server_name: Option<String>,

    /// Bind address for the mesh listener: an IP literal, IPv6 with or
    /// without brackets (`::`, `[::]`).
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

    /// The CA certificate (PEM) the mesh peers' certificates chain to. With
    /// `--mesh-tls-cert` and `--mesh-tls-key` the mesh listener serves TLS
    /// and requires peer certificates, and every dial presents this node's.
    #[arg(long, requires_all = ["mesh_tls_cert", "mesh_tls_key"])]
    mesh_tls_ca_cert: Option<String>,

    /// This mesh node's certificate (PEM), with the node's IP as a SAN.
    #[arg(long, requires_all = ["mesh_tls_ca_cert", "mesh_tls_key"])]
    mesh_tls_cert: Option<String>,

    /// This mesh node's private key (PKCS#8 PEM).
    #[arg(long, requires_all = ["mesh_tls_ca_cert", "mesh_tls_cert"])]
    mesh_tls_key: Option<String>,

    // ==================== WebRTC ====================
    /// Bind address for WebRTC UDP sockets (client-facing ICE candidate IP).
    /// Default: the unspecified address of `--host`'s family (`::`, dual-stack
    /// on Linux, for an IPv6 host; `0.0.0.0` otherwise), the candidate IP
    /// auto-detected via the routing table.
    /// Set to 127.0.0.1 (or ::1) for local development on the same machine.
    #[arg(long, help_heading = "WebRTC")]
    webrtc_bind_addr: Option<std::net::IpAddr>,

    /// STUN server for ICE candidate gathering (host:port).
    /// Set to your own STUN server for enterprise deployments that
    /// restrict outbound traffic to external STUN servers.
    /// Defaults to `stun.l.google.com:19302`. Set to "none" to disable.
    #[arg(long, help_heading = "WebRTC")]
    webrtc_stun_server: Option<String>,

    // ==================== Runtime ====================
    /// Explicit async runtime worker-thread count, from 1 to 4096 on every
    /// host (a refusal names the ceiling); a count above 4 x the CPUs
    /// available to the process starts with a warning, since a worker thread
    /// beyond the CPUs only waits for one. Leave unset to use tokio's default
    /// (`available_parallelism()`), which already honors the cgroup CPU quota
    /// on Rust 1.95+ and is therefore container-aware.
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

/// Parse a JWT role mapping from CLI format "idp_role=gateway_role". A
/// malformed entry is a configuration error, never a dropped mapping: with
/// the mapping dropped the map can end up empty, and an empty map grants the
/// gateway's admin role to any token whose roles claim says `admin`.
fn parse_role_mapping(mapping: &str) -> ConfigResult<(String, Role)> {
    let parts: Vec<&str> = mapping.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(ConfigError::InvalidValue {
            field: "jwt-role-mapping".to_string(),
            value: mapping.to_string(),
            reason: "expected 'idp_role=gateway_role'".to_string(),
        });
    }
    let idp_role = parts[0].to_string();
    let gateway_role = match parts[1].to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "jwt-role-mapping".to_string(),
                value: mapping.to_string(),
                reason: "the gateway role must be 'admin' or 'user'".to_string(),
            });
        }
    };
    Ok((idp_role, gateway_role))
}

/// Parse a control-plane API key from CLI format "id:name:role:key". A
/// malformed entry is a configuration error, never a dropped key: a dropped
/// key would leave the control plane open (or lock an operator out) with
/// nothing but a stderr line to show for it. The entry is not echoed, since
/// it carries the credential.
fn parse_control_plane_api_key(key_str: &str) -> ConfigResult<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(ConfigError::InvalidValue {
            field: "control-plane-api-keys".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'id:name:role:key'".to_string(),
        });
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        // The field is not named either: with the fields out of order it is the credential.
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "control-plane-api-keys".to_string(),
                value: "<redacted>".to_string(),
                reason: "the role must be 'admin' or 'user'".to_string(),
            });
        }
    };

    Ok(ApiKeyEntry::new(id, name, key, role))
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

// ==================== Secret sources ====================

/// A secret the command line, the environment or a file can give: the
/// names of its sources, for the resolution and for the refusals. A
/// refusal names sources and paths; none carries a value.
struct SecretSetting {
    /// What the secret is, for the refusals.
    what: &'static str,
    /// The `ConfigError` field a bad file is reported under.
    file_field: &'static str,
    /// The flag taking the value.
    flag: &'static str,
    /// The environment variable taking the value.
    env: &'static str,
    /// The flag naming a file that holds the value.
    file_flag: &'static str,
    /// The environment variable naming that file.
    file_env: &'static str,
}

const API_KEY_SETTING: SecretSetting = SecretSetting {
    what: "the shared API key",
    file_field: "api-key-file",
    flag: "--api-key",
    env: "SMG_API_KEY",
    file_flag: "--api-key-file",
    file_env: "SMG_API_KEY_FILE",
};

const TENANT_API_KEYS_SETTING: SecretSetting = SecretSetting {
    what: "the tenant API keys",
    file_field: "tenant-api-keys-file",
    flag: "--tenant-api-key",
    env: "SMG_TENANT_API_KEYS",
    file_flag: "--tenant-api-keys-file",
    file_env: "SMG_TENANT_API_KEYS_FILE",
};

const POSTGRES_DB_URL_SETTING: SecretSetting = SecretSetting {
    what: "the PostgreSQL connection URL",
    file_field: "postgres-db-url-file",
    flag: "--postgres-db-url",
    env: "SMG_POSTGRES_DB_URL",
    file_flag: "--postgres-db-url-file",
    file_env: "SMG_POSTGRES_DB_URL_FILE",
};

const REDIS_URL_SETTING: SecretSetting = SecretSetting {
    what: "the Redis connection URL",
    file_field: "redis-url-file",
    flag: "--redis-url",
    env: "SMG_REDIS_URL",
    file_flag: "--redis-url-file",
    file_env: "SMG_REDIS_URL_FILE",
};

/// An environment lookup: the value of a variable, `None` when it is not
/// set, and a refusal when it is set to bytes that are not UTF-8 (a source
/// that is set but unusable is refused, not passed over as unset). A
/// parameter rather than `std::env::var` so a test need not touch the
/// process environment.
type EnvLookup<'a> = &'a dyn Fn(&str) -> ConfigResult<Option<String>>;

/// The process environment as an [`EnvLookup`].
fn process_env(name: &str) -> ConfigResult<Option<String>> {
    env_value(name, std::env::var_os(name))
}

/// A variable's raw value as a string: `None` when the variable is unset,
/// a refusal naming the variable when its bytes are not UTF-8.
fn env_value(name: &str, raw: Option<std::ffi::OsString>) -> ConfigResult<Option<String>> {
    raw.map(|value| {
        value.into_string().map_err(|_| ConfigError::InvalidValue {
            field: name.to_string(),
            value: "<not UTF-8>".to_string(),
            reason: "set, but its value is not UTF-8".to_string(),
        })
    })
    .transpose()
}

impl SecretSetting {
    /// The file the flag or, failing that, the environment names.
    fn file(&self, flag_value: Option<&Path>, env: EnvLookup<'_>) -> ConfigResult<Option<PathBuf>> {
        Ok(match flag_value {
            Some(path) => Some(path.to_path_buf()),
            None => env(self.file_env)?.map(PathBuf::from),
        })
    }

    /// The refusal for a value and a file given together: ranking them
    /// would honour a stale secret in silence.
    fn given_twice(&self) -> ConfigError {
        ConfigError::IncompatibleConfig {
            reason: format!(
                "{} is given twice, by {} (or {}) and by {} (or {}); give one source",
                self.what, self.flag, self.env, self.file_flag, self.file_env
            ),
        }
    }

    /// The refusal for a file that cannot be used, naming the path, never
    /// the content.
    fn bad_file(&self, path: &Path, reason: &str) -> ConfigError {
        ConfigError::InvalidValue {
            field: self.file_field.to_string(),
            value: path.display().to_string(),
            reason: reason.to_string(),
        }
    }

    fn read_file(&self, path: &Path) -> ConfigResult<String> {
        std::fs::read_to_string(path)
            .map_err(|e| self.bad_file(path, &format!("the file cannot be read: {e}")))
    }

    /// The value of a single-valued secret and its source: the flag's,
    /// else the environment's, else the content of the file the flag or the
    /// environment names; the environment's value and the file's content
    /// have their surrounding whitespace (a trailing newline) dropped, since
    /// a Secret made from a file usually carries the editor's newline into
    /// both. A value and a file together are refused, as is a file that is
    /// empty or holds more than one line (a tenant key file or a
    /// concatenated pair in the place of one value is a mistake, not a
    /// value). An empty value from the flag or the environment passes here
    /// and is refused where the value is validated.
    fn resolve_value(
        &self,
        flag_value: Option<&str>,
        file_flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<(String, SecretSource)>> {
        let value = match flag_value {
            Some(value) => Some((value.to_string(), SecretSource::Flag)),
            None => {
                env(self.env)?.map(|value| (value.trim().to_string(), SecretSource::Environment))
            }
        };
        match (value, self.file(file_flag_value, env)?) {
            (Some(_), Some(_)) => Err(self.given_twice()),
            (Some(value), None) => Ok(Some(value)),
            (None, None) => Ok(None),
            (None, Some(path)) => {
                let content = self.read_file(&path)?;
                let value = content.trim();
                if value.is_empty() {
                    return Err(self.bad_file(&path, "the file is empty"));
                }
                if value.lines().count() > 1 {
                    return Err(self.bad_file(
                        &path,
                        "the file holds more than one line; it must hold the value alone",
                    ));
                }
                Ok(Some((value.to_string(), SecretSource::File)))
            }
        }
    }

    /// The entries of a list-valued secret and their source: the flag's
    /// occurrences, else the environment's value split at ',' and at line
    /// ends, else the lines of the file the flag or the environment names,
    /// blank lines and `#` comment lines skipped; the whitespace around an
    /// entry is dropped. Entries and a file together are refused, and so is
    /// a source that is set but holds no entry: the source was meant to
    /// close the serving routes, and an empty one would leave them open.
    fn resolve_entries(
        &self,
        flag_values: &[String],
        file_flag_value: Option<&Path>,
        env: EnvLookup<'_>,
    ) -> ConfigResult<Option<(Vec<String>, SecretSource)>> {
        let entries_of = |text: &str, separators: &[char]| -> Vec<String> {
            text.split(separators)
                .map(str::trim)
                .filter(|entry| !entry.is_empty() && !entry.starts_with('#'))
                .map(str::to_string)
                .collect()
        };
        let value = if flag_values.is_empty() {
            match env(self.env)? {
                Some(text) => {
                    let entries = entries_of(&text, &[',', '\n']);
                    if entries.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            field: self.env.to_string(),
                            value: "<empty>".to_string(),
                            reason: format!(
                                "set but holds no entry; {} are tenant_id:key entries separated by ',' or a newline",
                                self.what
                            ),
                        });
                    }
                    Some((entries, SecretSource::Environment))
                }
                None => None,
            }
        } else {
            Some((flag_values.to_vec(), SecretSource::Flag))
        };
        match (value, self.file(file_flag_value, env)?) {
            (Some(_), Some(_)) => Err(self.given_twice()),
            (Some(value), None) => Ok(Some(value)),
            (None, None) => Ok(None),
            (None, Some(path)) => {
                let content = self.read_file(&path)?;
                let entries = entries_of(&content, &['\n']);
                if entries.is_empty() {
                    return Err(self.bad_file(
                        &path,
                        "the file holds no entry; one tenant_id:key per line, blank and # lines skipped",
                    ));
                }
                Ok(Some((entries, SecretSource::File)))
            }
        }
    }
}

impl CliArgs {
    /// Phase 1b, resolve: the secrets the environment or a file may give in
    /// the command line's place, read into the flag fields so the phases
    /// after this one read flags alone; where each came from is kept for
    /// the start-up record. One rule for every secret: the command line
    /// beats the environment, and a value from either together with a file
    /// is refused rather than ranked. A store URL is read for the selected
    /// history backend alone, as its flag is. The environment lookup is a
    /// parameter so a test need not touch the process environment.
    fn resolve_secret_sources(&mut self, env: EnvLookup<'_>) -> ConfigResult<()> {
        let mut sources = SecretSources::default();
        if let Some((value, source)) = API_KEY_SETTING.resolve_value(
            self.integrations.api_key.as_deref(),
            self.integrations.api_key_file.as_deref(),
            env,
        )? {
            self.integrations.api_key = Some(value);
            sources.api_key = Some(source);
        }
        if let Some((entries, source)) = TENANT_API_KEYS_SETTING.resolve_entries(
            &self.integrations.tenant_api_keys,
            self.integrations.tenant_api_keys_file.as_deref(),
            env,
        )? {
            sources.tenant_api_keys = Some((source, entries.len()));
            self.integrations.tenant_api_keys = entries;
        }
        if self.model.history_backend == "postgres" {
            if let Some((value, source)) = POSTGRES_DB_URL_SETTING.resolve_value(
                self.integrations.postgres_db_url.as_deref(),
                self.integrations.postgres_db_url_file.as_deref(),
                env,
            )? {
                self.integrations.postgres_db_url = Some(value);
                sources.postgres_db_url = Some(source);
            }
        }
        if self.model.history_backend == "redis" {
            if let Some((value, source)) = REDIS_URL_SETTING.resolve_value(
                self.integrations.redis_url.as_deref(),
                self.integrations.redis_url_file.as_deref(),
                env,
            )? {
                self.integrations.redis_url = Some(value);
                sources.redis_url = Some(source);
            }
        }
        self.integrations.secret_sources = sources;
        Ok(())
    }
}

impl CliArgs {
    /// Build control plane authentication configuration from CLI args. A
    /// configuration that asks for authentication but cannot be honoured
    /// (a malformed key entry, an issuer without an audience or the reverse)
    /// is an error: silently dropping it would start the control plane open.
    fn build_control_plane_auth_config(&self) -> ConfigResult<ControlPlaneAuthConfig> {
        // Build JWT config if issuer and audience are provided
        let jwt = match (
            &self.integrations.jwt_issuer,
            &self.integrations.jwt_audience,
        ) {
            (Some(issuer), Some(audience)) => {
                // One malformed mapping refuses the start: dropped, it could widen access
                let role_mapping = self
                    .integrations
                    .jwt_role_mapping
                    .iter()
                    .map(|m| parse_role_mapping(m))
                    .collect::<ConfigResult<HashMap<String, Role>>>()?;

                let mut jwt_config = JwtConfig::new(issuer.clone(), audience.clone());
                jwt_config
                    .role_claim
                    .clone_from(&self.integrations.jwt_role_claim);
                jwt_config.role_mapping = role_mapping;
                if let Some(jwks_uri) = &self.integrations.jwt_jwks_uri {
                    jwt_config.jwks_uri = Some(jwks_uri.clone());
                }
                Some(jwt_config)
            }
            (Some(_), None) => {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "--jwt-issuer requires --jwt-audience (control-plane JWT authentication would otherwise be silently disabled)".to_string(),
                });
            }
            (None, Some(_)) => {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "--jwt-audience requires --jwt-issuer (control-plane JWT authentication would otherwise be silently disabled)".to_string(),
                });
            }
            (None, None) => None,
        };

        // Build API keys from CLI args; one malformed entry refuses the start
        let api_keys = self
            .integrations
            .control_plane_api_keys
            .iter()
            .map(|k| parse_control_plane_api_key(k))
            .collect::<ConfigResult<Vec<ApiKeyEntry>>>()?;

        Ok(ControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled: !self.integrations.disable_audit_logging,
        })
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

    /// The worker discovery provider selected by either spelling:
    /// `--service-discovery` is `--discovery-provider kubernetes`.
    fn selected_discovery_provider(&self) -> Option<DiscoveryProvider> {
        if self.discovery.service_discovery {
            Some(DiscoveryProvider::Kubernetes)
        } else {
            self.discovery.discovery_provider
        }
    }

    /// Selecting a discovery provider, by either spelling, turns IGW mode on.
    /// Returns whether this call turned it on.
    fn enable_igw_for_discovery(&mut self) -> bool {
        let enable = self.selected_discovery_provider().is_some() && !self.routing.enable_igw;
        self.routing.enable_igw |= enable;
        enable
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

    /// The metrics bind host: `--prometheus-host`, or the unspecified address
    /// of `--host`'s family when it is not given.
    fn metrics_host(&self) -> String {
        self.serving
            .prometheus_host
            .clone()
            .unwrap_or_else(|| MetricsConfig::default_host_for(&self.routing.host))
    }

    fn parse_mesh_socket_addr(
        host: &str,
        port: u16,
        field: &str,
    ) -> ConfigResult<std::net::SocketAddr> {
        bind_socket_addr(host, port).map_err(|e| ConfigError::InvalidValue {
            field: field.to_string(),
            value: host.to_string(),
            reason: format!("invalid mesh socket address: {e}"),
        })
    }

    /// Build mesh settings from the integration flags, validating addresses and TLS pairs.
    fn build_mesh_server_config(&self) -> ConfigResult<Option<MeshServerConfig>> {
        if !self.integrations.enable_mesh {
            return Ok(None);
        }

        let self_name = if let Some(name) = &self.integrations.mesh_server_name {
            validate_mesh_server_name(name)?;
            name.to_string()
        } else {
            let mut rng = rand::rng();
            let random_string: String = (0..4).map(|_| rng.sample(Alphanumeric) as char).collect();
            format!("Mesh_{random_string}")
        };

        let peer = self
            .integrations
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

        // Port 0 parses fine as a socket address, but it is meaningless as an
        // advertised one: peers gossip this port and cannot dial an ephemeral
        // bind. It would also reach router discovery as the fallback mesh port
        // for Pods whose annotation is missing or invalid, publishing `IP:0`
        // peers into the cluster state.
        if self.integrations.mesh_port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "mesh_port".to_string(),
                value: self.integrations.mesh_port.to_string(),
                reason: "mesh port cannot be 0; peers dial the advertised port, \
                         so it must be a fixed, routable one"
                    .to_string(),
            });
        }

        // The mesh port against the other listeners' is checked in
        // `server::startup`, for every way the server configuration is built.
        let bind_addr = Self::parse_mesh_socket_addr(
            &self.integrations.mesh_host,
            self.integrations.mesh_port,
            "mesh_host",
        )?;
        let (advertise_host, advertise_field) =
            if let Some(host) = self.integrations.mesh_advertise_host.as_deref() {
                (host, "mesh_advertise_host")
            } else {
                (self.integrations.mesh_host.as_str(), "mesh_host")
            };
        let advertise_addr = Self::parse_mesh_socket_addr(
            advertise_host,
            self.integrations.mesh_port,
            advertise_field,
        )?;

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
            mtls_config: self.mesh_mtls_config()?,
        }))
    }

    /// The mesh mTLS configuration from `--mesh-tls-ca-cert`, `--mesh-tls-cert`
    /// and `--mesh-tls-key` (all three or none, which the parser enforces);
    /// each file has to be readable at start-up, not at the first dial.
    fn mesh_mtls_config(&self) -> ConfigResult<Option<MTLSConfig>> {
        let (Some(ca), Some(cert), Some(key)) = (
            &self.integrations.mesh_tls_ca_cert,
            &self.integrations.mesh_tls_cert,
            &self.integrations.mesh_tls_key,
        ) else {
            return Ok(None);
        };
        for (field, path) in [
            ("mesh_tls_ca_cert", ca),
            ("mesh_tls_cert", cert),
            ("mesh_tls_key", key),
        ] {
            std::fs::metadata(path).map_err(|e| ConfigError::InvalidValue {
                field: field.to_string(),
                value: path.clone(),
                reason: format!("cannot read the file: {e}"),
            })?;
        }
        Ok(Some(MTLSConfig {
            ca_cert_path: ca.into(),
            server_cert_path: cert.into(),
            server_key_path: key.into(),
            ..MTLSConfig::default()
        }))
    }

    /// Build a routing policy with the same per-policy defaults as the CLI flags.
    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "passthrough" => PolicyConfig::Passthrough,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.routing.cache_threshold,
                balance_abs_threshold: self.routing.balance_abs_threshold,
                balance_rel_threshold: self.routing.balance_rel_threshold,
                eviction_interval_secs: self.routing.eviction_interval,
                max_tree_size: self.routing.max_tree_size,
                block_size: self.routing.block_size,
                balance_token_usage_threshold: self.routing.balance_token_usage_threshold,
                overload_token_usage_threshold: self.routing.overload_token_usage_threshold,
                overlap_decay: self.routing.overlap_decay,
                selection_temperature: self.routing.selection_temperature,
                cache_index: Self::parse_cache_index(&self.routing.cache_index),
                cache_ttl_secs: self.routing.cache_ttl_secs,
                cache_boundaries: self.routing.cache_boundaries.clone(),
                selection_policy: (self.routing.selection_policy != DEFAULT_SELECTION_POLICY)
                    .then(|| self.routing.selection_policy.clone()),
                selection_accounting_ttl_ms: self.routing.selection_accounting_ttl_ms,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
                load_metric: self.routing.power_of_two_load_metric,
            },
            "least_load" => PolicyConfig::LeastLoad {
                load_check_interval_secs: 5,
                kv_pressure_weight: self.routing.least_load_kv_pressure_weight,
                mean_prefill_tokens: self.routing.least_load_mean_prefill_tokens,
                default_throughput: self.routing.least_load_default_throughput,
                max_waiting_requests: self.routing.least_load_max_waiting_requests,
            },
            "bucket" => PolicyConfig::Bucket {
                balance_abs_threshold: self.routing.balance_abs_threshold,
                balance_rel_threshold: self.routing.balance_rel_threshold,
                bucket_adjust_interval_secs: 5,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: self.routing.prefix_token_count,
                load_factor: self.routing.prefix_hash_load_factor,
                balance_abs_threshold: self.routing.prefix_hash_balance_abs_threshold,
                cache_boundaries: self.routing.cache_boundaries.clone(),
            },
            "consistent_hashing" => PolicyConfig::ConsistentHashing,
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs: self.routing.eviction_interval,
                max_idle_secs: self.routing.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.routing.assignment_mode.as_deref(),
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

    /// Load the optional storage schema, reporting file and YAML errors as configuration errors.
    fn load_schema_config(&self) -> ConfigResult<Option<SchemaConfig>> {
        match &self.model.schema_config {
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

    /// Prefer an explicit Oracle DSN; otherwise require both wallet path and TNS alias.
    fn resolve_oracle_connect_details(&self) -> ConfigResult<OracleConnectSource> {
        if let Some(dsn) = self.integrations.oracle_dsn.clone() {
            return Ok(OracleConnectSource::Dsn { descriptor: dsn });
        }

        let wallet_path = self
            .integrations
            .oracle_wallet_path
            .clone()
            .ok_or_else(|| ConfigError::MissingRequired {
                field: "oracle_wallet_path or ATP_WALLET_PATH".to_string(),
            })?;

        let tns_alias = self.integrations.oracle_tns_alias.clone().ok_or_else(|| {
            ConfigError::MissingRequired {
                field: "oracle_tns_alias or ATP_TNS_ALIAS".to_string(),
            }
        })?;

        Ok(OracleConnectSource::Wallet {
            path: wallet_path,
            alias: tns_alias,
        })
    }

    /// Build Oracle storage settings after validating authentication and pool bounds.
    fn build_oracle_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<OracleConfig> {
        let (wallet_path, connect_descriptor) = match self.resolve_oracle_connect_details()? {
            OracleConnectSource::Dsn { descriptor } => (None, descriptor),
            OracleConnectSource::Wallet { path, alias } => (Some(path), alias),
        };
        let (username, password) = if self.integrations.oracle_external_auth {
            (
                self.integrations.oracle_user.clone().unwrap_or_default(),
                self.integrations
                    .oracle_password
                    .clone()
                    .unwrap_or_default(),
            )
        } else {
            (
                self.integrations.oracle_user.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "oracle_user or ATP_USER".to_string(),
                    }
                })?,
                self.integrations.oracle_password.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "oracle_password or ATP_PASSWORD".to_string(),
                    }
                })?,
            )
        };

        let pool_min = self
            .integrations
            .oracle_pool_min
            .unwrap_or_else(OracleConfig::default_pool_min);
        let pool_max = self
            .integrations
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
            .integrations
            .oracle_pool_timeout_secs
            .unwrap_or_else(OracleConfig::default_pool_timeout_secs);

        Ok(OracleConfig {
            wallet_path,
            connect_descriptor,
            external_auth: self.integrations.oracle_external_auth,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
            schema,
        })
    }

    /// Build PostgreSQL storage settings, requiring a URL and valid pool bounds.
    fn build_postgres_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<PostgresConfig> {
        let db_url = self
            .integrations
            .postgres_db_url
            .clone()
            .unwrap_or_default();
        let pool_max = self
            .integrations
            .postgres_pool_max_size
            .unwrap_or_else(PostgresConfig::default_pool_max);
        let pcf = PostgresConfig {
            db_url,
            pool_max,
            op_timeout_secs: self
                .integrations
                .postgres_op_timeout_secs
                .unwrap_or_else(PostgresConfig::default_op_timeout_secs),
            schema,
        };
        pcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(pcf)
    }

    /// Build Redis storage settings, requiring a URL and a nonzero pool size.
    fn build_redis_config(&self, schema: Option<SchemaConfig>) -> ConfigResult<RedisConfig> {
        let url = self.integrations.redis_url.clone().unwrap_or_default();
        let pool_max = self.integrations.redis_pool_max_size.unwrap_or(16);

        let retention_days = match self.integrations.redis_retention_days {
            Some(-1) => None, // Persistent
            // A retention of 0 days would expire every record as it is
            // written; -1 is the one spelling of persistent records.
            Some(d) if d <= 0 => {
                return Err(ConfigError::InvalidValue {
                    field: "redis_retention_days".to_string(),
                    value: d.to_string(),
                    reason: "Must be at least 1 (-1 keeps records persistent)".to_string(),
                });
            }
            Some(d) => Some(d as u64),
            None => Some(30), // Default 30 days
        };

        let rcf = RedisConfig {
            url,
            pool_max,
            op_timeout_secs: self
                .integrations
                .redis_op_timeout_secs
                .unwrap_or_else(RedisConfig::default_op_timeout_secs),
            retention_days,
            schema,
        };
        rcf.validate().map_err(|e| ConfigError::ValidationFailed {
            reason: e.to_string(),
        })?;
        Ok(rcf)
    }

    /// The flags whose value has a required shape and whose raw form never
    /// reaches `RouterConfig`, refused by name when the shape is wrong: a
    /// wrong shape used to be dropped or passed on in silence. The shapes of
    /// the fields `RouterConfig` carries (the CORS origins, the health-check
    /// endpoint, the request-id header names, the storage-context header
    /// names, an empty API key) are checked by `ConfigValidator`, on every
    /// construction path.
    fn validate_flag_formats(&self) -> ConfigResult<()> {
        let invalid = |field: &str, value: &str, reason: &str| ConfigError::InvalidValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: reason.to_string(),
        };
        // `cors_allowed_origins`, the health-check endpoint, the request-id
        // header names and an empty API key are checked by the shared
        // `ConfigValidator`, which every construction path runs (this CLI
        // through `builder.build()`).
        let buckets = &self.serving.prometheus_duration_buckets;
        if let Some(bad) = buckets.iter().find(|b| !b.is_finite() || **b <= 0.0) {
            return Err(invalid(
                "prometheus_duration_buckets",
                &bad.to_string(),
                "every bucket bound must be a finite number above 0",
            ));
        }
        if let Some(pair) = buckets.windows(2).find(|w| w[1] <= w[0]) {
            return Err(invalid(
                "prometheus_duration_buckets",
                &format!("{} {}", pair[0], pair[1]),
                "bucket bounds must ascend strictly",
            ));
        }
        for entry in &self.serving.storage_context_headers {
            if !entry.contains('=') {
                return Err(invalid(
                    "storage_context_headers",
                    entry,
                    "expected 'header-name=context-key'",
                ));
            }
        }
        for mapping in &self.integrations.jwt_role_mapping {
            match mapping.split_once('=') {
                Some((idp_role, gateway_role))
                    if !idp_role.is_empty()
                        && matches!(
                            gateway_role.to_ascii_lowercase().as_str(),
                            "admin" | "user"
                        ) => {}
                _ => {
                    return Err(invalid(
                        "jwt_role_mapping",
                        mapping,
                        "expected 'idp_role=admin' or 'idp_role=user'",
                    ))
                }
            }
        }
        Ok(())
    }

    /// The flags that only act with a companion, named when the companion is
    /// absent: each used to be ignored in silence. One row per warning, in
    /// the order they are printed: when it applies, and its text.
    fn ignored_companions(
        &self,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) -> Vec<String> {
        const PD: &str = "without --pd-disaggregation (or --epd-disaggregation)";
        const EPD: &str = "without --epd-disaggregation";
        let pd = self.workers.pd_disaggregation || self.workers.epd_disaggregation;
        let epd = self.workers.epd_disaggregation;
        // The selectors select Kubernetes pods: another provider ignores them
        // as the absence of any does.
        let provider = self.selected_discovery_provider();
        let kubernetes = provider == Some(DiscoveryProvider::Kubernetes);
        let selectors = match provider {
            None => "without --service-discovery".to_string(),
            Some(provider) => format!(
                "under --discovery-provider {}: it selects Kubernetes pods",
                provider.flag_value()
            ),
        };
        // Two worker sources are both honoured, which is worth a line of its
        // own, naming the source as it was given.
        let source = if self.discovery.service_discovery {
            "--service-discovery".to_string()
        } else {
            provider.map_or_else(String::new, |provider| {
                format!("--discovery-provider {}", provider.flag_value())
            })
        };
        let ignored = |flag: &str, needs: &str| format!("{flag} is ignored {needs}");
        let warnings = [
            (!prefill_urls.is_empty() && !pd, ignored("--prefill", PD)),
            (
                !self.workers.decode.is_empty() && !pd,
                ignored("--decode", PD),
            ),
            (
                self.workers.prefill_policy.is_some() && !pd,
                ignored("--prefill-policy", PD),
            ),
            (
                self.workers.decode_policy.is_some() && !pd,
                ignored("--decode-policy", PD),
            ),
            (!encode_urls.is_empty() && !epd, ignored("--encode", EPD)),
            (
                self.workers.encode_policy.is_some() && !epd,
                ignored("--encode-policy", EPD),
            ),
            (
                !self.discovery.selector.is_empty() && !kubernetes,
                ignored("--selector", selectors.as_str()),
            ),
            (
                !self.discovery.prefill_selector.is_empty() && !kubernetes,
                ignored("--prefill-selector", selectors.as_str()),
            ),
            (
                !self.discovery.decode_selector.is_empty() && !kubernetes,
                ignored("--decode-selector", selectors.as_str()),
            ),
            (
                !self.discovery.encode_selector.is_empty() && !kubernetes,
                ignored("--encode-selector", selectors.as_str()),
            ),
            (
                !self.discovery.router_selector.is_empty() && !kubernetes,
                ignored("--router-selector", selectors.as_str()),
            ),
            (
                provider.is_some() && !self.routing.worker_urls.is_empty(),
                format!(
                    "--worker-urls and {source} are both given: both are worker sources, \
                     the listed workers register first and the discovered ones join them"
                ),
            ),
            (
                self.serving.priority_scheduler_config.is_some()
                    && !self.serving.priority_scheduler_enabled,
                ignored(
                    "--priority-scheduler-config",
                    "without --priority-scheduler-enabled",
                ),
            ),
        ];
        warnings
            .into_iter()
            .filter_map(|(applies, warning)| applies.then_some(warning))
            .collect()
    }

    /// The companion warnings of [`ignored_companions`](Self::ignored_companions), one
    /// line each on stderr: the logger is not up when the flags are read.
    #[expect(clippy::print_stderr, reason = "pre-logger CLI configuration warnings")]
    fn warn_ignored_companions(
        &self,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) {
        for ignored in self.ignored_companions(prefill_urls, encode_urls) {
            eprintln!("WARNING: {ignored}");
        }
    }

    /// The routing mode the flags select. A vendor backend (`--backend
    /// openai|anthropic|gemini`) routes to that vendor whatever else is set;
    /// otherwise `--epd-disaggregation` wins over `--pd-disaggregation`, and
    /// neither is the regular mode over `--worker-urls`. IGW mode does not
    /// change the routing mode: it only affects router initialization.
    fn routing_mode(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> RoutingMode {
        let policy = |name: Option<&String>| name.map(|p| self.parse_policy(p));
        match (
            self.model.backend,
            self.workers.epd_disaggregation,
            self.workers.pd_disaggregation,
        ) {
            (Some(Backend::Openai), _, _) => RoutingMode::OpenAI {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (Some(Backend::Anthropic), _, _) => RoutingMode::Anthropic {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (Some(Backend::Gemini), _, _) => RoutingMode::Gemini {
                worker_urls: self.routing.worker_urls.clone(),
            },
            (_, true, _) => RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls: self.workers.decode.clone(),
                encode_policy: policy(self.workers.encode_policy.as_ref()),
                prefill_policy: policy(self.workers.prefill_policy.as_ref()),
                decode_policy: policy(self.workers.decode_policy.as_ref()),
            },
            (_, false, true) => RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls: self.workers.decode.clone(),
                prefill_policy: policy(self.workers.prefill_policy.as_ref()),
                decode_policy: policy(self.workers.decode_policy.as_ref()),
            },
            (_, false, false) => RoutingMode::Regular {
                worker_urls: self.routing.worker_urls.clone(),
            },
        }
    }

    /// The discovery configuration of the selected provider, or none. A
    /// provider's own flags are ignored under another provider, as the
    /// Kubernetes flags always were without discovery; the file flags have
    /// no other use, so giving them without the file provider is an error,
    /// as is the file provider without its manifest.
    fn discovery_config(&self) -> ConfigResult<Option<DiscoveryConfig>> {
        let provider = self.selected_discovery_provider();
        if provider != Some(DiscoveryProvider::File)
            && (self.discovery.discovery_file.is_some()
                || self.discovery.discovery_check_interval_secs.is_some())
        {
            return Err(ConfigError::IncompatibleConfig {
                reason: "--discovery-file and --discovery-check-interval-secs apply only to \
                         --discovery-provider file"
                    .to_string(),
            });
        }
        Ok(match provider {
            None => None,
            Some(DiscoveryProvider::Kubernetes) => {
                Some(DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
                    namespace: self.discovery.service_discovery_namespace.clone(),
                    port: self.discovery.service_discovery_port,
                    check_interval_secs: 60,
                    selector: Self::parse_selector(&self.discovery.selector),
                    encode_selector: Self::parse_selector(&self.discovery.encode_selector),
                    prefill_selector: Self::parse_selector(&self.discovery.prefill_selector),
                    decode_selector: Self::parse_selector(&self.discovery.decode_selector),
                    bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
                    worker_ports_annotation: "smg.ai/worker-ports".to_string(),
                    kv_connector_annotation: self.discovery.kv_connector_annotation.clone(),
                    kv_engine_id_annotation: self.discovery.kv_engine_id_annotation.clone(),
                    router_selector: Self::parse_selector(&self.discovery.router_selector),
                    router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                    model_id_source: self.discovery.model_id_from.clone(),
                }))
            }
            Some(DiscoveryProvider::File) => Some(DiscoveryConfig::File(FileDiscoveryConfig {
                path: self.discovery.discovery_file.clone().ok_or_else(|| {
                    ConfigError::MissingRequired {
                        field: "--discovery-file".to_string(),
                    }
                })?,
                check_interval_secs: self
                    .discovery
                    .discovery_check_interval_secs
                    .unwrap_or(DEFAULT_FILE_DISCOVERY_CHECK_INTERVAL_SECS),
            })),
        })
    }

    /// Every worker url the mode starts with, in the order the flags list
    /// them (encode, prefill, decode for the disaggregated modes): the first
    /// url that declares a wire protocol decides the connection mode.
    fn startup_worker_urls(mode: &RoutingMode) -> Vec<String> {
        match mode {
            RoutingMode::Regular { worker_urls }
            | RoutingMode::OpenAI { worker_urls }
            | RoutingMode::Anthropic { worker_urls }
            | RoutingMode::Gemini { worker_urls } => worker_urls.clone(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls
                .iter()
                .map(|(url, _)| url.clone())
                .chain(decode_urls.iter().cloned())
                .collect(),
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                ..
            } => encode_urls
                .iter()
                .chain(prefill_urls.iter())
                .map(|(url, _)| url.clone())
                .chain(decode_urls.iter().cloned())
                .collect(),
        }
    }

    /// `--backend` normally only steers the routing mode. Over ZMQ it
    /// additionally pins the startup workers' runtime: the shared EngineCore
    /// handshake carries no engine identity, so the wire protocol cannot be
    /// probed and must be declared. HTTP/gRPC keep auto-detection (None).
    fn startup_worker_runtime_type(&self, connection_mode: ConnectionMode) -> Option<RuntimeType> {
        if connection_mode != ConnectionMode::Zmq {
            return None;
        }
        match self.model.backend {
            Some(Backend::Vllm) => Some(RuntimeType::Vllm),
            Some(Backend::Tokenspeed) => Some(RuntimeType::TokenSpeed),
            Some(Backend::Sglang) => Some(RuntimeType::Sglang),
            _ => None,
        }
    }

    /// The `--model-alias` entries as a map. clap validated each entry's
    /// shape; here only the same alias naming two different canonical models
    /// is refused, which would make the routing outcome depend on argument
    /// order.
    fn model_aliases(entries: &[String]) -> ConfigResult<HashMap<String, String>> {
        let mut model_aliases: HashMap<String, String> = HashMap::new();
        for entry in entries {
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
        Ok(model_aliases)
    }

    /// The router configuration of the flags. A refusal is raised in this
    /// order: the flag shapes, the discovery flags, the schema file, the
    /// tenant keys, the history store, the model aliases, and last the
    /// shared validator over the built configuration, which the builder
    /// runs.
    fn to_router_config(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<RouterConfig> {
        self.validate_flag_formats()?;
        self.warn_ignored_companions(&prefill_urls, &encode_urls);

        let mode = self.routing_mode(prefill_urls, encode_urls);
        let policy = self.parse_policy(&self.routing.policy);
        let discovery = self.discovery_config()?;
        let metrics = Some(MetricsConfig {
            port: self.serving.prometheus_port,
            host: self.metrics_host(),
        });
        let trace_config = Some(TraceConfig {
            enable_trace: self.integrations.enable_trace,
            otlp_traces_endpoint: self.integrations.otlp_traces_endpoint.clone(),
        });
        let connection_mode = Self::determine_connection_mode(&Self::startup_worker_urls(&mode));
        let startup_worker_runtime_type = self.startup_worker_runtime_type(connection_mode);
        let history_backend = match self.model.history_backend.as_str() {
            "none" => HistoryBackend::None,
            "oracle" => HistoryBackend::Oracle,
            "postgres" => HistoryBackend::Postgres,
            "redis" => HistoryBackend::Redis,
            _ => HistoryBackend::Memory,
        };
        let schema = self.load_schema_config()?;
        let tenant_api_keys = self
            .integrations
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
        let model_aliases = Self::model_aliases(&self.discovery.model_alias)?;

        let builder = RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .cache_boundaries(self.routing.cache_boundaries.clone())
            .connection_mode(connection_mode)
            .startup_worker_runtime_type(startup_worker_runtime_type)
            .zmq_engine_count(self.routing.zmq_engine_count)
            .host(&self.routing.host)
            .port(self.routing.port)
            .health_check_port(self.routing.health_check_port)
            .admin_port(self.routing.admin_port)
            .admin_host(self.routing.admin_host.clone())
            .runtime_worker_threads(self.integrations.runtime_worker_threads)
            .max_payload_size(self.serving.max_payload_size)
            .max_buffered_request_bytes(self.serving.max_buffered_request_bytes)
            .stream_body_stall_timeout_secs(self.serving.stream_body_stall_timeout_secs)
            .header_read_timeout_secs(self.serving.header_read_timeout_secs)
            .idle_connection_timeout_secs(self.serving.idle_connection_timeout_secs)
            .request_timeout_secs(self.serving.request_timeout_secs)
            .sse_keepalive_secs(self.serving.sse_keepalive_secs)
            .upstream_pool_idle_timeout_secs(self.serving.upstream_pool_idle_timeout_secs)
            .worker_startup_timeout_secs(self.workers.worker_startup_timeout_secs)
            .worker_startup_delay_secs(self.workers.worker_startup_delay)
            .worker_startup_check_interval_secs(self.workers.worker_startup_check_interval)
            .job_queue_capacity(self.routing.job_queue_capacity)
            .job_queue_concurrency(self.routing.job_queue_concurrency)
            .load_monitor_interval_secs(self.workers.load_monitor_interval)
            .worker_stall_secs(self.workers.worker_stall_secs)
            .worker_wedge_secs(self.workers.worker_wedge_secs)
            .worker_stale_secs(self.workers.worker_stale_secs)
            .worker_warmup(
                self.routing.worker_warmup_secs,
                self.routing.worker_warmup_share,
                self.routing.worker_warmup_blocks,
                self.routing.worker_warmup_thin_ratio,
                self.routing.worker_warmup_divert_every,
            )
            .pd_admission_wait_secs(self.workers.pd_admission_wait_secs)
            .disable_load_monitoring(self.workers.disable_load_monitoring)
            .worker_overload_protection(
                self.routing.worker_overload_protection
                    && !self.routing.disable_worker_overload_protection,
            )
            .worker_overload_waiting_requests(Some(self.routing.worker_overload_waiting_requests))
            .worker_overload_token_usage(Some(self.routing.worker_overload_token_usage))
            .worker_overload_shed(self.routing.worker_overload_shed)
            .kv_indexer_ttl_secs(self.routing.kv_indexer_ttl_secs)
            .kv_indexer_max_entries(self.routing.kv_indexer_max_entries)
            .kv_index(self.routing.kv_index.clone())
            .engine_metrics(self.workers.engine_metrics)
            .multimodal_tensor_transport(self.workers.multimodal_tensor_transport)
            .multimodal_shm_min_bytes(self.workers.multimodal_shm_min_bytes)
            .multimodal_max_inflight_bytes(self.workers.multimodal_max_inflight_bytes)
            .mm_per_request_image_limit(self.workers.mm_per_request_image_limit.map(|v| v as usize))
            .mm_processing(self.workers.mm_processing)
            .mm_pixel_cache_mb(self.workers.mm_pixel_cache_mb)
            .mm_pixel_rdma(self.workers.mm_pixel_rdma)
            .mm_allowed_media_domains(self.workers.mm_allowed_media_domains.clone())
            .mm_allow_private_media_hosts(self.workers.mm_allow_private_media_hosts)
            .rdma_listen_ip(self.workers.rdma_listen_ip.clone())
            .rdma_slot_ttl_s(self.workers.rdma_slot_ttl_s)
            .log_mm_timing(self.workers.log_mm_timing)
            .max_concurrent_requests(self.serving.max_concurrent_requests)
            .queue_size(self.serving.queue_size)
            .queue_timeout_secs(self.serving.queue_timeout_secs)
            .prefill_max_inflight_requests_per_worker(
                self.workers.prefill_max_inflight_requests_per_worker,
            )
            .prefill_queue_size(self.workers.prefill_queue_size)
            .prefill_queue_timeout_secs(self.workers.prefill_queue_timeout_secs)
            .priority_scheduler_enabled(self.serving.priority_scheduler_enabled)
            .priority_scheduler_default_max_class(
                self.serving.priority_scheduler_default_max_class.clone(),
            )
            .priority_scheduler_config(self.serving.priority_scheduler_config.clone())
            .priority_scheduler_tenant_metric_top_n(
                self.serving.priority_scheduler_tenant_metric_top_n,
            )
            .tenant_rate_limit_enabled(self.serving.tenant_rate_limit_enabled)
            .tenant_rate_limit_config(self.serving.tenant_rate_limit_config.clone())
            .cors_allowed_origins(self.serving.cors_allowed_origins.clone())
            .retry_config(RetryConfig {
                max_retries: self.serving.retry_max_retries,
                initial_backoff_ms: self.serving.retry_initial_backoff_ms,
                max_backoff_ms: self.serving.retry_max_backoff_ms,
                backoff_multiplier: self.serving.retry_backoff_multiplier,
                jitter_factor: self.serving.retry_jitter_factor,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: self.serving.cb_failure_threshold,
                success_threshold: self.serving.cb_success_threshold,
                timeout_duration_secs: self.serving.cb_timeout_duration_secs,
                window_duration_secs: self.serving.cb_window_duration_secs,
            })
            .health_check_config(HealthCheckConfig {
                failure_threshold: self.serving.health_failure_threshold,
                success_threshold: self.serving.health_success_threshold,
                timeout_secs: self.serving.health_check_timeout_secs,
                check_interval_secs: self.serving.health_check_interval_secs,
                endpoint: self.serving.health_check_endpoint.clone(),
                disable_health_check: self.serving.disable_health_check,
                remove_unhealthy_workers: resolve_worker_auto_recovery(
                    self.serving.remove_unhealthy_workers,
                    self.selected_discovery_provider().is_some(),
                ),
                drain_settle_secs: self.serving.drain_settle_secs,
            })
            .tokenizer_cache(TokenizerCacheConfig {
                enable_l0: self.model.tokenizer_cache_enable_l0,
                l0_max_entries: self.model.tokenizer_cache_l0_max_entries,
                l0_max_memory: self.model.tokenizer_cache_l0_max_memory,
                enable_l1: self.model.tokenizer_cache_enable_l1,
                l1_max_memory: self.model.tokenizer_cache_l1_max_memory,
            })
            .disable_tokenizer_autoload(self.model.disable_tokenizer_autoload)
            .parser(ParserChoice::parse(&self.model.parser).unwrap_or_default())
            .history_backend(history_backend)
            .memory_config(MemoryConfig {
                max_responses: self.model.history_memory_max_responses,
                max_items: self.model.history_memory_max_items,
                retention_secs: (self.model.history_memory_retention_secs > 0)
                    .then_some(self.model.history_memory_retention_secs),
            })
            .log_level(&self.serving.log_level)
            .maybe_api_key(self.integrations.api_key.as_ref())
            .tenant_api_keys(tenant_api_keys)
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.serving.log_dir.as_ref())
            .maybe_jemalloc_prof_dir(self.serving.jemalloc_prof_dir.as_ref())
            .maybe_request_id_headers(
                (!self.serving.request_id_headers.is_empty())
                    .then(|| self.serving.request_id_headers.clone()),
            )
            .maybe_storage_context_headers(
                (!self.serving.storage_context_headers.is_empty())
                    .then(|| Self::parse_selector(&self.serving.storage_context_headers)),
            )
            .trust_tenant_header(self.serving.trust_tenant_header)
            .tenant_header_name(&self.serving.tenant_header_name)
            .maybe_rate_limit_tokens_per_second(self.serving.rate_limit_tokens_per_second)
            .max_background_responses(self.serving.max_background_responses)
            .maybe_model_path(self.model.model_path.as_ref())
            .maybe_tokenizer_path(self.model.tokenizer_path.as_ref())
            .maybe_chat_template(self.model.chat_template.as_ref())
            .model_aliases(model_aliases)
            .maybe_oracle(oracle)
            .maybe_postgres(postgres)
            .maybe_redis(redis)
            .maybe_reasoning_parser(self.model.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.model.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.model.mcp_config_path.as_ref())
            .dp_aware(self.routing.dp_aware)
            .pd_pairing_mode(
                PdPairingMode::parse(&self.routing.pd_pairing_mode).unwrap_or_default(),
            )
            .routing_key_override(RoutingKeyOverrideConfig {
                enabled: self.routing.routing_key_override,
                eviction_interval_secs: self.routing.eviction_interval,
                max_idle_secs: self.routing.max_idle_secs,
                assignment_mode: Self::parse_assignment_mode(
                    self.routing.assignment_mode.as_deref(),
                    ManualAssignmentMode::Delegate,
                ),
                headers: self.routing.routing_key_headers.clone(),
            })
            .retries(!self.serving.disable_retries)
            .upstream_http2(self.routing.upstream_http2)
            .circuit_breaker(!self.serving.disable_circuit_breaker)
            .enable_wasm(self.model.enable_wasm)
            .maybe_storage_hook_wasm_path(self.model.storage_hook_wasm_path.as_deref())
            .igw(self.routing.enable_igw)
            .rl(smg_rl::RlConfig {
                enabled: self.workers.enable_rl,
                control_timeout_secs: self.workers.rl_control_timeout_secs,
                fanout_concurrency: self.workers.rl_fanout_concurrency,
            })
            .dp_minimum_tokens_scheduler(self.routing.dp_minimum_tokens_scheduler)
            .maybe_server_cert_and_key(
                self.integrations.tls_cert_path.as_ref(),
                self.integrations.tls_key_path.as_ref(),
            );

        builder.build()
    }

    /// Combine validated router settings with listener, discovery, and authentication flags.
    fn to_server_config(&self, router_config: RouterConfig) -> ConfigResult<ServerConfig> {
        let service_discovery_config = router_config
            .discovery
            .as_ref()
            .map(|discovery| RuntimeDiscoveryConfig::from_config(discovery, &router_config.mode))
            .transpose()?;
        let mesh_discovery_config = router_config
            .discovery
            .as_ref()
            .and_then(MeshDiscoveryConfig::from_discovery);

        let prometheus_config = Some(PrometheusConfig {
            port: self.serving.prometheus_port,
            host: self.metrics_host(),
            duration_buckets: if self.serving.prometheus_duration_buckets.is_empty() {
                None
            } else {
                Some(self.serving.prometheus_duration_buckets.clone())
            },
        });

        // Build control plane auth config
        let control_plane_auth = {
            let config = self.build_control_plane_auth_config()?;
            if config.is_enabled() {
                Some(config)
            } else {
                None
            }
        };

        // ==================== Mesh Server ====================
        let mesh_server_config = self.build_mesh_server_config()?;

        Ok(ServerConfig {
            host: self.routing.host.clone(),
            port: self.routing.port,
            health_check_port: self.routing.health_check_port,
            runtime_worker_threads: self.integrations.runtime_worker_threads,
            router_config,
            max_payload_size: self.serving.max_payload_size,
            log_dir: self.serving.log_dir.clone(),
            log_level: Some(self.serving.log_level.clone()),
            log_json: self.serving.log_json,
            service_discovery_config,
            mesh_discovery_config,
            prometheus_config,
            request_timeout_secs: self.serving.request_timeout_secs,
            request_id_headers: if self.serving.request_id_headers.is_empty() {
                None
            } else {
                Some(self.serving.request_id_headers.clone())
            },
            shutdown_grace_period_secs: self.serving.shutdown_grace_period_secs,
            control_plane_auth,
            mesh_server_config,
            webrtc_bind_addr: self.integrations.webrtc_bind_addr,
            webrtc_stun_server: self.integrations.webrtc_stun_server.clone(),
            secret_sources: self.integrations.secret_sources,
        })
    }

    /// The mode label of the start-up banner. It follows the flags, not the
    /// built [`RoutingMode`]: IGW first, then the OpenAI and Anthropic
    /// backends, then the disaggregated modes, then the backend's name
    /// (`--backend gemini` is labelled `Regular (gemini)`, as it always was).
    fn banner_mode(&self) -> String {
        if self.routing.enable_igw {
            "IGW (Inference Gateway)".to_string()
        } else if matches!(self.model.backend, Some(Backend::Openai)) {
            "OpenAI Backend".to_string()
        } else if matches!(self.model.backend, Some(Backend::Anthropic)) {
            "Anthropic Backend".to_string()
        } else if self.workers.epd_disaggregation {
            "EPD Disaggregated".to_string()
        } else if self.workers.pd_disaggregation {
            "PD Disaggregated".to_string()
        } else if let Some(backend) = &self.model.backend {
            format!("Regular ({backend})")
        } else {
            "Regular".to_string()
        }
    }

    /// Phase 2, announce: the banner and the lines under it, printed before
    /// the configuration is built so an operator sees what was asked for
    /// even when it is refused; nothing is changed. The logger is not up
    /// yet: this is plain stdout.
    #[expect(
        clippy::print_stdout,
        reason = "pre-logger startup output: the banner and the flags' summary"
    )]
    fn announce(
        &self,
        igw_enabled_for_discovery: bool,
        prefill_urls: &[(String, Option<u16>)],
        encode_urls: &[(String, Option<u16>)],
    ) {
        if igw_enabled_for_discovery {
            println!("INFO: IGW mode automatically enabled because service discovery is turned on");
        }
        version::print_banner(&self.routing.host, self.routing.port, &self.banner_mode());
        if self.routing.enable_igw {
            return;
        }
        println!("Policy: {}", self.routing.policy);
        if self.workers.pd_disaggregation && !prefill_urls.is_empty() {
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", self.workers.decode);
        }
        if self.workers.epd_disaggregation {
            println!("Encode nodes: {encode_urls:?}");
            println!("Prefill nodes: {prefill_urls:?}");
            println!("Decode nodes: {:?}", self.workers.decode);
        }
    }

    /// Phase 3, configure: the router configuration, validated, wrapped in
    /// the server configuration. Every refusal is a [`ConfigError`] naming
    /// the flag; nothing has been bound or spawned when one is raised.
    fn configure(
        &self,
        prefill_urls: Vec<(String, Option<u16>)>,
        encode_urls: Vec<(String, Option<u16>)>,
    ) -> ConfigResult<ServerConfig> {
        let router_config = self.to_router_config(prefill_urls, encode_urls)?;
        router_config.validate()?;
        self.to_server_config(router_config)
    }
}

/// The text a version flag prints, read before anything else is parsed so a
/// command line with an error elsewhere still answers `--version`.
fn version_flag(args: &[String]) -> Option<String> {
    args.iter().find_map(|arg| match arg.as_str() {
        "--version" | "-V" => Some(version::get_version_string()),
        "--version-verbose" => Some(version::get_verbose_version_string()),
        _ => None,
    })
}

/// Phase 1, parse: clap over the command line with the `--prefill` /
/// `--encode` occurrences already taken out, `smg launch ...` and the bare
/// form alike. A usage error or `--help` ends the process here, before the
/// banner. Selecting a discovery provider turns IGW mode on here as well, so
/// the flags are final when this returns and the phases after it only read
/// them; the second value says whether this call turned IGW on, for the
/// announcement.
fn parse_cli_args(clap_args: Vec<String>) -> (CliArgs, bool) {
    let cli = Cli::parse_from(clap_args);
    let mut cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };
    let igw_enabled_for_discovery = cli_args.enable_igw_for_discovery();
    (cli_args, igw_enabled_for_discovery)
}

/// Phase 4, serve: the async runtime (tokio's container-aware default size
/// unless `--runtime-worker-threads` pins one; the size is logged by
/// `startup` once the log subscriber is up), then [`server::run`]: the
/// server until it stops, the log writers flushed and the tracer shut down
/// on every exit, a clean stop and a failed start-up alike. The runtime is
/// built only now: a refusal before this point costs no threads.
fn serve(server_config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = match server_config.runtime_worker_threads {
        Some(n) => tokio::runtime::Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build()?,
        None => tokio::runtime::Runtime::new()?,
    };
    server::run(&runtime, server_config)
}

/// The start-up sequence: parse, resolve the secrets' sources, announce,
/// configure, serve. The version flags are answered first, before clap sees
/// the line.
#[expect(clippy::print_stdout, reason = "the version flags' answer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_jemalloc_as_global_allocator();
    let args: Vec<String> = std::env::args().collect();
    if let Some(version) = version_flag(&args) {
        println!("{version}");
        return Ok(());
    }
    let command_line = CommandLine::split(&args);
    let (mut cli_args, igw_enabled_for_discovery) = parse_cli_args(command_line.clap_args);
    cli_args.resolve_secret_sources(&process_env)?;
    cli_args.announce(
        igw_enabled_for_discovery,
        &command_line.prefill_urls,
        &command_line.encode_urls,
    );
    let server_config = cli_args.configure(command_line.prefill_urls, command_line.encode_urls)?;
    serve(server_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise both generated builders and startup parsing on Rust's default
    /// thread stack. A subprocess contains stack overflow's process-wide abort.
    #[test]
    fn cli_startup_fits_a_two_mib_thread_stack() {
        const CHILD: &str = "SMG_CLI_STACK_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            std::thread::Builder::new()
                .stack_size(2 * 1024 * 1024)
                .spawn(|| {
                    use clap::CommandFactory;

                    Cli::command().debug_assert();
                    Cli::command_for_update().debug_assert();
                    for prefix in [&["smg"][..], &["smg", "launch"], &["smg", "start"]] {
                        let cli =
                            Cli::try_parse_from(prefix.iter().copied().chain(["--port", "31000"]))
                                .unwrap();
                        let mut args = match cli.command {
                            Some(Commands::Launch { args }) => args,
                            None => cli.router_args,
                        };
                        args.resolve_secret_sources(&|_| Ok(None)).unwrap();
                        let router = args.to_router_config(vec![], vec![]).unwrap();
                        let server = args.to_server_config(router).unwrap();
                        assert_eq!(server.port, 31000);
                        assert_eq!(server.router_config.port, 31000);
                    }
                })
                .unwrap()
                .join()
                .unwrap();
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::cli_startup_fits_a_two_mib_thread_stack",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "CLI startup exceeded its thread stack or failed: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    /// Parse top-level CLI args into the flattened `CliArgs` the binary uses.
    fn cli_args_from(args: &[&str]) -> CliArgs {
        let argv: Vec<String> = std::iter::once("smg".to_string())
            .chain(args.iter().map(|s| (*s).to_string()))
            .collect();
        Cli::parse_from(argv).router_args
    }

    /// `--prefill` / `--encode` take an optional trailing bootstrap port (a
    /// number or `none`, any case); one token rule parses the pairs and
    /// strips them from the line clap parses, so a following flag or url is
    /// kept, and a flag without a url is left for clap to refuse.
    #[test]
    fn url_port_flags_are_split_from_argv_by_one_token_rule() {
        let args: Vec<String> = [
            "smg",
            "--pd-disaggregation",
            "--prefill",
            "http://p1:1",
            "9001",
            "--prefill",
            "http://p2:2",
            "NONE",
            "--encode",
            "http://e1:3",
            "--prefill",
            "http://p3:4",
            "--decode",
            "http://d1:5",
            "--encode",
            "http://e2:6",
            "--prefill",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let line = CommandLine::split(&args);
        assert_eq!(
            line.prefill_urls,
            vec![
                ("http://p1:1".to_string(), Some(9001)),
                ("http://p2:2".to_string(), None),
                ("http://p3:4".to_string(), None),
            ]
        );
        assert_eq!(
            line.encode_urls,
            vec![
                ("http://e1:3".to_string(), None),
                ("http://e2:6".to_string(), None),
            ]
        );
        assert_eq!(
            line.clap_args,
            [
                "smg",
                "--pd-disaggregation",
                "--decode",
                "http://d1:5",
                "--prefill"
            ]
        );
    }

    /// A flag whose value has a required shape is refused by name when the
    /// shape is wrong, where it used to be dropped or passed on in silence;
    /// the well-formed spellings pass.
    #[test]
    fn malformed_flag_values_are_refused_by_name() {
        for (args, field) in [
            (
                vec!["--cors-allowed-origins", "http://[bad"],
                "cors_allowed_origins",
            ),
            (
                vec!["--cors-allowed-origins", "example.com"],
                "cors_allowed_origins",
            ),
            (
                vec!["--cors-allowed-origins", "https://example.com/"],
                "cors_allowed_origins",
            ),
            (
                vec!["--cors-allowed-origins", "https://Example.com"],
                "cors_allowed_origins",
            ),
            (
                vec!["--cors-allowed-origins", "https://example.com:443"],
                "cors_allowed_origins",
            ),
            (
                vec!["--cors-allowed-origins", "https://user@example.com"],
                "cors_allowed_origins",
            ),
            (
                vec!["--health-check-endpoint", "health"],
                "health_check.endpoint",
            ),
            (
                vec!["--prometheus-duration-buckets", "2", "1"],
                "prometheus_duration_buckets",
            ),
            (
                vec!["--prometheus-duration-buckets", "nan"],
                "prometheus_duration_buckets",
            ),
            (
                vec!["--prometheus-duration-buckets", "0"],
                "prometheus_duration_buckets",
            ),
            (
                vec!["--storage-context-headers", "noeq"],
                "storage_context_headers",
            ),
            (
                vec!["--request-id-headers", "bad header"],
                "request_id_headers",
            ),
            (
                vec!["--cors-allowed-origins", "*", "https://example.com"],
                "cors_allowed_origins",
            ),
            (vec!["--api-key", ""], "api_key"),
            (vec!["--jwt-role-mapping", "noeq"], "jwt_role_mapping"),
            (vec!["--jwt-role-mapping", "ops=root"], "jwt_role_mapping"),
        ] {
            let err = cli_args_from(&args)
                .to_router_config(vec![], vec![])
                .expect_err(&args.join(" "));
            match err {
                ConfigError::InvalidValue { field: named, .. } => {
                    assert_eq!(named, field, "{}", args.join(" "));
                }
                other => panic!("{}: {other}", args.join(" ")),
            }
        }
        cli_args_from(&[
            "--cors-allowed-origins",
            "https://example.com:8443",
            "--health-check-endpoint",
            "/health",
            "--prometheus-duration-buckets",
            "0.1",
            "1",
            "--storage-context-headers",
            "x-tenant=tenant",
            "--request-id-headers",
            "x-request-id",
            "--api-key",
            "secret",
            "--jwt-role-mapping",
            "ops=admin",
        ])
        .to_router_config(vec![], vec![])
        .expect("well-formed values pass");
        cli_args_from(&["--cors-allowed-origins", "*"])
            .to_router_config(vec![], vec![])
            .expect("a lone '*' is the open policy");
    }

    /// A flag that only acts with a companion is named when the companion is
    /// absent, where it used to be ignored in silence; with the companion
    /// present nothing is named.
    #[test]
    fn flags_without_their_companion_are_named() {
        let prefill = [("http://127.0.0.1:1".to_string(), None)];
        let encode = [("http://127.0.0.1:3".to_string(), None)];
        let ignored = cli_args_from(&[
            "--decode",
            "http://127.0.0.1:2",
            "--decode-policy",
            "random",
            "--prefill-policy",
            "random",
            "--encode-policy",
            "random",
            "--selector",
            "a=b",
            "--priority-scheduler-config",
            "/dev/null",
        ])
        .ignored_companions(&prefill, &encode);
        for flag in [
            "--prefill",
            "--decode",
            "--decode-policy",
            "--prefill-policy",
            "--encode",
            "--encode-policy",
            "--selector",
            "--priority-scheduler-config",
        ] {
            assert!(
                ignored.iter().any(|m| m.starts_with(flag)),
                "{flag} is named in {ignored:?}"
            );
        }
        let both = cli_args_from(&[
            "--worker-urls",
            "http://127.0.0.1:1",
            "--service-discovery",
            "--selector",
            "a=b",
        ])
        .ignored_companions(&[], &[]);
        assert!(
            both.iter()
                .any(|m| m.contains("--worker-urls") && m.contains("--service-discovery")),
            "{both:?}"
        );
        // Another provider ignores the selectors too, and the worker-urls
        // note names the provider that was given.
        let under_file = cli_args_from(&[
            "--discovery-provider",
            "file",
            "--selector",
            "a=b",
            "--worker-urls",
            "http://127.0.0.1:1",
        ])
        .ignored_companions(&[], &[]);
        assert!(
            under_file
                .iter()
                .any(|m| m.starts_with("--selector") && m.contains("--discovery-provider file")),
            "{under_file:?}"
        );
        assert!(
            under_file.iter().any(|m| m.starts_with("--worker-urls")
                && m.contains("--discovery-provider file")
                && !m.contains("--service-discovery")),
            "{under_file:?}"
        );
        let under_kubernetes =
            cli_args_from(&["--discovery-provider", "kubernetes", "--selector", "a=b"])
                .ignored_companions(&[], &[]);
        assert!(under_kubernetes.is_empty(), "{under_kubernetes:?}");
        let epd = cli_args_from(&["--epd-disaggregation", "--encode-policy", "random"])
            .ignored_companions(&[], &encode);
        assert!(epd.is_empty(), "{epd:?}");
        let paired = cli_args_from(&[
            "--pd-disaggregation",
            "--decode",
            "http://127.0.0.1:2",
            "--decode-policy",
            "random",
            "--service-discovery",
            "--selector",
            "a=b",
            "--priority-scheduler-enabled",
            "--priority-scheduler-config",
            "/dev/null",
        ])
        .ignored_companions(&prefill, &[]);
        assert!(paired.is_empty(), "{paired:?}");
    }

    /// The arguments of `smg launch ...`, the form a deployment uses.
    fn launch_args_from(args: &[&str]) -> CliArgs {
        let argv: Vec<String> = ["smg", "launch"]
            .into_iter()
            .map(str::to_string)
            .chain(args.iter().map(|s| (*s).to_string()))
            .collect();
        match Cli::parse_from(argv).command {
            Some(Commands::Launch { args }) => args,
            None => panic!("the launch subcommand"),
        }
    }

    /// A flag whose documented domain has a negative sentinel takes it in the
    /// space-separated spelling as in the `=` one, with and without the
    /// `launch` subcommand, and a flag without such a domain refuses the
    /// value by name instead of reporting an unexpected argument.
    #[test]
    fn negative_sentinels_parse_in_both_spellings() {
        let spaced = [
            "--max-concurrent-requests",
            "-1",
            "--prefill-max-inflight-requests-per-worker",
            "-1",
            "--redis-retention-days",
            "-1",
            "--rate-limit-tokens-per-second",
            "-1",
        ];
        // `parse_from` exits the process on a refusal: the sentinels are
        // tried first, so a refusal reads as the parser's message.
        for argv in [vec!["smg"], vec!["smg", "launch"]] {
            let argv: Vec<&str> = argv.into_iter().chain(spaced).collect();
            if let Err(e) = Cli::try_parse_from(argv.iter().copied()) {
                panic!("the sentinels are refused: {e}");
            }
        }
        for args in [cli_args_from(&spaced), launch_args_from(&spaced)] {
            assert_eq!(args.serving.max_concurrent_requests, -1);
            assert_eq!(args.workers.prefill_max_inflight_requests_per_worker, -1);
            assert_eq!(args.integrations.redis_retention_days, Some(-1));
            // Refused later by the validator, by name, as a negative rate.
            assert_eq!(args.serving.rate_limit_tokens_per_second, Some(-1));
        }
        assert_eq!(
            launch_args_from(&["--max-concurrent-requests=-1"])
                .serving
                .max_concurrent_requests,
            -1
        );
        for flag in ["--oracle-pool-min", "--oracle-pool-max"] {
            for argv in [vec!["smg", flag, "-1"], vec!["smg", "launch", flag, "-1"]] {
                let err = Cli::try_parse_from(argv).expect_err("an unsigned pool size refuses -1");
                let text = err.to_string();
                assert!(
                    text.contains(flag) && !text.contains("unexpected argument"),
                    "the refusal names the flag: {text}"
                );
            }
        }
    }

    /// The bound on a history store call is five seconds unless the flag of
    /// the backend sets it; zero is refused with the backend's other checks.
    #[test]
    fn store_call_bounds_default_to_five_seconds_and_follow_their_flags() {
        let stores = [
            "--postgres-db-url",
            "postgres://u:p@h:5432/d",
            "--redis-url",
            "redis://h:6379",
        ];
        let args = cli_args_from(&stores);
        assert_eq!(args.build_postgres_config(None).unwrap().op_timeout_secs, 5);
        assert_eq!(args.build_redis_config(None).unwrap().op_timeout_secs, 5);

        let mut argv = stores.to_vec();
        argv.extend([
            "--postgres-op-timeout-secs",
            "12",
            "--redis-op-timeout-secs",
            "3",
        ]);
        let args = cli_args_from(&argv);
        assert_eq!(
            args.build_postgres_config(None).unwrap().op_timeout_secs,
            12
        );
        assert_eq!(args.build_redis_config(None).unwrap().op_timeout_secs, 3);

        let mut argv = stores.to_vec();
        argv.extend([
            "--postgres-op-timeout-secs",
            "0",
            "--redis-op-timeout-secs",
            "0",
        ]);
        let args = cli_args_from(&argv);
        let err = args.build_postgres_config(None).unwrap_err().to_string();
        assert!(err.contains("op_timeout_secs"), "{err}");
        let err = args.build_redis_config(None).unwrap_err().to_string();
        assert!(err.contains("op_timeout_secs"), "{err}");
    }

    /// `-1` is the one spelling of persistent Redis records on the command
    /// line; 0 and any other negative value are refused by name.
    #[test]
    fn a_redis_retention_other_than_minus_one_below_one_is_refused() {
        let persistent =
            cli_args_from(&["--redis-url", "redis://h:6379", "--redis-retention-days=-1"]);
        assert_eq!(
            persistent.build_redis_config(None).unwrap().retention_days,
            None
        );
        for value in ["-2", "0"] {
            let flag = format!("--redis-retention-days={value}");
            let args = cli_args_from(&["--redis-url", "redis://h:6379", flag.as_str()]);
            match args.build_redis_config(None) {
                Err(ConfigError::InvalidValue {
                    field,
                    value: shown,
                    reason,
                }) => {
                    assert_eq!(field, "redis_retention_days");
                    assert_eq!(shown, value);
                    assert!(reason.contains("-1 keeps records persistent"), "{reason}");
                }
                other => panic!("expected {value} refused by name, got {other:?}"),
            }
        }
    }

    /// Mesh bind and advertise hosts go through the shared bind-host rule, so
    /// IPv6 works with or without brackets, as it does for `--host`.
    #[test]
    fn mesh_hosts_accept_both_ipv6_spellings() {
        for (host, advertise, expected_bind, expected_advertise) in [
            ("::", "::1", "[::]:39527", "[::1]:39527"),
            ("[::]", "[::1]", "[::]:39527", "[::1]:39527"),
            ("0.0.0.0", "127.0.0.1", "0.0.0.0:39527", "127.0.0.1:39527"),
        ] {
            let mesh = cli_args_from(&[
                "--enable-mesh",
                "--mesh-host",
                host,
                "--mesh-advertise-host",
                advertise,
            ])
            .build_mesh_server_config()
            .unwrap()
            .unwrap();
            assert_eq!(mesh.bind_addr.to_string(), expected_bind, "host {host}");
            assert_eq!(
                mesh.advertise_addr.to_string(),
                expected_advertise,
                "advertise {advertise}"
            );
        }
        let err = cli_args_from(&["--enable-mesh", "--mesh-host", "mesh-host"])
            .build_mesh_server_config()
            .err()
            .expect("a mesh host that is not an IP literal must be rejected");
        assert!(
            err.to_string().contains("invalid bind host 'mesh-host'"),
            "got: {err}"
        );
    }

    /// Three files for the mesh TLS flags (their content is not read here).
    fn mesh_tls_files(tag: &str) -> [String; 3] {
        let dir = std::env::temp_dir().join(format!("smg-mesh-tls-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        ["ca.pem", "node.pem", "node-key.pem"].map(|name| {
            let path = dir.join(name);
            std::fs::write(&path, b"test").unwrap();
            path.to_string_lossy().into_owned()
        })
    }

    #[test]
    fn mesh_tls_flags_configure_mtls_with_client_certificates_required() {
        let [ca, cert, key] = mesh_tls_files("all");
        let mesh = cli_args_from(&[
            "--enable-mesh",
            "--mesh-host",
            "127.0.0.1",
            "--mesh-tls-ca-cert",
            &ca,
            "--mesh-tls-cert",
            &cert,
            "--mesh-tls-key",
            &key,
        ])
        .build_mesh_server_config()
        .unwrap()
        .unwrap();

        let mtls = mesh.mtls_config.expect("the three flags configure mTLS");
        assert_eq!(mtls.ca_cert_path.to_string_lossy(), ca);
        assert_eq!(mtls.server_cert_path.to_string_lossy(), cert);
        assert_eq!(mtls.server_key_path.to_string_lossy(), key);
        assert!(mtls.require_client_cert, "peers must present a certificate");

        let plain = cli_args_from(&["--enable-mesh", "--mesh-host", "127.0.0.1"])
            .build_mesh_server_config()
            .unwrap()
            .unwrap();
        assert!(plain.mtls_config.is_none(), "no flags, no mTLS");
    }

    #[test]
    fn the_mesh_tls_flags_come_together() {
        let [ca, cert, _key] = mesh_tls_files("partial");
        for partial in [
            vec!["--mesh-tls-cert", cert.as_str()],
            vec![
                "--mesh-tls-ca-cert",
                ca.as_str(),
                "--mesh-tls-cert",
                cert.as_str(),
            ],
        ] {
            let argv: Vec<&str> = ["smg", "--enable-mesh", "--mesh-host", "127.0.0.1"]
                .into_iter()
                .chain(partial.iter().copied())
                .collect();
            let err = Cli::try_parse_from(argv).expect_err("a partial set is rejected");
            assert!(
                err.to_string().contains("--mesh-tls-"),
                "the error names the missing flag: {err}"
            );
        }
    }

    #[test]
    fn a_missing_mesh_tls_file_is_a_configuration_error() {
        let [ca, cert, _key] = mesh_tls_files("missing");
        let err = cli_args_from(&[
            "--enable-mesh",
            "--mesh-host",
            "127.0.0.1",
            "--mesh-tls-ca-cert",
            &ca,
            "--mesh-tls-cert",
            &cert,
            "--mesh-tls-key",
            "/nonexistent/node-key.pem",
        ])
        .build_mesh_server_config()
        .err()
        .expect("an unreadable key file is rejected at start-up");
        assert!(
            err.to_string().contains("mesh_tls_key"),
            "the error names the flag: {err}"
        );
    }

    /// With no `--prometheus-host`, the metrics listener follows the serving
    /// listener's address family, so an IPv6-bound gateway is scraped over
    /// IPv6 too; an explicit value is kept as written.
    #[test]
    fn metrics_host_defaults_to_the_serving_family() {
        for (args, expected) in [
            (vec!["--host", "[::]"], "::"),
            (vec!["--host", "::"], "::"),
            (vec!["--host", "[fd00::1]"], "::"),
            (vec![], "0.0.0.0"),
            (vec!["--host", "127.0.0.1"], "0.0.0.0"),
            (
                vec!["--host", "[::]", "--prometheus-host", "127.0.0.1"],
                "127.0.0.1",
            ),
            (vec!["--host", "0.0.0.0", "--prometheus-host", "::"], "::"),
        ] {
            let cli = cli_args_from(&args);
            assert_eq!(cli.metrics_host(), expected, "args {args:?}");
            let router_config = cli.to_router_config(vec![], vec![]).unwrap();
            assert_eq!(
                router_config.metrics.unwrap().host,
                expected,
                "args {args:?}"
            );
        }
    }

    /// A malformed `--control-plane-api-keys` entry or an unknown role used to
    /// be dropped with a stderr line, which started the control plane open:
    /// every such entry is now a configuration error, a valid one still works.
    #[test]
    fn control_plane_api_key_entries_fail_closed() {
        for bad in [
            &["--control-plane-api-keys", "bad"][..],
            &["--control-plane-api-keys", "i:n:bogus:s3cr3t"][..],
            // the fields out of order: the role field is the credential
            &[
                "--control-plane-api-keys",
                "ops:Ops Key:sk-live-abc123:admin",
            ][..],
            &["--control-plane-api-keys", ""][..],
            &[
                "--control-plane-api-keys",
                "i:n:admin:s3cr3t",
                "--control-plane-api-keys",
                "bad",
            ][..],
        ] {
            let err = cli_args_from(bad)
                .build_control_plane_auth_config()
                .expect_err("a malformed control-plane key entry must refuse the start");
            let text = err.to_string();
            assert!(text.contains("control-plane-api-keys"), "{bad:?}: {text}");
            assert!(
                !text.contains("s3cr3t")
                    && !text.contains("sk-live-abc123")
                    && !text.contains("'bad'"),
                "the entry (it carries the credential) must not be echoed: {text}"
            );
        }
        let config = cli_args_from(&["--control-plane-api-keys", "i:n:admin:s3cr3t"])
            .build_control_plane_auth_config()
            .expect("a valid entry");
        assert!(config.is_enabled());
        assert_eq!(config.api_keys.len(), 1);
    }

    /// `--jwt-issuer` without `--jwt-audience` (and the reverse) used to
    /// disable JWT authentication silently; both halves are required together.
    #[test]
    fn jwt_issuer_and_audience_are_required_together() {
        assert!(cli_args_from(&["--jwt-issuer", "https://issuer.example"])
            .build_control_plane_auth_config()
            .is_err());
        assert!(cli_args_from(&["--jwt-audience", "aud"])
            .build_control_plane_auth_config()
            .is_err());
        let config = cli_args_from(&[
            "--jwt-issuer",
            "https://issuer.example",
            "--jwt-audience",
            "aud",
            "--jwt-jwks-uri",
            "https://issuer.example/jwks",
        ])
        .build_control_plane_auth_config()
        .expect("a complete JWT configuration");
        assert!(config.jwt.is_some());
        assert!(cli_args_from(&[])
            .build_control_plane_auth_config()
            .expect("no auth configured")
            .jwt
            .is_none());
    }

    /// A malformed `--jwt-role-mapping` entry used to be dropped with a stderr
    /// line; an empty map grants the gateway's admin role to any token whose
    /// roles claim says `admin`, so every such entry is a configuration error.
    #[test]
    fn jwt_role_mapping_entries_fail_closed() {
        let jwt = [
            "--jwt-issuer",
            "https://issuer.example",
            "--jwt-audience",
            "aud",
        ];
        for bad in ["admin:user", "gw-admins=root", "gw-admins"] {
            let args: Vec<&str> = jwt
                .iter()
                .copied()
                .chain(["--jwt-role-mapping", bad])
                .collect();
            let err = cli_args_from(&args)
                .build_control_plane_auth_config()
                .expect_err("a malformed role mapping must refuse the start");
            assert!(err.to_string().contains("jwt-role-mapping"), "{bad}: {err}");
        }
        let args: Vec<&str> = jwt
            .iter()
            .copied()
            .chain(["--jwt-role-mapping", "gw-admins=admin"])
            .collect();
        let config = cli_args_from(&args)
            .build_control_plane_auth_config()
            .expect("a valid role mapping");
        let jwt_config = config.jwt.expect("JWT configured");
        assert!(matches!(
            jwt_config.role_mapping.get("gw-admins"),
            Some(Role::Admin)
        ));
    }

    /// A grouped ZMQ handshake needs at least one engine, so `0` (and any
    /// non-positive value) must be rejected at parse time rather than silently
    /// degrading to an ungrouped worker.
    #[test]
    fn zmq_engine_count_rejects_non_positive() {
        assert_eq!(
            cli_args_from(&["--zmq-engine-count", "2"])
                .routing
                .zmq_engine_count,
            Some(2)
        );
        assert_eq!(cli_args_from(&[]).routing.zmq_engine_count, None);

        for bad in ["0", "-1"] {
            let argv = ["smg", "--zmq-engine-count", bad];
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "--zmq-engine-count {bad} should be rejected"
            );
        }
    }

    /// The deprecated indexer prune flags are still accepted and flow into
    /// `RouterConfig`, where nothing reads them but the startup warning.
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

    /// `--kv-index` is deprecated: the chain index is the only KV index, so
    /// `chain` is accepted (and flows into `RouterConfig` for the startup
    /// warning), the removed `positional` index is refused with the reason,
    /// and any other spelling is an invalid value.
    #[test]
    fn kv_index_flag_takes_chain_only() {
        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.kv_index, None);

        let cli = cli_args_from(&["--kv-index", "Chain"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.kv_index.as_deref(), Some("chain"));
        router_config.validate().unwrap();

        let err = Cli::try_parse_from(["smg", "--kv-index", "positional"])
            .expect_err("the removed positional index is refused");
        let text = err.to_string();
        assert!(
            text.contains("positional KV index was removed") && text.contains("--kv-index"),
            "the refusal names the removal and the flag: {text}"
        );
        for value in ["run", "tree"] {
            let err = Cli::try_parse_from(["smg", "--kv-index", value])
                .expect_err("only chain is accepted");
            assert!(
                err.to_string().contains("takes chain only"),
                "{value}: {err}"
            );
        }
    }

    /// The retry-buffer cap must flow through both conversion paths.
    #[test]
    fn max_buffered_request_bytes_flows_into_both_config_paths() {
        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.max_buffered_request_bytes, 1_048_576);

        let cli = cli_args_from(&["--max-buffered-request-bytes", "4096"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.max_buffered_request_bytes, 4096);
        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.router_config.max_buffered_request_bytes, 4096);

        let zeroed = cli_args_from(&["--max-buffered-request-bytes", "0"])
            .to_router_config(vec![], vec![])
            .unwrap();
        assert_eq!(zeroed.max_buffered_request_bytes, 0);
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

    /// The PD admission wait is a router-only setting and must flow into
    /// `RouterConfig`, where the dispatch path latches it at startup.
    #[test]
    fn pd_admission_wait_flag_flows_into_router_config() {
        let cli = cli_args_from(&["--pd-admission-wait-secs", "5"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.pd_admission_wait_secs, 5);
        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.router_config.pd_admission_wait_secs, 5);

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.pd_admission_wait_secs, 30);
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

    /// The connection-level timeouts of the HTTP edge are router settings
    /// and must flow into `RouterConfig`, with zero kept as "disabled".
    #[test]
    fn connection_timeout_flags_flow_into_router_config() {
        let cli = cli_args_from(&[
            "--header-read-timeout-secs",
            "5",
            "--idle-connection-timeout-secs",
            "0",
        ]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.header_read_timeout_secs, 5);
        assert_eq!(router_config.idle_connection_timeout_secs, 0);

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.header_read_timeout_secs, 30);
        assert_eq!(defaults.idle_connection_timeout_secs, 90);
    }

    /// The SSE keep-alive interval is a router-only setting and must flow
    /// into `RouterConfig`; it is on by default.
    #[test]
    fn sse_keepalive_flag_flows_into_router_config() {
        let cli = cli_args_from(&["--sse-keepalive-secs", "40"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.sse_keepalive_secs, 40);

        let off = cli_args_from(&["--sse-keepalive-secs", "0"]);
        assert_eq!(
            off.to_router_config(vec![], vec![])
                .unwrap()
                .sse_keepalive_secs,
            0
        );

        let defaults = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(defaults.sse_keepalive_secs, 15);
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

    /// Mesh-router discovery has its own task and lifetime, but is still
    /// configured through Kubernetes `discovery`, which only a selected
    /// Kubernetes provider populates. `--router-selector` alone therefore
    /// still does nothing until router discovery gets its own config surface.
    #[test]
    fn router_selector_alone_does_not_yet_configure_mesh_discovery() {
        let cli = cli_args_from(&["--router-selector", "role=router"]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(router.discovery.is_none());

        let server = cli.to_server_config(router).unwrap();
        assert!(server.service_discovery_config.is_none());
        assert!(server.mesh_discovery_config.is_none());
    }

    #[test]
    fn mesh_discovery_is_absent_without_a_router_selector() {
        let cli = cli_args_from(&["--service-discovery", "--selector", "app=worker"]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        let server = cli.to_server_config(router).unwrap();
        assert!(server.service_discovery_config.is_some());
        assert!(server.mesh_discovery_config.is_none());
    }

    /// The router selector reaches the new independent mesh task unchanged.
    #[test]
    fn service_discovery_router_selector_reaches_mesh_discovery() {
        let cli = cli_args_from(&[
            "--service-discovery",
            "--selector",
            "app=worker",
            "--router-selector",
            "role=router",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        let server = cli.to_server_config(router).unwrap();
        let mesh = server.mesh_discovery_config.expect("mesh discovery config");
        assert_eq!(
            mesh.router_selector.get("role").map(String::as_str),
            Some("router")
        );
    }

    /// Port 0 parses as a socket address, so nothing downstream rejects it:
    /// it would be advertised to peers and used as the fallback mesh port for
    /// router Pods with no usable annotation, publishing `IP:0` peers.
    #[test]
    fn mesh_port_zero_is_rejected() {
        let cli = cli_args_from(&[
            "--enable-mesh",
            "--mesh-advertise-host",
            "10.0.0.1",
            "--mesh-port",
            "0",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        // `ServerConfig` is not `Debug`, so unwrap the error by hand.
        let Err(err) = cli.to_server_config(router) else {
            panic!("mesh port 0 must be rejected");
        };
        assert!(
            format!("{err}").contains("mesh port cannot be 0"),
            "unexpected error: {err}"
        );
    }

    /// The same configuration with a real port still builds, so the guard is
    /// not rejecting every meshed setup.
    #[test]
    fn mesh_port_nonzero_is_accepted() {
        let cli = cli_args_from(&[
            "--enable-mesh",
            "--mesh-advertise-host",
            "10.0.0.1",
            "--mesh-port",
            "39527",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        let server = cli.to_server_config(router).unwrap();
        let mesh = server.mesh_server_config.expect("mesh server config");
        assert_eq!(mesh.advertise_addr.port(), 39527);
    }

    #[test]
    fn kv_annotation_flags_flow_into_both_configs() {
        let cli = cli_args_from(&[
            "--service-discovery",
            "--selector",
            "app=worker",
            "--kv-connector-annotation",
            "example.com/connector",
            "--kv-engine-id-annotation",
            "example.com/engine-id",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        let Some(DiscoveryConfig::Kubernetes(discovery)) = &router.discovery else {
            panic!("expected Kubernetes discovery");
        };
        assert_eq!(discovery.kv_connector_annotation, "example.com/connector");
        assert_eq!(discovery.kv_engine_id_annotation, "example.com/engine-id");

        let server = cli.to_server_config(router).unwrap();
        let Some(RuntimeDiscoveryConfig::Kubernetes(discovery)) = &server.service_discovery_config
        else {
            panic!("expected Kubernetes runtime discovery");
        };
        assert_eq!(discovery.kv_connector_annotation, "example.com/connector");
        assert_eq!(discovery.kv_engine_id_annotation, "example.com/engine-id");

        let defaults = cli_args_from(&["--service-discovery", "--selector", "app=worker"])
            .to_router_config(vec![], vec![])
            .unwrap();
        let Some(DiscoveryConfig::Kubernetes(defaults)) = &defaults.discovery else {
            panic!("expected Kubernetes discovery");
        };
        assert_eq!(defaults.kv_connector_annotation, "smg.ai/kv-connector");
        assert_eq!(defaults.kv_engine_id_annotation, "smg.ai/kv-engine-id");
    }

    /// `--discovery-provider kubernetes` is `--service-discovery` under its
    /// new name: the same detail flags build the same configuration, worker
    /// and router discovery alike.
    #[test]
    fn discovery_provider_kubernetes_matches_service_discovery() {
        let details = [
            "--selector",
            "app=worker",
            "--service-discovery-port",
            "9000",
            "--service-discovery-namespace",
            "prod",
            "--kv-connector-annotation",
            "example.com/connector",
            "--model-id-from",
            "namespace",
            "--router-selector",
            "role=router",
        ];
        let build = |selection: &[&str]| {
            let args: Vec<&str> = selection.iter().chain(details.iter()).copied().collect();
            let cli = cli_args_from(&args);
            let router = cli.to_router_config(vec![], vec![]).unwrap();
            let discovery = router.discovery.clone();
            let server = cli.to_server_config(router).unwrap();
            (
                discovery,
                format!("{:?}", server.service_discovery_config),
                format!("{:?}", server.mesh_discovery_config),
            )
        };

        let legacy = build(&["--service-discovery"]);
        let tagged = build(&["--discovery-provider", "kubernetes"]);
        assert!(matches!(legacy.0, Some(DiscoveryConfig::Kubernetes(_))));
        assert_eq!(legacy, tagged);
    }

    /// Two spellings of one choice: giving both is a usage error, not a
    /// precedence rule to remember.
    #[test]
    fn service_discovery_and_discovery_provider_conflict() {
        let err = Cli::try_parse_from([
            "smg",
            "--service-discovery",
            "--discovery-provider",
            "kubernetes",
        ])
        .expect_err("both spellings must be rejected");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    /// `--discovery-provider file` builds a file configuration with the
    /// interval defaulted. Worker discovery runs from it; router discovery,
    /// configured only inside Kubernetes discovery, does not.
    #[test]
    fn discovery_provider_file_builds_a_file_config() {
        let cli = cli_args_from(&[
            "--discovery-provider",
            "file",
            "--discovery-file",
            "/run/smg/workers.json",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router.discovery,
            Some(DiscoveryConfig::File(FileDiscoveryConfig {
                path: "/run/smg/workers.json".into(),
                check_interval_secs: DEFAULT_FILE_DISCOVERY_CHECK_INTERVAL_SECS,
            }))
        );
        // Recovery by removal works here too: the next reread re-adds the worker.
        assert!(router.health_check.remove_unhealthy_workers);

        let server = cli.to_server_config(router).unwrap();
        assert!(matches!(
            server.service_discovery_config,
            Some(RuntimeDiscoveryConfig::File(_))
        ));
        assert!(server.mesh_discovery_config.is_none());

        let tuned = cli_args_from(&[
            "--discovery-provider",
            "file",
            "--discovery-file",
            "w.json",
            "--discovery-check-interval-secs",
            "5",
        ])
        .to_router_config(vec![], vec![])
        .unwrap();
        assert!(matches!(
            tuned.discovery,
            Some(DiscoveryConfig::File(FileDiscoveryConfig {
                check_interval_secs: 5,
                ..
            }))
        ));
    }

    #[test]
    fn file_provider_needs_a_manifest() {
        let err = cli_args_from(&["--discovery-provider", "file"])
            .to_router_config(vec![], vec![])
            .unwrap_err();
        assert!(
            matches!(err, ConfigError::MissingRequired { ref field } if field == "--discovery-file"),
            "{err}"
        );
    }

    /// The file flags configure nothing else, so a forgotten
    /// `--discovery-provider file` is reported instead of ignored.
    #[test]
    fn file_flags_need_the_file_provider() {
        for args in [
            &["--discovery-file", "w.json"][..],
            &["--discovery-check-interval-secs", "5"][..],
            &[
                "--service-discovery",
                "--selector",
                "app=w",
                "--discovery-file",
                "w.json",
            ][..],
        ] {
            let err = cli_args_from(args)
                .to_router_config(vec![], vec![])
                .unwrap_err();
            assert!(
                matches!(err, ConfigError::IncompatibleConfig { .. }),
                "{args:?}: {err}"
            );
        }
    }

    /// Under the file provider the Kubernetes flags are ignored, the router
    /// selector included, as they are without discovery.
    #[test]
    fn kubernetes_flags_are_ignored_under_the_file_provider() {
        let cli = cli_args_from(&[
            "--discovery-provider",
            "file",
            "--discovery-file",
            "w.json",
            "--selector",
            "app=w",
            "--service-discovery-namespace",
            "prod",
            "--router-selector",
            "role=router",
        ]);
        let router = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(matches!(router.discovery, Some(DiscoveryConfig::File(_))));
        let server = cli.to_server_config(router).unwrap();
        assert!(server.mesh_discovery_config.is_none());
    }

    /// IGW mode follows the selected provider, whichever spelling selected it.
    #[test]
    fn selecting_a_discovery_provider_enables_igw() {
        for selection in [
            &["--service-discovery"][..],
            &["--discovery-provider", "kubernetes"][..],
            &["--discovery-provider", "file", "--discovery-file", "w.json"][..],
        ] {
            let mut cli = cli_args_from(selection);
            assert!(cli.enable_igw_for_discovery(), "{selection:?}");
            assert!(cli.routing.enable_igw, "{selection:?}");
        }

        let mut already_on = cli_args_from(&["--service-discovery", "--enable-igw"]);
        assert!(!already_on.enable_igw_for_discovery());
        assert!(already_on.routing.enable_igw);

        let mut no_discovery = cli_args_from(&[]);
        assert!(!no_discovery.enable_igw_for_discovery());
        assert!(!no_discovery.routing.enable_igw);
    }

    /// `--worker-auto-recovery` defaults to the `--service-discovery`
    /// setting: recovery works by removal plus discovery re-registration, so
    /// it is on exactly when discovery can complete that loop, and off when
    /// removal would permanently shrink a static fleet. Explicit values win
    /// in both directions.
    #[test]
    fn worker_auto_recovery_follows_service_discovery_by_default() {
        let derived_on = cli_args_from(&["--service-discovery", "--selector", "app=w"])
            .to_router_config(vec![], vec![])
            .unwrap();
        assert!(derived_on.health_check.remove_unhealthy_workers);

        let derived_on_tagged =
            cli_args_from(&["--discovery-provider", "kubernetes", "--selector", "app=w"])
                .to_router_config(vec![], vec![])
                .unwrap();
        assert!(derived_on_tagged.health_check.remove_unhealthy_workers);

        let derived_off = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert!(!derived_off.health_check.remove_unhealthy_workers);

        let forced_off = cli_args_from(&[
            "--service-discovery",
            "--selector",
            "app=w",
            "--remove-unhealthy-workers=false",
        ])
        .to_router_config(vec![], vec![])
        .unwrap();
        assert!(!forced_off.health_check.remove_unhealthy_workers);

        let forced_on = cli_args_from(&["--remove-unhealthy-workers"])
            .to_router_config(vec![], vec![])
            .unwrap();
        assert!(forced_on.health_check.remove_unhealthy_workers);
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

    /// `--admin-port` and `--admin-host` flow into `RouterConfig`, the copy
    /// the validator checks and `startup` reads, and reach `ServerConfig`
    /// nested in it; unset, the control plane stays on the data listener.
    #[test]
    fn admin_listener_flags_flow_into_the_router_config() {
        let cli = cli_args_from(&["--admin-port", "9001", "--admin-host", "127.0.0.1"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.admin_port, Some(9001));
        assert_eq!(router_config.admin_host.as_deref(), Some("127.0.0.1"));
        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.router_config.admin_port, Some(9001));
        assert_eq!(
            server_config.router_config.admin_host.as_deref(),
            Some("127.0.0.1")
        );

        let router_config = cli_args_from(&[]).to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.admin_port, None);
        assert_eq!(router_config.admin_host, None);
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

    /// Protection is on by default on both signals, steering only, on both
    /// config paths.
    #[test]
    fn worker_overload_protection_defaults_to_on_and_steering_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(router_config.worker_overload_protection);
        assert_eq!(router_config.worker_overload_waiting_requests, Some(8));
        assert_eq!(router_config.worker_overload_token_usage, Some(0.8));
        assert!(!router_config.worker_overload_shed);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(server_config.router_config.worker_overload_protection);
        assert_eq!(
            server_config.router_config.worker_overload_waiting_requests,
            Some(8)
        );
        assert_eq!(
            server_config.router_config.worker_overload_token_usage,
            Some(0.8)
        );
        assert!(!server_config.router_config.worker_overload_shed);
    }

    /// The two opt-outs: `--disable-worker-overload-protection` switches the
    /// gateway thresholds off, `--worker-overload-shed` turns steering into
    /// refusal; both must survive into `ServerConfig.router_config`.
    #[test]
    fn worker_overload_opt_outs_flow_into_both_configs() {
        let cli = cli_args_from(&["--disable-worker-overload-protection"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(!router_config.worker_overload_protection);
        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(!server_config.router_config.worker_overload_protection);

        let cli = cli_args_from(&["--worker-overload-shed"]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(router_config.worker_overload_shed);
        assert!(router_config.worker_overload_protection);
        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(server_config.router_config.worker_overload_shed);
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
        // The flag changes nothing: the defaults are already on.
        assert_eq!(router_config.worker_overload_waiting_requests, Some(8));
        assert_eq!(router_config.worker_overload_token_usage, Some(0.8));

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

    /// Defaults: protection on, monitoring on (opt-out false) — both carried
    /// by the defaults here.
    #[test]
    fn overload_protection_and_monitoring_default_on_in_both_configs() {
        let cli = cli_args_from(&[]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert!(router_config.worker_overload_protection);
        assert!(!router_config.disable_load_monitoring);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert!(server_config.router_config.worker_overload_protection);
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

    /// The media placement and engine-side media flags must reach both
    /// `RouterConfig` and the wrapped `ServerConfig.router_config`.
    #[test]
    fn mm_settings_flow_into_both_configs() {
        let cli = cli_args_from(&[
            "--mm-processing",
            "worker",
            "--mm-pixel-cache-mb",
            "256",
            "--mm-pixel-rdma",
            "--rdma-listen-ip",
            "10.0.0.7",
            "--rdma-slot-ttl-s",
            "600",
            "--log-mm-timing",
            "--mm-allowed-media-domains",
            "media.example,cdn.example",
            "--mm-allow-private-media-hosts",
        ]);
        let domains = Some(vec!["media.example".to_string(), "cdn.example".to_string()]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.mm_processing, Some(MmProcessingMode::Worker));
        assert_eq!(router_config.mm_pixel_cache_mb, Some(256));
        assert!(router_config.mm_pixel_rdma);
        assert_eq!(router_config.rdma_listen_ip.as_deref(), Some("10.0.0.7"));
        assert_eq!(router_config.rdma_slot_ttl_s, Some(600));
        assert!(router_config.log_mm_timing);
        assert_eq!(router_config.mm_allowed_media_domains, domains);
        assert!(router_config.mm_allow_private_media_hosts);

        let server_config = cli.to_server_config(router_config).unwrap();
        let nested = &server_config.router_config;
        assert_eq!(nested.mm_processing, Some(MmProcessingMode::Worker));
        assert_eq!(nested.mm_pixel_cache_mb, Some(256));
        assert!(nested.mm_pixel_rdma);
        assert_eq!(nested.rdma_listen_ip.as_deref(), Some("10.0.0.7"));
        assert_eq!(nested.rdma_slot_ttl_s, Some(600));
        assert!(nested.log_mm_timing);
        assert_eq!(nested.mm_allowed_media_domains, domains);
        assert!(nested.mm_allow_private_media_hosts);
    }

    #[test]
    fn mm_settings_default_to_unset_in_both_configs() {
        let cli = cli_args_from(&[]);
        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(router_config.mm_processing, None);
        assert_eq!(router_config.mm_pixel_cache_mb, None);
        assert!(!router_config.mm_pixel_rdma);
        assert_eq!(router_config.rdma_listen_ip, None);
        assert_eq!(router_config.rdma_slot_ttl_s, None);
        assert!(!router_config.log_mm_timing);
        assert_eq!(router_config.mm_allowed_media_domains, None);
        assert!(!router_config.mm_allow_private_media_hosts);

        let server_config = cli.to_server_config(router_config).unwrap();
        assert_eq!(server_config.router_config.mm_processing, None);
        assert!(!server_config.router_config.log_mm_timing);
        assert!(!server_config.router_config.mm_allow_private_media_hosts);
    }

    #[test]
    fn mm_processing_rejects_unknown_modes_at_parse_time() {
        assert!(Cli::try_parse_from(["smg", "--mm-processing", "routers"]).is_err());
        assert!(Cli::try_parse_from(["smg", "--rdma-slot-ttl-s", "soon"]).is_err());
        assert!(Cli::try_parse_from(["smg", "--mm-pixel-cache-mb", "-1"]).is_err());
        let cli = cli_args_from(&["--mm-processing", "Router"]);
        assert_eq!(cli.workers.mm_processing, Some(MmProcessingMode::Router));
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
            "--multimodal-max-inflight-bytes",
            "2048",
            "--mm-per-request-image-limit",
            "128",
        ]);

        let router_config = cli.to_router_config(vec![], vec![]).unwrap();
        assert_eq!(
            router_config.multimodal_tensor_transport,
            Some(TransportMode::Shm),
            "transport mode must reach RouterConfig via to_router_config"
        );
        assert_eq!(router_config.multimodal_shm_min_bytes, Some(1024));
        assert_eq!(router_config.multimodal_max_inflight_bytes, Some(2048));
        assert_eq!(router_config.mm_per_request_image_limit, Some(128));

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
        assert_eq!(
            server_config.router_config.multimodal_max_inflight_bytes,
            Some(2048)
        );
        assert_eq!(
            server_config.router_config.mm_per_request_image_limit,
            Some(128)
        );

        // clap rejects a zero limit outright.
        assert!(Cli::try_parse_from(["smg", "--mm-per-request-image-limit", "0"]).is_err());
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

    /// Both values the validator bounds are checked on the command line's own
    /// path, after the flags are parsed and before the runtime is built: a
    /// count above the ceiling is refused naming the field and the
    /// ceiling, a listener address that is not an IP address naming its field
    /// and the value; the smallest count and a plain address pass.
    #[test]
    fn runtime_worker_threads_and_rdma_listen_ip_are_validated_on_the_cli_path() {
        // The builder validates as it builds the configuration, and the
        // configure phase validates once more before the runtime is built;
        // the refusal reads the same from either.
        let refusal = |args: &[&str]| match cli_args_from(args).to_router_config(vec![], vec![]) {
            Err(err) => err.to_string(),
            Ok(config) => config.validate().expect_err("refused").to_string(),
        };
        let text = refusal(&["--runtime-worker-threads", "100000"]);
        assert!(
            text.contains("runtime_worker_threads") && text.contains("Must be at most"),
            "{text}"
        );
        let text = refusal(&["--rdma-listen-ip", "999.1.1.1", "--mm-pixel-rdma"]);
        assert!(
            text.contains("rdma_listen_ip")
                && text.contains("999.1.1.1")
                && text.contains("expected an IP address"),
            "{text}"
        );
        cli_args_from(&[
            "--runtime-worker-threads",
            "1",
            "--rdma-listen-ip",
            "10.0.0.7",
            "--mm-pixel-rdma",
        ])
        .to_router_config(vec![], vec![])
        .unwrap()
        .validate()
        .unwrap();
    }

    /// The flag's help states the ceiling the validator refuses above and
    /// the multiple of the CPUs the start-up warns above, as the constants
    /// hold them.
    #[test]
    fn the_runtime_flag_help_names_the_bound() {
        use clap::CommandFactory;
        let help = Cli::command().render_long_help().to_string();
        let help = help.split_whitespace().collect::<Vec<_>>().join(" ");
        let ceiling = format!(
            "from 1 to {} on every host",
            smg::config::RUNTIME_WORKER_THREADS_CEILING
        );
        let advised = format!(
            "above {} x the CPUs available to the process starts with a warning",
            smg::config::RUNTIME_WORKER_THREADS_PER_CPU
        );
        assert!(help.contains(&ceiling) && help.contains(&advised), "{help}");
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

        let sglang = cli_args_from(&[
            "--backend",
            "sglang",
            "--worker-urls",
            "ipc:///tmp/smg-zmq/engine-0",
        ]);
        assert_eq!(
            sglang
                .to_router_config(vec![], vec![])
                .unwrap()
                .startup_worker_runtime_type,
            Some(RuntimeType::Sglang),
            "--backend sglang must pin the ZMQ startup worker runtime"
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

    // ==================== Secret sources ====================

    /// An environment lookup over a table: no test here touches the
    /// process environment.
    fn env_of<'a>(
        vars: &'a [(&'a str, &'a str)],
    ) -> impl Fn(&str) -> ConfigResult<Option<String>> + 'a {
        move |name| {
            Ok(vars
                .iter()
                .find(|(var, _)| *var == name)
                .map(|(_, value)| (*value).to_string()))
        }
    }

    fn resolved(args: &[&str], vars: &[(&str, &str)]) -> ConfigResult<CliArgs> {
        let mut cli = cli_args_from(args);
        cli.resolve_secret_sources(&env_of(vars))?;
        Ok(cli)
    }

    /// Every secret is read from its environment variable when its flag is
    /// absent, the tenant entries split at ',' and at line ends with the
    /// whitespace around them dropped, and the built configuration carries
    /// the values; the record names the environment as the source.
    #[test]
    fn secrets_are_read_from_the_environment_when_the_flags_are_absent() {
        let vars = [
            ("SMG_API_KEY", "env-shared-key"),
            (
                "SMG_TENANT_API_KEYS",
                "red:env-red, blue:env-blue\n green:env-green ,",
            ),
            (
                "SMG_POSTGRES_DB_URL",
                "postgresql://u:env-pg-pass@db.example:5432/smg",
            ),
            (
                "SMG_REDIS_URL",
                "redis://:env-redis-pass@cache.example:6379/0",
            ),
        ];

        let cli = resolved(&["--history-backend", "postgres"], &vars).expect("resolves");
        assert_eq!(cli.integrations.api_key.as_deref(), Some("env-shared-key"));
        assert_eq!(
            cli.integrations.tenant_api_keys,
            vec!["red:env-red", "blue:env-blue", "green:env-green"]
        );
        assert_eq!(
            cli.integrations.postgres_db_url.as_deref(),
            Some("postgresql://u:env-pg-pass@db.example:5432/smg")
        );
        assert_eq!(
            cli.integrations.secret_sources,
            SecretSources {
                api_key: Some(SecretSource::Environment),
                tenant_api_keys: Some((SecretSource::Environment, 3)),
                postgres_db_url: Some(SecretSource::Environment),
                redis_url: None,
            }
        );
        let config = cli
            .to_router_config(vec![], vec![])
            .expect("the resolved secrets build a configuration");
        assert_eq!(config.api_key.as_deref(), Some("env-shared-key"));
        assert_eq!(
            config
                .tenant_api_keys
                .iter()
                .map(|entry| (entry.tenant_id.as_str(), entry.key.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("red", "env-red"),
                ("blue", "env-blue"),
                ("green", "env-green")
            ]
        );
        assert_eq!(
            config.postgres.expect("a postgres store").db_url,
            "postgresql://u:env-pg-pass@db.example:5432/smg"
        );

        let cli = resolved(&["--history-backend", "redis"], &vars).expect("resolves");
        assert_eq!(
            cli.integrations.redis_url.as_deref(),
            Some("redis://:env-redis-pass@cache.example:6379/0")
        );
        assert_eq!(
            cli.integrations.secret_sources.redis_url,
            Some(SecretSource::Environment)
        );
        assert_eq!(cli.integrations.secret_sources.postgres_db_url, None);
        let config = cli.to_router_config(vec![], vec![]).expect("builds");
        assert_eq!(
            config.redis.expect("a redis store").url,
            "redis://:env-redis-pass@cache.example:6379/0"
        );
    }

    /// A value from the environment is trimmed as a file's content is: a
    /// Secret made from a file carries the editor's trailing newline into
    /// the variable, and no header could carry such a key.
    #[test]
    fn a_value_from_the_environment_is_trimmed_like_a_file() {
        let vars = [
            ("SMG_API_KEY", "env-shared-key\n"),
            (
                "SMG_POSTGRES_DB_URL",
                " postgresql://u:env-pg-pass@db.example:5432/smg\n",
            ),
        ];
        let cli = resolved(&["--history-backend", "postgres"], &vars).expect("resolves");
        assert_eq!(cli.integrations.api_key.as_deref(), Some("env-shared-key"));
        assert_eq!(
            cli.integrations.postgres_db_url.as_deref(),
            Some("postgresql://u:env-pg-pass@db.example:5432/smg")
        );
    }

    /// A variable set to bytes that are not UTF-8 is refused by name, not
    /// read as unset: the gateway would otherwise start with no serving key
    /// where the operator set one.
    #[test]
    fn a_variable_that_is_not_utf8_is_refused_not_passed_over() {
        assert_eq!(env_value("SMG_API_KEY", None).expect("unset"), None);
        assert_eq!(
            env_value("SMG_API_KEY", Some("k".into())).expect("set"),
            Some("k".to_string())
        );
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let raw = std::ffi::OsString::from_vec(vec![b'k', 0xff, b'y']);
            let err = env_value("SMG_API_KEY", Some(raw)).expect_err("not UTF-8");
            assert!(
                matches!(&err, ConfigError::InvalidValue { field, value, .. }
                    if field == "SMG_API_KEY" && value == "<not UTF-8>"),
                "{err}"
            );
        }
        let refusing = |name: &str| -> ConfigResult<Option<String>> {
            if name == "SMG_API_KEY" {
                Err(ConfigError::InvalidValue {
                    field: name.to_string(),
                    value: "<not UTF-8>".to_string(),
                    reason: "set, but its value is not UTF-8".to_string(),
                })
            } else {
                Ok(None)
            }
        };
        let mut cli = cli_args_from(&[]);
        let err = cli.resolve_secret_sources(&refusing).expect_err("refused");
        assert!(
            matches!(&err, ConfigError::InvalidValue { field, .. } if field == "SMG_API_KEY"),
            "{err}"
        );
        assert_eq!(cli.integrations.api_key, None);
    }

    /// The command line beats the environment for every secret, and the
    /// record says so.
    #[test]
    fn the_command_line_beats_the_environment() {
        let vars = [
            ("SMG_API_KEY", "env-shared-key"),
            ("SMG_TENANT_API_KEYS", "red:env-red"),
            (
                "SMG_POSTGRES_DB_URL",
                "postgresql://u:env@db.example:5432/smg",
            ),
        ];
        let cli = resolved(
            &[
                "--api-key",
                "flag-shared-key",
                "--tenant-api-key",
                "blue:flag-blue",
                "--history-backend",
                "postgres",
                "--postgres-db-url",
                "postgresql://u:flag@db.example:5432/smg",
            ],
            &vars,
        )
        .expect("resolves");

        assert_eq!(cli.integrations.api_key.as_deref(), Some("flag-shared-key"));
        assert_eq!(cli.integrations.tenant_api_keys, vec!["blue:flag-blue"]);
        assert_eq!(
            cli.integrations.postgres_db_url.as_deref(),
            Some("postgresql://u:flag@db.example:5432/smg")
        );
        assert_eq!(
            cli.integrations.secret_sources,
            SecretSources {
                api_key: Some(SecretSource::Flag),
                tenant_api_keys: Some((SecretSource::Flag, 1)),
                postgres_db_url: Some(SecretSource::Flag),
                redis_url: None,
            }
        );
    }

    /// Without a source, nothing is set and nothing is recorded.
    #[test]
    fn without_a_source_no_secret_is_set() {
        let cli = resolved(&["--history-backend", "postgres"], &[]).expect("resolves");

        assert_eq!(cli.integrations.api_key, None);
        assert!(cli.integrations.tenant_api_keys.is_empty());
        assert_eq!(cli.integrations.postgres_db_url, None);
        assert_eq!(cli.integrations.redis_url, None);
        assert_eq!(cli.integrations.secret_sources, SecretSources::default());
        assert!(!cli.integrations.secret_sources.any());
    }

    /// A secret file is read whole with its trailing newline dropped, a
    /// tenant key file one entry per line with blank and comment lines
    /// skipped; the file is named by its flag or by its environment
    /// variable, and the record names the file as the source.
    #[test]
    fn secrets_are_read_from_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let api_key = dir.path().join("api-key");
        let tenants = dir.path().join("tenants");
        let pg = dir.path().join("pg-url");
        let redis = dir.path().join("redis-url");
        std::fs::write(&api_key, "file-shared-key\n").unwrap();
        std::fs::write(
            &tenants,
            "# the tenants\r\n\nred:file-red\r\n  blue : file-blue  \n\n# trailing comment\n",
        )
        .unwrap();
        std::fs::write(&pg, "postgresql://u:file-pg@db.example:5432/smg").unwrap();
        std::fs::write(&redis, "  redis://:file-redis@cache.example:6379/1\n\n").unwrap();
        let (api_key, tenants, pg, redis) = (
            api_key.to_str().unwrap(),
            tenants.to_str().unwrap(),
            pg.to_str().unwrap(),
            redis.to_str().unwrap(),
        );

        let by_flag = resolved(
            &[
                "--api-key-file",
                api_key,
                "--tenant-api-keys-file",
                tenants,
                "--history-backend",
                "postgres",
                "--postgres-db-url-file",
                pg,
            ],
            &[],
        )
        .expect("resolves");
        assert_eq!(
            by_flag.integrations.api_key.as_deref(),
            Some("file-shared-key")
        );
        assert_eq!(
            by_flag.integrations.tenant_api_keys,
            vec!["red:file-red", "blue : file-blue"]
        );
        assert_eq!(
            by_flag.integrations.postgres_db_url.as_deref(),
            Some("postgresql://u:file-pg@db.example:5432/smg")
        );
        assert_eq!(
            by_flag.integrations.secret_sources,
            SecretSources {
                api_key: Some(SecretSource::File),
                tenant_api_keys: Some((SecretSource::File, 2)),
                postgres_db_url: Some(SecretSource::File),
                redis_url: None,
            }
        );
        let config = by_flag.to_router_config(vec![], vec![]).expect("builds");
        assert_eq!(
            config
                .tenant_api_keys
                .iter()
                .map(|entry| (entry.tenant_id.as_str(), entry.key.as_str()))
                .collect::<Vec<_>>(),
            vec![("red", "file-red"), ("blue", "file-blue")]
        );

        let by_env = resolved(
            &["--history-backend", "redis"],
            &[
                ("SMG_API_KEY_FILE", api_key),
                ("SMG_TENANT_API_KEYS_FILE", tenants),
                ("SMG_REDIS_URL_FILE", redis),
            ],
        )
        .expect("resolves");
        assert_eq!(
            by_env.integrations.api_key.as_deref(),
            Some("file-shared-key")
        );
        assert_eq!(by_env.integrations.tenant_api_keys.len(), 2);
        assert_eq!(
            by_env.integrations.redis_url.as_deref(),
            Some("redis://:file-redis@cache.example:6379/1")
        );
        assert_eq!(
            by_env.integrations.secret_sources,
            SecretSources {
                api_key: Some(SecretSource::File),
                tenant_api_keys: Some((SecretSource::File, 2)),
                postgres_db_url: None,
                redis_url: Some(SecretSource::File),
            }
        );
    }

    /// A value (from the flag or the environment) and a file for the same
    /// secret are refused, whichever way each is given; the refusal names
    /// the four sources and carries no value.
    #[test]
    fn a_value_and_a_file_for_one_secret_are_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("secret");
        std::fs::write(&file, "file-secret").unwrap();
        let file = file.to_str().unwrap();
        let backend = ["--history-backend", "redis"];
        /// The command line, the environment and the flag the refusal names.
        type Case<'a> = (Vec<&'a str>, Vec<(&'a str, &'a str)>, &'a str);
        let cases: Vec<Case> = vec![
            (
                vec!["--api-key", "flag-secret", "--api-key-file", file],
                vec![],
                "--api-key",
            ),
            (
                vec!["--api-key", "flag-secret"],
                vec![("SMG_API_KEY_FILE", file)],
                "--api-key",
            ),
            (
                vec![],
                vec![("SMG_API_KEY", "env-secret"), ("SMG_API_KEY_FILE", file)],
                "--api-key",
            ),
            (
                vec!["--api-key-file", file],
                vec![("SMG_API_KEY", "env-secret")],
                "--api-key",
            ),
            (
                vec![
                    "--tenant-api-key",
                    "red:flag-secret",
                    "--tenant-api-keys-file",
                    file,
                ],
                vec![],
                "--tenant-api-key",
            ),
            (
                vec!["--tenant-api-keys-file", file],
                vec![("SMG_TENANT_API_KEYS", "red:env-secret")],
                "--tenant-api-key",
            ),
            (
                vec![
                    "--history-backend",
                    "postgres",
                    "--postgres-db-url",
                    "postgresql://u:flag-secret@db.example/smg",
                    "--postgres-db-url-file",
                    file,
                ],
                vec![],
                "--postgres-db-url",
            ),
            (
                backend.to_vec(),
                vec![
                    ("SMG_REDIS_URL", "redis://:env-secret@cache.example/0"),
                    ("SMG_REDIS_URL_FILE", file),
                ],
                "--redis-url",
            ),
        ];
        for (args, vars, flag) in cases {
            let err = resolved(&args, &vars).expect_err(&format!("{args:?} {vars:?}"));
            let text = err.to_string();
            assert!(
                matches!(err, ConfigError::IncompatibleConfig { .. }),
                "{args:?}: {text}"
            );
            let file_flag = if flag == "--tenant-api-key" {
                "--tenant-api-keys-file".to_string()
            } else {
                format!("{flag}-file")
            };
            for named in [flag, file_flag.as_str()] {
                assert!(
                    text.contains(named),
                    "{args:?}: {named} missing from {text}"
                );
            }
            assert!(text.contains("SMG_"), "{args:?}: {text}");
            for secret in ["flag-secret", "env-secret", "file-secret"] {
                assert!(!text.contains(secret), "{args:?}: {text}");
            }
        }
    }

    /// A file that is missing, empty, or holds more than one value where
    /// one is expected, and a tenant key file without an entry, are
    /// refused under the file flag's name with the path; the content is
    /// never in the message.
    #[test]
    fn secret_files_that_cannot_be_used_are_refused_by_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("missing");
        let empty = dir.path().join("empty");
        let two_lines = dir.path().join("two-lines");
        let comments_only = dir.path().join("comments-only");
        std::fs::write(&empty, "\n").unwrap();
        std::fs::write(&two_lines, "first-secret\nsecond-secret\n").unwrap();
        std::fs::write(&comments_only, "# nothing here\n\n").unwrap();
        let path = |p: &Path| p.to_str().unwrap().to_string();
        let cases: Vec<(Vec<String>, &str, &str)> = vec![
            (
                vec!["--api-key-file".into(), path(&missing)],
                "api-key-file",
                "cannot be read",
            ),
            (
                vec!["--api-key-file".into(), path(&empty)],
                "api-key-file",
                "is empty",
            ),
            (
                vec!["--api-key-file".into(), path(&two_lines)],
                "api-key-file",
                "more than one line",
            ),
            (
                vec!["--tenant-api-keys-file".into(), path(&comments_only)],
                "tenant-api-keys-file",
                "holds no entry",
            ),
            (
                vec!["--tenant-api-keys-file".into(), path(&missing)],
                "tenant-api-keys-file",
                "cannot be read",
            ),
            (
                vec![
                    "--history-backend".into(),
                    "postgres".into(),
                    "--postgres-db-url-file".into(),
                    path(&two_lines),
                ],
                "postgres-db-url-file",
                "more than one line",
            ),
            (
                vec![
                    "--history-backend".into(),
                    "redis".into(),
                    "--redis-url-file".into(),
                    path(&missing),
                ],
                "redis-url-file",
                "cannot be read",
            ),
        ];
        for (args, field, reason) in cases {
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let err = resolved(&argv, &[]).expect_err(&args.join(" "));
            let text = err.to_string();
            match &err {
                ConfigError::InvalidValue {
                    field: named,
                    value,
                    reason: why,
                } => {
                    assert_eq!(named, field, "{text}");
                    assert!(value.starts_with(dir.path().to_str().unwrap()), "{text}");
                    assert!(why.contains(reason), "{text}");
                }
                other => panic!("{}: {other}", args.join(" ")),
            }
            assert!(!text.contains("first-secret"), "{text}");
        }
    }

    /// A tenant source that is set but holds no entry is refused: it was
    /// meant to close the serving routes. A malformed entry from the
    /// environment is refused where a flag's is, without an echo.
    #[test]
    fn tenant_entries_from_the_environment_are_checked_like_the_flags() {
        let err = resolved(&[], &[("SMG_TENANT_API_KEYS", " , \n")]).expect_err("no entry");
        assert!(
            matches!(&err, ConfigError::InvalidValue { field, .. } if field == "SMG_TENANT_API_KEYS"),
            "{err}"
        );

        let cli = resolved(&[], &[("SMG_TENANT_API_KEYS", "no-separator-secret")])
            .expect("the shape is checked by the parser");
        let err = cli
            .to_router_config(vec![], vec![])
            .expect_err("a tenant entry without ':' is refused");
        let text = err.to_string();
        assert!(text.contains("tenant-api-key"), "{text}");
        assert!(!text.contains("no-separator-secret"), "{text}");
    }

    /// A store URL source is read for the selected history backend alone:
    /// a PostgreSQL URL file named in the environment of a start on the
    /// memory backend is not opened, and the record does not name it.
    #[test]
    fn store_urls_are_resolved_for_the_selected_backend_alone() {
        let vars = [
            ("SMG_POSTGRES_DB_URL_FILE", "/nonexistent/pg-url"),
            ("SMG_REDIS_URL_FILE", "/nonexistent/redis-url"),
        ];

        let cli = resolved(&[], &vars).expect("the memory backend reads no store URL");
        assert_eq!(cli.integrations.postgres_db_url, None);
        assert_eq!(cli.integrations.redis_url, None);
        assert_eq!(cli.integrations.secret_sources, SecretSources::default());

        let err = resolved(&["--history-backend", "postgres"], &vars).expect_err("opened");
        assert!(
            matches!(&err, ConfigError::InvalidValue { field, .. } if field == "postgres-db-url-file"),
            "{err}"
        );
        let err = resolved(&["--history-backend", "redis"], &vars).expect_err("opened");
        assert!(
            matches!(&err, ConfigError::InvalidValue { field, .. } if field == "redis-url-file"),
            "{err}"
        );
    }

    /// The sources travel into the server configuration, for the start-up
    /// record.
    #[test]
    fn the_secret_sources_reach_the_server_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let api_key = dir.path().join("api-key");
        std::fs::write(&api_key, "file-shared-key\n").unwrap();

        let cli = resolved(
            &["--api-key-file", api_key.to_str().unwrap()],
            &[("SMG_TENANT_API_KEYS", "red:env-red")],
        )
        .expect("resolves");
        let server_config = cli.configure(vec![], vec![]).expect("configures");

        assert_eq!(
            server_config.secret_sources,
            SecretSources {
                api_key: Some(SecretSource::File),
                tenant_api_keys: Some((SecretSource::Environment, 1)),
                postgres_db_url: None,
                redis_url: None,
            }
        );
        assert_eq!(
            server_config.router_config.api_key.as_deref(),
            Some("file-shared-key")
        );
    }

    /// `--help` names the environment variables and the file flags of the
    /// secrets, and the variables clap reads itself are declared with their
    /// values hidden, so a set one never shows. Asserted on the command's
    /// arguments rather than by setting the variables: the other tests of
    /// this module parse through clap in parallel and would read them.
    #[test]
    fn the_help_names_the_sources_and_hides_the_values() {
        use clap::CommandFactory;

        let mut command = Cli::command();
        for id in ["oracle_password", "control_plane_api_keys"] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id().as_str() == id)
                .unwrap_or_else(|| panic!("{id} is an argument"));
            assert!(
                arg.is_hide_env_values_set(),
                "{id} would show its environment value in the help"
            );
        }
        let help = command.render_long_help().to_string();

        for named in [
            "SMG_API_KEY",
            "--api-key-file",
            "SMG_API_KEY_FILE",
            "SMG_TENANT_API_KEYS",
            "--tenant-api-keys-file",
            "SMG_POSTGRES_DB_URL",
            "--postgres-db-url-file",
            "SMG_REDIS_URL",
            "--redis-url-file",
            "[env: ATP_PASSWORD]",
            "[env: CONTROL_PLANE_API_KEYS]",
        ] {
            assert!(help.contains(named), "{named} missing from the help");
        }
        assert!(!help.contains("hunter2"), "a value in the help:\n{help}");
    }
}
