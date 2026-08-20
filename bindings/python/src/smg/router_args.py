from __future__ import annotations

import argparse
import dataclasses
import logging
import os

from smg.smg_rs import get_available_reasoning_parsers, get_available_tool_call_parsers

logger = logging.getLogger(__name__)


COMMON_POLICY_CHOICES = [
    "random",
    "round_robin",
    "passthrough",
    "cache_aware",
    "power_of_two",
    "least_load",
    "manual",
    "consistent_hashing",
    "prefix_hash",
]

PREFILL_POLICY_CHOICES = [*COMMON_POLICY_CHOICES, "bucket"]
ENCODE_POLICY_CHOICES = ["random", "round_robin", "consistent_hashing"]


def _parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated integer list (mirrors the CLI value_delimiter)."""
    return [int(item) for item in value.split(",") if item]


@dataclasses.dataclass
class RouterArgs:
    # Worker configuration
    worker_urls: list[str] = dataclasses.field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 30000
    # Dedicated port for liveness/readiness/health probes (k8s, load balancers, monitors), served from an
    # isolated runtime so probes are not starved by the request runtime.
    # None = dedicated probe listener off (routes stay on the main port).
    health_check_port: int | None = None

    # PD/EPD-specific configuration
    pd_disaggregation: bool = False  # Enable PD disaggregated mode
    epd_disaggregation: bool = False  # Enable Encode-Prefill-Decode disaggregated mode
    encode_urls: list[tuple] = dataclasses.field(
        default_factory=list
    )  # List of (url, bootstrap_port)
    prefill_urls: list[tuple] = dataclasses.field(
        default_factory=list
    )  # List of (url, bootstrap_port)
    decode_urls: list[str] = dataclasses.field(default_factory=list)

    # Routing policy
    policy: str = "cache_aware"
    encode_policy: str | None = None  # Specific policy for encode nodes in EPD mode
    prefill_policy: str | None = None  # Specific policy for prefill nodes in PD mode
    decode_policy: str | None = None  # Specific policy for decode nodes in PD mode
    worker_startup_timeout_secs: int = 1800
    worker_startup_check_interval: int = 30
    load_monitor_interval: int = 10
    cache_threshold: float = 0.3
    balance_abs_threshold: int = 64
    balance_rel_threshold: float = 1.5
    balance_token_usage_threshold: float = 1.0
    overload_token_usage_threshold: float = 1.0
    eviction_interval_secs: int = 60
    max_tree_size: int = 2**26
    block_size: int = 16
    least_load_kv_pressure_weight: float = 0.15
    least_load_default_throughput: float = 2000.0
    least_load_mean_prefill_tokens: int = 1024
    max_idle_secs: int = 4 * 3600
    # Routing-key assignment; defaults: random (manual policy), delegate (override)
    assignment_mode: str | None = None
    max_payload_size: int = 512 * 1024 * 1024  # 512MB default for large batches
    bucket_adjust_interval_secs: int = 5
    dp_aware: bool = False
    multimodal_tensor_transport: str | None = None
    multimodal_shm_min_bytes: int | None = None
    routing_key_override: bool = False
    dp_minimum_tokens_scheduler: bool = False
    enable_igw: bool = False  # Enable IGW (Inter-Gateway) mode for multi-model support
    api_key: str | None = None
    log_dir: str | None = None
    log_level: str | None = None
    log_json: bool = False
    # Service discovery configuration
    service_discovery: bool = False
    selector: dict[str, str] = dataclasses.field(default_factory=dict)
    service_discovery_port: int = 80
    service_discovery_namespace: str | None = None
    # PD/EPD service discovery configuration
    encode_selector: dict[str, str] = dataclasses.field(default_factory=dict)
    prefill_selector: dict[str, str] = dataclasses.field(default_factory=dict)
    decode_selector: dict[str, str] = dataclasses.field(default_factory=dict)
    router_selector: dict[str, str] = dataclasses.field(default_factory=dict)
    bootstrap_port_annotation: str = "sglang.ai/bootstrap-port"
    worker_ports_annotation: str = "smg.ai/worker-ports"
    model_id_from: str | None = None
    # Prometheus configuration
    prometheus_port: int | None = None
    prometheus_host: str | None = None
    prometheus_duration_buckets: list[float] | None = None
    # Request ID headers configuration
    request_id_headers: list[str] | None = None
    # HTTP header to storage hook context mapping
    storage_context_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    # Request timeout in seconds
    request_timeout_secs: int = 1800
    # Grace period in seconds to wait for in-flight requests during shutdown
    shutdown_grace_period_secs: int = 180
    # Standing-concurrency cap (-1 to disable); permits span the full response
    max_concurrent_requests: int = -1
    # Queue size for pending requests when max concurrent limit reached
    queue_size: int = 100
    # Maximum time (in seconds) a request can wait in queue before timing out
    queue_timeout_secs: int = 60
    # Token bucket refill rate (tokens per second). Unset or 0 = no refill
    rate_limit_tokens_per_second: int | None = None
    # CORS allowed origins
    cors_allowed_origins: list[str] = dataclasses.field(default_factory=list)
    # Retry configuration
    retry_max_retries: int = 5
    retry_initial_backoff_ms: int = 50
    retry_max_backoff_ms: int = 30_000
    retry_backoff_multiplier: float = 1.5
    retry_jitter_factor: float = 0.2
    disable_retries: bool = False
    # Health check configuration
    health_failure_threshold: int = 3
    health_success_threshold: int = 2
    health_check_timeout_secs: int = 5
    health_check_interval_secs: int = 60
    health_check_endpoint: str = "/health"
    disable_health_check: bool = False
    remove_unhealthy_workers: bool = False
    # Circuit breaker configuration
    cb_failure_threshold: int = 10
    cb_success_threshold: int = 3
    cb_timeout_duration_secs: int = 60
    cb_window_duration_secs: int = 120
    disable_circuit_breaker: bool = False
    model_path: str | None = None
    tokenizer_path: str | None = None
    chat_template: str | None = None
    # Disable automatic tokenizer loading at startup and worker registration
    disable_tokenizer_autoload: bool = False
    # Tokenizer cache configuration
    tokenizer_cache_enable_l0: bool = False
    tokenizer_cache_l0_max_entries: int = 10000
    tokenizer_cache_enable_l1: bool = False
    tokenizer_cache_l1_max_memory: int = 50 * 1024 * 1024  # 50MB
    # Parser configuration
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    # MCP server configuration
    mcp_config_path: str | None = None
    # Backend selection
    backend: str = "sglang"
    # WASM support
    enable_wasm: bool = False
    # Storage hooks (WASM)
    storage_hook_wasm_path: str | None = None
    # History backend configuration
    history_backend: str = "memory"
    oracle_wallet_path: str | None = None
    oracle_tns_alias: str | None = None
    oracle_connect_descriptor: str | None = None
    oracle_username: str | None = None
    oracle_password: str | None = None
    oracle_external_auth: bool = False
    oracle_pool_min: int = 1
    oracle_pool_max: int = 16
    oracle_pool_timeout_secs: int = 30
    postgres_db_url: str | None = None
    postgres_pool_max: int = 16
    redis_url: str | None = None
    redis_pool_max: int = 16
    redis_retention_days: int = 30
    schema_config: str | None = None
    # mTLS configuration for worker communication
    client_cert_path: str | None = None
    client_key_path: str | None = None
    ca_cert_paths: list[str] = dataclasses.field(default_factory=list)
    # Server TLS configuration
    server_cert_path: str | None = None
    server_key_path: str | None = None
    # Trace
    enable_trace: bool = False
    otlp_traces_endpoint: str = "localhost:4317"
    # Control plane authentication
    # API keys for control plane auth (list of tuples: id, name, key, role)
    control_plane_api_keys: list[tuple] = dataclasses.field(default_factory=list)
    control_plane_audit_enabled: bool = False
    # JWT/OIDC configuration for control plane auth
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_jwks_uri: str | None = None
    jwt_role_mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    # Mesh server configuration
    enable_mesh: bool = False
    mesh_server_name: str | None = None
    mesh_host: str = "0.0.0.0"
    mesh_advertise_host: str | None = None
    mesh_port: int = 39527
    mesh_peer_urls: list[str] = dataclasses.field(default_factory=list)
    # Append new fields here to preserve positional callers.
    model_aliases: dict[str, str] = dataclasses.field(default_factory=dict)
    worker_startup_delay: int = 0
    # DP engines per startup ZMQ worker (grouped worker; None/1 = ungrouped)
    zmq_engine_count: int | None = None
    prefix_token_count: int = 256
    prefix_hash_load_factor: float = 1.25
    prefix_hash_balance_abs_threshold: int = 10
    upstream_http2: bool = False
    overlap_decay: float = 0.0
    selection_temperature: float = 0.0
    upstream_pool_idle_timeout_secs: int = 3
    least_load_max_waiting_requests: int = 0
    stream_request_bodies_over: int = 0
    stream_body_stall_timeout_secs: int = 300
    # Ordered header names checked for the routing key; first valid wins
    routing_key_headers: list[str] = dataclasses.field(
        default_factory=lambda: ["x-smg-routing-key"]
    )
    # Token positions at which serving engines retain reusable prefix state
    cache_boundaries: list[int] = dataclasses.field(default_factory=list)
    # cache_aware index under-layer: "tree" or "hash"
    cache_index: str = "tree"
    # Seconds a cache-affinity placement stays routable
    cache_ttl_secs: int = 180
    # Control-plane job queue sizing (worker registration/removal jobs)
    job_queue_capacity: int = 1000
    job_queue_concurrency: int = 200
    # Absolute per-worker overload thresholds; both None disables the feature
    worker_overload_waiting_requests: int | None = None
    worker_overload_token_usage: float | None = None
    # Enable overload protection with the gateway default token ceiling (0.9)
    worker_overload_protection: bool = False
    # Restore the conditional load-monitor poll gate (default: poll always)
    disable_load_monitoring: bool = False

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
        use_router_prefix: bool = False,
        exclude_host_port: bool = False,
    ):
        """
        Add router-specific arguments to an argument parser.

        Args:
            parser: The argument parser to add arguments to
            use_router_prefix: If True, prefix all arguments with 'router-' to avoid conflicts
            exclude_host_port: If True, don't add host and port arguments (used when inheriting from server)
        """
        prefix = "router-" if use_router_prefix else ""

        # Repeatable list flags must accumulate across occurrences
        # (action="extend"/"append"), matching the Rust CLI.

        # Create argument groups for organized --help output
        worker_group = parser.add_argument_group(
            "Worker Configuration", "Settings for worker connections and URLs"
        )
        routing_group = parser.add_argument_group(
            "Routing Policy", "Load balancing and routing configuration"
        )
        pd_group = parser.add_argument_group(
            "PD/EPD Disaggregation", "Encode-Prefill-Decode and Prefill-Decode settings"
        )
        k8s_group = parser.add_argument_group(
            "Service Discovery (Kubernetes)", "Kubernetes-based worker discovery"
        )
        logging_group = parser.add_argument_group("Logging", "Log output configuration")
        prometheus_group = parser.add_argument_group(
            "Prometheus Metrics", "Metrics export configuration"
        )
        request_group = parser.add_argument_group(
            "Request Handling", "Request timeout and ID configuration"
        )
        rate_limit_group = parser.add_argument_group(
            "Rate Limiting", "Concurrent request and queue limits"
        )
        retry_group = parser.add_argument_group(
            "Retry Configuration", "Automatic retry behavior for failed requests"
        )
        cb_group = parser.add_argument_group(
            "Circuit Breaker", "Circuit breaker pattern configuration"
        )
        health_group = parser.add_argument_group(
            "Health Checks", "Worker health monitoring settings"
        )
        tokenizer_group = parser.add_argument_group(
            "Tokenizer", "Tokenizer and chat template configuration"
        )
        parser_group = parser.add_argument_group(
            "Parsers", "Reasoning and tool-call parser settings"
        )
        backend_group = parser.add_argument_group(
            "Backend", "Backend runtime and history storage selection"
        )
        oracle_group = parser.add_argument_group(
            "Oracle Database", "Oracle database backend configuration"
        )
        postgres_group = parser.add_argument_group(
            "PostgreSQL Database", "PostgreSQL database backend configuration"
        )
        redis_group = parser.add_argument_group(
            "Redis Database", "Redis database backend configuration"
        )
        tls_group = parser.add_argument_group(
            "TLS/mTLS Security", "TLS certificates for server and worker communication"
        )
        trace_group = parser.add_argument_group(
            "Tracing (OpenTelemetry)", "Distributed tracing configuration"
        )
        auth_group = parser.add_argument_group(
            "Control Plane Authentication", "API key and JWT/OIDC authentication"
        )

        if use_router_prefix:
            parser.add_argument(
                "--router-disable-arg-fallback",
                action="store_true",
                default=False,
                help=(
                    "When set, only use explicitly provided --router-* arguments and do not"
                    " fall back to backend arguments with the same name."
                ),
            )

        # Worker configuration
        if not exclude_host_port:
            worker_group.add_argument(
                "--host",
                type=str,
                default=RouterArgs.host,
                help=(
                    "Host address to bind the router server. Supports IPv4, IPv6 (e.g., ::, ::1),"
                    " or 0.0.0.0 for all interfaces"
                ),
            )
            worker_group.add_argument(
                "--port",
                type=int,
                default=RouterArgs.port,
                help="Port number to bind the router server",
            )

        worker_group.add_argument(
            "--worker-urls",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help=(
                "List of worker URLs. Supports IPv4 and IPv6 addresses"
                " (use brackets for IPv6, e.g., http://[::1]:8000 http://192.168.1.1:8000)"
            ),
        )
        worker_group.add_argument(
            f"--{prefix}upstream-http2",
            action="store_true",
            help=(
                "Speak HTTP/2 to workers via prior knowledge (h2c on cleartext),"
                " multiplexing every request to a worker over one connection."
                " Requires every HTTP worker to serve HTTP/2 without an upgrade"
                " handshake."
            ),
        )
        worker_group.add_argument(
            f"--{prefix}upstream-pool-idle-timeout-secs",
            type=int,
            default=RouterArgs.upstream_pool_idle_timeout_secs,
            help=(
                "Idle timeout in seconds for pooled upstream connections. Must"
                " stay below the backend HTTP server's keep-alive timeout (vLLM"
                " and SGLang default to 5). 0 keeps idle connections forever."
                " Defaults to 3."
            ),
        )
        worker_group.add_argument(
            f"--{prefix}health-check-port",
            type=int,
            default=RouterArgs.health_check_port,
            help=(
                "Dedicated port for liveness/readiness/health probes (Kubernetes, load"
                " balancers, uptime monitors)."
                " When set, those routes are also served on this port by an"
                " isolated runtime so probes are not starved by the request"
                " runtime. Unset = dedicated probe listener off (routes remain"
                " available on the main port)."
            ),
        )

        # Routing policy configuration
        routing_group.add_argument(
            f"--{prefix}policy",
            type=str,
            default=RouterArgs.policy,
            choices=COMMON_POLICY_CHOICES,
            help=(
                "Load balancing policy to use. In PD mode, this is used for both prefill and decode"
                " unless overridden"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}encode-policy",
            type=str,
            default=None,
            choices=ENCODE_POLICY_CHOICES,
            help=(
                "Specific policy for encode nodes in EPD mode."
                " If not specified, uses consistent_hashing"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}prefill-policy",
            type=str,
            default=None,
            choices=PREFILL_POLICY_CHOICES,
            help=(
                "Specific policy for prefill nodes in PD mode."
                " If not specified, uses the main policy"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}decode-policy",
            type=str,
            default=None,
            choices=COMMON_POLICY_CHOICES,
            help=(
                "Specific policy for decode nodes in PD mode."
                " If not specified, uses the main policy"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}cache-threshold",
            f"--{prefix}cache-match-threshold",
            type=float,
            default=RouterArgs.cache_threshold,
            help=(
                "Minimum matched-prefix share (0.0-1.0) before cache-aware routing"
                " pins a request to a worker already holding that prefix"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}prefix-token-count",
            type=int,
            default=RouterArgs.prefix_token_count,
            help=(
                "Number of prefix tokens hashed by the prefix_hash policy "
                "(untokenized requests hash four times as many characters)"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}prefix-hash-load-factor",
            type=float,
            default=RouterArgs.prefix_hash_load_factor,
            help="Load factor above which prefix_hash walks the ring (multiple of average load)",
        )
        routing_group.add_argument(
            f"--{prefix}prefix-hash-balance-abs-threshold",
            type=int,
            default=RouterArgs.prefix_hash_balance_abs_threshold,
            help=(
                "Absolute load difference over average a worker must also "
                "exceed before prefix_hash treats it as overloaded"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}least-load-kv-pressure-weight",
            type=float,
            default=RouterArgs.least_load_kv_pressure_weight,
            help="KV-pressure weight (seconds) for the least_load policy",
        )
        routing_group.add_argument(
            f"--{prefix}least-load-default-throughput",
            type=float,
            default=RouterArgs.least_load_default_throughput,
            help=(
                "Fallback generation throughput (tokens/s) for least_load when a"
                " backend reports no live throughput"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}least-load-mean-prefill-tokens",
            type=int,
            default=RouterArgs.least_load_mean_prefill_tokens,
            help=(
                "Mean prefill tokens for least_load's in-flight estimate when a"
                " request's token count is unknown at routing"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}least-load-max-waiting-requests",
            type=int,
            default=RouterArgs.least_load_max_waiting_requests,
            help=(
                "Per-worker waiting-queue cap for least_load: skip workers whose"
                " reported waiting requests (plus dispatches since their last"
                " poll) have reached this count; 0 disables"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}balance-abs-threshold",
            f"--{prefix}spill-abs-threshold",
            type=int,
            default=RouterArgs.balance_abs_threshold,
            help=(
                "Spill gate, absolute part. Balancing is triggered if"
                " `(max_load - min_load) > abs_threshold` and the relative threshold is also met."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}balance-rel-threshold",
            f"--{prefix}spill-rel-threshold",
            type=float,
            default=RouterArgs.balance_rel_threshold,
            help=(
                "Spill gate, relative part. Balancing is triggered if"
                " `max_load > min_load * rel_threshold` and the absolute threshold is also met."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}balance-token-usage-threshold",
            type=float,
            default=RouterArgs.balance_token_usage_threshold,
            help=(
                "Cache-aware KV-usage SPREAD threshold (0.0-1.0): the hottest minus"
                " coldest backend KV utilization above which cache affinity is"
                " abandoned for shortest-queue. Catches long-context KV imbalance that"
                " in-flight request counts miss, and is invariant to gateway replica"
                " count. Backend must report token_usage. Defaults to 1.0 (disabled)."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}overload-token-usage-threshold",
            type=float,
            default=RouterArgs.overload_token_usage_threshold,
            help=(
                "Cache-aware KV-utilization CEILING (0.0-1.0): when the hottest backend"
                " exceeds it, shed load off that engine regardless of spread. A safety"
                " valve for critically-saturated engines, best set high (e.g. 0.9)."
                " Backend must report token_usage. Defaults to 1.0 (disabled)."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}worker-overload-waiting-requests",
            type=int,
            default=RouterArgs.worker_overload_waiting_requests,
            help=(
                "Queued-request count AT OR ABOVE which a worker is considered"
                " overloaded and excluded from routing until the signal recovers;"
                " when every worker is overloaded, requests are shed immediately"
                " rather than queued. Unset disables overload protection. This"
                " signal is the queued (waiting) request count, summed across DP"
                " ranks. Must be >= 1: the comparison is inclusive, so 0 would veto"
                " every worker unconditionally."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}worker-overload-token-usage",
            type=float,
            default=RouterArgs.worker_overload_token_usage,
            help=(
                "KV-cache token usage AT OR ABOVE which a worker is considered"
                " overloaded and excluded from routing until the signal recovers;"
                " when every worker is overloaded, requests are shed immediately"
                " rather than queued. Unset disables overload protection. This"
                " signal is mean KV-cache token usage across DP ranks, the same one"
                " --balance-token-usage-threshold reads, applied as an absolute"
                " per-worker CEILING rather than a fleet-relative spread. Backend"
                " must report token_usage. Must be in (0.0, 1.0]: the comparison is"
                " inclusive, so 0.0 would veto every worker unconditionally."
                " Distinct from --overload-token-usage-threshold, which only"
                " de-ranks the hottest backend within cache-aware affinity; this"
                " flag removes the worker from routing entirely."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}worker-overload-protection",
            action="store_true",
            help=(
                "Enable worker overload protection with the gateway default"
                " thresholds. This flag alone applies"
                " --worker-overload-token-usage 0.9 and leaves"
                " --worker-overload-waiting-requests unset: KV token usage means"
                " the same thing on every engine, while a sensible"
                " waiting-requests ceiling is workload-dependent, so it has no"
                " universal default. Explicit thresholds override the default,"
                " and either threshold set on its own enables protection without"
                " this flag."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}overlap-decay",
            type=float,
            default=RouterArgs.overlap_decay,
            help=(
                "Cache-aware anti-hotspot decay: divide each candidate's overlap"
                " score by 1 + overlap_decay * x, where x is the worker's"
                " waiting-prefill backlog (blocks above the candidate minimum) per"
                " request block. Requires backend load reporting. Defaults to 0.0"
                " (disabled)."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}selection-temperature",
            type=float,
            default=RouterArgs.selection_temperature,
            help=(
                "Cache-aware softmax temperature over min-max normalized scores for"
                " event-driven selection. 0.0 is exact argmax; larger values spread"
                " picks across candidates. Defaults to 0.0."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}bucket-adjust-interval-secs",
            type=int,
            default=RouterArgs.bucket_adjust_interval_secs,
            help="Interval in seconds between bucket boundary adjustment operations",
        )
        routing_group.add_argument(
            f"--{prefix}eviction-interval-secs",
            type=int,
            default=RouterArgs.eviction_interval_secs,
            help="Interval in seconds between cache eviction operations",
        )
        routing_group.add_argument(
            f"--{prefix}max-tree-size",
            type=int,
            default=RouterArgs.max_tree_size,
            help="Maximum total size of each model's approximation tree for "
            "cache-aware routing (chars for HTTP, tokens for gRPC), shared "
            "across all workers; eviction keeps every tree at or under this "
            "bound",
        )
        routing_group.add_argument(
            f"--{prefix}block-size",
            type=int,
            default=RouterArgs.block_size,
            help="KV cache block size for event-driven cache-aware routing (default: 16)",
        )
        routing_group.add_argument(
            f"--{prefix}cache-boundaries",
            type=_parse_int_csv,
            default=[],
            help=(
                "Comma-separated token positions at which serving engines retain"
                " reusable prefix state; cache-affinity policies hash request"
                " heads at the deepest applicable boundary."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}cache-index",
            type=str,
            choices=["tree", "hash"],
            default=RouterArgs.cache_index,
            help=(
                "Index under-layer for cache_aware: 'tree' (radix prefix trees)"
                " or 'hash' (TTL'd exact-match placement map keyed on request"
                " heads at --cache-boundaries; token-bearing requests only —"
                " untokenized requests stay load-balanced). Defaults to 'tree'."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}cache-ttl-secs",
            type=int,
            default=RouterArgs.cache_ttl_secs,
            help=(
                "Seconds a cache-affinity placement stays routable; should"
                " approximate serving-engine cache retention. Defaults to 180."
            ),
        )
        routing_group.add_argument(
            f"--{prefix}max-idle-secs",
            f"--{prefix}sticky-key-idle-secs",
            type=int,
            default=RouterArgs.max_idle_secs,
            help=(
                "How long an unused sticky routing key stays pinned: keys idle"
                " beyond this many seconds are evicted from the sticky map"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}assignment-mode",
            type=str,
            default=RouterArgs.assignment_mode,
            choices=["random", "min_load", "min_group", "delegate"],
            help=(
                "Mode for assigning new routing keys: random, min_load (fewest"
                " requests), min_group (fewest routing keys), delegate (route via"
                " the underlying policy, then pin). Defaults to random for the"
                " manual policy and delegate for the routing-key override"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}max-payload-size",
            type=int,
            default=RouterArgs.max_payload_size,
            help="Maximum payload size in bytes",
        )
        routing_group.add_argument(
            f"--{prefix}stream-request-bodies-over",
            type=int,
            default=RouterArgs.stream_request_bodies_over,
            help=(
                "Forward request bodies larger than this many bytes to the"
                " worker as a raw stream instead of buffering, when the"
                " route's policy needs no request text and the worker applies"
                " no body mutation. Streamed bodies cannot be replayed, so"
                " those requests bypass router-level retries; bodies without"
                " a Content-Length header always buffer. 0 disables"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}stream-body-stall-timeout-secs",
            type=int,
            default=RouterArgs.stream_body_stall_timeout_secs,
            help=(
                "Abort a streamed request body once the upstream sender has"
                " waited on the client for this many seconds (408). The clock"
                " pauses while the worker applies backpressure, so a slow"
                " worker read never trips it. Applies only to bodies streamed"
                " via --stream-request-bodies-over. 0 disables"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}dp-aware",
            action="store_true",
            help="Enable data parallelism aware schedule",
        )
        routing_group.add_argument(
            f"--{prefix}routing-key-override",
            f"--{prefix}sticky-sessions",
            action="store_true",
            help=(
                "Sticky sessions: route every request of a conversation to the"
                " same worker, on any policy (keys derived from the request-id"
                " lineage, falling back to the routing-key headers)"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}routing-key-headers",
            type=str,
            nargs="*",
            action="extend",
            help=(
                "Ordered header names checked for the routing key; the first"
                " header present with a valid value wins"
            ),
        )
        routing_group.add_argument(
            f"--{prefix}dp-minimum-tokens-scheduler",
            action="store_true",
            help="Enable minimum tokens scheduler for data parallel group",
        )
        routing_group.add_argument(
            f"--{prefix}enable-igw",
            action="store_true",
            help="Enable IGW (Inference-Gateway) mode for multi-model support",
        )

        # PD/EPD-specific arguments
        pd_group.add_argument(
            f"--{prefix}pd-disaggregation",
            action="store_true",
            help="Enable PD (Prefill-Decode) disaggregated mode",
        )
        pd_group.add_argument(
            f"--{prefix}epd-disaggregation",
            action="store_true",
            help="Enable EPD (Encode-Prefill-Decode) disaggregated mode",
        )
        pd_group.add_argument(
            f"--{prefix}encode",
            nargs="+",
            action="append",
            help="Encode server URL and optional bootstrap port. Can be specified multiple times. "
            "Format: --encode URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, 'none', or omitted (defaults to none).",
        )
        pd_group.add_argument(
            f"--{prefix}prefill",
            nargs="+",
            action="append",
            help="Prefill server URL and optional bootstrap port. Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, 'none', or omitted (defaults to none).",
        )
        pd_group.add_argument(
            f"--{prefix}decode",
            nargs=1,
            action="append",
            metavar=("URL",),
            help="Decode server URL. Can be specified multiple times.",
        )
        pd_group.add_argument(
            f"--{prefix}worker-startup-timeout-secs",
            type=int,
            default=RouterArgs.worker_startup_timeout_secs,
            help=(
                "Timeout in seconds for worker startup and registration (default: 1800 / 30 minutes)."
                " Large models can take significant time to load into GPU memory."
            ),
        )
        pd_group.add_argument(
            f"--{prefix}worker-startup-delay",
            type=int,
            default=RouterArgs.worker_startup_delay,
            help="Grace period in seconds before the first worker startup check (default: 0)",
        )
        pd_group.add_argument(
            f"--{prefix}worker-startup-check-interval",
            type=int,
            default=RouterArgs.worker_startup_check_interval,
            help="Interval in seconds between checks for worker startup",
        )
        parser.add_argument(
            f"--{prefix}job-queue-capacity",
            type=int,
            default=RouterArgs.job_queue_capacity,
            help=(
                "Max pending control-plane jobs (worker add/remove, tokenizer, MCP, WASM)."
                " Size to fleet scale so a service-discovery reconcile pass can enqueue"
                " every worker without blocking (default: 1000)"
            ),
        )
        parser.add_argument(
            f"--{prefix}job-queue-concurrency",
            type=int,
            default=RouterArgs.job_queue_concurrency,
            help="Max control-plane jobs dispatched concurrently (default: 200)",
        )

        # Load monitoring
        parser.add_argument(
            f"--{prefix}load-monitor-interval",
            type=int,
            default=RouterArgs.load_monitor_interval,
            help="Interval in seconds between load monitor checks for PowerOfTwo routing (default: 10)",
        )
        parser.add_argument(
            f"--{prefix}disable-load-monitoring",
            action="store_true",
            help=(
                "Only poll worker loads when a load-aware routing policy,"
                " --engine-metrics, or worker overload protection needs the"
                " data. By default every worker group is polled from"
                " registration onward; this restores the old conditional gate"
                " (a load-aware policy is always fed regardless)."
            ),
        )

        # Multimodal tensor transport
        parser.add_argument(
            f"--{prefix}multimodal-tensor-transport",
            type=str,
            choices=["inline", "shm", "auto", "rdma"],
            default=RouterArgs.multimodal_tensor_transport,
            help="Multimodal tensor transport: inline (default), shm, auto, or rdma (NIXL lane; needs mm-rdma build)",
        )
        parser.add_argument(
            f"--{prefix}multimodal-shm-min-bytes",
            type=int,
            default=RouterArgs.multimodal_shm_min_bytes,
            help="Minimum multimodal tensor size (bytes) before the SHM transport is used",
        )

        # Logging configuration
        logging_group.add_argument(
            f"--{prefix}log-dir",
            type=str,
            default=None,
            help=(
                "Directory to store log files. If not specified, logs are only output to console."
            ),
        )
        logging_group.add_argument(
            f"--{prefix}log-level",
            type=str,
            default="info",
            choices=["debug", "info", "warn", "error"],
            help="Set the logging level. If not specified, defaults to INFO.",
        )
        logging_group.add_argument(
            f"--{prefix}log-json",
            action="store_true",
            default=RouterArgs.log_json,
            help="Output logs in JSON format",
        )

        # Service discovery configuration
        k8s_group.add_argument(
            f"--{prefix}service-discovery",
            action="store_true",
            help="Enable Kubernetes service discovery",
        )
        k8s_group.add_argument(
            f"--{prefix}selector",
            type=str,
            nargs="+",
            action="extend",
            default=None,
            help="Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)",
        )
        k8s_group.add_argument(
            f"--{prefix}service-discovery-port",
            type=int,
            default=RouterArgs.service_discovery_port,
            help="Port to use for discovered worker pods",
        )
        k8s_group.add_argument(
            f"--{prefix}service-discovery-namespace",
            type=str,
            help=(
                "Kubernetes namespace to watch for pods. If not provided, watches all namespaces"
                " (requires cluster-wide permissions)"
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}encode-selector",
            type=str,
            nargs="+",
            action="extend",
            default=None,
            help=(
                "Label selector for encode server pods in EPD mode"
                " (format: key1=value1 key2=value2)"
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}prefill-selector",
            type=str,
            nargs="+",
            action="extend",
            default=None,
            help=(
                "Label selector for prefill server pods in PD mode"
                " (format: key1=value1 key2=value2)"
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}decode-selector",
            type=str,
            nargs="+",
            action="extend",
            default=None,
            help=(
                "Label selector for decode server pods in PD mode (format: key1=value1 key2=value2)"
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}router-selector",
            type=str,
            nargs="+",
            action="extend",
            default=None,
            help=(
                "Label selector for router pod discovery in HA mesh mode (format: key1=value1 key2=value2)"
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}model-id-from",
            type=str,
            default=None,
            help=(
                "Override each worker's model ID from pod metadata."
                " Accepted values: 'namespace', 'label:<key>', 'annotation:<key>'."
                " The backend-discovered model name becomes an alias."
            ),
        )
        k8s_group.add_argument(
            f"--{prefix}model-alias",
            type=str,
            action="append",
            default=[],
            help=(
                "Accept an extra client-facing model name for a served model."
                " Format: <alias>=<canonical>. Repeat for multiple aliases."
                " Matching is case-sensitive."
            ),
        )
        # Prometheus configuration
        prometheus_group.add_argument(
            f"--{prefix}prometheus-port",
            type=int,
            default=29000,
            help=(
                "Port to expose Prometheus metrics (default: 29000)."
                " 0 binds an OS-assigned ephemeral port, logged at startup."
            ),
        )
        prometheus_group.add_argument(
            f"--{prefix}prometheus-host",
            type=str,
            default="0.0.0.0",
            help=(
                "Host address to bind the Prometheus metrics server. Supports IPv4, IPv6"
                " (e.g., ::, ::1), or 0.0.0.0 for all interfaces"
            ),
        )
        prometheus_group.add_argument(
            f"--{prefix}prometheus-duration-buckets",
            type=float,
            nargs="+",
            action="extend",
            help="Buckets for Prometheus duration metrics",
        )

        # Request handling configuration
        request_group.add_argument(
            f"--{prefix}request-id-headers",
            type=str,
            nargs="*",
            action="extend",
            help=(
                "Custom HTTP headers to check for request IDs (e.g., x-request-id x-trace-id)."
                " If not specified, uses common defaults."
            ),
        )
        request_group.add_argument(
            f"--{prefix}storage-context-headers",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help=(
                "Map HTTP headers into storage hook request context using HEADER=CONTEXT_KEY "
                "entries, for example x-tenant-id=tenant_id"
            ),
        )
        request_group.add_argument(
            f"--{prefix}request-timeout-secs",
            type=int,
            default=RouterArgs.request_timeout_secs,
            help="Request timeout in seconds",
        )
        request_group.add_argument(
            f"--{prefix}shutdown-grace-period-secs",
            type=int,
            default=RouterArgs.shutdown_grace_period_secs,
            help="Grace period in seconds to wait for in-flight requests during shutdown",
        )
        request_group.add_argument(
            f"--{prefix}cors-allowed-origins",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help="CORS allowed origins (e.g., http://localhost:3000 https://example.com)",
        )

        # Rate limiting configuration
        rate_limit_group.add_argument(
            f"--{prefix}max-concurrent-requests",
            type=int,
            default=RouterArgs.max_concurrent_requests,
            help=(
                "Maximum standing concurrent requests; each admission permit"
                " is held for the full response, including streaming bodies."
                " Set to -1 to disable."
            ),
        )
        rate_limit_group.add_argument(
            f"--{prefix}queue-size",
            type=int,
            default=RouterArgs.queue_size,
            help=(
                "Queue size for pending requests when max concurrent limit reached"
                " (0 = no queue, return 429 immediately)"
            ),
        )
        rate_limit_group.add_argument(
            f"--{prefix}queue-timeout-secs",
            type=int,
            default=RouterArgs.queue_timeout_secs,
            help="Maximum time (in seconds) a request can wait in queue before timing out",
        )
        rate_limit_group.add_argument(
            f"--{prefix}rate-limit-tokens-per-second",
            type=int,
            default=RouterArgs.rate_limit_tokens_per_second,
            help=(
                "Token bucket refill rate (tokens per second). Unset or 0 ="
                " no refill: --max-concurrent-requests bounds standing"
                " concurrency alone."
            ),
        )

        # Retry configuration
        retry_group.add_argument(
            f"--{prefix}retry-max-retries",
            type=int,
            default=RouterArgs.retry_max_retries,
            help="Maximum number of retry attempts for failed requests",
        )
        retry_group.add_argument(
            f"--{prefix}retry-initial-backoff-ms",
            type=int,
            default=RouterArgs.retry_initial_backoff_ms,
            help="Initial backoff delay in milliseconds before first retry",
        )
        retry_group.add_argument(
            f"--{prefix}retry-max-backoff-ms",
            type=int,
            default=RouterArgs.retry_max_backoff_ms,
            help="Maximum backoff delay in milliseconds between retries",
        )
        retry_group.add_argument(
            f"--{prefix}retry-backoff-multiplier",
            type=float,
            default=RouterArgs.retry_backoff_multiplier,
            help="Multiplier for exponential backoff between retries",
        )
        retry_group.add_argument(
            f"--{prefix}retry-jitter-factor",
            type=float,
            default=RouterArgs.retry_jitter_factor,
            help="Jitter factor (0.0-1.0) to add randomness to retry delays",
        )
        retry_group.add_argument(
            f"--{prefix}disable-retries",
            action="store_true",
            help="Disable retries (equivalent to setting retry_max_retries=1)",
        )

        # Circuit breaker configuration
        cb_group.add_argument(
            f"--{prefix}cb-failure-threshold",
            type=int,
            default=RouterArgs.cb_failure_threshold,
            help="Number of failures before circuit breaker opens",
        )
        cb_group.add_argument(
            f"--{prefix}cb-success-threshold",
            type=int,
            default=RouterArgs.cb_success_threshold,
            help="Number of successes in half-open state before closing circuit",
        )
        cb_group.add_argument(
            f"--{prefix}cb-timeout-duration-secs",
            type=int,
            default=RouterArgs.cb_timeout_duration_secs,
            help="Time in seconds before attempting to close an open circuit",
        )
        cb_group.add_argument(
            f"--{prefix}cb-window-duration-secs",
            type=int,
            default=RouterArgs.cb_window_duration_secs,
            help="Sliding window duration in seconds for tracking failures",
        )
        cb_group.add_argument(
            f"--{prefix}disable-circuit-breaker",
            action="store_true",
            help=(
                "Disable circuit breaker (equivalent to setting cb_failure_threshold to a very large value)"
            ),
        )

        # Health check configuration
        health_group.add_argument(
            f"--{prefix}health-failure-threshold",
            type=int,
            default=RouterArgs.health_failure_threshold,
            help=("Number of consecutive health check failures before marking worker unhealthy"),
        )
        health_group.add_argument(
            f"--{prefix}health-success-threshold",
            type=int,
            default=RouterArgs.health_success_threshold,
            help=("Number of consecutive health check successes before marking worker healthy"),
        )
        health_group.add_argument(
            f"--{prefix}health-check-timeout-secs",
            type=int,
            default=RouterArgs.health_check_timeout_secs,
            help="Timeout in seconds for health check requests",
        )
        health_group.add_argument(
            f"--{prefix}health-check-interval-secs",
            type=int,
            default=RouterArgs.health_check_interval_secs,
            help="Interval in seconds between runtime health checks",
        )
        health_group.add_argument(
            f"--{prefix}health-check-endpoint",
            type=str,
            default=RouterArgs.health_check_endpoint,
            help="Health check endpoint path",
        )
        health_group.add_argument(
            f"--{prefix}disable-health-check",
            action="store_true",
            default=RouterArgs.disable_health_check,
            help="Disable all worker health checks at startup",
        )
        health_group.add_argument(
            f"--{prefix}remove-unhealthy-workers",
            f"--{prefix}worker-auto-recovery",
            action="store_true",
            default=RouterArgs.remove_unhealthy_workers,
            help=(
                "Let workers recover after prolonged failure: unhealthy workers"
                " are removed so service discovery re-registers and re-probes"
                " them once their engine returns"
            ),
        )
        # Tokenizer configuration
        tokenizer_group.add_argument(
            f"--{prefix}model-path",
            f"--{prefix}model",
            type=str,
            default=None,
            help="Model path for loading tokenizer (HuggingFace model ID or local path)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-path",
            type=str,
            default=None,
            help="Explicit tokenizer path (overrides model_path tokenizer if provided)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}chat-template",
            type=str,
            default=None,
            help="Chat template path (optional)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}disable-tokenizer-autoload",
            action="store_true",
            default=RouterArgs.disable_tokenizer_autoload,
            help="Disable automatic tokenizer loading at startup. "
            "Use this when tokenizers are not needed (e.g., pure load balancing).",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-enable-l0",
            action="store_true",
            default=RouterArgs.tokenizer_cache_enable_l0,
            help="Enable L0 (whole-string exact match) tokenizer cache (default: False)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-l0-max-entries",
            type=int,
            default=RouterArgs.tokenizer_cache_l0_max_entries,
            help="Maximum number of entries in L0 tokenizer cache (default: 10000)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-enable-l1",
            action="store_true",
            default=RouterArgs.tokenizer_cache_enable_l1,
            help="Enable L1 (prefix matching) tokenizer cache (default: False)",
        )
        tokenizer_group.add_argument(
            f"--{prefix}tokenizer-cache-l1-max-memory",
            type=int,
            default=RouterArgs.tokenizer_cache_l1_max_memory,
            help="Maximum memory for L1 tokenizer cache in bytes (default: 50MB)",
        )

        # Parser configuration
        reasoning_parser_choices = get_available_reasoning_parsers()
        parser_group.add_argument(
            f"--{prefix}reasoning-parser",
            type=str,
            default=None,
            choices=reasoning_parser_choices,
            help="Specify the parser for reasoning models (e.g., deepseek_r1, qwen3)",
        )
        tool_call_parser_choices = get_available_tool_call_parsers()
        parser_group.add_argument(
            f"--{prefix}tool-call-parser",
            type=str,
            default=None,
            choices=tool_call_parser_choices,
            help="Specify the parser for tool-call interactions (e.g., json, qwen)",
        )
        parser_group.add_argument(
            f"--{prefix}mcp-config-path",
            type=str,
            default=None,
            help="Path to MCP (Model Context Protocol) server configuration file",
        )

        # Backend selection
        backend_group.add_argument(
            f"--{prefix}backend",
            type=str,
            default=RouterArgs.backend,
            choices=["sglang", "openai", "anthropic", "vllm", "tokenspeed"],
            help=(
                "Backend runtime to use (default: sglang). For ZMQ workers, vllm/"
                "tokenspeed also pin the wire protocol (it cannot be auto-detected)"
            ),
        )
        backend_group.add_argument(
            f"--{prefix}zmq-engine-count",
            type=int,
            default=RouterArgs.zmq_engine_count,
            help=(
                "DP engines per startup ZMQ worker: each ipc:// worker becomes a "
                "grouped worker whose handshake awaits this many engines on one "
                "socket set (vLLM and TokenSpeed; default: 1)"
            ),
        )
        backend_group.add_argument(
            f"--{prefix}enable-wasm",
            action="store_true",
            default=None,
            help="Enable WebAssembly (WASM) module support",
        )
        backend_group.add_argument(
            f"--{prefix}storage-hook-wasm-path",
            type=str,
            default=None,
            help="Path to a WASM component implementing storage hooks",
        )
        backend_group.add_argument(
            f"--{prefix}history-backend",
            type=str,
            default=RouterArgs.history_backend,
            choices=["memory", "none", "oracle", "postgres", "redis"],
            help="History storage backend for conversations and responses (default: memory)",
        )

        # Oracle configuration
        oracle_group.add_argument(
            f"--{prefix}oracle-wallet-path",
            type=str,
            default=os.getenv("ATP_WALLET_PATH"),
            help="Path to Oracle ATP wallet directory (env: ATP_WALLET_PATH)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-tns-alias",
            type=str,
            default=os.getenv("ATP_TNS_ALIAS"),
            help="Oracle TNS alias from tnsnames.ora (env: ATP_TNS_ALIAS).",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-connect-descriptor",
            type=str,
            default=os.getenv("ATP_DSN"),
            help="Oracle connection descriptor/DSN (full connection string) (env: ATP_DSN)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-username",
            type=str,
            default=os.getenv("ATP_USER"),
            help="Oracle database username (env: ATP_USER)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-password",
            type=str,
            default=os.getenv("ATP_PASSWORD"),
            help="Oracle database password (env: ATP_PASSWORD)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-external-auth",
            action="store_true",
            default=os.getenv("ATP_EXTERNAL_AUTH", "").lower() in ("1", "true", "yes"),
            help="Enable Oracle external authentication (env: ATP_EXTERNAL_AUTH)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-min",
            type=int,
            default=int(os.getenv("ATP_POOL_MIN", RouterArgs.oracle_pool_min)),
            help="Minimum Oracle connection pool size (default: 1, env: ATP_POOL_MIN)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-max",
            type=int,
            default=int(os.getenv("ATP_POOL_MAX", RouterArgs.oracle_pool_max)),
            help="Maximum Oracle connection pool size (default: 16, env: ATP_POOL_MAX)",
        )
        oracle_group.add_argument(
            f"--{prefix}oracle-pool-timeout-secs",
            type=int,
            default=int(os.getenv("ATP_POOL_TIMEOUT_SECS", RouterArgs.oracle_pool_timeout_secs)),
            help="Oracle connection pool timeout in seconds (default: 30, env: ATP_POOL_TIMEOUT_SECS)",
        )

        # Postgres configuration
        postgres_group.add_argument(
            f"--{prefix}postgres-db-url",
            type=str,
            default=os.getenv("POSTGRES_DB_URL"),
            help="PostgreSQL database connection URL (env: POSTGRES_DB_URL)",
        )
        postgres_group.add_argument(
            f"--{prefix}postgres-pool-max",
            type=int,
            default=int(os.getenv("POSTGRES_POOL_MAX", RouterArgs.postgres_pool_max)),
            help="Maximum PostgreSQL connection pool size (default: 16, env: POSTGRES_POOL_MAX)",
        )

        # Redis configuration
        redis_group.add_argument(
            f"--{prefix}redis-url",
            type=str,
            default=os.getenv("REDIS_URL"),
            help="Redis connection URL (env: REDIS_URL)",
        )
        redis_group.add_argument(
            f"--{prefix}redis-pool-max",
            type=int,
            default=int(os.getenv("REDIS_POOL_MAX", RouterArgs.redis_pool_max)),
            help="Maximum Redis connection pool size (default: 16, env: REDIS_POOL_MAX)",
        )
        redis_group.add_argument(
            f"--{prefix}redis-retention-days",
            type=int,
            default=int(os.getenv("REDIS_RETENTION_DAYS", RouterArgs.redis_retention_days)),
            help="Redis data retention in days (-1 for persistent, default: 30, env: REDIS_RETENTION_DAYS)",
        )

        # Schema configuration
        backend_group.add_argument(
            f"--{prefix}schema-config",
            type=str,
            default=None,
            help="Path to a YAML schema config file for storage table/column remapping",
        )

        # TLS/mTLS configuration
        tls_group.add_argument(
            f"--{prefix}client-cert-path",
            type=str,
            default=None,
            help="Path to client certificate for mTLS authentication with workers",
        )
        tls_group.add_argument(
            f"--{prefix}client-key-path",
            type=str,
            default=None,
            help="Path to client private key for mTLS authentication with workers",
        )
        tls_group.add_argument(
            f"--{prefix}ca-cert-paths",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help=(
                "Path(s) to CA certificate(s) for verifying worker TLS certificates."
                " Can specify multiple CAs."
            ),
        )
        tls_group.add_argument(
            f"--{prefix}tls-cert-path",
            type=str,
            default=None,
            help="Path to server TLS certificate (PEM format)",
        )
        tls_group.add_argument(
            f"--{prefix}tls-key-path",
            type=str,
            default=None,
            help="Path to server TLS private key (PEM format)",
        )

        # Tracing configuration
        trace_group.add_argument(
            f"--{prefix}enable-trace",
            action="store_true",
            help="Enable opentelemetry trace",
        )
        trace_group.add_argument(
            f"--{prefix}otlp-traces-endpoint",
            type=str,
            default="localhost:4317",
            help=(
                "Config opentelemetry collector endpoint if --enable-trace is set."
                " format: <ip>:<port>"
            ),
        )

        # Control plane authentication
        auth_group.add_argument(
            f"--{prefix}api-key",
            type=str,
            default=None,
            help=(
                "The api key used for the authorization with the worker."
                " Useful when the dp aware scheduling strategy is enabled."
            ),
        )
        auth_group.add_argument(
            f"--{prefix}control-plane-api-keys",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help=(
                "API keys for control plane authentication. Format: 'id:name:role:key'"
                " where role is 'admin' or 'user'."
                " Example: --control-plane-api-keys 'key1:Service Account:admin:secret123'"
                " 'key2:Read Only:user:secret456'"
            ),
        )
        auth_group.add_argument(
            f"--{prefix}control-plane-audit-enabled",
            action="store_true",
            default=False,
            help="Enable audit logging for control plane operations",
        )
        auth_group.add_argument(
            f"--{prefix}jwt-issuer",
            type=str,
            default=None,
            help=(
                "OIDC issuer URL for JWT authentication"
                " (e.g., https://login.microsoftonline.com/{tenant}/v2.0)"
            ),
        )
        auth_group.add_argument(
            f"--{prefix}jwt-audience",
            type=str,
            default=None,
            help=(
                "Expected audience claim for JWT tokens (usually the client ID or API identifier)"
            ),
        )
        auth_group.add_argument(
            f"--{prefix}jwt-jwks-uri",
            type=str,
            default=None,
            help=(
                "Explicit JWKS URI. If not provided, discovered from issuer"
                " via .well-known/openid-configuration"
            ),
        )
        auth_group.add_argument(
            f"--{prefix}jwt-role-mapping",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help=(
                "Mapping from IDP role/group names to gateway roles."
                " Format: 'idp_role=gateway_role'."
                " Example: --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'"
            ),
        )

        # Mesh server configuration
        mesh_group = parser.add_argument_group("Mesh Server")
        mesh_group.add_argument(
            f"--{prefix}enable-mesh",
            action="store_true",
            default=False,
            help="Enable mesh server for HA multi-router coordination",
        )
        mesh_group.add_argument(
            f"--{prefix}mesh-server-name",
            type=str,
            default=None,
            help="Mesh server name (default: auto-generated random name)",
        )
        mesh_group.add_argument(
            f"--{prefix}mesh-host",
            type=str,
            default="0.0.0.0",
            help="Mesh server bind address (default: 0.0.0.0)",
        )
        mesh_group.add_argument(
            f"--{prefix}mesh-advertise-host",
            type=str,
            default=None,
            help=(
                "Routable mesh address to advertise to peers."
                " Required when --mesh-host binds to 0.0.0.0."
            ),
        )
        mesh_group.add_argument(
            f"--{prefix}mesh-port",
            type=int,
            default=39527,
            help="Mesh server port (default: 39527)",
        )
        mesh_group.add_argument(
            f"--{prefix}mesh-peer-urls",
            type=str,
            nargs="*",
            action="extend",
            default=[],
            help="Peer mesh server addresses to join (format: host:port)",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace, use_router_prefix: bool = False) -> RouterArgs:
        """
        Create RouterArgs instance from parsed command line arguments.

        Args:
            args: Parsed command line arguments
            use_router_prefix: If True, look for arguments with 'router-' prefix
        """
        prefix = "router_" if use_router_prefix else ""
        cli_args_dict = vars(args)
        args_dict = {}
        disable_arg_fallback = bool(cli_args_dict.get(f"{prefix}disable_arg_fallback", False))

        for attr in dataclasses.fields(cls):
            # Auto strip prefix from args.
            # Prefer the prefixed version (e.g. --router-model-path) when
            # explicitly set, but fall back to the unprefixed version
            # (e.g. --model-path from the backend) when the prefixed key
            # exists but is None (argparse default).
            prefixed_key = f"{prefix}{attr.name}"
            if prefixed_key in cli_args_dict and cli_args_dict[prefixed_key] is not None:
                args_dict[attr.name] = cli_args_dict[prefixed_key]
            elif (
                not disable_arg_fallback
                and attr.name in cli_args_dict
                and cli_args_dict[attr.name] not in (None, "")
            ):
                args_dict[attr.name] = cli_args_dict[attr.name]

            # Special handling for CLI args with dashes vs dataclass fields with underscores
            # e.g. --tls-cert-path maps to tls_cert_path in args namespace,
            # but we might want server_cert_path in dataclass
            # Wait, dataclass fields are server_cert_path/server_key_path
            # CLI args are tls_cert_path/tls_key_path
            # We need to manually map them if names don't match

        # Map tls args to server cert/key path
        if f"{prefix}tls_cert_path" in cli_args_dict:
            args_dict["server_cert_path"] = cli_args_dict[f"{prefix}tls_cert_path"]
        if f"{prefix}tls_key_path" in cli_args_dict:
            args_dict["server_key_path"] = cli_args_dict[f"{prefix}tls_key_path"]

        # parse special arguments and remove "--encode", "--prefill", and "--decode" from cli_args_dict
        args_dict["encode_urls"] = cls._parse_encode_urls(
            cli_args_dict.get(f"{prefix}encode", None)
        )
        args_dict["prefill_urls"] = cls._parse_prefill_urls(
            cli_args_dict.get(f"{prefix}prefill", None)
        )
        args_dict["decode_urls"] = cls._parse_decode_urls(
            cli_args_dict.get(f"{prefix}decode", None)
        )
        args_dict["selector"] = cls._parse_selector(cli_args_dict.get(f"{prefix}selector", None))
        args_dict["encode_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}encode_selector", None)
        )
        args_dict["prefill_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}prefill_selector", None)
        )
        args_dict["decode_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}decode_selector", None)
        )
        args_dict["router_selector"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}router_selector", None)
        )
        args_dict["storage_context_headers"] = cls._parse_selector(
            cli_args_dict.get(f"{prefix}storage_context_headers", None)
        )
        args_dict["model_aliases"] = cls._parse_model_aliases(
            cli_args_dict.get(f"{prefix}model_alias", [])
        )

        # Mooncake-specific annotation
        args_dict["bootstrap_port_annotation"] = "sglang.ai/bootstrap-port"
        args_dict["worker_ports_annotation"] = "smg.ai/worker-ports"

        # Parse control plane API keys
        args_dict["control_plane_api_keys"] = cls._parse_control_plane_api_keys(
            cli_args_dict.get(f"{prefix}control_plane_api_keys", [])
        )

        # Parse JWT role mapping
        args_dict["jwt_role_mapping"] = cls._parse_jwt_role_mapping(
            cli_args_dict.get(f"{prefix}jwt_role_mapping", [])
        )

        return cls(**args_dict)

    def _validate_router_args(self):
        # Validate configuration based on mode
        if self.epd_disaggregation:
            if self.encode_policy:
                logger.info(f"Using --encode-policy '{self.encode_policy}' for encode nodes.")

        if self.pd_disaggregation:
            # Warn about policy usage in PD mode
            if self.prefill_policy and self.decode_policy and self.policy:
                logger.warning(
                    "Both --prefill-policy and --decode-policy are specified. "
                    "The main --policy flag will be ignored for PD mode."
                )
            elif self.prefill_policy and not self.decode_policy and self.policy:
                logger.info(
                    f"Using --prefill-policy '{self.prefill_policy}' for prefill nodes "
                    f"and --policy '{self.policy}' for decode nodes."
                )
            elif self.decode_policy and not self.prefill_policy and self.policy:
                logger.info(
                    f"Using --policy '{self.policy}' for prefill nodes "
                    f"and --decode-policy '{self.decode_policy}' for decode nodes."
                )

    @staticmethod
    def _parse_selector(selector_list):
        if not selector_list:
            return {}

        selector = {}
        # An item may hold several space-separated pairs (OME passes
        # `key1=value1 key2=value2` as a single argv entry).
        for item in selector_list:
            for token in item.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    selector[key] = value
        return selector

    @staticmethod
    def _parse_model_aliases(alias_list):
        if not alias_list:
            return {}

        aliases = {}
        for item in alias_list:
            if "=" not in item:
                raise ValueError(
                    f"Invalid model alias '{item}'. Expected format: <alias>=<canonical>"
                )
            alias, canonical = item.split("=", 1)
            if not alias or not canonical:
                raise ValueError(
                    f"Invalid model alias '{item}'. Alias and canonical model ID must be non-empty"
                )
            if alias == canonical:
                raise ValueError(
                    f"Invalid model alias '{item}'. Alias must differ from the canonical model ID"
                )
            previous = aliases.get(alias)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"Invalid model alias '{alias}'. It maps to both '{previous}' and '{canonical}'"
                )
            aliases[alias] = canonical
        return aliases

    @staticmethod
    def _parse_prefill_urls(prefill_list):
        """Parse prefill URLs from --prefill arguments.

        Format: --prefill URL [BOOTSTRAP_PORT]
        Example:
            --prefill http://prefill1:8080 9000  # With bootstrap port
            --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
            --prefill http://prefill3:8080       # Defaults to no bootstrap port
        """
        if not prefill_list:
            return []

        prefill_urls = []
        for prefill_args in prefill_list:
            url = prefill_args[0]

            # Handle optional bootstrap port
            if len(prefill_args) >= 2:
                bootstrap_port_str = prefill_args[1]
                # Handle 'none' as None
                if bootstrap_port_str.lower() == "none":
                    bootstrap_port = None
                else:
                    try:
                        bootstrap_port = int(bootstrap_port_str)
                    except ValueError:
                        raise ValueError(
                            f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"
                        )
            else:
                # No bootstrap port specified, default to None
                bootstrap_port = None

            prefill_urls.append((url, bootstrap_port))

        return prefill_urls

    @staticmethod
    def _parse_encode_urls(encode_list):
        """Parse encode URLs from --encode arguments.

        Format: --encode URL [BOOTSTRAP_PORT]
        Example:
            --encode http://encode1:8080 9000  # With bootstrap port
            --encode http://encode2:8080 none  # Explicitly no bootstrap port
            --encode http://encode3:8080       # Defaults to no bootstrap port
        """
        return RouterArgs._parse_prefill_urls(encode_list)

    @staticmethod
    def _parse_decode_urls(decode_list):
        """Parse decode URLs from --decode arguments.

        Format: --decode URL
        Example: --decode http://decode1:8081 --decode http://decode2:8081
        """
        if not decode_list:
            return []

        # decode_list is a list of single-element lists due to nargs=1
        return [url[0] for url in decode_list]

    @staticmethod
    def _parse_control_plane_api_keys(api_keys_list):
        """Parse control plane API keys from --control-plane-api-keys arguments.

        Format: id:name:role:key
        Example: --control-plane-api-keys 'key1:Service Account:admin:secret123'
        """
        if not api_keys_list:
            return []

        parsed_keys = []
        for key_str in api_keys_list:
            parts = key_str.split(":", 3)  # Split into at most 4 parts
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid API key format: '{key_str}'. Expected 'id:name:role:key'"
                )
            key_id, name, role, key = parts
            role_lower = role.lower()
            if role_lower not in ("admin", "user"):
                raise ValueError(f"Invalid role: '{role}'. Must be 'admin' or 'user'")
            parsed_keys.append((key_id, name, key, role_lower))
        return parsed_keys

    @staticmethod
    def _parse_jwt_role_mapping(role_mapping_list):
        """Parse JWT role mapping from --jwt-role-mapping arguments.

        Format: idp_role=gateway_role
        Example: --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'
        """
        if not role_mapping_list:
            return {}

        mapping = {}
        for mapping_str in role_mapping_list:
            if "=" not in mapping_str:
                raise ValueError(
                    f"Invalid role mapping format: '{mapping_str}'. Expected 'idp_role=gateway_role'"
                )
            idp_role, gateway_role = mapping_str.split("=", 1)
            gateway_role_lower = gateway_role.lower()
            if gateway_role_lower not in ("admin", "user"):
                raise ValueError(
                    f"Invalid gateway role: '{gateway_role}'. Must be 'admin' or 'user'"
                )
            mapping[idp_role] = gateway_role_lower
        return mapping
