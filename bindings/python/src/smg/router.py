from __future__ import annotations

from smg.router_args import RouterArgs
from smg.smg_rs import (
    BackendType,
    HistoryBackendType,
    PolicyType,
    PyApiKeyEntry,
    PyControlPlaneAuthConfig,
    PyJwtConfig,
    PyOracleConfig,
    PyPostgresConfig,
    PyRedisConfig,
    PyRole,
)
from smg.smg_rs import Router as _Router


def policy_from_str(policy_str: str | None) -> PolicyType:
    """Convert policy string to PolicyType enum."""
    if policy_str is None:
        return None
    policy_map = {
        "random": PolicyType.Random,
        "round_robin": PolicyType.RoundRobin,
        "passthrough": PolicyType.Passthrough,
        "cache_aware": PolicyType.CacheAware,
        "cache_aware_length": PolicyType.CacheAwareLength,
        "power_of_two": PolicyType.PowerOfTwo,
        "least_load": PolicyType.LeastLoad,
        "bucket": PolicyType.Bucket,
        "manual": PolicyType.Manual,
        "consistent_hashing": PolicyType.ConsistentHashing,
        "prefix_hash": PolicyType.PrefixHash,
    }
    return policy_map[policy_str]


def backend_from_str(backend_str: str | None) -> BackendType:
    """Convert backend string to BackendType enum."""
    if isinstance(backend_str, BackendType):
        return backend_str
    if backend_str is None:
        return BackendType.Sglang
    backend_map = {
        "sglang": BackendType.Sglang,
        "openai": BackendType.Openai,
        "anthropic": BackendType.Anthropic,
        # Engine backends: routing behaves like the default; over ZMQ they pin
        # the startup workers' wire protocol (vLLM EngineCore vs TokenSpeed).
        "vllm": BackendType.Vllm,
        "tokenspeed": BackendType.Tokenspeed,
    }
    backend_lower = backend_str.lower()
    if backend_lower not in backend_map:
        raise ValueError(
            f"Unknown backend: {backend_str}. Valid options: {', '.join(backend_map.keys())}"
        )
    return backend_map[backend_lower]


def history_backend_from_str(backend_str: str | None) -> HistoryBackendType:
    """Convert history backend string to HistoryBackendType enum."""
    if isinstance(backend_str, HistoryBackendType):
        return backend_str
    if backend_str is None:
        return HistoryBackendType.Memory
    backend_lower = backend_str.lower()
    if backend_lower == "memory":
        return HistoryBackendType.Memory
    elif backend_lower == "none":
        # Use getattr to access 'None' which is a Python keyword
        return getattr(HistoryBackendType, "None")
    elif backend_lower == "oracle":
        return HistoryBackendType.Oracle
    elif backend_lower == "postgres":
        return HistoryBackendType.Postgres
    elif backend_lower == "redis":
        return HistoryBackendType.Redis
    else:
        raise ValueError(f"Unknown history backend: {backend_str}")


def role_from_str(role_str: str) -> PyRole:
    """Convert role string to PyRole enum."""
    if role_str.lower() == "admin":
        return PyRole.Admin
    return PyRole.User


def build_control_plane_auth_config(
    args_dict: dict,
) -> PyControlPlaneAuthConfig | None:
    """Build control plane auth config from args dict."""
    api_keys = args_dict.get("control_plane_api_keys", [])
    jwt_issuer = args_dict.get("jwt_issuer")
    jwt_audience = args_dict.get("jwt_audience")
    audit_enabled = args_dict.get("control_plane_audit_enabled", False)

    # Check if any auth is configured
    has_api_keys = bool(api_keys)
    has_jwt = jwt_issuer is not None and jwt_audience is not None

    if not has_api_keys and not has_jwt:
        return None

    # Build API key entries
    py_api_keys = []
    for key_tuple in api_keys:
        # Tuple format: (id, name, key, role)
        key_id, name, key, role = key_tuple
        py_api_keys.append(
            PyApiKeyEntry(
                id=key_id,
                name=name,
                key=key,
                role=role_from_str(role),
            )
        )

    # Build JWT config if present
    jwt_config = None
    if has_jwt:
        jwt_config = PyJwtConfig(
            issuer=jwt_issuer,
            audience=jwt_audience,
            jwks_uri=args_dict.get("jwt_jwks_uri"),
            role_mapping=args_dict.get("jwt_role_mapping", {}),
        )

    return PyControlPlaneAuthConfig(
        jwt=jwt_config,
        api_keys=py_api_keys,
        audit_enabled=audit_enabled,
    )


class Router:
    """
    A high-performance router for distributing requests across worker nodes.

    Args:
        worker_urls: List of URLs for worker nodes that will handle requests. Each URL
            should include the protocol, host, and port (e.g.,
            ['http://worker1:8000', 'http://worker2:8000'])
        policy: Load balancing policy to use. Options:
            - PolicyType.Random: Randomly select workers
            - PolicyType.RoundRobin: Distribute requests in round-robin fashion
            - PolicyType.CacheAware: Distribute requests based on cache state and load
              balance
            - PolicyType.PowerOfTwo: Sample two random workers and route to the
              one with the lower expected wait (least-load scoring over the pair)
            - PolicyType.LeastLoad: Route to the worker with the lowest load score
              (in-flight requests plus KV-cache pressure)
        host: Host address to bind the router server. Supports IPv4, IPv6 (e.g., ::,
            ::1), or 0.0.0.0 for all interfaces. Default: '0.0.0.0'
        port: Port number to bind the router server. Default: 3001
        worker_startup_timeout_secs: Timeout in seconds for worker startup and
            registration. Large models can take significant time to load into GPU
            memory. Default: 1800 (30 minutes)
        worker_startup_delay: Grace period in seconds before the first worker
            startup check fires. Leaves the engine alone this long, then polls
            every worker_startup_check_interval. Default: 0
        worker_startup_check_interval: Interval in seconds between checks for worker
            initialization. Default: 30
        job_queue_capacity: Max pending control-plane jobs (worker add/remove,
            tokenizer, MCP, WASM). Size to fleet scale so a service-discovery
            reconcile pass can enqueue every worker without blocking.
            Default: 1000
        job_queue_concurrency: Max control-plane jobs dispatched concurrently.
            Default: 200
        cache_threshold: Cache threshold (0.0-1.0) for cache-aware routing. Routes to
            cached worker if the match rate exceeds threshold, otherwise routes to the
            worker with the smallest tree. Default: 0.5
        cache_boundaries: Token positions at which serving engines retain reusable
            prefix state; cache-affinity policies hash request heads at the deepest
            applicable boundary. Default: []
        cache_index: Index under-layer for cache_aware: 'tree' (radix prefix trees)
            or 'hash' (TTL'd exact-match placement map keyed on request heads at
            cache_boundaries; token-bearing requests only — untokenized requests
            stay load-balanced). Default: 'tree'
        cache_ttl_secs: Seconds a cache-affinity placement stays routable; should
            approximate serving-engine cache retention. Default: 180
        prefix_token_count: Number of prefix tokens hashed by the prefix_hash
            policy, or four times as many characters of the prompt when the
            request is untokenized. Size it past any shared system prompt so
            distinct conversations hash apart. Default: 256
        prefix_hash_load_factor: Load factor above which the prefix_hash policy
            walks the ring instead of using the hashed worker (multiple of the
            average load). Default: 1.25
        prefix_hash_balance_abs_threshold: Absolute load difference over average
            a worker must also exceed before prefix_hash treats it as
            overloaded. Guards the load factor against sampling noise when each
            router replica sees only a share of a worker's load. Default: 10
        balance_abs_threshold: Load balancing is triggered when (max_load - min_load) >
            abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache
            aware. Default: 32
        balance_rel_threshold: Load balancing is triggered when (max_load - min_load) >
            abs_threshold AND max_load > min_load * rel_threshold. Otherwise, use cache
            aware. Default: 1.0001
        eviction_interval_secs: Interval in seconds between cache eviction operations
            in cache-aware routing. Default: 60
        overlap_decay: Cache-aware anti-hotspot decay: each candidate's overlap score
            is divided by 1 + overlap_decay * x, where x is the worker's
            waiting-prefill backlog (blocks above the candidate minimum) per request
            block. Requires backend load reporting. Default: 0.0 (disabled)
        selection_temperature: Cache-aware softmax temperature over min-max
            normalized scores for event-driven selection. 0.0 is exact argmax;
            larger values spread picks across candidates. Default: 0.0
        max_payload_size: Maximum payload size in bytes. Default: 256MB
        max_tree_size: Maximum total size of each model's approximation tree for
            cache-aware routing (chars for HTTP, tokens for gRPC), shared across
            all workers; eviction keeps every tree at or under this bound.
            Default: 2^26
        dp_aware: Enable data parallelism aware schedule. Default: False
        dp_minimum_tokens_scheduler: Enable minimum tokens scheduler for data parallel group. Default: False
        upstream_http2: Speak HTTP/2 to workers via prior knowledge (h2c on
            cleartext), multiplexing every request to a worker over one
            connection. Requires every HTTP worker to serve HTTP/2 without an
            upgrade handshake. Default: False
        enable_igw: Enable IGW (Inference-Gateway) mode for multi-model support. When
            enabled, the router can manage multiple models simultaneously with per-model
            load balancing policies. Default: False
        api_key: The api key used for the authorization with the worker. Useful when
            the dp aware scheduling strategy is enabled. Default: None
        log_dir: Directory to store log files. If None, logs are only output to
            console. Default: None
        log_level: Logging level. Options: 'debug', 'info', 'warn', 'error'.
        service_discovery: Enable Kubernetes service discovery. When enabled, the
            router will automatically discover worker pods based on the selector.
            Default: False
        selector: Dictionary mapping of label keys to values for Kubernetes pod
            selection. Example: {"app": "sglang-worker"}. Default: {}
        service_discovery_port: Port to use for service discovery. The router will
            generate worker URLs using this port. Default: 80
        service_discovery_namespace: Kubernetes namespace to watch for pods. If not
            provided, watches pods across all namespaces (requires cluster-wide
            permissions). Default: None
        prefill_selector: Dictionary mapping of label keys to values for Kubernetes
            pod selection for prefill servers (PD mode only). Default: {}
        decode_selector: Dictionary mapping of label keys to values for Kubernetes pod
            selection for decode servers (PD mode only). Default: {}
        prometheus_port: Port to expose Prometheus metrics. Default: None
        prometheus_host: Host address to bind the Prometheus metrics server. Default:
            None
        pd_disaggregation: Enable PD (Prefill-Decode) disaggregated mode. Default:
            False
        prefill_urls: List of (url, bootstrap_port) tuples for prefill servers (PD
            mode only)
        decode_urls: List of URLs for decode servers (PD mode only)
        prefill_policy: Specific load balancing policy for prefill nodes (PD mode
            only). If not specified, uses the main policy. Default: None
        decode_policy: Specific load balancing policy for decode nodes (PD mode only).
            If not specified, uses the main policy. Default: None
        request_id_headers: List of HTTP headers to check for request IDs. If not
            specified, uses common defaults: ['x-request-id', 'x-correlation-id',
            'x-trace-id', 'request-id']. Example: ['x-my-request-id',
            'x-custom-trace-id']. Default: None
        storage_context_headers: Mapping of incoming HTTP headers to storage hook
            context keys. Example: {'x-tenant-id': 'tenant_id'}. Default: {}
        bootstrap_port_annotation: Kubernetes annotation name for bootstrap port (PD
            mode). Default: 'sglang.ai/bootstrap-port'
        worker_ports_annotation: Kubernetes annotation listing a pod's worker data
            ports (comma-separated) for pods running multiple engine servers.
            Absent = one worker at service_discovery_port. Default:
            'smg.ai/worker-ports'
        request_timeout_secs: Request timeout in seconds. Default: 600
        max_concurrent_requests: Maximum number of concurrent requests allowed for
            rate limiting. Default: 256
        queue_size: Queue size for pending requests when max concurrent limit reached
            (0 = no queue, return 429 immediately). Default: 100
        queue_timeout_secs: Maximum time (in seconds) a request can wait in queue
            before timing out. Default: 60
        rate_limit_tokens_per_second: Token bucket refill rate (tokens per second). If
            not set, defaults to max_concurrent_requests. Default: None
        cors_allowed_origins: List of allowed origins for CORS. Empty list allows all
            origins. Default: []
        health_failure_threshold: Number of consecutive health check failures before
            marking worker unhealthy. Default: 3
        health_success_threshold: Number of consecutive health check successes before
            marking worker healthy. Default: 2
        health_check_timeout_secs: Timeout in seconds for health check requests.
            Default: 5
        health_check_interval_secs: Interval in seconds between runtime health checks.
            Default: 60
        health_check_endpoint: Health check endpoint path. Default: '/health'
        model_path: Model path for loading tokenizer (HuggingFace model ID or local
            path). Default: None
        tokenizer_path: Explicit tokenizer path (overrides model_path tokenizer if
            provided). Default: None
        server_cert_path: Path to server TLS certificate (PEM format). Default: None
        server_key_path: Path to server TLS private key (PEM format). Default: None
    """

    def __init__(self, router: _Router):
        self._router = router

    @staticmethod
    def from_args(args: RouterArgs) -> Router:
        """Create a router from a RouterArgs instance."""

        args_dict = vars(args).copy()
        # Convert RouterArgs to _Router parameters
        args_dict["worker_urls"] = (
            []
            if (
                args_dict["service_discovery"]
                or args_dict["pd_disaggregation"]
                or args_dict["epd_disaggregation"]
            )
            else args_dict["worker_urls"]
        )
        args_dict["policy"] = policy_from_str(args_dict["policy"])
        args_dict["encode_urls"] = (
            args_dict["encode_urls"] if args_dict["epd_disaggregation"] else None
        )
        args_dict["prefill_urls"] = (
            args_dict["prefill_urls"]
            if args_dict["pd_disaggregation"] or args_dict["epd_disaggregation"]
            else None
        )
        args_dict["decode_urls"] = (
            args_dict["decode_urls"]
            if args_dict["pd_disaggregation"] or args_dict["epd_disaggregation"]
            else None
        )
        args_dict["encode_policy"] = policy_from_str(args_dict["encode_policy"])
        args_dict["prefill_policy"] = policy_from_str(args_dict["prefill_policy"])
        args_dict["decode_policy"] = policy_from_str(args_dict["decode_policy"])

        # Convert backend
        args_dict["backend"] = backend_from_str(args_dict.get("backend"))

        # Convert history_backend to enum first
        history_backend_raw = args_dict.get("history_backend", "memory")
        history_backend = history_backend_from_str(history_backend_raw)

        # Convert Oracle config if needed
        oracle_config = None
        if history_backend == HistoryBackendType.Oracle:
            # Prioritize TNS alias over connect descriptor
            tns_alias = args_dict.get("oracle_tns_alias")
            connect_descriptor = args_dict.get("oracle_connect_descriptor")

            # Use TNS alias if provided, otherwise use connect descriptor
            final_descriptor = tns_alias if tns_alias else connect_descriptor

            oracle_config = PyOracleConfig(
                password=args_dict.get("oracle_password"),
                username=args_dict.get("oracle_username"),
                connect_descriptor=final_descriptor,
                wallet_path=args_dict.get("oracle_wallet_path"),
                external_auth=args_dict.get("oracle_external_auth", False),
                pool_min=args_dict.get("oracle_pool_min", 1),
                pool_max=args_dict.get("oracle_pool_max", 16),
                pool_timeout_secs=args_dict.get("oracle_pool_timeout_secs", 30),
            )
        args_dict["oracle_config"] = oracle_config
        args_dict["history_backend"] = history_backend

        # Convert Postgres config if needed
        postgres_config = None
        if history_backend == HistoryBackendType.Postgres:
            postgres_config = PyPostgresConfig(
                db_url=args_dict.get("postgres_db_url"),
                pool_max=args_dict.get("postgres_pool_max", 16),
            )
        args_dict["postgres_config"] = postgres_config

        # Convert Redis config if needed
        redis_config = None
        if history_backend == HistoryBackendType.Redis:
            retention_days = args_dict.get("redis_retention_days", 30)
            # If retention_days is negative, it means persistent storage (None in Rust)
            retention_arg = None if retention_days < 0 else retention_days

            redis_config = PyRedisConfig(
                url=args_dict.get("redis_url"),
                pool_max=args_dict.get("redis_pool_max", 16),
                retention_days=retention_arg,
            )
        args_dict["redis_config"] = redis_config

        # Build control plane auth config
        args_dict["control_plane_auth"] = build_control_plane_auth_config(args_dict)

        # Remove fields that shouldn't be passed to Rust Router constructor
        fields_to_remove = [
            "oracle_wallet_path",
            "oracle_tns_alias",
            "oracle_connect_descriptor",
            "oracle_username",
            "oracle_password",
            "oracle_external_auth",
            "oracle_pool_min",
            "oracle_pool_max",
            "oracle_pool_timeout_secs",
            "postgres_db_url",
            "postgres_pool_max",
            "redis_url",
            "redis_pool_max",
            "redis_retention_days",
            # Control plane auth fields (converted to control_plane_auth)
            "control_plane_api_keys",
            "control_plane_audit_enabled",
            "jwt_issuer",
            "jwt_audience",
            "jwt_jwks_uri",
            "jwt_role_mapping",
        ]
        for field in fields_to_remove:
            args_dict.pop(field, None)

        return Router(_Router(**args_dict))

    def start(self) -> None:
        """Start the router server.

        This method blocks until the server is shut down.
        """
        self._router.start()
