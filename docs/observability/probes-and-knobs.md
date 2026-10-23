# Probes, metrics endpoints and observability knobs

What an operator points a scraper, a probe or a log shipper at, and the flags
and environment variables that turn each signal on. The families themselves
are in the [metrics catalog](metrics-catalog.md) and their labels in the
[label reference](metric-labels.md). Paths below are relative to the
repository root; the flag table is generated from the gateway's command-line
definitions by `python3 -I scripts/metrics_catalog.py`.

## Endpoints

| Route | Listener | Answer | What it tells you |
|---|---|---|---|
| `GET /liveness` | data port (`--port`), the probe port (`--health-check-port`) and the admin port (`--admin-port`) when set | `200 OK` while the process serves | The process is up and answering; it never turns 503, not even while draining. |
| `GET /health` | same as `/liveness` | `200 OK` | An alias of `/liveness` (the chart's default liveness probe). |
| `GET /readiness` | same as `/liveness` | `200 {"status":"ready", …}` or `503 {"status":"not ready","reason":…}` | Whether the gateway can take traffic. The reasons: `insufficient healthy workers` (no healthy worker for a model, or a prefill/decode role under its threshold), `tokenizer not yet registered` (a gRPC model whose tokenizer is still loading), `draining` (graceful shutdown began). A ready body carries `roles` with per-role counts on prefill/decode deployments and `models_without_media_processing` when a multimodal model has no media-capable worker. |
| `GET /health_generate` | data port | the worker's answer | A deep probe: a small generate request proxied to a worker. For a manual check, not for a kubelet probe (it generates tokens). |
| `GET /metrics` | `--prometheus-port` (default 29000; `--prometheus-host` picks the interface) | the Prometheus text format | Every family of the [catalog](metrics-catalog.md) that has been recorded, plus the pre-registered zero series. The chart names this port `metrics` on the Service and the ServiceMonitor scrapes it. |
| `GET /engine_metrics` | data port (admin route) | JSON | Pulls the HTTP workers' own `/metrics` on demand; gRPC workers are not included (their engine-side figures reach the gateway as load reports, the `smg_engine_*` families). |
| `GET /workers`, `GET /get_model_info`, `GET /get_server_info`, `GET /v1/models` | data port, or the admin port when set (`/workers`) | JSON | The fleet as the gateway sees it: workers with their health, type, connection mode and load; the served models. |
| `GET /metrics` on the engine pod | the Rust servicer's `--metrics-port` (`--metrics-host`; env `SMG_VLLM_SERVICER_METRICS_PORT` / `SMG_SGLANG_SERVICER_METRICS_PORT` / `SMG_TOKENSPEED_SERVICER_METRICS_PORT` and the `_HOST` forms) | the Prometheus text format | The `smg_servicer_*` families (labels `model`, `wire`, `rank`): requests by finish reason, refusals, errors by gRPC code, in-flight requests, prompt / cached / generated tokens, TTFT, inter-token, per-request mean and end-to-end latency histograms, the servicer's lifecycle state and start time. Off unless the port is given; the Python servicer exposes none. |

### How readiness is decided

The decision is O(1) per probe: a maintainer task re-derives it from the
worker registry on every worker event (bursts coalesced) and on a short
checkpoint interval, and publishes a snapshot the probe handlers read with one
atomic load. Probes therefore keep answering while the request runtime is
saturated, and `--health-check-port` moves them onto a listener with no
middleware on a runtime of its own, so a kubelet deadline is met even then.
The probe routes always stay on the data port too.

Once graceful shutdown begins, `/readiness` answers 503 with the reason
`draining` on every listener while `/liveness` and `/health` keep answering
200, so the load balancer stops sending new connections without the
orchestrator restarting the process. The drain lasts up to
`--shutdown-grace-period-secs`.

On prefill/decode deployments readiness needs one healthy worker per role by
default; `--prefill-min-ready-workers`, `--decode-min-ready-workers`,
`--<role>-expected-workers` and `--<role>-min-ready-percent` raise the bar
(the required count is the larger of the absolute minimum and the percentage
rounded up). A router whose workers register at runtime through
`POST /workers` is not ready until the first one does.

### Worker health probing

The gateway probes every worker's `--health-check-endpoint` (default
`/health`) every `--health-check-interval-secs` (60) with a
`--health-check-timeout-secs` (5) deadline. `--health-failure-threshold` (3)
consecutive failures take a worker out of rotation,
`--health-success-threshold` (2) successes bring it back;
`--disable-health-check` turns probing off. The probes feed
`smg_worker_health`, `smg_worker_health_checks_total` and the circuit
breaker's `smg_worker_cb_*` families; a worker's series leave `/metrics`
when its registration is removed.

## Flags

Every gateway flag has an environment-free command-line form; the Python
launcher exposes the same flags with a `--router-` prefix (`--router-prometheus-port`).

<!-- generated: observability-flags -->
| Flag | Default | Help heading | What it does |
|---|---|---|---|
| `--decode-expected-workers` | `unset` | Health Checks | Desired decode fleet size used as the fixed readiness percentage denominator. |
| `--decode-min-ready-percent` | `unset` | Health Checks | Minimum healthy percentage of expected decode workers (1-100). |
| `--decode-min-ready-workers` | `unset` | Health Checks | Absolute minimum number of healthy decode workers required (default 1). |
| `--disable-health-check` | `false` | Health Checks | Disable all worker health probing |
| `--drain-settle-secs` | `5` | Health Checks | Seconds to keep a Ready worker in `Draining` before removing it from the registry. Applies to all RemoveWorker submissions (K8s deletion, `--remove-unhealthy-workers`, manual API). Per-worker overrides are supported via `WorkerSpec::health.drain_settle_secs`. Set to `0` to remove immediately without draining. |
| `--health-check-endpoint` | `/health` | Health Checks | HTTP path probed on each worker |
| `--health-check-interval-secs` | `60` | Health Checks | Interval in seconds between health probes of each worker (at least 1; --disable-health-check turns probing off) |
| `--health-check-timeout-secs` | `5` | Health Checks | Timeout in seconds for a single health probe (at least 1) |
| `--health-failure-threshold` | `3` | Health Checks | Consecutive probe failures before a worker is taken out of rotation (at least 1) |
| `--health-success-threshold` | `2` | Health Checks | Consecutive probe successes before a worker returns to rotation (at least 1) |
| `--prefill-expected-workers` | `unset` | Health Checks | Desired number of prefill workers used as the stable readiness percentage denominator. This does not enable a percentage gate by itself; set `--prefill-min-ready-percent` to do that. |
| `--prefill-min-ready-percent` | `unset` | Health Checks | Minimum healthy percentage of the expected prefill workers (1-100). |
| `--prefill-min-ready-workers` | `unset` | Health Checks | Absolute minimum number of healthy prefill workers required. |
| `--remove-unhealthy-workers` | `unset` | Health Checks | Recover failed workers by removal: a worker that stays unhealthy long enough (Failed, ~12 minutes at the default thresholds) is removed from the registry so service discovery re-registers and re-probes it once its engine returns. Without this a Failed worker stays registered, out of rotation, and keeps being probed, so it rejoins in place as soon as it answers again. Unset, it follows discovery: with a discovery provider it removes only the workers that provider discovered, which it re-registers, while static and API-added workers, which nothing would re-add, recover in place. Pass it to remove every failed local worker, or =false to remove none. |
| `--engine-metrics` | `false` | Load Monitoring | Force GetLoads polling for smg_engine_* Prometheus gauges even without a load-aware routing policy. Routing-owned polls are always re-exported. A worker whose KV-event stream pushes its load feeds the gauges from those records and is not polled while they flow (GetLoads is the fallback). |
| `--access-log-fields` | `unset` | Logging | Optional fields of the access-log record, comma-separated: client_addr, forwarded_for, user_agent, tenant, key_id, request_bytes, response_bytes, worker (default: none, the record keeps the fields it has always had) |
| `--log-dir` | `unset` | Logging | Directory for the log files, created at start-up when missing (under a writable directory); an existing path must be a writable directory (unset: console only) |
| `--log-json` | `false` | Logging | Output logs as JSON |
| `--log-level` | `info` | Logging | Set the logging level |
| `--log-mm-timing` | `false` | Multimodal | Emit per-request multimodal timing at INFO. Falls back to `SMG_LOG_MM_TIMING`. |
| `--jemalloc-prof-dir` | `unset` | Profiling | Directory where `POST /heap_profile` (an admin route) writes a jemalloc heap profile of this gateway. Needs a build with the `jemalloc-profiling` feature, started with `_RJEM_MALLOC_CONF=prof:true,prof_active:true`; unset, the route answers 404 |
| `--prometheus-duration-buckets` | `unset` | Prometheus Metrics | Custom buckets for Prometheus duration metrics |
| `--prometheus-host` | `unset` | Prometheus Metrics | Host address to bind the Prometheus metrics server. Defaults to the unspecified address of `--host`'s family: `::` for an IPv6 host, `0.0.0.0` otherwise. |
| `--prometheus-port` | `29000` | Prometheus Metrics | Port to expose Prometheus metrics (at least 1). Each listener (--port, --prometheus-port, --health-check-port, --mesh-port) needs its own port. |
| `--shutdown-grace-period-secs` | `180` | Request Handling | Grace period in seconds to wait for in-flight requests during shutdown |
| `--enable-trace` | `false` | Tracing (OpenTelemetry) | Enable OpenTelemetry tracing |
| `--otlp-traces-endpoint` | `localhost:4317` | Tracing (OpenTelemetry) | OTLP collector endpoint (format: host:port) |
| `--admin-host` | `unset` | Worker Configuration | Bind host of the admin listener (needs --admin-port). Defaults to --host |
| `--admin-port` | `unset` | Worker Configuration | Port of the admin listener. When set, the control-plane routes (the admin routes /flush_cache, /start_profile, /stop_profile, /heap_profile, /get_loads, /parse/*, /wasm and /v1/tokenizers, the worker-registry routes /workers and the RL routes /v1/rl) are served there, with /health, /liveness and /readiness, and the data listener (--port) answers 404 for them: the control plane can sit on a private address, behind its own network policy, off the data port's ingress. Control-plane authentication applies on it as it does on the data listener. The public /loads stays on the data listener (a gateway registered as a worker is polled there) and names the backends as before. Each listener needs its own port. Unset = the control plane stays on the data listener |
| `--health-check-port` | `unset` | Worker Configuration | Dedicated port for liveness/readiness/health probes (Kubernetes, load balancers, uptime monitors, etc.).  When set, `/liveness`, `/readiness`, and `/health` are additionally served on this port by a middleware-free router running on its own single-worker runtime and OS thread, isolated from the request runtime so a saturated gateway cannot starve probes (and trigger the failed-probe restarts or depooling that follow) under load. The same probe routes always remain available on the main `--port` too. Unset = dedicated probe listener off. |
<!-- /generated: observability-flags -->

### Histogram buckets

`--prometheus-duration-buckets` replaces the request-duration edge set, which
the request duration, generation duration, TTFT and TPOT histograms share
(the default edges and the other sets are listed at the end of the
[catalog](metrics-catalog.md#bucket-edge-sets)). The KV-event, KV-index,
event-loop canary, cache match ratio, retry backoff and Symphony byte
histograms keep their own sets.

## Environment variables

| Variable | Effect |
|---|---|
| `SMG_CACHE_TRACE=1` | Logs the evidence of a routing decision (the candidates, scores and gates the policy saw) as an INFO line, one dispatch in every `SMG_CACHE_TRACE_SAMPLE` (default every dispatch); a failed request is always logged. |
| `SMG_CACHE_TRACE_HEADER=1` | Returns the decision record of a dispatch in the `x-smg-cache-trace` response header (not sampled). Independently of both switches, a request that sends `x-smg-cache-trace: 1` gets its own record. |
| `SMG_CACHE_TRACE_MAX_BYTES=B` | An evidence line longer than `B` bytes drops its candidate, score and gate lists and keeps the decision record. |
| `SMG_LOG_MM_TIMING=1` | Per-request multimodal timing at INFO (the `--log-mm-timing` flag's environment form). |
| `SMG_<ENGINE>_SERVICER_METRICS_PORT`, `SMG_<ENGINE>_SERVICER_METRICS_HOST` | The Rust servicer's `--metrics-port` / `--metrics-host` for `VLLM`, `SGLANG` and `TOKENSPEED`; the flags override them. |

## Logging

`--log-level` (`debug`, `info`, `warn`, `error`; default `info`) applies to
the workspace crates; dependencies log at WARN. `--log-json` switches the
console and file sinks to JSON lines with the request id and the trace id;
`--log-dir` adds a file sink. Sinks are non-blocking and batched: a sink that
cannot keep up drops lines and counts them in `smg_log_dropped_lines{sink}`
rather than stalling the request path, and a hot call site is throttled with
a suppressed count.

The access log writes one line per request with the method, the matched route,
the status, the latency, the request id and the trace id; `--access-log-fields`
adds `client_addr`, `forwarded_for`, `user_agent`, `tenant`, `key_id`,
`request_bytes`, `response_bytes` and `worker`. Probe responses (the 200s and
the expected 503s of `/readiness`) are logged at DEBUG so a tight probe
interval does not fill the log. `--request-id-headers` names the client
headers whose value becomes the request id (1 to 128 visible ASCII
characters; otherwise a fresh id is generated).

## Tracing

`--enable-trace` exports one `http_request` span per request, covering the
whole response stream, to the OTLP collector at `--otlp-traces-endpoint`
(`host:port`, default `localhost:4317`). W3C `traceparent` / `tracestate`
headers from the client are honoured and propagated to the workers over HTTP
headers and gRPC metadata; the Rust servicer hands the context to the
engine's own tracer, so the engine's spans for the request join the same
trace. The request id and the trace id are on the access log line of the
request, which is how a log line and a trace are joined.

## Helm chart keys

| Key | Default | Sets |
|---|---|---|
| `router.metrics.port` | `29000` | `--prometheus-port`; the `metrics` port of the container and the Service |
| `router.metrics.serviceMonitor.enabled`, `.interval`, `.labels` | `false`, `30s`, `{}` | A `ServiceMonitor` on the router's `metrics` port |
| `router.tracing.enabled`, `.otlpEndpoint` | `false`, `""` | `--enable-trace`, `--otlp-traces-endpoint` |
| `router.logging.level`, `.json`, `.dir` | `info`, `false`, `""` | `--log-level`, `--log-json`, `--log-dir` |
| `router.livenessProbe.path`, `router.readinessProbe.path` | `/health`, `/readiness` | The kubelet probes |
| `grafana.dashboard.enabled`, `.labels` | `false`, `{grafana_dashboard: "1"}` | The Grafana dashboard ConfigMap the sidecar loader picks up |

`deploy/helm/smg/examples/with-monitoring.yaml` turns the ServiceMonitor and
the dashboard on.
