# SMG Helm Chart

Helm chart for deploying the Shepherd Model Gateway (SMG) — a high-performance inference router for LLM deployments. Supports deploying the gateway with optional worker (inference engine) pods.

## Prerequisites

- Kubernetes >= 1.26
- Helm >= 3.12
- GPU nodes with `nvidia.com/gpu` resource (for workers)

## Quick Start

### Router only (external workers)

```bash
helm install smg deploy/helm/smg \
  --set router.workerUrls[0]=http://worker-1:8000 \
  --set router.workerUrls[1]=http://worker-2:8000
```

### Router + vLLM worker

```bash
helm install smg deploy/helm/smg \
  -f deploy/helm/smg/examples/with-vllm-http.yaml \
  --set huggingface.token=hf_xxx
```

### Router + SGLang worker

```bash
helm install smg deploy/helm/smg \
  -f deploy/helm/smg/examples/with-sglang.yaml \
  --set huggingface.token=hf_xxx
```

## Images

The gateway image is on Docker Hub, engine images are on GitHub Container Registry:

| Component | Registry | Image |
|-----------|----------|-------|
| Gateway | `docker.io` | `lightseekorg/smg:1.13.0` |
| vLLM worker | `ghcr.io` | `smg-project/smg:1.13.0-vllm-v0.31.0` |
| TRT-LLM worker | `ghcr.io` | `smg-project/smg:1.13.0-trtllm-1.3.0rc24` |
| SGLang worker | `ghcr.io` | `smg-project/smg:1.13.0-sglang-v0.5.21` |

Worker images default to `ghcr.io/smg-project/smg`. Set `workers[].image.tag` to the engine-specific tag; `workers[].image.registry` and `workers[].image.repository` override the path.

The TRT-LLM image is published, but the chart does not template `engine: trtllm` yet: `engineDefaults` knows `vllm` and `sglang`.

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Sections

| Section | Description |
|---------|-------------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability |
| `workers` | Worker (inference engine) deployments |
| `huggingface` | HuggingFace token for gated model downloads |
| `auth` | API key authentication, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) and the Secret its connection URL or credentials come from |
| `serviceAccount` | Service account creation and annotations |
| `rbac` | RBAC for reading pods through the Kubernetes API (Kubernetes discovery, mesh peer discovery) |

### Routing Policies

`cache_aware` (default), `round_robin`, `power_of_two`, `consistent_hashing`, `prefix_hash`, `manual`, `random`, `bucket`

### Workers

Each entry in `workers[]` creates a Deployment + Service:

```yaml
workers:
  - name: llama-8b                    # unique name
    engine: vllm                      # vllm or sglang
    model: meta-llama/Llama-3.1-8B-Instruct
    replicas: 1
    connectionMode: http              # http or grpc
    gpu:
      count: 1                        # GPUs per replica (tensor parallel)
    image:
      tag: "1.13.0-vllm-v0.31.0"     # engine image tag
```

Workers are automatically wired to the router via `--worker-urls`, unless the router discovers its workers ([Worker Discovery](#worker-discovery)). The URL scheme (`http://` or `grpc://`) is set based on `connectionMode`.

#### Connection Modes

| Mode | Description | Worker Command |
|------|-------------|----------------|
| `http` | OpenAI-compatible HTTP API | vLLM: `vllm.entrypoints.openai.api_server` |
| `grpc` | SMG gRPC pipeline | vLLM: `vllm.entrypoints.grpc_server` |

SGLang uses `sglang.launch_server` for both modes (adds `--grpc-mode` for gRPC).

#### Worker Metrics (gRPC mode)

A gRPC worker can serve Prometheus metrics from the engine pod itself:

```yaml
workers:
  - name: llama-8b
    engine: vllm
    connectionMode: grpc
    metrics:
      enabled: true
      port: 29100                       # default; must differ from the serving port
```

The chart then runs the engine behind the Rust servicer and gives its
metrics listener a port (the pod gets `SMG_<ENGINE>_SERVICER_IMPL=rust` and
`SMG_<ENGINE>_SERVICER_METRICS_PORT=<port>`, the variables the launcher reads
when the flags are absent), names the port `metrics` on the container and
the worker Service, and, when `router.metrics.serviceMonitor.enabled` is on,
adds one `<release>-worker` ServiceMonitor over the worker Services with the
router's interval and labels. The endpoint answers `GET /metrics` with the
`smg_servicer_*` families (requests by finish reason, refusals, errors,
in-flight requests, prompt / cached / generated tokens, the time-to-first-token,
inter-token and end-to-end latency histograms, the servicer's lifecycle state),
labelled `model`, `rank` and `wire`; it binds beside the gRPC listener before
the engine connects, so a scrape during start-up answers
`smg_servicer_state{state="starting"} 1`.

Off (the default), nothing changes: the launcher's own servicer choice stands
and the pod exposes no metrics port. The Python servicer has no endpoint, so
`extraArgs: ["--servicer-impl", "python"]` together with `metrics.enabled`
is refused by the launcher before the engine starts. The SGLang Rust servicer
serves neither PD/EPD disaggregation nor image and video input yet (it
advertises no vision support, so the gateway refuses media for that worker at
its gate), so a disaggregated or multimodal SGLang worker keeps `metrics` off.
A `metrics.port` equal to the serving port is refused when the chart renders.
An HTTP worker runs the engine's own server, which has its own `/metrics`; the
setting is ignored there.

#### Worker Discovery

`router.discovery.provider` selects where the router finds its workers at runtime, passed as `--discovery-provider` with the settings of that provider's block under `router.discovery`: `kubernetes`, `file`, `slurm` or `consul` (empty, the default, discovers nothing). `router.discovery.checkIntervalSecs` sets the provider's periodic pass (`--discovery-check-interval-secs`, 30 s when unset), under `router.serviceDiscovery` too. Selecting a provider turns IGW mode on in the gateway, and the chart's `workers` are then not passed as `--worker-urls`: the provider is the source of workers.

| Provider | Workers | Needs |
|---|---|---|
| `kubernetes` | pods matching `kubernetes.selector` in `kubernetes.namespace` (the release's by default; every namespace with `clusterWide`) | a Role, or a ClusterRole with `clusterWide`, which the chart creates |
| `file` | the workers listed in a JSON manifest, reread when it changes | the manifest: an existing ConfigMap (`file.configMap.name`, its key `file.configMap.key`), or a path in the image (`file.path`) |
| `slurm` | a worker on every node of each running job named in `slurm.jobNames` (or `encodeJobNames`, `prefillJobNames`, `decodeJobNames`), at `slurm.port` | an image with Slurm clients (below) |
| `consul` | the instances of `consul.service` with passing health checks | Consul's address, and the ACL token and TLS files from existing Secrets |

`router.serviceDiscovery.enabled` is the legacy spelling of `provider: kubernetes` and renders the arguments it always has; `router.discovery.kubernetes` takes the same settings. The chart refuses the two together, as the gateway refuses `--service-discovery` beside `--discovery-provider`. It also refuses a provider without what it cannot start without (a file provider without a manifest, Slurm without a port or a job name, Consul without a service, a Consul client certificate without its key), so the mistake surfaces at `helm install` rather than as a crash-looping pod.

The Role and RoleBinding (`get`, `list` and `watch` on pods) exist for what reads pods through the Kubernetes API: Kubernetes worker discovery, by either spelling, and mesh without `router.mesh.peerUrls`, which finds its peers among the release's router pods. The file, Slurm and Consul providers never call the API, so they get none unless `rbac.create` is set; nor does a mesh with `peerUrls`, which earlier chart versions gave one. A Role grants nothing outside its own namespace, so both are created in the namespace Kubernetes discovery watches (`kubernetes.namespace`, or `serviceDiscovery.namespace`), the release's by default; earlier chart versions always created them in the release's, where a discovery of another namespace was refused.

Mesh without `router.mesh.peerUrls` finds its peers through Kubernetes discovery, the only place the gateway runs router discovery. With no provider, the chart turns on Kubernetes discovery of the release's own workers for it, as it always has. Under `provider: kubernetes` it adds the router selector, naming the release so that another release's routers are not taken for peers, to the provider, whose namespace router discovery then watches: the provider has to watch the release namespace (no `clusterWide`, no other `namespace`), or another namespace finds no peer and every namespace finds other releases' routers. The file, Slurm and Consul providers cannot carry router discovery, so a mesh with one of them needs `router.mesh.peerUrls`; the chart refuses it without.

**File.** The ConfigMap is mounted whole at `/etc/smg/discovery`, not as a subPath, so the kubelet projects each update of it into the pod and the gateway's watch on that directory picks it up; the manifest path is `/etc/smg/discovery/<key>` unless `file.path` names another. The volume is optional: the gateway waits for a manifest that is not there yet, so the ConfigMap can be published after the release. The manifest format is in the gateway's file provider (`model_gateway/src/service_discovery/file.rs`); [`with-file-discovery.yaml`](examples/with-file-discovery.yaml) has an example.

**Slurm.** The provider runs `squeue` and `scontrol` (Slurm 23.11 or later) in the router container, and the published gateway image has no Slurm client. It needs an image built from the gateway's with those commands and what the cluster's authentication needs, its `slurm.conf` and MUNGE, or `auth/jwt` with `SLURM_JWT` (which `router.extraEnv` can take from a Secret); the chart mounts nothing for it. The router runs as UID 65532, so set `slurm.user` to the user whose jobs are the workers. Every node of a selected job runs one server on `slurm.port`, so worker jobs should run `--exclusive`.

**Consul.** The ACL token and the TLS files reach the gateway through Consul's own environment variables, never as arguments. `consul.token.existingSecret` sets `CONSUL_HTTP_TOKEN` from that Secret's `consul.token.secretKey` (default `token`). `consul.tls.existingSecret` mounts the Secret's `consul.tls.secretKeys` at `/etc/smg/consul-tls` and points `CONSUL_CACERT` (`caCert`, default `ca.crt`) and, when both are set, `CONSUL_CLIENT_CERT` and `CONSUL_CLIENT_KEY` (`clientCert`, `clientKey`) at them; give `consul.address` an `https://` scheme. The gateway reads them once, at start-up, so restart the router pods after rotating either Secret.

#### HuggingFace Token

Gated models (e.g., Llama) require a HuggingFace token:

```bash
helm install smg deploy/helm/smg \
  -f examples/with-vllm-http.yaml \
  --set huggingface.token=hf_xxx
```

The token is stored in a Kubernetes Secret and injected as `HF_TOKEN` env var into both the router and worker pods.

#### API Key

The shared API key the serving routes require (`auth.apiKey`) is written into a chart-managed Secret and mounted into the router pod as a file the gateway reads with `--api-key-file`; the key is never in the pod's arguments, so it is not in the Deployment object, the process listing or `/proc`. A key managed outside the chart's values is mounted the same way from an existing Secret:

```bash
kubectl create secret generic smg-auth --from-literal=api-key='<key>'
helm install smg deploy/helm/smg \
  --set auth.existingSecret=smg-auth \
  --set auth.apiKeySecretKey=api-key   # the default
```

The gateway also reads its other secrets from the environment or from files, which `router.extraEnv` can feed from Secrets: `SMG_TENANT_API_KEYS` (or `SMG_TENANT_API_KEYS_FILE`) for per-tenant keys; the store URLs and the Oracle credentials have values of their own ([Store URLs](#store-urls), [Oracle Credentials](#oracle-credentials)). On the command line a flag beats the environment, and a value together with a file for the same secret is refused at start-up. The launcher's reference in [the Python package's README](../../../bindings/python/README.md#secrets-from-the-environment-or-files) lists every name.

The gateway reads the key file once, at start-up, so a rotated key reaches a pod when the pod restarts. A rotated `auth.apiKey` rolls the router pods by itself: the pod template carries a `checksum/auth` annotation over the chart-managed Secret, so `helm upgrade` with a new key is a rolling update. A rotated `auth.existingSecret` is not seen by the chart; roll the pods by hand after updating it, or the running pods keep accepting the old key until they restart. By label, which covers both workload kinds and every release name: `kubectl rollout restart deployment,statefulset -l app.kubernetes.io/instance=<release>,app.kubernetes.io/component=router` (the workload is `<fullname>-router`, where `<fullname>` is `<release>-smg` unless the release name contains `smg` or `fullnameOverride` is set).

An older gateway image does not know `--api-key-file`; a chart of this version with such an image refuses to start rather than starting without the key.

#### Store URLs

The connection URL of the history store (`history.postgres.url`, `history.redis.url`) carries the database password, so the chart writes it into a chart-managed Secret and the router pod gets it as the environment variable the gateway reads (`SMG_POSTGRES_DB_URL`, `SMG_REDIS_URL`); the URL is never in the pod's arguments, so it is not in the Deployment object, the process listing or `/proc`. A URL managed outside the chart's values comes from an existing Secret the same way:

```bash
kubectl create secret generic smg-db-credentials --from-literal=postgres-url='postgresql://smg:<password>@postgres:5432/smg'
helm install smg deploy/helm/smg \
  --set history.backend=postgres \
  --set history.postgres.existingSecret=smg-db-credentials \
  --set history.postgres.secretKey=postgres-url   # the default (redis: redis-url)
```

A changed `url` rolls the router pods: the pod template carries a checksum of the chart-managed Secret. The checksum is a hash of the Secret's rendered contents, in which the URL is the only unknown, so a URL set in the values should carry a high-entropy password (anyone who can read the Deployment can test guesses against the hash); an existing Secret exposes no hash. The existing Secret's name must differ from the chart-managed one (`<fullname>-history`, so `smg-history` for a release named `smg`), or a later switch to `url` fails the upgrade because the chart cannot take over a Secret it did not create. A rotated existing Secret does not roll the pods, as the gateway reads the URL once at start-up; restart the router pods after rotating it (`kubectl rollout restart` of the router Deployment, or of the StatefulSet in mesh mode). With neither `url` nor `existingSecret` the chart passes no URL, and the gateway refuses to start a store backend without one, unless `router.extraEnv` provides the variable.

An older gateway image does not read `SMG_POSTGRES_DB_URL` / `SMG_REDIS_URL`; a chart of this version with such an image refuses to start the store backend rather than starting it without the URL.

#### Oracle Credentials

The Oracle backend's connect descriptor, database user and password (`history.oracle.dsn`, `user`, `password`) are written into the chart-managed Secret `<fullname>-secrets` (`<fullname>` is `<release>-smg` unless the release name contains `smg` or `fullnameOverride` is set: `smg-secrets` for a release named `smg`, `prod-smg-secrets` for one named `prod`) and the router pod gets them as the environment variables the gateway reads (`ATP_DSN`, `ATP_USER`, `ATP_PASSWORD`); none is in the pod's arguments. Credentials managed outside the chart's values come from an existing Secret holding the three keys:

```bash
kubectl create secret generic smg-oracle \
  --from-literal=oracle-dsn='<connect descriptor>' \
  --from-literal=oracle-user='<user>' \
  --from-literal=oracle-password='<password>'
helm install smg deploy/helm/smg \
  --set history.backend=oracle \
  --set history.oracle.existingSecret=smg-oracle \
  --set history.oracle.secretKeys.password=oracle-password   # the defaults: oracle-dsn, oracle-user, oracle-password
```

A changed credential rolls the router pods, as a changed store URL does: the pod template carries a checksum of the chart-managed Secret. The checksum is a hash of the Secret's contents, so a password set in the values should be a high-entropy one (anyone who can read the Deployment can test guesses against the hash); an existing Secret exposes no hash, and a rotation of it needs the same restart as a rotated store URL. With `existingSecret` all three variables are wired, so the Secret holds the three keys; a deployment that needs fewer (external authentication, a wallet with a TNS alias) sets the credentials it has under `history.oracle` or wires them with `router.extraEnv`. Earlier chart versions wrote the `<fullname>-secrets` Secret with these keys without reading it: an upgrade keeps the Secret and moves the credentials from the router's arguments to its environment. The existing Secret's name must differ from `<fullname>-secrets`, the Secret an earlier chart version created included: Helm deletes that Secret at the upgrade that stops rendering it (an `existingSecret` with no credentials in the values), and the router pods would then name a Secret that is gone; to keep using those credentials, copy them into a Secret with another name, as the `smg-oracle` example above does. The pool size (`history.oracle.poolMax`) stays an argument; the other Oracle settings (a wallet path, a TNS alias, external authentication, the pool bounds) are `ATP_*` variables for `router.extraEnv` or flags for `router.extraArgs`. Every gateway image with the Oracle backend reads these three variables.

#### Router Probes and Shutdown

The router's liveness probe is `httpGet /health`, which answers 200 while the process serves. Its readiness probe is `httpGet /readiness`, which answers 503 while the gateway has no healthy worker, while a tokenizer is not yet registered and while the gateway drains after SIGTERM, so the Service stops sending new connections to a pod that cannot take them (`router.livenessProbe.path` and `router.readinessProbe.path` override the routes). Both readiness bodies carry a `models` section, one entry per served model with the same predicates applied to that model's workers (`ready`, a `reason` when it is not, its healthy and registered worker counts), for a rollout or traffic switch that keys on one model while the status answers for the gateway as a whole.

PD readiness defaults to one healthy prefill and one healthy decode worker. For some
workloads, that remaining capacity is already insufficient: continued traffic can
worsen SLA violations until the service fails. Use `router.extraArgs` to set
`--prefill-min-ready-workers` and `--decode-min-ready-workers` to the minimum viable
P/D capacity. Each role also supports `--<role>-expected-workers` with
`--<role>-min-ready-percent` (1–100); the required count is the larger of the
absolute minimum (default 1) and the percentage rounded up. For example,
`--prefill-expected-workers 8 --prefill-min-ready-percent 75 --decode-min-ready-workers 2`
requires 6P and 2D. The expected size stays fixed when workers disappear; setting
it alone does not enable a percentage gate. These options apply to PD/EPD and are
also available in the Python launcher. `/readiness` returns 503 below either
threshold and includes per-role counts; the load balancer must honor readiness
to stop new traffic. The thresholds do not enforce an exact P:D ratio or shed
requests themselves.

A router that gets its workers at runtime through the admin API (`POST /workers`) rather than from `workers`, `router.workerUrls`, `router.serviceDiscovery` or `router.discovery` has no healthy worker when it starts, so `/readiness` answers 503 until one is registered, the Service has no endpoint for the pod, and a registration sent through the Service cannot reach it (a rolling update's new pod then never becomes Ready). Such deployments set `router.readinessProbe.path: /health`, or register their workers through a port-forward to the pod.

The gateway drains in-flight requests for `router.shutdownGracePeriodSeconds` (default 180, passed as `--shutdown-grace-period-secs`); the pod's `terminationGracePeriodSeconds` is that plus `router.terminationGraceMarginSeconds` (default 30), so the kubelet does not kill the drain at its default 30 s.

During shutdown, readiness fails immediately while listeners stay open for half of
`router.shutdownGracePeriodSeconds` by default (90 seconds with the default grace).
Set `router.shutdownSettleSeconds` (`--shutdown-settle-secs`) to cap this propagation
window; the effective window is the smaller of this value and half the grace.
The remaining grace drains existing requests. A value of 0 skips the propagation
wait (mesh departure still shares the total grace budget).

#### Dashboard and Alerts

`grafana.dashboard.enabled` ships a Grafana dashboard as a ConfigMap (label `grafana_dashboard: "1"` for the sidecar loader) with `namespace` and `job` variables that pick the release (every query is scoped to both) and 41 panels in seven rows: traffic (request rate, responses by status, 5xx ratio, errors by API code and router error type, token throughput), latency (TTFT and time per output token p50/p95/p99 and per model, request and HTTP duration p99), capacity and queues (admission in-flight and queued, rejections, engine running and waiting requests, KV cache usage and in-flight request age per worker, connections), workers (healthy count, health and circuit-breaker state per worker, health checks, overload shed and fallback, retries), cache and KV events (engine prefix cache hit rate, cache-aware decisions and match ratio, KV event lag, integrity counters, degraded ranks, index size), Symphony (outputs by outcome, malformed outputs by reason, fallbacks and refusals) and runtime (tokio busy ratio, event-loop stalls and delay, dropped log lines, memory, tasks, tokenizer cache). Every panel queries a family the gateway describes; the gateway's test suite holds the dashboard to the catalog.

`prometheusRule.enabled` adds a `PrometheusRule` with ten alerts, thresholds under `prometheusRule.alerts` (the defaults and their rationale are in `values.yaml`). Every expression selects this release's series through `prometheusRule.jobSelector` (empty = this release's namespace and the Service the ServiceMonitor scrapes, the Operator's default `namespace` and `job` labels, so a same-named release in another namespace stays out) and keeps `namespace` and `job` through its aggregation, so several releases on one Prometheus alert apart and a firing alert names its gateway:

| Alert | Severity | Fires when |
|---|---|---|
| `SmgGatewayDown` | critical | the router's scrape target is down, or no target matches the selector, for `scrapeMissingFor` |
| `SmgNoHealthyWorker` | critical | no registered worker is healthy for `noHealthyWorkerFor` |
| `SmgHighErrorRatio` | critical | the 5xx share of responses exceeds `errorRatio` for `errorRatioFor` |
| `SmgTtftP99High` | critical | p99 time to first token exceeds `ttftP99Seconds` for `ttftP99For` |
| `SmgKvUsageNearCapacity` | warning | a worker's KV token usage exceeds `kvUsage` for `kvUsageFor` |
| `SmgCircuitBreakerOpen` | warning | a worker's circuit breaker is open or half-open for `circuitBreakerOpenFor` (a worker failing requests while passing its probes cycles between the two) |
| `SmgAdmissionRejections` | warning | the admission queue rejects requests for `admissionRejectionsFor` |
| `SmgEventLoopStalls` | warning | more than `eventLoopStalls` event-loop stalls in ten minutes |
| `SmgDroppedLogLines` | warning | a log sink dropped lines in the last ten minutes |
| `SmgKvEventDegradedRanks` | warning | an engine rank's KV index stays degraded for `degradedRanksFor` |

Readiness itself is a kubelet signal: with kube-state-metrics in the cluster, `kube_pod_status_ready` on the router pods complements `SmgGatewayDown`.

#### Readiness Deadlines

`/readiness` follows the engine, not the router process. A worker from `router.workerUrls`, `router.serviceDiscovery` or `router.discovery` is registered only once its engine reports healthy: the gateway polls it every `--worker-startup-check-interval` (30 s) for at most `--worker-startup-timeout-secs` (1800 s), and a worker whose engine is not healthy by then is not registered (a discovered worker is tried again on a later discovery pass, one named in `router.workerUrls` is not). The router therefore answers `/readiness` 503 for the whole of the engine's bring-up (weights, initialisation, warm-up: tens of minutes for a large checkpoint streamed from object storage, 37 minutes measured for a two-node tensor-parallel gang), while `/health` and `/liveness` answer 200 from the first second, and it becomes Ready within a poll of the engine.

Every deadline on the router's readiness has to cover that bring-up:

- The router Deployment's `progressDeadlineSeconds` (`router.progressDeadlineSeconds`, 1920 by default): the gateway's start-up bound, 1800 s, plus the chart's tail after it (one start-up poll, the readiness probe's period, the registration) and a margin for the image pull and container start, which run on the rollout's clock before the gateway's; the Kubernetes default is 600. Kubernetes takes no action on it, but `kubectl rollout status`, Argo CD and Flux report the rollout as failed once it passes.
- `helm install --wait` and `helm upgrade --wait` time out after 5 minutes by default: pass `--timeout` with the engine's budget (with `--atomic`, the upgrade would otherwise be rolled back, or the install removed, while the engine loads).
- A serving controller that runs the router as a component of a model service, with an instance-ready or readiness deadline of its own: give the router component the engine component's budget. A controller whose router deadline was 30 minutes declared the router failed at the deadline, relocated it with an anti-affinity to its node that could not be scheduled, and left the service without an endpoint although the router became healthy seven minutes later; with the engine's four-hour budget on the router, the same service came up and served.
- `--worker-startup-timeout-secs` itself (`router.extraArgs`) for an engine that needs more than 30 minutes, and `router.progressDeadlineSeconds` with it.

The budget is not a startup probe on the router's `/readiness`: a startup probe that fails for its whole budget restarts the container, and the router process is not what is slow.

#### Startup Probes

Model loading takes from a minute to tens of minutes, depending on the checkpoint and where it is streamed from. Workers have startup probes with these defaults (`initialDelaySeconds` plus `failureThreshold` times `periodSeconds`: 360 s of budget):

- HTTP mode: `httpGet /health` with `initialDelaySeconds: 60`, `failureThreshold: 30`
- gRPC mode: `tcpSocket` probe (not all engines implement gRPC health checking yet)

Override per worker:

```yaml
workers:
  - name: large-model
    startupProbe:
      initialDelaySeconds: 180
      failureThreshold: 60
```

Size the budget past the longest load you expect: a startup probe that fails for its whole budget restarts the worker, and the load starts over. The worker Deployment's rollout deadline follows the budget (`initialDelaySeconds + failureThreshold * periodSeconds`, plus 600 s for what runs on the rollout's clock before the probe's starts: the node's provisioning, the image pull, the container start), so `kubectl rollout status`, Argo CD, Flux and `helm --wait` do not report a worker still loading within its budget as a failed rollout. On an autoscaled GPU pool or with a cold image cache, where the node and a multi-GB pull can take longer than that margin, set `progressDeadlineSeconds` on the worker entry past them; it overrides the default.

## Examples

| File | Scenario |
|------|----------|
| [`router-only.yaml`](examples/router-only.yaml) | Minimal router with external workers |
| [`with-vllm-http.yaml`](examples/with-vllm-http.yaml) | Router + vLLM worker (HTTP) |
| [`with-vllm-grpc.yaml`](examples/with-vllm-grpc.yaml) | Router + vLLM worker (gRPC) |
| [`with-sglang.yaml`](examples/with-sglang.yaml) | Router + SGLang worker |
| [`with-multi-model.yaml`](examples/with-multi-model.yaml) | Multiple models on different engines |
| [`with-postgres.yaml`](examples/with-postgres.yaml) | PostgreSQL history backend, its URL in a Secret |
| [`with-service-discovery.yaml`](examples/with-service-discovery.yaml) | K8s auto-discovery |
| [`with-file-discovery.yaml`](examples/with-file-discovery.yaml) | Workers from a manifest in a ConfigMap |
| [`with-consul-discovery.yaml`](examples/with-consul-discovery.yaml) | Workers from Consul, token and CA from Secrets |
| [`with-slurm-discovery.yaml`](examples/with-slurm-discovery.yaml) | Workers on Slurm job nodes (custom image) |
| [`with-ingress.yaml`](examples/with-ingress.yaml) | Ingress with TLS |
| [`with-monitoring.yaml`](examples/with-monitoring.yaml) | ServiceMonitor + Grafana dashboard + PrometheusRule |

## Testing

```bash
helm test smg
```

## Troubleshooting

### Worker not registering

Check worker pod logs:

```bash
kubectl logs -l smg.lightseek.org/worker=<worker-name> -f
```

Model loading can take several minutes. The router will register the worker once it passes the startup probe.

### Tokenizer download fails (read-only filesystem)

The router sets `HF_HOME=/tmp/hf-cache` to use the writable `/tmp` mount. If you see "Read-only file system" errors, ensure the `tmp` volume is mounted.

### Service discovery not finding pods

Verify RBAC is enabled and the selector matches your worker pod labels:

```bash
kubectl get role,rolebinding -l app.kubernetes.io/instance=smg
kubectl get pods -l <your-selector>
```
