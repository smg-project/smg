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
| Gateway | `docker.io` | `lightseekorg/smg:1.14.0` |
| Gateway with the Slurm clients | `docker.io` | `lightseekorg/smg:<version>-slurm`, beside every release image from 1.16.0 on |
| Policy controller | `docker.io` | `lightseekorg/smg:<version>-policy-controller`, beside every release image from 1.16.0 on |
| vLLM worker | `ghcr.io` | `smg-project/smg:1.14.0-vllm-v0.31.0` |
| TRT-LLM worker | `ghcr.io` | `smg-project/smg:1.14.0-trtllm-1.3.0rc24` |
| SGLang worker | `ghcr.io` | `smg-project/smg:1.14.0-sglang-v0.5.21` |

Worker images default to `ghcr.io/smg-project/smg`. Set `workers[].image.tag` to the engine-specific tag; `workers[].image.registry` and `workers[].image.repository` override the path.

The worker images are published for `linux/amd64` only. On a node of another architecture a worker's image pull fails (`ImagePullBackOff`): keep the workers on amd64 nodes with `workers[].nodeSelector` (`kubernetes.io/arch: amd64`), or set `workers[].image` to an image built for that node's architecture.

A tag is text: quote one made of digits alone (`tag: "1.16"`, or `--set-string` on the command line). `global.image.tag` and `workers[].image.tag` written as a number are refused at render, as `policyController.image.tag` is: a number is not rendered as written, and `1.10` would read as `1.1`.

The TRT-LLM image is published, but the chart does not template `engine: trtllm` yet: `engineDefaults` knows `vllm` and `sglang`.

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Sections

| Section | Description |
|---------|-------------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability |
| `workers` | Worker (inference engine) deployments |
| `policyController` | The standalone policy controller: its Deployment and Service, off by default |
| `huggingface` | HuggingFace token for gated model downloads |
| `auth` | API key authentication, control-plane keys, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) and the Secret its connection URL or credentials come from |
| `serviceAccount` | Service account creation and annotations |
| `rbac` | RBAC for reading pods through the Kubernetes API (Kubernetes discovery, mesh peer discovery) |

### Routing Policies

The chart's `router.policy` takes `cache_aware` (default), `round_robin`, `power_of_two`, `least_load`, `passthrough`, `consistent_hashing`, `prefix_hash`, `manual`, `random`, and `bucket`: the launcher's `--policy` choices, as listed in `values.schema.json`.

Under a disaggregation mode, a decode role with no `decodePolicy` runs under `router.policy`. The gateway refuses `bucket` for the decode role under `epd`, and under `pd` with `router.enableIgw`: set `router.disaggregation.decodePolicy` beside `router.policy: bucket` there.

### Disaggregated Serving

`router.disaggregation.mode` selects the gateway's `--pd-disaggregation` (`pd`) or `--epd-disaggregation` (`epd`); empty, the default, is a regular fleet. The role policies `prefillPolicy`, `decodePolicy` and `encodePolicy` are the launcher's `--prefill-policy`, `--decode-policy` and `--encode-policy` choices, which the schema lists (`bucket` is a prefill-side choice). A prefill or decode role with no policy of its own runs under the fleet's policy, and an encode role with none under `consistent_hashing`; on the decode pool a load policy (`decodePolicy: power_of_two`, the launcher's own example) is the lighter choice for the gateway, because a `cache_aware` decode pool also ingests the decode workers' cache events. The role workers are either static URLs, `prefill`, `decode` and `encode` lists of `{url, bootstrapPort}` rendered as `--prefill URL [BOOTSTRAP_PORT]`, `--decode URL` and `--encode URL [BOOTSTRAP_PORT]`, or, under Kubernetes discovery, the pods matching `router.discovery.kubernetes.prefillSelector`, `decodeSelector` and `encodeSelector`. A role given without its mode, or the encode role outside `epd`, is refused at render rather than ignored by the gateway at start-up; a typo in a role key is refused by the schema. A prefill-decode fleet found by label:

```yaml
router:
  disaggregation:
    mode: pd
    prefillPolicy: cache_aware
    decodePolicy: power_of_two
  discovery:
    provider: kubernetes
    kubernetes:
      prefillSelector: "app=engine role=prefill"
      decodeSelector: "app=engine role=decode"
```

Under a mode the gateway reads only its role workers. `router.workerUrls` and the regular Slurm `jobNames` are refused, the chart's own `workers` are passed only through a role or a discovery provider, and chart workers given no role fail the render. The Slurm role job names `prefillJobNames`, `decodeJobNames` and `encodeJobNames` follow the same rule as the role selectors.

### Admission Bound

`router.maxConcurrentRequests` is the gateway's `--max-concurrent-requests`: the in-flight requests it admits before shedding with 429 or 503, each permit held for the whole response, streaming bodies included. `-1` (the default) derives the bound from the host, 1024 per available core and at least 4096; `0` lifts the bound; a positive value is the bound.

### Native JPEG Decoder

`router.mmTurbojpeg` is the gateway's `--mm-turbojpeg`: what it does when libjpeg-turbo (`libturbojpeg.so.0`) cannot be loaded. `require` (the default) refuses to start, so a gateway never decodes JPEGs with the pure-Rust fallback, whose pixels differ from the engine's by a few levels. `prefer` decodes with the fallback, says so once at start-up and sets the gauge `smg_mm_turbojpeg_available` to 0. The published image ships the library; a custom image without it needs `prefer`.

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
      tag: "1.14.0-vllm-v0.31.0"     # engine image tag
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

Off (the default), the pod exposes no metrics port and the launcher's own
servicer choice stands: the Rust servicer, its default, unless the worker's
`servicerImpl` is `"python"` (the deprecated Python servicer, kept for the
deprecation window; the pod then gets `SMG_<ENGINE>_SERVICER_IMPL=python`).
The Python servicer has no metrics endpoint, so `servicerImpl: "python"` (or
`extraArgs: ["--servicer-impl", "python"]`) together with `metrics.enabled`
is refused by the launcher before the engine starts. A `metrics.port` equal to
the serving port is refused when the chart renders.
An HTTP worker runs the engine's own server, which has its own `/metrics`; the
setting is ignored there.

#### Parser

`router.parser` picks the parser that reads a chat completion's reasoning and
tool calls: `symphony` (one table per model family, the gateway's default from
the release after 1.13.0 on) or `legacy` (the older per-model parsers). Empty,
the default, renders no `--parser`, so the image's own default applies and the
value is safe on every image. A set value needs a gateway image of 1.13.0 or
later: the flag arrived in 1.13.0 (with `legacy` as its default there), and an
earlier image rejects it and exits at startup. The choice applies to gRPC and
ZMQ workers; an HTTP worker's engine parses its own output, so the value changes
nothing there. A deployment that set `--parser` through `router.extraArgs` removes
it there when it sets this value: the gateway refuses the flag twice and exits at
startup. A request's `X-SMG-Parser` header overrides the flag for that
request. `router.toolCallParser` and `router.reasoningParser` are read
differently by each parser: under Symphony, a model the registry knows keeps
its own table and the flags apply only to models it does not know; under
`legacy`, each flag takes only a legacy parser's name of its own kind: `hf`
and a Symphony-only name (`muse`, for one) are refused at startup, while a
name a legacy parser of that kind also has (`qwen` as `toolCallParser`, for
one) keeps that parser.

#### Worker Discovery

`router.discovery.provider` selects where the router finds its workers at runtime, passed as `--discovery-provider` with the settings of that provider's block under `router.discovery`: `kubernetes`, `file`, `slurm` or `consul` (empty, the default, discovers nothing). `router.discovery.checkIntervalSecs` sets the provider's periodic pass (`--discovery-check-interval-secs`, 30 s when unset), under `router.serviceDiscovery` too. Selecting a provider turns IGW mode on in the gateway, and the chart's `workers` are then not passed as `--worker-urls`: the provider is the source of workers.

| Provider | Workers | Needs |
|---|---|---|
| `kubernetes` | pods matching `kubernetes.selector` in `kubernetes.namespace` (the release's by default; every namespace with `clusterWide`) | a Role, or a ClusterRole with `clusterWide`, which the chart creates |
| `file` | the workers listed in a JSON manifest, reread when it changes | the manifest: an existing ConfigMap (`file.configMap.name`, its key `file.configMap.key`), or a path in the image (`file.path`) |
| `slurm` | a worker on every node of each running job named in `slurm.jobNames` (or `encodeJobNames`, `prefillJobNames`, `decodeJobNames`), at `slurm.port` | the gateway image's `-slurm` variant (below) |
| `consul` | the instances of `consul.service` with passing health checks | Consul's address, and the ACL token and TLS files from existing Secrets |

`router.serviceDiscovery.enabled` is the legacy spelling of `provider: kubernetes` and renders the arguments it always has; `router.discovery.kubernetes` takes the same settings. The chart refuses the two together, as the gateway refuses `--service-discovery` beside `--discovery-provider`. It also refuses a provider without what it cannot start without (a file provider without a manifest, Slurm without a port or a job name, Consul without a service, a Consul client certificate without its key), so the mistake surfaces at `helm install` rather than as a crash-looping pod.

The Role and RoleBinding (`get`, `list` and `watch` on pods) exist for what reads pods through the Kubernetes API: Kubernetes worker discovery, by either spelling, and mesh without `router.mesh.peerUrls`, which finds its peers among the release's router pods. The file, Slurm and Consul providers never call the API, so they get none unless `rbac.create` is set; nor does a mesh with `peerUrls`, which earlier chart versions gave one. A Role grants nothing outside its own namespace, so both are created in the namespace Kubernetes discovery watches (`kubernetes.namespace`, or `serviceDiscovery.namespace`), the release's by default; earlier chart versions always created them in the release's, where a discovery of another namespace was refused.

Mesh without `router.mesh.peerUrls` finds its peers through Kubernetes discovery, the only place the gateway runs router discovery. With no provider, the chart turns on Kubernetes discovery of the release's own workers for it, as it always has. Under `provider: kubernetes` it adds the router selector, naming the release so that another release's routers are not taken for peers, to the provider, whose namespace router discovery then watches: the provider has to watch the release namespace (no `clusterWide`, no other `namespace`), or another namespace finds no peer and every namespace finds other releases' routers. The file, Slurm and Consul providers cannot carry router discovery, so a mesh with one of them needs `router.mesh.peerUrls`; the chart refuses it without.

**File.** The ConfigMap is mounted whole at `/etc/smg/discovery`, not as a subPath, so the kubelet projects each update of it into the pod and the gateway's watch on that directory picks it up; the manifest path is `/etc/smg/discovery/<key>` unless `file.path` names another. The volume is optional: the gateway waits for a manifest that is not there yet, so the ConfigMap can be published after the release. The manifest format is in the discovery crate's file provider (`crates/service_discovery/src/file.rs`); [`with-file-discovery.yaml`](examples/with-file-discovery.yaml) has an example.

**Slurm.** The provider runs `squeue` and `scontrol` (Slurm 23.11 or later) in the router container. The published gateway image has no Slurm client and refuses to start the provider, naming the missing command; its `-slurm` variant (`global.image.tag: "<version>-slurm"`, the release tag with `-slurm` appended, published beside every release image from 1.16.0 on and as `ghcr.io/smg-project/smg:nightly-slurm`) carries Ubuntu's Slurm 23.11 client package and its JWT authentication plugin. A controller accepts commands up to two major releases older than itself (three from 24.11 on), so those clients speak to a controller on 23.11, 24.05, 24.11 or 25.05; a newer cluster needs an image with its own clients. What the cluster's authentication needs still comes from the deployment: its `slurm.conf`, and a running `munged`'s socket (a sidecar or the host; the image starts none and ships no MUNGE key) or `auth/jwt` with `SLURM_JWT` (which `router.extraEnv` can take from a Secret); the chart mounts nothing for it. The router runs as UID 65532, so set `slurm.user` to the user whose jobs are the workers. Every node of a selected job runs one server on `slurm.port`, so worker jobs should run `--exclusive`.

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

#### Control Plane

`router.adminPort` starts the admin listener (`--admin-port`): the control-plane routes (the admin routes, `/workers` and `/v1/rl`) are served on that container port, with the probes, and the data port answers 404 for them. The chart renders a second Service, `<fullname>-router-admin` (`router.adminService`, ClusterIP by default), so the control plane has a name of its own inside the cluster; the ingress never names it. `router.adminHost` binds the listener on another address. A loopback `router.adminHost` (`127.0.0.1`, `::1`) leaves the admin Service with nowhere to send traffic: Service traffic reaches the pod IP. A host without `router.adminPort`, or an admin port equal to another listener's, is refused at render.

The control-plane API keys (`auth.controlPlane`) reach the gateway as a file, like the shared key: `auth.controlPlane.apiKeys` (one `id:name:role:key` entry per line, role `admin` or `user`) is written into a chart-managed Secret, or `auth.controlPlane.existingSecret` names one, mounted at `/etc/smg/control-plane` and read with `--control-plane-api-keys-file`. A rotated `auth.controlPlane.apiKeys` rolls the router pods (`checksum/control-plane`); a rotated existing Secret needs the rollout restart described above. [`examples/with-admin-plane.yaml`](examples/with-admin-plane.yaml) sets both.

```bash
kubectl create secret generic smg-control-plane \
  --from-file=control-plane-api-keys=./control-plane-keys.txt
helm install smg deploy/helm/smg \
  --set router.adminPort=9001 \
  --set auth.controlPlane.existingSecret=smg-control-plane
```

#### Admission Config

`router.tenantRateLimit` and `router.priorityScheduler` carry the admission configuration the gateway reads from files: `enabled` renders the feature's flag (`--tenant-rate-limit-enabled`, `--priority-scheduler-enabled`), and `configMap.name` names a ConfigMap of yours, mounted whole at `/etc/smg/rate-limit` or `/etc/smg/scheduler` and read through `--tenant-rate-limit-config` or `--priority-scheduler-config` (`configMap.key` names the file: `rate-limit.yaml` and `scheduler.yaml` by default). The chart refuses what the gateway would refuse or ignore: the rate limit enabled without its ConfigMap, and a rate-limit or scheduler ConfigMap without its feature enabled.

```bash
kubectl create configmap smg-rate-limit --from-file=rate-limit.yaml=./rate-limit.yaml
helm install smg deploy/helm/smg \
  --set router.tenantRateLimit.enabled=true \
  --set router.tenantRateLimit.configMap.name=smg-rate-limit
```

The gateway reads both files once at start. An edit of a ConfigMap reaches the mounted file through the kubelet's symlink swap, but it takes effect on the next `kubectl rollout restart` of the router Deployment; the mounts stay whole ConfigMaps for a later file watch.

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

During shutdown, readiness fails immediately while listeners stay open for
`min(router.shutdownGracePeriodSeconds / 2, 5 seconds)` by default.
Set `router.shutdownSettleSeconds` (`--shutdown-settle-secs`) to adjust this
propagation window; the effective window is the smaller of this value and half
the grace. For example, a 180-second grace and a 30-second window leaves
150 seconds for draining existing requests. Longer windows also delay idle
shutdowns and local Ctrl+C exits. A value of 0 skips the propagation wait
(mesh departure still shares the total grace budget).

#### Dashboard and Alerts

`grafana.dashboard.enabled` ships a Grafana dashboard as a ConfigMap (label `grafana_dashboard: "1"` for the sidecar loader) with `namespace` and `job` variables that pick the release (every query is scoped to both) and 49 panels in eight rows: traffic (request rate, responses by status, 5xx ratio, errors by API code and router error type, token throughput), latency (TTFT and time per output token p50/p95/p99 and per model, request and HTTP duration p99), capacity and queues (admission in-flight and queued, rejections, engine running and waiting requests, KV cache usage and in-flight request age per worker, connections), workers (healthy count, health and circuit-breaker state per worker, health checks, overload shed and fallback, retries), cache and KV events (engine prefix cache hit rate, cache-aware decisions and match ratio, KV event lag, integrity counters, degraded ranks, index size), Symphony (outputs by outcome, malformed outputs by reason, fallbacks and refusals), runtime (tokio busy ratio, event-loop stalls and delay, dropped log lines, memory, tasks, tokenizer cache) and prefill-decode (time to first token p99 by phase and by model, rendezvous, bootstrap and KV transfer failures, prefill admission in flight, queued, rejections, waits and sheds, the admission wait, the admission window per decode worker, KV connector mode decisions). Every panel queries a family the gateway describes; the gateway's test suite holds the dashboard to the catalog.

`prometheusRule.enabled` adds a `PrometheusRule` with eleven alerts, thresholds under `prometheusRule.alerts` (the defaults and their rationale are in `values.yaml`). Every expression selects this release's series through `prometheusRule.jobSelector` (empty = this release's namespace and the Service the ServiceMonitor scrapes, the Operator's default `namespace` and `job` labels, so a same-named release in another namespace stays out) and keeps `namespace` and `job` through its aggregation, so several releases on one Prometheus alert apart and a firing alert names its gateway:

| Alert | Severity | Fires when |
|---|---|---|
| `SmgGatewayDown` | critical | the router's scrape target is down, or no target matches the selector, for `scrapeMissingFor` |
| `SmgNoHealthyWorker` | critical | no registered worker is healthy, or no worker is registered at all while the gateway's scrape is up, for `noHealthyWorkerFor` |
| `SmgHighErrorRatio` | critical | the 5xx share of responses, the probe routes (`/health`, `/readiness`, `/liveness`) excluded, exceeds `errorRatio` for `errorRatioFor` |
| `SmgHighRouterErrorRatio` | critical | the router's server-side failed share of its requests (upstream failures and streams that failed after their first token, which the edge counts as 200s; the client's own refusals are not counted) exceeds `errorRatio` for `errorRatioFor` |
| `SmgTtftP99High` | critical | p99 time to first token of the gRPC router's streamed answers (the HTTP router records none; a collected answer, and a streamed Responses answer on the Harmony pipeline, records no first token) exceeds `ttftP99Seconds` for `ttftP99For`, evaluated per namespace and job |
| `SmgKvUsageNearCapacity` | warning | a worker's KV token usage exceeds `kvUsage` for `kvUsageFor` |
| `SmgCircuitBreakerOpen` | warning | a worker's circuit breaker is open or half-open for `circuitBreakerOpenFor` (a worker failing requests while passing its probes cycles between the two) |
| `SmgAdmissionRejections` | warning | the concurrency limit's queue or the priority scheduler's per-class queues reject requests for `admissionRejectionsFor` |
| `SmgEventLoopStalls` | warning | more than `eventLoopStalls` event-loop stalls in ten minutes |
| `SmgDroppedLogLines` | warning | a log sink dropped lines in the last ten minutes |
| `SmgKvEventDegradedRanks` | warning | an engine rank's KV index stays degraded for `degradedRanksFor` |

Readiness itself is a kubelet signal: with kube-state-metrics in the cluster, `kube_pod_status_ready` on the router pods complements `SmgGatewayDown`.

The rules have unit tests under `tests/rules`: `tests/rules/run.sh` (needs `helm` and `promtool`) renders the chart's rules and replays each test file's traffic shape against them. The `PR Helm Chart` workflow (`.github/workflows/pr-helm.yml`) lints the chart and runs them on every pull request that touches it.

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
- gRPC mode: `grpc` probe on the serving port, the same budget. The probe asks the servicer's `grpc.health.v1` service, which every servicer serves (the Rust servicer for every engine, the Python servicers alike) and which answers `SERVING` once the engine serves. The pod therefore counts as started when the engine does, not when the servicer binds its port. The built-in gRPC probe needs Kubernetes 1.24 or later; the chart requires 1.26.

Override per worker:

```yaml
workers:
  - name: large-model
    startupProbe:
      initialDelaySeconds: 180
      failureThreshold: 60
```

Size the budget past the longest load you expect: a startup probe that fails for its whole budget restarts the worker, and the load starts over. The worker Deployment's rollout deadline follows the budget (`initialDelaySeconds + failureThreshold * periodSeconds`, plus 600 s for what runs on the rollout's clock before the probe's starts: the node's provisioning, the image pull, the container start), so `kubectl rollout status`, Argo CD, Flux and `helm --wait` do not report a worker still loading within its budget as a failed rollout. On an autoscaled GPU pool or with a cold image cache, where the node and a multi-GB pull can take longer than that margin, set `progressDeadlineSeconds` on the worker entry past them; it overrides the default.

#### Worker Shutdown

A gRPC worker on the Rust servicer drains in-flight streams on SIGTERM for up to `SMG_*_SERVICER_DRAIN_SECS` (default 30 s; set it through `extraEnv`), then stops its server (5 s) and terminates the engine (on vLLM, vLLM's own process manager first gives the engine cores its `--shutdown-timeout`, 5 s when it is 0, the default; then up to 30 s): 70 s, past Kubernetes' default grace of 30 s. Each worker's pod therefore gets `terminationGracePeriodSeconds: 90` (a 20 s margin) unless `workers[].terminationGracePeriodSeconds` says otherwise; raise it with the drain bound and with a vLLM `--shutdown-timeout N` (N in place of the default 5 s).

#### Policy Controller

`policyController.enabled` deploys the standalone policy controller (`smg-policy-controller`) beside the router: a Deployment and one Service, `<fullname>-policy-controller`. The controller runs on its own today: it serves its probes and its metrics and runs the load path, with a spec document it runs that document, with the election on its replicas elect a leader and a standby can follow it (Standby, below), and nothing in the chart points the router at it.

The controller's image is the router's with `-policy-controller` appended to the tag, the tag a release publishes it under: one version names both images. Each part of `policyController.image` left empty falls back to the router's: the registry to `global.image.registry`, the repository to `global.image.repository`, the tag to the router's tag with the suffix. A router on its `-slurm` or `-ol9` variant takes the same controller image: the variant's suffix is cut first. A registry that holds the router's image alone, a mirror for one, needs the controller's image under the suffixed tag too: the pod fails at the pull without it. For an image of your own, build one from `docker/policy-controller.Dockerfile` (`docker build --file docker/policy-controller.Dockerfile --tag <repository>:<tag> .` at the repository's root), push it and name `policyController.image.repository` and `policyController.image.tag`: a repository of your own is refused without its tag, and so is a router image named by a digest. A router tag that names a release older than the first with a controller image is refused too, until `policyController.image.tag` names the controller's tag: no such release published a controller image. A `null` removes a value's default, so with the controller on the schema refuses it for every value whose default is not empty: the chart leaves none of them to the cluster's or the controller's own default. The members of `securityContext` are apart: a `null` drops one, by design. With the controller off the block is not read, so `helm upgrade --reuse-values` on a release from before the block renders as before. To turn the controller on in such an upgrade, use `--reset-then-reuse-values` (Helm 3.14 and later) or pass the chart's whole `policyController` block in a values file: reused values hold none of the block's defaults, and the schema refuses the upgrade, naming each value it lacks, until all are given. A release that adds a value whose default is not empty refuses an upgrade with reused values the same way.

| Value | Default | What it sets |
|-------|---------|--------------|
| `policyController.replicas` | `1` | Replicas. With the election off every replica is a controller of its own and two would each lead alone: the chart refuses more than one, and an upgrade tells the old pod to stop before it makes the new one. With it on, one leads and the others stand by: one replica has a rollout's surge pod for its standby, and two or more keep one between rollouts too. See Standby below. |
| `policyController.election.enabled`, `.leaseName` | `false`, `""` | Leader election among the replicas over one `coordination.k8s.io` Lease in the release's namespace: see below. `leaseName` `""` is `<fullname>-policy-controller`; a name of your own is unique to one controller. |
| `policyController.spec` | `{}` | The policy spec document the controller runs, as a map: required with the controller on unless `policyController.specStoreClaim` names a store, and an empty one is then refused at render. The chart renders it as a ConfigMap, mounts that in the pod and passes `--spec-file`: see below. |
| `policyController.specStoreClaim` | `""` | A spec store in place of the document: the name of a PersistentVolumeClaim of yours. The chart mounts the claim, passes `--spec-store` on it and renders no ConfigMap: see below. |
| `policyController.ports.gateway`, `.control`, `.replication`, `.admin` | `30100`, `30101`, `30102`, `30103` | The listeners' container ports, passed as `--gateway-listen`, `--control-listen`, `--replication-listen` and `--admin-listen`. |
| `policyController.identityAsTold` | `[]` | Nothing is verified on a listener named here. `--identity-as-told`: the listeners, any of `gateway`, `control` and `replication`, that take a caller's stated name as its identity: see below. |
| `policyController.service.type`, `.annotations` | `ClusterIP`, `{}` | The Service, with three ports on the listeners' own numbers: `grpc` (the gateways' listener), `control` (the operators') and `metrics` (the admin listener). The standbys' listener has no Service. |
| `policyController.logging.level`, `.json` | `info`, `false` | `--log-level` (`debug`, `info`, `warn` or `error`) and `--log-json`. |
| `policyController.runtimeWorkerThreads` | `2` | `--runtime-worker-threads`. The controller's own default is one worker thread for every CPU it may use, which with no CPU limit is every core of the node. Raise it with the CPU limit; `0` leaves the flag out, so the count follows the limit. |
| `policyController.shutdownGracePeriodSeconds` | `30` | `--shutdown-grace-period-secs`. The pod's `terminationGracePeriodSeconds` is this plus `policyController.terminationGraceMarginSeconds` (10). |
| `policyController.shutdownSettleSeconds` | `null` | `--shutdown-settle-secs`; `null` keeps the controller's default of 5 seconds. |
| `policyController.podDisruptionBudget.enabled`, `.minAvailable` | `false`, `1` | A PodDisruptionBudget over the controller's pods. With one replica, `minAvailable: 1` lets no pod be evicted and a node drain waits on it for ever, as with the router's budget: run two replicas or more, or leave the budget off. The budget counts ready pods: a standby is ready once it is synced where `identityAsTold`, or a flag of `extraArgs`, names `replication` and the listeners serve no TLS, and at once otherwise, where it holds nothing. |
| `policyController.livenessProbe`, `.readinessProbe` | `/liveness`, `/readiness` | `httpGet` probes on the admin port, with `initialDelaySeconds` and `periodSeconds`. |
| `policyController.securityContext` | the router's | The container's security context. With the controller on the schema refuses a `null` here: to run with none, set each member to `null`. Reused values hold no default under it: the map given there is the whole context, and an empty one runs the container with none. |
| `policyController.resources`, `.nodeSelector`, `.tolerations`, `.affinity`, `.podAnnotations`, `.podLabels`, `.extraArgs`, `.extraEnv` | | As the router's values of the same names. |

With `router.metrics.serviceMonitor.enabled`, a third ServiceMonitor scrapes the Service's `metrics` port under the same interval and labels. With more than one replica and no `policyController.affinity`, the chart prefers one replica a node. The chart refuses at render what the controller refuses at start: two listeners on one port, a settle window over half the grace period, and a listener named in `policyController.identityAsTold` that is none of the three. See [`with-policy-controller.yaml`](examples/with-policy-controller.yaml).

**Identities as told.** Nothing is verified on a listener named in `policyController.identityAsTold`: whoever reaches its port acts as any principal of its scopes. None is named by default, and a listener that is not named refuses every call of its scopes but `Leader` as unauthenticated. The Service carries the gateways' and the operators' ports and a pod's address reaches all three listeners, so the standbys' listener is not out of reach for having no Service: name a listener only behind a mesh that verifies its callers, or a network policy that admits them alone. The value is a set, passed in the listeners' own order. An `--identity-as-told` flag in `policyController.extraArgs` adds to it: the ingestion listener, which the chart does not render, is named where it is turned on, and a listener named in both places is named once more, which the controller takes.

**Spec document.** `policyController.spec` holds the policy spec document (`apiVersion: smg.policy/v1`) as a map. With the controller on it is required, unless a spec store's claim is named (see below): the controller refuses a start that names no source of its spec, and the chart refuses an empty document at render, by name. The chart writes it as the one key `spec.yaml` of a ConfigMap, `<fullname>-policy-controller`, mounts that ConfigMap whole and read-only at `/etc/smg/policy-spec` and passes `--spec-file /etc/smg/policy-spec/spec.yaml`: the controller reads the file at its start and watches it from then on. The chart reads no line of the document and writes none: whether a document runs is the controller's word, and a `generation` line is the document's own. A document kept in a file of its own is handed through the values: a values file that holds it under `policyController.spec`, passed with `-f`. Helm reads the document before the chart does, by its own rules. It joins maps key by key, so a document given over reused values, or in a second values file, keeps every key the new one leaves out: give the document whole, in one place. It reads a plain `yes`, `no`, `on` or `off` as a boolean, as a key too, and `010` as 8: quote what is text. It reads a number of a values file as a 64-bit float: a whole number above 2^53 reaches the file rounded.

The policy controller crate's [README](../../../crates/policy_controller/README.md) explains the document's format.

A changed document moves no pod: the pod template holds no checksum of it. The kubelet turns the mounted file at its next sync, and the running controller applies the document in place as its next generation, or rejects one that does not run and keeps the generation it runs. A rejected document is one warning in the pod's log, `policy spec rejected: the running generation stays`, and one count of `smg_policy_spec_apply_total{result="rejected"}`. `smg_policy_spec_generation` is the generation that runs, and `GetPolicySpec` on the operators' listener answers its document with its number and its digest, to a caller that listener admits (see above). A pod that starts on a document that does not run refuses to start. A rejected document stays mounted. One that does not run refuses the pod's next start for any reason (a crash, an eviction, a drain) until the values hold a document that runs: put one back, or roll the release back. One refused only for its `generation` line (a change under the running line, or a line at or under it, as a rollback gives) runs at the pod's next start: a document that writes a line raises it at every change, a rollback's too, or writes none. With the election off that pod is the one controller, and nothing decides until then. An upgrade that moves the pods for another reason starts each new pod on the new document. Dry-run a document before such an upgrade: `ApplyPolicySpec` in its dry-run mode, on the leader's operators' listener, answers whether the document would run and applies nothing. With the election on the leader's mount decides, and a standby applies nothing from its own mount until it leads. A standby that follows its leader (`identityAsTold` names `replication`) runs the leader's document under the leader's number. A standby that follows nobody runs no generation until it leads: its `smg_policy_spec_generation` reads `0` and `GetPolicySpec` there says no document is held, so read the gauge on the leader. A pod that takes the lead applies its own mount's document as the next generation, unless that document ran on it since its mount last turned. So on a standby that follows its leader, a mount the kubelet has not turned yet brings no older document back. A standby that follows nobody ran none, and applies its mount's document as it stands. A standby says nothing of a mounted document that does not run: its silence is no verdict, and the pod tells it once it leads. Until the kubelets have turned both mounts, a hand-over inside that gap leads under the document the taker followed, or under the taker's copy where that is new to it. The digest says which.

**Spec store.** `policyController.specStoreClaim` names a PersistentVolumeClaim of yours in the release's namespace, and the controller then runs what an operator applies and no document of the values. The chart renders no ConfigMap, mounts the claim at `/var/lib/smg/spec` and passes `--spec-store /var/lib/smg/spec`. `ApplyPolicySpec` in its enforce mode, on the operators' listener, writes the document to the store before it runs, and `RollbackPolicySpec` runs one of the last eight generations again under a new number. A pod that starts runs the newest generation on file. A store with nothing on file starts ready and sheds every decision until its first apply: `smg_policy_spec_generation` reads `0` until then. The chart ships no client of the two calls: bring one of the controller's contract (`crates/policy_controller/contracts/policy_controller.proto`). A listener admits no caller until `policyController.identityAsTold` names it (see above), so name the operators' listener there, behind a mesh or a network policy, before the first apply.

The claim is yours and the chart renders none: a claim a release owns goes with the release, and the generations on file are the only copy of an applied document. Make it `ReadWriteOncePod` where the cluster offers that mode, else `ReadWriteOnce`, on a file system that links a file under a second name and locks one, as the store does. It holds eight documents. The pod's `fsGroup` is the container's `runAsUser`, so the user the image runs as writes the volume. With `runAsUser` set to `null` no group is named, and the volume must then be writable by the user the container runs as. One pod at a time holds the store, under a lock on a file the controller keeps there (`.lock`: leave it in place). A pod that cannot write the store and finds no lock file there exits at its start, where it would make one. One that finds the file, as on a store another user wrote, starts and runs the newest generation on file: each apply is then refused (`not persisted`, with the system's reason) and that generation runs on. Such a start is refused instead (`not locked`) where the file system gives no lock on a file opened to read, as a network file system may not, or where the pod cannot read the lock file. A temporary name a cut-off write left, which such a pod cannot remove, stays in place. A pod that stops still commits the applies it took, and a pod reads the store once, at its start: a pod started beside one that stops waits for the lock first, for its grace period at most (`policyController.shutdownGracePeriodSeconds`), and so runs the old pod's last apply. One still held at the end of that wait exits, and the kubelet starts it again. A pod that waits answers no probe: at the default probe timings, with a grace period past about a minute the liveness probe restarts it before that, and it waits again. An upgrade starts no pod beside another: on a store the Deployment recreates its pod, the old one gone before the new one is made. A pod replaced otherwise, after an eviction, a `kubectl delete pod` or a rollback begun while the old pod still stops, starts beside it and waits as said. Where the claim keeps the two on one node the lock holds; a claim several nodes mount keeps them apart only where its file system carries locks between nodes. The chart refuses at render what the controller refuses at start beside a store: a document in `policyController.spec`, since a start names one source of its spec, and the election, since a standby does not follow an applied generation yet. More than one replica is refused by the chart's own rule: each would lead alone. A release that moves from the document to a store sets `policyController.spec` to `null` and upgrades with `--reset-then-reuse-values` (Helm 3.14 or later) or `--reset-values`, which drops the release's other overrides, so they are given again: an empty map does not clear a reused document, and under `--reuse-values` a `null` does not either, since the old release's values stand in for the chart's own and hand the document back. On a new claim its pod starts on an empty store: apply the document again before the gateways depend on it, with no `generation` line or as generation 1, since a store takes a written line only where it is the next number.

**Election.** With `policyController.election.enabled` the replicas elect one leader over a Lease, and the chart renders four more objects: the Lease, a ServiceAccount of the controller's own, and a Role and RoleBinding that grant that account `get` and `update` on that one Lease and nothing else. The account's token is mounted in the controller's pods alone. The controller is passed `--lease-name`, and its pod's address as `--advertise-address`: the Lease's holder is named by an address the other replicas and the gateways can dial. The timings are the controller's own (a lease of 3 seconds, renewed every second); `policyController.extraArgs` carries `--lease-duration-ms`, `--lease-renew-period-ms` and `--lease-renew-deadline-ms` for others, and a Lease's duration is whole seconds. With the election off the pod mounts no token: the controller then calls no Kubernetes API.

The rollout follows the election and the spec's source. On, the Deployment rolls one pod more at a time with none unavailable: the new pod stands by, and the old leader's stop hands the Lease over. A stop whose release fails tries it twice more, a renew period apart: two seconds longer at the controller's own timings, inside the default grace period. If none lands the Lease runs out by itself. Off, every pod leads alone. On the document the Deployment then rolls with no pod more at a time (`maxSurge: 0`, `maxUnavailable: 1`): an upgrade tells the old pod to stop before it makes the new one. The old one decides nothing from that word on and may still be closing its streams when the new one starts, for its grace period at most; the controller is away between the two. A pod that does not obey decides until it is killed at the end of that period. On a spec store the Deployment recreates its pod (`Recreate`): the new pod is made once the old one is gone, so no controller decides from the old pod's stop signal until the new pod is ready. That is the old pod's whole stop at least: the settle window, 5 seconds by default, which a lower `policyController.shutdownSettleSeconds` shortens. On the document the strategy's type is `RollingUpdate` with the election on and off, so no upgrade between those changes it: the API refuses the rolling update's fields beside `Recreate`, and a server-side apply takes off a live object only what it put there, which leaves the server's own on a Deployment that named no strategy. A move from the document to a store, or back, changes the type: the store's form names the rolling update's fields as nothing, and the document's form names them itself, so an upgrade from it takes them off the live object. Turning the election on overlaps the two kinds: an elected pod starts while the lone one still leads. Scale the controller to zero first (`policyController.enabled=false`, or the Deployment by hand), wait until no controller pod remains, then turn it on. Turning it off tells the elected pods to stop before it makes the lone one. A changed `leaseName` on a running release is such a change as well: a pod elected over the new Lease may lead while the old Lease's leader still leads, until that leader's renew deadline, and the term starts again under the new name. Scale to zero before it too. The elected pod template names its account by `serviceAccountName` and by that field's deprecated alias `serviceAccount`: turning the election off, or a rollback past it, removes both, and new pods run under the namespace's default account.

The Lease is the release's object and its record is the cluster's:

- The chart renders the Lease with no `spec`: the holder, the times and the term are written by the controllers, and an upgrade has nothing to say of them. An apply that replaces objects wipes the record: the old leader fences itself at its next renewal, inside the renew deadline, and the replicas elect anew above the term they remember.
- The controller is granted no `create` (a create cannot be held to one name), so a Lease deleted by hand leaves no leader: every replica's election step is refused and none leads, the old leader fencing itself inside the renew deadline. Replicas that were ready stay ready, as followers that name no leader; one started after stays not ready. Replicas that follow their leader (Standby, below) turn not ready instead, which is the true state: the old leader within a renew period of the delete, logging `the lease moved; the fence is closed`, and a standby one load poll interval after its leader's last frame, 10 seconds by default (`--load-monitor-interval`, through `policyController.extraArgs`). The Service then has no ready endpoint, each pod answering 503 with the reason `syncing`. The pods cannot make the Lease themselves: apply the release again with its values (`helm upgrade`), which creates the Lease it finds missing. At the controller's default timings, about three seconds after the Lease stands one leads and the others follow, with no restart.
- A delivery tool that excludes Leases, as Argo CD's default `resource.exclusions` do since its version 3, never creates this one: every replica then stays not ready and nobody leads until the kind is taken out of the exclusion.
- The Lease carries the election's term. A deleted or replaced Lease restarts the term once no running controller remembers the last one, and turning the election off deletes the Lease. While a replica remembers a higher term than the Lease holds, a holder that never saw that term leads all the same: the replica names no leader, and takes the Lease above the term it remembers once it is free.
- Two controllers handed one `leaseName` in one namespace elect one leader between two stores: the chart cannot see it.

Two choices are deliberate. The Role holds `get` and `update` only: the controller calls no `create` and no `watch` with the chart's Lease in place. The Lease's name carries no revision: a name that moved at an upgrade would elect a second leader beside the first.

**Standby.** A replica that does not lead follows its leader only where the standbys' listener takes identities as told: `policyController.identityAsTold` names `replication`. It then dials the Lease holder's pod at `policyController.ports.replication`, takes the leader's workers and their loads and follows each write of them: at a take-over it holds them as the leader did, up to its lag. `/readiness` answers 503 with the reason `syncing` until the standby is synced with its leader, and 200 as a `follower` only while it is. A rollout therefore waits for the new pod's sync before it stops the old one, and a disruption budget counts a synced standby alone. `/metrics` carries `smg_policy_controller_replication_synced` (1 on a synced standby, 0 on any other replica), `smg_policy_controller_replication_lag_seconds` and `smg_policy_controller_replication_restores_asked_total`. Under the defaults none of this holds: every replica logs `this replica follows no leader`, a standby reads ready at once, and a take-over starts with no worker and no load. A standbys' listener that serves TLS follows nobody yet. With one replica a rollout's surge pod is the standby; two or more keep one between rollouts, for a pod that is lost and for a node drain, where no surge comes.

The standbys' listener has no Service: a standby dials the leader's pod. Its callers are the controller's own pods and nobody else, so what fences it (a mesh that verifies its callers, or a network policy) admits to the replication port the pods under the controller's selector labels: `app.kubernetes.io/name`, `app.kubernetes.io/instance` and `app.kubernetes.io/component: policy-controller`. A fence that keeps them out leaves every standby not ready, and a rollout waiting on it. The chart renders no such policy: one that selects the controller's pods isolates their other listeners too, and the chart cannot know what the cluster admits to those. An install with the election on says in its notes where a standby follows nobody: `replication` not named, or the listeners serving TLS. See [`with-policy-controller-standby.yaml`](examples/with-policy-controller-standby.yaml).

Naming `replication` on a running elected release whose listeners serve no TLS does not roll out. The old leader's listener takes no stated name and refuses the new pod, which stays not ready: `/readiness` answers 503 with the reason `syncing`, and the pod logs `this standby does not follow its leader` with the leader's refusal (`Unauthenticated`). The rollout waits until its progress deadline, an upgrade that waits for it times out, and nothing is down: the old pods serve as before. Two ways lead out: roll back, or scale the controller to zero (`policyController.enabled=false`, or the Deployment by hand), wait until no controller pod remains, and upgrade with the controller on again: reused values keep `enabled=false`. Deleting a pod is not one: the old ReplicaSet makes that pod again, and it or another old pod may take the Lease. Either way the first leader under the new values starts with no worker and no load, for no pod could follow the old one. `enabled=false` takes the Lease away too, so the election's term starts again, where by hand the Lease stays. Behind a server certificate none of this arises: the new pod follows nobody, reads ready at once and rolls out.

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
| [`with-policy-controller.yaml`](examples/with-policy-controller.yaml) | The standalone policy controller beside the router |
| [`with-policy-controller-standby.yaml`](examples/with-policy-controller-standby.yaml) | The policy controller elected, its standby following the leader |

An example that pins `global.image.tag` at a release with no controller image is refused with `policyController.enabled` until `policyController.image.tag` names one.

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
