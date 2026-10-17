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
| Gateway | `docker.io` | `lightseekorg/smg:1.3.3` |
| vLLM worker | `ghcr.io` | `lightseekorg/smg:1.3.3-vllm-v0.18.0` |
| TRT-LLM worker | `ghcr.io` | `lightseekorg/smg:1.3.3-trtllm-1.3.0rc8` |
| SGLang worker | `ghcr.io` | `lightseekorg/smg:1.3.3-sglang-v0.5.9` |

Worker images default to `ghcr.io`. Set `workers[].image.tag` to the engine-specific tag.

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
| `rbac` | RBAC for Kubernetes service discovery |

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
      tag: "1.3.3-vllm-v0.18.0"      # engine image tag
```

Workers are automatically wired to the router via `--worker-urls`. The URL scheme (`http://` or `grpc://`) is set based on `connectionMode`.

#### Connection Modes

| Mode | Description | Worker Command |
|------|-------------|----------------|
| `http` | OpenAI-compatible HTTP API | vLLM: `vllm.entrypoints.openai.api_server` |
| `grpc` | SMG gRPC pipeline | vLLM: `vllm.entrypoints.grpc_server` |

SGLang uses `sglang.launch_server` for both modes (adds `--grpc-mode` for gRPC).

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

The router's liveness probe is `httpGet /health`, which answers 200 while the process serves. Its readiness probe is `httpGet /readiness`, which answers 503 while the gateway has no healthy worker, while a tokenizer is not yet registered and while the gateway drains after SIGTERM, so the Service stops sending new connections to a pod that cannot take them (`router.livenessProbe.path` and `router.readinessProbe.path` override the routes).

A router that gets its workers at runtime through the admin API (`POST /workers`) rather than from `workers`, `router.workerUrls` or `router.serviceDiscovery` has no healthy worker when it starts, so `/readiness` answers 503 until one is registered, the Service has no endpoint for the pod, and a registration sent through the Service cannot reach it (a rolling update's new pod then never becomes Ready). Such deployments set `router.readinessProbe.path: /health`, or register their workers through a port-forward to the pod.

The gateway drains in-flight requests for `router.shutdownGracePeriodSeconds` (default 180, passed as `--shutdown-grace-period-secs`); the pod's `terminationGracePeriodSeconds` is that plus `router.terminationGraceMarginSeconds` (default 30), so the kubelet does not kill the drain at its default 30 s.

#### Startup Probes

Model loading takes 60-300 seconds. Workers have startup probes with generous defaults:

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
| [`with-ingress.yaml`](examples/with-ingress.yaml) | Ingress with TLS |
| [`with-monitoring.yaml`](examples/with-monitoring.yaml) | ServiceMonitor + Grafana dashboard |

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
