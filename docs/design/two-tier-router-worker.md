# Two-Tier SMG Router/Worker Architecture

Status: prototype on `feat/two-tier-smg-worker-zmq`

Scope: text generation only. LoRA management is excluded and will be handled
in a separate PR.

## Decision

Split SMG into a fleet-level Router and a node-local Worker. Keep the stable
cross-node boundary engine-neutral Rust gRPC, and let each Worker use the
colocated engine's native transport.

```mermaid
flowchart LR
    C[Client] --> R[SMG Router]
    R -->|WorkerControl + WorkerInference\nRust gRPC| W[SMG Worker]
    W -->|ZMQ IPC| V[vLLM]
    W -->|ZMQ IPC| T[TokenSpeed]
    W -.->|native gRPC loopback\n(adapter only, not launchable)| S[SGLang]
```

Dashed edges are implemented in the Worker but not reachable through
`smg serve` yet; see [Engine transports](#engine-transports).

The Router owns public APIs, authentication, tokenization, fleet membership,
admission, and request placement. The Worker owns engine readiness, topology,
bounded admission, draining, cancellation, and engine-specific translation.
The Router never connects directly to a node-local engine endpoint.

## Contracts

`WorkerControl` provides identity, capabilities, health, and topology.
`WorkerInference` provides engine-neutral tokenized generation, streaming, and
abort. Engine-specific protobufs and scheduler details remain behind the
Worker boundary.

Workers advertise `token_only_wire` when the engine-facing transport cannot
match string stops (currently ZMQ). The Router then retains string-stop
trimming and supplies tokenizer EOS ids as a vLLM EngineCore backstop across
the otherwise engine-neutral Worker hop. The primary EOS id travels with each
request (resolved from that request's tokenizer), never cached per Worker
connection, because one Worker can serve several models. A Worker must name
its engine transport in its topology attributes; the Router refuses to
register one that does not, rather than guess that string stops reach the
engine.

Router health probes map Worker states rather than collapsing them: DEGRADED
keeps the worker in rotation with a warning, DRAINING removes it from rotation
on the first probe (new work is already refused there), and STARTING or
NOT_SERVING count as failed probes.

The first version deliberately excludes embeddings, multimodal tensors, and
disaggregated execution until they have explicit contracts.

## Python binding

Python coordinators use a small PyO3 binding to start the Rust tonic Worker and
announce lifecycle transitions after scheduler processes fork. SGLang, vLLM,
and TokenSpeed use the same constructor arguments; Python is not called per
request.

The binding binds the control listener first and connects the engine
transport in the background, so STARTING is observable over gRPC for the
whole handshake (a model load, over ZMQ). Rust owns engine readiness: whatever
lifecycle Python announces, `GetHealth` reports SERVING only once the
transport is connected (an engine health probe over gRPC, the HELLO/INIT/READY
handshake over ZMQ), STARTING while it is connecting, and NOT_SERVING with the
error if it fails. The same state is served as standard `grpc.health.v1`, so
`smg serve`, Kubernetes probes, and `grpcurl` can wait for readiness without
speaking WorkerControl. Requests that arrive before the transport is up are
refused with UNAVAILABLE.

The hot path is:

`Router tonic client -> Worker Rust adapter -> engine-native transport`

This binding is intentionally invasive only at coordinator startup and
shutdown. Streaming, backpressure, cancellation, and protocol conversion stay
outside the GIL.

## Engine transports

- vLLM and TokenSpeed: same-host msgpack over ZMQ IPC, including the native
  HELLO/INIT/READY handshake and explicit abort.
- SGLang: a native Rust gRPC loopback adapter exists in the Worker, but no
  launch path reaches it yet, so `--router-worker-mode smg --backend sglang` is
  rejected at startup for every transport. `sglang.launch_server --grpc-mode`
  serves `sglang.grpc.scheduler.SglangScheduler` (what the Router's existing
  direct client speaks), while the Worker adapter dials
  `sglang.runtime.v1.SglangService`; since connecting only opens a channel, a
  mismatched pair would report SERVING and then fail every request with
  `Unimplemented`. Wiring the launch side up -- or direct scheduler IPC -- can
  land later without changing the Router/Worker contract.
- `grpc` remains a compatibility option for existing engine deployments.

## Implementation

- Versioned `WorkerControl` and `WorkerInference` protobufs and Rust clients.
- Rust Worker server with bounded admission, lifecycle states, streaming, and
  drop-triggered abort.
- Engine-neutral `EngineTransport` abstraction with native gRPC and ZMQ
  implementations.
- vLLM and TokenSpeed ZMQ adapters reuse the existing production ZMQ client.
- PyO3 lifecycle binding and standalone Worker sidecar accept the same engine
  transport arguments.
- `smg serve --router-worker-mode smg --connection-mode zmq` launches a
  colocated vLLM or TokenSpeed engine, Worker sidecar, and Router.
- SGLang is rejected for both `grpc` and `zmq` instead of silently selecting
  the wrong wire.

## Validation

Local checks pass for Rust build/clippy, the real IPC mapping-and-abort unit
test, command construction, lifecycle configuration, and Python lint/tests.

B200 GPU E2E passed with `Qwen/Qwen3-1.7B` using the latest official OSS
TokenSpeed runner image:

- image: `lightseekorg/tokenspeed-runner:latest`
- image digest: `sha256:d6067daeeb1fafecc531d45e282797076e1cd2e2c16eaa90712634dd76a709ca`
- TokenSpeed source: `c3ea3cd883048e4a4a444ec0481d270b19f0103d`
- verified path: Router HTTP -> Rust Worker gRPC -> msgpack ZMQ IPC -> TokenSpeed
- passed health, model discovery, non-streaming generation, SSE streaming, and
  client-disconnect cancellation; the Worker remained healthy afterward.

The same two-tier ZMQ path passed on B200 with the latest official OSS vLLM
image:

- image: `vllm/vllm-openai:latest` (`vLLM 0.28.0`)
- image digest: `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`
- verified path: Router HTTP -> Rust Worker gRPC -> msgpack ZMQ IPC -> vLLM
- passed health, model discovery, non-streaming generation, SSE through
  `[DONE]`, client-disconnect cancellation, and draining; while the Worker was
  draining, new generation returned 503 and Router liveness remained 200.

An earlier revision of this branch also passed SGLang non-streaming on B200,
using a `--grpc-port` launch that has since been reverted; that result does not
describe the current code, which rejects SGLang two-tier workers at startup.
Streaming did not pass in that same run (measured 2026-08-31, stock SGLang
0.5.18): the failure was in the engine's own streaming path, not in the
Router/Worker contract, and it was not isolated to a specific upstream issue
before the launch change was reverted. Treat it as untested rather than as a
known-broken combination, and re-measure against the SGLang version in use when
the launch path is wired up.

## Performance result

The [B200 TokenSpeed transport benchmark](../benchmarks/two-tier-transport-tokenspeed-b200-2026-08-31.md)
compared three 30-second runs at 1, 8, and 16 RPS. ZMQ reduced the Router +
Worker + engine-frontend CPU path by 41–43% and total service CPU by 5–12%,
with no throughput difference in the tested range. It did not improve TTFT:
at 8–16 RPS, p50 was 3–4 ms slower and p99 about 8 ms slower. E2E p50 and ITL
p99 generally improved, while the high-load E2E tail was mixed.

The result supports ZMQ as the colocated CPU-efficiency boundary, not as a
latency or peak-throughput claim. Trace and optimize its first-frame wakeup
path before claiming a TTFT improvement.

## Upstream alignment

- [SGLang native Rust gRPC RFC](https://github.com/sgl-project/sglang/issues/22558)
- [vLLM Rust frontend roadmap](https://github.com/vllm-project/vllm/issues/44280)
- [vLLM Rust frontend RFC](https://github.com/vllm-project/vllm/issues/40846)
