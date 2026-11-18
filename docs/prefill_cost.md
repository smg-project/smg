# Prefill cost routing

`prefill_cost` is an opt-in policy for regular native gRPC workers. The default
`cache_aware` policy is unchanged. IGW, HTTP, ZMQ and PD/EPD configurations are rejected
for this policy because those paths do not transfer its per-attempt lease.

```sh
smg --policy prefill_cost \
  --worker-urls grpc://worker1:8080 grpc://worker2:8080 \
  --block-size 16 \
  --prefill-cost-default-throughput 2000 \
  --prefill-cost-mean-prefill-tokens 1024 \
  --prefill-cost-kv-pressure-weight 0.15
```

The same policy and tunables are available through the Python launcher. Its
configuration form is:

```yaml
policy:
  type: prefill_cost
  block_size: 16
  default_throughput: 2000.0
  mean_prefill_tokens: 1024
  kv_pressure_weight: 0.15
```

Select it in the gateway configuration. A worker's `policy=prefill_cost` label
only inherits an already configured `prefill_cost` default; it cannot enable this
policy on a gateway using another default and bypass transport validation.

The throughput fallback is input tokens per second. Calibrate it to the serving
engines when measured input rates are unavailable. Generation throughput is never
substituted for input throughput.

## Decision and ownership

Every eligible worker, including workers with no cache hit, competes on estimated
input drain time plus a convex KV-pressure penalty. Native KV events give the
first prompt's reusable prefix. With no index, the cache credit is zero; the
policy still balances pending work. It does not build an approximate placement
tree or copy KV tensors between workers.

The input-work estimate includes backend queue work, a configurable running-work
estimate, local reservations and the incoming uncached prompt. Each live local
request also keeps a `mean_prefill_tokens` running-work estimate until completion,
including fully cached requests and requests already decoding. This prevents
zero-prefill reservations from disappearing behind stale idle reports. A reported queue
without token counts uses `waiting_requests * mean_prefill_tokens`. Valid input
rates normalize heterogeneous capacity; a missing rate uses nominal measured peer
capacity or the configured fallback. KV pressure contributes
`kv_pressure_weight * k / (1 - k)` seconds, with the existing bound near full KV.

One registry-owned ledger is shared across model policies and aliases. Selection
and reservation run under its lock, so the next local decision sees booked work
even before the first request reaches the engine. A request/attempt owns its
lease through selection, dispatch and the raw response stream:

- Refusal, dispatch failure and cancellation release the lease.
- Empty chunk/control frames keep the prefill reservation.
- The first output-token frame or a terminal response releases prefill work.
- Error, EOF or stream drop releases the remaining attempt owner.
- Retry starts a fresh reservation and gives back the previous attempt first.
- A batch prices the sum of all prompts, with cache credit only for the first
  affinity prompt; the other prompts are conservatively charged in full. Its
  prefill reservation remains until every sibling starts producing output or ends.
  The running-work estimate is scaled by the number of input items and held until
  the last sibling ends, so a partially finished batch can be overestimated.

Sticky overrides and the existing all-vetoed fallback retain their target. If
they bypass or replace a provisional choice, the actual worker receives a full
prompt reservation and the displaced reservation is rolled back.

## Limits and local validation

This is an estimate of work, not an exact TTFT predictor. Running requests include
decode, and the backend does not expose exact remaining running-prefill work.
Older sent reservations may overlap a load report and use a `max` approximation
to avoid double counting; it can undercount disjoint remote or in-transit work.
Unsent work and work sent after the sample are added separately. A timestamp does
not prove engine acceptance. A stale report can retain work until the next
refresh. Reservations are local to one gateway process; separate
gateway replicas do not share an atomic admission transaction.
Embedding calls retain their reservation until the unary batch completes and
conservatively keep it additive to backend reports throughout that batch.

The paired routing tests give both policies the same native index and backend
load feed. For two distinct 16k prompts with a shared 4k cached prefix, retained
reservations spread the new policy's second request to the cold worker while
`cache_aware` retains both at the holder. For two 64-token suffixes behind a large
warm prefix, both policies preserve the holder. Other cases cover queue and rate
fallback, namespace isolation, batch work, unavailable workers and concurrent
choices. Raw gRPC stream tests cover first-token, empty frames, replay,
disconnect, failure, EOF and batch sibling cleanup.

These CPU and protocol checks do not measure GPU latency or throughput. A serving
benchmark must hold engine settings, request arrivals, token IDs and actual cache
state fixed before drawing a TTFT or capacity conclusion.
