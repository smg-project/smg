# Prefill cost routing

`prefill_cost` routes requests by estimated input processing time. It accounts
for the prefix each worker has cached and the work already assigned to it, so
a busy worker can give up a cache hit when another worker can process the
whole prompt sooner.

## Usage

Start a regular native gRPC router:

```sh
smg --policy prefill_cost \
  --worker-urls grpc://worker1:8080 grpc://worker2:8080 \
  --block-size 16 \
  --prefill-cost-default-throughput 2000 \
  --prefill-cost-mean-prefill-tokens 1024 \
  --prefill-cost-kv-pressure-weight 0.15
```

The Python launcher accepts the same options. In a configuration file:

```yaml
policy:
  type: prefill_cost
  block_size: 16
  default_throughput: 2000.0
  mean_prefill_tokens: 1024
  kv_pressure_weight: 0.15
```

Select the policy in the router configuration. A worker's
`policy=prefill_cost` label inherits that configuration when the router's
default is also `prefill_cost`. With another default, the label follows the
unknown-policy fallback and leaves that default in place.

Use a regular worker pool with IGW disabled. HTTP and ZMQ workers and PD/EPD
pools are unsupported.

## Choosing a worker

The policy considers every eligible worker. Native KV events tell it how much
of the prompt each worker holds. Matching uses the request's cache namespace
and the engine's block size, with `--block-size` as the fallback. Until an
index is available, the whole prompt counts as input work.

For example, two 16k prompts sharing a cached 4k prefix each need 12k of new
prefill on the worker holding that prefix. If the first is still prefilling
when the second arrives, the second would add to that worker's outstanding
work. An idle peer can take it at the cost of recomputing the 4k prefix.
With a 64-token suffix behind a large cached prefix, reusing the cache costs
far less work than sending the whole prompt to a cold peer.

The score divides estimated input work by the worker's input-token rate and
adds a KV-pressure penalty. Input work includes the incoming uncached prompt,
the engine's queue and an allowance for running requests. A queue report
without token counts uses `waiting_requests * mean_prefill_tokens`. Running
requests use the same allowance, including requests already decoding; the
engine does not report their remaining prefill work.

The rate comes from valid prefill-throughput reports. A missing or invalid
rate uses the mean reported input rate of the eligible workers, or
`default_throughput` when none has one. All these rates are input tokens per
second. KV usage contributes `kv_pressure_weight * k / (1 - k)` seconds,
clamped near a full cache.

## Outstanding requests

SMG records a request's estimated work as part of selecting its worker. The
next selection in that process sees the assignment even before dispatch.
Model policies and aliases share these counters for each worker.

The first output token or a terminal response clears the prefill estimate;
empty response frames keep it. The running-request allowance stays until
the request ends, including fully cached requests. A refused or cancelled
attempt gives its work back, as does a failed dispatch. A retry releases the
old attempt before selecting again.

A batch counts every prompt, with cache credit only for the first. Its
prefill estimate stays until every input produces a token or ends, and its
running allowance stays until the last input ends. Partly finished batches
can therefore count more work than remains. An embedding batch continues to
count in full even after an engine load report includes it, until the batch
completes.

Sticky assignments keep their worker. When a sticky assignment or overload
fallback bypasses normal selection, the final worker is charged the full
prompt and any provisional assignment is released.

Backend load reports can include requests SMG has already counted. The score
takes the larger of the reported work and older dispatched work, then adds
work not yet dispatched and work sent after the report. A report can arrive
before a dispatched request reaches the engine, so this can undercount work
in transit. Stale reports retain their queue estimate until a new one arrives.
The counters belong to one gateway process; other gateway replicas see its
requests through engine load reports.

## Tuning

| Option | Default | Meaning |
|---|---|---|
| `--prefill-cost-default-throughput` | `2000` | Input tokens/s when no eligible worker has a valid input-rate report. Set this to the engines' measured input throughput. |
| `--prefill-cost-mean-prefill-tokens` | `1024` | Work allowance for each running request, and for each queued request whose token count is unknown. |
| `--prefill-cost-kv-pressure-weight` | `0.15` | Weight of the KV-pressure penalty, in seconds. Increasing it moves requests away from workers with full caches sooner. |
| `--block-size` | `16` | KV block size used before the engine reports its own. |
