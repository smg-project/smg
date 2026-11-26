# The policy spec

A policy spec tells Grand Central how to choose a worker for each request. It is one document, in YAML or JSON. This page lists its fields, with their defaults and limits.

The [README](../README.md) is the first page to read. [REFERENCE.md](../REFERENCE.md) holds the rest of the detail, and calls the policy spec "the policy document".

## A small spec

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy]
    score: [least-load]
    pick: best
```

This spec has one *profile*, named `default`. A profile is one way to choose a worker: some filters, some scorers and one picker. The spec has no bindings, so its one profile serves every request that takes one worker.

These rules hold for the whole spec:

- `apiVersion` must be `smg.policy/v1`, and the spec needs at least one profile.
- YAML and JSON use the same types.
- In YAML, quote a string that looks like a number or a boolean.
- A key without a value is null. Write `[]` or `{}` for an empty list or map.
- The [JSON schema](../policyspec.schema.json) includes reserved fields and the stock stage names.
- Validate a spec in the host that will run it. That check says which fields and stages can run there.

## Top-level fields

| Field | What it holds | Default |
|---|---|---|
| `apiVersion` | The format's version. Only `smg.policy/v1` is accepted. | Required |
| `generation` | The spec's number, an unsigned 64-bit integer. | `0` |
| `profiles` | The profiles by name, from 1 to 65535. | Required |
| `view` | What a decision may read. See [The view](#the-view). | See there |
| `index` | Prefix index settings by model. See [Index](#index). | Empty |
| `signals` | Named values of a worker's report. See [Signals](#signals). | Empty |
| `plans.pd-pair` | The rules of a pair. See [Prefill and decode](#prefill-and-decode). | See there |
| `bindings` | The bindings, in order. See [Bindings](#bindings). | Empty |
| `explain.sample` | Each deciding thread keeps every Nth pick or shed it counts, for explain. N is 0 to 4294967295. | `0`: keeps none |

A `generation` of zero asks the service to number the spec. See [Updating a policy spec](../REFERENCE.md#updating-a-document).

In a host's own process the engine counts a request only where an id names it. See [Explain and metrics](../REFERENCE.md#explain-and-metrics).

## Profiles

A *stage* is one filter, one scorer or one picker. A profile names its stages in these fields:

| Field | What it holds | Default |
|---|---|---|
| `filter` | The filters, in order. | Empty |
| `score` | The scorers. | Empty |
| `pick` | The one picker. | Required |
| `sent_ttl_ms` | How many milliseconds a send counts for. Above zero. | Unset |
| `sent_report_delay_ms` | The least milliseconds between a send and the report that ends it. Above zero. | Unset |

A stage is written as a bare name, or as a mapping of one name to its settings:

```text
score:
  - least-load: { weight: 3 }
  - least-requests
```

[Built-in stages](../REFERENCE.md#built-in-stages) lists every stage with its settings.

### The tally of sends

A profile can keep a *tally of sends*: what the host sent each worker since that worker's last load report. A stage reads the tally through `StageBuild::dispatch_credit`.

A send leaves the tally at the first of these events:

- A load report of the worker, sampled at or after the send. Where the profile sets `sent_report_delay_ms`, the sample must come at least that long after the send.
- The worker leaves, joins again, is replaced or changes its number of ranks.
- The end of `sent_ttl_ms`, where the profile sets it.

The tally keeps 8,192 sends for a worker, and for each rank of a data-parallel worker, whether or not the profile sets `sent_ttl_ms`. One more send lets the oldest of them go.

A profile that sets `sent_ttl_ms` or `sent_report_delay_ms` is refused unless one of its stages reads the tally. Of the built-in stages, these read it:

- `least-load`
- `least-cost` with a `sent.` term
- a `has-room` rule with a `sent.` term

`resident-work` keeps a tally of its own, which `sent_ttl_ms` does not reach.

## The view

The `view` block bounds what a decision reads of a request and of a worker.

| Field | What it holds | Limits | Default |
|---|---|---|---|
| `headers` | The request headers a stage may read. | Up to 256 unique lowercase names. | Empty |
| `block_size` | The fallback block size, in tokens. | Above zero. | Unset |
| `boundaries` | Token positions that prefix keying uses. | Above zero, strictly ascending. | Empty |
| `max_block_hashes` | How many prefix block hashes a request carries. | 1 to 8190. | 256 |
| `max_report_age_ms` | A limit on a load report's age, in milliseconds. | 1 to 3599999. | Unset |
| `labels` | The worker labels a stage may read. | Up to 64 unique names. | Empty |
| `zones` | The zones a stage may tell apart. | Up to 256 unique names. | Empty |
| `engines` | The engine families a stage may tell apart. | Up to 64 unique names. | Empty |

Some fields have rules of their own:

- **`headers`.** The order matters: the wire names a header by its position in this list. Seven names are denied: `authorization`, `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key` and `x-goog-api-key`. An embedded host's validation can deny more.
- **`block_size`.** It applies when a token request gives no block size. Omitted, the prefix stages fall back to 16. A text request states its own window, in bytes. Where the engine hashes the text itself, that window is four bytes for each token of the block size.
- **`boundaries`.** A `hash` index entry requires at least one. Where a `consistent-hash` scorer sets `prefix_tokens`, that depth and every boundary must hold one whole block at least and `max_block_hashes` whole blocks at most.
- **`labels`.** A label is `key` or `key=value`, with no empty side.

An applied spec can read fewer block hashes than `max_block_hashes`:

- It reads none when no stage needs hashes.
- It reads none of a kind no stage declares. A text request is hashed only where a stage reads text windows.
- A host asks `AppliedSpec::block_hashes_read` with the request's kind before it hashes.

### The age of a load report

`max_report_age_ms` limits the age of a load report that these stages count. A report exactly at the limit still counts. For an older report, each takes its unreported path:

- `least-load`, `resident-work` and `least-cost`
- `has-room` with a rule
- the load gates of `prefix-affinity` and `warmup-slice`

Other readers follow their own rules:

- A host-registered stage reads a report under the limit through `build.load_report_reader()`.
- A host-registered stage that reads `row.load` itself sees the report at any age.
- Overload verdicts can still reflect older reports.
- Plugins interpret report age themselves.

## Index

`index` maps a model pattern to the settings of that model's prefix index. A pattern is an exact id, `prefix*`, `*suffix` or `*`.

| Field | What it holds | Default |
|---|---|---|
| `kind` | `tree` or `hash`. | `tree` |
| `ttl_s` | How many seconds a `hash` placement lives. | 180 |
| `max_size` | The size bound of a `tree`. | 67108864 |
| `sweep_s` | How many seconds pass between sweeps. | 120 |

Each number is above zero. `max_size` counts a tree in one of two ways:

- For text, it counts the characters, plus 1024 for each node a worker stands on.
- Otherwise it counts the tokens.

Where several patterns match one model, the most specific wins:

1. An exact id.
2. The longest literal prefix or suffix.
3. `*`.

A spec that holds a `prefix*` and a `*suffix` of equal length is refused.

A host can read these settings for an index of its own. The crate's engine reads `kind` and `max_size` alone, for a pool whose profile reads sent prefixes:

- Under `tree`, the engine bounds the model's record of sent prefixes at `max_size` divided by the block size. The record holds one placement for each block a worker was sent.
- At the defaults that is 4,194,304 placements. The record takes about 55 bytes a block, so about 230 MB a model.
- Under `hash`, the pool keeps no record.
- The entry read is the one for the workers' primary model id. A request that names the model by another id shares that record and its settings.

## Signals

`signals` declares named values of a worker's report for the stages to read. It holds up to 64 names. A name is a lowercase letter followed by lowercase letters, digits or underscores, at most 63 bytes.

| Field | What it holds | Default |
|---|---|---|
| `unit` | The unit a report must state: 1 to 31 ASCII characters, no space. | Required |
| `max_age_ms` | The oldest value that still reads, from 1 to 3599999. | Unset |
| `across_ranks` | `sum`, `max`, `min` or `mean`. | `sum` |

- The unit is compared byte for byte with the unit a report states.
- A value older than `max_age_ms` reads as no value.
- `across_ranks` says how one worker's value is made where its report holds several ranks.
- A spec with no signals is unchanged, and so is its digest.

See also [Signals a decision can read](../REFERENCE.md#signals-a-decision-can-read).

## Bindings

A *binding* says which requests a profile serves. Bindings are ordered, and the first match wins.

```yaml
apiVersion: smg.policy/v1
profiles:
  chat:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  batch:
    filter: [healthy]
    pick: round-robin
bindings:
  - match: { model: "batch-*" }
    profile: batch
  - match: { model: "*" }
    profile: chat
```

Here a model whose id starts with `batch-` uses the `batch` profile. Every other model uses `chat`.

A binding matches a request on two fields:

| Field | What it takes | Default |
|---|---|---|
| `match.model` | An exact id, `prefix*`, `*suffix` or `*`. | `*` |
| `match.phase` | A list of `single`, `prefill`, `decode` and `encode`. | Empty: every phase |

An empty `bindings` list works with one profile. That profile then serves every model under the plan `single`.

## Plans

A binding's `plan` says which roles one decision fills. Each role is one *side* of the plan, with a profile of its own.

| `plan` | Profiles the binding names | What the engine picks |
|---|---|---|
| `single` (default) | `profile` or `profiles.single`. | One worker in the requested role, with optional alternates. |
| `pd-pair` | `profiles.prefill` and `profiles.decode`. | A prefill worker and a compatible decode worker, in one decision. |
| `epd` | `profiles.encode`, `profiles.prefill` and `profiles.decode`. | A pair, then an encoder for each encode item. |

A plan with several sides serves every one of them. If its binding lists phases, the list must include each side.

### Prefill and decode

```yaml
apiVersion: smg.policy/v1
profiles:
  prefill:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
plans:
  pd-pair:
    pairing: lenient
bindings:
  - match: { model: "*" }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
```

The engine picks the pair in this order:

1. The decode profile's hard filters constrain which prefill workers have a partner.
2. The decode profile then sees the workers compatible with the selected prefill.
3. If it yields no target, the engine can try another prefill from the pick.

`plans.pd-pair.pairing` says how strictly a prefill pairs with a decode. It rules the pair inside `epd` too.

| `pairing` | What it rejects |
|---|---|
| `off` | Nothing: it skips the descriptor checks. |
| `lenient` (default) | Known differences. |
| `strict` | Also a missing runtime or transport, a hard fact missing on one side, and different versions when both are provided. |

These rules stand beside the modes:

- Under `lenient` and `strict`, both sides may omit a KV layout fact, and a shared explicit protocol settles the handoff after the runtime check.
- In every mode, an HTTP-only worker pairs only with an HTTP-only worker. Two workers on any other wire cannot pair when their known runtimes differ, even under `off`.
- The store derives no compatibility. Whoever holds the worker objects derives it from them (`PairClasses::derive`), under the pairing mode of the spec in force.
- An embedded host that uses the store can supply the runtime and a KV handoff descriptor.

`plans.pd-pair.decode` says when the decode side is picked: `with-prefill` (the default) or `after-prefill`. See [A decode chosen after its prefill](../REFERENCE.md#a-decode-chosen-after-its-prefill).

Workers of the standalone service pair by their labels:

- Over the registration RPC, the pairing descriptor accepts only a bootstrap port. The runtime comes from `spec.runtime`.
- A worker registered this way, or found by the controller's own discovery, pairs by what its labels report, as a gateway reads them.
- The labels report its transfer backend or `kv_connector`, its version, its KV layout facts and its operator's `pairing_protocol`.
- The controller's discovery writes a found worker's pairing protocol and KV connector among that worker's labels.
- Use `lenient` or `off` for workers whose labels report no transport.

A host that supplies its own admission facts can also add `has-capacity` and `has-room` to the profiles.

### Encode with a pair

```yaml
apiVersion: smg.policy/v1
profiles:
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
```

The engine first picks a pair whose prefill can accept the encoders' output. It then picks an encoder for each item, among that prefill's compatible encoders. A request with no encode items needs no encoder.

Use this spec with `InProcess::decide_encode_at`. Pass the request's `EncodeItems` and an `EncodePlaced`. A host that decides over its own rows calls `Engine::decide_encode_pair`, and supplies the worker offers and the compatibility tables as well.

- The encode profile can key each item by content hash.
- A push handoff can place by one supplied key.
- Images and videos require a vision-capable encoder, and a request can require a vision-capable decode worker.
- The decision writes the encoder choices to `EncodePlaced`. All choices are made before dispatch.

The entry and its refusal names stand in [`encode.rs`](../src/engine/framework/encode.rs).

## Where the rest stands

These parts of a binding are still in the reference:

- [Overload, shadows and reserved fields](../REFERENCE.md#bindings-and-plans), with `sessions` and `routing_key`
- [Session keys](../REFERENCE.md#session-keys)

A spec's life is there too:

- [Updating a policy spec](../REFERENCE.md#updating-a-document)
- [Decisions and refusals](../REFERENCE.md#decisions-and-refusals)
- [The presets](../REFERENCE.md#presets), each as a whole spec
