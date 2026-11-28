# The built-in stages

A profile of a policy spec names its stages: some filters, some scorers and one picker. This page lists every stage of the stock catalog, with its rule, its parameters and their defaults.

The [README](../README.md#built-in-stages) has the short tables. The [policy spec's page](policy-spec.md) shows where a profile names its stages.

## Filters

A *hard* filter may drop every worker. A *soft* filter steps aside when it would drop the last ones.

<!-- filters: begin -->
| Name | Kind | What it reads | What it keeps |
|---|---|---|---|
| `healthy` | Hard | The ready, draining, breaker and stall flags, and the asking host's veto mask. | Ready workers clear of those vetoes. |
| `not-overloaded` | Hard | The latched overload bit and the host's overload lift. | Workers that are not overloaded, and every worker under the host's lift. |
| `has-capacity` | Hard | The host's bit for a free prefill admission slot. | Workers whose bit is open. |
| `has-room` | Soft | Written bare, the host's bit for a room in the decode running window. Written with a rule, the values the rule names. | Workers with room. |
| `not-tried` | Soft | Earlier attempts and the host's exclusions. | Workers outside both lists, while any remain. |
<!-- filters: end -->

No filter takes a parameter under its bare name. `has-room` alone can be written with a rule.

### `not-overloaded`

- No policy spec names an overload threshold. A worker's registration states the worker's thresholds.
- Without them a worker takes its program's default: 8 queued requests and 0.8 active KV usage, unless the program sets others.
- A worker of the `external` runtime takes no default when it registers. A standalone controller judges it by its registration's thresholds alone.
- The binding's `overload` block switches that default on or off.
- [Overload](policy-spec.md#overload) has the rules for each host.

### The host's two bits

`has-capacity` and the bare `has-room` each read a bit the host sets.

- The room `has-room` asks about is a free one, or the room the request holds there already.
- A row the store builds reads both bits open.
- For capacity, the host that asks gates it at its own commit.
- On a pair's prefill side, a shed under `has-capacity` says a wait for room. Where the side's own run sheds, a picker's decline aside, it is named for a prefill with an open partner that stands but for its limit.
- It is named too where the pair's default takes a prefill at its limit, under a profile that lists it as a hard filter.
- Behind the pair's default and its question of saturation, where no prefill at the request's places stands but for the limit, the side's stage is read with it set aside.
- The limit's place on that side changes no answer there, under filters whose word on a row does not turn on how often they are asked.

### `has-room` with a rule

`has-room` takes a rule the spec writes:

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter:
      - healthy
      - has-room:
          used: { load.resident_tokens: 1, sent.tokens: 1, request.prompt_tokens: 1 }
          capacity: 200000
          headroom: 0.05
    score: [least-load]
    pick: best
```

- A worker has room where `used <= (1 - headroom) * capacity`.
- `used` adds each named value times its coefficient, in the document's units.
- The names are the ones `least-cost` reads, and two of the request's own.
- `request.prompt_tokens` is the request's size: its token count, and for a request of text alone its bytes over four.
- `request.max_output_tokens` is the most tokens the request's answer may hold.
- Under a capacity that is a number, `used` names at least one field or signal of the worker. The request's own values alone are refused there: every worker would read the same sum.
- A worker that lacks a named value has no room: one with no report under a `load.` name, or one whose report does not count its resident tokens.
- Under `view.max_report_age_ms` an older report reads as none.
- The filter stays soft. Where no worker has room, every worker stands, and the filter sheds nothing.
- A `sent.` name reads the profile's count of sends, so the profile may set `sent_ttl_ms`.
- `capacity` is a number above zero, or a declared signal written `signal.<name>`.
- A worker whose capacity reads zero or less has no room, and so has one whose capacity has no fresh value.
- `headroom` is from 0 to under 1, and zero where the rule leaves it out.
- An empty map is refused by the part it lacks.
- A null under the name is refused with both forms named: a rule with `used` and `capacity`, or the bare name, which reads the host's bit. That holds whichever reader takes the document.
- A profile that names the rule is asked a filter at a time. The bare name still folds into the one test of a row.

## Scorers

Every scorer takes a `weight`: a finite number above zero, 1 by default. The weights' sum and its reciprocal must be finite. A fraction's range includes its ends unless its line says otherwise.

<!-- scorers: begin -->
| Name | What it reads | What it favors |
|---|---|---|
| `least-load` | Load reports, requests in flight, and the sends that no report shows yet. | The shortest estimated wait. |
| `least-requests` | Requests in flight. | The lowest count. |
| `fewest-sessions` | Each worker's count of active sessions. | The fewest keys with requests in flight. |
| `session-affinity` | The request's pin from the host. | The pinned worker: it scores 1 and the others 0. |
| `prefix-affinity` | The matched prefix depth and the request's size, with requests in flight and an optional gate on active KV usage. | The deepest holders that meet its spill and pressure rules. |
| `kv-pressure` | Nothing of its own: its score is flat. | No worker. It sets the gate that `prefix-affinity` and `warmup-slice` read. |
| `warmup-slice` | A worker's admission age, its cache's growth and size, the prefix depth and requests in flight. | Young or thin workers, for a share of the misses. A single request warms nobody while the model's index holds no block. |
| `bucket` | The prefill prompt's size and the tokens booked on each worker. | The owner of the prompt's size range, or a usable owner near it. |
| `consistent-hash` | The routing key, or a prefix key where the stage sets one. | The key's owner on a ring, then its successors. |
| `resident-work` | Reported resident context, requests, the host's counts and the sends no report shows yet. | The lowest total work. |
| `least-cost` | The fields its `terms` name. | Workers within the cost margin of the least weighted sum. |
<!-- scorers: end -->

`least-requests`, `fewest-sessions` and `session-affinity` take a weight and nothing else.

### `least-load`

- The wait is estimated from queued and running token work, the drain rate and active KV pressure.
- With no reports the scorer uses the counts of requests in flight.
- It reads its sends from the profile's tally of sends. `StageBuild::dispatch_credit` hands a registered scorer that tally, and the `sent.` terms of `least-cost` read it. So a profile records each send once.
- The profile's `sent_ttl_ms` ends a send for this scorer too.
- A send that states no size counts at `mean_prefill_tokens`.

| Parameter | Default | Bound |
|---|---|---|
| `kv_pressure_weight` | 0.15 | Finite, zero or above. |
| `mean_prefill_tokens` | 1024 | Above zero. |
| `default_throughput` | 2000 tokens a second | Finite, above zero. |

`default_throughput` is used when no compared worker reports a live rate.

### `prefix-affinity`

- A hit the index names counts at any depth.
- A predicted hit must exceed `threshold`.
- For a request decided inside a tree the host lends, the stage reads the match's own share: the match's tokens or characters over the request's.
- Where that share is over `threshold`, every worker the match names is a holder, though the match is under one block.
- The validator refuses a positive `decay`, a positive `accounting_ttl_ms` and `spill.uncached_credit: true`.

| Parameter | Default | Bound or values |
|---|---|---|
| `threshold` | 0.3 | 0 to 1. |
| `source` | `merged` | `index` reads cache events alone. `predictor` reads sent prefixes alone. |
| `spill.abs` | 64 requests | 0 to 4,294,967,295. |
| `spill.rel` | 1.5 | Finite, 1 or above. |
| `spill.prefix_margin_tokens` | 0 | 0 to 4,294,967,295. |

### `kv-pressure`

- The affinity stages that use the gate read the spread and the peak of reported active KV occupancy.
- Each limit at 1 or above is off.
- A profile has one gate: its `kv-pressure` entries write the same limits, or the document is refused. Limits that are off count as the same.

| Parameter | Default | Bound |
|---|---|---|
| `spread` | 1 | Finite, above zero. |
| `ceiling` | 1 | Finite, above zero. |

### `warmup-slice`

- The slice gives young or thin workers a share of the misses.
- A single request warms nobody while the model's index of cache events holds no block. A pair's side still warms a young worker there. So does a profile with a plugin's stage, or with warm-up stages that rule differently.
- It can divert shallow hits to refill a thin cache.
- It favors a warming worker on a miss only while that worker is under the profile's spill gate.
- The gate is the `spill.rel` and `spill.abs` of the profile's `prefix-affinity` entries: the defaults for a bare name.
- Beside a `warmup-slice` those entries write the same two limits, or the document is refused.

| Parameter | Default | Meaning |
|---|---|---|
| `secs` | 60 | The age rule. Zero switches it off. Where a host names the warming rows, also how long a thin row that holds nothing is named. |
| `share` | 0.25 | The share of the misses, 0 to 1. |
| `blocks` | 1024 | The growth rule. |
| `thin_ratio` | 0.5 | A share of the median index size, 0 to 1. Zero switches the thin-cache rule off. |
| `divert_every` | 8 | At most one hit in this many is diverted to a thin worker. Zero switches the diversion off. |
| `miss_share` | 0.3 | The overlap treated as a miss, 0 to 1. |

### `bucket`

- The scorer spills to the least booked worker when both spread thresholds are crossed.
- At a phase other than prefill every worker ties.
- The standalone validator refuses this stage, because it requires lifecycle reports.

| Parameter | Default | Bound |
|---|---|---|
| `abs_threshold` | 64 tokens | 0 to 4,294,967,295. |
| `rel_threshold` | 1.5 | Finite, 1 or above. |
| `adjust_interval_s` | 5 | Above zero. The booking window and the adjustment cadence. |

### `consistent-hash`

A load bound can pass the owner over. Both parameters are unset by default.

| Parameter | Default | Bound |
|---|---|---|
| `prefix_tokens` | Unset | At least one block of `view.block_size`, and at most `view.max_block_hashes` blocks. |
| `bounded_load.factor` | 1.25, with the `bounded_load` block present | Finite, 1 or above. |
| `bounded_load.abs_margin` | 10, with the block present | 0 to 4,294,967,295. |

### `resident-work`

- The scorer uses the counts of requests in flight when no compared report counts resident tokens.
- `request_tokens` is required. It is a whole number, zero included: the token cost added for each request.

### `least-cost`

| Parameter | Default |
|---|---|
| `terms` | Required. |
| `margin` | Zero. |
| `unreported` | Required when a term reads a report or a signal. |

**The terms.** `least-cost` adds each named value times its coefficient, in the document's units.

- `candidate.in_flight` comes from the host.
- The report gives `load.running`, `load.queued`, `load.waiting_uncached_tokens`, `load.kv_usage_permille`, `load.active_kv_permille` and `load.resident_tokens`.
- `sent.requests` and `sent.tokens` come from the profile's count of sends.
- A name written `signal.<name>` reads a signal the spec declares.
- Resident tokens are present only when the report counts them.
- Waiting tokens are missing when they read zero beside a queue: a report of a queue's depth alone carries no token count.

For example, `terms: { load.queued: 11, candidate.in_flight: 2.5 }` adds 11 for each queued request and 2.5 for each request in flight.

**The sends.** `sent.requests` counts the requests the deciding host sent the worker that no report of it shows yet. `sent.tokens` counts the prompt tokens of those sends.

- A send counts the tokens it stated, and a send of text alone its bytes over four.
- A send that states no tokens and no text adds no tokens.
- Both values are present for a worker with no report.
- A stage that names `sent.tokens` asks its host for each request's size (`Needs::prompt_size`), as one that names `request.prompt_tokens` does.
- So a host that can send a request on without reading its prompt reads the text under such a document.

For example, `terms: { load.queued: 11, sent.requests: 11 }` prices a request just sent like one the report already shows queued.

**When a send stops counting.** A send stops counting at the first of these events:

- A report sampled at or after the send. Where the profile sets `sent_report_delay_ms`, the report is at least that long after it.
- The profile's `sent_ttl_ms`.
- The worker is removed, joins again, is replaced or changes its number of ranks.
- As a rule a row keeps 8,192 sends, and a send past them lets the row's oldest go.

These rules hold for the reports:

- A standalone controller's decision is given its load monitor's newest snapshot, and a send ends at a report there too.
- Such a controller gives none yet to a document that holds a stage that keeps reports. There no report ends a send: the other events of the list above do.
- A report older than `view.max_report_age_ms` still ends sends as a fresh one does.
- A `least-cost` that names a `sent.` term asks the host for load reports even where no term reads one, because a report ends sends.
- A profile that sets `sent_ttl_ms` needs a stage that reads sends, and a `least-cost` with a `sent.` term is one.

**A worker that lacks a value.** With `unreported: { over-least: { request_cost: 13 } }`, a row missing a named value costs 13 for each request in flight above the least reported cost. Where no row reports, it costs that above zero.

- The reported costs include their `sent.` terms.
- A row priced over the least takes no `sent.` term of its own.
- `unreported: missing-as-zero` sums the available named values and keeps a row's sends in its sum.

**The margin.** The absolute margin and the relative share of the least cost's magnitude add together. Each row within that margin scores 1. With this scorer alone, `best` draws from those rows before the others.

**The bounds.**

- A coefficient can be positive or negative, and must not be zero.
- Every numeric setting is finite.
- The weight, a coefficient's magnitude, the absolute margin and the request cost are at most `1e15`.
- The relative margin is at most `1e6`.
- Both parts of the margin and the request cost are zero or above.

### A request hashed by its text

A host with no token ids can hash a request's text in their place.

- It cuts the text into fixed windows of bytes and hashes each whole window, as many as the spec's stages read. A text shorter than one window, and not empty, is hashed whole.
- A window is the model's block size times four: 64 bytes at the default 16.
- The request says `hash_kind: text` and states the window's size in bytes as its `block_size`.
- It states no token count: its `prompt_tokens` is 0, and its `prompt_bytes` is the text's length in bytes.
- Its `prompt_chars` is the length in characters, where a stage of the document reads a request's size.

The stages read such a request's size in three ways:

- A reader with no classic twin prices it by its bytes over four (`prompt_size()`): the request term of `has-room`, `cache_aware`, and the `sent.` terms for its send.
- `least-load` and `resident-work` credit its send at their own mean prefill.
- `bucket` sizes it by its characters.

Over the wire these rules hold:

- A frame carries the bytes over four as its one count. So a controller's stages read that number, `bucket` among them.
- A text request that carries hashes and states no window is refused.
- A caller of the crate's shared link is refused it on its own side, and the stream it shares stays up.
- The serial client sends it, reads the stream's refusal and opens another stream at its next ask.

`prefix-affinity` reads these hashes through the record of sent prefixes, under `source: merged` or `predictor`.

- Cache events are token blocks, so the index of cache events is never asked for a text request.
- The record of sent prefixes speaks for every worker of a text request's list, whatever the index holds for the worker.
- Under an index of the kind `hash` the record keys no text request: [Index](policy-spec.md#index) has the rule.
- A text request under a cache partition is hashed under the partition's whole identity, the extra key included. So it shares no window with a request of another partition.

## Pickers

<!-- pickers: begin -->
| Name | What it reads | How it picks |
|---|---|---|
| `best` | The combined scores. | The highest first, with a seeded draw inside the score's tie band. |
| `round-robin` | The worker list and the ordered set of survivors. | The list's next worker, or the next turn of a rotation over the survivors. |
| `random` | The survivors and the request's seed. | It draws a worker and then alternates. |
| `power-of-two` | The scores of two random draws. | The better of the two. Where the host supplies suitable engine groups, it draws from two engines, takes the better row's engine and picks that engine's best row. |
| `count-first` | The counts of requests in flight, then the combined score. | The lowest count, by score among the rows at it. It orders higher-count alternates by count. |
| `power-of-two-count-first` | Two random rows. | By the count of requests in flight, then by the combined score on equal counts. |
| `passthrough` | The survivors' order. | The first worker in the host's order. |
| `manual` | The binding's session assignment and pins. | The assigned or pinned worker. It uses `min_load` with no rule or for `delegate`. |
| `best-with-temperature` | Nothing. | The name is reserved, and the validator refuses it. |
<!-- pickers: end -->

- `best` takes `sample`, unset by default. Write `best: { sample: 16 }` for a sample. Valid sizes are 1 to 64.
- `power-of-two` and `power-of-two-count-first` require a scorer.
- The validator refuses scorers with `manual`.
- The reserved `temperature` of `best-with-temperature` must be finite and zero or above.
- The other pickers take no parameter.

### `round-robin`

- A single request's first attempt, held to no part of its list, takes its list's next worker where the filters keep it.
- It does so at every list size where the list has a counter of its own, else from 16 workers.
- Otherwise the picker advances a rotation for the set of survivors.

### `best` with a sample

- The sampled path needs a host that names prefix holders and warming rows, and stages that can work on a draw.
- It runs at 16 or more rows and more than twice the sample size.
- It makes at most twice the sample size in draws, to keep up to the sample size in distinct usable rows.
- The named lists join the draw.
- Small pools, an empty draw, a missing named view, or stages that need the whole pool use the full path.

Two more rules hold for scores:

- `power-of-two` scores its compared set.
- A normalized scorer can cast its full weight as a vote between two rows.
