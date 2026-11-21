# Policy controller

The policy controller chooses which workers serve a request. One versioned YAML or JSON document composes health and capacity filters, weighted scores for load or cache locality, and a picker with alternates. Bind profiles by model and phase, keep related requests on a worker, or choose compatible workers for split requests. The same Rust library runs in a host process and behind the standalone gRPC service. The service validates a new document before it replaces the running one and can explain sampled picks and sheds.

## First decision

From a checkout, run the small in-process example with three ready workers. It checks that the worker with one request in flight comes first, followed by the workers with three and five. It needs no running service.

```sh
cargo test -p smg-policy-controller --locked --lib the_pick_orders_targets_best_first_and_names_the_survivors
```

This document describes that profile. With one profile, an omitted or empty `bindings` list uses it for every model and phase under the `single` plan.

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-tried]
    score: [least-requests]
    pick: best
```

1. **Filter.** Start with the model's pool for the requested role. Hard filters run first, in their written order. An empty hard pass usually sheds. A soft filter that would empty the pool leaves it as it was before that filter. Overload and liveness have [fallback rules](#decisions-and-refusals).
2. **Score.** Each scorer rates the rows it is asked to compare from 0 to 1, higher first. The combined score is a weighted average. A bare name has weight 1. A mapping puts the weight beside that scorer's other parameters. In the next document the weights give `least-load` three quarters of the combined score.
3. **Pick.** Every profile has one picker. A pick carries up to four ordered targets. A shed gives a reason and the stage or constraint that stopped the decision. The picker can narrow the comparison before scoring.

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - least-load: { weight: 3 }
      - least-requests
    pick: best
```

## Built-in stages

### Filters

<!-- filters: begin -->
| Name | What it reads and keeps | Parameters and defaults |
|---|---|---|
| `healthy` | Hard. Reads ready, draining, breaker and stall flags and the asking host's veto mask. Keeps ready workers clear of those vetoes. | None |
| `not-overloaded` | Hard. Reads the latched overload bit and the host's overload lift. | None on the stage. The binding's `overload` block sets thresholds, 8 queued requests or 0.8 active KV usage by default. |
| `has-capacity` | Hard. Reads the host's bit for a free prefill admission slot. A row the store builds reads the bit open, and the host that asks gates capacity at its own commit. | None |
| `has-room` | Soft. Written bare, reads the host's bit for free room in the decode running window. A row the store builds reads the bit open. Written with a rule, keeps the workers whose named fields sum to at most the capacity less the headroom. | None for the bare name. The rule takes `used` (required), `capacity` (required) and `headroom` (zero). |
| `not-tried` | Soft. Reads earlier attempts and the host's exclusions. Keeps rows outside both lists while any remain. | None |
<!-- filters: end -->

`has-room` takes a rule the document writes:

```text
filter:
  - healthy
  - has-room:
      used: { load.resident_tokens: 1, sent.tokens: 1, request.prompt_tokens: 1 }
      capacity: 200000
      headroom: 0.05
```

- A worker has room where `used <= (1 - headroom) * capacity`. `used` adds each named value times its coefficient, in the document's units.
- The names are the ones `least-cost` reads, and `request.prompt_tokens`, the request's size.
- `used` names at least one field of the worker. Under a capacity that is a number the request's size alone is refused: every worker would read the same sum.
- A worker that lacks a named value has no room: one with no report under a `load.` name, or one whose report does not count its resident tokens. Under `view.max_report_age_ms` an older report reads as none.
- The filter stays soft. Where no worker has room, every worker stands, and the filter sheds nothing.
- A `sent.` name reads the profile's count of sends, so the profile may set `sent_ttl_ms`.
- `capacity` is a number above zero, and `headroom` is from 0 to under 1. An empty map is refused by the part it lacks. A null under the name is refused with both forms named: a rule with `used` and `capacity`, or the bare name, which reads the host's bit. YAML's own typed read takes that null for an empty map, and refuses it by the part it lacks.
- A profile that names the rule is asked a filter at a time. The bare name still folds into the one test of a row.

### Scorers

Every scorer takes a finite positive `weight`, default 1. The sum and its reciprocal must be finite. Fractions in the table include their endpoints unless stated otherwise.

<!-- scorers: begin -->
| Name | What it reads and favors | Parameters and defaults |
|---|---|---|
| `least-load` | Load reports, in-flight counts and dispatches the reports have yet to reflect. Favors the shortest estimated wait from queued and running token work, drain rate and active KV pressure. With no reports it uses in-flight counts. It reads the dispatches from the profile's tally of sends, the one `StageBuild::dispatch_credit` hands a registered scorer and the `sent.` terms of `least-cost` read, so a profile records each send once. The profile's `sent_ttl_ms` ends a send for it too. A send that states no size counts at `mean_prefill_tokens`. | `kv_pressure_weight` 0.15 (finite, ≥0), `mean_prefill_tokens` 1024 (positive), `default_throughput` 2000 tokens/s (finite, positive, used when no compared row reports a live rate). |
| `least-requests` | In-flight request counts. Favors the lowest. | Weight only |
| `fewest-sessions` | Active session counts per worker. Favors the fewest keys with requests in flight. | Weight only |
| `session-affinity` | The request's pin from the host. Scores that worker 1 and the others 0. | Weight only |
| `prefix-affinity` | Matched prefix depth and request size, with in-flight counts and an optional active KV gate. Favors the deepest holders that meet the spill and pressure rules. Index hits count at any depth. A predicted hit must exceed `threshold`. | `threshold` 0.3 (0–1), `source` `merged` (each worker is read by what it has: its cache events once the store has heard one from it, and the prefixes sent to it until then. `index` reads cache events alone, and `predictor` reads sent prefixes alone), `spill.abs` 64 requests, `spill.rel` 1.5 (finite, ≥1), `spill.prefix_margin_tokens` 0. The validator refuses positive `decay`, positive `accounting_ttl_ms` and `spill.uncached_credit: true`. |
| `kv-pressure` | A flat score. The other affinity stages that use this gate read the spread and peak of reported active KV occupancy. | `spread` 1 and `ceiling` 1 (finite, positive). Each limit at 1 or above is off. |
| `warmup-slice` | Worker admission age, cache growth and size, prefix depth and in-flight counts. Gives young or thin workers a share of misses and can divert shallow hits to refill a thin cache. | `secs` 60 for age, `share` 0.25 of misses, `blocks` 1024 for growth, `thin_ratio` 0.5 of median index size, `divert_every` 8 hits, `miss_share` 0.3 for overlap treated as a miss. The three fractions are 0–1. Zero disables the corresponding age, thin-cache or diversion rule for `secs`, `thin_ratio` or `divert_every`. |
| `bucket` | Prefill prompt size and booked tokens. Favors a size range's owner or a nearby usable owner. It spills to the least booked worker when both spread thresholds are crossed. Other phases tie. | `abs_threshold` 64 tokens, `rel_threshold` 1.5 (finite, ≥1), `adjust_interval_s` 5 for the booking window and adjustment cadence (positive). The standalone validator refuses this stage because it requires lifecycle reports. |
| `consistent-hash` | The routing key, or a prefix key when configured. Favors the owner on a ring, then its successors. A load bound can pass the owner over. | `prefix_tokens` unset. `bounded_load` unset. With that block present, `factor` defaults to 1.25 (finite, ≥1) and `abs_margin` to 10. |
| `resident-work` | Reported resident context, requests, host counts and dispatch credit. Favors the lowest total work. Uses in-flight counts when no compared report counts resident tokens. | `request_tokens` required. A whole number including 0, the token cost added per request. |
| `least-cost` | Workers within the cost margin of the least weighted sum of the named fields. | `terms` (required), `margin` (zero), `unreported` (required when a term reads a report) |
<!-- scorers: end -->

`least-cost` adds each named value times its coefficient in the document's units. It reads `candidate.in_flight` from the host and `load.running`, `load.queued`, `load.waiting_uncached_tokens`, `load.kv_usage_permille`, `load.active_kv_permille`, and `load.resident_tokens` from the report. It reads `sent.requests` and `sent.tokens` from the profile's count of sends: the requests the deciding host sent the worker that no report of it shows yet, and the prompt tokens those sends stated. Both are present for a worker with no report. A send that states no size adds no tokens. Resident tokens are present only when the report counts them. Waiting tokens are missing when they read zero beside a queue: a report of a queue's depth alone carries no token count.

For example, `terms: { load.queued: 11, candidate.in_flight: 2.5 }` adds 11 for each queued request and 2.5 for each request in flight.

For example, `terms: { load.queued: 11, sent.requests: 11 }` prices a request just sent like one the report already shows queued. A send stops counting at a report sampled at or after it, at the profile's `sent_ttl_ms`, or when the worker is removed. A row keeps 8,192 sends, and a send past them retires the row's oldest 256. While a retirement runs, up to one more batch lands above the bound. A profile that sets `sent_ttl_ms` needs a stage that reads sends, and a `least-cost` with a `sent.` term is one. A report older than `view.max_report_age_ms` still ends the sends before it. A `least-cost` that names a `sent.` term asks the host for load reports even where no term reads one, because a report ends the sends before it.

With `unreported: { over-least: { request_cost: 13 } }`, a row missing a named value costs 13 per request in flight above the least reported cost, or above zero if no row reports. The reported costs include their `sent.` terms. A row priced over the least takes no `sent.` term of its own. `unreported: missing-as-zero` sums the available named values and keeps a row's sends in its sum. The absolute margin and the relative share of the magnitude of the least cost add together. Each row within that margin scores 1. With this scorer alone, `best` draws from those rows before the others.

Coefficients can be positive or negative and must be nonzero. Numeric settings are finite. The weight, coefficient magnitudes, absolute margin, and request cost are at most `1e15`, and the relative margin is at most `1e6`. Both margin parts and the request cost are nonnegative.

A host with no token ids can hash a request's text in their place. It cuts the text into fixed windows of bytes and hashes each whole window. A window is the model's block size times four, so 64 bytes at the default 16. The request then says `hash_kind: text`, states the window's size in bytes as its `block_size`, and counts its `prompt_tokens` as the text's bytes over four where it has no better count. Over the wire a text request that carries hashes and states no window is refused. A caller of the crate's shared link is refused it on its own side, and the stream it shares stays up. The serial client sends it, reads the stream's refusal and opens another stream at its next ask. `prefix-affinity` reads these hashes through the record of sent prefixes, under `source: merged` or `predictor`. Cache events are token blocks, so the index of cache events is never asked for a text request. A text request under a cache partition is hashed under the partition's whole identity, the extra key included, so it shares no window with a request of another partition.

### Pickers

<!-- pickers: begin -->
| Name | What it reads and how it picks | Parameters and defaults |
|---|---|---|
| `best` | Combined scores. Highest first, with a seeded draw inside the score's tie band. | `sample` is unset by default. Use `best: { sample: 16 }` for a sample. Valid sizes are 1–64. |
| `round-robin` | The ordered survivor set. Advances a rotation for that set. | None |
| `random` | The survivors and the request's seed. Draws a worker and then alternates. | None |
| `power-of-two` | Scores of two random draws. Uses two engines when the host supplies suitable engine groups. | None. Requires a scorer. |
| `count-first` | In-flight counts, then combined score for the lowest-count rows. Orders higher-count alternates by count. | None |
| `power-of-two-count-first` | Two random rows, then in-flight count, then combined score on equal counts. | None. Requires a scorer. |
| `passthrough` | Survivor order. Takes the first worker in the host's order. | None |
| `manual` | The binding's session assignment and pins. Uses `min_load` with no rule or for `delegate`. | None. The validator refuses scorers with this picker. |
| `best-with-temperature` | Reserved name. The validator refuses it. | Its reserved `temperature` must be finite and ≥0. |
<!-- pickers: end -->

The sampled `best` path needs a host that names prefix holders and warming rows, and stages that can work on a draw. At 16 or more rows and more than twice the sample size, it makes at most twice that many draws to keep up to the sample size in distinct usable rows. The named lists join the draw. At a host's door the holders that fill a sample are asked the pass's own test of a row, up to the first that passes. Where the pass would refuse every one of them (workers the request already tried, ones over their thresholds, ones that are not ready, ones the asking gateway vetoes), the lookup is made again without them, up to four times, and the next deepest holders are named instead. Where no lookup names a holder the pass would take, the first one's holders stand. Small pools, an empty draw, a missing named view, or stages that need the whole pool use the full path. Power-of-two scores its compared set. A normalized scorer can cast its full weight as a vote between two rows.

## Signals a decision can read

The engine reports a profile's input requirements through `Needs`. The host supplies the request and worker views. Native stages can read the full view. Plugins see the projection their host configures. See [the Rust view types](src/engine/view.rs) and [the wire records](contracts/policy_plugin.proto).

| Signal | Where it comes from |
|---|---|
| Model, phase, tenant, namespace seed, priority, streaming, output limit, prompt size, media counts, retry attempt and seed | The caller's `RequestFacts`. The remote caller sends these in `DecideRequest`. A frame with raw prompt text receives `Busy`. The built-in binding matcher reads model and phase. |
| Request hashes and headers | The caller sends token-block or text-window hashes, their kind and their block size. A token request may leave the size out and reads the document's. A text request states its window's size in bytes: over the wire, one that carries hashes and states none is refused by `facts.block_size`. The applied document bounds the prefix read. The caller projects headers as positions in `view.headers` and resolves the routing or session key. |
| Ready, draining, stall, breaker and reachability | Membership and health probes, plus the asking gateway's vetoes and wire constraints. A liveness veto is a doubt. A held veto represents the asking gateway's breaker and cannot be lifted by the liveness fallback. The standalone controller has no separate routing breaker for workers. |
| Running, queued and waiting uncached work, KV usage and active KV usage, drain rate, resident tokens, admission window, sample age and source | Host load reports. The standalone leader polls the worker's load RPC and can scrape HTTP metrics for a registered runtime with a supported parser. Fields depend on the reporter. Use `view.max_report_age_ms` to age out reports for the affected scorers and KV gates. Overload is a separate latched verdict. |
| Host requests in flight and dispatch credit | Lease bookings and lifecycle reports. Read counts through `RequestFacts::in_flight` so a stage also sees counts lent beside rows. The load scorers can account for dispatched work until a report covers it. |
| Pins, requests in flight for one key, and active-key counts | The host's session facts, or the engine's session rule and the store's pins of record. The standalone controller counts a key's requests in flight by the leases it holds, for a rule with an in-flight cap. Its decision view does not supply active-key counts or a premarked session-hit row. Its session rule keeps pins using the caller's key. |
| Cache depth, age, growth and size | An embedded host can submit KV events to `MemoryStore` and name prefix holders, or supply an affinity column itself. It can also supply sent-prefix predictions. With regular standalone ingestion, worker age is available, and the stock controller records the prefixes it sends at each single lease's issue. The stock controller reads cache events from each gRPC worker whose row asks for them and names a model. A controller of your own built without the `worker-host` feature reads none. |
| Role, ranks, wires, stable worker key, zone, engine family and labels | Membership. The document projects labels to bits, and zone and family to positions from 1. An unlisted or absent zone or family is 0. |
| Capacity, room, pair and encode compatibility, worker vision | The host supplies admission bits and worker facts. `MemoryStore` builds pairing tables and encode classes from membership for its library callers. Its candidate rows read both admission bits open, since the store knows no admission window. |
| Pool generation, size and membership, reachability and host constraints | The pool view and `Constraints` carry the pool's facts, exclusions, vetoes, a wire mask and an optional subset hint. A standalone request can narrow by common wires or specify a zero subset to exclude the pool. Hosts can supply further pool facts such as a regime or zone capacity. The standalone view supplies neither. |
| Named values the document declares under `signals` | Worker load reports, pushed or polled: the values a report states by name, and the two fixed token counts. The store keeps each declared value beside its rank's load sample and publishes it with the row, for a profile whose `Needs` set `signal_cells`. A native stage that resolved a signal's column reads them. No plugin contract carries them, and a standby does not follow them. |

An embedded host can implement or use the store's ingress traits for membership, health, loads, cache events, reservations, sessions and leases. `MemoryStore` has an event-fed chain index that holds up to 1024 workers per model. Its event sink reports sequence gaps and supports replay and resync. A snapshot that its stream cuts short keeps the blocks it had shown: the worker stays placed by them until the next snapshot starts. A host can read the validated `index` settings through `PolicySpec::index_for` and use them to configure its own index. The bundled store uses the same index layout for every document. It also keeps, a model, a record of the prefixes sent to workers the store has heard no cache event from, where the document's prefix stage reads them and the model's index `kind` is `tree`. The stage reads them unless it says `source: index`. A single lease's issue writes a placement, in process and in the stock controller alike, and the record holds `max_size` over the block size placements at most. A placement is a prediction made at the decision. A decision turned back after its lease's issue keeps it, and the next request with that prefix reads it as a predicted hit, which must exceed the stage's `threshold`. The record speaks for a worker until the store hears a cache event from it, and again once the worker is reported gone to the event sink. A worker is read by its cache events from the first applied batch that carries one. A pair's lease records nothing, and a standby starts with an empty record. A text request is recorded as a token request is, by the hashes of its windows. Readers of the record wait for a writer: under many callers that never pause, a fleet whose requests mostly grow decides slower than one that reads no record. A host with a store of its own keeps the same record with the crate's `PrefixPredictor`: it folds a request's keys once (`PrefixPredictor::fold_into`), reads by a shared reference, and writes with a tick of the host's clock. `kv_index::salt::request_text_hashes_with_seed` makes a text's window hashes under a cache namespace's seed, and `engine::TEXT_BYTES_PER_TOKEN` is the four bytes a token. A window of text and a block of token ids over the same bytes never share a hash. A host hashes a request by its model's block size: the size the store has on record for the model, and the document's `view.block_size` where it has none. A registration can state the size, and a wider registration raises it. Once the model's engines store a block, the store keeps the engines' own size: the widest block a model's indexed stores named. A worker need not state it. A model first hashed by the document's size and later by a size its engines state loses the prefixes sent before, until they are sent again. [Store API](src/store/mod.rs) · [Sent-prefix predictor](src/tracking/prefix_predictor.rs)

## Run it

### In a host process

Validate a `PolicySpec` to get an `Engine`. Call `Engine::decide` with request facts, rows, constraints and reusable `Scratch`. For a host that needs booking, apply an `AppliedSpec` to `MemoryStore` and call `StoreBackedEngine` or `DecisionGuard`. The door runs on the caller's thread. It issues the lease for the first target. The host takes a target, reports dispatch and completion or release, and calls `StoreBackedEngine::sweep` on its timer. The door decides a pair when asked at its prefill: that one request decides both sides. It returns `PairSide` for a decode or an encode sent on its own, and `Unleased` or `Faulted` if it cannot book or decide a pick. `StoreBackedEngine::decide_encode_pair` and `DecisionGuard::decide_encode_pair` also place an `epd` request's encode items. A pick can state what its first worker holds of the request's prefix (`Pick::overlap`): the leading blocks held, the request's blocks, where the match came from, and whether the prefix margin kept that worker in front. The profile's first `prefix-affinity` scorer says it. A host that calls the library asks through `Engine::decide_stated`. The controller's `Decide` answer does not carry it yet. [Engine example](src/engine/framework.rs) · [StoreBackedEngine API](src/engine/store_backed_engine.rs) · [DecisionGuard API](src/tracking/decision_guard.rs)

The crate defaults to `wasm`. Use `default-features = false` for the library without that runtime. `ingest` adds load polling and scrape helpers. `controller` adds the service, its protocol, command line and gRPC plugin host, and enables `ingest` and `operator`. `client` enables the reference client and watcher and implies `controller`. `kubernetes` adds the namespaced Lease backend. `operator` adds the operators' door for a host's own process. It builds no server, and it builds on every target. `worker-host` adds an optional registry of worker objects that follows membership. In the controller build those objects use a lazy gRPC connector. The feature also has the cache feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A controller built with the feature fills its index itself: every replica opens a cache-event stream to each gRPC worker whose row asks for events and names a model. The stock controller is built with it: its binary requires `controller` and `worker-host`. `discovery` lets the controller run a worker discovery provider, and enables `controller` and `worker-host`: see [Worker discovery](#worker-discovery). `test-util` supports the crate's tests and benches. [Cargo features](Cargo.toml)

A host built with `worker-host` can keep a worker object for each membership row. If the host gets health from the rows, use `Objects::new(registry).with_status_followed()` and call `follow(&store)` when the store moves. The holder copies each row's status on a status change or a snapshot read. By default, a status change in the store's tail does nothing to the object.

For example, a worker registered `Pending` can take requests once its row turns `Ready` and the host's breaker and vetoes allow it. If a snapshot changes just that row's status to `NotReady`, the same object stops taking requests. The registry tells its subscribers each change, and a promotion puts an open breaker on trial, as a health probe's promotion does.

### A decode chosen after its prefill

A host in process sets `request.constraints.decode_later = true` on a pair's request. The first pick names both sides and books the prefill alone. The host can decide the decode again once it has reported the prefill's dispatch, and before it reports the prefill's end.

A stage that keeps state by issue hears the issue of every decode before its dispatch. Under a pair asked with `decode_later`, a decode call issues the decode side and the stage hears that issue. A decode sent with no call goes out as the first pick's: the stage hears its issue at that dispatch, just ahead of the dispatch's own word. The decision told there holds each target's workers and ranks, with no trace and the default row position, 0, and the lease names the target taken. A stage reads the worker by its id there.

The lease of a later decode's issue is dated when its side is filed: at the decode call, or as its dispatch is told. It is never dated at the pair's first decision, nor by the date a gateway gives the dispatch. The lease's later words keep the first decision's instant.

For example, after the prefill is sent, call `handle.dispatched(sent_at)` and then `handle.decide_decode(&request, now)`. The new pick names the held prefill with each decode it can use, on the loads of that instant. Its first decode is booked and has five seconds to be dispatched. `handle.take_decode(index)` takes an alternate and `handle.dispatched_decode(sent_at)` reports the send.

A handle keeps its first pick where the call sheds or is refused. A decode side the host gave up with `released_decode` takes no call after: `decide_decode` answers `DecisionRefused::Decode(DecodeRefused::DecodeStands)` and asks no door. `decide_decode` answers the decision the handle holds, so its answer does not tell a shed from a pick. `handle.lease().due` does: a pick moves it to the call's deadline, and a shed leaves the first. A host that skips the call can send a decode of the first pick: its dispatch books that side.

A host that manages the lease itself calls `door.decide_decode(gateway, &request, id, now)`. A pick carries the decode side's lease under the same id, and a shed carries none. The lease keeps the generation and the time of the pair's first issue. The handle counts the decode's dispatch from the first decision, so a stage dates it by its own dispatch. A stage hears the decode's issue only where the lease's generation still stands: behind a reload the new engine decides the side and hears nothing of the lease. A refusal of the store comes back as `DecisionRefused::Decode` with the store's reason.

A host that files pairs directly through `LeasesStore::issue_pair` can set the decode side to `LeaseState::Chosen`. The side keeps the pick's decode targets without booking a worker or starting a dispatch deadline. A side still chosen leaves the store when its prefill ends.

Once the prefill is dispatched, `store.issue_decode(gateway, id, decode, now)` can replace that choice once, while the prefill is held. It books the new first target and gives the decode five seconds to dispatch. A dispatch on a chosen side can also book a target from the original pick. An index outside the pick uses its first target.

`issue_decode` returns `DecodeRefused::NoLease` when the gateway has no lease under that id, `NotChosen` when the lease was filed without a chosen side, `PrefillNotHeld` when its prefill is waiting or has ended, and `DecodeStands` when its chosen side has already been issued or dispatched, or its gateway ended it.

`LeasesStore::expire` returns both sides as the sweep left them. For example, a completed decode stays `Completed` when its prefill is released as abandoned.

### In the gateway

The gateway shipped in this checkout runs the controller in its own process with `--policy-controller on`. Without the flag, routing uses the gateway's classic policies. `--print-policy-spec` prints the document the routing flags mean, and exits. The Python launcher takes `--policy-controller`, `--policy-spec` and the four options of a gateway that follows a controller. It does not take `--print-policy-spec`.

**The document.** The gateway runs the policy document its routing flags mean. To run a document of your own, add `--policy-spec FILE`. A document replaces the routing flags: the command line refuses a routing flag typed beside `--policy-spec`, and the Python launcher refuses one that is not at its default. A document that names no generation starts as number 1. A program that embeds the gateway and registers stages of its own names them in that document, as [Register a Rust stage](#register-a-rust-stage) shows.

**What it serves.** The regular HTTP router asks the controller in the gateway's process for the worker of a typed request or a transcription. It refuses by name, with `not_decided_by_policy_controller` (HTTP 501), a request that carries `x-smg-target-worker` for a model whose routing policy is consistent hashing: the classic policy holds such a request to the worker it names, and the controller reads no named worker yet. Under any other policy the header changes nothing, as under the classic policies. The regular gRPC router asks the controller for each request it places. That router refuses three shapes by name with `not_decided_by_policy_controller` (HTTP 501): a request with media input, one with a context demand that a worker's window binds, and a retry under a decoding rule. The gRPC router of a prefill and decode pair asks it for the pair. The HTTP router of a pair asks it for each attempt's pair and takes the prefill through prefill admission. It refuses a request that carries `x-smg-target-worker` by the same name, where the pair's prefill or decode policy is consistent hashing. Both do so where the gateway's policy document plans pairs and the gateway follows no controller's address. Elsewhere each returns `not_decided_by_policy_controller` (HTTP 501) for every request: under a document that plans no pair, and under a controller's address. The gRPC router of an encode, prefill and decode plan returns it for every request. Under the policy controller the document that the routing settings mean plans one family for every model: pairs under the prefill-decode mode, single workers under the regular mode. A gateway of many routers serves a model of the other family under the classic policies and refuses it under the controller. To serve both, supply a policy document that binds each family: a `pd-pair` binding that matches the prefill and decode phases, ahead of a `single` binding. A provider's router, the realtime routes and the proxied GET routes serve as they do under the classic policies.

**Workers.** With the flag on and no controller's address, the gateway keeps its workers as it does without the flag. Its registry holds them, registration and removal run the same worker workflows, its health checks probe them, and the worker API answers as it does under the classic policies. Under a controller's address the gateway's store mirrors that controller, and the gateway's worker workflows write no row there. There a row's worker object is built from the row, and reads the defaults in what no row states.

**Rows.** The gateway serves the workers its policy controller decides over on its control plane: its own registered workers, or the policy store's rows under a controller's address. `GET /policy/snapshot` returns them in registration key order under the generation that runs, at most 1,024 a call. `models` keeps the rows that serve one of the models it names, at most 128, and `limit` cuts the page. A page also ends at 2 MiB of rows, and its `truncated` says that rows were left out. The page has no cursor, so a fleet past the cap is read by model. The call needs the `admin` role, and each address is told without its userinfo.

**Cache events.** With the flag on and no controller's address, the gateway reads its workers' cache events as it does without the flag. Under a controller's address the gateway reads no events: the controller does.

**Provider modes.** In a provider mode (`openai`, `anthropic`, `gemini`) without `--enable-igw`, every router of the gateway is the provider's, and the setting moves nothing. Such a router places its requests by no routing policy, so the gateway starts no controller and says so in one log line at the start. A `--policy-spec` document or a `--policy-controller-address` beside it is refused by name. With `--enable-igw` the gateway also serves self-hosted workers, and the controller stays in force for them.

**Following a controller.** With the flag on, `--policy-controller-address <host:port>` makes the gateway follow a standalone controller's workers. It takes the controller's worker watch into its policy store, a snapshot first and then each change, and it discovers, registers and probes no worker itself. It keeps one worker object for each row. The object's status is the row's, so the controller's health verdict decides which workers the gateway may send to. `--policy-controller-tls-ca-cert`, `--policy-controller-tls-cert` and `--policy-controller-tls-key` name the PEM files of that connection. A client certificate needs its key and the authority's file. Readiness waits for the first whole snapshot and its worker objects, and the worker API's writes answer HTTP 409 `WORKER_CONTROLLER_OWNED`. The mode builds on Linux and macOS. A request is still decided in the gateway's own process, over the rows the controller tells: the gateway asks the controller for no decision yet.

### Standalone service

Build from a checkout with `protoc` available. The service feature uses the worker protocol files from the checkout. The stock binary requires the `controller` and `worker-host` features. With either one missing, cargo builds no binary and says nothing. Save the first policy document above as `policy.yaml`, then run the following from the checkout root.

```sh
cargo build --release --locked -p smg-policy-controller --features controller,worker-host
target/release/smg-policy-controller --controller-id controller-0 --spec-file policy.yaml \
  --gateway-listen 127.0.0.1:30100 --control-listen 127.0.0.1:30101 \
  --replication-listen 127.0.0.1:30102 --admin-listen 127.0.0.1:30103 \
  --identity-as-told gateway,control
```

Register workers with `RegisterWorker` on the control listener. Use `Session` and `Decide` on the gateway listener to ask for a worker. The service returns targets and leases for the caller to dispatch and report. For a complete local example of registration and the decision stream, see [the controller's wire test](tests/controller.rs). The deployment chart can mount `policyController.spec` as a file or `policyController.specStoreClaim` as a store. [Deployment chart](../../deploy/helm/smg/README.md#policy-controller)

| Flags | Default and effect |
|---|---|
| `--help`, `--version` | Show help or version and exit. |
| `--controller-id` | Reads `HOSTNAME` if omitted. Use letters, digits, `.`, `_` and `-`. |
| `--spec-file`, `--spec-store` | Choose exactly one. A watched YAML or JSON file, or an existing directory managed through the spec calls. |
| `--gateway-listen`, `--control-listen`, `--replication-listen`, `--admin-listen` | Bind all addresses by default at ports 30100, 30101, 30102 and 30103 respectively. Admin uses HTTP. |
| `--ingestion-listen` | Optional separate ingestion listener. The gateway listener carries that scope by default. Both ingestion calls are unimplemented. |
| `--identity-as-told` | Comma-separated listener names, empty by default. A named listener accepts one `x-smg-principal` value as the caller's identity. Use it on loopback or behind a trusted proxy. |
| `--tls-cert`, `--tls-key` | Optional PEM server chain and key together for the gRPC listeners. |
| `--gateway-client-ca`, `--ingestion-client-ca`, `--control-client-ca`, `--replication-client-ca` | Optional PEM client CA per scope. Both scopes of a shared listener must use the same CA or both omit it. Requires server TLS and conflicts with `--identity-as-told` on that listener. |
| `--lease-file`, `--lease-name`, `--lease-namespace` | A process-local lease by default. File lease for replicas on one host with an existing parent directory. A named Lease needs the `kubernetes` feature. Its namespace defaults to the client's namespace. The two shared lease types are exclusive. |
| `--advertise-address` | A dialable host or host:port. A bare host uses the gateway port. Required for a shared lease when the gateway binds a wildcard address. |
| `--lease-duration-ms`, `--lease-renew-deadline-ms`, `--lease-renew-period-ms` | 3000, 2000, 1000. The duration must exceed the deadline, which must exceed the positive period. The named Lease uses whole seconds. |
| `--load-monitor-interval` | 10 seconds, positive. Load polling interval and the freshness window for a stream report to stand in for a poll. |
| `--shutdown-settle-secs`, `--shutdown-grace-period-secs` | 5 and 30 seconds. Grace is positive and includes settle. Settle is at most half of grace. Readiness drops before the listeners close. |
| `--decide-shards` | 1–256, available parallelism capped at 8 by default. Each model maps to one shard with room for 128 jobs. |
| `--runtime-worker-threads` | 1–4096 when set. Unset uses the async runtime's default. |
| `--log-level`, `--log-json` | `info` by default, also accepts `debug`, `warn`, `error`. `RUST_LOG` can set the filter. Text output by default. `NO_COLOR`, `FORCE_COLOR` and `CLICOLOR_FORCE` control text color, with force taking precedence. |

Startup rejects missing or duplicate scopes, duplicate listener names or addresses, unusable TLS material, and invalid lease addresses or timings. A client CA requires a verified certificate at transport level. This build has no certificate-to-principal mapping. Calls requiring a principal use a listener configured with `--identity-as-told`. A stated subject is a visible ASCII token of at most 253 bytes without commas. The service checks the scope before the principal. [Flags](src/service/controller/run.rs) · [Listener configuration](src/service/controller/config.rs)

### Worker discovery

A controller built with the `discovery` feature looks for its workers by itself. The feature is off by default, and the stock image does not build it yet.

```sh
cargo build --release --locked -p smg-policy-controller --features discovery
```

The binary then takes the gateway's worker discovery flags, under the same names and with the same defaults. `--discovery-provider` selects one of `kubernetes`, `file`, `slurm` and `consul`. `--service-discovery` still selects Kubernetes.

| Flags | Default and effect |
|---|---|
| `--discovery-provider` | One of `kubernetes`, `file`, `slurm` and `consul`. Unset by default: the controller runs no provider. Give this or `--service-discovery`, not both. |
| `--service-discovery` | The older spelling of `--discovery-provider kubernetes`. |
| `--discovery-check-interval-secs` | Seconds between periodic passes, 30 when unset. It must be positive. A start that failed is tried again after it. Refused with no provider. |
| `--api-key`, `--api-key-file` | The key sent to discovered workers, or a file that holds it alone. Refused with no provider. |
| `--selector` | Kubernetes: the label selector of regular worker pods, as `key=value`. |
| `--encode-selector`, `--prefill-selector`, `--decode-selector` | Kubernetes: the label selectors of encode, prefill and decode pods. Under any provider a prefill or decode one selects the prefill and decode pair, and an encode one beside them selects all three roles. The file and Consul providers refuse a record outside the selected roles, and Slurm refuses job names that do not fit them. |
| `--service-discovery-port` | Kubernetes: 80. The port of a discovered pod that has no `smg.ai/worker-ports` annotation. |
| `--service-discovery-namespace` | Kubernetes: the namespace to watch. Every namespace when unset. |
| `--model-id-from` | Kubernetes: take each worker's model id from pod metadata: `namespace`, `label:<key>` or `annotation:<key>`. |
| `--kv-connector-annotation`, `--kv-engine-id-annotation` | Kubernetes: `smg.ai/kv-connector` and `smg.ai/kv-engine-id`. The pod annotations that name a worker's KV connector and its engine ids. |
| `--discovery-file` | File: the JSON manifest that lists the workers. Required with `--discovery-provider file`. It may be created after startup. |
| `--slurm-job-name`, `--slurm-encode-job-name`, `--slurm-prefill-job-name`, `--slurm-decode-job-name` | Slurm: the job names whose nodes are regular, encode, prefill and decode workers. Each can be given more than once. |
| `--slurm-port` | Slurm: the inference port on every node of a discovered job. Required with `--discovery-provider slurm`. |
| `--slurm-bootstrap-port` | Slurm: the bootstrap port of discovered encode and prefill workers. |
| `--slurm-user`, `--slurm-account`, `--slurm-partition` | Slurm: discover this user's jobs (the user the controller runs as when unset), and only the jobs of this account or partition. |
| `--consul-service` | Consul: the service whose instances with passing health checks are the workers. Required with `--discovery-provider consul`. |
| `--consul-address`, `--consul-datacenter` | Consul: the HTTP API address (`CONSUL_HTTP_ADDR`, else 127.0.0.1:8500) and the datacenter (the agent's own when unset). |

```sh
smg-policy-controller --controller-id controller-0 --spec-file policy.yaml \
  --discovery-provider file --discovery-file /etc/smg/workers.json
```

- The replica that leads runs the provider. A standby runs none.
- The provider stops when the replica loses the lead, when its lease runs out, and at shutdown. The next leader starts its own.
- A provider that cannot start is logged as an error and tried again after the check interval. So is a provider that ends by itself.
- A Slurm or Consul flag, or `--discovery-file`, given with another provider or with none, is refused at the start. The Kubernetes flags are not. Under another provider the role selectors select the roles, as the table says, and the other Kubernetes flags, `--selector` among them, do nothing. A selector that is not `key=value` is still refused, and so is a malformed `--model-id-from`.
- The controller checks discovery settings as the gateway does and refuses bad settings at the start. A refusal of one setting's value names the flag that sets it. A refusal of settings that do not fit the roles names `--discovery-provider` and says that the roles follow the selectors and job names given. A Slurm bootstrap port with no encode or prefill job name is refused that way, and so is a Slurm job name given for two roles.
- The key sent to discovered workers comes from `--api-key` or `SMG_API_KEY`, or from a file named by `--api-key-file` or `SMG_API_KEY_FILE`. Each flag beats its own variable. A key and a key file together are refused, from any source. The key never prints.
- The key is read only when a provider is selected. A key flag with no provider is refused.
- The controller has no routing mode. A prefill or decode selector or job name selects the prefill and decode pair. An encode one beside them selects all three roles. Under Slurm a job's role comes from the job-name list that names it, and the names must fit the selected roles.
- `--router-selector` and `--model-alias` stay with the gateway.

One discovery limit stands today. [Discovery registers no workers yet](#what-is-not-built-yet). [Discovery flags](src/service/controller/discovery/flags.rs)

### Remote calls from a gateway

A gateway you host can call the standalone service. Use the generated client or the crate's `service::client::Client` or `service::client::blocking::Client`. The reference client dials and asks `Leader`, opens a session, sends one decision at a time, and reports `dispatched`, `completed` and `released` when its caller asks. Use `connect_with` for a configured TLS endpoint. For concurrent requests, clone a `service::client::Link`: it shares one session and one decide stream. An ask takes one credit and waits for its own answer inside its budget. Reports go through a bounded queue, and a full queue ends the link. The client exposes `welcome`, `term_floor`, `gateway_id`, `hashes_cut`, `session_end` and `watch_workers`. `Watcher` can apply worker events and resume positions to a local store. The host handles dispatch, fallbacks, retries, lease ownership and reconnection. There is no bundled operator CLI. [Reference client](src/service/client.rs) · [Worker watcher](src/service/watcher.rs)

| Scope | Call | What this build serves |
|---|---|---|
| Any gRPC listener | `Leader` | Reports the known leader, address and term. One lease covers all models. It takes no principal. |
| Gateway | `Session` | Start with `Hello` naming the caller and a protocol range containing 1. `Welcome` returns identity and limits. Its spec generation is 0 and it leaves needs and hashing unset. Send dispatch, completion, release or gateway veto frames. A new session replaces the earlier one for that gateway. Hello can restore vetoes. Lease and pin restoration via `HelloPage`, `SpecAck` and pushing new specs to gateway sessions are unserved. |
| Gateway | `Decide` | Bidirectional stream for a present gateway. Accepts single decisions and prefill entry requests that ask for the prefill/decode pair in one answer with a lease per side under the same ID. Under `epd` the served prefill entry still picks only the pair. A separate decode ask under either pair binding, or an encode ask under `epd`, gives `Busy` with an unspecified reason. The wire carries no encode item, so use the in-process door or an embedded engine for the full encode plan. |
| Gateway | `WatchWorkers` | Snapshot followed by membership and status changes, for all models or a named set. Resume from a term and sequence. A position outside the retained history can require a new snapshot. The service refuses `with_needs`. |
| Ingestion | `PushLoad`, `PushKvEvents` | Return `UNIMPLEMENTED` after admission. |
| Control | `RegisterWorker` | `ADD` (default), `REPLACE`, or `UPSERT` by URL. An upsert with changed role, wire or models makes a new incarnation. `wait_ready` requires probes disabled. |
| Control | `PatchWorker`, `DeregisterWorker` | Guard by worker ID and revision. Patch sets and removes labels and can replace overload thresholds. Deregister can mark draining, or remove. A drain needs a later remove. |
| Control | `GetSnapshot` | Leader's redacted `SLICE_WORKERS` and `SLICE_SPEC`. Name the slices explicitly. Empty requests ask for all ten and are refused. The spec slice includes retained generations with a store. |
| Control | `Explain` | Read a sampled decision by gateway and request ID on the leader. |
| Control | `ApplyPolicySpec`, `GetPolicySpec`, `RollbackPolicySpec` | Validate, read or change documents as described under [Updating a document](#updating-a-document). |
| Control | `StepDown`, `WatchRegime` | Return `UNIMPLEMENTED` after admission. |
| Replication | `Replicate` | Standby follows the leader's term, replays retained events or restores a snapshot. Automatic following uses the advertised leader host and the same fixed replication port on each replica, in cleartext with `--identity-as-told replication`. |

The controller exposes the gRPC health service on each gRPC listener. The admin listener serves `/liveness` and `/health` even while draining, `/readiness` (200 for leader and ready follower, 503 otherwise), `/version` and `/metrics`. [Controller protocol](contracts/policy_controller.proto)

### Holding a decision

A host that decides in process calls `DecisionGuard::decide` with its door. A host with the `client` feature can share a `Link` to a controller and ask for the same kind of handle.

```rust
let now = Instant::now();
let mut handle = DecisionGuard::<&StoreBackedEngine>::asked(&link, &request, now, budget).await?;
```

Give the controller time to decide in `request.facts.budget_micros`. On the wire, zero means that time is spent. A request built for the door states no budget with 0, so set one before you ask: the handle answers a zero budget `DecisionUnavailable::Late` before anything is sent. A request that states a hard demand (`Constraints::demands`) is answered `DecisionUnavailable::Unsent` before it is sent, since the wire carries no demand. The separate `budget` is how long the link waits for the answer. The handle counts the time to dispatch and the lease's dispatch span from `now`.

Use `take` to choose a target, then say `dispatched` and `completed`, or `released` if the request is given up. For a pair, use `dispatched_decode` for the decode and name each side in its completion. Dropping a handle gives up any sides still owed. A linked handle reports on the session, and the controller writes the completion on the worker that served. The controller's session rule covers leases when the link ends.

Linked targets carry worker ids and ranks. Candidate positions stay with the controller, and a shed reads `controller` as its stage name. An answer the handle cannot read returns `DecisionUnavailable::Unsent` and the link releases its leases as late. The caller handles an unanswered request.

### Following a controller's workers

With the `client` feature, a host can keep a copy of the controller's worker rows in a `MemoryStore` through a `Watcher`. Use `Watcher::open_on(&link)` to take the snapshot and the changes that follow it over the link's channel. The watch opens within the same five-second bound as `Client::watch_workers`. Once open, it has its own stream, so the link can answer asks and carry reports while the watch waits. Use one link for the gateway id. A second session with that id takes the first one's place.

A watch can outlive the link. A host that follows only while the link stands can wait for either to end.

```rust
let mut following = watcher.open_on(&link).await?;
let end = loop {
    tokio::select! {
        biased;
        () = link.closed() => break None,
        event = following.next() => match event {
            Ok(true) => {}
            result => break Some(result),
        },
    }
};
```

`None` means the link ended. `Some` gives the watch's own end. `Link::ended()` gives the cause once the link ends. A closed send queue can refuse an ask or a report before its stream reader sees the end. A full report queue ends the link at once.

Drop the stream before opening the watch again. The watcher asks for changes after the last event it took, or for a snapshot if it has no place. When a link ends, the host dials a new one. An ended link lets its channel go, and `Link::watch_workers` on it answers `Error::LinkEnded`.

When the watch starts from a snapshot, its last page puts the rows into use through `MemoryStore::watch_from`. The watch carries worker rows and their status. The store keeps its own load reports for workers still present.

For example, if the watch starts again while a request is running, its lease stays on file. If its worker is still present, the request still counts toward that worker's in-flight total and the worker keeps its gateway vetoes. If the worker has left, the lease ends when the host reports the request's end. The store keeps its session pins through the snapshot.

### Worker credentials

A host built with `worker-host` can hand its worker objects a keeper with `Objects::with_credentials`. The keeper has one entry per address an object dials, and each entry is bound to the rows that host's registrations wrote. If two rank rows name `http://worker:8000` as their engine, the keeper holds their credential once at `http://worker:8000`, bound to both rows.

The host calls `keep` before it writes a registration, with the facts it writes and `None` where the registration has no credential. `keep` returns a token. The host calls the token's `accepted` with the row the door answered, whether or not the write moved the row, and its `put_back` where the write was refused. It answers at the door's answer, with no await between the two. A token dropped with no answer is put back. Where either answer returns `true`, the host calls `Objects::lay_again` with the same facts, and the holder builds the address's objects again.

One registration is in flight at an address. A `keep` at an address whose last `keep` has no answer yet takes nothing and returns `RegistrationInFlight`. The host waits, or tells its caller to ask again.

A worker object takes its address's credential only where a registration of this host bound its row. A worker that another replica registered is built here with no key. So is a worker that takes an address after its last worker left: it is another row. While a registration is in flight, an object built at its address takes what it stated, and the answer settles it.

Registering the same worker again with a new key moves no row: the door answers `MembershipWriteOutcome::Unmoved` for equal facts. The host calls `accepted` all the same. It returns `true`, and `lay_again` builds the worker's objects with the new key.

The holder forgets by itself. Each of its reads drops the rows that left from the keeper, whoever wrote the removal, and an entry goes with its last row. The keeper lives in the host's process and starts empty at restart. Membership rows and their watches carry no credentials. The stock controller builds its worker objects without a keeper.

A host that keeps credentials builds its registration line with `worker_host::registration_into_store_keeping_credentials`. The line reads the keeper from the holder it is handed (`Objects::with_credentials`) and takes the store that holder follows. It returns `worker_host::NoCredentials` for a holder built with no keeper. The write keeps what each worker states ahead of its row, answers the keeper with the door's answer, and has the address laid again where the keeper says so. The write runs in the holder's turn: a replacement by id is refused as moved where another object stands under its URL by then, since a registration of another credential alone lays another object and moves no row. A holder takes a keeper or kept objects (`Objects::with_built_kept`), never both: the later call stands. `registration_into_store` keeps nothing.


## Extend it

### Register a Rust stage

Implement `Filter`, `Scorer` or `Picker`. Add a factory to `StageRegistry::builtin()` and pass it to `PolicySpec::validate_in`, `Controller::bind_with` or `service::controller::run`. Each stage name is unique across the catalog. A simple scorer factory accepts the document's `weight`. It runs once at registration, where its stage's name is checked, and at every validation, including a dry run. Keep its effects inside the stage it returns.

The example registers a scorer with `stages.scorer("pack", || Box::new(Pack))?`. It favors workers with more reported running requests. The document names it in `score: [pack]` and uses `not-overloaded` to steer away at the threshold.

```sh
cargo build --release --locked -p policy-controller-example
target/release/policy-controller-example --controller-id controller-0 \
  --spec-file examples/policy-controller/policy.yaml
```

A host that holds a store of its own, with no controller around it, validates a document over the same catalog. `AppliedSpec::validate_in(&document, &stages, &denied_headers)` returns the generation that `MemoryStore::apply` takes. The host numbers its documents with `generation`. The store takes a number above the one that stands, so it refuses a document with no number.

A scorer can take settings of its own from the document. Register it with `scorer_with`. The factory reads the entry's parameters into a type of its own and returns the scorer, or refuses a value.

```rust
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreferWorker {
    preferred: u64,
}

impl Lifecycle for PreferWorker {}

impl Scorer for PreferWorker {
    fn name(&self) -> &'static str {
        "prefer-worker"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (&row, score) in kept.iter().zip(out) {
            *score = if candidates[row as usize].id.0 == self.preferred {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

stages.scorer_with("prefer-worker", |build| {
    Ok(Box::new(build.params::<PreferWorker>()?))
})?;
```

```text
score:
  - least-requests
  - prefer-worker: { preferred: 89, weight: 7 }
```

- `weight` is the profile's and stays out of the parameters. A bare name hands the factory an empty map.
- The factory runs for each entry that names it, at every start, apply and dry run. A value it refuses with `StageError::bad` is reported at its field's path in the document. A parameter that fails to deserialize is reported at the entry's path.
- Derive `deny_unknown_fields` on the parameters' type. Without it a misspelled key is ignored and its default runs.
- The keys of a parameter map carry no order. The factory reads them in key order, and two files that differ in that order alone have one digest. A key written twice is refused.
- A stage is asked what it needs while the document that names it is built, and not at a decision. The engine keeps the answer, so a stage's `needs` must give the same answer every time it is asked.
- The scorer's `name()` must return the name it is registered under. `scorer`, `filter` and `picker` run the factory once at registration and refuse another name there. `scorer_with` runs no factory then, so validation refuses the name.
- A `scorer_with` factory can call `build.dispatch_credit()` for the profile's `DispatchCredit`: what the host sent each row since that row's last load report. The scorer reads a row's `CreditedDispatches { requests, tokens, unsized_requests }` with `read`. `tokens` holds the prompt tokens the sends stated. `unsized_requests` counts the sends that stated none, and the scorer chooses their price. The type is open to further numbers: code outside the crate reads the fields and matches them with `..`. A scorer that prices no send of unstated size reads the pair `DispatchCounts { requests, tokens }` with `pass(facts).counts(row)`. Where the read releases no sends, it reads two words for each row that holds sends, where the whole read reads three. A send is a dispatch the host reports. It stops counting at a report of the row sampled at or after it, at the profile's `sent_ttl_ms`, or at the row's removal. When a send arrives with 8,192 already held on a row, the tally retires the row's oldest batch, at any age.
- A `scorer_with` or `filter_with` factory can call `build.load_report_reader()` for the view's reading of a load report. A scorer keeps the `LoadReportReader` value, opens one pass a decision with `pass(facts)` and asks `of(row)` for a row's report. A filter calls `read(facts, row)` in `keep`. Under `view.max_report_age_ms` a report older than the limit reads as none, as it does for the built-in scorers. A report exactly at the limit is still read. A document that sets no limit reads every report. A stage that reads `row.load` itself gets the report at any age.

A filter can take settings from the document too. Register it with `filter_with`, and write its entry as the name mapped to its parameters.

```rust
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MinFree {
    slots: u32,
}

impl Filter for MinFree {
    fn name(&self) -> &'static str {
        "min-free"
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        facts.in_flight(candidate).saturating_add(self.slots) <= 8
    }
}

stages.filter_with("min-free", |build| {
    let rule: MinFree = build.params()?;
    if rule.slots == 0 {
        return Err(StageError::bad("slots", rule.slots, "above zero"));
    }
    Ok(Box::new(rule))
})?;
```

```text
filter:
  - healthy
  - min-free: { slots: 2 }
```

- A filter has no weight, so the whole map is the factory's. A bare name hands it an empty map.
- The rules of a scorer's parameters hold. The factory runs for each entry that names it, at every start, apply and dry run. A value it refuses is reported at its field's path, `profiles.<profile>.filter[<index>].<name>.<field>`. The keys carry no order, and a key written twice is refused.
- A filter registered with `filter` takes no parameter, and neither does a stock filter but `has-room`. Whatever a document writes under such a name is refused.
- `filter_with` runs no factory at registration, so validation refuses a filter that answers another name.
- The factory can call `build.dispatch_credit()` for the profile's `DispatchCredit`, as a scorer's can. A profile whose filter reads it may set `sent_ttl_ms`.
- A Rust caller that builds a `ProfileSpec` by hand holds its filters as `document::FilterSpec`. `"healthy".into()` reads as before. A `String` pushed or compared does not: use `.into()` and `FilterSpec::name()`.

A picker can take settings from the document too. Register it with `picker_with`, and write the pick entry as the name mapped to its parameters.

```rust
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PastFirst {
    rows: u32,
}

impl Lifecycle for PastFirst {}

impl Picker for PastFirst {
    fn name(&self) -> &'static str {
        "past-first"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        let from = (self.rows as usize).min(kept.len());
        Best.pick(facts, candidates, &kept[from..], &scores[from..], band, out);
    }
}

stages.picker_with("past-first", |build| {
    let rule: PastFirst = build.params()?;
    if rule.rows > 2 {
        return Err(StageError::bad("rows", rule.rows, "at most 2"));
    }
    Ok(Box::new(rule))
})?;
```

```text
pick:
  past-first: { rows: 1 }
```

- A picker has no weight, so the whole map is the factory's. A bare name hands it an empty map.
- The rules of a filter's parameters hold. The factory runs for each profile that names the picker, at every start, apply and dry run. A value it refuses is reported at its field's path, `profiles.<profile>.pick.<name>.<field>`. The keys carry no order, and a key written twice is refused.
- A picker registered with `picker` takes no parameter. Neither does a stock picker, but `best` and `best-with-temperature`, which the validator builds. Whatever a document writes under such a name is refused.
- `picker_with` runs no factory at registration, so validation refuses a picker that answers another name.
- A picker built by `picker_with` can resolve a signal's column through its build context, and its needs then set `signal_cells`. A picker registered with `picker` resolves none.
- A Rust caller that matches on `document::PickSpec` adds a wildcard arm: the type is non-exhaustive and holds `Added`.

A stage can read a named value that the document declares under `signals`. The factory asks the build context for the signal's column and the stage keeps it. In a decision the stage reads a row's value with `SignalColumn::read(facts, row)`. The read returns `None` when the host handed no cells, when the row has no value, and when the value is older than the signal's `max_age_ms`.

```rust
stages.scorer_with("least-pending", |build| {
    let pending = build.signal_column("pending_tokens")?;
    Ok(Box::new(LeastPending { pending }))
})?;
```

A filter registered with `filter_with` resolves a column the same way. A stage that keeps a column sets `signal_cells` in the `Needs` it declares. Validation refuses a stage that keeps a column without the need, and one that sets the need without a column. A name the document does not declare is refused at the stage's entry.

A host hands the values beside the rows: `RequestFacts::signal_cells` for a single pool, `PairPools::prefill_signal_cells` and `PairPools::decode_signal_cells` for a pair, and `EncodePool::signal_cells` for the encoders. A pair decision and an encode pair read each pool's cells: cells on the request are not read there. `SignalCells::new(cells, signals)` takes one cell per declared signal for each row, in the rows' order, with a row's cells in the order of the names. It owns its cells: a host keeps the one it built and refills it in place through `cells_mut()`, so no decision allocates for them. A value is finite: where a host has none, or its source states one that is not finite, it writes `SignalCell::NONE`. `AppliedSpec::signal_columns()` lists the declared signals in that order.

The store keeps the values for you. A worker's load report can state values by name, pushed or polled. The store keeps the ones the document declares, where the report states them under the declared unit, and a decision through the door reads each row's value beside the row. The two fixed token counts are offered under `kv_tokens_used` and `kv_tokens_capacity`, in `tokens`.

What to expect of a stored value:

- It stays until a newer one arrives or its `max_age_ms` passes. A report that leaves a name out removes nothing.
- A value is as old as its report. A value reported with its own age is that much older. The stock controller dates a polled report by the stamp its answer carries, else at the start of its poll.
- A row and its values are two writes. One decision can read a new row beside the values of the report before.
- Where one row's report holds several ranks, the row has a value only where every rank states the name under one unit. The ranks' values are then made one as `across_ranks` says, under the oldest sample among them.
- A new document keeps the values of every name it declares again under the same unit and the same `across_ranks`. A name whose unit or `across_ranks` changed starts with no value. A value a report made under the document before is dropped where its name changed so, and counted as `undeclared`.
- A report offers `kv_tokens_used` and `kv_tokens_capacity` where its source states them, zero too. A source that carries no count leaves the stored value, so declare `max_age_ms` on a signal that reads one. A rank with no token capacity offers neither.
- A worker that leaves loses its values.
- Values are not replicated. After a failover, a document that reads signals decides by each stage's fallback until the next reports arrive: about one report period.
- A metrics scrape states no value by name.
- A report's list is held to the worker crate's bounds before it is read: its first 64 entries, each with a name of at most 63 bytes, a unit of at most 31 bytes and a finite value. A name's first statement stands. The controller counts what its own poll leaves out in `smg_policy_controller_load_values_dropped_total`.

A host's own feeder can write a value through `SignalSink::signal_value`. It reads the declared columns from the sink first (`SignalSink::signal_columns`), once a report, as the load drivers do. Each write names the column the value belongs to and the unit the value is in. A value is taken where the document in force still declares that column with the same unit and `across_ranks`, the stated unit is the column's, and the value is finite and of magnitude at most 1e15, the bound a cost rule's own parameters are held to. The answer says what became of the write: `Written`, `Undeclared`, `WrongUnit`, `OutOfRange`, `NotAMember` or `Older`. The controller counts each outcome in `smg_policy_controller_signal_writes_total`.

A host that keeps no store builds the cells itself, as before.

The stock schema takes any stage name, written bare or mapped to an object, and lists the stock names in each entry's description. Validate an extended document against the host's catalog: the validator refuses a name the catalog does not hold, by its path. Stateful scorers and pickers can observe issue, dispatch, completion, release, membership and signals when the host supplies them. A stage that keeps a number of each worker's load report says so through `Lifecycle::keeps_loads`. The controller then tells it every report before a decision can read that report, through `Lifecycle::on_signal`. A stage that does not say so is told no report. In process the load signal names two more things: the worker's slot in the host's counts and the report's sample instant. A host that builds the signal itself fills both. A scorer of your own can state a pick's overlap too: implement `Scorer::states_overlap` and `Scorer::said`. Both have a default that says nothing. `said` is asked once a decision, of the first target alone. The standalone validator refuses stages that declare a need for the full lifecycle. [Runnable Rust example](../../examples/policy-controller) · [StageRegistry API](src/stages/registry.rs)

A gateway binary of your own can take the same catalog. Build it as the `stages` function of [`examples/policy-controller`](../../examples/policy-controller) does, set it on `server::Extensions` and start through the gateway library's `main_with`.

```rust
smg::stock_allocator!();

fn main() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let extensions = Extensions::new().stages(stages()?);
    main_with(extensions)
}
```

The library sets no allocator. Write `smg::stock_allocator!();` once at the root of your `main.rs` to run on jemalloc with the stock executable's server options. That one line is all: the gateway finds out at its start which allocator serves the process, and publishes the allocator's metrics where it is jemalloc. On MSVC and musl the macro expands to nothing. A build without the macro runs on the system allocator, or on an allocator of its own. The gateway then says once on stderr that the process does not run on the stock allocator and that the allocator's statistics are off. The check needs the `jemalloc-stats` feature, which is on by default, and is not made on MSVC or musl.

The gateway validates its policy document over that catalog at startup, where the policy controller is selected. A refusal stops the start and names what was refused. The gateway keeps its tenant header out of the document's view.

The document the routing flags mean names stock stages alone. To run a stage of yours, name it in a document of your own and start the gateway with `--policy-controller on` and `--policy-spec FILE`. The document replaces the routing flags: a routing flag typed beside it is refused by name.

The same gateway startup can take a worker connector. With an `Arc<dyn EngineConnector>` in `connector`, use `Extensions::new().stages(stages()?).connector(connector)`. The gateway uses it for worker probes and engine connections. Delegate to `BackendConnector::shared()` for workers that use the gateway's own client. Inference over gRPC or ZMQ requires that client. If the connection provides the engine's load reports, let the worker runtime be the only poller of that engine's load.

### Load a WebAssembly component

Compile a component for the `smg:policy` package, at the version the WIT file names. The `policy` world takes records. `policy-packed` takes byte blocks with the layout given in the WIT. The host refuses imports from WASI. It provides log, clock, counter and seeded random calls. A component implements `init`, a decision resource with filter, score and pick, and `observe`. A score answer can carry `tie-band` on the scores' 0 to 1 scale: rows within that band of the best score tie, and the host uses its default band where the guest sends none. The host keeps instances per decider and meters their calls.

This host function loads the repository's policy component, using the default session bonus, and makes a scorer with an in-flight-count fallback. Use the returned scorer in a `WeightedScorer` entry of a `Profile`.

```rust
use std::{path::Path, sync::Arc};
use smg_policy_controller::{
    stages::LeastRequests,
    plugins::shared::OpenDecisions,
    plugins::wasm::{Plugin, PluginSpec, Runtime, WasmScorer},
};

fn component_scorer(
    decisions: &Arc<OpenDecisions>,
    path: &Path,
) -> Result<WasmScorer, Box<dyn std::error::Error>> {
    let runtime = Arc::new(Runtime::new(decisions)?);
    let spec = PluginSpec::new("component-score", std::fs::read(path)?);
    let plugin = Plugin::load(&runtime, spec)?;
    Ok(WasmScorer::new(plugin.bind(), Some(Box::new(LeastRequests))))
}
```

Use one `OpenDecisions` for the host's plugin runtimes. It accounts for the decision budget across plugins and sides. `PluginSpec` takes bytes, an optional ABI pin, opaque config, memory cap (16 MiB per instance by default), frame budget, fuel calibration (1000 units per microsecond by default), log limit (10 lines/s at level 2 by default), exposure, needs, counters, event subscription, fallback policy and breaker settings. A frame budget of zero takes the decision's remaining budget. The loader refuses zero memory or fuel, a component that fails to compile or link, an unsupported world, ABI or import, and a failing `init`. `Plugin::check_memory_budget` can check a host's total memory limit per decider. A trap or invalid output discards the instance. [Component build and host example](../../examples/wasm/wasm-guest-policy) · [WIT](contracts/smg-policy.wit) · [Host configuration](src/plugins/wasm/config.rs)

### Call a gRPC plugin

Implement the `PolicyPlugin` service with the supplied proto. Its `Decide` stream starts with a hello and an acknowledgement with a protocol version (1 or 2) and supported stages. It handles decision frames by ID. An optional `Observe` stream sends ordered lifecycle events. A filter or scorer can start its call while native stages work. A picker bound on its own receives the combined scores at the pick. A scoring answer can carry `tie_band` on the scores' 0 to 1 scale: rows within that band of the best score tie. The host uses its default band when the field is unset. Use a Unix socket with a checked peer user ID, or TCP with mutual TLS and a pinned server name.

This host function connects to a plugin that's listening on a Unix socket under the same user and makes a picker. Use it as a profile's picker.

```rust
use std::{path::PathBuf, sync::Arc};
use smg_policy_controller::{
    stages::Best,
    plugins::grpc::{GrpcPicker, GrpcPluginSpec, Plugin, Runtime, Transport},
    plugins::shared::OpenDecisions,
};

fn socket_picker(
    decisions: &Arc<OpenDecisions>,
    path: PathBuf,
) -> Result<GrpcPicker, Box<dyn std::error::Error>> {
    let runtime = Runtime::new(decisions)?;
    let spec = GrpcPluginSpec::new("socket-pick", Transport::Unix { path, peer_uid: None });
    let plugin = Plugin::load(&runtime, spec)?;
    Ok(GrpcPicker::new(plugin.bind(), Some(Box::new(Best))))
}
```

`GrpcPluginSpec` takes the transport, opaque config, exposure, needs, counter names, event subscription, failure policy, breaker settings, the per-decision budget and the connect budget (100 ms by default). A frame sees the whole candidate view up to `max_candidates` (128 by default). Beyond it the stage falls back. Cold or unavailable streams fall back while opening in the background. Call `Plugin::warm` on a deciding thread to establish its stream in advance. A failed initial handshake or initialization can refuse loading the plugin. The host bounds its waits and frames. [Service contract](contracts/policy_plugin.proto) · [Host configuration](src/plugins/grpc/config.rs) · [Test service](tests/support/grpc_plugin.rs)

### Plugin inputs and fallback

Host code sets up both plugin kinds. A policy document can name a registered stage adapter through the host's catalog. It has no `plugins` field. A host building a profile in Rust can bind several adapters to one plugin binding. Catalog registration requires a distinct stage name for each entry, and each adapter takes its name from its plugin.

Exposure defaults to hashed tenant and session identities and no headers. A host can withhold identities, expose the clear form it holds, and expose selected allow-listed header positions. Request hashes are passed only when declared in the plugin's needs, up to 8190. Lifecycle events default to none. Scorers and pickers receive the subscribed events. The gRPC observe queue holds 1024 events. Gaps in the sequence identify lost events.

A decline takes the configured stage fallback and leaves a healthy instance or stream in use. A trap, invalid output, timeout or open breaker can also trigger fallback. Surviving positions must be unique and in range. Each scored row needs one finite value, which the host clamps to 0–1. The pick must be a permitted position. With no fallback stage, the default `Open` policy keeps all rows, gives a flat score or ranks the combined scores, respectively. `Closed` sheds at a filter or picker and still gives a flat score at a scorer. The breaker defaults to five failures in ten seconds and a 30-second cooldown. Its latency rule can also open it when moving p50 exceeds half of the decision budget or p99 exceeds the budget. Hosts can read `Plugin::stats` and declared `Plugin::counters` in code. [Shared plugin settings](src/plugins/shared/config.rs)

## Policy document reference

Use `apiVersion: smg.policy/v1` and at least one profile. Both YAML and JSON use the same types. Quote strings that look like numbers or booleans in YAML. A key without a value is null. Use `[]` or `{}` for an empty list or map. The [JSON schema](policyspec.schema.json) includes reserved fields and stock stage names. Use validation in the intended host to check which fields and stages can run.

| Document field | Meaning and limits |
|---|---|
| `apiVersion` | Required. Only `smg.policy/v1` is accepted. |
| `generation` | Unsigned 64-bit number, default 0. Zero asks the service to number it. See [Updating a document](#updating-a-document). |
| `profiles.<name>` | One to 65535 profiles. `filter` and `score` default to empty. `pick` is required. Stage mappings have one name. `sent_ttl_ms` is optional and positive. It sets how many milliseconds a send counts for in the profile's tally of sends (`StageBuild::dispatch_credit`). Left unset, a send ends at a report of the row sampled at or after it, or at the row's removal. The 8,192 sends a row holds bound the tally either way. A profile whose stages ask for no tally refuses it. Of the built-in stages, `least-load` asks for it, and so do a `least-cost` and a `has-room` rule with a `sent.` term. `resident-work` keeps a tally of its own, which the limit does not reach. See [Built-in stages](#built-in-stages). |
| `view.headers` | Up to 256 unique lowercase HTTP header names, in wire position order. Default empty. Denied names are `authorization`, `proxy-authorization`, `cookie`, `set-cookie`, `x-api-key`, `api-key` and `x-goog-api-key`. Embedded validation can deny more, such as the gateway's tenant header. |
| `view.block_size` | Positive fallback block size, in tokens, when a token request gives none. The prefix stages fall back to 16 if omitted. A text request states its own window and never reads this. |
| `view.boundaries` | Positive, strictly ascending token positions, default empty. Used by prefix keying and required for a `hash` index entry. With a prefix-keyed scorer every boundary and `prefix_tokens` must cover at least one whole configured block and fit the hash cap. |
| `view.max_block_hashes` | 1–8190, default 256. The generation can read fewer, down to zero when no stage needs hashes. It reads zero of a kind no stage declares: a text request is hashed only where a stage reads text windows, and a host asks `AppliedSpec::block_hashes_read` with the request's kind before it hashes. |
| `view.max_report_age_ms` | 1–3599999 ms, default unset. `least-load`, `resident-work`, `least-cost`, `has-room` with a rule and the load gates of `prefix-affinity` and `warmup-slice` use their unreported paths for older reports. A host-registered stage reads a report under the limit through `build.load_report_reader()`. One that reads `row.load` itself sees it at any age. Overload verdicts can still reflect older reports, and plugins interpret report age themselves. A report at the limit still counts. |
| `view.labels`, `view.zones`, `view.engines` | Unique names, default empty. At most 64, 256 and 64 respectively. Labels are `key` or `key=value` with nonempty sides. |
| `index.<model glob>` | Defaults to an empty map. An entry has `kind` (`tree` by default, or `hash`), `ttl_s` (180 seconds for a hash placement), `max_size` (67108864 characters for text or tokens otherwise for a tree), and `sweep_s` (120 seconds). Numbers are positive. These are settings a host can read for its index. The crate's own store reads `kind` and `max_size` for a pool whose profile reads sent prefixes, and none of the others. Under `tree` it bounds the model's record of sent prefixes at `max_size` over the block size, one placement a block a worker sent it: 4,194,304 placements at the defaults, about 277 MB a model. Under `hash` the pool keeps no record. Exact keys win, then the longest literal prefix or suffix, then `*`. Equal-length prefix and suffix globs are refused. |
| `signals.<name>` | Defaults to an empty map. Up to 64 names. A name is a lowercase letter followed by lowercase letters, digits or underscores, at most 63 bytes. An entry has `unit` (required, 1 to 31 ASCII characters with no space, compared byte for byte with the unit a report states) and `max_age_ms` (optional, 1 to 3,599,999: a value older than this reads as no value). `across_ranks` is optional: `sum` (the default), `max`, `min` or `mean`. It says how one row's value is made where the row's report holds several ranks. A document with no signals is unchanged, and so is its digest. |
| `plans.pd-pair.pairing` | `off`, `lenient` (default), or `strict`. Used for prefill/decode compatibility, including the pair in `epd`. |
| `bindings` | Ordered, first match wins. Empty works with one profile. |
| `explain.sample` | 0–4294967295. Zero is the default and disables sampling. |

### Bindings and plans

`match.model` defaults to `*`. It takes an exact ID, `prefix*`, `*suffix`, or `*`. `match.phase` can list `single`, `prefill`, `decode`, and `encode`. An empty list matches all phases. A multi-sided plan with a nonempty phase list must include each of its sides.

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

| `bindings[].plan` | Profiles the binding names | What the engine picks |
|---|---|---|
| `single` (default) | `profile` or `profiles.single`. | One worker in the requested role, with optional alternates. |
| `pd-pair` | `profiles.prefill` and `profiles.decode`. | A prefill worker and a compatible decode worker, in one decision. The decode hard filters first constrain which prefills have a partner. The decode profile then sees compatible workers for the selected prefill. If it yields no target the engine can try another prefill from the pick. |
| `epd` | `profiles.encode`, `profiles.prefill` and `profiles.decode`. | A pair with a prefill that can accept the encoders' output, then an encoder for each item over that prefill's compatible encoders. A request with no encode items needs no encoder. |

`bindings[].overload` defaults to `enabled: true`. Disable it to turn off binding-level thresholds. A worker registration can still set thresholds of its own. `waiting_requests` defaults to 8 and must be positive. `token_usage` defaults to 0.8 and must be above 0 and at most 1. The store judges the sum of queued requests and mean active KV usage over an engine's ranks. A report without that split carries its total usage there. Reaching either limit sets the veto. `shed_when_all_vetoed` defaults to false and controls the single-plan fallback and plain ring behavior. For a decode pool used by a pair, the binding selected at prefill supplies its overload policy.

`bindings[].sessions` configures sticky placement under `single`. `bindings[].routing_key` names the sources of a request's key, for a ring or a sessions rule. See [Session keys](#session-keys). A multi-sided plan refuses a sessions rule or a picker that places by session assignment.

A `single` binding can name a second profile with `shadow: next`. Request handling runs the live profile, and only its decision serves the request. The in-process door decides the shadow beside each live decision of that binding, on the caller's thread and ahead of the decision's commit. The shadow reads the rows, the counts and the session pins the live decision read, so a twin of the live profile agrees where its stages keep no state of their own. A stage that keeps one, as a rotation does, moves only where its shadow is asked. It issues no lease and tells no stage.

The door counts what each shadow said, by the binding's position in its document and one of four kinds (`StoreBackedEngine::shadowed`):

| Kind | Meaning |
|---|---|
| `agreed` | The same first target, or the same outcome without a pick. |
| `other-target` | Both picked, and the first targets differ. |
| `other-outcome` | One picked, shed or queued where the other did something else. |
| `skipped` | The shadow was due and not asked. |

The count is a position's: a later generation's binding at the same position counts on, and a binding past the sixteenth counts with the sixteenth.

A shadow is asked over the rows its own profile would be handed. Where that profile is named cached-block rows of its own, the door names them again for the shadow, from the same load and the same instant as the live decision. It names them once: a live naming looks again past a sample full of refused holders, and a shadow's naming does not. So `cache_aware` is shadowed behind `least_load` or `random`, and so is a shadow of another sample than the live profile's. A pool keeps one warm-up rule and one sent-prefix record. They are the live profile's. Where the live profile is named no rows, they are the shadow profile's: each lease records the live decision's worker under the prefixes the shadow's naming folded, so the shadow reads where the requests went. A request whose shadow is not asked, as under the budget's floor, leaves no entry there, so that shadow can read colder than its profile would live. Behind a live profile that is named rows of its own, a shadow of another warm-up rule or another record is skipped and counted. A model keeps one sent-prefix record, so a shadow named rows of its own is also skipped and counted where another binding's live profile reads that record for a model the shadowed binding decides. The engine reads this from the document alone: the bindings' model patterns, phase lists and order. A shadow profile that reads no cached blocks is asked over the whole pool, as it decides live.

A request with less than 1 ms of its stated budget left skips its shadow and keeps the budget for its answer. A request that states no budget is shadowed. A shadow that panics costs its request nothing: the live answer leaves as decided and the door's fault count (`StoreBackedEngine::faults`) rises by one. A shadowed request waits for one more decision. A document that names no shadow decides as before.

Use native stages and a profile that no binding uses live. A served controller does not ask a shadow yet.

A host that drives the engine itself runs the shadow's stages through `Engine::shadow`. It makes the request's live decision through `Engine::decide_keeping`, and calls `Engine::shadow` with that decision's scratch. Under a sessions rule the shadow then starts from the session pins that decision read, whether or not the decision is committed first. Behind `Engine::decide` or `Engine::decide_stated`, or with any other scratch, it reads the pins as they stand. A host that shadows a request decides it through `Engine::decide_keeping`, each time it decides it. A debug build stops on a mix of the two entries on one request.

The validator checks reserved fields for shape and refuses their active use. This includes `match.tenant`, `match.priority`, `match.header`, `match.prompt_tokens`, a nonempty `match.labels`, `canary`, `override: allowed`, `admission.enabled: true`, and `sessions.issue_session_id: true`. The bounds use `min` and `max`. A header matcher uses `name` and `values`. A canary would name a `profile` and exactly one of `percent` (1–100), `tenants` or `header`. Use `override: locked` (the default) and leave admission and session ID issuance off.

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

An embedded host using the store can supply the runtime and a KV handoff descriptor. `lenient` rejects known differences. `strict` also rejects missing runtime or transport, a value missing on one side for a hard fact, and different versions when both are provided. Both sides may omit a KV layout fact. A shared explicit protocol settles the handoff after the runtime check. `off` skips descriptor checks. In every mode, HTTP-only workers pair within their wire class and workers on the other side of that division cannot pair if their known runtimes differ. The store derives compatibility from membership and rebuilds it when the pairing mode changes.

Over the standalone registration RPC, the pairing descriptor accepts only a bootstrap port, and the runtime comes from `spec.runtime`. Use `lenient` or `off` for workers registered this way. A host supplying its own admission facts can also add `has-capacity` and `has-room` to the profiles.

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

Use this document with `StoreBackedEngine::decide_encode_pair`, or `DecisionGuard::decide_encode_pair` to hold the decision. Pass the request's `EncodeItems` and an `EncodePlaced`. The door reads the worker offers and the compatibility tables from the store. A host that decides over its own rows calls `Engine::decide_encode_pair` and supplies those as well. The encode profile can key each item by content hash. A push handoff can place by one supplied key. Images require a vision-capable encoder, and a request can require a vision-capable decode worker. The decision writes encoder choices to `EncodePlaced`. All choices are made before dispatch. [Encode entry and refusal names](src/engine/framework/encode.rs)

### Session keys

The host resolves a key and supplies it as `RequestFacts::session_key`, or as `facts.session_key` on the wire. The header fields specify the host's inputs. Sending the header in a `DecideRequest` alone does not create a key.

| `sessions` field | Default and behavior |
|---|---|
| `sticky_header` | Required without `routing_key`. It must be listed in `view.headers`. |
| `headers` | Empty uses the sticky header alone. A nonempty lookup list must contain it. Names must be unique and allow-listed. |
| `assignment` | `delegate` (default) uses the profile. `random` draws, `min_load` uses host requests in flight, and `min_group` uses active session counts. With `manual`, `delegate` uses `min_load` and a keyless request follows the same assignment without a pin. |
| `ttl_s` | 14400 seconds. Positive. Use 4294967295 for no expiry. |
| `sweep_s` | 120 seconds. Positive. With finite TTL, the sweep must be finite. |
| `inflight_cap` | 2. Positive. 65535 disables it. At the cap the session rule tries a fresh placement. A pin moves only to another worker below the cap. The cap reads the key's counts. The door and the standalone controller hold them from their leases, and a host that decides over its own rows feeds them. |
| `recovery` | `original` (default) prefers the first pin when both pins are usable again. `latest` prefers the most recently dispatched pin. |
| `issue_session_id` | False. True is refused. |

The stock controller counts a key's requests by the leases it holds. A request counts on the worker its lease books, from its decision to its end, over every gateway the controller serves. Decisions made at the same moment for one key each read the count before the others' leases are on file, so a burst of them can pass the cap. A request decided while its rule had no cap is not counted, so after a change that sets a cap a key can pass it until those requests end.

**When a worker leaves.** A worker that leaves membership keeps its keys' pins for the absence's time: the longest `ttl_s` among the document's session rules. A rule that names none counts as four hours, and so does a document with no rule. Meanwhile its keys read as not routable, and the rule places their requests. A worker registered again inside that time as the worker it was (the same role, models and wires) takes its pins back under its new id, and `recovery` then rules as for two usable pins. Another worker under the same registration key takes none. A worker that stays away is forgotten at the absence's end, by the store's tick or the next keyed decision. The table and the store each keep 4,096 absent workers at most. A standby, and a gateway whose store watches a controller, take the same steps at the writes they follow, and count the absence by their own clock.

**Where the key comes from.** A binding names the sources of its requests' key under `routing_key`. The host reads them in order, and the first that holds a value is the key. A ring places by that key, and a sessions rule pins it. An entry marked `client` is read apart, for a request the other entries give no key.

```yaml
apiVersion: smg.policy/v1
profiles:
  consistent_hashing:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: "*" }
    profile: consistent_hashing
    routing_key:
      sources:
        - header: x-smg-routing-key
        - rid: true
        - header: x-forwarded-for
          client: true
        - header: cookie
          client: true
      strip_lineage: true
```

| Field | Default | Meaning |
|---|---|---|
| `sources` | required | One to eight entries, read in order, and up to four client entries beside them. A block of client entries alone is refused. An entry is `header: <name>`, or `rid: true` for the request's own id. |
| `max_bytes`, on an entry | 128 | The most bytes of a value the source hands, 8192 at most. A longer value is passed over. So is an empty one, and one that is not UTF-8. |
| `client`, on an entry | `false` | The header tells one client from another, not one session. Its value places a request that has no session key on a plain ring, and does nothing else: it lifts no overload veto, pins no session and is counted under no key. The first client entry whose header the request carries is read whole. A value that is empty or not visible ASCII gives no key. Refused beside `rid` and `max_bytes`. A header may be named both ways, once without `client` and once with it. |
| `strip_lineage` | `false` | Take the turn and retry suffixes off a key: `conv_t2_r1` keys as `conv`. A header is measured against `max_bytes` as sent, and the request's own id without its suffixes. A suffix's counter has any number of digits. |

A source need not be in `view.headers`: its value enters no view. A credential header is refused. A key from a header that `view.headers` does not list, `cookie` for one, leaves the host as its fingerprint alone, so no stage, plugin or record sees its text.

A binding names its sources once. With `routing_key`, a sessions rule leaves out `sticky_header` and `headers`. Without it, the rule's own headers are the sources. A binding with neither gives its requests no key.

A host reads a binding's sources with `AppliedSpec::key_sources_for` and derives a request's key with `KeySources::derive`. A host in process does both inside the door: it implements `engine::RequestFromSnapshot` and calls `StoreBackedEngine::decide_under` or `DecisionGuard::decide_under`. The door hands `RequestFromSnapshot::make` the world the decision is made under, and with it the applied document. The key's sources and the rule that pins the key are then of one generation, whatever is applied meanwhile. The maker is asked once for each run of the decision: a second time only where the store refused the first run's lease. A host derives the client's key with `KeySources::derive_client` and states it as `RequestFacts::client_key`. The wire's request facts carry it to a standalone controller as `client_key`, so a controller at an address places a request with no session key on the plain ring by its client's key, as the host's own process does. A plugin's frame carries zero in that field under every exposure.

A plain ring under `single` can keep a key on an otherwise usable worker across the overload veto. This requires the sole scorer to be `consistent-hash` without prefix or load bound, bare `best`, and a steering binding. Any session rule must delegate and either have its cap disabled or have the key's counts held. The door and the standalone controller hold them from their leases. A host that decides over its own rows sets `session_counted`. With shedding enabled, a vetoed owner passes the request to a usable pin or ring successor if one exists. With a bounded prefix ring, an owner strictly above both load bounds is passed over. The mean includes the incoming request. Where boundaries supply a shallower key it is tried before the profile's keyless scoring path.

## Guarantees and limits

### Decisions and refusals

A pick's first target is booked at issue by the door or service. The caller can dispatch an alternate and report which target it took. Report completion or release so the lease can end. An undispatched lease expires after five seconds. A repeat marked `again` ends an earlier lease still waiting for dispatch. The service checks the leadership term and budget as it makes the decision and checks the term and lease at send. Decisions read one published world and keep the issuing engine for lease events. A new generation builds fresh stages. Outstanding leases still belong to their issuing engine while the store holds them.

- A single profile's empty pass can rerun filters with the overload veto lifted, then with liveness doubts lifted. It favors fewer in-flight requests on fallback. `shed_when_all_vetoed: true` keeps overload in force and can still lift liveness alone. Draining, not-ready and breaker holds stay refused by the health check. Split plans shed at their own vetoed sides without this single-plan fallback.
- Unmatched bindings, no endpoints, an empty filter pass, a declined picker and incompatible pair or encode workers are sheds. The answer says the reason and the stopping stage. The built-ins emit no queue tickets. Hosts decide how to map sheds to request status and retry behavior.
- `Decide` has a 64 KiB incoming frame limit, including unknown fields. It requires a request ID of 1–256 bytes, known phase and hash kind, and at most 256 headers with positions in 0–255. A nonzero session hash must decode to a nonzero fingerprint for its model and phase. Zero means no key. A malformed or oversized frame can end the stream. The client cuts tail hashes to fit or returns an error when cutting them cannot make the message fit.
- `Welcome` grants 128 credits per decide stream and up to 256 streams per gateway. Requests spend credit. A credit frame replenishes it. `Busy` reasons are `queue_depth`, `no_credit`, `budget_gone_on_arrival`, `not_leader` and `unspecified`. They cover a full shard queue, exhausted stream credit, expired budgets or dispatch span, a fenced replica, and cases such as raw prompt text, an unsupported side, a decider fault or exit, spent lease IDs or a document still awaited. The caller owns any local fallback.
- The session expects `Hello` within five seconds, as its first frame, with a matching principal and protocol 1 in range. A gateway can keep one current session and get ten seconds of grace after disconnect. The table holds up to 4096 gateways, up to 4096 vetoes per gateway and eight kinds per worker per gateway. A `Hello` veto count must match the list. An unspecified veto kind, excess vetoes or an invalid lifecycle frame can end the stream. Old terms and replaced sessions can end calls. The reference client applies its term floor to decision and session answers. It waits up to five seconds per operation step, with a separately streamed watch.
- Workers registered over gRPC take one rank and one HTTP or gRPC wire. A new row must supply its role and wire. The service refuses a host-local wire, rank width above one, nonempty topology, or extra pairing descriptor fields. It accepts at most 128 labels per row, a 2048-byte URL, 256-byte names for model, runtime and zone, 128-byte label keys, 4096-byte label values and a 16 KiB encoded row. No count bounds a row's model IDs: the row's size does. The URL and model IDs cannot contain blanks or control characters. A scrape needs an interval and uses the registered runtime's parser. Probes default to three failures, two successes, a 60-second interval and a five-second timeout. Duplicate adds, missing IDs or URLs, stale revisions and another maker's row have separate statuses. A registration over a draining row is the URL's next incarnation, whatever its facts: a drain ends with the registration it named, and only a patch or a second drain keeps it.
- `WatchWorkers` allows four watches per principal. It can require a fresh snapshot when history no longer matches, and ends a stalled watch when polling resumes after 30 seconds. Worker watch pages hold up to 128 rows and 2 MiB of row data. Watches and `GetSnapshot` accept up to 128 model IDs. `GetSnapshot` can cap the worker rows returned and takes one snapshot at a time with up to 32 waiting. It refuses unknown or unavailable slices, invalid filter fields, and clear identifiers. `Explain` refuses a decision ID or an incomplete key and returns `NOT_FOUND` on a miss.
- A shared lease keeps rings of 4096 watched events and 65536 load events for replication. History gaps require a snapshot. The leader holds up to four snapshot restores at once and can release a stalled restore after 60 seconds.
- A call on the wrong scope returns `PERMISSION_DENIED`. A scoped call without a principal returns `UNAUTHENTICATED`. Leader-only operations can return `FAILED_PRECONDITION` with leader details, or `UNAVAILABLE` during a term transition. Unserved calls return `UNIMPLEMENTED` after admission.

### When a gateway loses its stream

A gateway asks the stock controller for decisions on a stream. A pick carries a **lease** that books its worker, or both sides of a pair under one id. A side that no dispatch names within five seconds is given up.

A stream can end with requests unanswered. Two rules keep one lease standing for one request.

- **The stream's end.** A decide stream that ends gives up the lease of every decision it had not sent yet. No gateway could use them.
- **A request asked again.** A gateway whose stream ended sends each unanswered request again and marks it `again` (`DecideRequest.again` in [`contracts`](contracts)). The controller ends the lease of the earlier ask at once, where it still waits for a dispatch, and decides the request anew.

What a gateway must hold to:

- The mark is for a request whose stream has ended, and the gateway reads nothing more from that stream. On a stream that is still open the earlier answer can still arrive, and it would carry a lease the mark has ended.
- A request sent twice without the mark is two requests and two leases.
- A repeat that the stream answers `Busy` before a decider takes it ends nothing. The next ask of that request carries the mark again.
- The earlier lease is found by a 64-bit fingerprint of the request's id, among that gateway's own leases.

Two counters tell what happened, on the admin listener's `/metrics`:

| counter | what it counts |
|---|---|
| `smg_policy_controller_decide_asked_again_total` | requests marked `again`, by `earlier_lease`: `ended` where the earlier lease still waited, `none` where no lease of the request waited |
| `smg_policy_controller_decide_answers_dropped_total` | answers no stream took, by `cause`: `stream_gone` where the stream had ended, before the answer was made or with it still in the stream's queue, and `queue_full` |

After a stream's end most repeats read `none`: the stream gave the lease up, or the first ask was never decided. That is no error.

### Updating a document

**File.** Use a path that leads to a regular file of at most 128 KiB. The reader follows symlinks. A `.json` suffix selects JSON. The controller checks the file at startup. The watch polls after one second and reads a changed identity after two matching looks. A rejected update leaves the running generation in place and is logged and counted once per settled file state while leading. Unchanged documents count as neither apply nor rejection. A fast same-size edit can escape detection if the file identity does not move. An update that stays unstable can be rejected.

**Store.** `--spec-store` uses an existing directory with a lock and support for hard links. It runs with a process-local lease. Startup loads the newest file or starts with no document in an empty store. An empty store can report ready and shed requests until the first apply. Enforce persists before publishing and retains the newest eight files normally. A competing start waits up to the shutdown grace for the lock. Startup runs the newest file under the number in its name, also where the host's store already runs that document under a lower number. If the store refuses a generation after its file is written and the file cannot be removed, the apply fails as `not persisted` and says both. The store removes that file before its next write. A process that starts before that write reads it as a generation on file, and runs it if it is the newest.

**Generation.** A zero or omitted number starts at 1 and increments on change. A file can claim a higher number. Enforce through the store takes the next number. Changed content under the standing number, a stale number, or an out-of-order store number is refused. An identical document with the standing or automatic number is a no-op. The digest is a canonical `sha256:` digest of parsed content without the generation line.

**Calls.** `ApplyPolicySpec` requires a mode. Use `yaml` or `json` for the document format. `DRY_RUN` on either source checks the candidate under that source's rules. `ENFORCE` writes only under a store. Both can take the current `expected_digest` as a guard. A verdict can include before and after needs, or one issue with a field path when one applies. `AUDIT` is unserved. `GetPolicySpec` returns this replica's standing document, or a generation by number if held. It returns `NOT_FOUND` when none is held. `RollbackPolicySpec` under a store reapplies a named retained generation using a new number. Enforce and rollback under a file return `read_only` before document validation. Each dry-run and write queue has room for up to 32 waiting. Both use the same 128 KiB document cap as the file, with a separate bound on decoded YAML expansion.

**In process.** A host that embeds the store takes the same door with the crate's `operator` feature. `document::operator::DocumentManager::new(store, catalog, denied_headers)` makes it. `apply` takes the host's document as bytes, YAML or JSON. It checks the document over the host's catalog and denied headers, and runs it. A changed document that leaves `generation` out runs as the next generation. One that states a higher number runs under that number. `dry_run` checks a document and applies nothing. A refused apply counts once in `apply_counts`, and a dry run counts nothing. `document::operator::MAX_DOCUMENT_BYTES` is the 128 KiB bound, and `Format::named` reads `yaml` or `json`. `status`, `document` and `standing` read the generation that runs, and `apply_counts` the applies. `EnforceRefused` and `NotHeld` are errors a host can print. `NotHeld` says what was asked, what is held, what stands and what a spec store has on file. `document::operator::snapshot::worker_rows_page` gives the worker rows an operator reads. An address is told without userinfo, and so is a provider's key that is a URL. A provider's key that is no URL and holds an `@` reads as `<unparsable address>`. No spec store is bound in process. `writable` says so, and `enforce` and `roll_back` are refused there with `not persisted: no spec store is bound`. The spec store itself builds on Linux and macOS. The gateway is such a host. With its policy controller on, it applies its policy document through this door at startup. `GET /policy/spec` on its control plane returns that document with its generation and digest, and `POST /policy/spec:validate` runs a dry run of the body against it. Both need the `admin` role, and nothing is applied in the gateway after startup.

**Shared lease.** A standby follows the leader's document if replication is configured and working. It validates its own file at startup and watches without applying it while following. On promotion it may apply its file after it settles. A file already observed running since the last change is skipped. An unnumbered document the standby never saw run can take effect on promotion even if its file is older. An explicit generation keeps a stale number from taking over. A new leader awaiting its first document can return `Busy`. File lease renewals are durable writes. With the default timings, a slow renewal can stop decisions, close sessions or lose the lease. For slower storage, the lease flags can give more time, for example `--lease-duration-ms 15000 --lease-renew-deadline-ms 10000 --lease-renew-period-ms 2000`.

**Sends.** A profile's tally of sends (`StageBuild::dispatch_credit`: what the host sent each row since the row's last load report) carries over to the profile of the same name in the next document, when both have a stage that reads it. The two generations share the entries. Each reads them under its own `sent_ttl_ms`, and a send that one read releases has left the other too. A `least-load` in the new document keeps the dispatches its profile's name held under the standing one, so its estimate does not restart at a new document. A profile under a new name starts empty. A changed `sent_ttl_ms` applies from the next read. A refused document takes nothing. `MemoryStore::apply` makes the carry before it publishes. A host with a store of its own calls `Engine::succeed` where it swaps engines. A dispatch reported for a worker that the store no longer lists at that rank adds nothing to the tally.

The validator refuses unknown document fields and a non-null `kind`, wrong API version, malformed or duplicate map keys, invalid names or ranges, missing or conflicting bindings, missing plan sides, unsafe headers, unknown stages, unsupported active fields and stage factory errors. The one exception is the parameters of a scorer you register: its own type reads them, and must deny unknown fields to refuse one. A refusal can be about the whole document. The validator messages in these two examples also appear in startup or apply errors.

<!-- refused -->
```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [ready]
    pick: random
```

```text
profile "default": unknown filter "ready"
```

<!-- refused -->
```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    score: [least-load]
    pick:
      best-with-temperature: { temperature: 0.5 }
```

```text
profiles.default.pick: the best-with-temperature picker is not in this build
```

### Explain and metrics

Set `explain: { sample: N }` to keep every Nth pick or shed per decider. Zero turns it off and one keeps every pick or shed. Each decider allocates its 1024-slot ring on its first sampled write and overwrites its oldest records. With sampling on, a decision takes its decider's ring lock only when it keeps a record. That write can wait for an `Explain` read of the ring. At R eligible decisions a second on that decider, a full ring covers about `1024 × N / R` seconds. A new generation restarts counting. Older samples can still be read under the same term. The remote lookup uses `gateway_id` and `request_id`. Use the subject the gateway authenticates as on the gateway listener for `gateway_id`, because `Explain` hashes it the same way as the decide stream. Use `StoreBackedEngine::explain` in process with the store gateway ID. `StoreBackedEngine::explained` returns that record with its names: the plan, the profile, and that profile's picker, filters and scorers in their declared order. The names are `None` once another generation stands, or when the request matched no binding. A shed made ahead of every profile names the plan alone. A subset hint of zero on a pool with members is one example. In process, `Trace::made_under()` gives the profile's position, or `None` for such a shed. On the wire, read `trace.no_profile` first: when it is true, the decision was made ahead of every profile and `trace.profile` is 0. A term change makes old samples unavailable on the service.

A record echoes the lookup key and gives the decision and its trace without a lease. It keeps up to three top scores. A sampled record of a shadowed decision also holds the shadow's kind (`shadow`) and the shadow's first target (`shadow_target`). The control call keeps the record's original binding position and trace. `binding` is 0 when no binding matched or when the first binding matched. The name fields are empty for unmatched decisions and records from older generations. It leaves the decision ID, full candidate list, view digest, plugin projections and lifecycle outcome unset. It omits request facts, headers and session keys.

The admin `/metrics` endpoint exports the following series. A spec source adds the two spec series.

| Metric | What it measures |
|---|---|
| `smg_policy_controller_state` | One gauge per `state` (`starting`, `leader`, `syncing`, `follower`, `draining`). The current state is 1. |
| `smg_policy_controller_start_time_seconds` | Startup time since the Unix epoch. |
| `smg_policy_controller_replication_synced` | 1 for a synced standby. |
| `smg_policy_controller_replication_lag_seconds` | Seconds since a frame last left the standby at the leader's head, or 0 when leading or before such a frame. |
| `smg_policy_controller_replication_restores_asked_total` | Standby's restore requests. |
| `smg_policy_controller_load_polls_total` | Poll arbitration per engine and interval, by `mode` (`poll`, `fallback`, `suppressed`). |
| `smg_policy_controller_load_poll_failures_total` | Poll failures by `reason`. |
| `smg_policy_controller_load_scrapes_total` | Successful HTTP load scrapes. |
| `smg_policy_controller_load_scrape_failures_total` | Scrape failures by `reason`. The poll and scrape label values are `transport`, `answer`, `deadline`, `no_load`, `unsupported`, and `client`. |
| `smg_policy_controller_load_rows_total` | Rank load rows written by the load driver. |
| `smg_policy_controller_load_values_dropped_total` | Named values that the list bounds removed from the load driver's polled answers. |
| `smg_policy_spec_generation` | Standing spec generation. |
| `smg_policy_spec_apply_total` | Applies by `result` (`applied` or `rejected`). |
| `smg_policy_controller_decide_busy_total` | Busy answers by wire `reason`. |
| `smg_policy_controller_decide_no_decision_total` | Unspecified Busy by `cause` (`fault`, `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`, or `document_awaited`). |
| `smg_policy_controller_decide_hashes_cut_total` | Requests whose hash list exceeded what the generation reads. |
| `smg_policy_controller_decide_answers_dropped_total` | Dropped answers by `cause` (`stream_gone` or `queue_full`). |
| `smg_policy_controller_decide_asked_again_total` | Requests marked `again`, by `earlier_lease` (`none` or `ended`). |
| `smg_policy_controller_decide_candidates_undialed_total` | Candidates removed by the asking gateway's wire constraints. |
| `smg_policy_controller_decide_budget_outlasted_total` | Decisions cut at the final decision check because the budget or dispatch span was too short. |
| `smg_policy_controller_members_outside_list` | Members not classified by a configured list, by `list` (`engines` or `zones`). |
| `smg_policy_controller_overload_flips_total` | Load writes that changed an overload verdict. |
| `smg_policy_controller_lifecycle_faults_total` | Panics contained while applying dispatch, completion, release or dispatch expiry. |
| `smg_policy_controller_signal_writes_total` | Named values written to the store, by `outcome` (`written`, `undeclared`, `wrong_unit`, `out_of_range`, `not_a_member` or `older`). |

Plugin stats are available through the host APIs. Both kinds count answers, declines, exhausted decision budgets, breaker bypass and openings, stalls and observed events. WebAssembly also reports traps, fuel use and exhaustion, invalid replies, overlong calls, instantiations, lost events and unread timing. gRPC reports plugin Busy, invalid replies, overlarge pools, cold or refused frames, misses and their attribution, connections, stream failures, late answers and dropped observe events. Declared counters are separate from these host stats. [WebAssembly stats](src/plugins/wasm/plugin.rs) · [gRPC stats](src/plugins/grpc/plugin.rs)

## Gateway routing metrics

The embedded host counts each ask and times its call to the controller. Each ask counts as
`pick`, `shed`, `queue`, or `refused`. The `mode` is `embedded`. The `model` is the
resolved model. The `profile` is empty on a refusal, and for a request that is shed ahead of
every profile, as when no binding matches its model. It is also empty when the host needs a
name after that generation has been replaced. When a commit finds no sendable target, each
subsequent ask counts on its own.

The embedded column gives the gateway family to query. `Same` carries the classic names over.
The last column also records families whose embedded writers are pending. Four names in it
are planned, and no code writes them yet: `smg_policy_lease_expired_total{kind}`,
`smg_policy_leases{kind,state}`, `smg_policy_fallback_total{reason}` and
`smg_policy_deadline_misses_total`. The shipped dashboard uses the classic names. Decision
series appear after their first observation. The standalone controller keeps the
`smg_policy_controller_` prefix for its process metrics, including role, replication,
election, and load polling.

| Classic gateway | Embedded gateway | Class and plan |
| --- | --- | --- |
| `smg_worker_selection_total{policy}` | `smg_worker_selection_total{profile}` | Kept. The embedded profile label comes with the router's send path. |
| `smg_worker_overload_shed_total`, `smg_worker_overload_fallback_total`, `smg_worker_liveness_fallback_total` | Same | Kept. The embedded writers will use the decision's outcome and lifted veto. |
| `smg_pd_admission_waits_total`, `smg_pd_admission_sheds_total`, `smg_pd_admission_window`, `smg_pd_prefill_admission_inflight`, `smg_pd_prefill_admission_queued`, `smg_pd_prefill_admission_rejections_total`, `smg_pd_prefill_admission_wait_seconds` | Same | Kept. The gateway owns admission. |
| `smg_routing_key_source_total{source}` | Same | Kept. The gateway reads `header` or `rid` as the source. |
| `smg_cache_aware_policy_branch_total`, `smg_manual_policy_branch_total{branch}`, `smg_consistent_hashing_policy_branch_total`, `smg_prefix_hash_policy_branch_total` | `smg_policy_decisions_total{model,profile,outcome,mode}` | Renamed. The new family groups by outcome. The standalone controller's `smg_policy_controller_decide_no_decision_total` corresponds to `outcome="refused"`. |
| `smg_policy_inflight_reconciled_total{policy}` | — | Planned rename to `smg_policy_lease_expired_total{kind}`. No code writes the new name yet. |
| `smg_cache_aware_match_ratio`, `smg_kv_index_lookup_seconds`, `smg_cache_placement_entries` | Same | Moved to the process that holds the index. Match ratio is observed only when the scorer reports the chosen worker's overlap. Embedded observations from the index are pending. |
| `smg_cache_tree_chars`, `smg_cache_tree_tokens`, `smg_cache_tree_tenants`, `smg_manual_policy_cache_entries` | — | Ends with the classic trees and manual cache. |
| `smg_cache_aware_prefix_margin_held_total` | — | Open. An embedded successor requires the scorer to report whether the margin held without an extra read. Otherwise the counter ends with classic routing. |
| — | `smg_policy_decision_seconds{mode,outcome}` | Added. One observation per ask. |
| — | — | Planned additions. Embedded writers are pending for `smg_policy_leases{kind,state}`, `smg_policy_spec_generation`, and `smg_policy_spec_apply_total{result}`. The standalone controller already publishes the two spec families when it has a spec source. No code writes `smg_policy_leases` yet. |
| — | — | Planned renames of the standalone controller's `smg_policy_controller_decide_busy_total` to `smg_policy_fallback_total{reason}` for fallback and of its `smg_policy_controller_decide_budget_outlasted_total` to `smg_policy_deadline_misses_total` for decision deadlines. No code writes the new names yet. |
| `smg_router_requests_total`, `smg_router_request_errors_total`, `smg_router_request_duration_seconds`, `smg_router_tokens_total`, `smg_router_ttft_seconds`, `smg_router_dispatch_wait_seconds`, `smg_router_tpot_seconds`, `smg_http_responses_total`, `smg_http_request_duration_seconds`, `smg_http_inflight_request_age_count`, `smg_http_connections_active` | Same | Kept. Request and connection metrics. |
| `smg_admission_inflight`, `smg_admission_queue_depth`, `smg_admission_queue_rejected_total`, `smg_scheduler_admit_total`, `smg_scheduler_inflight`, `smg_scheduler_queue_depth` | Same | Kept. Admission and scheduling. |
| `smg_engine_running_requests`, `smg_engine_waiting_requests`, `smg_engine_token_usage`, `smg_engine_cache_hit_rate`, `smg_worker_health`, `smg_worker_cb_state`, `smg_worker_health_checks_total`, `smg_worker_retries_total`, `smg_worker_retries_exhausted_total` | Same | Kept. Worker reports and resilience. |
| `smg_kv_event_lag_seconds`, `smg_kv_event_gaps_total`, `smg_kv_event_resyncs_total`, `smg_kv_event_missed_batches_total`, `smg_kv_event_stream_errors_total`, `smg_kv_event_degraded_ranks`, `smg_kv_index_entries` | Same | Kept. Gateway cache event and index metrics. |
| `smg_symphony_outputs_total`, `smg_symphony_malformed_total`, `smg_symphony_fallback_total`, `smg_symphony_refused_total`, `smg_symphony_calls_suppressed_total`, `smg_tokenizer_cache_lookups_total` | Same | Kept. Output parsing and tokenization. |
| `smg_tokio_worker_busy_ratio`, `smg_tokio_event_loop_stalls_total`, `smg_tokio_event_loop_delay_seconds`, `smg_tokio_alive_tasks`, `smg_tokio_global_queue_depth`, `smg_log_dropped_lines` | Same | Kept. Runtime and logging. |
| `smg_allocator_resident_bytes`, `smg_allocator_allocated_bytes` | Same | Kept. Available when the gateway runs with jemalloc statistics. |
| `smg_pd_prefill_duration_seconds`, `smg_pd_kv_transfer_duration_seconds`, `smg_pd_decode_ttft_seconds`, `smg_pd_ttft_seconds`, `smg_pd_rendezvous_failures_total`, `smg_pd_bootstrap_failures_total`, `smg_pd_kv_transfer_failures_total`, `smg_pd_kv_connector_mode_total` | Same | Kept. Prefill and decode dispatch. |

The manual policy branches `occupied_hit`, `occupied_miss`, `vacant`, and
`no_routing_id` correspond to `pick`. `no_healthy_workers` corresponds to `shed`.
`cap_respill` corresponds to `pick` when the policy returns a worker and `shed` when it
returns none. The outcome count groups these branches together.

An embedded selection uses the existing opt-in routing capture. Its `x-smg-cache-trace`
selection row names the profile in `policy`, uses origin `policy_controller`, and carries
a `decision` with up to three scored worker addresses. The gateway uses redacted addresses
for the selected worker and the top candidates. The same row appears in the dispatch log
when tracing is configured to log.

## Presets

`PolicySpec::preset("least_load")` builds a complete document with generation 1. Each preset has one profile, the same three filters and a binding for every model. The copies below leave the number to the source. `PolicySpec::power_of_two` also accepts `ExpectedWait`, `Requests`, `LeastRequests` or `ResidentTokens { request_tokens }`. The resident variant ranks the full set by resident work. `pair_presets` and `epd_presets` compose side profiles from preset names and add `has-capacity` to the prefill side and `has-room` to the decode side. Over `MemoryStore` or the standalone controller a row reads both bits open, so these documents place pairs as written, and the host that asks gates capacity at its own commit. [Preset constructors](src/document/spec.rs)

<details>
<summary>Preset documents</summary>

### `random`

Draws among the workers that pass its filters. The picker uses no score. The overload filter can still use load reports.

<!-- preset: random -->
```yaml
apiVersion: smg.policy/v1
profiles:
  random:
    filter: [healthy, not-overloaded, not-tried]
    pick: random
bindings:
  - match: { model: "*" }
    profile: random
```

### `round_robin`

Rotates through a survivor set. Changes to that set change the rotation, and retained rotations are bounded.

<!-- preset: round_robin -->
```yaml
apiVersion: smg.policy/v1
profiles:
  round_robin:
    filter: [healthy, not-overloaded, not-tried]
    pick: round-robin
bindings:
  - match: { model: "*" }
    profile: round_robin
```

### `passthrough`

Takes the first survivor in host order.

<!-- preset: passthrough -->
```yaml
apiVersion: smg.policy/v1
profiles:
  passthrough:
    filter: [healthy, not-overloaded, not-tried]
    pick: passthrough
bindings:
  - match: { model: "*" }
    profile: passthrough
```

### `power_of_two`

Compares the estimated waits of two draws. A full filter pass can still be needed.

<!-- preset: power_of_two -->
```yaml
apiVersion: smg.policy/v1
profiles:
  power_of_two:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: power-of-two
bindings:
  - match: { model: "*" }
    profile: power_of_two
```

### `least_load`

Favors the best estimated wait, including active KV pressure. Uses in-flight counts when the compared set has no reports.

<!-- preset: least_load -->
```yaml
apiVersion: smg.policy/v1
profiles:
  least_load:
    filter: [healthy, not-overloaded, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    profile: least_load
```

### `consistent_hashing`

Favors the routing key's ring owner while it is usable. With no session key the request is placed by its client's key, read from `x-forwarded-for`, else `cookie`. With neither, the scores tie. The binding names the routing key header as the key's source, and the two client headers behind it. A sessions rule added to the binding reads the same key and names no header of its own.

<!-- preset: consistent_hashing -->
```yaml
apiVersion: smg.policy/v1
profiles:
  consistent_hashing:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
bindings:
  - match: { model: "*" }
    profile: consistent_hashing
    routing_key:
      sources:
        - header: x-smg-routing-key
        - header: x-forwarded-for
          client: true
        - header: cookie
          client: true
```

### `prefix_hash`

Uses the routing key first, then the prompt's prefix: every block from the first through the last whole block at the deepest reached boundary, or at the configured 256-token depth when no boundary is reached. Two prompts that differ anywhere in that prefix take different keys, so they spread over the workers. When no key can be used, the in-flight-count scorer places the request. The binding names the routing key header as the key's source. A sessions rule added to the binding names no header of its own.

<!-- preset: prefix_hash -->
```yaml
apiVersion: smg.policy/v1
profiles:
  prefix_hash:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - consistent-hash:
          weight: 8
          prefix_tokens: 256
          bounded_load: { factor: 1.25, abs_margin: 10 }
      - least-requests
    pick: best
bindings:
  - match: { model: "*" }
    profile: prefix_hash
    routing_key:
      sources:
        - header: x-smg-routing-key
```

### `manual`

Uses a host-resolved key from the routing header with random first placement and preference for the original pin on recovery. This preset sweeps every 60 seconds and disables the in-flight cap.

<!-- preset: manual -->
```yaml
apiVersion: smg.policy/v1
view:
  headers: [x-smg-routing-key]
profiles:
  manual:
    filter: [healthy, not-overloaded, not-tried]
    pick: manual
bindings:
  - match: { model: "*" }
    profile: manual
    sessions:
      sticky_header: x-smg-routing-key
      assignment: random
      ttl_s: 14400
      sweep_s: 60
      inflight_cap: 65535
```

### `cache_aware`

Blends warm-up, prefix reuse and least-load. The default KV gate is off. The stock controller reads cache events from each gRPC worker whose row asks for them and names a model. Where no worker publishes them, the preset works from the prefixes sent to each worker, worker age and load.

The preset's prefix stage names no source, so it reads `merged`. A worker the store has heard a cache event from is placed by its cache events. A worker it has heard none from is placed by the prefixes sent to it, so the preset keeps a returning prompt on its worker on a fleet with no cache events at all. A worker that asks for cache events and never publishes one is placed by the prefixes sent to it too. Once the store applies a batch of its that carries a cache event, the worker is placed by its cache events alone, and what the record holds of it ages out under the record's bound. Set `source: index` to read cache events alone, or `source: predictor` to read sent prefixes alone.

<!-- preset: cache_aware -->
```yaml
apiVersion: smg.policy/v1
profiles:
  cache_aware:
    filter: [healthy, not-overloaded, not-tried]
    score:
      - warmup-slice: { weight: 8 }
      - prefix-affinity: { weight: 4 }
      - kv-pressure
      - least-load
    pick:
      best: { sample: 16 }
bindings:
  - match: { model: "*" }
    profile: cache_aware
```

### `bucket`

Groups prefill prompts by size in a host that reports the lifecycle. The standalone validator refuses this preset.

<!-- preset: bucket -->
```yaml
apiVersion: smg.policy/v1
profiles:
  bucket:
    filter: [healthy, not-overloaded, not-tried]
    score: [bucket]
    pick: best
bindings:
  - match: { model: "*" }
    profile: bucket
```

</details>

## Cost and regression gates

The decision benchmarks time view construction and policy execution separately, typically at 8, 64 and 512 candidates. They cover presets, dispatch credit, overloaded pools, sticky hits and misses, spec validation, cache locality, sampling and prefix keying. The store benchmarks time membership and load writes, publication, view building, update journals, KV ingestion and pairing. Library benches also measure encode placement, pair and encode class construction, concurrent callers, the sent-prefix predictor, election, and both plugin kinds. The `decide_stream` bench measures loopback round-trip percentiles and throughput under offered load, separately from in-process work. [Benchmark sources](benches)

```sh
cargo bench -p smg-policy-controller --locked --bench decision
```

| Gate | Work measured | Threshold by row |
|---|---|---|
| [Engine](../../.github/bench-gates/policy-engine.toml) | View build, decisions and spec parse/validation. | 10–15% |
| [Affinity](../../.github/bench-gates/policy-engine-affinity.toml) | Affinity decisions and selected sizes of view builds. | 10% |
| [Bucket](../../.github/bench-gates/policy-engine-bucket.toml) | Bucket decisions, including prefill across models and with stale rows. | 10–20% |
| [Power of two](../../.github/bench-gates/policy-engine-power-of-two.toml) | The preset's load and count variants. | 10% |
| [Prefix hash](../../.github/bench-gates/policy-engine-prefix-hash.toml) | Ring owner and load spill paths, with and without boundaries. | 10–15% |
| [Store](../../.github/bench-gates/policy-store.toml) | Store views, load and membership writes, publication and selected pair and many-model cases. | 10–40% |
| [Store journal](../../.github/bench-gates/policy-store-journal.toml) | Journaled writes, view build and journal reads. | 10–30% |
| [Gateway policy selection](../../.github/bench-gates/policy-selection.toml) | The gateway's regular placement benchmarks. | 10% |

The gates run three interleaved measurements per side when timing is needed. Each benchmark ID uses the first matching threshold. A time regression must exceed that threshold in the median of the runs, with every changed run slower than every base run. Overlap is reported as suspect. Some gates also repeat suspect measurements or check for executable placement effects before failing a row. Missing gated IDs are detected. These are relative regressions for the measured paths. The plugin and wire round-trip benchmarks have no dedicated policy gate here. [Gate rules and report](../../scripts/bench_gate.py)

## A worker's row

The controller keeps one row a worker. A registration states the row's facts. One that states a fact past its bound is refused by the fact's name.

| fact | bound |
|---|---|
| the URL | 2,048 bytes |
| the runtime, the zone, a model id | 256 bytes each |
| the labels | 128 of them, a key of 128 bytes, a value of 4,096 |
| the row as a whole | 16,384 bytes |

No count bounds a row's model ids: the row's size does. A control call's row is weighed as the wire frames it. A row that a host writes in its own process is weighed by the store: 160 bytes for its numbers, and for each text its bytes and five more. That measure is never under the wire's weight, so a row the store takes fits a frame.

A host that links the library with the `worker-host` feature writes rows at a membership door, `worker_host::MembershipWriter`. The door lays a found worker's facts over the row at its URL, marks a row draining and takes a row away. Only a row's maker lays facts over it, save for a refresh of its labels, and facts that name another URL than the call's are refused. A health loop's refresh of a worker has an entry of its own, `MembershipWriter::refreshed`. The labels are the engine's word, so the entry asks no maker and the row keeps the maker it has. The refreshed worker may differ from its row in the labels alone: one that states any other fact is refused, and nothing is written. The labels are bounded as a registration's are, and equal labels write nothing. A row that is draining takes no refresh. A removal names the row its asker saw, by id and revision, and the next section says who may remove whose rows. `worker_host::StoreMembershipWriter` is the door over a store alone, for a host with no election. The controller's own door also refuses a fact that its log does not carry to a standby: a pairing descriptor, an encode offer, the vision bit, a priority, a drain window, an empty zone, and a zero where the row means none. It also refuses a worker of the host-local ZMQ wire whose row names no gateway, as it refuses a control call's registration of that wire. One gateway holds that socket, and such a row would stand in every gateway's pool. `worker_host::facts` reads a worker object back to a row's facts. The steps of the next section call the door. The stock controller does not run them yet.

The controller probes its workers while it leads. Its binary is built with the `worker-host` feature, which holds the prober; a library build without the feature probes nothing. A row's health policy sets the check interval, the timeout and the two thresholds. The controller counts the interval and the timeout in whole seconds, one at least. A new row is probed at once, then once an interval. `success_threshold` passed probes in a row ready a row. `failure_threshold` failed ones take a ready row out, and three times as many more fail it. A pending row fails after ten times `failure_threshold` probes without becoming ready. A replica that does not lead probes nothing: its rows keep the status its leader's log tells. A row whose policy turns probes off is not probed: its door readied it at its registration, and it stays ready. A probe removes no row.

## Running worker jobs over a local store

A host that keeps the membership store in its own process can use `registration_into_store` and `removal_from_store` with the `worker-host` feature. The steps write rows through the membership door. The host calls `Objects::follow` when the store moves, and the object holder writes the registry. The host supplies the job queue and runs the workflows.

`write_workers_at_door` waits for creation or the host's last checks. It takes the three job modes. An `Upsert` registers an absent URL and refreshes a standing row. A `CreateOnly` job adds a row under a free URL, and fails with "already exists" where a row stands, whoever made it. A `CreateOnly` job of a worker of several ranks is refused by name before any rank is written, until the door lays a worker's ranks in one turn. A `ReplaceById` job is laid over the row behind the registry's worker of that id, at the revision the job names. It takes a worker of one rank: a job that builds several is refused before any rank is written, as the gateway's own replacement is. It fails where that worker left or moved. It also fails while that worker drains: a removal wins over a replacement that lands inside its drain window. A replacement that changes labels, checks or thresholds is made in place, and the worker keeps its id. One that changes the role, the wire or the models is the URL's next life, and the worker takes a new id. `registration_into_store` takes the object holder for that read. It is a step of the local kind: it skips a registration of another kind, whose workers the host's own steps write. API and startup jobs write as control calls. A discovery job writes as provider `discovery`, using the spec's address as its key. For example, a job for `http://worker:8000` uses that one key even if the job builds two ranks at different registered URLs. Health jobs cannot register workers. If one rank is refused, the rows written before it stand.

A removal selects workers, marks their rows with `drain_workers_at_door`, waits for them to settle, then takes their rows with `remove_workers_at_door`. The settle waits the drain window the worker was registered with, one day at most: its row carries a window other than the default, and the object built from the row takes it. A longer window ends the removal at the day. Each write checks the row id and revision. API and startup jobs can remove control rows. Discovery can remove rows of provider `discovery` under any key. Health uses the row's maker for its scope. `Disabled` permits no rows, `DiscoveryOwned` permits provider rows, and `AllLocal` also permits control rows. The controller's door includes the asker in its removal audit lines. A removal the door refuses for its asker fails at the drain with the door's sentence, where the job drained nothing: another maker's row, or a row outside the health scope. Each step drops from the job's lists a worker it leaves standing, so a host step behind the removal reads the workers it took or found gone. An object no row stands for stays in the lists for the host's own steps.

Each removal write has one attempt. A worker that an `Upsert` job finds again while its removal settles is registered anew: its row takes the URL's next life, out of the drain, and the pending removal leaves it.

## What is not built yet

- A client that marks a request as asked again. The controller serves the mark. The crate's own client does not set it yet.
- A served call that decides the decode under a held prefill lease. The stock controller picks both sides in one decision.
- Encode placement through a served controller. The in-process door places the encode items with `StoreBackedEngine::decide_encode_pair`. The served controller decides an `epd` request as its pair alone because the wire carries no item.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A request's hard demands but one. A host that decides in process states the runtime a retry stays on (`Constraints::demands`), a worker's registration states its runtime (`WorkerFacts::demand_facts`), and the door keeps the workers that state it before any stage runs. A registration that changes what a worker states starts another life of the worker, as a change of its wires does. A pair's sides keep no demand yet: a pair's request that states one is shed under the stage `wires`, whatever its sides state, until the store narrows a pair by its demands. The stock controller's `Decide` request has no field for a demand yet, and a worker it registers states none. A request's context window, its constraint's decider and its rank pin are not kept yet.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- Shadow execution through a served controller, shadow maps by side, and plugin stages in a shadow profile.
- `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd`.
- The `best-with-temperature` picker, and three settings of `prefix-affinity`: `decay`, `accounting_ttl_ms` and `spill.uncached_credit`.
- The `bucket` scorer in the stock controller. The library runs it. The stock controller refuses a document that names it, because it does not yet report every request's end to a stage.
- The record of sent prefixes beyond a single lease. A pair's lease records nothing, and a standby starts with an empty record.
- A text request's affinity on a worker that publishes cache events. Once the store has heard a cache event from a worker it reads that worker by its cache events alone, and cache events hold token blocks.
- The wire's `prompt_text`. A controller does not hash a foreign host's text yet. It answers such a request `Busy`, as before.
- Cache events from an engine that asks for a key, or from a data-parallel worker's rows above rank 0. The stock controller builds its worker objects without credentials, so an engine that asks for a key gives it no stream. Its `PushKvEvents` call is not served. Every replica of the stock controller opens a cache-event stream to each gRPC worker whose row asks for events and names a model, and keeps those streams while the active document reads no index. A host built with the `worker-host` feature has the feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A host that opens a row's feed without the row's cache group state clears the row's blocks and cursor first, so the subscription starts at the head of the stream.
- One more input in the stock controller. A worker's circuit-breaker bit has one source there: a veto of the asking gateway that holds, as its breaker's open window does. Such a veto takes the worker out of that gateway's decisions, and no rule lifts it. Its row reads as stalled too, as under every veto of that gateway, so a filter that reads the stalled bit refuses it. The controller keeps no breaker of its own.
- An engine connection over the host-local ZMQ wire. A controller built with the `worker-host` feature hands each worker object a gRPC connector alone, and an object connects to its engine at its first use. A worker that serves only the ZMQ wire gets no connection there, so nothing that controller reads from an engine reaches it for that worker. The controller sends such a worker no health probe: a row that names its one gateway is host-local, and the controller cannot dial it. Its row turns ready by its registration's connect signal, which a row whose policy turns probes off gets; with its probes on, nothing readies it yet. No field of a registration or of a worker object names a gateway yet, so the controller's doors refuse a worker of that wire.
- A probe with a key. The controller's probe of a worker carries no API key until a host hands the worker objects a keeper of credentials. An HTTP worker's URL userinfo goes out as Basic authentication, as the HTTP client sends it for every call. A gRPC worker's channel cuts the userinfo, so its probe carries no credential.
- A count of the controller's probes in its own metrics. The worker crate counts each probe in a recorder the controller does not install.
- A probe schedule kept across a change of leader. A new leader probes at once each worker that has no verdict on that replica, and each other worker after its own interval.
- The worker crate's load collector in the controller's own service. The `worker-host` feature has the two sinks. `worker_host::HealthAdapter` writes probe outcomes and connect signals to the store's rows. `worker_host::LoadAdapter` writes the load collector's pushed records and poll answers to the store, while its host leads. The controller's own service runs the health checker over `HealthAdapter` while it leads. It runs no load collector over `LoadAdapter` yet: its leader polls its workers' loads itself.
- A driver for the leader's own doubt of a worker. The store takes the write and a decision reads it as the row's stalled bit, for every gateway (`HealthSink::leader_bits`, `HealthAdapter::project`). No driver calls the projection yet. Until one does, a worker the leader's liveness check doubts reads as routable to a decision unless a gateway's own veto masks it.
- A plugin named in the document. The format has no field for one.
- Four calls of the contract: `PushLoad`, `PushKvEvents`, `StepDown` and `WatchRegime`. The stock controller answers each as not served.
- Eight of the ten slices of the snapshot call, `GetSnapshot`. The stock controller serves the worker rows and the spec's status. It refuses a request for any other slice by the slice's name, and a request that names no slice asks for all ten. It also refuses a request for identifiers in clear.
- The audit mode of the apply call.
- Registration of the workers that discovery finds. A provider hands each worker to the controller's job queue, and nothing sets that queue yet. Until then a pass finds workers and registers none.
- A gateway that asks a standalone controller for its decisions. Under `--policy-controller-address` the gateway follows the controller's workers and still decides in its own process: see [In the gateway](#in-the-gateway).
