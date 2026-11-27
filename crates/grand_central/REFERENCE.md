# Grand Central

Grand Central chooses which workers serve a request. One versioned YAML or JSON document composes health and capacity filters, weighted scores for load or cache locality, and a picker with alternates. Bind profiles by model and phase, keep related requests on a worker, or choose compatible workers for split requests. The same Rust library runs in a host process and behind the standalone gRPC service. The service validates a new document before it replaces the running one and can explain sampled picks and sheds.

## First decision

From a checkout, run the small in-process example with three ready workers. It checks that the worker with one request in flight comes first, followed by the workers with three and five. It needs no running service.

```sh
cargo test -p smg-grand-central --locked --lib the_pick_orders_targets_best_first_and_names_the_survivors
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
3. **Pick.** Every profile has one picker. A pick carries up to four ordered targets. A shed gives a reason and a stage or constraint: the one that stopped the decision, but for two sheds of a pair. The picker can narrow the comparison before scoring.

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

The stock filters, scorers and pickers stand on [a page of their own](docs/stages.md), each with its rules, its parameters and their defaults.

## Signals a decision can read

The engine reports a profile's input requirements through `Needs`. The host supplies the request and worker views. Native stages can read the full view. Plugins see the projection their host configures. See [the Rust view types](src/engine/view.rs) and [the wire records](contracts/policy_plugin.proto).

| Signal | Where it comes from |
|---|---|
| Model, phase, tenant, namespace seed, priority, streaming, output limit, prompt size, media counts, retry attempt and seed | The caller's `RequestFacts`. The remote caller sends these in `DecideRequest`. A frame with raw prompt text receives `Busy`. The built-in binding matcher reads model and phase. |
| Request hashes and headers | The caller sends token-block or text-window hashes, their kind and their block size. A token request may leave the size out and reads the document's. A text request states its window's size in bytes: over the wire, one that carries hashes and states none is refused by `facts.block_size`. The applied document bounds the prefix read. The caller projects headers as positions in `view.headers` and resolves the routing or session key. |
| Ready, draining, stall, breaker and reachability | Membership and health probes, plus the asking gateway's vetoes and wire constraints. A liveness veto is a doubt. A held veto represents the asking gateway's breaker and cannot be lifted by the liveness fallback. The standalone controller reads a worker's breaker on the worker's own object, beside the asking gateway's vetoes. |
| Running, queued and waiting uncached work, KV usage and active KV usage, drain rate, resident tokens, admission window, sample age and source | Host load reports. The standalone leader's load monitor asks each worker for its load: the load RPC over gRPC, and over HTTP the worker's native load route or the `/metrics` gauges of a supported runtime. It also takes the load records on a worker's stream, and asks no load of a worker whose records are fresh. Its decisions read those reports, and a document that holds a stage that keeps reports is given none yet. The store holds no load row. Fields depend on the reporter. A worker whose load call fails keeps its last report where that report states a running or waiting request, as old as its sample, until its load call answers again, a load record comes on its stream or it leaves, and no further than the first load poll that finds the sample 30 seconds old; past that poll it has no report. Use `view.max_report_age_ms` to age out reports for the affected scorers and KV gates. Overload is a separate latched verdict. |
| Host requests in flight and dispatch credit | Lease bookings and lifecycle reports. Read counts through `RequestFacts::in_flight`. The load scorers can account for dispatched work until a report covers it. |
| Pins, requests in flight for one key, and active-key counts | The host's session facts, or the engine's session rule. The standalone controller's engine keeps a session table of its own, one for the controller's life. It counts a key's requests on each worker's object, by the leases it holds. The lease of every keyed request counts, under any sessions rule and under none: a single's on its worker, and a pair's on each side's worker. A record's count of active keys counts the sessions of those leases. No row is premarked as a session hit. Its session rule keeps pins using the caller's key. |
| Cache depth, age, growth and size | An embedded host can supply an affinity column itself. It can also supply sent-prefix predictions. With regular standalone ingestion, worker age is available, and the stock controller records the prefixes it sends at each single request's decision, unless a sessions rule placed the request and keeps it from the profile. The stock controller reads cache events from each gRPC worker whose row asks for them and names a model. A controller of your own built with the `controller` feature does too. |
| Role, ranks, wires, stable worker key, zone, engine family and labels | Membership. The document projects labels to bits, and zone and family to positions from 1. An unlisted or absent zone or family is 0, and so is every zone in a standalone controller's decision, where a worker's object states none. |
| Capacity, room, pair and encode compatibility, worker vision | The host supplies admission bits and worker facts. `MemoryStore` holds no candidate row: it knows no admission window and derives no class. A host derives pair classes itself, by `PairClasses::derive`, and encode classes by `EncodeClasses::derive`. |
| Pool generation, size and membership, reachability and host constraints | The pool view and `Constraints` carry the pool's facts, exclusions, vetoes, a wire mask and an optional subset hint. A standalone request can narrow by common wires or specify a zero subset to exclude the pool: the controller applies both ahead of the stages, which are shown the places the gateway may take, no wire mask and no positive hint. Hosts can supply further pool facts such as a regime or zone capacity. The standalone view supplies neither. |
| Named values the document declares under `signals` | Worker load reports, pushed or polled: the values a report states by name, and the two fixed token counts. `InProcess` keeps none: it reads them in the load snapshot its decision holds, for a single request, for each side of a pair and for each side of an encode request. A native stage that resolved a signal's column reads them. In a standalone controller's decision that snapshot is the one its load monitor last published, and a document that holds a stage that keeps reports is given none yet. No plugin contract carries them, and a standby does not follow them. |

An embedded host can implement or use the store's ingress traits for health and leases. One that keeps no worker objects reads its membership through `WorkersStore`, which holds no write, and writes a row as a decided change (`MemoryStore::follow`). It feeds cache events through the worker host, under the `worker-host` feature. It hands the holder of its worker objects the KV tables (`worker_host::KvTables`, `Objects::with_kv_tables`), and `worker_host::KvHost` over that holder gives each event subscription its row. The holder makes a row's lane where it states the row and retires it where the row leaves, and the crate's adapter alone writes blocks through a lane: nothing outside the crate writes the tables. A holder handed no tables makes no lane, and its host indexes nothing. The tables hold an event-fed chain index a model, each for up to 1024 workers. The adapter reports sequence gaps and supports replay and resync. A snapshot that its stream cuts short keeps the blocks it had shown: the next snapshot's start replaces them. A host reads a rank's cursor, a row's indexed blocks and the resyncs counted under a reason from the tables (`KvTables::cursor`, `KvTables::indexed_blocks`, `KvTables::resyncs`). A controller's tables are at `Controller::kv_tables`. A host can read the validated `index` settings through `PolicySpec::index_for` and use them to configure its own index. The KV tables use the same index layout for every document. A standalone controller's engine keeps a record of the prefixes sent to its workers, and its prefix stage reads it unless the stage says `source: index`. The stock controller writes a placement at each single request's decision, unless a sessions rule placed the request and keeps it from the profile. The record holds `max_size` over the block size placements at most. A placement is a prediction made at the decision. A decision turned back after its lease's issue keeps it, and the next request with that prefix reads it as a predicted hit, which must exceed the stage's `threshold`. In a standalone controller's decision the engine's record speaks for every worker on a text request or under `source: predictor`, and on a pair's side for each worker the KV tables hold no block for. A single request with token ids reads the record while the model's tables hold no block: once they hold one, the tables decide it alone. The store tells the engine each word of its membership: the record forgets a worker that left, was replaced, stopped serving a model or changed its role. The record stands across a new document. A pair's lease records nothing, and a standby starts with an empty record. A text request is recorded as a token request is, by the hashes of its windows. Readers of the record wait for a writer: under many callers that never pause, a fleet whose requests mostly grow decides slower than one that reads no record. A host with a store of its own keeps the same record with the crate's `PrefixPredictor`: it folds a request's keys once (`PrefixPredictor::fold_into`), reads by a shared reference, and writes with a tick of the host's clock. `kv_index::salt::request_text_hashes_with_seed` makes a text's window hashes under a cache namespace's seed, and `engine::TEXT_BYTES_PER_TOKEN` is the four bytes a token. A window of text and a block of token ids over the same bytes never share a hash. A host hashes a request by its model's block size: the size the KV tables have on record for the model (`KvTables::block_size`), and the document's `view.block_size` where they have none. A registration can state the size, and a wider registration raises it. Once the model's engines store a block, the tables keep the engines' own size: the widest block a model's indexed stores named. A worker need not state it. A model first hashed by the document's size and later by a size its engines state loses the prefixes sent before, until they are sent again. [Store API](src/store/mod.rs) · [Sent-prefix predictor](src/tracking/prefix_predictor.rs)

## Run it

The standalone service stands on [a page of its own](docs/service.md): its flags, its worker discovery, the calls it serves and the client a host calls it with.

### In a host process

Validate a `PolicySpec` to get an `Engine`. Call `Engine::decide` with request facts, rows, constraints and reusable `Scratch`. A host that needs booking makes an `InProcess` from an `AppliedSpec` and asks it, itself or through a `DecisionGuard`, lending its worker lists with each request. `InProcess` runs on the caller's thread. It issues the lease for the first target. The host takes a target, reports dispatch and completion or release, and calls `DecisionEngine::sweep` on its timer. `InProcess` decides a pair when asked at its prefill: that one request decides both sides. It returns `PairSide` for a decode or an encode request that is a side of its plan's pair or that hands more than one list, and `Faulted` if it cannot decide a pick. A single pick of `InProcess` states what its first worker holds of the request's prefix (`Pick::overlap`): the leading blocks held, the request's blocks and where the match came from. `InProcess` reads them from that worker's entry of the rows its decision was given: the rows it named from the host's data, a column the host filled, or the rows the host named. It asks no stage, so no stage's threshold applies. A worker the rows do not name is a miss of the request's blocks. A request decided inside a tree its host lends is given the rows the tree's match names and the match's own two numbers: its size and the request's, in the tree's unit, tokens or characters. A pick of a worker named among the match's holders states those two numbers and names the tree as where they came from (`OverlapSource::Tree`), with nothing rounded to a block. Any other worker is a miss of the request's blocks. Past the profile's sample the holders named are a draw. A pair's first pick states the same of its prefill, which its first target names, from the prefills' column, and nothing of its decode. Nothing is stated for a pair's decode decided again, for an encode request's pair, for a pick the sessions rule keeps from the profile, or under a profile that reads no affinity. The word on the prefix margin stays `false`. `Engine::decide` states nothing, and the controller's `Decide` answer does not carry it. [Engine example](tests/unit/engine/framework/tests/profile_run.rs) · [InProcess API](src/engine/in_process.rs) · [DecisionGuard API](src/tracking/decision_guard.rs)

A host that keeps one `InProcess` for its life gives it a later document with `InProcess::apply`. The call refuses a document whose generation is not above the standing one. The next decision runs under the new document, and a decision under way ends under the one it began under.

The session table and the record of sent prompts stand across the call. The stages of a profile whose name and text did not change stand too, with what they hold. That holds where the new document changes no `signals`, and its `view` is the same or differs in its header list alone. The paragraph "What a new document keeps" of [Updating a document](#updating-a-document) has the rules, a tally of sends among them.

Every other stage starts empty, and is told the workers that stand in the registry the host handed before the call returns. A stage that stands is told none of them again. A stage that keeps load reports is told each worker's report at the engine task's next upkeep.

`InProcess::spec` answers a `SpecHandle`: a counted reference that reads as the document and keeps showing the document it was asked under. Ask `place_for` and `not_alone` of the handle, and hand it to `InProcess::pick` and `InProcess::pick_among` with the profile it answered. A host that reads one word of the document at each request asks `DecisionEngine::with_spec`, which borrows the document for the call.

A dispatch, a completion or a release of a lease issued under an earlier document is told to the stages that stood under that document and stand still, and to no stage built since. Its dispatch is counted once by every tally of sends in force, kept or built new.

The crate defaults to `wasm`. Use `default-features = false` for the library without that runtime. `controller` adds the service, its protocol, command line and gRPC plugin host, and enables `operator`, `worker-host` and `in-process`, the engine over a host's worker objects. `client` enables the reference client and watcher and implies `controller`. `kubernetes` adds the namespaced Lease backend. `operator` adds the operators' door for a host's own process. It builds no server, and it builds on every target. `worker-host` adds an optional registry of worker objects, one for each membership row, written through the holder's write calls. In the controller build those objects use a lazy gRPC connector. The feature also has the cache feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A controller built with the feature fills its index itself: every replica opens a cache-event stream to each gRPC worker whose row asks for events and names a model. The stock controller is built with it: its binary requires `controller`, which enables it. `discovery` lets the controller run a worker discovery provider, and enables `controller`: see [Worker discovery](docs/service.md#worker-discovery). `test-util` supports the crate's tests and benches. [Cargo features](Cargo.toml)

A host built with `worker-host` can keep a worker object for each membership row. Use `Objects::new(registry)` and write each row through its write calls (`Objects::upsert`, `Objects::patch`, `Objects::remove`). A row a store took as a decided change around the holder (`MemoryStore::follow`) has no object until a snapshot is laid whole, and the holder's registration of its key is refused until then. A standby and a `Watcher` also lay it at the next event that tells the row and that the store takes.

A status told the same way does nothing to the object: a worker object's status leads its row's. A host writes a status on the worker object, and the store is told: the connect signal with `Objects::connect_row`, a drain with `Objects::drain_row`, and a probe's verdict through `HealthAdapter`. A `Watcher` lays the status its watch tells.

For example, a worker registered `Pending` can take requests once its connect signal makes it `Ready` and the host's breaker and vetoes allow it. The registry tells its subscribers each change, and a promotion puts an open breaker on trial, as a health probe's promotion does.

### A decode chosen after its prefill

A host in process sets `request.constraints.decode_later = true` on a pair's request. The first pick names both sides and books the prefill alone. The host can decide the decode again once it has reported the prefill's dispatch, and before it reports the prefill's end.

A stage that keeps state by issue hears the issue of every decode before its dispatch. Under a pair asked with `decode_later`, a decode call issues the decode side and the stage hears that issue. A decode sent with no call goes out as the first pick's: the stage hears its issue at that dispatch, just ahead of the dispatch's own word. A stage that a document put in force since the pair's first decision built hears none of the three: it never heard the pair's issue. The decision told there holds each target's workers and ranks, with no trace and the default row position, 0, and the lease names the target taken. A stage reads the worker by its id there. One issue can go unheard in a host's own process: where a stage panics at an issue, the stages behind it do not hear that issue, nor a pair's decode side's where the prefill side's panicked. They still hear that side's dispatch and its end.

The lease of a later decode's issue is dated when its side is filed: at the decode call, or as its dispatch is told. It is never dated at the pair's first decision, nor by the date a gateway gives the dispatch. The lease's later words keep the first decision's instant.

For example, after the prefill is sent, call `handle.dispatched(sent_at)` and then `handle.decide_decode(&request, now)`. The new pick names the held prefill with each decode it can use, on the loads of that instant. Its first decode is booked and has five seconds to be dispatched. `handle.take_decode(index)` takes an alternate and `handle.dispatched_decode(sent_at)` reports the send.

A handle keeps its first pick where the call sheds or is refused. A decode side the host gave up with `released_decode` takes no call after: `decide_decode` answers `DecisionRefused::Decode(DecodeRefused::DecodeStands)` and asks no door. `decide_decode` answers the decision the handle holds, so its answer does not tell a shed from a pick. `handle.lease().due` does: a pick moves it to the call's deadline, and a shed leaves the first. A host that skips the call can send a decode of the first pick: its dispatch books that side.

A host that manages the lease itself calls `door.decide_decode(gateway, &request, id, now)`. A pick carries the decode side's lease under the same id, and a shed carries none. The lease keeps the generation and the time of the pair's first issue. The handle counts the decode's dispatch from the first decision, so a stage dates it by its own dispatch. A stage hears the decode's issue only where it heard the pair's first issue. Behind a new document the engine in force decides the side and tells the issue to the stages it kept since the pair's document, on a controller and in its host's own process alike. A refusal comes back as `DecisionRefused::Decode` with its reason.

A host that files pairs directly through `LeasesStore::issue_pair` can set the decode side to `LeaseState::Chosen`. The side keeps the pick's decode targets without booking a worker or starting a dispatch deadline. A side still chosen leaves the store when its prefill ends.

Once the prefill is dispatched, `store.issue_decode(gateway, id, decode, now)` can replace that choice once, while the prefill is held. It books the new first target and gives the decode five seconds to dispatch. A dispatch on a chosen side can also book a target from the original pick. An index outside the pick books none, and the side stays chosen.

`issue_decode` returns `DecodeRefused::NoLease` when the gateway has no lease under that id, `NotChosen` when the lease was filed without a chosen side, `PrefillNotHeld` when its prefill is waiting or has ended, and `DecodeStands` when its chosen side has already been issued or dispatched, or its gateway ended it.

`LeasesStore::expire` returns both sides as the sweep left them. For example, a completed decode stays `Completed` when its prefill is released as abandoned.

## Extend it

### Register a Rust stage

Implement `Filter`, `Scorer` or `Picker`. Add a factory to `StageRegistry::builtin()` and pass it to `PolicySpec::validate_in`, `Controller::bind_with` or `service::controller::run`. Each stage name is unique across the catalog. A simple scorer factory accepts the document's `weight`. It runs once at registration, where its stage's name is checked, and at every validation, including a dry run. Keep its effects inside the stage it returns.

The example registers a scorer with `stages.scorer("pack", || Box::new(Pack))?`. It favors workers with more reported running requests. The document names it in `score: [pack]` and uses `not-overloaded` to steer away at the threshold.

```sh
cargo build --release --locked -p grand-central-example
target/release/grand-central-example --controller-id controller-0 \
  --spec-file examples/grand-central/policy.yaml
```

A filter or a scorer you register opens a stream of its own where it draws, `Draw::for_stage(facts.seed, self.name())`. The seed's own stream, `Draw::new(facts.seed)`, is the one a picker that draws reads, as `best` does between tied rows: a stage that draws one of two from it draws that picker's value, and of two equal rows the picker then never takes the other.

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
- A `scorer_with` factory can call `build.dispatch_credit()` for the profile's `DispatchCredit`: what the host sent each row since that row's last load report. The scorer reads a row's `CreditedDispatches { requests, tokens, unsized_requests, estimated_tokens }` with `read`. `tokens` holds the prompt tokens the sends stated. `unsized_requests` counts the sends that stated none, and the scorer chooses their price. `estimated_tokens` is one price: the bytes of each such send's text over four, summed. The type is open to further numbers: code outside the crate reads the fields and matches them with `..`. A scorer that takes that price reads the pair `DispatchCounts { requests, tokens }` with `pass(facts).counts(row)`: its `tokens` are the stated ones and the estimates together, as the `sent.` terms read them. Where the read releases no sends, it reads three words for each row that holds sends, where the whole read reads four. A send is a dispatch the host reports. It stops counting at a report of the row sampled at or after it (at least the profile's `sent_report_delay_ms` after it, where the profile sets one), at the profile's `sent_ttl_ms`, or at the row's removal. In a standalone controller's decision the report is its load monitor's. A document that holds a stage that keeps reports is given none yet, so there a send ends at `sent_ttl_ms` or at the row's removal. When a send arrives with 8,192 already held on a row, the tally lets the row's oldest go, at any age.
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

A stage can read a named value that the document declares under `signals`. The factory asks the build context for the signal's column and the stage keeps it. In a decision the stage reads a row's value with `SignalColumn::read(facts, row)`. The read returns `None` when the decision's facts hold no cells, when the row has no value, and when the value is older than the signal's `max_age_ms`.

```rust
stages.scorer_with("least-pending", |build| {
    let pending = build.signal_column("pending_tokens")?;
    Ok(Box::new(LeastPending { pending }))
})?;
```

A filter registered with `filter_with` resolves a column the same way. A stage that keeps a column sets `signal_cells` in the `Needs` it declares. Validation refuses a stage that keeps a column without the need, and one that sets the need without a column. A name the document does not declare is refused at the stage's entry.

A host hands the values beside the rows: `RequestFacts::signal_cells` for a single pool, `PairPools::prefill_signal_cells` and `PairPools::decode_signal_cells` for a pair, and `EncodePool::signal_cells` for the encoders. A pair decision and an encode pair read each pool's cells: cells on the request are not read there. `SignalCells::new(cells, signals)` takes one cell per declared signal for each row, in the rows' order, with a row's cells in the order of the names. It owns its cells: a host keeps the one it built and refills it in place through `cells_mut()`, so no decision allocates for them. A value is finite: where a host has none, or its source states one that is not finite, it writes `SignalCell::NONE`. It makes every other cell by `SignalCell::of(value, sampled_at_micros)`. `AppliedSpec::signal_columns()` lists the declared signals in that order.

`InProcess` makes the cells for you. A worker's load report can state values by name, pushed or polled. The engine reads the ones the document declares, where the report states them under the declared unit. The two fixed token counts are offered under `kv_tokens_used` and `kv_tokens_capacity`, in `tokens`.

What to expect of a value it reads:

- A value is as old as its report. A value reported with its own age is that much older. The stock controller's load monitor dates a polled report when it reads the answer.
- Where one row's report holds several ranks, the row has a value only where every rank states the name under one unit. The ranks' values are then made one as `across_ranks` says, under the oldest sample among them. A worker that pushes its load dates each rank by that rank's own record, so a rank that stops reporting ends the row's value at `max_age_ms`.
- A report offers `kv_tokens_used` and `kv_tokens_capacity` where its source states them, zero too. A worker that pushes its load also offers a count its record leaves out, for each of its ranks: the count as that rank last stated it, as old as that statement, the age rounded up to a millisecond. So declare `max_age_ms` on a signal that reads one. A rank with no token capacity offers neither.
- Values are not replicated, and neither are load reports or overload verdicts: a controller that follows holds none, and a lead's reports end with the lead. A controller that takes the lead starts its load monitor empty: a document that reads them decides by each stage's fallback, and no worker reads overloaded, until the first poll of that start lands, or the first load record on a worker's stream. The poll goes out at the start. One exception stands until the worker crate's stop awaits every poll loop: a loop that a registration, a removal or a change of interval retired just ahead of a lead's end can still publish its reports and write its verdicts behind that end. They stand through the follow, and a decision admitted in the instant before the next start clears them reads them.
- A metrics scrape states no value by name.
- A report's list is held to the worker crate's bounds before it is read: its first 64 entries, each with a name of at most 63 bytes, a unit of at most 31 bytes and a finite value. A name's first statement stands. The controller counts what its own poll leaves out in `smg_grand_central_load_values_dropped_total`.

A host that asks `Engine::decide` itself builds the cells itself. A host of `InProcess` builds none for a single request, for a pair or for an encode request: that engine reads the values in the load snapshot its decision holds, each side of a pair from its own list, and reads no cells from the request. An encode request decided over its three lists reads them a side too: its pair's two sides as a pair's, and its encode side from its encoder list, where the encode profile of its pair's binding reads a named signal. That engine honours a value's own age, its signal's `max_age_ms` and the bounds of a report's list. It keeps no value: a decision reads what its snapshot's reports state, so a pushed record that leaves a name out reads as no value until a record states it again. The two fixed token counts are the worker's report's to keep: behind a record that leaves one out, a rank is still read its last stated count, until the signal's `max_age_ms`. There a worker object that stands for several ranks reads one value a name: its ranks' values made one as `across_ranks` says, under the oldest sample among them, because the worker's report states each rank's own sample instant. Its report is read where it lists as many ranks as the worker's registration states it stands for: its `dp_size`, with no rank of its own. A report of several ranks is read of no other worker, and a report that lists fewer ranks than it counts gives no value. A report of one rank is read as the worker's whole, so the first record of a worker of several ranks stands for it until a second rank's record. A worker registered one row a rank reads each rank's own values, pushed or polled.

The stock schema takes any stage name, written bare or mapped to an object, and lists the stock names in each entry's description. Validate an extended document against the host's catalog: the validator refuses a name the catalog does not hold, by its path. Stateful scorers and pickers can observe issue, dispatch, completion, release, membership and signals when the host supplies them. A stage that observes membership may be told a join again of a worker that stands, and a leave of a worker it did not hear join. `InProcess::apply` tells the stages a new document builds each worker that stands in the host's registry as joined, and then the list of them, beside the registry's own words. A stage takes each word as a statement of what stands. A stage that keeps a number of each worker's load report says so through `Lifecycle::keeps_loads`. Its host then tells it the reports through `Lifecycle::on_signal`. A controller tells none yet: under it a document that holds such a stage reads no load report. At each load snapshot it takes, the engine in a gateway's own process (`InProcess`) tells the reports whose sample instant is new, for the workers its task follows in the host's registry, before it keeps the snapshot for its decisions. A stage that does not say so is told no report, and no stock stage says so. In process the load signal names two more things: the report's sample instant and the named values the report states. `InProcess` states a report's named values under a document in which a stage that keeps load reports keeps a signal's column. That stage reads its column's value through `SignalColumn::told`, with the value's own sample instant and at any age. A report that lists fewer ranks than it counts states none, and no plugin is told them. A host that builds the signal itself makes it with `Signal::load` and puts its report's values on it with `Signal::with_signals`. A scorer of your own has a place for a pick's overlap too: `Scorer::states_overlap` and `Scorer::said`. Both have a default that says nothing. No entry of the engine asks them now. The standalone validator refuses stages that declare a need for the full lifecycle. [Runnable Rust example](../../examples/grand-central) · [StageRegistry API](src/stages/registry.rs)

### Load a WebAssembly component

Compile a component for the `smg:policy` package, at the version the WIT file names. The `policy` world takes records. `policy-packed` takes byte blocks with the layout given in the WIT. The host refuses imports from WASI. It provides log, clock, counter and seeded random calls. A component implements `init`, a decision resource with filter, score and pick, and `observe`. A score answer can carry `tie-band` on the scores' 0 to 1 scale: rows within that band of the best score tie, and the host uses its default band where the guest sends none. The host keeps instances per decider and meters their calls.

This host function loads the repository's policy component, using the default session bonus, and makes a scorer with an in-flight-count fallback. Use the returned scorer in a `WeightedScorer` entry of a `Profile`.

```rust
use std::{path::Path, sync::Arc};
use smg_grand_central::{
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

Use one `OpenDecisions` for the host's plugin runtimes. It accounts for the decision budget across plugins and sides. `PluginSpec` takes bytes, an optional ABI pin, opaque config, memory cap (16 MiB per instance by default), frame budget, fuel calibration (1000 units per microsecond by default), log limit (10 lines/s at level 2 by default), exposure, needs, counters, event subscription, fallback policy and breaker settings. A frame budget of zero takes the default frame of 100 ms, clamped to the decision's remaining budget as a stated one is. The loader refuses zero memory or fuel, a component that fails to compile or link, an unsupported world, ABI or import, and a failing `init`. `Plugin::check_memory_budget` can check a host's total memory limit per decider. A trap or invalid output discards the instance. [Component build and host example](../../examples/wasm/wasm-guest-policy) · [WIT](contracts/smg-policy.wit) · [Host configuration](src/plugins/wasm/config.rs)

### Call a gRPC plugin

Implement the `PolicyPlugin` service with the supplied proto. Its `Decide` stream starts with a hello and an acknowledgement with a protocol version (1 or 2) and supported stages. It handles decision frames by ID. An optional `Observe` stream sends ordered lifecycle events. A filter or scorer can start its call while native stages work. A picker bound on its own receives the combined scores at the pick. A scoring answer can carry `tie_band` on the scores' 0 to 1 scale: rows within that band of the best score tie. The host uses its default band when the field is unset. Use a Unix socket with a checked peer user ID, or TCP with mutual TLS and a pinned server name.

This host function connects to a plugin that's listening on a Unix socket under the same user and makes a picker. Use it as a profile's picker.

```rust
use std::{path::PathBuf, sync::Arc};
use smg_grand_central::{
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

A decline takes the configured stage fallback and leaves a healthy instance or stream in use. A trap, invalid output, timeout or open breaker can also trigger fallback. Surviving positions must be unique and in range. Each scored row needs one finite value, which the host clamps to 0–1. The pick must be a permitted position. With no fallback stage, the default `Open` policy keeps all rows, gives a flat score or ranks the combined scores, respectively. `Closed` sheds at a filter or picker and still gives a flat score at a scorer. The breaker defaults to five failures in ten seconds and a 30-second cooldown. Its latency rule can also open it when moving p50 exceeds half of the frames the plugin was given in the decision, at most the decision budget, or p99 exceeds them. Give a plugin a budget of at least twice its usual time. Hosts can read `Plugin::stats` and declared `Plugin::counters` in code. [Shared plugin settings](src/plugins/shared/config.rs)

## Policy document reference

The document's fields, its bindings and its plans stand on a page of their own: [The policy spec](docs/policy-spec.md).

## Guarantees and limits

### Decisions and refusals

A pick's first target is booked at issue by the door or service. The caller can dispatch an alternate and report which target it took. Report completion or release so the lease can end. An undispatched lease expires after five seconds. A repeat marked `again` ends an earlier lease still waiting for dispatch. A decode call's repeat takes back that call's issue alone. The service checks the leadership term and budget as it makes the decision and checks the term and lease at send. A decision takes its engine and the asking gateway's vetoes from one published world, reads the worker list the worker objects' registry publishes, and keeps the issuing engine for lease events. A new generation builds fresh stages for each profile it does not keep: the paragraph "What a new document keeps" of [Updating a document](#updating-a-document) has the rule. Outstanding leases still belong to their issuing engine while the store holds them.

- A single profile's empty pass can rerun filters with the overload veto lifted, then with liveness doubts lifted. It favors fewer in-flight requests on fallback. `shed_when_all_vetoed: true` keeps overload in force and can still lift liveness alone. Draining, not-ready and breaker holds stay refused by the health check. In a host's own process a pair's decode pool takes the overload half of that fallback at its first decision. A decode asked again behind its prefill takes both halves. The encoders of an `epd` decision shed without it: as saturated under `shed_when_all_vetoed: true`, and for a missing endpoint under `encoders` otherwise.
- Unmatched bindings, no endpoints, an empty filter pass and a declined picker are sheds, and so are incompatible pair or encode workers in a host's own process. The built-ins emit no queue tickets. Hosts decide how to map sheds to request status and retry behavior.
- A shed's answer says the reason and the stopping stage or constraint, with two exceptions for a pair. One shed as saturated names the overload veto, whichever stage ended its pass.
- A pair's shed with no endpoint, behind a pass the overload veto ended, names the stage that refused a worker with room of the list judged. Where none has room it is the first ready worker's, or else the list's first. A list judged with no worker names `partnered`.
- `Decide` has a 64 KiB incoming frame limit, including unknown fields. It requires a request ID of 1–256 bytes, known phase and hash kind, and at most 256 headers with positions in 0–255. A nonzero session hash must decode to a nonzero fingerprint for its model and phase. Zero means no key. A malformed or oversized frame can end the stream. The client cuts tail hashes to fit or returns an error when cutting them cannot make the message fit.
- `Welcome` grants 128 credits per decide stream and up to 256 streams per gateway. Requests spend credit. A credit frame replenishes it. `Busy` reasons are `queue_depth`, `no_credit`, `budget_gone_on_arrival`, `not_leader` and `unspecified`, beside four of a decode call its lease refuses (`decode_no_lease`, `decode_not_chosen`, `decode_prefill_not_held`, `decode_stands`). The five cover a full shard queue, exhausted stream credit, expired budgets or dispatch span, a fenced replica, and cases such as raw prompt text, an unsupported side, a decider fault or exit, spent lease IDs or a document still awaited. The caller owns any local fallback.
- The session expects `Hello` within five seconds, as its first frame, with a matching principal and protocol 1 or 2 in range. A gateway can keep one current session and get ten seconds of grace after disconnect. The table holds up to 4096 gateways, up to 4096 vetoes per gateway and eight kinds per worker per gateway. A `Hello` veto count must match the list. An unspecified veto kind, excess vetoes or an invalid lifecycle frame can end the stream. Old terms and replaced sessions can end calls. The reference client applies its term floor to decision and session answers. It waits up to five seconds per operation step, with a separately streamed watch.
- Workers registered over gRPC take one rank and one HTTP or gRPC wire. A new row must supply its role and wire. The service refuses a host-local wire, rank width above one, nonempty topology, or extra pairing descriptor fields. It accepts at most 128 labels per row, a 2048-byte URL, 256-byte names for model, runtime and zone, 128-byte label keys, 4096-byte label values and a 16 KiB encoded row. No count bounds a row's model IDs: the row's size does. The URL and model IDs cannot contain blanks or control characters. An encode offer's connector is any text, and the row's size alone bounds it. The service reads it past the blanks around it, and a blank one states none. A scrape needs an interval and uses the registered runtime's parser. The stock controller's load monitor reads an HTTP worker by the controller's own load interval, whatever the row's scrape interval. Probes default to three failures, two successes, a 60-second interval and a five-second timeout. Duplicate adds, missing IDs or URLs, stale revisions and another maker's row have separate statuses. A registration over a draining row is the URL's next incarnation, whatever its facts: a drain ends with the registration it named, and only a patch or a second drain keeps it.
- `WatchWorkers` allows four watches per principal. It can require a fresh snapshot when history no longer matches, and ends a stalled watch when polling resumes after 30 seconds. Worker watch pages hold up to 128 rows and 2 MiB of row data. Watches and `GetSnapshot` accept up to 128 model IDs. `GetSnapshot` can cap the worker rows returned and takes one snapshot at a time with up to 32 waiting. It refuses unknown or unavailable slices, invalid filter fields, and clear identifiers. `Explain` refuses a decision ID or an incomplete key and returns `NOT_FOUND` on a miss.
- A shared lease keeps a ring of 4096 watched events for replication. Load reports are not replicated. History gaps require a snapshot. The leader holds up to four snapshot restores at once and can release a stalled restore after 60 seconds.
- A call on the wrong scope returns `PERMISSION_DENIED`. A scoped call without a principal returns `UNAUTHENTICATED`. Leader-only operations can return `FAILED_PRECONDITION` with leader details, or `UNAVAILABLE` during a term transition. Unserved calls return `UNIMPLEMENTED` after admission.

### When a gateway loses its stream

A gateway asks the stock controller for decisions on a stream. A pick carries a **lease** that books its worker. A side that no dispatch names within five seconds is given up. A pair's decode side that was issued with its prefill is held while that prefill is dispatched. Once the prefill has ended, the side has at least five seconds to be dispatched.

A stream can end with requests unanswered. Two rules keep one lease standing for one request.

- **The stream's end.** A decide stream that ends gives up the lease of every decision it had not sent yet. No gateway could use them.
- **A request asked again.** A gateway whose stream ended sends each unanswered request again and marks it `again` (`DecideRequest.again` in [`contracts`](contracts)). The controller ends the lease of the earlier ask at once, where it still waits for a dispatch, and decides the request anew.

What a gateway must hold to:

- The mark is for a request whose stream has ended, and the gateway reads nothing more from that stream. On a stream that is still open the earlier answer can still arrive, and it would carry a lease the mark has ended.
- A request sent twice without the mark is two requests and two leases.
- A repeat that the stream answers `Busy` before a decider takes it ends nothing. The next ask of that request carries the mark again.
- The earlier lease is found by a 64-bit fingerprint of the request's id, among that gateway's own leases.
- A decode call is the exception to each rule above. The pair's lease it names stays: an answer that was not sent, or a repeat with the mark, gives back that call's one issue. A repeat without the mark is refused `decode_stands` while the decode side stands, and is a first call where it does not. No fingerprint is searched.

Two counters tell what happened, on the admin listener's `/metrics`:

| counter | what it counts |
|---|---|
| `smg_grand_central_decide_asked_again_total` | requests marked `again`, by `earlier_lease`: `ended` where the earlier lease still waited, `none` where no lease of the request waited |
| `smg_grand_central_decide_answers_dropped_total` | answers no stream took, by `cause`: `stream_gone` where the stream had ended, before the answer was made or with it still in the stream's queue, and `queue_full` |

After a stream's end most repeats read `none`: the stream gave the lease up, or the first ask was never decided. That is no error.

### Updating a document

**File.** Use a path that leads to a regular file of at most 128 KiB. The reader follows symlinks. A `.json` suffix selects JSON. The controller checks the file at startup. The watch polls after one second and reads a changed identity after two matching looks. A rejected update leaves the running generation in place and is logged and counted once per settled file state while leading. Unchanged documents count as neither apply nor rejection. A fast same-size edit can escape detection if the file identity does not move. An update that stays unstable can be rejected.

**Store.** `--spec-store` uses an existing directory with a lock and support for hard links. It runs with a process-local lease. Startup loads the newest file or starts with no document in an empty store. An empty store can report ready and shed requests until the first apply. Enforce persists before publishing and retains the newest eight files normally. A competing start waits up to the shutdown grace for the lock. Startup runs the newest file under the number in its name, also where the host's store already runs that document under a lower number. If the store refuses a generation after its file is written and the file cannot be removed, the apply fails as `not persisted` and says both. The store removes that file before its next write. A process that starts before that write reads it as a generation on file, and runs it if it is the newest.

**Generation.** A zero or omitted number starts at 1 and increments on change. A file can claim a higher number. Enforce through the store takes the next number. Changed content under the standing number, a stale number, or an out-of-order store number is refused. An identical document with the standing or automatic number is a no-op. The digest is a canonical `sha256:` digest of parsed content without the generation line.

**Calls.** `ApplyPolicySpec` requires a mode. Use `yaml` or `json` for the document format. `DRY_RUN` on either source checks the candidate under that source's rules. `ENFORCE` writes only under a store. Both can take the current `expected_digest` as a guard. A verdict can include before and after needs, or one issue with a field path when one applies. `AUDIT` is unserved. `GetPolicySpec` returns this replica's standing document, or a generation by number if held. It returns `NOT_FOUND` when none is held. `RollbackPolicySpec` under a store reapplies a named retained generation using a new number. Enforce and rollback under a file return `read_only` before document validation. Each dry-run and write queue has room for up to 32 waiting. Both use the same 128 KiB document cap as the file, with a separate bound on decoded YAML expansion.

**In process.** A host that embeds the store takes the same door with the crate's `operator` feature. `document::operator::DocumentManager::new(store, catalog, denied_headers)` makes it. `apply` takes the host's document as bytes, YAML or JSON. It checks the document over the host's catalog and denied headers, and runs it. A changed document that leaves `generation` out runs as the next generation. One that states a higher number runs under that number. `dry_run` checks a document and applies nothing. A refused apply counts once in `apply_counts`, and a dry run counts nothing. `document::operator::MAX_DOCUMENT_BYTES` is the 128 KiB bound, and `Format::named` reads `yaml` or `json`. `status`, `document` and `standing` read the generation that runs, and `apply_counts` the applies. `EnforceRefused` and `NotHeld` are errors a host can print. `NotHeld` says what was asked, what is held, what stands and what a spec store has on file. `document::operator::snapshot::worker_rows_page` gives the worker rows an operator reads. An address is told without userinfo, and so is a provider's key that is a URL. A provider's key that is no URL and holds an `@` reads as `<unparsable address>`. No spec store is bound in process. `writable` says so, and `enforce` and `roll_back` are refused there with `not persisted: no spec store is bound`. The spec store itself builds on Linux and macOS.

**A host's file.** A host that starts from a file names the file's form with `document::operator::Format::of` (a `.json` suffix selects JSON) and reads its bytes with `Format::read`, the read an apply passes too.

**Shared lease.** A standby follows the leader's document if replication is configured and working. It validates its own file at startup and watches without applying it while following. On promotion it may apply its file after it settles. A file already observed running since the last change is skipped. An unnumbered document the standby never saw run can take effect on promotion even if its file is older. An explicit generation keeps a stale number from taking over. A new leader awaiting its first document can return `Busy`. File lease renewals are durable writes. With the default timings, a slow renewal can stop decisions, close sessions or lose the lease. For slower storage, the lease flags can give more time, for example `--lease-duration-ms 15000 --lease-renew-deadline-ms 10000 --lease-renew-period-ms 2000`.

**Sends.** A profile's tally of sends (`StageBuild::dispatch_credit`: what the host sent each row since the row's last load report) carries over to the next document, when both have a stage that reads it. A `least-load` in the new document keeps the dispatches held under the standing one, so its estimate does not restart at a new document. The profiles of a document that set the same `sent_ttl_ms` and `sent_report_delay_ms` read one tally, and it carries over to the tally under the same two settings in the next document, whichever profiles read it there: a profile under a new name keeps the sends where its settings stand. Under the same two settings the two generations share the entries, and a send that one read releases has left the other too. A tally under two settings that no standing tally has shares none: it takes a copy of the sends the standing tallies hold, for each rank of a worker those of the tally that holds the most, so a changed `sent_ttl_ms` applies from the next read. A send that every standing tally has released is in no copy, and a tally starts empty where no tally stands. A refused document takes nothing. `MemoryStore::apply` makes the carry before it publishes, and `InProcess::apply` before its new document decides. A host with a store of its own calls `Engine::succeed` where it swaps engines. A dispatch reported to a served controller for a worker that its worker objects no longer hold at that rank adds nothing to the tally.

**A running engine.** `InProcess::apply` puts a document in force on an engine that stays. A worker list that stands across N documents keeps N values in its cell, and a read under the newest passes N - 1 of them, until the host publishes the list again. A dispatch said behind a new document for a decision made under an earlier one is counted once by every tally of sends in force, whether the tally stands since that document or was built since. The session table hears it. A pair's decode asked again behind a new document tells its issue to the stages that stood under the pair's own document. A header's name and a profile's name in a document are kept for the life of the process. `Members::standing` and `Members::facts_of` run under the lock a change of membership is told under: neither may call `forget`, `joined` or `apply`. `InProcess::spec` counts a reference at each call, and `DecisionEngine::with_spec` borrows the document for the call.

**What a new document keeps.** A profile keeps its stages where its name and its block are the same in both documents, and so are the document's `signals` and its `view`. One difference of the `view` keeps them too: another header list. No stage is built from that list. A stage that reads a header by a place its host wrote is kept as it stands: it reads that place of the new list.

A profile is built anew too where one of its stages, built again, says otherwise of itself than the standing one. That is another name, other words of a request's life to hear, or anything else the engine reads of a stage when it builds a profile. A factory of the host's own may build such a stage: build the same stage from the same text.

A tally of sends stands where a profile that reads it is kept. A profile built new beside it reads the same tally, and a send is counted once. A tally none of whose readers is kept is built new, and carries over as the paragraph "**Sends.**" says. Profiles under the same `sent_ttl_ms` and `sent_report_delay_ms` read one tally, and a document that states neither puts them all under one.

The keep stands between two documents validated over the stock stage table, or over one `StageRegistry` of the host's own as it stood at both validations. A host that builds its table anew for each document keeps nothing. A stage that a validation built for a kept profile is let go at the call.

**A replaced document.** A document that `InProcess::apply` replaces lives until its last decision ends, its last `SpecHandle` is let go, and each thread that read it last asks an engine again or ends: at most one replaced document for each thread. The stages the new document did not keep are dropped by the last of those holders: as a rule a thread that asks the engine, at its next entry of an engine (a decision, a lease's word or a read of the document) or at its end. A stage whose drop blocks holds that call meanwhile. A drop that panics there unwinds out of that call. It aborts the process where the call is made while the thread already unwinds, as a `DecisionGuard` dropped with a report owed speaks for its lease, and at a thread's end.

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

Set `explain: { sample: N }` to keep every Nth pick or shed per decider. Zero turns it off and one keeps every pick or shed. Each decider allocates its 1024-slot ring on its first sampled write and overwrites its oldest records. With sampling on, a decision takes its decider's ring lock only when it keeps a record. That write can wait for an `Explain` read of the ring. At R eligible decisions a second on that decider, a full ring covers about `1024 × N / R` seconds. A new generation restarts counting. Older samples can still be read under the same term. The remote lookup uses `gateway_id` and `request_id`. Use the subject the gateway authenticates as on the gateway listener for `gateway_id`, because `Explain` hashes it the same way as the decide stream. A record comes with its names: the plan, the profile, and that profile's picker, filters and scorers in their declared order. A shed made ahead of every profile names the plan alone. A subset hint of zero on a pool with members is one example. In process, `Trace::made_under()` gives the profile's position, or `None` for such a shed. On the wire, read `trace.no_profile` first: when it is true, the decision was made ahead of every profile and `trace.profile` is 0. A term change makes old samples unavailable on the service.

A record echoes the lookup key and gives the decision and its trace without a lease. It keeps up to three top scores. The control call keeps the record's original binding position and trace. `binding` is 0 when no binding matched or when the first binding matched. The name fields are empty for unmatched decisions and records from older generations. It leaves the decision ID, full candidate list, view digest, plugin projections and lifecycle outcome unset. It omits request facts, headers and session keys.

The admin `/metrics` endpoint exports the following series. A spec source adds the two spec series.

| Metric | What it measures |
|---|---|
| `smg_grand_central_state` | One gauge per `state` (`starting`, `leader`, `syncing`, `follower`, `draining`). The current state is 1. |
| `smg_grand_central_start_time_seconds` | Startup time since the Unix epoch. |
| `smg_grand_central_replication_synced` | 1 for a synced standby. |
| `smg_grand_central_replication_lag_seconds` | Seconds since a frame last left the standby at the leader's head, or 0 when leading or before such a frame. |
| `smg_grand_central_replication_restores_asked_total` | Standby's restore requests. |
| `smg_grand_central_load_polls_total` | Poll arbitration per worker and interval, by `mode` (`poll`, `fallback`, `suppressed`), for a worker on a connection. A worker is one rank of its engine, so the count differs from one an engine only for an engine of several ranks. |
| `smg_grand_central_load_poll_failures_total` | Failures of a poll over a connection, by `reason`. |
| `smg_grand_central_load_scrapes_total` | Load reads over HTTP that brought a report: the native load route or the `/metrics` gauges. The ranks of one engine share one read. |
| `smg_grand_central_load_scrape_failures_total` | Load reads over HTTP that brought none, by `reason`. The poll and scrape label values are `transport`, `answer`, `deadline`, `no_load`, `unsupported`, and `client`. |
| `smg_grand_central_load_rows_total` | Rank load rows of polled answers that the load monitor published. |
| `smg_grand_central_load_values_dropped_total` | Named values that the list bounds removed from the load monitor's polled answers. |
| `smg_policy_spec_generation` | Standing spec generation. |
| `smg_policy_spec_apply_total` | Applies by `result` (`applied` or `rejected`). |
| `smg_grand_central_decide_busy_total` | Busy answers by wire `reason`. |
| `smg_grand_central_decide_no_decision_total` | Unspecified Busy by `cause` (`fault`, `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`, or `document_awaited`). |
| `smg_grand_central_decide_hashes_cut_total` | Requests whose hash list exceeded what the generation reads. |
| `smg_grand_central_decide_answers_dropped_total` | Dropped answers by `cause` (`stream_gone` or `queue_full`). |
| `smg_grand_central_decide_asked_again_total` | Requests marked `again`, by `earlier_lease` (`none` or `ended`). |
| `smg_grand_central_decide_candidates_undialed_total` | Candidates removed by the asking gateway's wire constraints. |
| `smg_grand_central_decide_budget_outlasted_total` | Decisions cut at the final decision check because the budget or dispatch span was too short. |
| `smg_grand_central_members_outside_list` | Members not classified by a configured list, by `list` (`engines` or `zones`). |
| `smg_grand_central_overload_flips_total` | Overload verdicts of a worker that changed, either way. A lead's end takes every verdict off and counts each one. |
| `smg_grand_central_lifecycle_faults_total` | Panics contained while applying dispatch, completion, release, dispatch expiry or a decode call's issue. |
| `smg_policy_faults_total` | Panics the engine held and counted, by `kind` (`decision`, `event` or `upkeep`): the family an embedded gateway publishes. It is read from the store at each scrape and counted over every engine the store built, so it never starts again at a new policy document. `decision` is a decision that panicked, a shadow's beside a live decision among them. A live decision's panic also counts as `fault` in `smg_grand_central_decide_no_decision_total`. `event` is a stage that panicked at a change of membership. A stage's panic at a lease's word is held by the controller itself. It counts as `fault` where it unwinds the decision that told the word (the issue of a single request's or a pair's lease, the release of a lease the decision gives up), and in `smg_grand_central_lifecycle_faults_total` at every other word, a decode call's issue among them. `upkeep` is a run of the engine's upkeep that panicked at the controller's sweep. |

Plugin stats are available through the host APIs. Both kinds count answers, declines, exhausted decision budgets, breaker bypass and openings, stalls and observed events. WebAssembly also reports traps, fuel use and exhaustion, invalid replies, overlong calls, instantiations, lost events and unread timing. gRPC reports plugin Busy, invalid replies, overlarge pools, cold or refused frames, misses and their attribution, connections, stream failures, late answers and dropped observe events. Declared counters are separate from these host stats. [WebAssembly stats](src/plugins/wasm/plugin.rs) · [gRPC stats](src/plugins/grpc/plugin.rs)

## Gateway routing metrics

`Same` carries the classic names over.
The last column also records families whose embedded writers are pending. Four names in it
are planned, and no code writes them yet: `smg_policy_lease_expired_total{kind}`,
`smg_policy_leases{kind,state}`, `smg_policy_fallback_total{reason}` and
`smg_policy_deadline_misses_total`. The shipped dashboard uses the classic names. The
standalone controller keeps the `smg_grand_central_` prefix for its process metrics,
including role, replication, election, and load polling.

| Classic gateway | Embedded gateway | Class and plan |
| --- | --- | --- |
| `smg_worker_selection_total{policy}` | `smg_worker_selection_total{profile}` | Kept. The embedded profile label comes with the router's send path. |
| `smg_worker_overload_shed_total`, `smg_worker_overload_fallback_total`, `smg_worker_liveness_fallback_total` | Same | Kept. The embedded writers will use the decision's outcome and lifted veto. |
| `smg_pd_admission_waits_total`, `smg_pd_admission_sheds_total`, `smg_pd_admission_window`, `smg_pd_prefill_admission_inflight`, `smg_pd_prefill_admission_queued`, `smg_pd_prefill_admission_rejections_total`, `smg_pd_prefill_admission_wait_seconds` | Same | Kept. The gateway owns admission. |
| `smg_routing_key_source_total{source}` | Same | Kept. The gateway reads `header` or `rid` as the source. |
| `smg_cache_aware_policy_branch_total`, `smg_manual_policy_branch_total{branch}`, `smg_consistent_hashing_policy_branch_total`, `smg_prefix_hash_policy_branch_total` | — | — |
| `smg_policy_inflight_reconciled_total{policy}` | — | Planned rename to `smg_policy_lease_expired_total{kind}`. No code writes the new name yet. |
| `smg_cache_aware_match_ratio`, `smg_kv_index_lookup_seconds`, `smg_cache_placement_entries` | Same | Moved to the process that holds the index. A host observes the match ratio from the overlap a single pick of `InProcess` states of the chosen worker (`Pick::overlap`). The gateway observes it for a single request and once for a pair, from the placement the one call answers (`Placement::overlap`), where the request goes to the worker its pick's first target names. A pair's pick states what its prefill holds and nothing of its decode, so a pair observes once where a classic pair observes each `cache_aware` side. An encode request's pair states none yet, so it observes nothing. Inside a lent tree the pick states the tree's own two numbers, so the gateway divides what a classic tree decision divides. Three things still differ there. The count is the chosen worker's, a miss where it is no holder of the match, where classic observes the match whoever it chose. A request of more than 65,535 tokens or characters states both counts halved. A host's own prefix stage, or a plugin's, still reads a holder's whole blocks. |
| `smg_cache_tree_tokens`, `smg_cache_tree_chars`, `smg_cache_tree_tenants` | Same | Kept for the trees the gateway lends the engine: their size pass writes a token tree's tokens, a string tree's size and each tree's workers. Under an `index` entry of the longest cadence no pass runs, and none of the three is written for its trees. |
| `smg_manual_policy_cache_entries` | — | Ends with the classic manual cache. |
| `smg_cache_aware_prefix_margin_held_total` | — | Open. An embedded successor requires the scorer to report whether the margin held without an extra read. Otherwise the counter ends with classic routing. |
| — | `smg_policy_shadow_decisions_total{binding,side,kind}` | New with the engine. Shadow decisions beside live ones, read from the engine at each scrape. `binding` is the binding's position in the policy document from 0, and `15` stands for every binding from the sixteenth on. `side` is `single`, `prefill`, `decode` or `decode-again`. `kind` is `agreed`, `other-target`, `other-outcome` or `skipped`. A shadow that panicked is in the engine's fault count and not here. A classic gateway publishes no such family. |
| — | `smg_policy_faults_total{kind}` | New with the engine. Panics the engine held and counted, read from the engine at each scrape. `kind` is `decision`, `event` or `upkeep`. `decision` is a decision that panicked, a shadow's beside a live decision among them (`DecisionEngine::faults`). `event` is a stage that panicked at a lease event or a change of membership (`InProcess::unheard`). `upkeep` is a step of the engine's upkeep that panicked: its sweep, the registry's follow, a load report or a fill of worker lists (`InProcess::upkeep_faults`). Each kind has its series from the first scrape. A classic gateway publishes no such family. The standalone controller publishes it on its own metrics page, as its table of series says. |
| — | — | Planned additions. Embedded writers are pending for `smg_policy_leases{kind,state}`, `smg_policy_spec_generation`, and `smg_policy_spec_apply_total{result}`. The standalone controller already publishes the two spec families when it has a spec source. No code writes `smg_policy_leases` yet. |
| — | — | Planned renames of the standalone controller's `smg_grand_central_decide_busy_total` to `smg_policy_fallback_total{reason}` for fallback and of its `smg_grand_central_decide_budget_outlasted_total` to `smg_policy_deadline_misses_total` for decision deadlines. No code writes the new names yet. |
| `smg_router_requests_total`, `smg_router_request_errors_total`, `smg_router_request_duration_seconds`, `smg_router_tokens_total`, `smg_router_ttft_seconds`, `smg_router_dispatch_wait_seconds`, `smg_router_tpot_seconds`, `smg_http_responses_total`, `smg_http_request_duration_seconds`, `smg_http_inflight_request_age_count`, `smg_http_connections_active` | Same | Kept. Request and connection metrics. |
| `smg_admission_inflight`, `smg_admission_queue_depth`, `smg_admission_queue_rejected_total`, `smg_scheduler_admit_total`, `smg_scheduler_inflight`, `smg_scheduler_queue_depth` | Same | Kept. Admission and scheduling. |
| `smg_engine_running_requests`, `smg_engine_waiting_requests`, `smg_engine_token_usage`, `smg_engine_cache_hit_rate`, `smg_worker_health`, `smg_worker_cb_state`, `smg_worker_health_checks_total`, `smg_worker_retries_total`, `smg_worker_retries_exhausted_total` | Same | Kept. Worker reports and resilience. |
| `smg_kv_event_lag_seconds`, `smg_kv_event_gaps_total`, `smg_kv_event_resyncs_total`, `smg_kv_event_missed_batches_total`, `smg_kv_event_stream_errors_total`, `smg_kv_event_degraded_ranks`, `smg_kv_index_entries` | Same | Kept. Gateway cache event and index metrics. |
| `smg_symphony_outputs_total`, `smg_symphony_malformed_total`, `smg_symphony_fallback_total`, `smg_symphony_refused_total`, `smg_symphony_calls_suppressed_total`, `smg_tokenizer_cache_lookups_total` | Same | Kept. Output parsing and tokenization. |
| `smg_tokio_worker_busy_ratio`, `smg_tokio_event_loop_stalls_total`, `smg_tokio_event_loop_delay_seconds`, `smg_tokio_alive_tasks`, `smg_tokio_global_queue_depth`, `smg_log_dropped_lines` | Same | Kept. Runtime and logging. |
| `smg_allocator_resident_bytes`, `smg_allocator_allocated_bytes` | Same | Kept. Available when the gateway runs with jemalloc statistics. |
| `smg_pd_prefill_duration_seconds`, `smg_pd_kv_transfer_duration_seconds`, `smg_pd_decode_ttft_seconds`, `smg_pd_ttft_seconds`, `smg_pd_rendezvous_failures_total`, `smg_pd_bootstrap_failures_total`, `smg_pd_kv_transfer_failures_total`, `smg_pd_kv_connector_mode_total` | Same | Kept. Prefill and decode dispatch. |

## Presets

`PolicySpec::preset("least_load")` builds a complete document with generation 1. Each preset has one profile, the same three filters and a binding for every model. The copies below leave the number to the source. `PolicySpec::power_of_two` also accepts `ExpectedWait`, `Requests`, `LeastRequests` or `ResidentTokens { request_tokens }`. The resident variant ranks the full set by resident work. `pair_presets` and `epd_presets` compose side profiles from preset names and add `has-capacity` to the prefill side and `has-room` to the decode side. The standalone controller places a pair at its prefill, and reads both bits open. [Preset constructors](src/document/spec.rs)

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

Rotates through a worker list or a survivor set. A single request's first attempt, held to no part of its list, takes the next worker of its list's rotation where the filters keep that worker: at every list size over a list that has a counter of its own, as the lists of the engine over worker lists (`InProcess`) have, and from 16 workers over any other. Every other placement rotates through its survivor set, a retry among them: changes to that set change the rotation, and retained rotations are bounded.

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
        - header: x-smg-target-worker
          target: true
```

### `prefix_hash`

Uses the routing key first, then the prompt's prefix: every block from the first through the last whole block at the deepest reached boundary, or at the configured 256-token depth when no boundary is reached. On a ring the request's caller lends, the prefix ends at the token of that boundary or depth, and no block is rounded off. Two prompts that differ anywhere in that prefix take different keys, so they spread over the workers. When no key can be used, the in-flight-count scorer places the request. The binding names the routing key header as the key's source. A sessions rule added to the binding names no header of its own.

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

Blends warm-up, prefix reuse and least-load. The default KV gate is off. The stock controller reads cache events from each gRPC worker whose row asks for them and names a model, and its decisions read them. There the preset works from the cached prefixes of each worker the index holds a block for, and from worker age and its load monitor's reports. It works from the prefixes sent to every worker for a text request and while the model's index holds no block, and on a pair's side from the prefixes sent to each worker the index holds no block for. While the model's index holds no block, a single request warms no worker; a pair's side still warms a young one.

The preset's prefix stage names no source, so it reads `merged`. Set `source: index` to read cache events alone, or `source: predictor` to read sent prefixes alone. A standalone controller's decision reads the cache events of its own workers' streams, as a gateway's does.

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

The decision benchmarks time view construction and policy execution separately, typically at 8, 64 and 512 candidates. They cover presets, dispatch credit, overloaded pools, sticky hits and misses, spec validation, cache locality, sampling and prefix keying. The store benchmarks time membership writes, publication, update journals and pairing. Library benches also measure encode placement, pair and encode class construction, concurrent callers, the sent-prefix predictor, election, and both plugin kinds. The `decide_stream` bench measures loopback round-trip percentiles and throughput under offered load, separately from in-process work. [Benchmark sources](benches)

```sh
cargo bench -p smg-grand-central --locked --bench decision
```

| Gate | Work measured | Threshold by row |
|---|---|---|
| [Engine](../../.github/bench-gates/grand-central-engine.toml) | View build, decisions and spec parse/validation. | 10–15% |
| [Affinity](../../.github/bench-gates/grand-central-engine-affinity.toml) | Affinity decisions and selected sizes of view builds. | 10% |
| [Bucket](../../.github/bench-gates/grand-central-engine-bucket.toml) | Bucket decisions, including prefill across models and with stale rows. | 10–20% |
| [Power of two](../../.github/bench-gates/grand-central-engine-power-of-two.toml) | The preset's load and count variants. | 10% |
| [Prefix hash](../../.github/bench-gates/grand-central-engine-prefix-hash.toml) | Ring owner and load spill paths, with and without boundaries. | 10–15% |
| [Store](../../.github/bench-gates/policy-store.toml) | Load and membership writes, publication and selected pair and many-model cases. | 10–40% |
| [Store journal](../../.github/bench-gates/policy-store-journal.toml) | Journaled writes and journal reads. | 10–30% |
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

No count bounds a row's model ids, and no bound of its own holds an encode offer's connector: the row's size does. A control call's row is weighed as the wire frames it. A row that a host writes in its own process is weighed by the store. It charges 160 bytes for the row's numbers, and for each text its bytes and five more. It charges more where the worker states a fact: 22 bytes for a request's hard demands, 3 where it takes images and 11 where its encode offer states anything. A priority other than zero costs 7 bytes, and a drain window 7. That measure is never under the wire's weight, so a row the store takes fits a frame.

A host's registration and removal steps write rows at a membership door, `worker_host::MembershipWriter`. The door lays a found worker's facts over the row at its URL, marks a row draining and takes a row away. Only a row's maker lays facts over it, save for a refresh of its labels, and facts that name another URL than the call's are refused. A health loop's refresh of a worker has an entry of its own, `MembershipWriter::refreshed`. The labels are the engine's word, so the entry asks no maker and the row keeps the maker it has. The refreshed worker may differ from its row in the labels alone: one that states any other fact is refused, and nothing is written. The labels are bounded as a registration's are, and equal labels write nothing. A row that is draining takes no refresh. A removal names the row its asker saw, by id and revision, and the next section says who may remove whose rows. The crate holds one door, the controller's own, for a replica that may lead, and none over a store alone. The controller's door also refuses a fact that its log does not carry to a standby: a pairing descriptor, an empty zone, an encode offer's empty connector, and a zero where the row means none. It also refuses a worker of the host-local ZMQ wire whose row names no gateway, as it refuses a control call's registration of that wire. One gateway holds that socket, and such a row would stand in every gateway's pool. `worker_host::facts` reads a worker object back to a row's facts. The steps of the next section call the door. A controller built with the `discovery` feature runs them for the workers its discovery finds.

The controller probes its workers while it leads. Its binary is built with the `worker-host` feature, which holds the prober; a library build without the feature probes nothing. A row's health policy sets the check interval, the timeout and the two thresholds. The controller counts the interval and the timeout in whole seconds, one at least. A new row is probed at once, then once an interval. `success_threshold` passed probes in a row ready a row. `failure_threshold` failed ones take a ready row out, and three times as many more fail it. A pending row fails after ten times `failure_threshold` probes without becoming ready. A replica that does not lead probes nothing: its rows keep the status its leader's log tells. A row whose policy turns probes off is not probed, and it stays ready. Its door readied it at its registration, where the call's term held through its connect signal, or at an upsert of that registration. A probe removes no row.

## Running worker jobs over a local store

A host that keeps the membership store in its own process can use `registration_into_store` and `removal_from_store` with the `worker-host` feature. The steps write rows through the membership door, which the host hands them. The door must write through the holder's write calls (`Objects::upsert`, `Objects::patch`, `Objects::remove`), and the object holder writes the registry with each row. The host supplies the job queue and runs the workflows.

`write_workers_at_door` waits for creation or the host's last checks. It takes the three job modes. An `Upsert` registers an absent URL and refreshes a standing row. A `CreateOnly` job adds a row under a free URL, and fails with "already exists" where a row stands, whoever made it. A `CreateOnly` job of a worker of several ranks is refused by name before any rank is written, until the door lays a worker's ranks in one turn. A `ReplaceById` job is laid over the row behind the registry's worker of that id, at the revision the job names. It takes a worker of one rank: a job that builds several is refused before any rank is written, as the gateway's own replacement is. It fails where that worker left or moved. It also fails while that worker drains: a removal wins over a replacement that lands inside its drain window. A replacement that changes labels, checks or thresholds is made in place, and the worker keeps its id. One that changes the role, the wire or the models is the URL's next life, and the worker takes a new id. `registration_into_store` takes the object holder for that read. It is a step of the local kind: it skips a registration of another kind, whose workers the host's own steps write. API and startup jobs write as control calls. A discovery job writes as provider `discovery`, using the spec's address as its key. For example, a job for `http://worker:8000` uses that one key even if the job builds two ranks at different registered URLs. Health jobs cannot register workers. If one rank is refused, the rows written before it stand.

A removal selects workers, marks their rows with `drain_workers_at_door`, waits for them to settle, then takes their rows with `remove_workers_at_door`. The settle waits the drain window the worker was registered with, one day at most: its row carries a window that a registration states, and the object built from the row takes it. A found worker's row carries one only where it is not the default. A longer window ends the removal at the day with the worker still draining, and one job waits the longest window among the workers it drains. Each write checks the row id and revision. API and startup jobs can remove control rows. Discovery can remove rows of provider `discovery` under any key. Health uses the row's maker for its scope. `Disabled` permits no rows, `DiscoveryOwned` permits provider rows, and `AllLocal` also permits control rows. The controller's door includes the asker in its removal audit lines. A removal the door refuses for its asker fails at the drain with the door's sentence, where the job drained nothing: another maker's row, or a row outside the health scope. Each step drops from the job's lists a worker it leaves standing, so a host step behind the removal reads the workers it took or found gone. An object no row stands for stays in the lists for the host's own steps.

Each removal write has one attempt. A worker that an `Upsert` job finds again while its removal settles is registered anew: its row takes the URL's next life, out of the drain, and the pending removal leaves it.

The leader's load monitor takes the load records on a worker's stream, and a standby drops them. The `worker-host` feature has no load sink over the store, only the health checker's: `worker_host::HealthAdapter` hands probe outcomes to the worker objects and tells the store's rows the status each object takes. The controller's own service runs the health checker over `HealthAdapter` while it leads.

## What is not built yet

- A client that marks a request as asked again. The controller serves the mark. The crate's own client does not set it yet.
- A client that asks a served controller for a pair's later decode or for its decode call. The controller serves both. The crate's own client states neither yet.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- A pair's shadows at an encode request's first decision, beside a side its sessions rule or its picker's assignment places and through a served controller, and plugin stages in a shadow profile.
- `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd` that states `rule_placed: untold`, through the stock controller. Its file of leases keeps no word of a pair's side. In a host's own process a rule runs on both sides of a pair under either plan.
- The `best-with-temperature` picker, and three settings of `prefix-affinity`: `decay`, `accounting_ttl_ms` and `spill.uncached_credit`.
- The `bucket` scorer in the stock controller. The library runs it. The stock controller refuses a document that names it, because it does not yet report every request's end to a stage.
- The record of sent prefixes on a standby. A standby starts with an empty record.
- The wire's `prompt_text`. A controller does not hash a foreign host's text yet. It answers such a request `Busy`, as before.
- Cache events from an engine that asks for a key, or from a data-parallel worker's rows above rank 0. The stock controller builds its worker objects without credentials, so an engine that asks for a key gives it no stream. Every replica of the stock controller opens a cache-event stream to each gRPC worker whose row asks for events and names a model, and keeps those streams while the active document reads no index. A host built with the `worker-host` feature has the feed's two parts: `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row. A host that opens a row's feed without the row's cache group state clears the row's blocks and cursor first, so the subscription starts at the head of the stream.
- One more input in the stock controller. A worker's circuit-breaker bit is read from two places there: the worker's own object, and a veto of the asking gateway that holds, as its breaker's open window does. Such a veto takes the worker out of that gateway's decisions, and no rule lifts it. Its row reads as stalled too, as under every veto of that gateway, so a filter that reads the stalled bit refuses it. Nothing in the controller records a request's end on the object's breaker yet, so that bit reads closed.
- An engine connection over the host-local ZMQ wire. A controller built with the `worker-host` feature hands each worker object a gRPC connector alone, and an object connects to its engine at its first use. A worker that serves only the ZMQ wire gets no connection there, so nothing that controller reads from an engine reaches it for that worker. The controller sends such a worker no health probe: a row that names its one gateway is host-local, and the controller cannot dial it. Its row turns ready by its registration's connect signal, which a row whose policy turns probes off gets; with its probes on, nothing readies it yet. No field of a registration or of a worker object names a gateway yet, so the controller's doors refuse a worker of that wire.
- A probe with a key. The controller's probe of a worker carries no API key until a host hands the worker objects a keeper of credentials. An HTTP worker's URL userinfo goes out as Basic authentication, as the HTTP client sends it for every call. A gRPC worker's channel cuts the userinfo, so its probe carries no credential.
- A count of the controller's probes in its own metrics. The worker crate counts each probe in a recorder the controller does not install.
- A probe schedule kept across a change of leader. A new leader probes at once each worker that has no verdict on that replica, and each other worker after its own interval.
- A driver for the leader's own doubt of a worker. The store takes no such write, and a standalone controller's decision reads a worker's stalled bit on the worker's own object and in the asking gateway's vetoes. The object's bit is written by the worker crate's liveness check, which runs with its load monitor: a replica runs that monitor while it leads.
- A plugin named in the document. The format has no field for one.
- Two calls of the contract: `StepDown` and `WatchRegime`. The stock controller answers each as not served.
- Seven of the nine slices of the snapshot call, `GetSnapshot`. The stock controller serves the worker rows and the spec's status. It refuses a request for any other slice by the slice's name, and a request that names no slice asks for all nine. It also refuses a request for identifiers in clear.
- The audit mode of the apply call.
- A key for the workers discovery finds. The controller refuses one at its start. It keeps no worker key, and a replica that took the lead could not learn the key of a worker found before it: a pass registers no worker whose row stands.
- Flags for the timing of a discovery registration, and a found worker's cache-event word written again at an apply.
- A client identity and a private CA for the dial of a found worker.
- A registration that ends ahead of its dial where another maker holds the worker's row.
- Discovery in the stock image, which does not build the feature.
- A gateway that asks a standalone controller for its decisions.
