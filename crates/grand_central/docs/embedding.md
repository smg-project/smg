# Embed the engine

A host can link Grand Central and decide in its own process, with no service to call. This page covers the calls such a host makes, the signals its decisions read and the Cargo features it builds with.

The [policy spec's page](policy-spec.md) lists a spec's fields, and [Run the service](service.md) covers the standalone service. [REFERENCE.md](../REFERENCE.md) holds the rest of the detail, and calls the policy spec "the policy document".

## Decide in your own process

A host decides at one of two levels: the engine alone, or the engine with booking.

**The engine alone.**

- Validate a `PolicySpec` to get an `Engine`.
- Call `Engine::decide` with request facts, pool facts, rows, constraints and a reusable `Scratch`.

**With booking.**

- A host that needs booking makes an `InProcess` from an `AppliedSpec`. That type needs the `in-process` Cargo feature.
- The host asks `InProcess` for each decision, itself or through a `DecisionGuard`. It lends its worker lists with each request.
- `InProcess` runs on the caller's thread.
- A pick's lease books the pick's first target.
- A single request's pick may come with no lease. The engine issues one where something under the spec in force would hear the lease's words.
- The host takes a target, then reports its dispatch, and its completion or its release.
- Of a pick with no lease the host owes the engine no word. It reads the target from the decision (`DecisionGuard::decision`), since `DecisionGuard::take` answers `None` for such a pick.
- A host that hands a single list asks the engine itself, with `DecisionEngine::decide_under`. A place alone can answer it (`Decided::Alone`): the host sends to that worker and owes the engine no word.
- A whole answer (`Decided::Whole`) is held with `DecisionGuard::holding`. A guard has nothing to hold of a place: `DecisionGuard::decide_under` answers one as `DecisionRefused::Faulted`.

**The upkeep.**

- `InProcess::start` builds the engine over the host's worker registry, its load channel and its cache indexes. It also answers the engine's own task, which the host spawns once, on a runtime with its timer on.
- That task runs the engine's upkeep once a second. It also follows the registry and the load channel, both from its first run.
- On a runtime with no timer the task ends at its first poll with a panic. The engine decides on as one whose task was never spawned.
- An engine whose task is never spawned follows neither. A decision reads no load snapshot but one its request gives (`RequestMaker::loads`).
- Its stages hear of the registry at an `InProcess::apply` alone. That call tells each stage it builds new the workers that stand there, and tells a stage it keeps nothing.
- `DecisionEngine::sweep` is that upkeep alone. A host that runs no task calls it on its own timer. Such a host builds its engine with `InProcess::new`, hands no registry, and says each join and removal itself (`InProcess::joined`, `InProcess::forget`).
- Those two calls name a worker by its address. A host whose workers state ids of their own hands its registry, and spawns the task.
- A host that hands no registry says each standing worker's join again behind each `InProcess::apply`. A stage that call builds new has heard none.

**Pairs and refusals.**

- `InProcess` decides a pair when asked at its prefill: that one request decides both sides.
- A decode or an encode request is refused as `DecisionRefused::PairSide` where it is a side of its plan's pair, or hands more than one list.
- The engine catches a stage's panic in a decision and counts it (`DecisionEngine::faults`). A panic in the live decision answers `DecisionRefused::Faulted`, and a shadow's costs its request nothing.
- It also catches a stage's panic at a lease's word or at a change of membership, and counts it apart (`InProcess::unheard`). The request, the registration or the removal goes on.
- The drop of a retired stage can panic outside that catch: [Updating a document](../REFERENCE.md#updating-a-document) says where it unwinds.

**What a pick says of the prefix.**

- A single pick of `InProcess` states what its first worker holds of the request's prefix (`Pick::overlap`): the leading blocks held, the request's blocks and where the match came from.
- A request's count past 65,535 halves both counts together until it fits.
- `InProcess` reads them from that worker's entry of the rows its decision was given. Those are the rows it named from the host's data, a column the host filled, or the rows the host named.
- It asks no stage, so no stage's threshold applies.
- A worker the rows do not name is a miss of the request's blocks.
- A pair's first pick states the same of its prefill, which its first target names, from the prefills' column. It states nothing of its decode.
- Nothing is stated for a pair's decode decided again, nor for an encode request's pair.
- Nothing is stated for a pick the sessions rule keeps from the profile, nor under a profile that reads no affinity.
- The word on the prefix margin stays `false`.
- `Engine::decide` states nothing, and the controller's `Decide` answer does not carry it.

**Inside a tree the host lends.**

- A request decided inside a tree its host lends is given the rows the tree's match names, and the match's own two numbers.
- The two are the match's size and the request's, in the tree's unit: tokens or characters.
- A pick of a worker named among the match's holders states those two numbers, and names the tree as their source (`OverlapSource::Tree`).
- Nothing is rounded to a block. A request's size past 65,535 halves both numbers together until it fits.
- Any other worker is a miss of the request's blocks.
- Past the profile's sample the holders named are a draw.

[Engine example](../tests/unit/engine/framework/tests/profile_run.rs) · [InProcess API](../src/engine/in_process.rs) · [DecisionGuard API](../src/tracking/decision_guard.rs)

## A later spec

A host that keeps one `InProcess` for its life gives it a later spec with `InProcess::apply`.

- The call refuses a spec whose generation is not above the standing one's.
- The next decision runs under the new spec. A decision under way ends under the spec it began under.
- A decision made with a `SpecHandle` taken before the call still runs under the handle's spec.

**What stands across the call.**

- The session table stands.
- The record of sent prefixes stands as a rule: [The record of sent prefixes](#the-record-of-sent-prefixes) names the new specs that can start one anew.
- The stages of a profile whose name and text did not change stand too, with what they hold.
- That holds where the new spec changes no `signals`, and its `view` is the same or differs in its header list alone.
- A profile is built anew too where one of its stages, built again, describes itself otherwise than the standing one. A host's own factory may build such a stage.
- A host with a stage table of its own validates both specs over the one table. A table built anew for each spec keeps no stage.
- The paragraph "What a new document keeps" of [Updating a document](../REFERENCE.md#updating-a-document) has the rules, a tally of sends among them.

**What starts anew.**

- Every other stage is built new by its factory. A tally of sends it reads follows the reference's rules named above.
- Before the call returns, the engine tells each scorer and picker built new the workers that stand in the registry the host handed. A filter hears no membership.
- A stage that panics at that word is held and counted (`InProcess::unheard`), and the stages behind it do not hear the word.
- A stage that stands is told none of them again by the call. The engine's task reads the registry whole once more at its next upkeep.
- A stage that keeps load reports is told the reports at hand at the engine task's next upkeep.

**Reading the spec in force.**

- `InProcess::spec` answers a `SpecHandle`: a counted reference that reads as the spec, and keeps showing the spec it was asked under.
- Ask `place_for` and `not_alone` of the handle. Hand it to `InProcess::pick` and `InProcess::pick_among` with the profile it answered.
- A host that reads one word of the spec at each request asks `DecisionEngine::with_spec`, which borrows the spec for the call.

**A lease of an earlier spec.**

- What a stage hears of a lease, it hears of such a lease where it stood under the lease's spec and stands still. No stage built since hears it.
- Where its dispatch counts among sends, every tally of sends in force counts it once, kept or built new. A dispatch told while an apply is under way can miss a tally that apply builds new.

## Worker objects

A host built with `worker-host` can keep a worker object for each membership row.

- Make the holder with `Objects::new(registry)`.
- Write each row through the holder's write calls: `Objects::upsert`, `Objects::patch` and `Objects::remove`.
- A row registered by a decided change around the holder (`MemoryStore::follow`) stands apart from the holder. The holder's registration of its key is refused until a snapshot is laid whole.
- A standby and a `Watcher` also lay such a row at the next event that tells it and that the store takes.
- The worker object's status is the one that counts. A status told the store the same way does nothing to the object.
- A host writes a status on the worker object, and the store is told.

| What the host says | The call |
|---|---|
| A connect signal | `Objects::connect_row` |
| A drain | `Objects::drain_row` |
| A probe's verdict | `HealthAdapter` |

A `Watcher` lays the status its watch tells.

For example, a worker registered `Pending` can take requests once its connect signal makes it `Ready` and the host's breaker and vetoes allow it. The registry tells its subscribers each change. A promotion puts an open breaker on trial, as a health probe's promotion does.

## Cargo features

The crate defaults to `wasm`. Use `default-features = false` for the library without that runtime. The last column names the crate's own features that a feature turns on directly.

| Feature | What it adds | It enables |
|---|---|---|
| `wasm` | The WebAssembly plugin host. | |
| `in-process` | `InProcess`, the engine over a host's worker objects. | |
| `operator` | The operators' door for a host's own process. It builds no server, and it builds on every target. | |
| `worker-host` | An optional registry of worker objects, one for each membership row, and the cache feed. | |
| `controller` | The service, its protocol, its command line and the gRPC plugin host. | `operator`, `worker-host`, `in-process` |
| `client` | The reference client and the watcher. | `controller` |
| `discovery` | A worker discovery provider that the controller runs. See [Worker discovery](service.md#worker-discovery). | `controller` |
| `kubernetes` | The namespaced Lease backend. | |
| `test-util` | Support for the crate's tests and benches. | |

**The `worker-host` feature.**

- Its worker objects are written through the holder's write calls.
- The stock controller hands its holder a gRPC connector (`Objects::with_connector`), which makes each object's engine connection. Without one, an object of a gRPC or a host-local worker refuses its connect.
- The cache feed has two parts. `worker_host::KvAdapter` writes a worker's cache events to the index, and `worker_host::KvHost` gives each event subscription its row.
- A controller built with the feature fills its index itself. Every replica asks a gRPC engine for its cache events through one row: the engine's row of rank 0, or its one row with no rank. That row must ask for events and name a model.
- The stream is its engine's: a row of a rank above 0 opens no stream of cache events, and its engine's rank 0 stream feeds it.
- The stock controller is built with it: its binary requires `controller`, which enables it.

[Cargo features](../Cargo.toml)

## A decode chosen after its prefill

A host in process sets `request.constraints.decode_later = true` on a pair's request.

- The spec's word for it is `plans.pd-pair.decode: after-prefill`, which a host reads with `AppliedSpec::decode_pick`.
- The first pick names both sides and books the prefill alone.
- The host can decide the decode again once it has reported the prefill's dispatch, and before it reports the prefill's end.

### The calls on a guard

For example, after the prefill is sent, call `guard.dispatched(sent_at)` and then `guard.decide_decode_under(&maker, now)`. The maker lends the pair's two lists, as at the first decision. A call that lends none is decided over no worker, and the guard keeps the decode the first pick named.

- The new pick names the held prefill with decodes it can use, four at most.
- Its first decode is booked. The lease's dispatch deadline, the `due` of `guard.lease()`, moves to five seconds past the call.
- `guard.take_decode(index)` takes an alternate, and `guard.dispatched_decode(sent_at)` reports the send.
- A guard keeps the pick it holds where the call sheds or is refused.
- The call answers the decision the guard holds, so its answer does not tell a shed from a pick. `guard.decode_chosen()` does: it is `false` behind a pick, and still `true` behind a shed.
- A shed leaves the deadline where it stood.
- A host that skips the call can send a decode of the first pick: its dispatch books that side.
- A decode side the host gave up with `released_decode` takes no call after. While its prefill runs, a decode call then answers `DecisionRefused::Decode(DecodeRefused::DecodeStands)` and asks the engine nothing.

### What the engine answers the call

- The guard asks its engine with its own view of the pair, a `HeldDecision`. The crate alone builds that view: a guard lends its own, and a file of leases lends one of a pair it holds.
- A pick carries the decode side's lease under the pair's id, and a shed carries none.
- That lease keeps the generation and the time of the pair's first issue.
- The guard counts the decode's dispatch from the first decision.
- Behind a new spec the engine in force decides the side. It tells the issue to the stages it kept since the pair's spec, on a controller and in its host's own process alike.
- A refusal comes back as `DecisionRefused::Decode` with its reason, and a decision that panics as `DecisionRefused::Faulted`.

### What a stage hears

- A scorer or a picker hears a side's issue and its dispatch unless the sessions rule keeps that side's pick from the profile, or a stage panics as below. Each side of a pair answers for itself.
- A plugin bound as a scorer or a picker hears the events its spec lists (`PluginSpec::observes`, `GrpcPluginSpec::observes`), and none by default.
- A filter hears no word of a lease.
- A stage that keeps state by issue hears the issue of each decode it is told of, before that decode's dispatch.
- Under a pair asked with `decode_later`, a decode call issues the decode side. That issue is the one the stage hears.
- A decode sent with no call goes out as the first pick's. The stage hears its issue at that dispatch, just ahead of the dispatch's own word.
- A stage that a later spec built new hears none of these words: it never heard the pair's issue.
- The lease told at that dispatch names the target taken.
- At that dispatch a guard lends the stage the decision it holds. A controller's file lends each target's workers and ranks, with no trace beside the generation and the default row position, 0: there a stage reads a worker by its id.
- One issue can go unheard in a host's own process. Where a stage panics at an issue, the stages behind it do not hear that issue. `InProcess::unheard` counts it.
- Where a pair's two sides are issued together, a panic at the prefill side's issue leaves the decode side's untold to every stage. The guard still tells that side's dispatch and its end.
- A decode sent with no call has its issue and its dispatch told in one word. A panic at that issue leaves the dispatch untold to every stage.

**The date of a later decode's issue.**

- The lease of a later decode's issue is dated when its side is filed: at the decode call, or as its dispatch is told. A guard dates it by the instant its host hands, and the controller by its own clock at the frame's read.
- It is never dated at the pair's first decision.
- A controller dates it by its own clock, and never by the date a gateway gives the dispatch.
- The lease's later words keep the first decision's instant.

### A host that files leases itself

- A host that files pairs directly through `LeasesStore::issue_pair` can set the decode side to `LeaseState::Chosen`.
- The side keeps the pick's decode targets without booking a worker or starting a dispatch deadline.
- A side still chosen leaves the store when its prefill ends.
- Once the prefill is dispatched, `store.issue_decode(gateway, id, decode, now)` can replace that choice once, while the prefill is held.
- It books the new first target and gives the decode five seconds to dispatch.
- A dispatch on a chosen side can also book a target from the original pick. An index outside the pick books none, and the side stays chosen.
- `LeasesStore::expire` returns both sides as the sweep left them. For example, a completed decode stays `Completed` when its prefill is released as abandoned.

`issue_decode` refuses with a reason:

| `DecodeRefused` | When |
|---|---|
| `NoLease` | The gateway has no lease under that id. |
| `NotChosen` | The lease was filed without a chosen side. |
| `PrefillNotHeld` | Its prefill does not stand dispatched. |
| `DecodeStands` | Its chosen side has already been issued or dispatched, or its gateway ended it, or it took as many calls as one lease takes. |

A guard answers the same four from its own note of the pair. A guard over a link asks its controller nothing and is refused: such a guard sends no decode call.

## Signals a decision can read

The engine reports a profile's input requirements through `Needs`. The host supplies the request and worker views. Native stages can read the full view. Plugins see the projection their host configures. See [the Rust view types](../src/engine/view.rs) and [the wire records](../contracts/policy_plugin.proto).

| Signal | Where it comes from |
|---|---|
| Model, phase, tenant, namespace seed, priority, streaming, output limit, prompt size, media counts, retry attempt and seed | The caller's `RequestFacts`. |
| Request hashes and headers | The caller: hashes of token blocks or of text windows, and headers as positions in `view.headers`. |
| Ready, draining, stall, breaker and reachability | Membership and health probes, plus the asking gateway's vetoes and wire constraints. |
| Running, queued and waiting uncached work, KV usage and active KV usage, drain rate, resident tokens, admission window, sample age and source | Host load reports. |
| Host requests in flight and dispatch credit | Lease bookings and lifecycle reports. |
| Pins, requests in flight for one key, and active-key counts | The host's session facts, or the engine's session rule. |
| Cache depth, age, growth and size | An affinity column or sent-prefix predictions the host supplies, or the standalone service's own ingestion. |
| Role, ranks, wires, stable worker key, zone, engine family and labels | Membership. |
| Capacity, room, pair and encode compatibility, worker vision | Admission bits and worker facts the host supplies. |
| Pool generation, size and membership, reachability and host constraints | The pool view and `Constraints`. |
| Named values the spec declares under `signals` | Worker load reports, pushed or polled. |

The lists below take the rows in that order.

**The request's own facts.**

- A remote caller sends these facts in `DecideRequest`.
- An otherwise valid frame with raw prompt text receives `Busy`.
- The built-in binding matcher reads model and phase.

**Hashes and headers.**

- The caller sends token-block or text-window hashes, their kind and their block size.
- A token request may leave the size out, and then reads the spec's.
- A text request states its window's size in bytes. Over the wire, one that carries hashes and states none is refused by `facts.block_size`.
- The applied spec bounds the prefix read (`AppliedSpec::block_hashes_read`). A host hashes that many blocks and no more, or hands the first that many of a longer list.
- The caller projects headers as positions in `view.headers`, and resolves the routing or session key.

**Health.**

- A liveness veto is a doubt.
- A held veto represents the asking gateway's breaker, and the liveness fallback cannot lift it.
- The standalone controller reads a worker's breaker on the worker's own object, beside the asking gateway's vetoes.

**Load reports.**

- The standalone leader's load monitor polls the load of each ready worker the controller dials. The ranks of one engine that are due share one call.
- A row that one gateway alone reaches is asked for no load.
- Over gRPC it calls the load RPC. Over HTTP it reads the worker's native load route, or the `/metrics` gauges of a supported runtime.
- It also takes the load records on a worker's stream, and asks no load of a worker whose records are fresh.
- Its decisions read those reports. A spec that holds a stage that keeps reports is given none yet.
- The store holds no load row, and a report's fields depend on its reporter.
- A worker whose load call fails keeps its last report, where that report states a running or waiting request. The report stays as old as its sample.
- That report stands until the worker's load call answers again, a load record comes on its stream or the worker leaves.
- It stands no further than the first load poll that finds the sample 30 seconds old. Past that poll the worker has no report.
- Use `view.max_report_age_ms` to age out reports for the affected scorers and KV gates.
- Overload is a separate latched verdict.

**Requests in flight.**

- Read counts through `RequestFacts::in_flight`.
- The load scorers can account for dispatched work until a report covers it.

**Sessions.**

- The standalone controller's engine keeps a session table of its own, one for the controller's life.
- It counts a key's requests on each worker's object, by the leases it holds.
- A lease filed under a key counts for that key, under any sessions rule and under none. A single's counts on its worker, and a pair's on the worker of each side that is issued.
- A record's count of active keys counts the sessions of those leases.
- No row is premarked as a session hit.
- Its session rule keeps pins using the caller's key.

**The cache.**

- An embedded host can supply an affinity column itself. It can also supply sent-prefix predictions.
- With regular standalone ingestion, worker age is available.
- The stock controller records the prefixes it sends. [The record of sent prefixes](#the-record-of-sent-prefixes) has the rules.
- The stock controller asks its gRPC engines for cache events, by the rows [Cargo features](#cargo-features) names. A controller of your own built with the `controller` feature does too.

**Membership.**

- The spec projects labels to bits, and zone and family to positions from 1.
- An unlisted or absent zone or family is 0.
- So is every zone in a standalone controller's decision, where a worker's object states none.

**Capacity and classes.**

- `MemoryStore` holds no candidate row: it knows no admission window and derives no class.
- A host derives pair classes itself, by `PairClasses::derive`, and encode classes by `EncodeClasses::derive`.

**The pool and its constraints.**

- The pool view and `Constraints` carry the pool's facts, exclusions, vetoes, a wire mask and an optional subset hint.
- A standalone request can narrow by common wires, or state a zero subset to exclude the pool.
- The controller applies both ahead of the stages. The stages are shown the places the gateway may take, no wire mask and no positive hint.
- Hosts can supply further pool facts, such as a regime or zone capacity. The standalone view supplies neither.

**Named signals.**

- The values are the ones a report states by name, and the two fixed token counts.
- `InProcess` keeps none. It reads them in the load snapshot its decision holds: for a single request, for each side of a pair and for each side of an encode request.
- A native stage that resolved a signal's column reads them.
- In a standalone controller's decision that snapshot is the one its load monitor last published. No snapshot is given under a spec that holds a stage that keeps reports.
- No plugin contract carries them, and a standby does not follow them.

### Feeding the store and the cache index

- An embedded host can implement or use the store's ingress traits for health and leases.
- One that keeps no worker objects reads its membership through `WorkersStore`, which holds no write, and writes a row as a decided change (`MemoryStore::follow`).
- It feeds cache events through the worker host, under the `worker-host` feature.
- It hands the holder of its worker objects the KV tables (`worker_host::KvTables`, `Objects::with_kv_tables`). `worker_host::KvHost` over that holder gives each event subscription its row.
- The holder makes a row's lane where it states the row's facts, and retires it where the row leaves. A row has a lane under its first model, while its signals ask for events.
- A row stated while its model's index is full has no lane until a later statement of its own finds room.
- The crate's adapter alone writes blocks through a lane: nothing outside the crate writes the tables.
- A holder handed no tables makes no lane, and its host indexes nothing.
- The tables hold one event-fed chain index for each model, for up to 1024 workers each.
- The adapter reports sequence gaps, and supports replay and resync.
- A snapshot that its stream cuts short keeps the blocks it had shown: the next snapshot's start replaces them.
- A host reads the tables through four calls. They answer a rank's cursor (`KvTables::cursor`), a row's indexed blocks (`KvTables::indexed_blocks`), the resyncs counted under a reason (`KvTables::resyncs`) and a model's block size (`KvTables::block_size`).
- A controller's tables are at `Controller::kv_tables`.
- A host can read the validated `index` settings through `PolicySpec::index_for`, and use them to configure its own index.
- The KV tables use the same index layout for every spec.

### The record of sent prefixes

- A standalone controller's engine keeps a record of the prefixes sent to its workers. It keeps the records by model, for a pool whose prefix stage reads sent prefixes.
- A stage that says `source: index` reads cache events alone, and no such record.
- A decision can write a placement there: a single request's, and each side of a pair's.
- It writes for the first target its maker says the caller sends to. On the stock controller that is the pick's first. Under an encode request it writes none where the maker says the caller sends to a placed encoder no more.
- It writes none for a pick the sessions rule keeps from the profile.
- A model's record takes its room from the index's `max_size`, and has room for one placement at least. The index's `kind` says how the record is keyed. [Index](policy-spec.md#index) has both rules.
- A placement is a prediction made at the decision.
- A decision turned back after its lease's issue keeps the placement it wrote. A later request with that prefix can read it as a predicted hit, which must exceed the stage's `threshold`.
- A decision does not wait to write its placement: a write that meets the record's lock held is kept by its thread and written behind the thread's next picks. A write that eight newer ones push out, or that its thread ends with, is left out.
- Which record names a request's holders, this one or the model's KV tables, is by the profile's prefix stages and by the request. [The cache records](design.md#45-the-cache-records) in the design has the rule.
- The store tells the engine each word of its membership. The record forgets a worker that left, was replaced, stopped serving a model or changed its role.
- As a rule the record stands across a new spec. A new spec can start a model's record anew: one that states another `kind` for the model's index, or other `view.boundaries` under a `hash` index.
- A standby starts with an empty record.
- Under a `tree` index a text request is recorded as a token request is, by the hashes of its windows. Under `hash` it is not recorded.
- Readers of the record wait for a writer. Under many callers that never pause, a fleet whose requests mostly grow decides slower than one that reads no record.
- A host with a store of its own keeps a `tree` index's record with the crate's `PrefixPredictor`. It folds a request's keys once (`PrefixPredictor::fold_into`), reads by a shared reference, and writes with a tick of the host's clock.

### Hashing a request

- `kv_index::salt::request_text_hashes_with_seed` makes a text's window hashes under a cache namespace's seed. `engine::TEXT_BYTES_PER_TOKEN` is the four bytes a token.
- Under one namespace's seed, a window of text and a block of token ids over the same bytes hash apart: the text kind is folded into the seed.
- A host hashes a request by its model's block size. That is the size the KV tables have on record for the model (`KvTables::block_size`), and the spec's `view.block_size` where they have none.
- The tables take the size from the model's event subscriptions. A worker's registration there can state it, and its engines' cache events state their own. The doc of `KvTables::block_size` says which of the two stands.
- A worker need not state it.
- The size is dropped where the model's last worker with an event subscription leaves.
- A model first hashed by the spec's size and later by a size its engines state can lose the prefixes sent before, until they are sent again.

[Store API](../src/store/mod.rs) · [Sent-prefix predictor](../src/tracking/prefix_predictor.rs)

## Keys and shadows: a host's calls

These calls belong to a host's own code, not to a spec.

**Keys.**

- The host resolves a key and supplies it as `RequestFacts::session_key`, or as `facts.session_key` on the wire.
- The header fields say what the host reads. Sending the header in a `DecideRequest` alone creates no key.
- A host reads a binding's sources with `AppliedSpec::key_sources_for` and derives a request's key with `KeySources::derive`.
- A host in process does both inside `InProcess`: through `InProcess::place`, or with its own `engine::RequestMaker` through `InProcess::decide_under` or `DecisionGuard::decide_under`.
- `InProcess` hands `RequestMaker::make` the applied spec of the world the decision is made under. So the key's sources and the rule that pins the key are of one generation, whatever is applied meanwhile.
- A host derives the client's key with `KeySources::derive_client` and states it as `RequestFacts::client_key`.
- The wire's request facts carry it to a standalone controller as `client_key`. So a controller at an address places a request with no session key on the plain ring by its client's key, as the host's own process does.
- A gRPC plugin's frame carries zero in that field under every exposure. A WebAssembly guest's request record has no such field.

**Shadows.**

- A host reads every shadow count in one call, `InProcess::shadow_counts`. The fault count is `DecisionEngine::faults`.
- A host that drives the engine itself runs a single binding's shadow through `Engine::shadow`.
- It makes the request's live decision through `Engine::decide_keeping`, and calls `Engine::shadow` with that decision's scratch.
- Under a sessions rule the shadow then starts from the session pins that decision read, whether or not the decision is committed first.
- With a scratch that holds no reading of that decision, it reads the pins as they stand.
- A host that shadows a request decides it through `Engine::decide_keeping`, each time it decides it. A debug build can stop on a mix of the two entries on one request.
