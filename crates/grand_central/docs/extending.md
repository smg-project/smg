# Extend the engine

A host can add policy code of its own in three ways. This page covers each way, what a plugin is shown and what happens when one fails.

| Way | What you write | Where it runs |
|---|---|---|
| [A Rust stage](#register-a-rust-stage) | A `Filter`, a `Scorer` or a `Picker`, and its factory in the host's stage catalog. | In the host's own binary, trusted as a stock stage is. |
| [A WebAssembly component](#load-a-webassembly-component) | A component for the `smg:policy` package. | Inside the host, under a memory cap and a metered budget. |
| [A gRPC plugin](#call-a-grpc-plugin) | A `PolicyPlugin` service. | Behind a Unix socket, or TCP with mutual TLS. |

Host code sets up both plugin kinds, and [Plugin inputs and fallback](#plugin-inputs-and-fallback) covers what they share. [The built-in stages](stages.md) lists the stock stages. [Embed the engine](embedding.md) covers a host that decides in its own process.

## Register a Rust stage

Implement `Filter`, `Scorer` or `Picker`. Add a factory to `StageRegistry::builtin()` and hand the catalog to `PolicySpec::validate_in`, `Controller::bind_with` or `service::controller::run`. Each stage name is unique across the catalog.

| Call | It registers | Its entry in a spec |
|---|---|---|
| `scorer` | A scorer that takes a `weight` alone. | The bare name, at weight one, or `{ name: { weight: 2 } }`. |
| `scorer_with` | A scorer that reads parameters beside `weight`. | The bare name, or the name mapped to its parameters. |
| `filter` | A filter that takes no parameter. | The bare name. A map under it is refused. |
| `filter_with` | A filter that reads parameters. | The bare name, or `{ name: { .. } }`. |
| `picker` | A picker that takes no parameter. | The bare name. A map under it is refused. |
| `picker_with` | A picker that reads parameters. | The bare name, or `pick: { name: { .. } }`. |

- A factory handed to `scorer`, `filter` or `picker` runs once at a registration that finds its name free: its stage's name is checked there. It runs again where a validation reads an entry that names the stage, a dry run too.
- Under one of the engine's own refusal names, such as `binding`, such a factory runs at registration alone: validation refuses a spec that names the stage.
- A factory handed to `scorer_with`, `filter_with` or `picker_with` does not run at registration. [A stage with parameters](#a-stage-with-parameters) says when it runs.
- Keep a factory's effects inside the stage it returns.
- A stage's `name()` must return the name it is registered under. `scorer`, `filter` and `picker` refuse another name at registration. Validation refuses it at every build, for all six calls.

### The example program

The [example program](../../../examples/grand-central) registers a scorer with `stages.scorer("pack", || Box::new(Pack))?`. The scorer favors workers with more reported running requests. Its spec names it in `score: [pack]`, and uses `not-overloaded` to steer away at the threshold.

```sh
cargo build --release --locked -p grand-central-example
target/release/grand-central-example --controller-id controller-0 \
  --spec-file examples/grand-central/policy.yaml
```

### A stage that draws

- A filter or a scorer you register opens a stream of its own where it draws: `Draw::for_stage(facts.seed, self.name())`.
- The seed's own stream is `Draw::new(facts.seed)`. `best` reads it between tied rows.
- A stage that draws one of two from that stream can draw the very number `best` draws between two tied rows. The two choices are then one draw, not two.

### A host with a store of its own

- A host that holds a store of its own, with no controller around it, validates a spec over the same catalog.
- `AppliedSpec::validate_in(&document, &stages, &denied_headers)` returns the generation that `MemoryStore::apply` takes.
- The host numbers its specs with `generation`. The store takes a number above the one that stands, so it refuses a spec with no number.

### A stage with parameters

A stage can take settings of its own from the spec. Register it with `scorer_with`, `filter_with` or `picker_with`. The factory reads the entry's parameters into a type of its own and returns the stage, or refuses a value.

**A scorer.**

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

**A filter.** Write its entry as the name mapped to its parameters.

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

**A picker.** Write the pick entry as the name mapped to its parameters.

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

**The rules the three kinds share.**

- A bare name hands the factory an empty map.
- A scorer's factory and a filter's run once for each entry that names the stage, as a validation reaches it: at a start, an apply or a dry run. A picker's runs likewise, once for each profile that names it.
- Under one of the engine's own refusal names such a factory runs for none.
- For `scorer_with`, `filter_with` and `picker_with`, a name mapped to a value that is no map is refused ahead of the factory. So is a `scorer_with` entry whose `weight` is not a finite number above zero.
- A value the factory refuses with `StageError::bad` is reported at its field's path in the spec. A refusal by `StageError::other` is reported at the stage's own path, which ends at its name. So is a parameter's deserialization error that the factory passes on.
- Put `#[serde(deny_unknown_fields)]` on the parameters' type. Without it a struct like those above ignores a misspelled key: a field with a default takes it, and a required field is reported missing.
- The keys of a parameter map carry no order. The factory reads them in key order, and two files that differ in that order alone have one digest. A key written twice is refused.
- A stage is asked what it needs while the spec that names it is built, and not at a decision. The engine keeps the answer, so a stage's `needs` must give the same answer every time.

**Where the three kinds part.**

| What | A scorer | A filter | A picker |
|---|---|---|---|
| The map its factory gets | The entry's map less `weight`, which is the profile's. | The whole map: a filter has no weight. | The whole map: a picker has no weight. |
| The path of a value refused with `StageError::bad` | `profiles.<profile>.score[<index>].<name>.<field>` | `profiles.<profile>.filter[<index>].<name>.<field>` | `profiles.<profile>.pick.<name>.<field>` |
| Stock stages that take parameters | Every stock scorer: its `weight` at least. | `has-room` alone. | `best` alone. `best-with-temperature` is read and refused: it is not in this build. |

- Whatever a spec writes under a stock stage that takes no parameter is refused.
- A Rust caller that makes a `ProfileSpec` by `ProfileSpec::of(pick)` and assigns its filters holds them as `document::FilterSpec`. `"healthy".into()` reads as before. A `String` pushed or compared does not: use `.into()` and `FilterSpec::name()`.
- A Rust caller that matches on `document::PickSpec` adds a wildcard arm: the type is non-exhaustive and holds `Added`.

### What a factory can ask

A factory of the three `_with` calls takes a build context, `StageBuild`. It reads its parameters there, and it can ask three things more.

| Call | What it hands |
|---|---|
| `build.dispatch_credit()` | The profile's tally of sends, a `DispatchCredit`. |
| `build.load_report_reader()` | The view's reading of a load report, a `LoadReportReader`. |
| `build.signal_column(name)` | The column of a signal the spec declares. [Named signals](#named-signals) covers it. |

**The tally of sends.**

- A factory of any of the three `_with` calls can call `build.dispatch_credit()`. The tally holds the host's sends to each row that still count, as the rules below say.
- A profile may set `sent_ttl_ms` or `sent_report_delay_ms` only where one of its stages asks for the tally. Otherwise the validator refuses the field.
- The stage reads a row's `CreditedDispatches { requests, tokens, unsized_requests, estimated_tokens }` with `read`.
- `tokens` holds the prompt tokens the sends stated, each send's count held under 2^31. `unsized_requests` counts the sends that stated none, and the stage chooses their price.
- `estimated_tokens` is one price: the text bytes the host stated for each such send, over four, summed.
- The type is open to further numbers: code outside the crate reads the fields and matches them with `..`.
- A scorer that takes that price reads the pair `DispatchCounts { requests, tokens }` with `pass(facts).counts(row)`. Its `tokens` are the stated ones and the estimates together, as the `sent.` terms read them.
- As a rule a read that releases no sends reads three words for each row that holds sends, where the whole read reads four.

**When a send stops counting.**

- A send is a dispatch the host reports.
- As a rule it stops counting at a report of the row sampled at or after it, or at the profile's `sent_ttl_ms`.
- Membership words end it too, among them a removal, a replacement or a resize of the row's worker and a join told of it.
- Where the profile sets `sent_report_delay_ms`, the report that ends a send is sampled at least that long after it.
- In a standalone controller's decision the report is its load monitor's. A spec that holds a stage that keeps reports is given none yet, so there no report ends a send.
- When a send arrives with 8,192 or more held on a row, the tally lets one of the row's sends go at any age, or none. As a rule it is the oldest. None goes where the row, the new send counted, holds no more than 8,192.

**The load report reader.**

- A factory of any of the three `_with` calls can call `build.load_report_reader()`.
- A scorer keeps the `LoadReportReader` value, opens one pass a decision with `pass(facts)` and asks `of(row)` for a row's report. A filter calls `read(facts, row)` in `keep`.
- Under `view.max_report_age_ms` a report older than the limit reads as none, as it does for the built-in scorers. A report exactly at the limit is still read.
- A spec that sets no limit reads every report. A stage that reads `row.load` itself gets the report at any age.

### Named signals

A stage can read a named value that the spec declares under `signals`. The factory asks the build context for the signal's column and the stage keeps it. In a decision the stage reads a row's value with `SignalColumn::read(facts, row)`.

```rust
stages.scorer_with("least-pending", |build| {
    let pending = build.signal_column("pending_tokens")?;
    Ok(Box::new(LeastPending { pending }))
})?;
```

- The read returns `None` when the decision's facts hold no cells, when the row has no value, and when the value is older than the signal's `max_age_ms`.
- A filter registered with `filter_with` resolves a column the same way, and so does a picker registered with `picker_with`. A stage registered with `scorer`, `filter` or `picker` resolves none.
- A stage that keeps a column sets `signal_cells` in the `Needs` it declares. Validation holds the two together: it refuses a stage whose factory resolved a column without the need, and one that sets the need where its factory resolved none.
- For a name the spec does not declare, `signal_column` returns an error. A factory that passes it on with `?` has its stage refused at the stage's entry.

**A host that asks `Engine::decide` itself builds the cells.**

- It hands the values beside the rows: `RequestFacts::signal_cells` for a single pool, `PairPools::prefill_signal_cells` and `PairPools::decode_signal_cells` for a pair, and `EncodePool::signal_cells` for the encoders.
- A pair decision and an encode pair read each pool's cells: cells on the request are not read there.
- `SignalCells::new(cells, signals)` takes one cell per declared signal for each row. The cells stand in the rows' order, with a row's cells in the order of the names.
- `AppliedSpec::signal_columns()` lists the declared signals in that order.
- A `SignalCells` owns its cells. A host keeps the one it built and refills it in place through `cells_mut()`: it builds no new one for a decision.
- The count of cells per row stays as built. For a spec that declares another count of signals, build a new `SignalCells`, which can take the old one's `Vec`. A spec that declares no signal needs no cells.
- A value is finite, of magnitude at most 1e15. Where a host has none, or its source states one outside that bound, it writes `SignalCell::NONE`. It makes every other cell by `SignalCell::of(value, sampled_at_micros)`, which checks nothing.

**`InProcess` makes the cells for you.**

- A worker's load report can state values by name, pushed or polled. The engine reads the ones the spec declares, where the report states them under the declared unit.
- The two fixed token counts are offered under `kv_tokens_used` and `kv_tokens_capacity`, in `tokens`.
- A host of `InProcess` builds no cells, for a single request, for a pair or for an encode request. That engine reads the values in the load snapshot its decision holds. Where the request's profile reads a named signal, cells on the request are set aside.
- Each side of a pair reads its own list. An encode request decided over its three lists reads them a side too: its pair's two sides as a pair's, and its encode side from its encoder list.
- The encode side reads them where the encode profile of its pair's binding reads a named signal.
- That engine honours a value's own age and its signal's `max_age_ms`. The bounds of a report's list are kept where the list is read off the wire.
- A decision of that engine reads what its snapshot's reports state. So a pushed record that leaves a name out reads as no value until a report states it again.
- The two fixed token counts are the worker's report's to keep. Behind a record that leaves one out, a rank is still read its last stated count, until the signal's `max_age_ms`.

**A worker of several ranks under `InProcess`.**

- A worker object that stands for several ranks reads one value a name. Its ranks' values are made one as `across_ranks` says, under the oldest sample among them: the worker's report states each rank's own sample instant.
- Its report is read where it lists as many ranks as the worker's registration states it stands for: its `dp_size`, with no rank of its own.
- A report of several ranks is read of no other worker, and a report that lists fewer ranks than it counts gives no value.
- A report of one rank is read as the worker's whole. So the first record of a worker of several ranks stands for it until a second rank's record.
- A worker registered one row a rank reads each rank's own values, pushed or polled.

**What to expect of a value.**

- A value is as old as its report. A value reported with its own age is that much older.
- The stock controller's load monitor dates a polled report when it takes the answer in.
- It takes a group's answers in by batches. So a report can be dated some 20 ms after its answer came, and more on a busy host.
- A polled report that keeps a rank's pushed record takes the earlier of two instants. One is the poll's own. The other is that of the last pushed record the monitor took for the worker, of whatever rank. Each kept rank keeps its own record's instant.
- Where one row's report lists several ranks, the row has a value only where every rank it lists states the name under one unit. The listed values are then made one as `across_ranks` says, under the oldest sample among them.
- A worker that pushes its load dates each rank by that rank's own record. So a rank that stops reporting ends the row's value once its last record is older than `max_age_ms`.
- A report offers the two fixed counts as `kv_tokens_used` and `kv_tokens_capacity` where its source states them, zero too. A rank with no token capacity offers neither fixed count. A value its list states under one of the two names stands in the fixed count's place.
- A worker that pushes its load also offers a count its record leaves out, for each of its ranks: the count as that rank last stated it. That count carries the age the monitor measures since that statement, rounded up to a millisecond and cut at 4,294,967,295 ms.
- So declare `max_age_ms` on a signal that reads one of the two counts.
- A metrics scrape states no value by name.
- A report's list off the wire is held to the worker crate's bounds: a pushed record's where the record is read, a polled answer's where the load monitor takes it.
- Those are its first 64 entries, each with a name of 1 to 63 bytes, a unit of at most 31 bytes and a finite value. Of a name stated twice, the first entry inside the bounds stands.
- The controller counts what its own poll leaves out in `smg_grand_central_load_values_dropped_total`.

**Values behind a change of lead.**

- Values are not replicated, and neither are load reports or overload verdicts. A controller that follows holds none, and a lead's reports end with the lead.
- A controller that takes the lead starts its load monitor empty, and its first poll goes out at the start.
- Until that poll lands, or the first load record on a worker's stream, a spec that reads them decides by each stage's fallback. No worker reads overloaded until then.
- One exception stands until the worker crate's stop awaits every poll loop. A registration, a removal or a change of interval can retire a loop just ahead of a lead's end. That loop can still publish its reports and write its verdicts behind that end.
- They stand through the follow. A decision admitted in the instant before the next start clears them reads them.
- Another stands too: a replica that loses the lead and takes it again before its monitor reads the loss keeps the monitor, and what it holds.

### The schema and the validator

- The stock schema takes any stage name, written bare or mapped to an object. It lists the stock names a spec can write bare as examples of each entry.
- Validate an extended spec against the host's catalog: the validator refuses a name the catalog does not hold. Its refusal carries the path of the field that names the stage, such as `profiles.<profile>.score`.
- The standalone validator refuses stages that declare a need for the full lifecycle.

### What a stateful stage hears

- Stateful scorers and pickers can observe issue, dispatch, completion, release, membership and signals when the host supplies them.
- The engine tells a stage the issue, the dispatch, the completion and the release only where the stage names the word's kind (`Lifecycle::hears`). By default a stage that declares `needs_lifecycle` names every one, and another names none.
- The engine reads a scorer's or a picker's answer when the engine is built for a spec. A stage whose answer changes later is told by its first answer still.
- A fallback stage is the exception. A plugin's scorer or picker adapter passes it each word the adapter is told, whatever the fallback names, unless the plugin's own call at that word panics.
- A stage that observes membership may be told a join again of a worker that stands, and a leave of a worker it did not hear join. A stage takes each word as a statement of what stands.
- Where the host handed a registry, `InProcess::apply` tells the scorers and pickers it builds new the workers that stand there: each as joined, then the list of them. A stage behind one that panics at a word does not hear that word.
- The call tells a stage kept from the spec before none of them again. [A later spec](embedding.md#a-later-spec) has the rules.
- A scorer of your own has a place for a pick's overlap too: `Scorer::states_overlap` and `Scorer::said`. Both have a default that says nothing, and no entry of the engine asks them now.

**A stage that keeps load reports.**

- A stage that keeps a number of each worker's load report says so through `Lifecycle::keeps_loads`. Its host then tells it the reports through `Lifecycle::on_signal`.
- `InProcess` tells a stage that does not say so no report, and no stock stage says so.
- A controller tells none yet: under it a spec that holds such a stage reads no load report.
- At each load snapshot it takes, `InProcess` tells the reports whose sample instant is new, before it keeps the snapshot for its decisions. It tells them for the workers its task follows in the host's registry.
- In process the load signal names two more things: the report's sample instant and the named values the report states.
- `InProcess` states a report's named values under a spec in which a stage that keeps load reports keeps a signal's column.
- That stage reads its column's value through `SignalColumn::told`, with the value's own sample instant and at any age.
- A report that lists fewer ranks than it counts states none, and no plugin is told them.
- A host that builds the signal itself makes it with `Signal::load` and puts its report's values on it with `Signal::with_signals`.

[Runnable Rust example](../../../examples/grand-central) · [StageRegistry API](../src/stages/registry.rs)

## Load a WebAssembly component

- Compile a component for the `smg:policy` package, at the version the WIT file names.
- The `policy` world takes records. `policy-packed` takes byte blocks with the layout given in the WIT.
- The host refuses imports from WASI. It provides log, clock, counter and seeded random calls.
- A component implements `init`, a decision resource with filter, score and pick, and `observe`.
- A score answer can carry `tie-band` on the scores' 0 to 1 scale. The host uses its default band where the guest sends none.
- Beside other scorers the band counts by the plugin's share of the weights.
- The host keeps instances per decider and meters their calls.

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

- Use one `OpenDecisions` for the host's plugin runtimes. It accounts for the decision budget across plugins and sides.
- `PluginSpec` takes bytes, an optional ABI pin and opaque config. It also takes exposure, needs, counters, an event subscription, a fallback policy and breaker settings.
- Four more of its settings have a default, in the table below.
- The loader refuses zero memory or fuel, a component that fails to compile or link, an unsupported world, ABI or import, and a failing `init`.
- `Plugin::check_memory_budget` can check a host's total memory limit per decider.
- A trap or invalid output discards the instance.

| Setting of `PluginSpec` | Default |
|---|---|
| The memory cap | 16 MiB an instance. |
| The fuel calibration | 1,000 units a microsecond. |
| The log limit | 10 lines a second, at level 2. |
| The frame budget | 100 ms where it is zero, clamped to the decision's remaining budget as a stated one is. |

[Component build and host example](../../../examples/wasm/wasm-guest-policy) · [WIT](../contracts/smg-policy.wit) · [Host configuration](../src/plugins/wasm/config.rs)

## Call a gRPC plugin

- Implement the `PolicyPlugin` service with the supplied proto.
- Build the host with the crate's `controller` feature: `plugins::grpc` stands behind it.
- Its `Decide` stream starts with a hello and an acknowledgement with a protocol version (1 or 2) and supported stages. It handles decision frames by ID.
- An optional `Observe` stream sends ordered lifecycle events.
- A filter or scorer can start its call while native stages work. A picker bound on its own receives the combined scores at the pick.
- A scoring answer can carry `tie_band` on the scores' 0 to 1 scale. The host uses its default band when the field is unset.
- Beside other scorers the band counts by the plugin's share of the weights.
- Use a Unix socket with a checked peer user ID, or TCP with mutual TLS and a pinned server name.

This host function connects to a plugin that listens on a Unix socket under the same user and makes a picker. Use it as a profile's picker.

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

- `GrpcPluginSpec` takes the transport, opaque config, exposure, needs, counter names, an event subscription, a failure policy and breaker settings. It also takes the plugin's budget for a frame and the connect budget, 100 ms by default.
- A frame sees the whole candidate view up to `max_candidates`, 128 by default. Beyond it the stage falls back.
- A decision that finds its stream cold, connecting or failed falls back at once and waits for no connect. It asks for the stream in the background where the stream is cold, or failed at least 100 ms ago.
- A decision that falls back sooner asks for no stream: for example behind an open breaker, with no budget left, or over `max_candidates`.
- Call `Plugin::warm` on a deciding thread to establish its stream in advance.
- A failed initial handshake or initialization can refuse loading the plugin.
- The host bounds a decision's waits and frames.

[Service contract](../contracts/policy_plugin.proto) · [Host configuration](../src/plugins/grpc/config.rs) · [Test service](../tests/support/grpc_plugin.rs)

## Plugin inputs and fallback

Host code sets up both plugin kinds. A policy spec can name a registered stage adapter through the host's catalog. Under one of the engine's own refusal names it cannot: validation refuses that spec. A spec has no `plugins` field.

A host building a profile in Rust can bind several adapters to one plugin binding. Catalog registration requires a distinct stage name for each entry, and each adapter takes its name from its plugin.

**What a plugin is shown.**

- Exposure defaults to hashed tenant and session identities and no headers.
- A host can withhold identities, expose the clear form it holds, and expose selected allow-listed header positions.
- Request hashes are passed only when declared in the plugin's needs, up to 8,190.
- Lifecycle events default to none. Scorer and picker adapters forward the subscribed events, and a filter adapter forwards none.
- Among the adapters made from one binder's bindings, the first scorer or picker made is the one that forwards. `Plugin::bind` takes a binder of its own.
- So a catalog factory that keeps one binding has one adapter forward: the first made from that binder, which can be a name check's or an earlier validation's. Where it is one of those, no adapter made from that binder for the spec in force forwards.
- One that calls `Plugin::bind` for each adapter has each scorer or picker adapter forward, so a plugin named in two profiles can hear one event twice.
- A gRPC stream on protocol version 1 is not sent `Membership::Left`.
- The gRPC observe queue holds 1,024 events. Gaps in the sequence identify lost events.

**When a plugin fails.**

- A well-formed decline takes the configured stage fallback and leaves a healthy instance or stream in use.
- A trap, invalid output, timeout or open breaker can also trigger fallback.
- Surviving positions must be unique and in range. A scoring answer holds one finite value for each candidate it was shown, which the host clamps to 0–1. The pick must be a permitted position.
- With no fallback stage, the plugin's failure policy decides, as the table below says.

| Failure policy | At a filter | At a scorer | At a picker |
|---|---|---|---|
| `Open`, the default | Keeps the rows it was given. | Gives a flat score. | Ranks the combined scores. |
| `Closed` | Keeps no row. | Gives a flat score. | Names no target, and the request is shed. |

A hard filter that keeps no row empties the pool, as any hard filter can. A soft one that keeps none steps aside.

**The breaker.**

- The breaker defaults to five failures in ten seconds and a 30-second cooldown.
- Its latency rule can also open it. A sample is the time a decision counts as the plugin's own, as a share of the frames of its calls there.
- The frames' sum is cut at the budget the decision began with. A charged gRPC call counts the plugin's own frame whole, whatever the decision had left for it.
- The samples stand in windows of 128 at most. A plugin can hold several windows, as one for each deciding thread.
- The rule reads a window at each 64th sample. It opens the breaker where the middle share is past one half, the lower of two middle ones, or the second-slowest is past the whole.
- A share counts in thousandths, rounded down.
- Give a plugin a budget of at least twice its usual time.
- Hosts can read `Plugin::stats` and declared `Plugin::counters` in code.

[Shared plugin settings](../src/plugins/shared/config.rs)
