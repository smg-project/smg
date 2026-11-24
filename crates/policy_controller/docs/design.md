# The policy controller: design

How the policy controller is built: its parts, the data it reads and keeps, and what happens to a
request. It is written for a reader who knows the gateway and has not followed the work. The
README beside this folder is the user's guide (how to write a policy document, how to run the
service). This file is the design.

It describes the crate as it stands on main and what is planned. It was read whole at commit
`882be6f4f` (2026-11-22). Four subjects were read again at commit `ce421ddd3` (2026-11-23): the
gateway's option (sections 1 and 9), who sends a worker's data (4.3, 4.5, 10.1 and 10.2), the one
engine (5.7) and a gateway under a controller (11). Every part has one of three marks:

- **[built]**: on main as described.
- **[partly]**: on main in part. The text says which part.
- **[planned]**: not on main.

Section 14 collects the marks in one table. Section 15 lists the decisions that are open, each
with its options, their cost and one recommendation. Section 16 is the order of work.

Paths without a crate name are under `crates/policy_controller/src/`.

## Words used here

| Word | Meaning |
|---|---|
| worker | One inference engine endpoint that a request can be sent to. |
| worker list | The workers of one model that serve one role, in a fixed order. The gateway keeps its lists by role and protocol, and the controller by role alone. |
| classic | The gateway's routing policies of today (`--policy round_robin`, `cache_aware` and the others). |
| policy document | A YAML or JSON file that says how workers are chosen. The code calls it a policy spec (`PolicySpec`, `--policy-spec FILE`). Section 3. |
| profile | One way to choose: filters, then scorers, then one picker. |
| stage | A filter, a scorer or a picker. |
| binding | A rule of the document: which requests use which profile. |
| engine | The code that takes a document, a request and a worker list and answers with a decision. |
| host | A program that embeds the engine: the gateway, or the controller. |
| controller | The standalone service built from this crate. Gateways can ask it over gRPC. |
| decision | The engine's answer: a pick (up to four targets, best first) or a refusal with a reason. A target is one worker, or a pair's two. |
| lease | The engine's note that a decision's request is on its way. It ends when the request ends. |
| prefill, decode, encode | The roles of workers that serve one request in stages. |
| single request, pair, encode request | The three shapes of a request. One worker serves a single request. A prefill worker and a decode worker serve a pair. An encode request is a pair that also carries items for encode workers. |
| routing key | A value read from a request, where its binding says. Policies use it to keep the requests of one key on one worker. |

## 1. What it is

The policy controller chooses the worker that serves a request.

It is one crate, `smg-policy-controller`. Its core is the engine. The same engine is meant to run
in two places:

- inside the gateway process, in place of the classic policies;
- inside the controller, a standalone service that gateways ask over gRPC.

| Mode | Who decides | Who finds the workers and holds their data | Mark |
|---|---|---|---|
| Classic (the default) | the gateway's classic policies | the gateway | **[built]** |
| Engine in the gateway | the engine, inside the gateway | the gateway, in the same structures as classic | **[partly]** |
| Engine in a controller | the controller | the controller | **[partly]** |

On main the gateway calls the engine behind an option that is off by default (section 9). It
once called the engine through an earlier integration. That integration was taken out first
(stage 0 of section 16), so that the crate could be finished before the gateway depended on it.
The engine then came back as one change.

### The seven rules

Everything below follows from seven rules.

1. **The engine decides every request.** Where the engine is switched on, no request is handed to
   a classic policy and no classic policy object is built.
2. **One set of worker data in a process.** Workers, their health and their loads live in the
   worker crate's registry and load monitor. Cache records live in the kv-index crate's
   structures. The engine reads them in place. It keeps no second copy and runs no second
   discovery, probe or subscription.
3. **One engine.** The gateway and the controller run the same decision code over the same kinds
   of data.
4. **The host side is small.** A host builds the engine at start, calls it at one place for each
   request and tells it what became of the request. A host has no policy logic: it never asks
   which policy is running.
5. **Switched off, it costs a test.** With the option off no engine is built and classic's own
   selection decides. A request then pays two things: the test of the option where it is placed,
   and what the five placement functions and the load guard cost for carrying an answer of either
   side. That cost is counted for each policy in the gateway's decision bench and shown on the
   page of each change to those functions (section 13.2).
6. **Classic is the reference.** Under the document that a set of routing flags means, the engine
   gives the answer the classic policy gives. A difference is a defect, or a decision written in
   section 15.
7. **The crate first.** A behaviour is built and proven inside the crate, against classic's
   recorded answers and at classic's speed, before any host code is written for it.

## 2. The parts

```text
                     policy document (YAML or JSON, with a generation number)
                                      |
  +-----------------------------------v-----------------------------------+
  |  smg-policy-controller                                                |
  |                                                                       |
  |   document:   bindings  ->  plan  ->  profile                         |
  |                                        filters -> scorers -> picker   |
  |   engine:     one decision, one lease, the lease's events             |
  |   state:      what each policy remembers between decisions            |
  |               (rotation, hash ring, sessions, sent prefixes)          |
  |   controller: the gRPC service, election, replication  (a feature)    |
  +--------^-------------------------^------------------------^-----------+
           | read in place           | read in place          | read in place
  +--------+----------+   +----------+----------+   +---------+----------+
  |  smg-worker       |   |  smg-worker         |   |  kv-index          |
  |  worker registry  |   |  load monitor       |   |  cache records     |
  |  worker objects   |   |  load snapshot      |   |  (events, trees)   |
  +--------^----------+   +----------^----------+   +---------^----------+
           | filled by               | filled by              | filled by
     discovery, registration,   the workers' load       the workers' cache
     health checks              reports                 events
```

There are two hosts:

- **the gateway**: it already owns a worker registry, a load monitor and the cache records,
  because its classic policies use them. It hands the same objects to the engine.
- **the controller**: it owns a registry, a load monitor and cache records of its own, fills them
  itself and answers gateways over gRPC.

Dependencies point one way. The crate depends on `smg-worker` and `kv-index`. It does not depend
on the gateway.

### The crate today, by size

A snapshot: the lines of each folder under `src/`, tests included, counted at commit `4ac871fd1`.
Later changes do not update these counts.

| Folder | Lines | Files | What it holds |
|---|---|---|---|
| `document/` | 21,614 | 15 | The policy document: its types, validation, the presets of the ten policy names, the conversion from routing settings, apply and roll back. |
| `engine/` | 80,067 | 67 | The engine, the profile runner (`framework.rs`, 20,369 lines), the request and worker types. |
| `policies/` | 12,568 | 12 | One file for each built-in policy. |
| `stages/` | 10,365 | 8 | The built-in filters, scorers and pickers and their table. |
| `tracking/` | 10,075 | 12 | The lease, its guard, the placement, the dispatch credit, the record of sent prefixes. |
| `store/` | 29,896 | 28 | The controller's in-memory store: worker records, loads, cache index, leases. |
| `service/` | 80,546 | 59 | The controller: the gRPC service, election, replication, discovery. |
| `worker_host/` | 18,895 | 19 | What lets a controller hold worker objects of the worker crate. |
| `load/` | 5,842 | 10 | A worker's load report as a stage reads it. The controller's own load poller, counted here, has left since. |
| `plugins/` | 16,296 | 21 | Runtimes for user stages in WebAssembly and over gRPC. |
| other | 498 | 4 | `lib.rs`, the binary's `main`, two small files. |
| **Total** | **286,662** | **255** | |

The engine (section 5.7) is `engine/policy_engine.rs` and the folder beside it: 33,581 lines with
their tests. The record writer and its tests are 3,354 more.

## 3. The policy document **[built]**

A policy document says how workers are chosen. It has a generation number. A new document
replaces the old one whole, under the next number.

### 3.1 Its shape

```rust
// document/spec.rs
pub struct PolicySpec {
    pub api_version: String,                     // always "smg.policy/v1"
    pub generation: u64,                         // the document's number
    // (one private field reads a `kind` line only to refuse it)
    pub view: ViewSpec,                          // what a decision may read beyond the basic facts
    pub index: BTreeMap<String, IndexSpec>,      // settings of the record of sent prefixes, by model
    pub signals: BTreeMap<String, SignalSpec>,   // named values of a worker's report that stages read
    pub profiles: BTreeMap<String, ProfileSpec>, // the profiles (required, as `apiVersion` is)
    pub plans: PlansSpec,                        // rules of the prefill-decode plan
    pub bindings: Vec<BindingSpec>,              // in order; the first match wins; an empty
                                                 // list binds a lone profile to everything
                                                 // and is refused beside several profiles
    pub explain: ExplainSpec,                    // which decisions are kept to be explained:
                                                 // every Nth
}

pub struct ProfileSpec {
    pub filter: Vec<FilterSpec>,  // the filters, in order: each removes workers
    pub score: Vec<ScoreSpec>,    // the scorers, each with a weight
    pub pick: PickSpec,           // exactly one picker
    pub sent_ttl_ms: Option<u32>, // how long a dispatched request counts in the profile's records
    pub sent_report_delay_ms: Option<u32>, // how long behind a dispatch a load report ends its count
}

pub struct BindingSpec {
    pub matcher: Match,                      // which requests the binding takes (written `match`)
    pub profile: Option<String>,             // the profile of a single decision
    pub plan: Option<PlanName>,              // `single` (default), `pd-pair` or `epd`
    pub profiles: BTreeMap<String, String>,  // a profile for each side of a pair
    pub overload: Option<OverloadSpec>,      // the overload switch and shed word of these workers
    pub routing_key: Option<RoutingKeySpec>, // where a request's routing key is read from
    pub sessions: Option<SessionsSpec>,      // sticky sessions
    pub shadow: Option<ShadowSpec>,          // a second profile run for comparison only
    // a canary, `override: allowed` and an admission switched on are in the schema and are
    // refused: section 14
}

pub struct Match {
    pub model: String,     // an exact id, `prefix*`, `*suffix` or `*`
    pub phase: Vec<Phase>, // single, prefill, decode, encode; empty covers every phase
    // tenant, priority, header, prompt_tokens and labels are in the schema; a document that
    // writes one is refused as not in this build
}
```

A binding matches a request by its model and its phase. Bindings are read in document order and
the first match wins.

### 3.2 Two documents

The default policy, `cache_aware`, as a document:

```yaml
apiVersion: smg.policy/v1
generation: 1
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

A prefill-decode pair with round robin on both sides:

```yaml
apiVersion: smg.policy/v1
generation: 1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: []
    pick: round-robin
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: []
    pick: round-robin
bindings:
  - match: { model: "*" }
    plan: pd-pair
    profiles: { prefill: prefill, decode: decode }
```

### 3.3 From routing flags to a document

A user who never writes a document still gets one. The crate turns the routing settings into the
document they mean. The settings include the policy name and its options, the mode, the overload
switch and shed word, and the sticky-session options.

```text
routing settings ──> RoutingSettings::policy_document() ──> PolicySpec ──> validate ──> AppliedSpec
                     (document/routing_settings.rs, a pure function)
```

Each of the ten policy names has a preset. The presets are in `policies/<name>.rs`, and section 7
lists them. A preset also writes what its binding needs: a routing-key policy gets its key
sources, and `manual` gets its sessions rule.

The mode and the gateway decide what the document binds:

| Gateway | Mode | What the document binds |
|---|---|---|
| one router | single workers | one profile, bound to every model |
| one router | prefill-decode, over HTTP | two profiles under one pair binding, over the prefill and decode phases; behind it, the main policy's profile, bound to every other request |
| one router | prefill-decode, over gRPC or ZMQ | the pair binding alone |
| one router | encode-prefill-decode | three profiles under one binding of the `epd` plan, and no single request |
| many routers | single workers, or prefill-decode | the pair binding and, behind it, the main policy's profile for every other request; under `manual` with single workers, its one profile alone |
| many routers | encode-prefill-decode | three profiles under one binding of the `epd` plan, over the encode, prefill and decode phases; behind it, the main policy's profile, bound to every other request |

A gateway of many routers serves pairs and single workers, however it reaches them, so its
document binds both. In a gateway of one router the second binding of a prefill-decode mode is
there for one request, the token count. Over HTTP the gateway still places it as a single request,
on one prefill worker, and classic places it by the model's policy, which is the main policy
unless a worker's label names another (below). Over gRPC or ZMQ the mode counts
with no worker, as the encode mode does, so the document binds no single request. The settings
state how the gateway reaches its workers.

The document plans the mode's own sides alone, with no single request, in other cases too. Among
them:

- The main policy states a setting that no document says.
- In a gateway of one router, or in the encode mode of many routers, a policy of one of the two
  families (the mode's own sides, the single requests) names a block size and a policy of the
  other places by a prefix ring, which counts its key in another size.

A request of the family left out is then refused: the token count in a gateway of one router,
each single request in a gateway of many routers.

A user may supply a document instead. Both launchers refuse beside it, by name, each routing flag
that the document replaces. The command line refuses such a flag where it was typed, even at its
default value. The Python launcher cannot tell whether a flag was typed, and refuses one that is
off its default. A configuration built by hand is compared instead. The document is accepted when
the routing settings say the same document or say nothing. Otherwise the start is refused, and the
error names the first difference. A gateway's start also refuses a supplied document whose
bindings differ in an overload word, and one whose overload block is on where the gateway states
neither threshold (decision 10). A supplied document that binds no single request refuses each
single request, in any gateway. For a prefill-decode mode of one router over HTTP that is the
token count.

**A worker's `policy` label [planned].** With the classic policies a registration can name its
model's policy. The engine follows no such label today.

What classic does: the registry builds a model's policy at the registration that finds the model
with none, from that worker's `policy` label, for the worker's primary model. Where two such
registrations overlap, the policy inserted last stands. The model keeps the policy until the
last of the gateway's own workers for it has left, and the label of a worker registered
meanwhile changes nothing (`PolicyRegistry::on_worker_added` and `on_model_unserved`,
`model_gateway/src/policies/registry.rs`). A worker of any role names the policy this way. The
policy so named places the model's single requests: on its regular workers, and the token count
of a prefill-decode mode over HTTP on a prefill worker. A pair's and an encode request's workers
are placed by their side's policy. A label names a policy in any letter case, with all its
underscores or with none. A label that names the default policy's own type takes the settings'
options, any other name takes that policy's defaults, and an unknown name leaves the default.

What is planned: the label becomes one more input of the document built from the routing
settings (`RoutingSettings::policy_document()`), by classic's rule. The registration that finds
the model with none decides: its label adds a binding of that model's single requests, ahead of
the binding of every model (section 5.6, step 4), and the binding leaves with the model's last
worker. Where the document binds no single request, a label adds no binding, and the model's
single requests stay refused. The name is resolved as classic resolves it. The default policy's
own type gives the profile the settings state. Any other name gives the binding the settings
would write for a single request under that policy, with the policy at the defaults classic
builds it with by name. An unknown name adds no binding.

The gateway then puts the new document in force on the running engine. A document put in force
keeps the stages of every profile that did not change: one whose name stands in both documents
and whose stages are built from the same text (the profile's own block, `view` and `signals`).
Such a profile's stages keep what they hold, and a decision under way and a new one run the same
stage objects. A profile that changed, and a new one, start empty. So a label's binding resets
nothing of another model's policy where it leaves the document's `view` and `signals` as they
stand. A label whose binding writes into them, as `manual` writes its header into the view,
restarts every profile's stages. Where the new document cannot be built, the document in force
stays and the registration is answered with a warning.

That holds where the document is built from the routing settings. Where the operator supplies a
document, and on a controller, the document decides, and a registration whose label names
another policy is answered with a warning. As a first step the gateway warns at a registration
whose label it does not follow.

### 3.4 A validated document

Validation turns a `PolicySpec` into an `AppliedSpec`. An `AppliedSpec` holds the built profiles
and bindings with their stage objects. It also holds what is resolved once from the document: the
block size, the hash depth, each binding's routing-key sources and the signals.

What a policy remembers lives in several places, among them:

| What | Where |
|---|---|
| what a stage remembers | its stage object, inside the `AppliedSpec`: it belongs to one generation of the document |
| what belongs to one worker list (round robin's counter) | the engine's value for that list (section 5.2) |
| the record of sent prefixes | the engine object (section 5.1) |
| a sessions rule's pins | the engine object: one table for the engine's life |

`DocumentManager` (`document/operator/document_manager.rs`) applies, lists and rolls back
generations. It can keep the last eight generations in a folder, one file each, and roll back to
one of them.

### 3.5 The stage interfaces

A filter, a scorer and a picker are three traits (`engine/framework.rs`). Each has two required
methods, shown here. The others have default bodies and are not shown: most are switches and
short paths that the built-in stages use.

```rust
pub trait Filter {
    fn name(&self) -> &'static str;
    // keep or drop one worker for this request
    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, cx: &Constraints<'_>) -> bool;
}

pub trait Scorer: Lifecycle {
    fn name(&self) -> &'static str;
    // write one score for each surviving worker
    fn score(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], kept: &[u32],
             out: &mut [f32]) -> TieBand;
}

pub trait Picker: Lifecycle {
    fn name(&self) -> &'static str;
    // order up to four targets
    fn pick(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], kept: &[u32],
            scores: &[f32], band: TieBand, out: &mut Targets);
}

// What a stage may hear about the requests it helped place. All optional.
pub trait Lifecycle {
    // on_decision_issued, on_dispatched, on_completed, on_released, on_membership, on_signal
}
```

The built-in stages:

| Kind | Names |
|---|---|
| filter | `healthy`, `not-overloaded`, `has-capacity`, `has-room`, `not-tried` |
| scorer | `least-load`, `least-requests`, `fewest-sessions`, `session-affinity`, `prefix-affinity`, `kv-pressure`, `warmup-slice`, `bucket`, `consistent-hash`, `resident-work`, `least-cost` |
| picker | `best`, `round-robin`, `random`, `power-of-two`, `count-first`, `power-of-two-count-first`, `passthrough`, `manual`, `best-with-temperature` |

A program that embeds the crate can add stages of its own under new names
(`StageRegistry::filter`, `scorer`, `picker`) and validate documents against that table.

**[built]** Such a stage keeps building when the crate grows. For each type below, the table says
whether it is marked non-exhaustive and how code writes it:

| What | Types | Marked | How code writes it |
|---|---|---|---|
| the events a stage hears | `Signal`, `Membership`, `ReleaseReason`, `TransportClass`, `EventKind` | yes | a `match` on one keeps an arm for the rest, and a case with named fields is matched with `..` and made by its call |
| the answers a stage gives as an enum | `ScoreSpread`, `ScoreFront`, `DrawUnit` | yes | as the events are |
| the records a caller builds | `Needs`, `Outcome` | yes | a stage writes its `Needs` as one chain from `Needs::NONE`, a call a flag. A host makes an `Outcome` by `Outcome::reported` |
| the records a stage answers with | `Shortlist`, `HeadRule` | yes | each is made by its call |
| the records a stage only reads | `RequestFacts`, `Candidate`, `Constraints`, `PoolFacts`, `LoadRow`, `AffinityFacts`, `Demands`, `DemandFacts`, `Target`, `Pair`, `Pick`, `Overlap`, `Lease` | not yet | each gains fields, so code that builds one writes the rest as `..Default::default()` and never matches it whole |
| the enums closed by design | `Decision`, `ShedReason`, `Side`, `Phase`, `Role`, `Said` | no | a caller answers every case of them, and a new case must stop its build |

## 4. The data the engine reads

Rule 2 says the engine keeps no copy of worker data. This section says where the data lives: in
structures of the worker crate and of the kv-index crate. The policy crate adds structures of its
own, among them the engine's hash table (section 4.5) and the controller's value for a worker
list and its holder (section 4.1).

### 4.1 The worker registry **[built]** (`smg-worker`)

The registry holds every worker object and publishes, for each model, the list of its workers.

```rust
// crates/worker/src/registry.rs (the fields that matter here)
pub struct WorkerRegistry<G> {
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>, // every worker, by id
    model_index: ModelIndex<G>,                       // model id -> the published value of that model
    event_tx: broadcast::Sender<WorkerEvent>,         // Registered, Removed, Replaced, StatusChanged
    // ...
}

// The value the registry builds each time a model's list changes.
pub trait Generation: std::fmt::Debug + Send + Sync + 'static {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self; // build the value for this list
    fn members(&self) -> &Arc<[Arc<dyn Worker>]>;   // the list it was built for
}
```

`G` is the host's own type. Whenever a worker joins, leaves or is replaced, the registry builds a
new list and a new `G` for each model the worker serves, and readers get the new value whole. A
published list is never changed.

The gateway's `G` holds the model's workers and its lists by role and protocol (prefill workers
over gRPC, regular workers over HTTP, and so on). The controller's `G` holds its lists by role
alone, with the workers of every protocol in one list.

The controller's holder lends each role's list with its cell and with what its rows state of each
worker (`Objects::lent`).

Beside a pair's two lists it hands their pair classes (`Objects::pair_classes`): each worker's
class by its place in the list as lent, under the store's own pairing rule over what the rows
state. Those classes state no first partner, so a controller makes no pair on one known runtime
ahead of another. A row whose facts change gets a new worker object, so the registry builds a
new list and a new `G` for it. The classes are derived once for a `G` and a pairing mode, at the
first ask, and stand with it.

A controller knows no admission window of a gateway. So it reads a pair's admission open
(`OpenAdmission`), as the rows its store builds state it. The gateway that asks is to gate
capacity when it commits **[planned]**: no gateway calls a controller yet (section 11).

All of it stands in a build with the `worker-host` and `in-process` features, as a controller's
is. A controller's decision of a pair, at its prefill and at its decode call, reads it (section
10.1).

Beside its encoder list and its prefill list the holder hands the two columns an encode request's
decision reads (`Objects::encode_columns`). They are what the encode side reads of each encoder
and the encode class each prefill names, by a worker's place in the list as lent. The classes are
the store's own encode rule over what the rows state. The columns stand in the same build. No
decision of a controller reads them yet: an encode request is answered `Busy` (section 10.1).

### 4.2 The worker object **[built]** (`smg-worker`)

```rust
// crates/worker/src/identity/basic.rs (shortened)
pub struct BasicWorker {
    pub metadata: WorkerMetadata,                 // the worker as configured: URL, models, role, protocol
    pub runtime: ArcSwap<WorkerRuntime>,          // its changing state
    pub circuit_breaker: ArcSwap<CircuitBreaker>, // closed, open or half-open
    // connections and clients
}

// crates/worker/src/runtime.rs (the fields a decision reads)
pub struct WorkerRuntime {
    status: AtomicU8,                              // Pending, Ready, NotReady, Failed, Draining
    load_counter: AtomicUsize,                     // requests in flight on this worker
    worker_routing_key_load: WorkerRoutingKeyLoad, // requests in flight for each routing key
    overloaded: AtomicBool,                        // the overload flag
    stall: AtomicU8,                               // the liveness flag and its reason
    // health-check counters, prefill admission, warm-up marks ...
}
```

Who writes it: the health checker (status), the load monitor (overload, liveness), the request's
load guard (the in-flight counters), request outcomes (the circuit breaker). A controller serves no
request: the leases its store holds on file write the in-flight counter there (`HostCount`).

These are the main writers, not all of them. The status is also written at a drain, at a
registration's activation, by the registry itself where a worker leaves or comes from a peer's
record, and on a controller by its worker host, which follows the store's row. The liveness flag
is also written on the word of the stream a worker pushes on (section 4.3), of a passed probe
and of a request's own token stream.

A decision reads a worker's routing state in one read: `Worker::routing_state()` gives ready,
circuit breaker, stalled, overloaded, the in-flight count and the count of requests processed.

### 4.3 The load reports **[built]** (`smg-worker`)

```rust
// crates/worker/src/load/state.rs
pub struct LoadSnapshot {
    entries: HashMap<Arc<str>, LoadEntry>, // worker URL -> its last load report and when it was taken
}
pub type LoadReceiver = watch::Receiver<Arc<LoadSnapshot>>; // how a reader follows new snapshots
```

The load monitor polls each worker (every 10 seconds by the gateway's default) and takes the load
records that workers push. A worker whose pushed records are fresh is not polled. A new snapshot
is published after each poll, and for pushed records at most once in 20 ms. A snapshot is never
changed after it is published.

**Who opens the stream.** A worker pushes on a stream that its host opens to it. The worker
crate's collector connects to the worker and subscribes, and the worker answers on that stream
(`crates/worker/src/kv_events/`). It sends its cache events with the load records that ride
them, or its load records alone, as its host asks. A gateway asks a worker for its cache events
where what places the worker reads them. Otherwise it asks for the load records alone, and opens
no stream where its load monitor wants no loads of the worker. On main a controller opens no
stream for load records alone. It asks a worker for its cache events where the worker's
registration asks for them, on one stream for each engine: a data-parallel engine's rank 0
streams every rank's events. The collector hands the load records to the load monitor and the
cache events to the index of cache events (section 4.5). Only a worker dialed over gRPC has such
a stream; the monitor polls the others. A worker dials nobody for this, so it needs no address
of a gateway or of a controller.

The host that opens the stream is the process that holds the workers' data (the table of
section 1): the gateway in classic and with the engine in the gateway, the controller with the
engine in a controller (section 10.2). So a worker's data takes one way in every mode. Only the
process that opens the stream changes. On main a controller's collector hands its load monitor no
load record yet: that monitor is fed by its polls alone (section 10.2).

### 4.4 The in-flight counters **[built]** (`smg-worker`)

A route creates one `WorkerLoadGuard` for the worker it chose. The guard adds one to the worker's
in-flight counter, and to its routing key's counter. It lives as long as the request. At its drop
it takes one off and tells the worker's `RequestCompletionSink` that a request ended. Every path
passes through it: success, error and a client that went away.

### 4.5 The cache records

A cache-aware policy sends a prompt to a worker that already holds its beginning. Three records
can name that worker.

| Record | What it holds | Written by | Lives in |
|---|---|---|---|
| A. the index of cache events | for each block of tokens, which workers hold it | the workers' own cache events | `kv-index`: `ChainIndex`, wrapped by the gateway's `KvIndex`; one for each model |
| B. the token tree | the token prefixes this process sent to each worker | the policy, at each decision | `kv-index`: `TokenTree`, one for each model in the `ModelTrees` of each policy object |
| C. the string tree | the same for text prompts | the policy, at each decision | `kv-index`: `StringTree`, one for each model in the same `ModelTrees` |

A prefill-decode setup has a classic policy object for each side, so each side keeps its own
trees.

Classic reads exactly one record for a request:

| The request | The model | Record |
|---|---|---|
| has token ids | has an index of cache events that holds at least one block | A |
| has token ids | has no index, or its index is empty | B |
| has no token ids (text) | any | C |

This is classic under its default, `--cache-index tree`. Under `--cache-index hash` classic reads
none of the three: it keeps a table of placements with a time to live for requests with token
ids, and balances a text request by load. This design covers the default.

Both trees are read and written in one walk: the walk finds the longest stored prefix, hands the
match to the policy, and stores the request for the worker the policy chose. So no write is lost
when many requests decide at once. A tree is brought back under a size bound
(67,108,864 tokens or characters for a model by default) by an eviction pass every 120 seconds.
Between two passes it grows past the bound.

A request's prompt is stored under its **cache partition** (cache salt, extra value, adapter), so
that two tenants' prompts never match. The gateway derives the partition once for each request.

**The engine today [built].** It reads record A through two small traits that the host
implements over its own index (`CacheIndexes`, `CacheIndex`, in `engine/policy_engine/overlap.rs`).
For what it sent itself it keeps a record of its own: a hash table behind one lock for each model
(`tracking/prefix_predictor.rs`), not a tree. That record has three faults, each seen on a cluster
(section 13.3):

1. It holds at most the first 256 blocks of a prompt (4,096 tokens by default). A match needs a
   0.3 share of the prompt, so past 13,653 tokens a prompt stops matching.
2. Its write gives up when the lock is held, so writes are lost when many requests arrive at once.
3. It is neither read nor written for a worker the index holds a block for. Under a document
   whose prefix stage reads the record alone (`source: predictor`), such a worker is never named.

One more limit is in its reading of record A: it asks the index by at most the first 256 blocks of
a prompt, where classic asks by every block. Two conversations alike in their first 4,096 tokens
then tie. A local probe showed it (18 of 32 next turns kept, where classic kept every one); the
cluster run did not.

**The engine as designed [partly].** It follows classic's rule above: one request reads one
record. For B and C the engine keeps no tree of its own. The host lends its own trees, through
the trait it already answers for cache events (`CacheIndexes`). The engine decides inside the
lent tree's one walk, as classic does, and the walk stores the request once, for the worker the
request goes to.

- **[built]** A single request that hands token ids, for a model with no index of cache events,
  under a profile that reads sent prefixes and names the holders' rows, as the `cache_aware`
  preset does: the host lends its token tree (`CacheIndexes::token_tree`) and hands the
  request's entry already named under its cache partition (`RequestMaker::token_key`), so the
  naming stays written once, in the host. The engine calls the tree's walk alone. The tree's size
  bound and a worker that left stay the host's to keep.
- **[built]** A host that lends no tree, or names no entry, is decided by the hash table as
  before.
- **[built]** A request placed by the one call (`PolicyEngine::place`) hands its entry too. Its
  caller states the entry with the prompt (`Prompt::entry`), and the call's own maker asks it
  where a decision reads the tree.
- **[built]** The tree is written for the worker the request goes to. The tree's walk asks the
  request's maker which target of the pick its caller sends to (`RequestMaker::sends_to`), best
  first, and stores the prompt for the first one: for no worker where the caller sends to none.
  The one call answers by the read it makes of a target itself: ready, and admitted by the
  route's rule. So a call that decides again stores the prompt once, for the worker it sends
  to, and a call shed under the rule's name stores nothing. A maker that does not answer is
  stored for the pick's first target.
- **[built]** The one call takes a target by the answer the tree's walk was given on it: a
  worker's state is read, and its route's rule asked, once a decision. So the tree names the
  worker the prompt went to. A worker that stops being ready behind that read is still sent to,
  as under classic, which also stores its selection ahead of the send. The hash table is written
  by the same answer: for the first target its caller sends to, and for none where it sends to
  none.
- **[planned]** A request given as text is decided the same way in the host's string tree.
- **[planned]** The hash table leaves the single decision. Until then a model with an index of
  cache events reads the index and the hash table together, as today.
- **[built]** A prefill-decode pair decided over worker lists reads the hash table and the
  host's index of cache events for each side, by the side's own list, and writes each side's
  pick into the hash table. An encode request's first decision reads neither yet. Classic keeps
  trees for each side. **[planned]** The engine reads the host's trees for each side.

A decision of the engine publishes no insert to other gateways.

**What the workers send and what they do not.** Only record A comes from the workers: the cache
events they push fill the index and nothing else. Records B and C are the deciding process's own
account of the prompts it sent, written by its policy at each decision. No worker pushes them and
no worker's stream carries them. In classic the gateway's policy keeps them. With the engine in
the gateway the gateway keeps them and lends them to the engine. On main it lends none yet
(section 9.2). Until it does, a request under the engine reads the hash table, and the index
beside it where its model has one. A prefill-decode pair is lent no tree: each of its sides
reads the hash table, and the index where its model has one, by the side's own list. An encode
request's first decision reads neither yet. A controller is to
keep trees of its own **[planned]**, and section 11 says what its contract must hold for that. On
main a controller's engine reads its hash table alone (section 10.1).

One change was made outside the policy crate first **[built]**: classic's trees moved out of
its policy object into one store type of the kv-index crate (`ModelTrees`), with no behaviour
change. Classic's policy object keeps its trees in that type. The gateway keeps the token trees
it lends the engine in the same type **[built]**.

### 4.6 Who writes and who reads

| Structure | Written by | Read by classic | Read by the engine |
|---|---|---|---|
| a model's worker list | the registry, at each membership change | yes | yes, the same list |
| the worker object's state | health checker, load monitor, load guard (on a controller, the leases on file), outcomes | yes | yes, once for each decision |
| `LoadSnapshot` | the load monitor | `least_load`, `cache_aware`, `power_of_two` | yes, the latest snapshot; a controller's deciding thread gives it with each request, where the document keeps no report |
| the index of cache events | the workers' event streams | `cache_aware` | yes, through `CacheIndex`; a controller's engine is lent none yet |
| the two trees | the deciding policy: classic's, or the engine inside the tree's walk | `cache_aware` | the host's own trees, lent: the token tree **[built]**, the string tree **[planned]** |

## 5. The engine

The engine is in `engine/`.

### 5.1 The engine object **[built]**

A gateway has one engine object. It is built once from the policy document and holds the
document's stages and what they remember. It holds no worker of its own: the registry that its
host hands it (section 8.1) holds them. A controller builds a new engine object for each document
it applies (section 10.2).

```rust
// engine/policy_engine.rs
pub struct PolicyEngine {
    document: InForce,    // the validated document (profiles, bindings, stage objects and their state), its
                          // number, what is read of it once (which lease events its stages need) and its
                          // shadow counts
    clock: DecisionClock, // one clock for every decision and every load report's age
    leases: OwnWord,      // the id of the last lease issued: one atomic counter
    sessions: Arc<SessionTable>, // the pins of every key a sessions rule placed: one table for the
                          // engine's life
    affinity: Affinity,   // cache affinity: the host's index of cache events, and the engine's own
                          // record of the prefixes it sent, one for each model
    epochs: AtomicU64,    // how many list values were made
    faults: AtomicU64,    // decisions that panicked
    unheard: AtomicU64,   // lease events and membership changes that panicked inside a stage
    upkeep_faults: AtomicU64, // steps of the engine's own task that panicked: the upkeep, the registry's follow,
                          // a telling of load reports
    loads: task::Loads,   // the host's load channel, once handed, and the newest load snapshot
                          // the engine's own task keeps at hand
    members: membership::Handed, // the host's worker registry, once handed: the engine's own task
                          // follows its events
    lists: HandedLists,   // the host's worker lists, once handed: the engine's own task fills their
                          // cells ahead of their requests
}
```

It compiles under the crate feature `in-process`.

### 5.2 The engine's value for one worker list **[built]**

For each list the registry publishes, the engine keeps one value. It keeps the value in a cell
that the host lends:

- The host keeps an empty cell beside each list it publishes (`ListCell`). It lends the list and
  its cell together with each request (`LentList`).
- The engine makes its value into the cell at the list's first ask, and reads it there from then
  on. Where the host handed the engine its lists (section 8.2), the engine's own task asks first.
  Otherwise the first decision over the list does.
- When the membership changes, the registry publishes a new list with an empty cell. The old
  value is dropped with the old list.

A host may know a fact of its workers that their objects do not hold. It lends such a fact with
the list (`LentList::facts`): one value for each worker, in the list's order. The engine writes it
into the worker's record (section 5.3) where it fills the cell. Three such facts stand today:

- whether a worker takes images;
- the id the host knows the worker by, where the host keeps ids of its own;
- what the worker states for a request's hard demands, which stands in a column beside the
  records.

The fill makes the list's records.

```rust
// engine/policy_engine.rs
pub(crate) struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>, // the host's own list of worker objects, shared, not copied
    rotation: OwnWord,               // round robin's counter for this list
    records: OnceLock<PoolRecords>,  // the unchanging half of each worker's record (5.3)
    signal: OnceLock<(Box<str>, Phase, bool)>, // whether a decision over this list reads a named signal
    wires: OnceLock<WireModes>,      // which protocols (HTTP, gRPC, ZMQ) the list's workers speak
    alone: OnceLock<(Box<str>, Option<AloneProfile>)>, // the profile one pick can decide (5.6)
    shadow: OnceLock<(Box<str>, Option<ShadowWord>)>,  // the shadow of this list's model, where the
                                     // document names one
    affinity: PoolAffinity,          // this list's lookup table for cache affinity
}
```

The list of workers in it never changes. The counter is one atomic word. The cached values are
built once or replaced whole.

### 5.3 One worker as a decision sees it **[built]**

A filter, a scorer or a picker never touches a worker object. It reads one fixed-size record for
each worker, 64-byte aligned, in a buffer that belongs to the deciding thread.

```rust
// engine/view.rs
pub struct Candidate {
    // identity: does not change while the list lives
    pub position: u32,       // the worker's place in the host's list; a decision names workers by it
    pub id: WorkerId,        // the id its host states, or a 64-bit fingerprint of the worker's address
    pub key: u32,            // stable identity used by hash-based placement
    pub role: Role,          // regular, prefill, decode or encode
    pub dp_rank: u16,        // data-parallel rank; a rank is a candidate of its own
    pub dp_size: u16,        // how many ranks its engine has
    pub wire_modes: WireModes, // protocols it serves
    pub engine_kind: u16,    // engine family, as a position in the document's list
    pub connector: u16,      // prefill-decode connector, as a position in the document's list
    pub pair_class: u32,     // which decodes a prefill can pair with
    pub zone: u16,           // zone, as a position in the document's list
    pub labels: u64,         // one bit for each label the document declares (at most 64)
    pub ec_compatible: bool, // usable for the encode side of an encode-prefill-decode request
    pub vision: bool,        // takes pixel media: an image or a video
    pub host_local: bool,    // reachable from one gateway only
    pub engine: u32,         // its engine's place among the list's engines; ranks share one

    // live: read again at every decision
    pub health: Health,      // healthy, breaker_open, stalled, draining, overloaded
    pub in_flight: u32,      // requests the host has on the worker now
    pub load: Option<LoadRow>,       // the worker's latest load report, if it sends any
    pub load_sampled_at_micros: u64, // when that report was taken
    pub session_hit: bool,   // the request's routing key is pinned to this worker
    pub session_in_flight: u16,      // requests of this routing key in flight here
    pub active_sessions: u16,        // routing keys with a request in flight here
    pub has_capacity: bool,  // the host admits one more prefill here
    pub has_room: bool,      // the host admits one more decode here
}
```

The identity half is filled once for a list and cached (`PoolRecords`, `engine/records.rs`). The
live half is read from the worker object once for each decision. The load report comes from the
load snapshot; a snapshot's reports are converted once for a list and cached.

The field `engine` numbers the list's data-parallel engines: records that state one number are
ranks of one engine. `PolicyEngine` writes the numbers once for a list, where it finds the list's
engines. A default record leaves 0 on every record, which would read as one engine. So the field
tells engines apart only where the records' maker numbered them. `PolicyEngine` does, in process
and in a controller. A host that writes its own records numbers them itself, as the crate's
example does (`crates/policy_controller/examples/in_process/main.rs`). The number holds within
one generation of one list and is no id beyond it.

### 5.4 What a request states **[partly]**

```rust
pub struct Request<'a> {
    pub id: &'a [u8],                 // the request's id, as the host names it
    pub facts: RequestFacts<'a>,      // the request, as a policy may read it
    pub constraints: Constraints<'a>, // what only the deciding host knows
}

// engine/view.rs (the fields a stage reads; 32 in all)
pub struct RequestFacts<'a> {
    pub model: &'a str,            // the resolved model id
    pub phase: Phase,              // Single, Prefill, Decode or Encode
    pub tenant: Identity<'a>,      // the tenant
    pub priority: i32,             // priority band
    pub streaming: bool,
    pub max_output_tokens: u32,
    pub prompt_tokens: u32,        // the prompt's token ids, counted; 0 for a request of text alone
    pub prompt_chars: u32,         // the text's characters, counted where a stage reads a size
    pub prompt_bytes: u32,         // the text's bytes; `prompt_size()` is the tokens, else these over four
    pub block_hashes: &'a [u64],   // one hash for each whole block of the prompt, in order
    pub hash_kind: HashKind,       // the blocks are token blocks or text blocks
    pub block_size: u32,           // tokens (or bytes) in one block
    pub namespace_seed: u64,       // the cache partition, as one seed
    pub session_key: Identity<'a>, // the routing key, derived as the binding says
    pub client_key: u64,           // fingerprint of the client's key, 0 if none
    pub attempt: u8,               // 0 for the first try
    pub tried: &'a [WorkerId],     // workers earlier attempts already used
    pub headers: &'a [Header<'a>], // the headers the document allows a policy to read
    pub media: MediaCounts,        // how many images, audio and video inputs; an image or a video asks vision
    pub seed: u64,                 // the only randomness a stage gets; kept in the trace
    pub budget_micros: u32,        // time left for the decision
    pub now_micros: u64,           // the host's clock at the decision
    pub ring: Option<&'a dyn LentRing>, // the hash ring the host keeps, lent to this decision
    pub ring_positions: &'a [u64], // where the request stands on that ring, as the host states it
    // eight more fields: `affinity`, `named` and `signal_cells`, and the five that 5.8 removes
}

pub struct Constraints<'a> {
    pub excluded: &'a [WorkerId], // workers this host just failed to reach
    pub wire_modes: WireModes,    // the protocols this host can dial for the request
    pub overload_lifted: bool,    // the overload veto does not apply to this decision
    pub decode_later: bool,       // a pair's decode will be chosen after the prefill has run
    pub subset_hint: Option<u32>, // decide over at most this many workers
    pub demands: Demands,         // what the request needs of its worker
    pub stalled: &'a [u32],       // positions the host itself holds back
    pub held: &'a [u32],          // those of `stalled` that no rule may lift
    pub places: Option<&'a [u32]>, // the places of the list the request may take; `None`: every place
}
```

What is not built in the engine that stays: a request that states a subset hint, an encode
request that states a demand, or protocols that not every worker of its list serves is refused.
The refusal names the input, except an encode request's demand, which is shed under the stage
`wires`. A pair's demand is kept a side at a time: each side of its two lists is held to the
workers that meet it as that side's role is held to it, and a side it leaves nobody sheds the pair
under that demand's stage, the prefills' first.
A pair's encode items are decided over a host's three lists, with the pair's lease issued
(`PolicyEngine::decide_encode_at`). The trait has no method for them: the one it had took no
lists and could only refuse, and it was removed.
The encoders are the one list the engine narrows by protocol itself. An encoder is named by its
place in its list, and what its host says of it and of each prefill is laid by that place, so the
host has no narrowed list to hand. The decision leaves out each encoder that serves no protocol
the request states, and a request with items is refused under `wires` where the list holds
encoders and none serves one. The pair's two lists are refused as a pair's are.
The other inputs are honoured by the old engine only.
A request that states places of its list (`Constraints::places`) is refused too, and the refusal
names `places`, where the request is a pair or an encode request: one list of places cannot hold
a pair's two lists, or an encode request's three. A single request that states places is decided
among them.

A pair's host states each list's places beside its two lists (`PairLists::prefill_places`,
`PairLists::decode_places`), and each side is decided among its own. An encode request's host
states its encoders' places as well (`EncodeLists::encoders_places`). The encode pass decides its
pair over the workers it admits for the items, and numbers each side's places with them as it
numbers a mask: a place whose worker it left out opens none. An item's encoder is one at the
encoders' places, and a prefill that only other encoders hand to is not the pair's. Where a list's
places name workers and none of them takes the request, though a worker outside them does, the
request is shed under `places`: it is held off every worker that takes it. No gap of the membership
says that, and the gaps stay read over every worker. Places that name no worker of a list are no
such case. Such places of a side shed the request as they shed its pair alone, and such places of
the encoders shed it as a list with no open encoder does.
A pair's host states its standing vetoes the same way, the places each list's vetoes mask and those
a hold stands on (`PairLists::prefill_stalled`, `prefill_held`, `decode_stalled`, `decode_held`).
Each side reads its own, under an encode request too, and a pair's or an encode request's own mask
is refused under the name `places`. An encode request's host states the encoders' vetoes beside their
list as well (`EncodeLists::encoders_stalled`, `encoders_held`), and the encode side reads them.

### 5.5 The decision **[built]**

```rust
// engine/decision.rs
pub enum Decision { Pick(Pick), Shed(Shed), Queue(Queue) }

pub struct Pick {
    pub targets: Targets,                // up to four targets, best first
    pub trace: Trace,                    // how the decision was made
    pub session_verdict: SessionVerdict, // what the decision did with the routing key
}
pub struct Target {
    pub position: u32,      // the worker's place in the list the host handed
    pub id: WorkerId,
    pub dp_rank: u16,
    pub pair: Option<Pair>, // the decode worker, when this target is a pair's prefill
}
pub struct Shed {           // the request is refused
    pub reason: ShedReason,
    pub emptied_by: &'static str, // the filter or picker that left no worker
    pub retry_after_secs: u16,
    pub wait_allowed: bool,       // the host may hold the request (scale from zero)
    pub trace: Trace,
}
pub enum ShedReason {
    Saturated,      // the overload filter left no worker
    FlowLimit,      // a queue is full
    NoEndpoints,    // another filter left no worker, or the host handed no list
    PolicyDeclined, // the picker declined, no binding matches, or the request states what this
                    // engine does not honour
    Deadline,       // the decision's time budget ran out
}
pub struct Trace {
    pub profile: u16,                      // which profile decided
    pub survivors: Survivors,              // how many workers passed the filters
    pub top: [(WorkerId, f32); TRACE_TOP], // the best workers and their scores
    pub top_len: u8,                       // how many of `top` are set
    pub seed: u64,
    pub generation: u64,                   // which generation of the document
    pub lifted: LiftedVeto,                // a soft veto that was lifted for this pick
}
```

`Queue` (the request waits) is defined; no built-in stage issues one **[planned]**.

The HTTP status codes belong to the host. The engine only says why.

### 5.6 One decision, step by step **[built]**

The host calls `PolicyEngine::decide_at(maker, clock)`. `maker` is the host's object for the
request (section 8). The call runs on the caller's thread.

The engine waits on no lock of its own. A stage that keeps state takes its own short lock (one
shard of the session table, or the record of sent prefixes). A decision that meets the one build
of a list's `records`, `wires` or `alone` waits for it (5.2).

**The short path.** A request is *plain* when it names no tried or excluded worker and states no
hint, no demand, no lifted overload veto, no later decode, no held-back worker and no places of its
list.

A profile is decided by *one pick* when its picker needs nothing of the request, it has no scorer
and no filter beyond the health tests, and its binding has no sessions rule and names no shadow.
On main three pickers say so: round robin, `random` and `passthrough`.

1. The engine asks `maker.plain()`. The request takes the short path where all of this holds:
   - the request is plain and a single one;
   - the host handed one list;
   - every worker of the list is dialed over a protocol the request states (where it states
     any);
   - the list's model is bound to a profile that one pick decides.

   The picker then names a worker by its own rule, and the engine answers with its position.

   | Picker | Its rule |
   |---|---|
   | round robin | It adds one to the list's counter and takes the worker it lands on if that worker is eligible. Where it is not, the engine reads the list once and the rotation picks among the eligible workers. |
   | `random` | It draws a position below the list's size from the calling thread's own draw sequence, and again while the worker drawn is not eligible, eight draws at most. Past them the engine reads the list once and draws among the eligible workers. |
   | `passthrough` | It takes the first eligible worker in the list's order. |

   Where no worker of the list is eligible, the request takes the full path.
2. If no stage of the document needs lease events, that is all. No request is built, no clock is
   read, no lease exists, and the host owes the engine nothing more.

**The full path**, in logical order (the code interleaves some steps to save work):

3. **The request is built**: `maker.make(&spec)`. It is built under the document in force, so the
   routing key is derived by the same generation that decides.
4. **The binding.** The first binding in document order whose model pattern and phase match gives
   the plan and the profile. No match is a refusal, `PolicyDeclined`, naming `binding`.
5. **Short answers for a routing key.** A request with a routing key may be answered from the
   key's pin in the session table, or from the workers its key reaches on the ring its host
   lends. Only the records of the workers the key names are read.
6. **Cache affinity.** A profile that reads which worker holds the prompt's beginning has those
   workers named before the pass. After it the prompt is recorded for the first target the
   request's caller sends to, and for none where it sends to none (section 4.5).
7. **Records.** One `Candidate` for each worker is written into the thread's buffer. Over a list
   of 16 workers or more, a profile whose picker draws first (`random`, `round-robin`,
   `passthrough`, power of two, `best` with a sample) has only the drawn workers' records written.
   For a request held to places no worker is drawn first: every worker's record is written.
   Under a binding that names a shadow every worker's record is written, for the two decisions.
8. **Filters.** Each filter removes records. If a filter leaves nothing, the binding's rule for a
   list wholly over its thresholds may still pick among the workers the overload filter removed.
   Otherwise the decision is a refusal that names the filter: `Saturated` for the overload filter,
   `NoEndpoints` for any other. A single request with a routing key keeps its key's worker over
   the overload veto, beside free workers too, where its binding does not shed, the plain ring
   alone places its profile and every worker it may take is a single worker. A sessions rule adds
   conditions of its own. Over workers of another role the veto stands, and the key passes on to
   the next worker its ring reaches.
9. **Scorers and the picker.** Each scorer scores the survivors. The scores are combined by the
   document's weights. The picker orders up to four targets. No target is a refusal,
   `PolicyDeclined`, naming the picker.
10. **The lease.** If a stage needs lease events and the decision is a pick, the engine issues a
    lease (section 6) and tells the stages, unless a sessions rule keeps the pick from them.
11. The answer is the decision with its lease.

**A retry** names the workers it tried and takes the full path. How its tried workers are widened:

- Where its list holds several workers of one engine, the ranks of a data-parallel engine, the
  engine over worker lists names every worker of a tried worker's engine as tried before the
  key's pin is read and the filters run. So `not-tried` passes the whole engine over while
  another worker is left, and hands it again where none is.
- A tried worker the list no longer holds names itself alone: the list does not say which engine
  it was a rank of.
- A pair's retry is widened the same way, each of its two lists by its own engines, and an
  encode request's over its three.

A decision that panics is caught, counted and answered as refused. It never takes the host down.
A stage that panics at a word of a lease, its issue among them, is held and counted apart: the
request keeps its decision and its lease.

A request whose host hands no list (`WorkerLists::None`) is decided over no worker: it is shed
with no endpoint under the stage `none`, and its trace names no profile. A controller hands none
for a model no worker serves, and for a gateway that dials no worker of the model's list.

What one decision writes:

- the thread's own buffer;
- on the short path under `round-robin`, one atomic add on the list's counter;
- one atomic add for a lease id, where a stage needs lease events;
- a session pin, where a sessions rule decides;
- the record of sent prefixes, where a cache-affinity profile decides;
- once for each new load snapshot and list, the converted load reports, which allocates.

On the short path it allocates nothing. A draw that looked up load reports counts them toward its
list's conversion where it takes the list's claim, which it tries and never waits for.

### 5.7 One engine **[partly]**

Main has one engine, `PolicyEngine`, behind the trait `DecisionEngine`
(`engine/decision_engine.rs`). It decides over its host's worker objects, read in place. It keeps
no lease: the host hands the lease back with each event. This document also calls it the kept
engine.

A second engine stood beside it: the store-backed engine, also called the old engine here. It
decided over records copied into the policy store and kept its leases there. Its type is deleted,
a step of stage 4 (section 16). The store's copies of the worker data that only it read are
deleted in the steps behind it (sections 10.2 and 10.3). A controller decides a pair at its
prefill and its decode call over worker lists too; an encode request is answered `Busy` until
it follows (decision 5).

The types the two engines shared stay where they were moved ahead of the deletion: `Request`,
`DecideOutcome`, `DecisionRefused` and `DecodeRefused` in the trait's file, and
`Said` (section 6) beside `Lease`. `GatewayId` is still defined in the store.

### 5.8 What changes in the kept engine **[planned]**

- It honours what it refuses today: an encode request's pair held to some workers of its two
  lists, and its encoders held to some of theirs (`EncodeLists::decide` decides the request
  among the places its host states for each of the three lists, section 5.4, which is
  built); an encode request's demands (a single request's and a pair's are kept, which is built).
- The request fields that only the old engine's host fills leave with it. `pinned_at`,
  `lent_in_flight` and `model_fingerprint` have yet to go.
  Another, `session_counted`, stays, which is built: by it a host says that somebody counts a
  key's requests on each worker, and a sessions rule with a cap asks nothing more of it. Where
  that count stands on a worker object is the request's maker's to say
  (`RequestMaker::session_counted_under`, a provided method): under the key's text for a host
  whose own load guard counts, and under the session's number for a controller, which counts
  by the leases it holds (10.2).
  `Constraints::pins`, the session pins a host kept outside the engine, is removed with
  its trait `PinSource`: no host lent any. A session's worker is to outlive a document in the
  engine's own table, which the front shares with its document's engine
  (`Engine::share_sessions`), once a running engine takes a new document.
- The trait's method that the kept engine answers with a constant, `decide_encode_pair`, is built
  or removed. It is removed, and the guard's maker of that name with it, which is done: a host
  asks the engine's own entry with its three lists (`PolicyEngine::decide_encode_at`) and holds
  the answer (`DecisionGuard::holding`).
- The record of sent prefixes leaves the single decision. The two trees a request is then
  decided in are the host's own, lent (section 4.5).

`signal_cells` stays, and it is built for a single request: where the request's profile reads a
named signal, the kept engine fills the field itself, from the reports of the load snapshot the
decision holds. Such a decision converts its list's reports where no decision under its snapshot
has, ahead of a routing key's short answer too. A row that is one rank of a data-parallel worker
reads its own rank's values. A worker object that stands for several ranks reads one value a
name: its ranks' values made one as the signal's `across_ranks` says, under the oldest sample
among them. The worker crate's report states each rank's own sample instant, so a rank that
stopped reporting ends the worker's value at the signal's `max_age_ms`. The report is read where
it lists as many ranks as the object's registration states it stands for, its width with no rank
of its own: a pushed report counts the rows it holds, so the object's word is what says a rank is
still to report. A report of several ranks is read of no other object, and a report that lists
fewer ranks than it counts gives no value. A report of one rank is read as the worker's whole, so
the first record of an object of several ranks stands for it until a second rank's record.

A pair is lent them a side. Where the prefill profile reads a named signal, the prefill side
reads its prefill list's cells, and the decode side reads its decode list's where the decode
profile reads one. Both are of the snapshot the decision holds, and a decode decided again
reads those of its own call. The pair's pools carry each side's cells
(`PairPools::prefill_signal_cells`, `decode_signal_cells`), so the request is not copied.

An encode request decided over its three lists is lent them a side too. The two sides of its
pair are lent as a pair's first decision's. Its encode side reads its encoder list's cells
(`EncodePool::signal_cells`) where the encode profile of its pair's binding reads a named signal:
that side runs under the binding the prefill lookup resolves to, whatever binding covers the
encode phase first, and the engine's answer for an encode decision carries that profile's needs.

## 6. The lease and its events **[built]**

A lease is the engine's note that a decision's request is on its way. For a single request it
exists only when a stage of the document wants to hear what became of requests. A pair's pick
always has one.

```rust
// tracking/lease.rs
pub struct Lease {
    pub id: LeaseId,
    pub kind: LeaseKind,
    pub side: Side,                      // Single, Prefill, Decode or Encode
    pub target_index: u8,                // which target of the pick it books
    pub booked: WorkerId,                // the worker it books
    pub generation: u64,                 // the document generation that decided
    pub session: Session,                // the request's session
    pub session_verdict: SessionVerdict, // copied from the pick
    pub issued_micros: u64,              // the host's clock at issue
    pub tokens: u32,                     // the token count the request stated, 0 for text alone
    pub chars: u32,                      // its text's characters, which `bucket` books by
    pub bytes: u32,                      // its text's bytes; `prompt_size()` as the facts'
}

pub enum Said {
    Dispatched(Dispatch, Instant),         // the request was sent to a target
    Completed(Outcome),                    // one side ended, with its outcome
    Released(Option<Side>, ReleaseReason), // given up before completion
}
```

```text
decide ──> lease issued ──> Dispatched ──> Completed
                │                └───────> Released (client gone, send failed, worker removed)
                └── never dispatched ──> Released when the host drops its guard
                                         (the old engine releases it 5 s after issue)
```

The kept engine stores nothing about a lease. The host keeps what it was handed and hands it back
with each event (`DecisionEngine::say`). The crate has a guard for that, `DecisionGuard`
(`tracking/decision_guard.rs`): it holds one decision's lease, has `dispatched`, `completed` and
`released`, and releases at its drop.

What an event does:

| Event | What the stages do with it |
|---|---|
| Dispatched | the session's pin follows the worker the request went to; each profile counts what it sent since the worker's last load report; scorers and pickers that asked hear it. A pick a sessions rule keeps from the profile moves the pin alone. A pin at its rule's cap moved already, where the request that met the cap was decided, to the worker it was placed on: the dispatch finds it there, or fronts the later target the request went to |
| Completed | scorers and pickers that asked hear the outcome (status, time to first token, tokens in and out, duration) |
| Released | the same stages hear the reason |

A sessions rule places some requests alone, behind the profile's filters and with no scorer or
picker of the profile asked: it follows a pin of the request's key, or its own assignment places
the key. The rule says whether such a request is the profile's own (`rule_placed`). Under `told`,
the default, it is: the stages hear its issue and its dispatch,
the profile counts it among its sends, and its prompt is recorded as behind any pick. Under
`untold` it is kept from the profile: no stage hears its issue or its dispatch, no profile counts
it, and neither the engine's record of sent prefixes nor a lent token tree takes its prompt. The
session's pin still follows its dispatch, and its completion and its release are told as any
request's, so a stage matches an end by the lease's id. The routing settings write `untold` for
the sticky override's rule alone: a gateway's sticky map keeps a request it places from the
gateway's policy. The kept engine keeps the word on the pick, and one function reads it
(`lease_events::stages_hear`) at four places: a single's lease words, the record's two writes, the
tree's store, and the hand-over of a pin followed by one row. Such a pin that is the profile's own
goes through the lookup that names a request's holders with its answer made, so its prompt is
recorded as a pin's the whole decision follows. A controller asks the same function where it files
a pick's lease, and keeps the answer on the lease's record (`LeaseRecord::untold`): no stage hears
that issue, and the dispatch it tells off the file moves the session's pin and reaches no stage
and no count of sends. Its decision is the engine's own, so its record of sent prefixes follows the
word too.

Upkeep, about once a second (`sweep`): expired session keys are removed, and the record of sent
prefixes gets a bounded step of clean-up. A record leaves the engine's map once no list's table
holds it and it holds no worker's place.

## 7. The ten built-in policies **[built]**

Each classic policy name has a preset in the crate. Every preset starts with the filters
`healthy`, `not-overloaded` and `not-tried`. In a pair, the prefill side adds `has-capacity` and
the decode side adds `has-room`.

| Policy | Scorers | Picker | What it remembers between decisions |
|---|---|---|---|
| `random` | none | `random` | nothing: it draws from the decision's seed, and the one pick from the calling thread's own draw sequence |
| `passthrough` | none | `passthrough` | nothing: the first survivor |
| `round_robin` | none | `round-robin` | one counter for each worker list, which the one pick and a first attempt's whole decision both turn, at every list size; in a fixed table, one rotation for each set of surviving workers, which a refused turn, a retry and a request held to places turn, and one for each pool whose draw is handed no list counter |
| `power_of_two` | `least-load` | `power-of-two` | what was dispatched to each worker since its last load report |
| `least_load` | `least-load` | `best` | the same |
| `consistent_hashing` | `consistent-hash` | `best` | nothing where the host lends its ring and states the request's place; else one hash ring, of the workers its decisions have read |
| `prefix_hash` | `consistent-hash` by the routing key, else by the prompt's first 256 tokens, with a bounded-load rule; `least-requests` | `best` | the same ring |
| `manual` | none | `manual`, with a sessions rule | routing key to pinned worker, with an expiry |
| `cache_aware` | `warmup-slice`, `prefix-affinity`, `kv-pressure`, `least-load` | `best` over a sample of 16 | the record of sent prefixes; the warm-up counters; the dispatch count of `least-load` |
| `bucket` | `bucket` | `best` | for each model, the ranges of prompt sizes each worker takes, and a time window of booked tokens |

Whether each preset gives classic's answer is section 13.

## 8. What a host does

This is the contract between the engine and the program that embeds it. The gateway and a
controller keep it differently on main.

- **At start.** The gateway takes steps 1 to 3 of section 8.1's start. It hands no lists yet
  (step 4). A controller takes step 3 alone: its holder lends each role's list with its cell
  (section 4.1). Its store builds an engine object for each document it applies, with no
  hand-over (section 10.2).
- **For each request.** The gateway makes the one call of section 8.2 and keeps the placement. A
  controller states worker ids of its own and keeps no placement: it calls the entry that issues
  nothing and files its own record.

### 8.1 On main today **[built]**

A host's start has these steps.

1. **It starts the engine with its own data, once.** The call is
   `PolicyEngine::start(spec, WorkerData { registry, loads, cache_indexes })`. It builds the
   engine from the document and takes three things of the host:
   - its worker registry (the host implements `Members`, and the worker crate's registry is one);
   - its load monitor's channel;
   - its indexes of cache events (the host implements `CacheIndexes` and `CacheIndex`). The same
     hand-over also lends the token tree, where the host keeps one (`CacheIndexes::token_tree`).

   A host with no load monitor or no index says `None`. The call answers the engine, which lives
   as long as the process, and the engine's own task.
2. **It spawns that task, once,** on a runtime of its choice, which needs its timer on. The task
   does five things:
   - it runs the upkeep once a second;
   - it keeps the newest load snapshot at hand;
   - it follows the registry and tells the policies a worker that joined or was removed (the
     table below);
   - it tells each new load report to the policies that keep them;
   - it fills the cells of the lists the host handed, ahead of their requests.
3. **It keeps an empty cell beside each published worker list**: a `ListCell` in its value of a
   membership generation. The cell is lent with the list as a `LentList`, with `facts` where the
   host knows a fact of its workers that their objects do not hold.
4. **It hands its worker lists by model, once**: `engine.set_lists(lists)`, beside the start,
   which does not take them yet. The host implements `ModelLists` over its registry. The engine's
   task fills a list's cell behind the registry's word of the membership change. A later call is
   ignored.
5. **It hands its load monitor's channel again where it builds another monitor**:
   `engine.set_loads(channel)`. An engine keeps no channel where its document reads no load
   report and no named signal, and none of its policies keeps a report.

A caller that builds its engine with `PolicyEngine::new(spec)`, as the crate's own cases and
bench do over lists made by hand, makes each hand-over itself:

- `engine.set_registry(registry)`. With no registry the caller says `engine.forget(address)` when
  a worker is removed and `engine.joined(address)` when one joins. It says a removal and a
  registration at one address in that order.
- `engine.affinity().set_indexes(indexes)`.
- `engine.set_loads(channel)`.
- It spawns `engine.run()`.

A host makes none of these calls but `engine.set_loads`, where it builds another load monitor.

The table says what the engine's task tells the policies at each event of the registry. The task
names a worker by the id the host states of it (`Members::facts_of`), which is the id the host's
lists name it by. A worker whose host states none is named by the fingerprint of its address.

| The registry's event | The policies are told |
|---|---|
| `Registered` | a join |
| `Removed` | a leave, under the id told at the worker's join |
| `Replaced` | nothing where the host states the same id of the new object; a leave of the old id and a join of the new one where it states another; nothing of an object the task does not hold |
| `StatusChanged` | nothing: a decision reads a worker's status on its object |

Three cases need more words:

- **Two workers at one address.** A worker removed and another registered at its address come as
  a `Removed` and a `Registered`: a leave and then a join. Where the registration is heard first,
  the worker is read as its id's next life: a leave and a join of the id, and the removal then
  tells nothing. That holds under the registry's own id of a worker too: a removal tells a leave
  only of the worker the task holds.
- **A reader that falls behind.** The events are a broadcast that drops its oldest for a reader
  that falls behind. The task then reads the registry whole, tells what changed, and last tells
  the ids that stand. So a stage keeps nothing of a worker whose whole life the channel dropped.
  The task does the same behind a channel that closed.
- **A host that states ids.** The task reads the registry whole in the same way at its next
  upkeep behind every registration, replacement and removal of a host known to state ids, from
  the first ask it answers with one. Such a host may have stated an id of a worker and taken it
  back ahead of the event.

The session table is told nothing: a pin on a worker that is gone stands until the session's idle
time or two other workers end it.

For each request a host implements a small trait and calls the engine. A controller does so for
each frame it decides. The gateway makes the one call of section 8.2, and the crate implements the
trait for that call. A host writes `model` and `make`; every other method has a default body:

```rust
// engine/decision_engine.rs
pub trait RequestMaker {
    fn model(&self) -> &str;                          // the request's model
    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w>; // builds the request
    fn plain(&self) -> Option<Plain>;                 // what a plain request states, if it is plain
    fn prompt_chars(&self) -> u32;                    // the length of its text in characters, asked for a plain request's lease
    fn prompt_bytes(&self) -> u32;                    // the same length in bytes
    fn session_counted_under(&self) -> SessionCountedUnder; // where a worker object counts its session key's requests
    fn workers(&self) -> WorkerLists<'_>;             // the worker list, a pair's two lists, or an encode request's three
    fn loads(&self) -> Option<&Arc<LoadSnapshot>>;    // a request's snapshot of the host's own; without one the engine reads its own
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>>; // the prompt's entry in the host's token tree, where the host lends one
    fn sends_to(&self, target: &Target) -> bool;      // whether its caller sends to this target of a pick: the prompt is stored for the first such
}

pub enum WorkerLists<'a> {
    None,
    Single(LentList<'a>),    // one list, for a single request or one that stands alone, lent with its cell
    Pair(PairLists<'a>),     // a pair's prefill list and decode list
    Encode(EncodeLists<'a>), // an encode request's three lists: its pair's two and its encoders
}

pub enum Decided {
    Alone(AlonePick),           // a position in the list; nothing is owed to the engine afterwards
    Whole(DecideOutcome, bool), // a full decision with its lease
}
```

A host that calls the engine through this trait alone has more to do itself. Among its duties:

- derive the routing key by the document's rule and hash the prompt into blocks, inside `make`;
- try the targets in order, checking each worker's health again;
- keep the decision's lease and report each event;
- turn a refusal into a status.

That is too much. The gateway's earlier integration needed a module of its own of 19 files and
13,856 lines (tests included) for it, and a branch in every route. For a single request and for a
pair, `place` of section 8.2 now derives the key, hashes the prompt and tries the targets, and its
`Placement` keeps the lease. The host still says the two events and turns a refusal into a status.

**A lent hash ring.** A host that keeps a hash ring of a model's workers can lend it with a
request. The request's facts carry the ring, behind one trait (`LentRing`), and where the request
stands on it (`RequestFacts::ring`, `RequestFacts::ring_positions`).

There is a trap for the lender: classic asks its ring for two different texts.

| Policy | What classic asks its ring for |
|---|---|
| `prefix_hash` | the sixteen hex digits of the hash of the bytes a request is placed by, also where those bytes are a routing key's (`ring_key` and `find_worker_with_load_balance` in `model_gateway/src/policies/prefix_hash.rs`) |
| `consistent_hashing` | the routing key's own text (`find_by_consistent_hash` in `model_gateway/src/policies/consistent_hashing.rs`) |

A lender that states a request's place from the wrong text sends routing keys to other workers
than classic does, and no test in the crate sees it.

The trait keeps the two questions apart: `position` answers for a key's own text, `head_position`
for a head of the request, each by the lender's own recipe.

A caller of the one call states the ring alone (`PlaceRequest::ring`). The engine reads where the
routing value stands on it with the request's keys, under a profile whose ring stage places a value
by its own text. Under `prefix_hash` the engine cuts the request into the heads classic cuts it
into, a routing value first, and asks the ring where each stands (`LentRing::head_position`). A
ring whose lender places no head answers nothing: such a request is lent no ring, the lender is
asked about no further head, and the stage's own ring places the request. A request that has no
head states no place and is lent no ring either.

### 8.2 The contract as designed **[partly]**

Everything in the list above that is the same for every host moves into the crate. A host does
three things. The names below may change; the contract may not.

**1. At start, hand over what it already has, once.**

```rust
// engine/policy_engine/start.rs
let (engine, task) = PolicyEngine::start(spec, WorkerData {
    registry,                              // the worker registry: the engine's task follows its events
    loads: Some(load_monitor.subscribe()), // the load monitor's channel
    cache_indexes: Some(kv_monitor),       // the host's indexes of cache events
});
engine.set_lists(lists);                   // the worker lists by model: the task fills their cells
runtime.spawn(task);                       // the engine's own background task
```

The engine lives as long as the process, so a placement holds a plain reference to it.

The start answers the task, and the host spawns it on a runtime of its choice: the start spawns
nothing. The call is marked `#[must_use]`, with the reason that the task does nothing until it is
spawned. So the compiler warns a caller that drops the whole answer. It does not warn a caller
that keeps the engine and drops the task. That caller's engine keeps no load snapshot, runs no
upkeep, tells the policies no worker, tells no policy a load report and fills no cell ahead of a
request.

The start takes no worker lists. Each request gives its own. The engine's task asks only the
lists a host hands it beside the start, to fill their cells (the fifth job below).

The engine's background task does what a host did by hand. Five of its jobs are built
(`PolicyEngine::run`):

1. the upkeep, once a second;
2. the newest load snapshot kept at hand, swapped when the monitor publishes one, so that a
   decision reads it without touching the channel;
3. a worker that joined or was removed told to the policies, read from the events of the
   registry the host hands at the start (section 8.1);
4. each new load report told to a policy that keeps a number of it;
5. a model's cells filled ahead of its requests, for a host that handed its lists.

A host that builds another load monitor hands its channel again (`engine.set_loads`): a channel
ends with its monitor.

The task's first upkeep runs at its first poll, where it reads the registry and takes the load
channel. So a host relies on both once the task has been polled once, not when the start
returns.

The fourth job serves a policy that keeps a number of each worker's load report
(`Lifecycle::keeps_loads`). At each load snapshot the task takes, it tells such a policy the
reports that are new in it, for the workers it follows in the host's registry, and only then keeps
the snapshot at hand. A report is new where its sample instant is not the one the task told last
of its worker. The channel keeps one snapshot, so of two reports of one worker the task took no
snapshot between, the earlier is told to no policy. A told report states its named values too, one
cell a signal the document declares and in the document's order, under a document a stage of which
keeps a signal's column: a stage reads its column's cell (`SignalColumn::told`) for the time of the
call and copies out what it keeps. A report that is not the whole of its worker object states
none, as a list's cells hold none of it, and neither plugin view carries the values yet.

The fifth job fills a model's cells ahead of its requests, for a host that handed its lists
(`PolicyEngine::set_lists`). The task asks the host for lists (`ModelLists::lists_of`):

- behind a word of a registration, a replacement or a removal, for the lists of each model the
  word's worker serves;
- behind a whole read of the registry, for the lists of each model a standing worker serves.

It asks each handed list's value, which fills a cell that holds none, so the list's first
request reads a filled cell. Words that stand ready together are heard first, and each model is
asked once behind the last of them. The upkeep asks what is owed at that minute too, with words
still standing. Lists handed behind the task's first poll are owed at its next upkeep, every
list that stands once.

The task asks at those moments alone, and it fills the lists that stand at its ask. So six kinds
of list are filled ahead only where a later ask of their model finds them, and by their first
request otherwise:

- one a request asks ahead of the task;
- one published again ahead of the task's ask;
- one the registry publishes with no event (the worker crate's registry does so where a worker's
  model cards are refreshed);
- one of a model its word's worker no longer states;
- one lent under no model a worker serves;
- one the host builds behind the task's ask.

These faults are held and counted with the task's faults (`PolicyEngine::upkeep_faults`):

- a panic of the host's own code where a worker names its models, which costs that worker's
  fills alone;
- a list refused for its facts, which costs its own fill alone;
- a panic of the host's own code where it hands a model's lists, which costs the lists of that
  model not yet handed.

The policies are told the word all the same.

The host keeps one empty cell beside each worker list it publishes. That is one field in the
host's `Generation` type. The engine fills the cell at the list's first ask: the task's where
the host handed its lists (the fifth job above), and the first decision's otherwise.

The cell is built. A request's `WorkerLists` name each list as a `LentList`, and the decision
that needs a list asks its cell. The start does not take the lists' trait yet. A host hands its
lists by `PolicyEngine::set_lists`, and the field of the start's data comes with the change in
which the gateway hands its own lists.

**2. For each request, one call.**

```rust
// engine/policy_engine/place.rs, behind the feature `in-process`: built for a single request, a pair and an encode request
pub struct PlaceRequest<'a> {
    pub model: &'a str,
    pub lists: WorkerLists<'a>,       // one list, a pair's two lists with their pair classes, or an encode request's three
    pub prompt: Prompt<'a>,           // token ids or text, and the seeds of its cache partition
    pub headers: &'a dyn HeaderRead,  // the engine reads the headers the document names
    pub request_id: Option<&'a str>,
    pub tenant: Option<&'a str>,
    pub priority: i32,
    pub streaming: bool,
    pub max_output_tokens: u32,
    pub media: MediaCounts,
    pub demands: Demands,             // what the request needs of its worker
    pub encode: Option<EncodeItems<'a>>, // the items of an encode request
    pub attempt: u8,
    pub tried: &'a [WorkerId],        // workers earlier attempts used
    pub budget_micros: u32,           // time left for the decision
    pub decode_again: bool,           // the caller asks a pair's decode again behind its prefill
    pub ring: Option<&'a dyn LentRing>, // the hash ring the caller keeps of the model's workers
    pub admits: Option<WorkerRule<'a>>, // the route's own rule, asked of the pick
}

pub type WorkerRule<'a> = &'a dyn Fn(&dyn Worker) -> bool;

impl PolicyEngine {
    pub fn place<'a>(&'static self, request: &PlaceRequest<'a>) -> Result<Placement<'a>, Unplaced>;
}

pub struct Placement<'a> {               // built, see part 3
    pub worker: Arc<dyn Worker>,         // where the request goes
    pub decode: Option<Arc<dyn Worker>>, // a pair's decode worker
    pub encoders: Encoders,              // an encode request's encoders, one an item: one pointer wide
    pub routing_key: Option<&'a str>,    // the key the document derived, for the in-flight counters
    pub profile: &'static str,           // the label of the profile that picked, for the host's own counts
    pub decode_profile: Option<&'static str>, // a pair's decode side's profile, by its label
    // private: the lease, when a stage of the document wants events
}

pub enum Unplaced {
    Shed { reason: ShedReason, emptied_by: &'static str, retry_after_secs: u16 },
    Queued,
    PrefillsFull, // the host's admission holds back every prefill of a pair that has an open partner: the host waits for room
    NoWorkers,
    Refused(DecisionRefused),
}
```

Inside `place` the crate derives the routing key, hashes the prompt, decides, and tries the
targets best first: health and circuit breaker are read again, and the route's rule is asked of
the picked worker. If none holds, it decides again over what is left. The host sees a worker or a
reason.

Built so far, for a single request, a pair over its two lists and an encode request over its three: the key, the hashes, the decision, and its targets tried
best first by health, circuit breaker and the route's rule. Where none holds, a lease the pick was issued is given up at once and the decision is made again, told to leave out the workers that could not be sent to, three decisions a call at most, or a single request's fourth where its third found the workers the rule takes; then the request has no worker
(`NoWorkers`). Planned: the later decisions of a pair and of an encode request held to the workers the rule takes.

A pair's target holds where both its workers can be sent to.

A profile's `not-tried` filter leaves those workers out while another worker remains, and a
profile that lists no such filter decides over the whole list again, or over the workers the rule takes once a single request's call is held to them. Of a pair's target they are
the sides that could not be sent to: a side that held stays a candidate. A worker the rule
refuses is left out like one that is not ready. Until a pair's later decisions are held to the
workers the rule takes, a pair whose rule takes few workers can have no worker though its lists
hold one the rule takes: where three decisions' picks are all refused, where that worker is
overloaded and a refused one is not, and where an earlier attempt tried it.

Behind a single request's first decision none of whose targets holds, where the rule refused one
that was ready, the rule is asked of every worker of the list once, ready or not, and the later
decisions are held to the workers it takes. So the rule's one worker of a long list is found, one
over its overload thresholds is answered by the binding's rule for a list wholly over them, and
one an earlier attempt tried is handed again where no other is left. A request whose rule takes
no worker of the list is shed with no endpoint, under `admits`. A single request's later decision
is given what the call has left of the request's budget, and a request with none left when a
decision is due is shed under `budget`. A pair's decisions are each given the whole budget.

**[built]** A single request can name its worker. A binding's `routing_key` block names a target
source (`KeySourceSpec::target`): a header whose number names the one worker the request goes
to. The preset of `consistent_hashing` names `x-smg-target-worker` so, as classic's policy of
that name reads it, and the routing settings write the same source for that policy alone. The
one call reads the header once, ahead of its decision. The number counts, from 0 and in the
list's order, the workers that are available by the worker's own test and that the route's rule
takes, as classic counts them (`model_gateway/src/routers/common/placement.rs:1187-1188`,
`:1225-1234`; `model_gateway/src/policies/consistent_hashing.rs:154-160`, `:262-264`). The call's
one decision is held to the place of the worker the number reaches (`Constraints::places`): a
retry is handed that worker again, and the request goes to that worker or to none. Where the
text is no number, fewer workers can be sent to, or the worker cannot take the request after
all, the request is shed with no endpoint, under `target-worker`. Whether a binding names a
target source is decided when the document is applied: a request under a document that names
none pays one test of that word and no read of a header.

Where no worker of the list can be sent to, there is none to count. Classic steers such a
request to its least-loaded worker, as it steers any request, and lifts no veto for its routing
value (`placement.rs:1256-1258`; `consistent_hashing.rs:262-264`). The engine places it as a
request that names no worker and carries no routing value, by the binding's rule for workers
that are all vetoed.

One thing differs from classic there, by rule. Classic reads the header once a side for a pair,
each side's policy over its own workers (`placement.rs:2519`, `:2540`, `:2616`). **[planned]**
The engine reads no pair's and no encode request's header for a target yet.

An encode request is placed over its three lists as a pair is over its two: three decisions a
call at most, each given the whole budget, and none held to the workers the rule takes. One
decision names its pair and each item's encoder, and the pair's targets are tried as a pair's
are. The placement names each item's encoder beside the pair (`Placement::encoders`). An item has
one encoder: where it cannot be sent to, the decision is given up and made again with that
encoder left out, and each side of the pair that could not be sent to either. An encoder is read by its health and its circuit breaker alone: the route's
rule is asked of the pair's two workers. Three lists with no item are placed as their pair. A
request whose decode is held to the workers that take images books its decode with its prefill,
whatever its caller states of asking the decode again: a decode asked again reads no image fact.
That decode never stands chosen.

A retry stays on its wire by its demand alone. The host gives one number to each runtime and
transport it dials. It writes that number as each worker's fact and as the retry's demand, and
it states no wire modes for a list it dials whole. The engine compares the two numbers and
reads no more of them. So a pool on two transports is handed as one list, and no host keeps a
list by wire.

Two things keep the short path of section 5.6 as cheap as it is today. The route's rule is asked
of the one picked worker, and of a single request's whole list only behind a decision all of whose ready picks it refuses; a rule that is the same for every request of a
route belongs to the list itself. And a plain request touches neither the load snapshot nor the
other fields of the request.

A request decided inside a lent token tree has the rule asked inside the tree's walk, of the
pick's targets best first, so the tree stores the prompt for a worker the rule takes. The call
takes that answer and asks the rule of that worker no second time.

**3. Keep the placement with the request.**

```rust
// tracking/placement.rs, behind the feature `in-process`
impl Placement<'_> {
    pub fn sent(&mut self, side: Side);                    // one side of the request went out
    pub fn ended(&mut self, side: Side, outcome: Outcome); // one side ended
    pub fn id(&self) -> WorkerId;                // the worker's id, for the next attempt's `tried`
    pub fn decode_id(&self) -> Option<WorkerId>; // a pair's decode worker's id
    pub fn owes_events(&self) -> bool;           // a side under a lease has not ended
    pub fn kept(self) -> Placement<'static>;     // without the borrowed key, to keep to the end
    // a pair's decode decided again behind its prefill's answer, the request stated again
    pub fn decode_again(&mut self, request: &PlaceRequest<'_>) -> Result<(), Unplaced>;
    // dropped with a side not ended: the engine hears a release before the side went out,
    // and an end of class cancelled after it
}
```

What a placement holds, and so what these calls do, depends on the request:

- **A single request.** Where the document's stages want no events, its placement holds nothing
  and these calls do nothing.
- **A pair.** Its placement always holds a lease (section 6).
- **An encode request.** Its placement always holds a lease too: it reports its pair's two sides
  as a pair does, and nothing of an encode side. No lease stands for an encoder, so `sent` and
  `ended` under the encode side tell no stage and end no side. The host that sends an encode
  call counts it on the worker itself, as the gateway's encode stage does.

`place` answers a single request, a pair and an encode request with a placement, and no host
makes one itself.

**A pair's decode decided again.** A pair's decode can be decided again on its placement:

1. Under a document that picks the decode after the prefill, a caller whose request states
   `decode_again` is placed with its prefill booked alone. The decode the first decision named
   stands chosen.
2. When the prefill has answered, the caller calls `decode_again` with the request stated
   again, as its first call stated it.
3. The engine makes the request anew from what the caller stated, as it makes a first call's,
   and decides under the binding of the model stated there. Every fact of the decode's decision
   is that request's: the pair's two lists as they stand then, the request's size, its routing
   value and the workers it tried among them. The engine decides the decode on the records of
   that instant, behind the prefill that was sent.
4. The placement takes the first decode the engine names that can be sent to. Its routing key
   and its two profile labels stay as the first call wrote them.

The order is part of the contract. The caller asks the decode again, or says the send of the
first decision's decode, before it says the prefill's end: a decode still chosen at that end
leaves with it. Behind a decode the call took, the prefill's end is said when the prefill
answers, and the decode's send at its own later instant.

Where the call takes no decode, the placement stands as it stood. A shed is answered by its
name. A decision none of whose decodes can be sent to answers `NoWorkers` and gives up the side
it issued. The caller may then ask again, send the first decision's decode, or answer its
client.

**The sides of a placement.** A host that keeps a value with each send, as a gateway keeps a
worker's load guard, takes a kept placement apart into its sides. A side says its dispatch by one
call and its end at its drop. Whoever reads the side's answer reports its outcome through a
handle on the side's end:

```rust
// tracking/placement_side.rs, behind the feature `in-process`
impl Placement<'static> {
    // a single request's one side; a pair's prefill and its decode
    pub fn sides(self) -> (PlacementSide, Option<PlacementSide>);
}
impl PlacementSide {
    pub fn sent(&mut self); // this side of the request went out
    pub fn profile(&self) -> &'static str; // the label of the profile that picked this side
    // a pair's decode side: its decode decided again, answered the decode the request goes to
    pub fn decode_again(&mut self, request: &PlaceRequest<'_>) -> Result<Arc<dyn Worker>, Unplaced>;
    pub fn side_end(&self) -> SideEnd; // a handle on this side's end
    // dropped with no outcome reported: a release where the side never went out, an end of
    // class cancelled where it did
}
impl SideEnd {
    pub fn reported(&self, outcome: Outcome); // the side ended with this outcome
    pub fn is_empty(&self) -> bool;           // a report through it reaches nobody
}
```

**A handle on a side's end.** The value a host keeps with a send is often not where the answer
is read: a streamed response's guard is held in the response's body, and another task reads the
worker's answer. So whoever reads the answer holds a handle, which is cheap to clone and may
outlive the side.

The side's end is said once the side and every handle on it are gone. It is said with the
outcome last reported or, where none was, in the words of a dropped side: a release or an end of
class cancelled. Into a reported outcome the side writes its own name, the worker that served
and the span from its dispatch to the report. A host that asks no handle, and a handle dropped
with no report, change nothing.

Under a document none of whose stages takes an end, a handle is empty and costs nothing: no
allocation and no lock. Under one that does, a side shares its placement with its handles: one
allocation a side.

A pair's two sides share the pair's one placement behind a lock of that one request. The
crate keeps the order of a pair's words itself: a prefill side let go while the decode still
stands chosen holds its word back until the decode side speaks, and says it then as it stood.
A prefill that went out says its end, with the outcome reported or with the instant of its own
drop. One that never went out is released. So a host that sends a pair's decode and no prefill
orders nothing by hand.

A host that keeps sides asks the decode again on the pair's decode side, once the prefill
side's send was said. It may let the prefill side go before that call or behind it. A prefill's
end that waits is said behind the decode the call took. Behind a call that took none, it still
waits. A side that is not a pair's decode side is refused by name, with no decision made.

Nothing else is asked of a host that makes the one call (`place`). In particular it never names
a policy, never looks inside a decision, never keeps a lease and never runs a timer for the
engine.

### 8.3 What is reused and what is added

The design adds no second mechanism where one exists.

| Need | Reused, as it exists | Added |
|---|---|---|
| a model's worker lists | the registry's published lists (`Generation`) | one field beside each list: the engine's value |
| a worker's health, breaker, overload, in-flight count | the worker object, read in place | nothing |
| load reports | the load monitor's `LoadSnapshot` | nothing |
| which worker holds a prompt, by cache events | the kv-index crate's index, through two traits | the gateway's two readers of its index for those traits (section 9.2) |
| which prefill pairs with which decode | the gateway's pair index (`PdPairIndex`) | the pair classes read from it, one file (section 9.2) |
| which worker was sent a prompt | the host's own trees, of the kv-index crate's two tree types | one more question on the trait the host already answers for cache events |
| a request's lifetime on a worker | the load guard every route keeps | one optional field (decision 4) |
| the place where a worker is chosen | classic's five placement functions | one call in each |
| the rank a request is sent with | the worker object's own rank, where its registration states one | nothing: no rank choice runs behind the pick of a worker registered rank by rank, and the classic rank option is refused at the start beside the engine (section 9.2 names the two places a rank is still chosen) |
| the lease's reporting | `DecisionGuard` | `Placement`, which holds it |
| proof against classic | classic's recorders and their sheets | replays through the kept engine |

## 9. The gateway as a host **[partly]**

After stage 0 the gateway did not call the engine. This section is the one change that brought it
back, behind an option that is off by default (section 9.3).

### 9.1 Where classic chooses a worker

Every classic route ends in one of five functions:

| Function | Used by |
|---|---|
| `placement::select_single` | HTTP typed requests; the token-count route |
| `placement::select_single_of` | the gRPC regular stage |
| `placement::select_from` | transcription |
| `placement::select_pair` | the HTTP and gRPC prefill-decode routes |
| the encode stage's own selection | the gRPC encode-prefill-decode stage |

Four are in `model_gateway/src/routers/common/placement.rs`. They take what a classic policy
reads: the request's text or token ids, its headers, its request id, its cache partition, the
route's own rule for a worker, and the workers already tried.

Three places choose a worker with no policy at all (a proxied GET takes the first worker; a
realtime session takes the least loaded; a third-party provider's router takes the first worker
of its pool). They are not policy decisions and stay as they are.

### 9.2 The change

```text
request ──> route (HTTP or gRPC, any family)
              │  has what classic placement reads today
              ▼
        the five placement functions
              │
              ├─ engine off ──> the classic policy object                 (unchanged)
              │
              └─ engine on  ──> engine.place(request) ──> Placement
              ▼
        the load guard for the chosen worker   (exists today; lives as long as the request)
              ▼
        send ──> response ──> the guard is dropped ──> the engine hears: ended, or released
```

- **Start.** One function builds the engine from the routing settings or a supplied document and
  hands it the worker registry, the load monitor's channel and the cache-event monitor. With
  the engine on, no classic policy object is built. So the start makes the token trees the
  engine is lent **[built]**, where the document reads the prefixes sent to each worker. The
  policy registry keeps them, classic's size pass runs over them on its timer, and a worker that
  left is taken out of them. The trees take the document's numbers: its block size is a tree's
  page, and a model's `index` entry states the size its tree is held to and how often.

  The start makes the policy crate's one start call as the last step of the context's build. The document is checked first against the table of stages the gateway's host hands the start, or the stock table where it hands none, and a refused document ends the start before the gateway listens. The engine's task follows the registry's events and tells the document's stages a worker that joined and a worker that was removed. With the engine on, the gateway builds its policy registry with no classic policy object: a stand-in that places nothing stands in its default and encode slots, and a worker's registration and a router's build add no policy.

- **Each request.** Four of the five functions ask the engine. They fill `PlaceRequest` from the inputs
  they already take. A route names the answer's type and makes its load guard from the answer, in the places it made one before.

  The fifth, the encode stage's own selection, takes its pair from `select_pair` and refuses a request that carries an item for the encode workers by name **[planned]**: the gateway does not yet hand the engine an encode request's three lists.

  The pair's function makes its call twice in one case, as a stopgap: where the first call answers that none of its workers could be sent to, the request carries a rule of its route, and that rule takes a prefill and a decode it pairs with that can both be sent to, the overload veto apart. The second call holds both pools to the workers the rule takes. The engine is to hold a pair's later decisions to the rule inside its own call, as it holds a single request's (section 8.2) **[planned]**.

  Where the document asks a pair's decode again, the two pair routes make one more call between the prefill's answer and the decode's dispatch (section 8.2).

  A retry held to its wire is placed over the pool's whole list: the gateway holds it there by the route's own rule for a worker, and a pair's retry by the places of each pool on that wire. It does not yet give each runtime and transport it dials a number, as a worker's fact and as the retry's demand (section 8.2) **[planned]**.

- **The request's end.** The placement stays with the request's load guard, which every route
  already keeps to the end of the response, streamed or not (decision 4).

  The engine hears the dispatch where the guard is made, or at the send for a guard a route makes
  ahead of its send, and the end where the guard is dropped. It is told no outcome. A stage hears
  the end of a request that was sent as a cancelled completion, with the time since its dispatch
  and no other measure. It hears a guard dropped before its send as a release, abandoned. So a
  plugin stage that learns from outcomes reads every request sent in process as cancelled.

- **Membership.** The gateway's value for a model's workers gets one cell for each list, which
  the engine fills.

  The pair classes a gateway hands with a pair's two lists are derived under the pairing of the document in force: beside a supplied document the registry takes that document's `plans.pd_pair.pairing` at its build, and no request reads it.

  On the wire that carries one runtime's request the classes also state each prefill's first partners: its partners on its own known runtime (`PartnerClasses::number_first`). The engine makes a pair among the prefills with a first partner open and those partners while there is one, so a pair on one known runtime goes ahead of a pair with a worker whose runtime is unknown, as under classic. The other wire states no first partner, and a fleet whose every partner would go first states none either.
- **The rank.** A data-parallel worker registered rank by rank (every registration reads the worker's width and registers one worker a rank) is the shape the engine serves in full. Each rank is then a worker object, and a record of its own in its list (`Candidate::dp_rank`, section 5.3). The record reads its own rank's named values (section 5.8). A stage that keeps load reports is told each rank's own report, under that rank (`Signal::Load`). `Candidate::engine` is the same number on every rank of one such worker. The request is sent with the rank of the record the engine picked: where a record is one rank, no rank choice runs behind the pick. A rank is chosen behind the pick in two places, each by classic's own code, with the engine on as under classic, and each inside a worker registered once for all its ranks: on the HTTP pair route, under a connector that holds both sides of a pair to one rank, the route's own turn names the rank both sides are sent to; and under the gateway's `--grouped-dp-sticky hash`, the gRPC dispatch pins a single request that has a sticky key, and where the runtime allows it a pair's prefill side, to that key's rank. That option is refused beside a supplied document.

  The gateway's rank option (`dp_minimum_tokens_scheduler`) is refused at the start beside the engine, under the settings' own document and under a supplied one. A worker object that stands for several ranks states no rank, and outside those two places its request is sent with none. A route that can name no rank to its worker, the multipart transcription, sends with none to the engine of the record picked, a rank's record included: the engine places the request among its ranks, and the record's count holds it.

The change also wrote again three small things that stage 0 took out with the earlier
integration: the gateway's two readers of its own index of cache events for the engine (about
70 lines on main today), the pair classes read from the gateway's pair index (one file of 404
lines), and the cell beside each published list.

The bar for that change:

1. It adds no file of policy logic to the gateway and no branch on a policy's name.
2. No route gets an engine branch. The call sites are the five functions, and one call in the two
   pair routes between a prefill's answer and its decode's dispatch, where the document asks the
   decode again.
3. With the engine off, a classic decision pays what rule 5 names and no more, counted in the
   gateway's decision bench.

   With the engine off, the one call of the second point is still made for a pair sent prefill first: it returns at its first test, of the engine side of the pair's first answer, and reads no pair index. The decision bench's pair route does not make that call, so what a route pays to make it is not counted yet.
4. With the engine on, every classic test of the gateway passes under the settings' own document.

### 9.3 The options

| Option | Meaning |
|---|---|
| `--policy-controller on\|off` | who decides, for the whole process; default `off` |
| `--policy-spec FILE` | run this document in place of the one the routing flags mean |
| `--print-policy-spec` | print the document the routing flags mean, and exit |

They came back with the change (decision 9). No released version has them.

In a gateway whose every router is a third-party provider's, `on` changes nothing: no engine
starts, the classic policies stay, and the start says so in one log line. A document supplied
beside such a mode is refused at the start, by the mode's name.

## 10. The controller **[partly]**

The controller is the binary `smg-policy-controller`. This section fixes the direction. The
switch is made in steps: 10.1 says what is on main.

### 10.1 On main

It starts from a policy document (a watched file, or a folder it keeps). It serves gRPC on three
listeners (gateways, operators, standbys; a fourth, for ingestion, only when asked for) and HTTP
on an admin listener (probes, version, metrics). It elects a leader by a leader lease, held in the
process, in a file, or in a Kubernetes Lease object. That lease is not the lease of section 6.

Its data is one in-memory store (`store/memory.rs`):

| Data | Filled by | Note |
|---|---|---|
| worker records | the `RegisterWorker`, `PatchWorker`, `DeregisterWorker` calls | discovery is in the code behind a feature the image leaves out, and registers nothing yet |
| worker objects | a task that follows the worker records | a second form of the same workers, of the worker crate's type; each states its registration's overload thresholds over one default, 8 queued requests and 0.8 usage, an external record's over none, and the load monitor judges by them while the replica leads; an object's status follows its record's, and a single request's decision reads the workers here |
| health | the worker crate's health checker over the worker objects, on the leader | copied into the records |
| loads | the worker crate's load monitor over the worker objects, every 10 seconds while the replica leads | `PushLoad` is not served; a single request's decision is given the monitor's snapshot where the document reads reports and keeps none; the store's load table is written by nothing |
| cache index | a follower of each worker's cache events | an index of the store's own; `PushKvEvents` is not served; a single request's decision reads none yet, only the engine's own record of sent prompts |
| leases | decisions | not replicated to a standby; a lease that counts for its session key steps that key's count on its worker's object, which a controller's in-flight cap reads (10.2); a single request's session pins are the engine's own table, one for each document |

The gRPC service (`crates/policy_controller/contracts/policy_controller.proto`):

| Call | For | On main |
|---|---|---|
| `Decide` (stream) | a gateway asks for one placement | served; no gateway calls it |
| `Session` (stream) | a gateway reports dispatched, completed, released | served in part: hello, dispatched, completed, released and a gateway's veto. The policy document is never pushed, and a gateway's replay of its leases and pins is refused. No gateway calls it |
| `WatchWorkers` (stream) | a gateway follows the worker list | served by the leader; the contract lets a standby serve it too, and none does yet |
| `RegisterWorker`, `PatchWorker`, `DeregisterWorker` | operators | served |
| `ApplyPolicySpec`, `GetPolicySpec`, `RollbackPolicySpec` | operators | served where the controller keeps a folder; under a watched file an apply answers read-only, and a controller with standbys must use a file |
| `GetSnapshot`, `Explain` | operators | served by the leader; `Explain` only for decisions that were sampled |
| `Leader` | anyone | served by every replica |
| `Replicate` (stream) | a standby follows the leader | served |
| `PushLoad`, `PushKvEvents`, `StepDown`, `WatchRegime` | | not served |

A `Decide` request of the single phase is decided on one of a fixed set of threads (the model
picks the thread) by `PolicyEngine::decide_without_lease`, over the list of the model's workers
that the worker objects' registry publishes. A pair's request at its prefill is decided there
too, over the registry's two lists of its model. Its answer carries a lease a side, or the
prefill side's alone where its frame stated the later decode. A decode call that names a lease
goes to the thread that keeps that lease's file. One that names the
pair's lease is decided behind the prefill that lease holds, by
`PolicyEngine::decide_decode_without_lease`, and its pick is issued under that lease. A prefill
request, or a decode request that names no lease, stands alone where its plan pairs no side of
its phase: it is decided as a single request is, over the registry's list of its own phase's
workers, and its lease is a single's. The token count of a prefill-decode mode (section 3.3) is
not decided this way: it is a single request over the prefill workers, the wire cannot say so,
and a controller reads the single workers' list for it, which such a fleet does not have. An
encode request, and a decode request that names no lease under a pair plan, are answered `Busy`,
and their gateway decides them.

### 10.2 As designed **[partly]**

Rule 2 and rule 3 apply to the controller as they do to the gateway: one set of worker data, in
the worker crate's structures, and the one engine.

| Data | On main | As designed |
|---|---|---|
| workers | records in the store, and worker objects kept in step with them | worker objects in a worker-crate registry; beside each, the code that keeps the objects holds what a registration stated and the object does not hold as stated (10.3); the membership log that watchers and standbys read stays |
| lists | the store's published copy; a single request's decision reads the registry's published list of the model's single workers, a pair's, at its prefill and at its decode call, the lists of its two sides, and a prefill or a decode request that stands alone the list of its own phase's workers | the registry's published lists by role and protocol, with the engine's value beside each (decision 10) |
| health | the worker crate's health checker over the worker objects, on the leader; each outcome is copied into the store's record, an object's status follows its record's, and a decision reads the object's | the same checker; a decision reads the object's own status; a gateway's veto on a worker stays in the store |
| loads | the worker crate's load monitor over the worker objects, while the replica leads: a decision is given its snapshot where the document reads reports and keeps none, and reads the overload verdict on the object; the crate's own poller runs no more, and the store's load table is written by nothing and read by no decision | the worker crate's load monitor, publishing `LoadSnapshot` (decision 10). Every worker is judged by the thresholds its object states, so the loads of every worker the controller dials are read: no new cost against classic as it runs by default, whose gateway hands each worker of an engine the same two defaults and an external worker none at its registration, as a controller's objects state them. A gateway started with `--disable-load-monitoring` polls by need, and this monitor polls every group |
| cache events | an index of the store's own; a decision reads none yet | the kv-index crate's index, filled by the worker crate's collector |
| sessions | the engine's session table for a single request, one for each document | the engine's session table |
| leases | the store's lease tables, in memory, one for each deciding thread | stay: a controller must remember what it issued to gateways |
| decisions | a single request, a pair at its prefill, its decode call, and a prefill or a decode request that stands alone: `PolicyEngine` over the registry's lists; an encode request, and a decode request that names no lease under a pair plan: none, the request is answered `Busy` | `PolicyEngine::decide_without_lease` over the registry's lists |
| discovery | not registering | the discovery crate through its host interface, registering through the worker crate's job queue |

**Who sends a worker's data.** Nothing changes for a worker. It pushes its load records and its
cache events on the stream its host opens (section 4.3), and here the host is the controller. The
controller runs the worker crate's collector and load monitor over its own worker objects, as a
gateway does in classic. A gateway under a controller opens no such stream (section 11). On main
the controller already opens the workers' streams with that collector, for their cache events,
and runs that load monitor while it leads. The monitor is fed by its polls alone there: the load
records that ride the streams are not handed to it yet.

The contract has two calls that would push the same data to a controller, `PushLoad` and
`PushKvEvents`. They are a relay's calls, not a worker's. A relay is a gateway in front of a
worker that only it can reach, and it would forward what that worker's stream brought. Neither
call is served, and this design names no use for them.

What leaves the crate with that switch: the store's copies of the worker data that a worker
object holds, and the adapters that write worker state into the
store. The crate's own load poller has left.
Some facts of a worker are not carried to its worker object today, and some stand there in
a narrower form than a registration states: 10.3 names them and where they are kept.

**The first step, on main.** A single request's decision reads the model's single workers in the
order they registered, where the store's copy was sorted by id. It reads a worker's status, its
breaker and its stalled bit on the worker object, and the count of requests in flight there, which
the store steps for each lease on file. A record made of a worker object states one protocol and
no zone.
The asking gateway's reach, its wires and its standing vetoes are stated to the engine as places
of the list. A new document is a new engine object: its session pins and its record of sent
prompts start empty, and that record is told no worker's leave. A sessions rule's cap reads a
key's requests on the worker object too: the store steps that count for the lease of each keyed
single request on file, with a sessions rule or with none, under the session's number written as
text. The object's count of keys with a request in flight follows the same leases. A pair's lease
counts for no key. A
registration, a drain or a deregistration is on the worker objects when its call answers: a
decision behind the acknowledgment reads the fleet the call left. A controller that starts over
standing rows holds their worker objects before it serves.

A standby's applied records are on its worker objects when its applier goes on to the next event.
A replica that takes the lead holds its last leader's last change in them at once.

### 10.3 The controller's worker data **[partly]**

One worker's data stands several times in a controller, whole or in part. The membership log
tells the data once more:

| Copy | Where | Read by |
|---|---|---|
| the worker record | the store's table of records (`store/memory.rs`) | the registration calls, for a record's status; the load poller, and the follower of cache events for whether a worker's events are indexed, each until it reads the worker objects; a watcher's and a standby's first copy |
| a copy of the record's facts | beside the worker objects (`worker_host/objects.rs`) | the code that keeps the objects, to tell a record that changed under its object |
| the worker object | the worker-crate registry | the health checker, the follower of cache events and a decision |
| the remainder: 17 of the 24 facts and one bit of an eighteenth, with the record's id and revision | beside each worker object (`worker_host/remainder.rs`) | a decision, for the record's id, whether the worker takes images, its wire modes and from where it is reached. A pair's classes read its engine kind and its pairing descriptor too. What the worker states for a request's hard demands is lent with the list, and no frame states a demand yet |
| the published copy of the records | the store | a pick's booking and a gateway's vetoes, among others; no request is decided over it |
| the membership log's event | `service/replication/log.rs` | a watcher behind its first copy; a standby |

A leader's write of a record is decided over the worker object since step 1 below, and the
table follows it. The record still leads a status and a first copy, and the
object follows it there. After the switch the object leads: a worker's data
stands once, as its worker object with its remainder beside it, and the log still tells it. The
remainder is the copy that stays. The table of records and the second copy of the facts end
with step 4 below. The published copy ends when its readers, a pick's booking and a gateway's
vetoes among them, read the worker objects: a later step **[planned]**.

**What stands beside an object [built].** `RowRemainder` (`worker_host/remainder.rs`) holds
what a registration stated and the worker object does not hold as stated. The code that keeps
the objects writes one with every object it builds. `Objects::row_of` gives a worker's
whole record back from the object and its remainder, with no read of the store's table.
A decision reads the remainder's facts, among them the record's id, whether the worker takes
images, its wire modes and from where it is reached. A pair's classes read its engine kind and its
pairing descriptor too. The controller's registration calls decide a write over a whole record
read back (step 1 below). Of the 24 facts a registration states:

- Six stand on the object whole, and the remainder holds none of them: the address, the role,
  the models, the labels, the cache block size and the bootstrap port. So does the status.
- Eight stand twice, in a narrower form on the object and as stated in the remainder; of the
  engine's address the remainder holds one bit, whether a registration named one. The rank
  and the width share a line:

| Fact | On the object | What the object cannot say |
|---|---|---|
| the engine's address | the address the object dials, where that is not the worker's own | whether a registration named the worker's own address as its engine or named none |
| rank and width | a width above one, and a rank unless the worker is its engine whole | a width of one or of none, and a rank stated beside either |
| the wire modes | the one protocol the object speaks | the other protocols of the set |
| the engine's kind | the runtime the worker crate reads the name as | a name that crate does not know, and a known name as it was written |
| the health policy | the checks in whole seconds, off where the controller cannot dial the worker | a part of a second, and the policy of a worker built with its checks off |
| the overload thresholds | where a program asks for them, as the controller's does, an engine's worker's own over the program's defaults and an external worker's own alone | which of them an engine's worker stated |
| the drain window | the window in force | a window stated equal to the object's default, which reads back as none |

- Ten are not carried to the worker object: the incarnation, the zone, the pairing facts, the
  encode offer, vision, the admission window, the priority, from where the worker is reached,
  how its signals arrive and the demand facts. Three of them have a place of the object's own
  that a record does not fill: the object's priority is unsigned where a record's is signed,
  the object derives its pairing facts when it is built, and it reads its admission window
  from a label. The remainder holds the ten, with the record's id and revision, who registered
  it and when.

A fact leaves the remainder when the worker object gets a field that holds it as stated
(decision 10).

**Behind the switch [partly].** Four steps, each small:

1. **[built]** A leader's write of a record (a registration, a patch, a removal) is decided
   over the worker objects, by the code that keeps them, in one turn of its own behind the
   door's. It reads the record back from the object that stands, decides by the registration
   rule, builds the object of the record the change leaves, tells the store the decided
   change, and puts the object in the registry. A removal takes the object out first. So the
   object stands when the write returns. The table of records follows on a leader as it does
   on a standby, by one body, and the membership log is told from it. In a controller the
   code that keeps the objects alone writes a worker object's own facts, so the record a change
   in place writes to the table is the object's record read back whole. A second writer of an
   object's facts in a controller would break that, as a gateway's refresh of a worker's models
   is one in a gateway. The locks in their
   order: the door's turn, the keeper's turn, the store's membership lock, and no lock of the
   store over a write of the registry. The store takes a decided change only where the record
   it was decided over still stands at the revision the keeper read, tested under the
   membership lock: a write that landed between the two is never written over, and the keeper
   reads the store again before it answers. Until step 3 a first copy is still cut from the table,
   and a call reads the status of a record there. Until step 3 a status is still the
   table's word: a drain and a connect signal write the table, the store hands a record the
   status its table holds, and a call follows the store into the objects before it answers. A
   change the store took and the registry did not get is laid by the task that follows the
   table, while that task stands. A change that moves a fact the object holds builds the object again: the
   new object takes the standing one's place under its id, with its running state, as the code
   that keeps the objects does today for a record that changed.
2. A standby applies the leader's events to its own worker objects **[built]**. A call reads the
   record that stands from the worker objects **[built]**. A gateway's watcher hands its
   watch's events to its own worker objects **[built]**. A watcher's and a
   standby's first copy is cut from the objects and their remainders behind step 3, because a
   copy's statuses must be the ones the log last told **[planned]**.
3. No health outcome is copied into a record: a decision reads the object's status, as a single
   request's does on main.
4. The table of records, the second copy of the facts, the two conversions between a record
   and an object, and the task that follows the table are deleted.

**One rule for a registration's write [built].** What a registration makes of the record under
its address is one function beside the other membership rules (`WorkerFacts::laid_over`,
`store/membership.rs`): from the record that stands and the facts stated, the change and the
record it leaves. The store's own write decides by it over the table's record. It then writes
the decided change by the one body a standby's applier writes a leader's change by
(`MemoryStore::decided`). The rule's second caller is the code that keeps the worker
objects, over the record read back from an object (step 1).

**A standby's events through the worker objects [built].** A standby's applier hands each
membership event of its leader's log, and each restore, to the code that keeps the worker
objects (the first part of step 2). For an event with a record that code builds the object,
hands the store the event at its place in the log, and puts the object in the registry where
the store took the event. A removal's object leaves the registry first. A record changed in
place keeps the instant it was admitted at. A status is still the table's word, and it is
copied onto the object in the same turn. A restore is the store's restore and the read of its
records in one turn of that code. A first copy is still cut from the table: a later
part of step 2.

**A call reads the worker objects [built].** A registration call reads the record that stands
before it writes: the record under an address, the record under an id, and an engine's records.
It reads them back from the worker objects (the second part of step 2), in the turn of the code
that keeps them and behind that code's read of the store to its end. So a call's test and the
write behind it read the record from one copy, and a call that waited behind another writer reads what that
writer left. A status is still the table's word, so a record read this way carries the status
the table holds. An engine's records are found by the list of ranks kept beside the objects,
where the store passed over every record under the membership lock: that read of the table is
deleted. A first copy stays on the table until step 3. Its statuses must be the ones the log
last told at its place, and until an object's status leads the table alone holds them under
the lock that orders the log.

**A watcher's events through the worker objects [built].** A gateway's watcher hands each
event of a controller's worker watch, and the watch's first copy at its last page, to the
code that keeps the gateway's worker objects (the third part of step 2), as a standby's
applier does. So the object of a record stands when the event that told it returns, and the
host of the watcher follows nothing behind the watch. A first copy keeps the object of a
record that stands through it. The store's two entries for a watcher, a told change and a
first copy, have no caller in the product but that code.

A lease keeps naming its worker by the record's id, never by a place in a list: a gateway
reports on a lease later and across membership changes.

## 11. A gateway under a controller **[planned]**

One more option, `--policy-controller-address HOST:PORT`, puts a gateway under a controller. Then
the controller finds the workers and decides; the gateway sends the requests.

```text
gateway                                           controller (leader)
   │── WatchWorkers ──────────────────────────────>│  worker list: addresses, roles, up or down
   │<──────────────── snapshot, then each change ──│
   │                                               │
   │── Session: hello ────────────────────────────>│
   │<──────────────── welcome ─────────────────────│
   │<──────────────── the policy document ─────────│
   │── Session: accepted, or refused ─────────────>│
   │                                               │
   │  for each request:                            │
   │── Decide: request facts ─────────────────────>│  PolicyEngine::decide_without_lease over its own lists
   │<──────────────── decision and lease ──────────│
   │── (the gateway sends the request to the worker)
   │── Session: dispatched ───────────────────────>│
   │── Session: completed, or released ───────────>│
```

- The gateway keeps what is its own: the connections to the workers it dials, its circuit
  breakers, its in-flight counts and its rate limits. It runs no discovery, no health checker and
  no load monitor.
- A rate limiter is no part of this design: the gateway keeps its own. Its later home is reserved
  in the schema: a binding's `admission` block, which a document may not switch on today
  (section 14). A limit written there would be decided with the placement. Three things stand to
  build on. A decision can already refuse a request, with a reason and a time to retry. A lease's
  events say when a request starts and ends. A request states its tenant to the engine
  (`PlaceRequest::tenant`); the gateway fills none yet, and no binding may match by tenant yet.
  That block is not the admission window of a pair's prefills, which stays the host's
  (section 8.2).
- About the workers, the controller tells a gateway only their list, on `WatchWorkers`: each
  worker's address, role, models and status, with the facts a decision reads of it. A gateway
  reads those facts in two places: where a route applies its own rule for a worker, and where it
  decides by itself because no answer came in time (both below). It is sent no load report and no
  cache event: that data goes to the controller alone (section 10.2).
- The gateway's side is the same five functions of section 9, made asynchronous: an answer from a
  controller is a round trip, so each route's call changes by that much.
- The request crosses in the contract's own form. The prompt crosses as hashes of its blocks, not
  as token ids or text. A route's own rule for a worker crosses as the ids of the workers the
  gateway excludes. So a controller matches a prompt by whole blocks. For the two trees of
  section 4.5 the contract must hold token ids, or the trees must be keyed by block hashes. The
  controller's own design settles which.
- The gateway tells the controller what became of each request, on `Session`: its dispatch, then
  its completion with the outcome (status, time to first token, tokens in and out, duration), or
  its release with a reason. For a request the controller placed, a completion or a release ends
  its lease, or the side of a pair's lease that it names. The leases on file are the controller's
  count of each worker's requests in flight, which every decision by load reads. A controller's
  stages also learn of the requests they placed from these reports (section 6). A request that
  the gateway placed by itself holds no lease on file, so it enters no count of requests in
  flight: its dispatch names no worker (decision 6). Its completion names the worker that served
  it and still counts toward that worker's health. In process the gateway tells its own engine no
  outcome yet (section 9.2).
- When no answer comes in time (no credit on the stream, a change of leader, the controller
  gone), the gateway decides by itself with the same engine, under the document the controller
  pushed, over the worker list it follows (decision 6). That list stops changing while the
  controller is gone.
- Open: where a gateway under a controller gets a worker's API key. The contract's worker entry
  has no field for one.

On main the controller's half is served (section 10.1): `Decide`, `Session` in part and
`WatchWorkers`. No gateway calls it: the gateway has no option for a controller's address, and
the document is not pushed yet.

## 12. User stages **[partly]**

| Way | On main |
|---|---|
| A stage compiled into the program by its builder | **[built]**: `StageRegistry`, and validation against that table |
| A stage in WebAssembly | the runtime exists as library code under the feature `wasm`, which is on in the crate's default build; no document can name it |
| A stage served over gRPC | the client exists as library code under the feature `controller`; no document can name it |

The two runtimes are 16,039 lines that no document reaches. A `plugins` block in the document and
a loader are **[planned]** and not scheduled (decision 8).

## 13. How it is proven

Rule 6 says classic is the reference. Rule 7 says the proof is inside the crate.

A gateway may run a document that names a builder's own stages: the builder hands its stage table at the start, and the supplied document is checked against that table (section 12). Such a document has no classic twin to be compared with: rule 6 speaks of the document that a set of routing flags means. Its proof is a case of its own (rule 7).

### 13.1 Classic's recorded answers

A *sheet* is a file of recorded lines: for one set of workers and one request, the answer the
classic code gave. A *recorder* is an ignored test beside classic's own code that runs it and
writes the sheet. The sheets are in `crates/policy_controller/tests/classic/sheets/`. A crate
test replays each line through the crate and compares.

On main: 19 sheets, 489 lines.

| | Sheets | Lines |
|---|---|---|
| replayed and passing | 18 | 484 |
| recorded, replayed by nothing (`constraint_decider`) | 1 | 5 |
| replayed through the engine that stays (`PolicyEngine`) | **4**, and 31 lines of `prefix_hash` | **65** |
| replayed lines held to an answer of the crate's own, not classic's | | 7 |

A replay of all fifteen decision sheets through `PolicyEngine` over worker objects is in work and not on main yet: on main the three hard-demand sheets (`retry_wire`, `media_refs`, `context_window`; 26 lines), `sticky_rank` (8 lines) and 31 lines of `prefix_hash` replay through it. Measured
with it locally on 2026-11-22, on main (`096c0f5ea`): of the 408 lines that recorded a decision
then (397 of the 415 of point 1 below, and the 11 of a rank sheet that has left with its rule), the engine answers 286 as classic recorded and 28 differently, and 94 cannot
be asked yet. Of the 94, 25 are declined because the request states a hard demand, 16 need a
word a request cannot state, 52 are encode requests and 1 is refused at validation. Runs with
one input left off the worker objects (the requests in flight, the load reports, the status,
the ring) lose lines, so the replay can fail.

Five things follow, and they are the first work of section 16.

1. **Only the three hard-demand sheets, `sticky_rank` and 31 lines of `prefix_hash` (65 lines) go through the engine that stays** in main's own tests. The other replays
   call the decision function over hand-built candidates. Fifteen sheets
   (415 lines) record a decision, and each has to replay through `PolicyEngine` over worker
   objects. The other four (74 lines: `overload` 25, `pair_index` 23, `minimum_tokens` 18,
   `sticky_rank` 8) record an answer of the host, which the kept engine is never asked for. It
   reads the overload verdict that the host's load monitor wrote on the worker object. It is
   handed each prefill's partners with a pair's two lists. It names the rank a worker
   registered, never one of the ranks behind a worker that stands for all of them. In a host's
   own process those answers are classic's own code, which stays, so the 74 lines hold there
   with no replay. For the two rank sheets that rests on a rule of the gateway's change
   (section 9.2). On main `pair_index` replays through the old engine's side, `sticky_rank` replays over rank objects through `PolicyEngine`, and `overload` (25 lines) replays as recorded through the worker crate's load monitor, built as a host builds it, where the worker is one object for all its ranks. Eleven of its lines answer differently where each rank is a worker object, by the one rule stage 4's line names (section 16). `minimum_tokens` (18 lines) replays as recorded through the rank cache of that load monitor, where the worker is one object for all its ranks or the object of its one rank. No test replays it through the engine's pick among rank objects, which stage 4's line counts (section 16).
2. **23 lines were held to the crate's own answer** (`minimum_tokens` 10, `overload` 6,
   `sticky_rank` 4, `prefix_hash` 2, `manual` 1). The 16 of `minimum_tokens` and `overload` were held by the old engine's store. The 6 of `overload` replay as classic recorded now, through the host's load monitor (point 1). The 10 of `minimum_tokens` replay as classic recorded now too, through that monitor's rank cache (point 1); three of them are rules of the scorer and the picker, which the kept engine shares where it picks among rank objects. The 4 of `sticky_rank` are held the same through `PolicyEngine`, by the ring and the picker. The
   other three are the kept engine's. The 2 of `prefix_hash` differ from classic there too (a
   key folds whole blocks). One of them turns in the crate on a ring its caller lends
   (decision 3). The other measures which tokens are in a prompt's head, which is then the
   caller's recipe: it is proven at stage 3, by a gateway test of classic's own function. The
   1 of `manual` replays as classic recorded now: a sessions rule with no cap runs on the two
   sides of a pair, each side by its own session, where the host decides in process. A served
   controller refuses the rule under a pair: its file of leases keeps one session verdict for
   a pair's two sides. Each
   of the 23 is made equal to classic or named as an accepted difference: two in stage 1, one
   at stage 3, the 20 in stage 4.
3. **Five policies have no sheet over a healthy pool**: `round_robin`, `random`, `passthrough`,
   `least_load` and `consistent_hashing`. Their only recorded lines are over pools in trouble.
   New recorders are decision 2.
4. **Two limits on "as classic recorded".** A pair's refusal names no side in the engine's
   answer, so the pair sheet's refusals are compared by their reason alone, where classic names
   the side. A change of stage 1 makes a pair's refusal name its side. The 12 ring lines of
   `all_vetoed` name their workers by the key's order on the engine's own ring: they prove the
   walk past a vetoed worker, not that a key lands on classic's worker (decision 3). With a ring
   its caller lends, the 12 answer as recorded over hand-built candidates
   (`the_classic_ring_rows_replay_on_a_ring_their_caller_lends`). That the lent ring is classic's
   own is proven at stage 3.
5. **Decision 3 (a) parts the `prefix_hash` sheet in two.** 31 of its 47 lines say where their
   requests stand on a ring: 30 list their workers in a request's ring order and one has
   nothing to hash. They replay on a ring their caller lends and hold their recorded answers,
   the held ten-token line among them. That replay calls the decision function over hand-built
   candidates. Through `PolicyEngine::place` over worker objects, with the ring lent and no place
   stated, the 31 hold their recorded answers too: the engine cuts each request into its heads
   and the line's ring answers for each
   (`the_classic_prefix_hash_rows_replay_through_the_one_call_with_no_place_stated`). The other
   16 measure which tokens are in a prompt's key (15 lines) or how
   2,000 prompts spread over eight workers (1 line); the second held line is one of the 15.
   With a lent ring a request's place is the host's to state, so these 16 test the host's
   recipe. In the crate they stay on the crate's own ring with its own hashes, which a caller
   that lends no ring still runs: 15 hold classic's shares, as on main, and one is held apart.
   A gateway never runs that ring. Classic's key recipe is inside its policy's own selection
   today, and its ring lookup takes a key and not a position. So at stage 3 the gateway's
   change cuts both into functions that classic and the lender call, and a gateway test puts
   all 47 lines through them, classic's ring and the engine, count for count.

Hosted CI's first test line of the crate enables the feature `client`. It takes `controller`,
which takes the feature the kept engine is behind, so that line runs that engine's own tests.
Decision 11 says how this was closed.

### 13.2 Speed

Classic's cost is counted in the gateway's decision bench: instructions for one decision, for
each classic policy at 8, 64 and 512 workers. That count is the mark the engine has to meet.

The crate gets one bench of the engine over worker objects **[planned]**. Every change to a
policy shows its cell before and after, with classic's count beside it.

### 13.3 On a cluster

The engine has run beside classic on a cluster twice: 72 engines, 1,024 clients, single requests
in conversations of several turns. "Same worker" is the share of a conversation's next turns that
went to the worker of the turn before.

**Through the gateway's option, 2026-11-23.** One gateway ran with `--policy-controller` off and
with it on (section 9.3), over the same fleet. With the option on, the engine ran the document
that the routing flags mean. For `cache_aware` on the earlier head the flags themselves made it.
For the other rows it was supplied as a file: the document the earlier head's gateway prints for
those flags, with its profile renamed. Two builds of the change that brought the engine back were
run, both before it merged: an earlier head, and the head that merged as `4da61bd83`. A run
measures 480 seconds behind a warm-up of 120, with no bursts. No request failed. A cell is the
mean of its side's runs.

| Policy | Build | Runs a side | Who decided | Requests a second | First token p50 / p95 / p99 ms | Same worker |
|---|---|---|---|---|---|---|
| `power_of_two` | earlier head | 3 | classic | 302.15 | 397.02 / 661.99 / 803.00 | 0.0157 |
| `power_of_two` | earlier head | 3 | engine | 302.19 | 397.66 / 662.58 / 802.89 | 0.0157 |
| `least_load` | earlier head | 2 | classic | 301.80 | 363.17 / 573.02 / 678.75 | 0.0160 |
| `least_load` | earlier head | 2 | engine | 302.46 | 364.60 / 577.24 / 688.19 | 0.0163 |
| `cache_aware` | earlier head | 2 | classic | 420.03 | 207.15 / 420.97 / 522.69 | 0.9963 |
| `cache_aware` | earlier head | 2 | engine | 419.55 | 207.93 / 424.22 / 528.68 | 0.9964 |
| `cache_aware` | merged head | 2 | classic | 420.28 | 207.18 / 423.07 / 525.23 | 0.9965 |
| `cache_aware` | merged head | 2 | engine | 419.43 | 208.10 / 422.53 / 527.16 | 0.9965 |

A difference is told apart here when it is larger than each side's spread: the largest of a
side's runs less its smallest. The spreads stand in parentheses, classic's first. With two runs a
side, told apart means seen twice, not proven.

- `power_of_two`: not told apart on any measure of the table.
- `cache_aware` on the merged head: not told apart on any measure. The engine served 0.85
  requests a second fewer, inside classic's own spread (1.44 and 0.42).
- `cache_aware` on the earlier head: the engine served 0.48 requests a second fewer (0.11 %;
  0.38 and 0.10), and its p99 was 6.0 ms later (1.15 %; 2.67 and 5.26). Its p50 and p95 are not
  told apart.
- `least_load`: the first token is later with the engine, by 1.4 ms at p50 (0.39 %; 0.84 and
  0.24), 4.2 ms at p95 (0.74 %; 1.44 and 0.58) and 9.4 ms at p99 (1.39 %; 5.47 and 1.13). The
  requests a second are not told apart.

One measured run of the earlier head is left out of the table. Under a supplied `cache_aware`
document that head applied no cache event: 397.97 requests a second, same worker 0.8942. The
merged head applies them, as its rows show.

Not run through the option: a pair, an encode request, bursts, any other policy, a controller in
its own process, or any commit of main later than that merge.

**Through the earlier integration, 2026-11-22.** One run for each line, through the gateway
integration that stage 0 took out. It showed the faults of the engine's own record of sent
prefixes (section 4.5), under a document whose prefix stage reads that record alone. The two-tree
lines ran on a build 32 commits later than the others, so their speed is not proven to be the
record's alone.

| Load | Who decided | First token p50 / p95 ms | Same worker |
|---|---|---|---|
| steady | classic, by cache events | 207.95 / 423.78 | 0.9960 |
| steady | engine, by its own record | 240.55 / 532.34 | 0.8944 |
| steady | engine, record as two trees (a closed draft) | 206.41 / 420.10 | 0.9979 |
| bursts | classic, by cache events | 480.34 / 848.99 | 0.9982 |
| bursts | engine, by its own record | 834.53 / 1497.56 | 0.5150 |
| bursts | engine, record as two trees (a closed draft) | 482.12 / 853.12 | 0.9988 |
| steady, the record beside cache events (a 90-second look) | engine, by its own record | 359.44 / 578.56 | 0.0647 |
| steady, the record beside cache events | engine, record as two trees (a closed draft) | 208.81 / 424.85 | 0.9970 |

In the steady run with the old record, 12,050 of 12,936 next turns moved where the earlier prompt
was 14,000 tokens or more, and 1,881 of 139,710 where the later prompt was under 13,000. Classic
ran with cache events in every line; its own token tree was not run on the cluster.

The two-tree record costs memory: the gateway held 380 to 839 MiB in the steady run with it,
against 282 to 476 MiB with the old record, and 283 to 707 MiB against 250 to 425 MiB under
bursts. It is bounded by the tree's size bound and the time between two eviction passes.

## 14. Status

| Part | Mark | What is missing |
|---|---|---|
| The policy document: profiles, bindings by model and phase, plans, a binding's shed word and its refusal of an overload number, routing key, sessions | **[built]** | |
| A binding's overload switch (`overload.enabled`) under worker objects | **[partly]** | a gateway that runs a supplied document reads it at its start, where it builds its workers' default: one word for the bindings a host runs (decision 10); a controller does not read it there yet |
| Bindings by tenant, priority, header, prompt size, labels; a session id issued by the engine (`issue_session_id`) | **[planned]** | in the schema; a document that writes one is refused as not in this build |
| `canary`, `override: allowed`, `admission` | **[planned]** | in the schema; a document that writes one is refused, and the error names the field |
| `shadow`, `explain.sample` | **[partly]** | the kept engine runs a single request's shadow, but not one that reads affinity rows of its own; a pair's shadows run in no engine, and `explain.sample` is kept by the served controller alone (decision 7) |
| The ten presets and the conversion from routing settings | **[built]** | |
| A worker's `policy` label names its model's policy | **[planned]** | section 3.3: a warning at a registration first, then a binding for the model in the document built from the routing settings; a document put in force keeps the stages of a profile that did not change |
| The stage interfaces and a builder's own stages | **[built]** | |
| The engine over worker lists (`PolicyEngine`) | **[partly]** | refuses a subset hint, an encode request's demand, mixed protocols; of the recorded lines only the three hard-demand sheets and `sticky_rank` (34), and 31 lines of `prefix_hash`, replay through it |
| `round_robin`, `random`, `passthrough`, `least_load`, `power_of_two` on that engine | **[partly]** | no sheet over a healthy pool for four of them |
| `cache_aware` on that engine | **[partly]** | a single request whose maker names its entry by token ids (`RequestMaker::token_key`) is decided in the host's token tree, where the host lends it, the model has no index of cache events and the profile reads sent prefixes and names the holders' rows, as the preset does: the record's first two faults do not reach it. Every other request the engine names the holders of reads the engine's own record, with its three faults: a request whose caller names no entry is one. The string tree is planned |
| `consistent_hashing`, `prefix_hash` on that engine | **[partly]** | placed on the ring the host lends (decision 3): through `place` the plain ring at the place the engine reads with the request's keys, and `prefix_hash` at the places the host's ring answers for the request's heads, where it answers for a head; that both are classic's is proven at stage 3; a host that lends none is placed on the crate's ring, which may be another worker than classic's |
| `manual` and sticky sessions on that engine | **[partly]** | their sheets replay through the decision function, not through the engine |
| `bucket` on that engine | **[partly]** | the same |
| Prefill-decode pairs on that engine | **[partly]** | the sheet of classic's pair choice (`pair_select`) replays through the pair's decision function, not through the engine; one of its 40 lines is held to an answer of the crate's own; the 8 lines behind the prefill admission's bound (`pair_bound`) and the 10 lines of a pair on one known runtime (`pair_runtime`) replay through that function as recorded |
| Encode-prefill-decode on that engine | **[partly]** | `PolicyEngine::decide_encode_at` decides it over a host's three lists and issues the pair's lease, and `place` places it; the entry declines a request held to a wire that a worker of its pair's two lists is not dialed on, places its items on the encoders dialed on a wire it states, declines a request with items where the list holds encoders and none is, and declines one that states the later decode where its decode is held to the workers that take images: a decode asked again takes no items |
| The lease and its events; the guard | **[built]** | |
| The host contract of section 8.2 (`place`, `Placement`) | **[partly]** | `Placement` is built, with its call for a pair's decode decided again, and `place` for a single request, for a pair and for an encode request, with the route's rule and the decision made again, three decisions a call at most or a single request's fourth where the third found the workers the route's rule takes, and a single request's later decisions held to those workers; the later decisions of a pair and of an encode request held to them are not |
| The crate's bench of the kept engine | **[planned]** | |
| The gateway as a host | **[partly]** | section 9, behind `--policy-controller on`, which is off by default: single requests and pairs on the HTTP and gRPC routes, a pair's decode asked again, a host's table of stages at the start; an encode request that carries an item is refused by name until the gateway hands the engine its three lists; every classic test of the gateway with the engine on (the bar's fourth point) is not met yet |
| The controller: service, election, replication, document apply and roll back | **[partly]** | a session never pushes the document and refuses a gateway's replay; a standby serves neither the worker list nor the snapshot (section 10.1) |
| The controller on the one engine and the worker crate's structures | **[partly]** | a single request, a pair at its prefill and its decode call are decided by `PolicyEngine` over the registry's lists; an encode request is answered `Busy`; a decision reads no cache event, and no load report under a document that holds a stage that keeps reports (section 10.2) |
| The controller finding its own workers | **[partly]** | discovery registers nothing yet |
| A gateway under a controller | **[planned]** | section 11 |
| User stages in WebAssembly and over gRPC | **[partly]** | library code that no document reaches |

**When it is ready.** Three tests, all of them: requests are decided over the real data (the
worker crate's registry, loads and cache records, not a copy); the speed is classic's, by count;
and it has served traffic on a cluster beside classic.

## 15. Decisions

Each is the maintainer's. Work inside the crate goes on where a decision is not needed for it.

All eleven at a glance. An answer can be as short as the numbers with their letters.

| # | Decision | Recommended | Needed for |
|---|---|---|---|
| 1 | The name of a cache partition | (a) the host hands the entry already named | the host's two trees, read for `cache_aware` (the token tree's entry is built) |
| 2 | Recorders for the sheets that are missing | (a) new recorders beside classic's code, for the ring, the sticky override and `least_load`; (b) a crate test that names classic's rule, for `round_robin`, `random` and `passthrough` | proof against classic for those policies |
| 3 | The hash ring | (a) the host lends its ring now; (b) one shared ring with the controller's switch | `consistent_hashing` and `prefix_hash` |
| 4 | Where a placement stays during a request | (a) inside the load guard | the gateway's integration (stage 3) |
| 5 | How the controller comes onto the one engine | (a) whole, in one switch | stage 4 |
| 6 | No answer in time under a controller | (b) the gateway decides by itself with the same engine | stage 5 |
| 7 | Shadow and explain | (a) built on the kept engine before the old one is deleted | stage 4 |
| 8 | User stages in WebAssembly and over gRPC | (a) left as library code | nothing now |
| 9 | The gateway's options | three with stage 3, the address and TLS options with stage 5 | stages 3 and 5 |
| 10 | What the controller needs in the worker crate | one list type, and a load monitor that can be stopped; the other worker facts are kept beside the object; recommended: one set of overload thresholds for each worker | stage 4 |
| 11 | The crate in CI and in the published set | (b) one line in CI's test script names the engine's feature; (a) out of the published set until stage 3 | CI's run of the kept engine's tests; stage 0 |

### Decision 1: the name of a cache partition

*Needed for:* the host's two trees, which the engine reads for `cache_aware` (section 4.5). A
request's entry in a tree must be named as classic names it.

| Option | Cost |
|---|---|
| (a) The host hands the entry already named, with the length of the partition's marker at its start. The engine takes it as it takes the prompt's hash seeds today. | About 70 lines in the crate. Nothing outside it now; about 15 lines (an estimate) in the gateway's one integration, calling classic's own two naming functions. The naming is then the host's to keep: a host that hands a plain entry for a partitioned request lets two tenants' prompts match. |
| (b) Classic's naming (336 lines) and its two functions move into the kv-index crate. | A move across 12 files, 10 of them in the gateway. No behaviour changes. |

Under (a) the engine also states for itself which of the three records a request reads (the rule
of section 4.5), about 12 lines. Classic has that rule inline, so there is no function to call.
Classic's `cache_aware` sheet has lines for all three records (12, 13 and 3) and holds the
engine's rule equal to classic's. The other way is one function in the kv-index crate that
classic calls too: about 90 lines, and it touches the gateway.

**Recommendation: (a).** Nothing is copied and nothing moves. Work has started on it, inside the
crate only.

### Decision 2: recorders for the sheets that are missing

*Needed for:* proof against classic for `round_robin`, `least_load`, the hash ring, and the sticky
override under five more policies.

| Option | Cost |
|---|---|
| (a) New recorders are allowed as ignored tests beside classic's code. | Files under `model_gateway/`, tests only; nothing the gateway runs changes. One small change for each recorder. |
| (b) No new recorder: a crate test states classic's rule and names classic's line. | Nothing outside the crate. Weak for a hash, a table of sessions or a sum of loads. |

**Recommendation: (a)** for the ring, the sticky override and `least_load`; (b) for `round_robin`,
`random` and `passthrough`.

### Decision 3: the hash ring

*The fact:* the crate's ring and classic's ring use different hashes. In a local comparison 879
of 1,000 routing keys landed on another worker at 8 workers, 1,973 of 2,000 at 64 and 1,996 of
2,000 at 512. A session would move when a gateway changes to the engine, and two gateways in
different modes would disagree.

| Option | Cost |
|---|---|
| (a) The host lends the ring it keeps; the engine asks it which workers a routing key reaches. | Crate only: one deletion and four changes. The crate's ring beside a worker list is deleted (1,728 lines removed, 230 added, 17 files). The four changes are about 580, 270, 450 and 550 changed lines, about 370 of them product code and 1,050 proof. The host lends its ring through one trait (`LentRing`) and states where a request stands on it. The crate then writes no hash and no ring for such a host, so a request is placed on classic's own ring. The price in proof: 16 of the 47 recorded `prefix_hash` lines test which tokens are in a key. In the crate they keep replaying on the crate's own ring, share for share as on main. A gateway then never runs that ring, so for a gateway they are proven at stage 3 (section 13.1), where the change also cuts classic's key recipe and classic's ring lookup into functions that classic and the lender both call. A controller has no classic ring to be lent: until (b) it keeps its own ring and places a routing key on another worker than a gateway would. |
| (b) Classic's ring (one file, 508 lines) moves to a crate both sides use. | A pure move, about 15 lines changed beside it. The gateway, the engine and the controller then build one ring type. |
| (c) The crate's ring is changed to classic's hashes. | About 125 lines. Classic's recipe written twice; a gateway would keep two rings for a model. |

**Recommendation: (a) now, (b) with the controller's switch.** Not (c). Work on (a) has started,
inside the crate only.

### Decision 4: where a placement stays during a request

*Needed for:* the gateway's integration (section 9).

| Option | Cost |
|---|---|
| (a) Inside the load guard: the worker crate's `WorkerLoadGuard` gets one optional field. | One small addition to the worker crate. No route gets a new value to keep: every route already keeps the guard to the end of the response. |
| (b) Each route keeps the placement beside its guard. | No change to the worker crate. Every route changes, which is what made the earlier integration wide. |

**Recommendation: (a).** Taken with the integration, not before.

### Decision 5: how the controller comes onto the one engine

| Option | Cost |
|---|---|
| (a) Whole: the kept engine is finished and proven in the crate, the controller's parts are added beside the old engine, one small change switches the deciding threads, one deletion follows. | Nothing of the new path runs in a controller until the switch. The types both engines share move out of the old engine's file first (section 5.7). |
| (b) Document by document, behind a test at apply. | A test at apply, two load pollers for a time, a temporary call, and refactors inside the old engine that are deleted afterwards. |

**Recommendation: (a).**

**As taken.** The deciding threads were switched for a single request first (section 10.1), and
the store-backed engine's type was deleted behind that step. A pair at its prefill and its decode
call are decided by the one engine over worker lists too. An encode request is answered `Busy`
until it follows.

### Decision 6: a gateway under a controller that gets no answer in time

| Option | Cost |
|---|---|
| (a) Refuse the request (503). | The simplest. Every blink of the controller is an outage of the gateways. |
| (b) The gateway decides by itself with the same engine, under the document the controller pushed, over the worker list it follows. | The gateway has the engine from section 9. The document comes by the controller's push, which the contract has and the controller does not send yet. Gateways and controller must be built with the same stages. The gateway decides without load reports and cache events, over a worker list that no longer changes: a worker that dies meanwhile is seen by the gateway's circuit breaker only. The controller's in-flight counts miss a request that a gateway placed by itself: its dispatch message names no worker. |
| (c) A fixed simple policy, such as round robin. | Small. The fallback behaves unlike the policy in force. |

**Recommendation: (b).** The contract was written for it: it has a push of the document "for
every fallback engine".

### Decision 7: shadow and explain

The kept engine runs a single request's shadow, and counts as skipped one that reads affinity
rows of its own. A pair's shadows run in no engine, and the explain sample is kept by the
served controller alone. A document
that writes `canary`, `override: allowed` or `admission` is refused. No engine runs them.

| Option | Cost |
|---|---|
| (a) Build shadow and explain on the kept engine before the old engine is deleted. | Work inside the crate, after the policies are proven. Both features survive the switch. |
| (b) Delete them with the old engine and refuse them in a document until rebuilt. | The switch is sooner. Two documented features are gone for a time. |

**Recommendation: (a).** They are how an operator tries a policy safely. They wait behind the
proof of the ten policies.

### Decision 8: user stages in WebAssembly and over gRPC

16,039 lines that no document reaches.

| Option | Cost |
|---|---|
| (a) Leave them as library code until the engine and both hosts are done. | Nothing new. The WebAssembly runtime stays in the crate's default build, and both runtimes' tests stay in its gate. |
| (b) Add the `plugins` block and a loader now. | New work beside the essential work. |
| (c) Remove them until they are needed. | A large deletion; a rebuild later. |

**Recommendation: (a).**

### Decision 9: the gateway's options

A new option needs the maintainer's yes. The integration of section 9 brings three:
`--policy-controller on|off`, `--policy-spec FILE`, `--print-policy-spec`. A gateway under a
controller (section 11) brings `--policy-controller-address` and three TLS options.

**Recommendation:** the first three with section 9's change, the others with section 11's.

### Decision 10: what the controller needs in the worker crate

*Needed for:* section 10.2. Not needed before the controller's switch.

| Need | Option recommended | Cost |
|---|---|---|
| A model's lists by role and protocol, with the engine's value beside each | one type in the worker crate that the gateway and the controller both use | 279 lines added; the gateway moves onto it at its integration |
| A load monitor that a controller can stop when it stops leading, and start again | one public `stop()` on the worker crate's monitor | about 110 lines with its test |
| The facts of a worker that a decision reads and a worker object does not hold today (zone, the encode offer, vision, the admission window, the demand facts, pairing) | the controller keeps them beside the object (10.3, built); a fact moves to a field of the worker crate's object when that crate has one that holds it as stated, checked against the crate first | 245 lines of code for what is kept beside the object |

One more point under this decision. A document can write an `overload` block for each binding, and
a worker may state thresholds of its own. A worker object holds one set. **Recommendation:** one
set for each worker, the worker's own values over one default of the program, as the classic
policies have it. What that leaves of a binding's block:

- **Its two numbers [built].** A block names none. A document that names one is refused when it
  is applied, by the number's path, and the refusal says where a number is stated: a worker's
  registration, or its program's default. The two that a gateway's routing settings state are
  the defaults its worker objects are built with, and the crate's translation of those settings
  writes neither onto a binding. A gateway states them on its command line, beside a supplied
  document too.
- **Its shed word [built].** A binding keeps whether a request is shed when every worker is
  vetoed. A decision reads it.
- **Its switch [partly].** Whether the program's default vetoes at all is one word for the
  bindings a host runs, because a worker object has one default. The host reads it where it
  builds that default. A gateway that runs a supplied document reads it at its start, and
  refuses a document whose bindings disagree on it, or on the shed word: its routes read one
  shed word for the process. It also refuses one whose block is on where the gateway states
  neither number. With the switch off the gateway builds its worker objects with no default. A
  binding with no block says the defaults. A controller's reader of the switch is planned: it
  does not read it on its worker objects yet. A worker's own thresholds veto either way.

### Decision 11: the crate in CI and in the published set

| Question | Fact | Options | Recommendation |
|---|---|---|---|
| How does CI run the kept engine's own tests? | The kept engine compiles only under the feature `in-process`. The feature `controller` takes it. The first of CI's four test lines of the crate in `scripts/ci_rust_tests.sh` enables `client`, which takes `controller`, so that line runs the cases behind `in-process` and the test target `caller`. | (a) The crate's default build includes the feature: one line in the crate's manifest. Measured: two of CI's four lines then run the 82 cases, in under a second more. But a default build of the crate goes from 147 to 272 packages, the controller's image and three bench workflows then build the engine in, and two READMEs become wrong, one of them outside the crate. (b) CI's first test line names the feature: one line in the script, outside the crate. That line builds the same 364 packages with it and without it, and nothing else builds differently. (c) The crate's test-only dependency on itself names the feature: inside the crate, and no product build changes. But every build that takes the crate's test-only dependencies then builds the engine too: CI's two lines that test one feature alone, the lint of `operator` alone, and the three bench workflows that (a) names. Not measured. | Closed a fourth way, inside the crate: `controller` takes `in-process` and `worker-host`, the two features the controller's switch (section 10.2) builds on. A build with `controller` or `client` then compiles the kept engine and the worker objects, and a controller built with `controller` alone keeps worker objects as the stock binary does. No line outside the crate changed. |
| Is the crate published before it is ready? | The earlier integration added the crate to the published set: four lines in two release workflows and the release version check. | (a) Stage 0's change takes the four lines out, and they return with the gateway's integration (stage 3). (b) They stay. | (a). |

## 16. The order of work

The crate first (rule 7). Each stage ends with a test, and the next does not start before it.

| Stage | Work | Ends when |
|---|---|---|
| 0 | The earlier gateway integration is taken out. | The gateway routes by classic alone; every classic test and bench cell is as before. |
| 1 | **The policies, proven in the crate.** Each classic policy decided by `PolicyEngine` over worker objects: its sheet replays through that engine, and its bench cell meets classic's count. | Every line of the 15 sheets that record a decision (415 of section 13.1's 489) replays through the kept engine over worker objects. Each difference from classic is fixed or accepted as a named difference. Under decision 3 (a) the 16 `prefix_hash` lines that test a key's tokens replay there with no ring lent, and their proof for a gateway is stage 3's. |
| 2 | **The host contract.** `place`, `Placement`, the engine's own upkeep and membership events (section 8.2). | A test host in the crate serves every plan through `place` alone. |
| 3 | **The gateway, one change** (section 9), off by default. | The four points of section 9.2's bar; under decision 3 (a), the 47 `prefix_hash` lines and the 12 ring lines of `all_vetoed` read as recorded, count for count, through the gateway's own entry with the engine on and the recorder's own workers and prompts; a cluster run beside classic. |
| 4 | **The controller on the one engine** (section 10.2); the old engine is deleted. | The controller's own tests pass on `PolicyEngine`; the store-backed engine is gone; each of the 74 lines of the four host sheets (section 13.1) is answered by the host's code that the controller takes from the worker crate, or by a rule of the crate's with its differences named. One difference is named so far, and 11 `overload` lines answer by it: where ranks are worker objects, a threshold that judged an engine's summed queue judges each rank's own queue, and a usage threshold each rank's own usage in place of the mean (two ranks that queue 4 requests each under a threshold of 8 are vetoed as one worker and not as two). The 18 `minimum_tokens` lines are answered as recorded by the worker crate's rank cache behind that monitor, which no code of a controller asks. They are not answers of the engine: it serves no rank option (`refuse_unsupported` in `src/document/routing_settings.rs` refuses classic's at the start), and where ranks are worker objects it picks among them by its profile. That pick was measured once, on 2026-11-23 over `55c35100d`, outside the tests: under a resident-work profile with a picker of two it names the recorded ranks on 5 of the 18 lines and other ranks on 13, by four rules. On 6 lines classic finds no rank on a worker its cache holds nothing of, and the engine places on a ready object that has no report (`with_no_survivor_reporting_resident_tokens_the_in_flight_count_decides`). On 3 a rank that the answer omits or reports with no token capacity is no rank to classic, and the engine scores it from its best reporting peer (`a_worker_without_resident_tokens_scores_from_the_best_reporting_peer`). On 3 the engine reads resident tokens and draws among equal ranks, where classic reads used tokens and takes the lowest rank (`equal_resident_work_ties_and_a_token_apart_does_not`). On 1 classic keeps a worker's last answer and its bookings behind a failed poll, and the monitor publishes no report of that worker (`publish_polled` in the worker crate's load monitor). Three of the lines move with how the caller counts a pick in flight and with the picker, and the count stays at 5 or 6. |
| 5 | **A gateway under a controller** (section 11). | A cluster run: gateways under a controller beside classic, with the controller stopped and started under load. |

### Stage 1, the first changes

Each is one small change inside the crate, with its proof. Change 2 is a series, each behind
the one before.

| # | Change | Proof inside the crate |
|---|---|---|
| 0 | Two doc links that break the docs build with default features | the docs build, before and after |
| 1 | The bench of the engine over worker objects, with `passthrough` decided by one pick | the bench's cells before and after |
| 2 | `cache_aware` reads and writes the host's two trees: the host lends its token tree and hands the entry, and a tokenized request is decided inside that tree's walk; a text request; the choice between the three records; the old record leaves | crate cases that fail before; the bench's cell; classic's `cache_aware` sheet through the engine once change 3's replay over worker objects is on main |
| 3 | A pinned session is followed first, as classic does | two crate cases that fail before; the bench's cell; the 18 recorded sticky lines through the engine over worker objects, which prove the reads from the worker object |
| 4 | A request can be held to some workers of its list | its cases |
| 5 | A decode decided again keeps its first decode | a failing case at the service's entry |
| 6 | A list's records take vision and demands from the host | the context-window sheet through records made from worker objects |
| 7 | One type owns a rank's dispatches (no behaviour changes) | the crate's tests; the bench's cell equal |
| 8 | The encode facts of a host's lists, written by one function | its case |
| 9 | A pair's side wholly over its thresholds follows its binding's rule | classic's pair sheet |
| 10 | A sampling pick steps aside for a request held to some workers | its cases |
| 11 | A row the keyed decision reads by its place honours a request held to some workers | its cases |
| 12 | A sampling pick converts a list's load reports once | the bench's cell before and after |
| 13 | A request with no routing key under a sessions rule is decided by one pick | a case that counts the records read, failing before; the bench's cell; the `manual` and sticky lines unmoved |
| 14 | A pair is decided with the host's load reports, as a single request is | a case that fails before; the bench's pair cell |

One more change ran beside them: the types both engines share moved out of the old engine's file
and the store's lease file into the kept engine's modules, with no behaviour change (section 5.7).

Behind them, in the crate: `random` and `round_robin` details of the short path; pairs and encode
requests over worker lists; the hash ring a host lends (decision 3). Waiting for a decision:
new recorders (2), the controller's parts (5, 10), shadow and explain on the
kept engine (7), the published set (11).
