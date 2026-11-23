# The policy controller: design

How the policy controller is built: its parts, the data it reads and keeps, and what happens to a
request. It is written for a reader who knows the gateway and has not followed the work. The
README beside this folder is the user's guide (how to write a policy document, how to run the
service). This file is the design.

It describes the crate as it stands on main (read at commit `882be6f4f`, 2026-11-22) and what is
planned. One thing on that commit is on its way out: the gateway still calls the engine through
an earlier integration, which a pull request of its own removes first (stage 0 of section 16).
Where the text speaks of the gateway, it speaks of main with that integration taken out. Every
part has one of three marks:

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
| policy document | A YAML or JSON file that says how workers are chosen. Section 3. |
| profile | One way to choose: filters, then scorers, then one picker. |
| binding | A rule of the document: which requests use which profile. |
| engine | The code that takes a document, a request and a worker list and answers with a decision. |
| host | A program that embeds the engine: the gateway, or the controller. |
| controller | The standalone service built from this crate. Gateways can ask it over gRPC. |
| decision | The engine's answer: a pick (up to four workers, best first) or a refusal with a reason. |
| lease | The engine's note that a decision's request is on its way. It ends when the request ends. |
| prefill, decode, encode | The roles of workers that serve one request in stages. |

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

On main the gateway still calls the engine through an earlier integration. That integration is
taken out first (stage 0 of section 16), so that the crate can be finished before the gateway
depends on it. Section 9 says how the gateway gets the engine back, as one small change.

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
   selection decides; a request pays the test of the option where it is placed and what the five
   placement functions and the load guard cost for carrying an answer of either side, counted for
   each policy in the gateway's decision bench and shown on the page of each change to those
   functions (section 13.2).
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

Lines of each folder under `src/`, tests included, counted on main.

| Folder | Lines | Files | What it holds |
|---|---|---|---|
| `document/` | 20,553 | 15 | The policy document: its types, validation, the presets of the ten policy names, the conversion from routing settings, apply and roll back. |
| `engine/` | 56,487 | 33 | Both engines, the profile runner (`framework.rs`, 19,125 lines), the request and worker types. |
| `policies/` | 12,512 | 12 | One file for each built-in policy. |
| `stages/` | 10,200 | 8 | The built-in filters, scorers and pickers and their table. |
| `tracking/` | 7,506 | 9 | The lease, its guard, the dispatch credit, the old record of sent prefixes. |
| `store/` | 44,236 | 38 | The controller's in-memory store: worker records, loads, cache index, sessions, leases. |
| `service/` | 79,311 | 49 | The controller: the gRPC service, election, replication, discovery. |
| `worker_host/` | 17,656 | 12 | What lets a controller hold worker objects of the worker crate. |
| `load/` | 6,313 | 10 | The controller's own load poller. |
| `plugins/` | 16,039 | 21 | Runtimes for user stages in WebAssembly and over gRPC. |
| other | 432 | 4 | `lib.rs`, the binary's `main`, two small files. |
| **Total** | **271,245** | **211** | |

The engine that stays (section 5.7) is `engine/policy_engine.rs` and the folder beside it: 6,228
lines with their tests, and 1,818 more for the record writer and its tests.

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
    pub profiles: BTreeMap<String, ProfileSpec>, // the profiles (the one required field)
    pub plans: PlansSpec,                        // rules of the prefill-decode plan
    pub bindings: Vec<BindingSpec>,              // in order; the first match wins; an empty
                                                 // list binds the first profile to everything
    pub explain: ExplainSpec,                    // how many decisions are kept to be explained
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
    pub overload: Option<OverloadSpec>,      // the overload thresholds of the workers decided here
    pub routing_key: Option<RoutingKeySpec>, // where a request's routing key is read from
    pub sessions: Option<SessionsSpec>,      // sticky sessions
    pub shadow: Option<ShadowSpec>,          // a second profile run for comparison only
    // canary, override and admission are in the schema and are refused: section 14
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

A user who never writes a document still gets one. The crate turns the routing settings (the
policy name and its options, the mode, the overload options, the sticky-session options) into the
document they mean:

```text
routing settings ──> RoutingSettings::policy_document() ──> PolicySpec ──> validate ──> AppliedSpec
                     (document/routing_settings.rs, a pure function)
```

Each of the ten policy names has a preset. For single workers it is one profile bound to every
model. A prefill-decode mode gets two profiles under one pair binding. A preset also writes what
its binding needs: a routing-key policy its key sources, `manual` its sessions rule. The presets
are in `policies/<name>.rs`. Section 7 lists them.

A user may instead supply a document. It is accepted when the routing settings say the same
document or say nothing; otherwise the start is refused and the error names the first
difference.

### 3.4 A validated document

Validation turns a `PolicySpec` into an `AppliedSpec`: the built profiles and bindings with their
stage objects, and what is resolved once from the document (block size, hash depth, each
binding's routing-key sources, the signals). What a stage remembers lives in its stage object
inside the `AppliedSpec`, so it belongs to one generation of the document. Two things live beside
it. What belongs to one worker list (round robin's counter) is in the
engine's value for that list (section 5.2). The record of sent prefixes is in the engine object
(section 5.1).

Applying, listing and rolling back generations is `DocumentManager`
(`document/operator/document_manager.rs`). It can keep the last eight generations in a folder,
one file each, and roll back to one of them.

### 3.5 The stage interfaces

A filter, a scorer and a picker are three traits (`engine/framework.rs`). Each has two required
methods, shown here. The others have default bodies (6 on a filter, 13 on a scorer, 15 on a
picker) and are not shown: most are switches and short paths that the built-in stages use.

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

## 4. The data the engine reads

Rule 2 says the engine keeps no copy of worker data. This section says where the data lives. None
of these structures is the policy crate's.

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
worker (`Objects::lent`). Beside a pair's two lists it hands their pair classes
(`Objects::pair_classes`): each worker's class by its place in the list as lent, under the store's
own pairing rule over what the rows state. A row's wires, engine kind and pairing descriptor can
change under a worker object that stays, with no new list. The holder counts those changes, and the
next ask of the classes derives them again. A controller knows no admission window of a gateway.
So it reads a pair's admission open (`OpenAdmission`), as the rows its store builds state it, and
the gateway that asks gates capacity when it commits. All of it stands in a build with the
`in-process` feature, and no decision path calls it yet: the controller decides a pair over its
store's rows today.

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

What a decision reads of a worker: `Worker::routing_state()` gives ready, circuit breaker,
stalled, overloaded and the in-flight count in one read.

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

### 4.4 The in-flight counters **[built]** (`smg-worker`)

A route creates one `WorkerLoadGuard` for the worker it chose. The guard adds one to the worker's
in-flight counter (and to its routing key's counter), lives as long as the request, and at its
drop takes one off and tells the worker's `RequestCompletionSink` that a request ended. Every
path passes through it: success, error and a client that went away.

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
  for a pick's first target, as before.
- **[planned]** A request given as text is decided the same way in the host's string tree.
- **[planned]** The hash table leaves the single decision. Until then a model with an index of
  cache events reads the index and the hash table together, as today.
- A prefill-decode pair reads no record of sent prefixes in the engine today. Classic keeps
  trees for each side. **[planned]** The engine reads the host's trees for each side.

A decision of the engine publishes no insert to other gateways.

One change was made outside the policy crate first **[built]**: classic's trees moved out of
its policy object into one store type of the kv-index crate (`ModelTrees`), with no behaviour
change. Classic's policy object keeps its trees in that type. The engine's host keeps the trees
it lends in the same type **[planned]**.

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

A process has one engine object. It is built once from the policy document and holds the
document's stages and what they remember. It holds no worker of its own: the registry its host hands it (section 8.1) holds them.

```rust
// engine/policy_engine.rs
pub struct PolicyEngine {
    document: InForce,    // the validated document (profiles, bindings, stage objects and their state), its
                          // number, what is read of it once (which lease events its stages need) and its
                          // shadow counts
    clock: DecisionClock, // one clock for every decision and every load report's age
    leases: OwnWord,      // the id of the last lease issued: one atomic counter
    affinity: Affinity,   // cache affinity: the host's index of cache events, and the engine's own
                          // record of the prefixes it sent, one for each model
    epochs: AtomicU64,    // how many list values were made
    faults: AtomicU64,    // decisions that panicked
    unheard: AtomicU64,   // lease events that panicked inside a stage
    upkeep_faults: AtomicU64, // steps of the engine's own task that panicked: the upkeep, the registry's follow,
                          // a telling of load reports
    loads: task::Loads,   // the host's load channel, once handed, and the newest load snapshot
                          // the engine's own task keeps at hand
}
```

It compiles under the crate feature `in-process`.

### 5.2 The engine's value for one worker list **[built]**

For each list the registry publishes, the engine keeps one value. The host keeps an empty cell
beside each list it publishes (`ListCell`) and lends the two together with each request (`LentList`). The engine makes its value into the cell inside the first decision that asks the list, and reads it there from then on. When the
membership changes the registry publishes a new list with an empty cell, and the old value is
dropped with the old list.

A host that knows a fact of its workers that their objects do not hold lends it with the list
(`LentList::facts`). It lends one fact for each worker, in the list's order,
and the engine writes it into the worker's record (5.3) where it fills the cell. Three such facts stand today:
whether a worker takes images, the id the host knows the worker by where it keeps ids of its own,
and what the worker states for a request's hard demands, which stands in a column beside the
records. The fill makes the list's records.

```rust
// engine/policy_engine.rs
pub(crate) struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>, // the host's own list of worker objects, shared, not copied
    rotation: OwnWord,               // round robin's counter for this list
    records: OnceLock<PoolRecords>,  // the unchanging half of each worker's record (5.3)
    wires: OnceLock<WireModes>,      // which protocols (HTTP, gRPC) the list's workers speak
    alone: OnceLock<(Box<str>, Option<AloneProfile>)>, // the profile one pick can decide (5.6)
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
    pub vision: bool,        // takes images
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

`engine` is `PolicyEngine`'s number. It writes each record its engine's place where it finds the
list's engines, once for a list, so records that state one place are ranks of one data-parallel
engine. A default record leaves 0 on every record, which would read as
one engine, so a stage reads the field where `PolicyEngine` decides: in process, and in a
controller for a single request. The
number holds within one generation of one list and is no id beyond it.

### 5.4 What a request states **[partly]**

```rust
pub struct Request<'a> {
    pub id: &'a [u8],                 // the request's id, as the host names it
    pub facts: RequestFacts<'a>,      // the request, as a policy may read it
    pub constraints: Constraints<'a>, // what only the deciding host knows
}

// engine/view.rs (the fields a stage reads; 30 in all)
pub struct RequestFacts<'a> {
    pub model: &'a str,            // the resolved model id
    pub phase: Phase,              // Single, Prefill, Decode or Encode
    pub tenant: Identity<'a>,      // the tenant
    pub priority: i32,             // priority band
    pub streaming: bool,
    pub max_output_tokens: u32,
    pub prompt_tokens: u32,        // prompt size, counted or estimated by the host
    pub block_hashes: &'a [u64],   // one hash for each whole block of the prompt, in order
    pub hash_kind: HashKind,       // the blocks are token blocks or text blocks
    pub block_size: u32,           // tokens (or bytes) in one block
    pub namespace_seed: u64,       // the cache partition, as one seed
    pub session_key: Identity<'a>, // the routing key, derived as the binding says
    pub client_key: u64,           // fingerprint of the client's key, 0 if none
    pub attempt: u8,               // 0 for the first try
    pub tried: &'a [WorkerId],     // workers earlier attempts already used
    pub headers: &'a [Header<'a>], // the headers the document allows a policy to read
    pub media: MediaCounts,        // how many images, audio and video inputs
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
    pub pins: Option<&'a dyn PinSource>, // session pins kept outside the engine (the old engine)
    pub places: Option<&'a [u32]>, // the places of the list the request may take; `None`: every place
}
```

What is not built in the engine that stays: a request that states a subset hint, a pair's demand,
or protocols that not every worker of its list serves is refused. The refusal names the input,
except a pair's demand, which is shed under the stage `wires`.
A pair's encode items are decided over a host's three lists, with the pair's lease issued
(`PolicyEngine::decide_encode_at`). The trait's method for them takes no lists and refuses.
The encoders are the one list the engine narrows by protocol itself. An encoder is named by its
place in its list, and what its host says of it and of each prefill is laid by that place, so the
host has no narrowed list to hand. The decision leaves out each encoder that serves no protocol
the request states, and a request with items is refused under `wires` where the list holds
encoders and none serves one. The pair's two lists are refused as a pair's are.
The other inputs are honoured by the old engine only.
A request that states places of its list (`Constraints::places`) is refused too, and the refusal
names `places`, where the request is a pair: one list of places cannot hold a pair's two lists
(5.8). A single request that states places is decided among them.

A pair's host states each list's places beside its two lists (`PairLists::prefill_places`,
`PairLists::decode_places`), and each side is decided among its own. `EncodeLists::decide` refuses
lists whose pair states places, under the same name: the encode pass decides its pair over the
workers it admits for the items and reads no place.
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
    pub targets: Targets,                // up to four workers, best first
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
request (section 8). The call runs on the caller's thread. The engine waits on no lock of its own; a
stage that keeps state takes its own short lock (one shard of the session table, or the record of
sent prefixes). A decision that meets the one build of a list's `records`, `wires` or `alone`
waits for it (5.2).

**The short path.** A request is *plain* when it names no tried or excluded worker and states no
hint, no demand, no lifted overload veto, no later decode, no held-back worker and no places of its
list. A profile is decided by *one pick* when its picker needs nothing of the request, it has no
scorer and no filter beyond the health tests, and its binding has no sessions rule and names no
shadow. On main two pickers say so: round robin and `passthrough`.

1. The engine asks `maker.plain()`. If the request is plain and a single one, the host handed one
   list, every worker of the list serves the protocols the request states, and the list's model
   is bound to a profile that one pick decides, the picker names a worker by its own rule and the
   engine answers with its position. Round robin adds one to the list's counter and takes the
   worker it lands on if that worker is eligible. Past a refused worker it takes one more turn
   for each 32 workers of the list, eight more at the most. Where its turns name no worker, the
   engine reads the list once and the rotation picks among the eligible workers. `passthrough`
   takes the first eligible worker in the list's order. Where no worker of the list is eligible,
   the request takes the full path.
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
   workers named before the pass, and the pick recorded after it.
7. **Records.** One `Candidate` for each worker is written into the thread's buffer. Over a list
   of 16 workers or more, a profile whose picker draws first (`random`, `round-robin`,
   `passthrough`, power of two, `best` with a sample) has only the drawn workers' records written.
   For a request held to places no worker is drawn first: every worker's record is written.
   Under a binding that names a shadow every worker's record is written, for the two decisions.
8. **Filters.** Each filter removes records. If a filter leaves nothing, the binding's rule for a
   list wholly over its thresholds may still pick among the workers the overload filter removed.
   Otherwise the decision is a refusal that names the filter: `Saturated` for the overload filter,
   `NoEndpoints` for any other.
9. **Scorers and the picker.** Each scorer scores the survivors. The scores are combined by the
   document's weights. The picker orders up to four targets. No target is a refusal,
   `PolicyDeclined`, naming the picker.
10. **The lease.** If a stage needs lease events and the decision is a pick, the engine issues a
    lease (section 6) and tells the stages, unless a sessions rule keeps the pick from them.
11. The answer is the decision with its lease.

**A retry** names the workers it tried and takes the full path. Where its list holds several
workers of one engine, the ranks of a data-parallel engine, the engine over worker lists names
every worker of a tried worker's engine as tried before the key's pin is read and the filters
run. So `not-tried` passes the whole engine over while another worker is left, and hands it
again where none is. A tried worker the list no longer holds names itself alone: the list does
not say which engine it was a rank of. A pair's retry is widened the same way, each of its two
lists by its own engines.

A decision that panics is caught, counted and answered as refused. It never takes the host down.

A request whose host hands no list (`WorkerLists::None`) is decided over no worker: it is shed
with no endpoint under the stage `none`, and its trace names no profile. A controller hands none
for a model no worker serves, and for a gateway that dials no worker of the model's list.

What one decision writes: the thread's own buffer; on the short path one atomic add on the list's
counter; one atomic add for a lease id where a stage needs lease events; a session pin where a
sessions rule decides; the record of sent prefixes where a cache-affinity profile decides; and,
once for each new load snapshot and list, the converted load reports, which allocates. On the
short path it allocates nothing. A draw that looked up load reports counts them toward its list's
conversion where it takes the list's claim, which it tries and never waits for.

### 5.7 Two engines today, one that stays **[partly]**

Main has two engines behind one trait (`DecisionEngine`, `engine/decision_engine.rs`).

| | `PolicyEngine` | the store-backed engine |
|---|---|---|
| decides over | the host's worker objects, read in place | records copied into the policy store |
| keeps leases | no: the host hands the lease back with each event | yes, in the store |
| used by | the gateway's earlier integration, until stage 0 takes it out; the controller's service, for a single request | the controller's service, for a lease's words and its counters; it decides no request there |
| size | 6,228 lines with its folder | 12,461 lines with its folder, plus the store's tables |

**The design keeps `PolicyEngine` and removes the store-backed engine [partly].** The controller
gets worker objects of its own (section 10), so its service calls the same engine a gateway
calls. The switch is made in steps (decision 5): a single request is decided by `PolicyEngine`
on main (10.1), the store-backed engine is deleted right behind that step, and a pair's phases,
answered `Busy` meanwhile, are then decided by the one engine.

The two engines still share types. `Request`, `DecideOutcome`, `DecisionRefused`, `DecodeRefused`
and `SweepOutcome` are defined in the trait's file, and `Said` (section 6) beside `Lease`. They
were moved there with no behaviour change, because deleting the old engine would otherwise break
the kept one. `GatewayId` is still defined in the store.

### 5.8 What changes in the kept engine **[planned]**

- It honours what it refuses today: an encode request's pair held to some workers of its two
  lists (`PairLists::decide` keeps a pair to the positions its host states, and a single
  request states the positions of its list in `Constraints::places` and is decided, which is
  built); a pair's demands (a single request's are kept, which is built).
- Five request fields that only the old engine's host fills leave with it (`pinned_at`,
  `lent_in_flight`, `lent_session_in_flight`, `session_counted`, `model_fingerprint`),
  and `Constraints::pins`. One of them needs a successor: a sessions rule with
  a cap asks `session_counted` of its host today.
- The trait's method that the kept engine answers with a constant, `decide_encode_pair`, is built
  or removed.
- The record of sent prefixes leaves the single decision. The two trees a request is then
  decided in are the host's own, lent (section 4.5).

`signal_cells` stays, and it is built for a single request: where the request's profile reads a
named signal, the kept engine fills the field itself, from the reports of the load snapshot the
decision holds. Such a decision converts its list's reports where no decision under its snapshot
has, ahead of a routing key's short answer too. A row that is one rank of a data-parallel worker
reads its own rank's values. One limit stands: a worker object that stands for several ranks
reads no named value, because its report holds one sample instant for all its ranks. So a
signal's `across_ranks` has no effect under the kept engine until the worker crate's report
states each rank's own sample instant. An encode request decided over its three lists is not
lent cells yet: neither its encoders nor the two sides of its pair.

A pair is lent them a side. Where the prefill profile reads a named signal, the prefill side
reads its prefill list's cells, and the decode side reads its decode list's where the decode
profile reads one. Both are of the snapshot the decision holds, and a decode decided again
reads those of its own call. The pair's pools carry each side's cells
(`PairPools::prefill_signal_cells`, `decode_signal_cells`), so the request is not copied.

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
    pub tokens: u32,                     // the prompt size the request stated
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
| Dispatched | the session's pin follows the worker the request went to; each profile counts what it sent since the worker's last load report; scorers and pickers that asked hear it. A pick a sessions rule keeps from the profile moves the pin alone |
| Completed | scorers and pickers that asked hear the outcome (status, time to first token, tokens in and out, duration) |
| Released | the same stages hear the reason |

A sessions rule places some requests alone, with no stage asked: it follows a pin of the request's
key, or its own assignment places the key. The rule says whether such a request is the profile's
own (`rule_placed`). Under `told`, the default, it is: the stages hear its issue and its dispatch,
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
recorded as a pin's the whole decision follows. A controller that tells a dispatch off a lease on
file has no such word yet: it takes the document, and its stages hear such a request as any other.

Upkeep, about once a second (`sweep`): expired session keys are removed, and the record of sent
prefixes gets a bounded step of clean-up.

## 7. The ten built-in policies **[built]**

Each classic policy name has a preset in the crate. Every preset starts with the filters
`healthy`, `not-overloaded` and `not-tried`. In a pair, the prefill side adds `has-capacity` and
the decode side adds `has-room`.

| Policy | Scorers | Picker | What it remembers between decisions |
|---|---|---|---|
| `random` | none | `random` | nothing: it draws from the decision's seed |
| `passthrough` | none | `passthrough` | nothing: the first survivor |
| `round_robin` | none | `round-robin` | one counter for each worker list, which the one pick and a single request's draw over 16 workers or more both turn; in a fixed table, one rotation for each set of surviving workers and one for each pool whose draw is handed no list counter |
| `power_of_two` | `least-load` | `power-of-two` | what was dispatched to each worker since its last load report |
| `least_load` | `least-load` | `best` | the same |
| `consistent_hashing` | `consistent-hash` | `best` | nothing where the host lends its ring and states the request's place; else one hash ring, of the workers its decisions have read |
| `prefix_hash` | `consistent-hash` by the routing key, else by the prompt's first 256 tokens, with a bounded-load rule; `least-requests` | `best` | the same ring |
| `manual` | none | `manual`, with a sessions rule | routing key to pinned worker, with an expiry |
| `cache_aware` | `warmup-slice`, `prefix-affinity`, `kv-pressure`, `least-load` | `best` over a sample of 16 | the record of sent prefixes; the warm-up counters; the dispatch count of `least-load` |
| `bucket` | `bucket` | `best` | for each model, the ranges of prompt sizes each worker takes, and a time window of booked tokens |

Whether each preset gives classic's answer is section 13.

## 8. What a host does

This is the contract between the engine and the program that embeds it. Its start is the same
for the gateway and for the controller. For each request the two differ. The gateway makes the
one call of section 8.2 and keeps the placement. A controller states worker ids of its own and
keeps no placement: it calls the entry that issues nothing and files its own record.

### 8.1 On main today **[built]**

At start:

| The host does | With |
|---|---|
| starts the engine with its own data, once | `PolicyEngine::start(spec, WorkerData { registry, loads, cache_indexes })`. It builds the engine from the document and takes the host's worker registry (the host implements `Members`, and the worker crate's registry is one), its load monitor's channel and its indexes of cache events (the host implements `CacheIndexes` and `CacheIndex`). The same hand-over also lends the token tree, where the host keeps one (`CacheIndexes::token_tree`). A host with no load monitor or no index says `None`. It answers the engine, which lives as long as the process, and the engine's own task |
| spawns the task the start answered, once | on a runtime of its choice, which needs its timer on: the upkeep once a second, the newest load snapshot kept at hand, the follow of the registry, which tells the policies a worker that joined or was removed (the table below), and each new load report told to the policies that keep them |
| keeps an empty cell beside each published worker list | a `ListCell` in its value of a membership generation, lent with the list as a `LentList`, with `facts` where the host knows a fact of its workers that their objects do not hold |
| hands its load monitor's channel again where it builds another monitor | `engine.set_loads(channel)`; an engine whose document reads no load report and no named signal, and none of whose policies keeps a report, keeps none |

A caller that builds its engine with `PolicyEngine::new(spec)`, as the crate's own cases and bench
do over lists made by hand, makes each hand-over itself: `engine.set_registry(registry)`, or with
no registry `engine.forget(address)` when a worker is removed and `engine.joined(address)` when
one joins, a removal and a registration at one address in that order;
`engine.affinity().set_indexes(indexes)`; `engine.set_loads(channel)`; and it spawns
`engine.run()`. A host makes none of these calls but `engine.set_loads`, where it builds another
load monitor.

What the engine's task tells the policies at each event of the registry. It names a worker by the
id the host states of it (`Members::facts_of`), which is the id the host's lists name it by; a
worker whose host states none is named by the fingerprint of its address.

| The registry's event | The policies are told |
|---|---|
| `Registered` | a join |
| `Removed` | a leave, under the id told at the worker's join |
| `Replaced` | nothing where the host states the same id of the new object; a leave of the old id and a join of the new one where it states another; nothing of an object the task does not hold |
| `StatusChanged` | nothing: a decision reads a worker's status on its object |

A worker removed and another registered at its address come as a `Removed` and a `Registered`,
so a leave and then a join. Where the registration is heard first, the worker is read as its
id's next life: a leave and a join of the id, and the removal then tells nothing. That holds
under the registry's own id of a worker too: a removal tells a leave only of the worker the
task holds. The events
are a broadcast that drops its oldest for a reader that falls behind: the task then reads the
registry whole, tells what changed, and last tells the ids that stand, so a stage keeps
nothing of a worker whose whole life the channel dropped. It does the same behind a channel
that closed, and at its next upkeep behind every registration, replacement and removal of a
host known to state ids, from the first ask it answers with one: such a host may have stated an
id of a worker and taken it back ahead of
the event. The session table is told nothing: a pin on a worker that is gone stands until the
session's idle time or two other workers end it.

For each request the host implements a small trait and calls the engine. A host writes `model`
and `make`; the other four methods have default bodies:

```rust
// engine/decision_engine.rs
pub trait RequestMaker {
    fn model(&self) -> &str;                          // the request's model
    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w>; // builds the request
    fn plain(&self) -> Option<Plain>;                 // what a plain request states, if it is plain
    fn workers(&self) -> WorkerLists<'_>;             // the worker list, a pair's two lists, or an encode request's three
    fn loads(&self) -> Option<&Arc<LoadSnapshot>>;    // a request's snapshot of the host's own; without one the engine reads its own
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>>; // the prompt's entry in the host's token tree, where the host lends one
    fn sends_to(&self, target: &Target) -> bool;      // whether its caller sends to this target of a pick: the tree stores the first such
}

pub enum WorkerLists<'a> {
    None,
    Single(LentList<'a>),    // one list for a single request, lent with its cell
    Pair(PairLists<'a>),     // a pair's prefill list and decode list
    Encode(EncodeLists<'a>), // an encode request's three lists: its pair's two and its encoders
}

pub enum Decided {
    Alone(AlonePick),           // a position in the list; nothing is owed to the engine afterwards
    Whole(DecideOutcome, bool), // a full decision with its lease
}
```

Then the host itself has to: derive the routing key by the document's rule and hash the prompt
into blocks, inside `make`; try the targets in order, checking each worker's health again; keep
the decision's lease and report each event; turn a refusal into a status. For a single request and for a pair
`place` of section 8.2 now derives the key, hashes the prompt and tries the targets, and its
`Placement` keeps the lease. The host still says the two events and turns a refusal into a
status.

That is too much. The gateway's earlier integration needs a module of its own of 19 files and
13,856 lines (tests included) for it, and a branch in every route.

A host that keeps a hash ring of a model's workers can lend it with a request: the request's facts
carry the ring, behind a trait of two methods (`LentRing`), and where the request stands on it
(`RequestFacts::ring`, `RequestFacts::ring_positions`). A trap for the lender. Under `prefix_hash`
classic asks its ring for the sixteen hex digits of the hash of the bytes a request is placed by,
also where those bytes are a routing key's (`model_gateway/src/policies/prefix_hash.rs:223`,
`:292`, `:300`). Under `consistent_hashing` it asks for the routing key's own text
(`model_gateway/src/policies/consistent_hashing.rs:84`, `:109`). A lender that states a request's
place from the wrong text sends routing keys to other workers than classic does, and no test in
the crate sees it.

A caller of the one call states the ring alone (`PlaceRequest::ring`). The engine reads where the
routing value stands on it with the request's keys, under a profile whose ring stage places a value
by its own text. Under `prefix_hash` such a request is lent no ring yet: the stage's own ring
places it.

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
runtime.spawn(task);                       // the engine's own background task
```

The engine lives as long as the process, so a placement holds a plain reference to it.

The start answers the task and the host spawns it on a runtime of its choice: the start spawns
nothing. The call is marked `#[must_use]`, with the reason that the task does nothing until it is
spawned: the compiler warns a caller that drops the whole answer. It does not warn a caller that
keeps the engine and drops the task, whose engine keeps no load snapshot, runs no upkeep, tells
the policies no worker and tells no policy a load report. The start takes no worker lists: each
request gives its own, and the engine's task reads none.

The engine's background task does what a host did by hand. Four of its jobs are built
(`PolicyEngine::run`): the upkeep once a second, and the newest load snapshot kept at hand,
swapped when the monitor publishes one, so that a decision reads it without touching the
channel. The third is a worker that joined or was removed told to the policies, read from the events of the registry the host hands at the start (section 8.1). A host that
builds another load monitor hands its channel again (`engine.set_loads`): a channel ends with
its monitor. The task's first upkeep runs at its first poll, where it reads the registry and
takes the load channel: a host relies on both once the task has been polled once, not when the
start returns.

The fourth job serves a policy that keeps a number of each worker's load report
(`Lifecycle::keeps_loads`). At each load snapshot the task takes, it tells such a policy the
reports that are new in it, for the workers it follows in the host's registry, and only then keeps
the snapshot at hand. A report is new where its sample instant is not the one the task told last
of its worker. The channel keeps one snapshot, so of two reports of one worker the task took no
snapshot between, the earlier is told to no policy.

The host keeps one empty cell beside each worker list it publishes. That is one field in the
host's `Generation` type. The engine fills the cell at the list's first decision.

The cell is built. A request's `WorkerLists` name each list as a `LentList`, and the decision
that needs a list asks its cell. The start does not take the lists' trait yet.

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
    PrefillsFull, // the host's admission holds back every prefill of a pair: the host waits for room
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

Where the document's stages want no events, a single request's placement holds nothing and
these calls do nothing. A pair's placement always holds a lease (section 6), and so does an
encode request's: it reports its pair's two sides as a pair does, and nothing of an encode side.
No lease stands for an encoder, so `sent` and `ended` under the encode side tell no stage and
end no side. The host that sends an encode call counts it on the worker itself, as the gateway's
encode stage does. `place` answers a single request, a pair and an encode request with a
placement, and no host makes one itself.

A pair's decode can be decided again on its placement. Under a document that picks the decode
after the prefill, a caller that states `decode_again` is placed with its prefill booked alone,
and the decode the first decision named stands chosen. When the prefill has answered, the caller
asks `decode_again` with the request stated again, as its first call stated it. The engine makes
the request of it anew, as it makes a first call's, and decides under the binding of the model it
states. Every fact of the decode's decision is that request's: the pair's two lists as they stand
then, the request's size, its routing value and the workers it tried among them. It decides the
decode on the records of that instant, behind the prefill that was sent, and the placement takes
the first decode it names that can be sent to. The placement's routing key and its two profile
labels stay as the first call wrote them. The order is part of the contract: the caller asks, or
says the send of the first decision's decode, before it says the prefill's end, and a decode
still chosen at that end leaves with it. Behind a decode the call took, the prefill's end is said
when the prefill answers, and the decode's send at its own later instant. Where the call takes no
decode the placement stands as it stood. A shed is answered by its name. A decision none of
whose decodes can be sent to answers `NoWorkers` and gives up the side it issued; the caller may
then ask again, send the first decision's decode, or answer its client.

A host that keeps a value with each send, as a gateway keeps a worker's load guard, takes a
kept placement apart into its sides. A side says its dispatch by one call and its end at its
drop, with no outcome:

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
    // dropped: a release where the side never went out, an end of class cancelled where it did
}
```

A pair's two sides share the pair's one placement behind a lock of that one request. The
crate keeps the order of a pair's words itself: a prefill side dropped while the decode still
stands chosen holds its end back until the decode side speaks, and says it then with the
instant of its own drop.

A host that keeps sides asks the decode again on the pair's decode side, once the prefill
side's send was said. It may let the prefill side go before that call or behind it: a
prefill's end that waits is said behind the decode the call took, and behind a call that took
none it still waits. Any other side is refused by name, with no decision made.

Nothing else is asked of a host that makes this call. In particular it never names a policy,
never looks inside a decision, never keeps a lease and never runs a timer for the engine.

### 8.3 What is reused and what is added

The design adds no second mechanism where one exists.

| Need | Reused, as it exists | Added |
|---|---|---|
| a model's worker lists | the registry's published lists (`Generation`) | one field beside each list: the engine's value |
| a worker's health, breaker, overload, in-flight count | the worker object, read in place | nothing |
| load reports | the load monitor's `LoadSnapshot` | nothing |
| which worker holds a prompt, by cache events | the kv-index crate's index, through two traits | the gateway's two readers of its index for those traits, about 130 lines (section 9.2) |
| which prefill pairs with which decode | the gateway's pair index (`PdPairIndex`) | the pair classes read from it, one file of 282 lines (section 9.2) |
| which worker was sent a prompt | the host's own trees, of the kv-index crate's two tree types | one more question on the trait the host already answers for cache events |
| a request's lifetime on a worker | the load guard every route keeps | one optional field (decision 4) |
| the place where a worker is chosen | classic's five placement functions | one call in each |
| the rank a request is sent with | the worker object's own rank, where its registration states one | nothing: no rank choice runs behind the pick of a worker registered rank by rank, and the classic rank option is refused at the start beside the engine (section 9.2 names the two places a rank is still chosen) |
| the lease's reporting | `DecisionGuard` | `Placement`, which holds it |
| proof against classic | classic's recorders and their sheets | replays through the kept engine |

## 9. The gateway as a host **[partly]**

After stage 0 the gateway did not call the engine. This section is the shape of the one change
that brings it back, behind an option that is off by default (section 9.3).

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
  the engine on, no classic policy object is built. So the start function owns the trees the
  engine is lent **[planned]**: it keeps them, runs their size pass on a timer and tells them a
  worker that left.

  The start makes the policy crate's one start call as the last step of the context's build. The document is checked first against the table of stages the gateway's host hands the start, or the stock table where it hands none, and a refused document ends the start before the gateway listens. The engine's task follows the registry's events and tells the document's stages a worker that joined and a worker that was removed. Where the configuration runs the policy controller the gateway builds its policy registry with no classic policy object: a stand-in that places nothing stands in its default and encode slots, and a worker's registration and a router's build add no policy.

- **Each request.** Four of the five functions ask the engine. They fill `PlaceRequest` from the inputs
  they already take. A route names the answer's type and makes its load guard from the answer, in the places it made one before.

  The fifth, the encode stage's own selection, takes its pair from `select_pair` and refuses a request that carries an item for the encode workers by name **[planned]**: the gateway does not yet hand the engine an encode request's three lists.

  Where the document asks a pair's decode again, the two pair routes make one more call between the prefill's answer and the decode's dispatch (section 8.2).

  A retry held to its wire is placed over the pool's whole list: the gateway holds it there by the route's own rule for a worker, and a pair's retry by the places of each pool on that wire. It does not yet give each runtime and transport it dials a number, as a worker's fact and as the retry's demand (section 8.2) **[planned]**.

- **The request's end.** The placement stays with the request's load guard, which every route
  already keeps to the end of the response, streamed or not (decision 4).

  The engine hears the dispatch where the guard is made, or at the send for a guard a route makes ahead of its send, and the end where the guard is dropped. It is told no outcome, so a plugin stage that learns from outcomes learns nothing in process.

- **Membership.** The gateway's value for a model's workers gets one cell for each list, which
  the engine fills.

  The pair classes a gateway hands with a pair's two lists are derived under the pairing of the document in force: beside a supplied document the registry takes that document's `plans.pd_pair.pairing` at its build, and no request reads it.
- **The rank.** A data-parallel worker registered rank by rank (the gateway's `--dp-aware`) is the shape the engine serves in full. Each rank is then a worker object, and a record of its own in its list (`Candidate::dp_rank`, section 5.3). The record reads its own rank's named values (section 5.8). A stage that keeps load reports is told each rank's own report, under that rank (`Signal::Load`). `Candidate::engine` is the same number on every rank of one such worker. The request is sent with the rank of the record the engine picked: where a record is one rank, no rank choice runs behind the pick. A rank is chosen behind the pick in two places, each by classic's own code, with the engine on as under classic, and each inside a worker registered once for all its ranks: on the HTTP pair route, under a connector that holds both sides of a pair to one rank, the route's own turn names the rank both sides are sent to; and under the gateway's `--grouped-dp-sticky hash`, the gRPC dispatch pins a single request that has a sticky key, and where the runtime allows it a pair's prefill side, to that key's rank. That option is refused beside a supplied document.

  The gateway's rank option (`dp_minimum_tokens_scheduler`) is refused at the start beside the engine, under the settings' own document and under a supplied one. A worker object that stands for several ranks states no rank, and outside those two places its request is sent with none.

The change also writes again three small things that stage 0 takes out with the earlier
integration: the gateway's two readers of its own index of cache events for the engine (about
130 lines on main today), the pair classes read from the gateway's pair index (one file of 282
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

They come back with the change (decision 9). No released version has them.

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
| worker objects | a task that follows the worker records | a second form of the same workers, of the worker crate's type; each states its registration's overload thresholds over one default, 8 queued requests and 0.8 usage, and no monitor judges by them yet; an object's status follows its record's, and a single request's decision reads the workers here |
| health | the worker crate's health checker over the worker objects, on the leader | copied into the records |
| loads | the crate's own poller, every 10 seconds | `PushLoad` is not served; a single request's decision reads none yet |
| cache index | a follower of each worker's cache events | an index of the store's own; `PushKvEvents` is not served; a single request's decision reads none yet, only the engine's own record of sent prompts |
| sessions, leases | decisions (leases only) | not replicated to a standby; a single request's session pins are the engine's own table, one for each document, and no decision reads the store's pins |

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
that the worker objects' registry publishes. A request of a prefill, a decode or an encode phase
is answered `Busy`, and its gateway decides it.

### 10.2 As designed **[partly]**

Rule 2 and rule 3 apply to the controller as they do to the gateway: one set of worker data, in
the worker crate's structures, and the one engine.

| Data | On main | As designed |
|---|---|---|
| workers | records in the store, and worker objects kept in step with them | worker objects in a worker-crate registry; beside each, the code that keeps the objects holds what a registration stated and the object does not hold as stated (10.3); the membership log that watchers and standbys read stays |
| lists | the store's published copy; a single request's decision reads the registry's published list of the model's single workers | the registry's published lists by role and protocol, with the engine's value beside each (decision 10) |
| health | the worker crate's health checker over the worker objects, on the leader; each outcome is copied into the store's record, an object's status follows its record's, and a single request's decision reads the object's | the same checker; a decision reads the object's own status; a gateway's veto on a worker stays in the store |
| loads | the worker crate's load monitor over the worker objects, while the replica leads: a single request's decision is given its snapshot where the document reads reports and keeps none, and reads the overload verdict on the object; the crate's own poller and table still run, and no decision reads them | the worker crate's load monitor, publishing `LoadSnapshot` (decision 10). Every worker is judged by the thresholds its object states, so the loads of every worker the controller dials are read: no new cost against classic, whose gateway hands every worker the same two defaults |
| cache events | an index of the store's own; a single request's decision reads none yet | the kv-index crate's index, filled by the worker crate's collector |
| sessions | the engine's session table for a single request, one for each document; the store's pins of record are not written at dispatch, and no decision reads them | the engine's session table |
| leases | the store's lease tables, in memory, one for each deciding thread | stay: a controller must remember what it issued to gateways |
| decisions | a single request: `PolicyEngine::decide_without_lease` over the registry's list; a pair's phases: none, the request is answered `Busy` | `PolicyEngine::decide_without_lease` over the registry's lists |
| discovery | not registering | the discovery crate through its host interface, registering through the worker crate's job queue |

What leaves the crate with that switch: the store-backed engine, the store's copies of the worker
data that a worker object holds, the crate's own load poller, and the adapters that write worker
state into the store. Some facts of a worker are not carried to its worker object today, and
some stand there in a narrower form than a registration states: 10.3 names them and where they
are kept.

**The first step, on main.** A single request's decision reads the model's single workers in the
order they registered, where the store's copy was sorted by id. It reads a worker's status, its
breaker and its stalled bit on the worker object, and the count of requests in flight there, which
the store steps for each lease on file. A record made of a worker object states one protocol and
no zone.
The asking gateway's reach, its wires and its standing vetoes are stated to the engine as places
of the list. A new document is a new engine object: its session pins and its record of sent
prompts start empty, and that record is told no worker's leave. A sessions rule's cap counts no
request of a key: a key stays on its worker past the cap, as under a rule with no cap. A
registration, a drain or a deregistration is on the worker objects when its call answers: a
decision behind the acknowledgment reads the fleet the call left. A controller that starts over
standing rows holds their worker objects before it serves.

A standby's applied records reach its worker objects at the next pass of the task that follows the
worker records. A replica that takes the lead reads its last leader's last change at that pass.

### 10.3 The controller's worker data **[partly]**

One worker's data stands four times whole in a controller, and a fifth time in part: the
remainder holds 17 of its 24 facts as stated and one bit of an eighteenth. The membership log
tells the data once more:

| Copy | Where | Read by |
|---|---|---|
| the worker record | the store's table of records (`store/memory.rs`) | the registration calls; a watcher's and a standby's first copy |
| a copy of the record's facts | beside the worker objects (`worker_host/objects.rs`) | the code that keeps the objects, to tell a record that changed under its object |
| the worker object | the worker-crate registry | the health checker, the follower of cache events and a single request's decision |
| the remainder: 17 of the 24 facts and one bit of an eighteenth, with the record's id and revision | beside each worker object (`worker_host/remainder.rs`) | a single request's decision, for the record's id, whether the worker takes images, its wire modes and from where it is reached. What the worker states for a request's hard demands is lent with the list, and no frame states a demand yet |
| the published copy of the records | the store | a pick's booking and a gateway's vetoes; no request is decided over it |
| the membership log's event | `service/replication/log.rs` | a watcher behind its first copy; a standby |

The record leads and the object follows it. After the switch the object leads: a worker's data
stands once, as its worker object with its remainder beside it, and the log still tells it. The
remainder is the copy that stays. The table of records and the second copy of the facts end
with step 4 below, and the published copy with the store-backed engine (10.2).

**What stands beside an object [built].** `RowRemainder` (`worker_host/remainder.rs`) holds
what a registration stated and the worker object does not hold as stated. The code that keeps
the objects writes one with every object it builds or takes. `Objects::row_of` gives a worker's
whole record back from the object and its remainder, with no read of the store's table, for an
object built from its record or handed in with it. An object that was handed in and kept while
its record moved states the six facts it holds whole as they were handed in, and as the
engine's address the one it dials, where the moved record names an engine. A single request's
decision reads four of the remainder's facts: the record's id, whether the worker takes images,
its wire modes and from where it is reached. Nothing in the controller's service reads a whole
record back yet. Of the 24 facts a registration states:

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
| the overload thresholds | where a program asks for them, as the controller's does, the worker's own over the program's defaults | which of them the worker stated |
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

**Behind the switch [planned].** Four steps, each small:

1. A registration builds the worker object first and tells the membership log; no table of
   records leads. The object, its remainder, the count of membership changes and the log's
   event change in one turn, and a first copy is cut in the same turn, as the store's lock
   orders them today. A change that moves a fact the object holds builds the object again: the
   new object takes the standing one's place under its id, with its running state, as the code
   that keeps the objects does today for a record that changed.
2. A standby applies the leader's events to its own worker objects, and a watcher's and a
   standby's first copy is cut from the objects and their remainders.
3. No health outcome is copied into a record: a decision reads the object's status, as a single
   request's does on main.
4. The table of records, the second copy of the facts, the two conversions between a record
   and an object, and the task that follows the table are deleted.

**One rule for a registration's write [built].** What a registration makes of the record under
its address is one function beside the other membership rules (`WorkerFacts::laid_over`,
`store/membership.rs`): from the record that stands and the facts stated, the change and the
record it leaves. The store's own write decides by it over the table's record. It then writes
the decided change by the one body a standby's applier writes a leader's change by
(`MemoryStore::follow`). Step 1 gives the rule its second caller: the code that keeps the worker
objects, over the record read back from an object.

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
- The gateway's side is the same five functions of section 9, made asynchronous: an answer from a
  controller is a round trip, so each route's call changes by that much.
- The request crosses in the contract's own form. The prompt crosses as hashes of its blocks, not
  as token ids or text. A route's own rule for a worker crosses as the ids of the workers the
  gateway excludes. So a controller matches a prompt by whole blocks. For the two trees of
  section 4.5 the contract must hold token ids, or the trees must be keyed by block hashes. The
  controller's own design settles which.
- When no answer comes in time (no credit on the stream, a change of leader, the controller
  gone), the gateway decides by itself with the same engine, under the document the controller
  pushed, over the worker list it follows (decision 6). That list stops changing while the
  controller is gone.
- Open: where a gateway under a controller gets a worker's API key. The contract's worker entry
  has no field for one.

On main none of this is wired. A gateway given an address only follows the controller's worker
list and decides every request itself, and it refuses prefill-decode and encode requests. That
code leaves with the earlier integration.

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

On main: 18 sheets, 482 lines.

| | Sheets | Lines |
|---|---|---|
| replayed and passing | 15 | 434 |
| recorded, replayed by nothing (`constraint_decider`, `overload`, `minimum_tokens`) | 3 | 48 |
| replayed through the engine that stays (`PolicyEngine`) | **5** | **45** |
| replayed lines held to an answer of the crate's own, not classic's | | 12 |

A replay of all fourteen decision sheets through `PolicyEngine` over worker objects is in work and not on main yet: on main the four hard-demand sheets (`retry_wire`, `media_refs`, `context_window`, `rank_pin`; 37 lines) and `sticky_rank` (8 lines) replay through it. Measured
with it locally on 2026-11-22, on main (`096c0f5ea`): of the 408 lines that record a decision
(point 1 below), the engine answers 286 as classic recorded and 28 differently, and 94 cannot
be asked yet. Of the 94, 25 are declined because the request states a hard demand, 16 need a
word a request cannot state, 52 are encode requests and 1 is refused at validation. Runs with
one input left off the worker objects (the requests in flight, the load reports, the status,
the ring) lose lines, so the replay can fail.

Five things follow, and they are the first work of section 16.

1. **Only the four hard-demand sheets and `sticky_rank` (45 lines) go through the engine that stays** in main's own tests. The other replays
   call the decision function over hand-built candidates. Fourteen sheets
   (408 lines) record a decision, and each has to replay through `PolicyEngine` over worker
   objects. The other four (74 lines: `overload` 25, `pair_index` 23, `minimum_tokens` 18,
   `sticky_rank` 8) record an answer of the host, which the kept engine is never asked for. It
   reads the overload verdict that the host's load monitor wrote on the worker object. It is
   handed each prefill's partners with a pair's two lists. It names the rank a worker
   registered, never one of the ranks behind a worker that stands for all of them. In a host's
   own process those answers are classic's own code, which stays, so the 74 lines hold there
   with no replay. For the two rank sheets that rests on a rule of the gateway's change
   (section 9.2). On main `pair_index` replays through the old engine's side, `sticky_rank` replays over rank objects through `PolicyEngine`, and `overload` and `minimum_tokens` (43 lines) are replayed by nothing: the store that judged them is no longer read, and they wait for the host's code
   until stage 4.
2. **23 lines were held to the crate's own answer** (`minimum_tokens` 10, `overload` 6,
   `sticky_rank` 4, `prefix_hash` 2, `manual` 1). The 16 of `minimum_tokens` and `overload` were held by the old engine's store and are not replayed now (point 1); three of them are rules of the scorer and the picker, which the kept engine shares. The 4 of `sticky_rank` are held the same through `PolicyEngine`, by the ring and the picker. The
   other three are the kept engine's. The 2 of `prefix_hash` differ from classic there too (a
   key folds whole blocks). One of them turns in the crate on a ring its caller lends
   (decision 3). The other measures which tokens are in a prompt's head, which is then the
   caller's recipe: it is proven at stage 3, by a gateway test of classic's own function. The
   1 of `manual` cannot be asked (validation refuses a sessions rule under a pair plan). Each
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
   candidates. Through `PolicyEngine` over worker objects with a lent ring no recorded line is
   measured yet. The other 16 measure which tokens are in a prompt's key (15 lines) or how
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

One cluster run exists, of 2026-11-22: 72 engines, 1,024 clients, one run for each line, through
the gateway integration that stage 0 takes out. It has to be repeated through the new one.
"Same worker" is the share of a conversation's next turns that went to the worker of the turn
before.

| Policy | Who decided | Requests a second | First token p50 / p95 ms | Same worker |
|---|---|---|---|---|
| `cache_aware` | classic | 419.89 | 207.95 / 423.78 | 0.9960 |
| `cache_aware` | engine | 419.61 | 206.41 / 419.63 | 0.9961 |
| `power_of_two` | classic | 303.67 | 393.22 / 654.35 | 0.0152 |
| `power_of_two` | engine | 302.68 | 394.21 / 658.95 | 0.0156 |
| `least_load` | classic | 302.26 | 361.16 / 569.44 | 0.0159 |
| `least_load` | engine | 302.77 | 363.26 / 575.41 | 0.0161 |

On these three policies the engine and classic are not told apart: two runs of one setting
differ by as much.

The same run showed the faults of the engine's own record of sent prefixes (section 4.5), under a
document whose prefix stage reads that record alone. The two-tree lines ran on a build 32 commits
later than the others, so their speed is not proven to be the record's alone.

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
| The policy document: profiles, bindings by model and phase, plans, overload, routing key, sessions | **[built]** | |
| Bindings by tenant, priority, header, prompt size, labels; a session id issued by the engine (`issue_session_id`) | **[planned]** | in the schema; a document that writes one is refused as not in this build |
| `canary`, `override: allowed`, `admission` | **[planned]** | in the schema; a document that writes one is refused, and the error names the field |
| `shadow`, `explain.sample` | **[partly]** | the kept engine runs a single request's shadow, but not one that reads affinity rows of its own; a pair's shadows run in no engine, and `explain.sample` is kept by the served controller alone (decision 7) |
| The ten presets and the conversion from routing settings | **[built]** | |
| The stage interfaces and a builder's own stages | **[built]** | |
| The engine over worker lists (`PolicyEngine`) | **[partly]** | refuses a subset hint, a pair's demand, mixed protocols; of the recorded lines only the four hard-demand sheets and `sticky_rank` (45) replay through it |
| `round_robin`, `random`, `passthrough`, `least_load`, `power_of_two` on that engine | **[partly]** | no sheet over a healthy pool for four of them |
| `cache_aware` on that engine | **[partly]** | a single request whose maker names its entry by token ids (`RequestMaker::token_key`) is decided in the host's token tree, where the host lends it, the model has no index of cache events and the profile reads sent prefixes and names the holders' rows, as the preset does: the record's first two faults do not reach it. Every other request the engine names the holders of reads the engine's own record, with its three faults: a request whose caller names no entry is one. The string tree is planned |
| `consistent_hashing`, `prefix_hash` on that engine | **[partly]** | placed on the ring the host lends (decision 3): through `place` the plain ring alone, at the place the engine reads with the request's keys; `prefix_hash` at the places a host that writes the request's facts itself states; that both are classic's is proven at stage 3; a host that lends none is placed on the crate's ring, which may be another worker than classic's |
| `manual` and sticky sessions on that engine | **[partly]** | their sheets replay through the decision function, not through the engine |
| `bucket` on that engine | **[partly]** | the same |
| Prefill-decode pairs on that engine | **[partly]** | the sheet of classic's pair choice (`pair_select`) replays through the pair's decision function, not through the engine; five of its 40 lines are held to an answer of the crate's own |
| Encode-prefill-decode on that engine | **[partly]** | `PolicyEngine::decide_encode_at` decides it over a host's three lists and issues the pair's lease, and `place` places it; the entry declines a request held to a wire that a worker of its pair's two lists is not dialed on, places its items on the encoders dialed on a wire it states, declines a request with items where the list holds encoders and none is, and declines one that states the later decode where its decode is held to the workers that take images: a decode asked again takes no items |
| The lease and its events; the guard | **[built]** | |
| The host contract of section 8.2 (`place`, `Placement`) | **[partly]** | `Placement` is built, with its call for a pair's decode decided again, and `place` for a single request, for a pair and for an encode request, with the route's rule and the decision made again, three decisions a call at most or a single request's fourth where the third found the workers the route's rule takes, and a single request's later decisions held to those workers; the later decisions of a pair and of an encode request held to them are not |
| The crate's bench of the kept engine | **[planned]** | |
| The gateway as a host | **[partly]** | section 9, behind `--policy-controller on`, which is off by default: single requests and pairs on the HTTP and gRPC routes, a pair's decode asked again, a host's table of stages at the start; an encode request that carries an item is refused by name until the gateway hands the engine its three lists; every classic test of the gateway with the engine on (the bar's fourth point) is not met yet |
| The controller: service, election, replication, document apply and roll back | **[partly]** | a session never pushes the document and refuses a gateway's replay; a standby serves neither the worker list nor the snapshot (section 10.1) |
| The controller on the one engine and the worker crate's structures | **[partly]** | a single request is decided by `PolicyEngine` over the registry's list; a pair's phases are answered `Busy`; a decision reads no cache event, and no load report under a document that holds a stage that keeps reports, and a sessions rule's cap counts nothing (section 10.2) |
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
| (a) The host lends the ring it keeps; the engine asks it which workers a routing key reaches. | Crate only: one deletion and four changes. The crate's ring beside a worker list is deleted (1,728 lines removed, 230 added, 17 files). The four changes are about 580, 270, 450 and 550 changed lines, about 370 of them product code and 1,050 proof. The host lends its ring through a trait of two methods and states where a request stands on it. The crate then writes no hash and no ring for such a host, so a request is placed on classic's own ring. The price in proof: 16 of the 47 recorded `prefix_hash` lines test which tokens are in a key. In the crate they keep replaying on the crate's own ring, share for share as on main. A gateway then never runs that ring, so for a gateway they are proven at stage 3 (section 13.1), where the change also cuts classic's key recipe and classic's ring lookup into functions that classic and the lender both call. A controller has no classic ring to be lent: until (b) it keeps its own ring and places a routing key on another worker than a gateway would. |
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

**As taken.** The deciding threads were switched for a single request first (section 10.1). The
store-backed engine is deleted right behind that step, and a request of a prefill, a decode or an
encode phase, answered `Busy` meanwhile, is then decided by the one engine over worker lists.

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

One more point under this decision. A document writes overload thresholds for each binding, and
a worker may state its own. A worker object holds one set. **Recommendation:** one set for each
worker, the worker's own values over one default of the program, as the classic policies have
it. A binding keeps whether its thresholds veto at all and whether a request is shed when every
worker is vetoed. Under the one engine its two numbers have no reader: the two that a gateway's
routing settings state are already the defaults its worker objects are built with, and the
crate's translation of those settings, which writes one that is off its default onto every
binding today, stops writing them there. A document that names one is then refused when it is
applied; that refusal is planned, not built, and the validator takes such a document today.

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
| 1 | **The policies, proven in the crate.** Each classic policy decided by `PolicyEngine` over worker objects: its sheet replays through that engine, and its bench cell meets classic's count. | Every line of the 14 sheets that record a decision (408 of section 13.1's 482) replays through the kept engine over worker objects. Each difference from classic is fixed or accepted as a named difference. Under decision 3 (a) the 16 `prefix_hash` lines that test a key's tokens replay there with no ring lent, and their proof for a gateway is stage 3's. |
| 2 | **The host contract.** `place`, `Placement`, the engine's own upkeep and membership events (section 8.2). | A test host in the crate serves every plan through `place` alone. |
| 3 | **The gateway, one change** (section 9), off by default. | The four points of section 9.2's bar; under decision 3 (a), the 47 `prefix_hash` lines and the 12 ring lines of `all_vetoed` read as recorded, count for count, through the gateway's own entry with the engine on and the recorder's own workers and prompts; a cluster run beside classic. |
| 4 | **The controller on the one engine** (section 10.2); the old engine is deleted. | The controller's own tests pass on `PolicyEngine`; the store-backed engine is gone; each of the 74 lines of the four host sheets (section 13.1) is answered by the host's code that the controller takes from the worker crate, or by a rule of the crate's with its differences named. |
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
