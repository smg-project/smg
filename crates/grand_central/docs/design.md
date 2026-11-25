# Grand Central: design

How Grand Central is built: its parts, the data it reads and keeps, and what happens to a
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

Paths without a crate name are under `crates/grand_central/src/`.

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

Grand Central chooses the worker that serves a request.

It is one crate, `smg-grand-central`. Its core is the engine. The same engine is meant to run
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
   and what the four placement functions and the load guard cost for carrying an answer of either
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
  |  smg-grand-central                                                    |
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
| `store/` | 29,896 | 28 | The controller's in-memory store: worker records, leases. Its cache-event tables, counted here, have moved to the worker host since. Its load rows, counted here, have left since. |
| `service/` | 80,546 | 59 | The controller: the gRPC service, election, replication, discovery. |
| `worker_host/` | 18,895 | 19 | What lets a controller hold worker objects of the worker crate. |
| `load/` | 5,842 | 10 | A worker's load report as a stage reads it. The controller's own load poller, counted here, has left since. |
| `plugins/` | 16,296 | 21 | Runtimes for user stages in WebAssembly and over gRPC. |
| other | 498 | 4 | `lib.rs`, the binary's `main`, two small files. |
| **Total** | **286,662** | **255** | |

The engine (section 5.7) is `engine/in_process.rs` and the folder beside it: 33,581 lines with
their tests. The record writer and its tests are 3,354 more.

## 3. The policy document **[built]**

A policy document says how workers are chosen. It has a generation number. A new document
replaces the old one whole, under a number above the old one's.

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
    pub sent_ttl_ms: Option<u32>, // how long a send stays in the count of sends the profile reads
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
- In the encode mode of many routers, the controller's check refuses the document of both
  families, whatever it refuses. Two such main policies: one that places by a prefix ring the
  view carries no block for (its depth, or any cache boundary of the view, holds no whole block
  or more blocks than a request carries), and one that sets what the build does not run. The
  mode's own document is checked where it is applied, and its refusal still ends a start: sides
  that state no policy run the main one.

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

**A worker's `policy` label [partly].** With the classic policies a registration can name its
model's policy. The engine follows no such label today: the document's builder takes one, no
gateway hands it one yet, and a registration that states one is answered with a warning that it
names no policy (`model_gateway/src/workflow/steps/shared/update_policies.rs`).

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

What is built: the label is one more input of the document built from the routing settings
(`RoutingSettings::policy_document_with_labels`), which takes each labelled model's id with the
name of its policy. A label adds a binding of that model's single requests, ahead of the binding
of every model (section 5.6, step 4). Where the document binds no single request, a label adds no
binding, and the model's single requests stay refused. The default policy's own type gives the
profile the settings state. Any other name gives the binding the settings would write for a
single request under that policy, with the policy at the defaults classic builds it with by name
(`PolicySettings::of_label`). A label whose binding the document cannot hold beside the
settings' own is refused by its model and its policy. The call then answers no document: its host
leaves that label out and calls again, and the other labels build without it. A label that names
another policy for a model whose id is empty or holds a `*` is refused the same way: a binding
reads a `*` at its start or its end as a pattern, and the document refuses one anywhere else.
Classic keeps a model's policy under the id as written, so there such a model follows its label.

One difference from classic stays. Models that name one policy by their labels are bound to one
profile, so a stage that keeps a history of its own keeps one for all of them: the count round
robin draws a new list's start from, and the warm-up's counts of misses and hits with its
shallow mark. A model whose label names the default policy shares the main profile's the same
way. Classic builds a policy object for each such model, so there each keeps its own. A profile
for each labelled model would end the difference.

What is planned: the gateway hands the builder its labels by classic's rule. The registration
that finds the model with none decides, and the binding leaves with the model's last worker. The
name is resolved as classic resolves it, and an unknown name adds no binding.

The gateway then puts the new document in force on the running engine. A document put in force keeps
the stages of a profile that did not change, under the rules of section 5.1: its name stands in
both documents, its stages are built from the same text (the profile's own block, `view` and
`signals`), over one stage table, and each stage built again says of itself what the standing one
says. Such a profile's stages keep what they hold, and a decision under way and a new one run the
same stage objects. A profile that changed, and a new one, start empty, and read the count of
sends that the profiles kept beside them read. So a label's binding resets
nothing of another model's policy where it leaves the document's `view` and `signals` as they
stand. A label whose binding writes into them restarts every profile's stages. A label writes
no signal, and into the view only headers the view does not list yet: a `manual` label its key's
headers, and under the sticky override a label of a policy the override places adds the
override's headers, beside a main policy whose binding lists none (`consistent_hashing`). A
`cache_aware` label also writes the document's `index` block where the settings wrote none.
Where the new document cannot be built, the document in force stays and the registration is
answered with a warning.

That holds where the document is built from the routing settings. Where the operator supplies a
document, and on a controller, the document decides, and a registration whose label names
another policy is answered with a warning. The gateway takes that first step today: with the
engine on, it warns at each registration that states a label.

### 3.4 A validated document

Validation turns a `PolicySpec` into an `AppliedSpec`. An `AppliedSpec` holds the built profiles
and bindings with their stage objects. It also holds what is resolved once from the document: the
block size, the hash depth, each binding's routing-key sources and the signals.

What a policy remembers lives in several places, among them:

| What | Where |
|---|---|
| what a stage remembers | its stage object, inside the `AppliedSpec`. It belongs to one generation of the document, or to several where a document put in force on the same engine object keeps an unchanged profile's stages (section 5.1) |
| what belongs to one worker list (round robin's counter) | the engine's value for that list (section 5.2) |
| the record of sent prefixes | the engine object (section 5.1) |
| a sessions rule's pins | the engine object: one table for the engine's life |

`DocumentManager` (`document/operator/document_manager.rs`) applies, lists and rolls back
generations. It can keep the last eight generations in a folder, one file each, and roll back to
one of them.

A gateway reads the supplied file by the crate's one read of a document (`Format::read`), the read an apply and a controller's file pass. A file named `.json` is read as JSON, any other name as YAML (`Format::of`). Either form is read into values first, and the document is built from those values by one typed read. So a YAML document reads as JSON does: text that looks like a number, a boolean or a null is written in quotes, a key with nothing under it is refused where a list or a map stands, a list item with nothing in it is refused in a list of the spec's own, and a tag is refused. A number reads one way in both forms and in every build: a whole number of 64 bits as itself, and any other as the nearest double. The crate's manifest asks the JSON library for its exact reading (`float_roundtrip`). A field of single precision takes the number by a cast, so one that rounds past the largest single reads as infinite there and the spec's check refuses it by its field. A key written twice is refused by name in both forms. Three numbers read apart, by the forms' own rules: a zero written `-0`, a number past a double, and a YAML whole number in hex, octal or binary. A refusal names the field by its path, or the line and column where the file itself does not read.

### 3.5 The stage interfaces

A filter, a scorer and a picker are three traits (`engine/framework.rs`). Each has two required
methods, shown here. The others have default bodies, and all but one are not shown: most are
switches and short paths that the built-in stages use.

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
    // a gate of the scorer's own weighs load against cache affinity
    fn balances_load_against_affinity(&self) -> bool { false }
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

**[built]** One default method is shown, `balances_load_against_affinity`: a scorer of a program's
own that weighs load against affinity must answer it, because a pair's lift rests on it. A scorer
that answers yes places a request by who holds the prompt's beginning, and a gate of its own bounds
where it sends the request by the workers' load: `prefix-affinity` moves a request off that worker
once it is loaded well past the others. Under a binding that steers, a pair's prefill profile with
such a scorer is decided with the overload filter lifted, so a prompt stays on the prefill that
holds its beginning while that worker is over its thresholds. The lift rests on that answer alone,
as classic's rests on its policy's answer of the same name:
`LoadBalancingPolicy::balances_load_against_affinity` (`model_gateway/src/policies/mod.rs`), read
where `select_pair` places a pair (`model_gateway/src/routers/common/placement.rs`).
`prefix-affinity` and `warmup-slice` answer yes, by their spill gate. What a stage reads says
nothing of a gate: a stage that reads the cache records and answers no keeps the filter for its
profile, a scorer a program adds answers for itself, and a plugin's stage answers no. A filter or a
picker has no such word: one that reads the cache records keeps the filter for its profile.

**[built]** Such a stage keeps building when the crate grows. For each type below, the table says
whether it is marked non-exhaustive and how code writes it:

| What | Types | Marked | How code writes it |
|---|---|---|---|
| the events a stage hears | `Signal`, `Membership`, `ReleaseReason`, `TransportClass`, `EventKind` | yes | a `match` on one keeps an arm for the rest, and a case with named fields is matched with `..` and made by its call |
| the answers a stage gives as an enum | `ScoreSpread`, `ScoreFront`, `DrawUnit` | yes | as the events are |
| the records a caller builds | `Needs`, `Outcome` | yes | a stage writes its `Needs` as one chain from `Needs::NONE`, a call a flag. A host makes an `Outcome` by `Outcome::reported` |
| the records a stage answers with | `Shortlist`, `HeadRule` | yes | each is made by its call |
| the view a stage only reads | `RequestFacts`, `Candidate`, `Constraints`, `PoolFacts`, `LoadRow`, `AffinityFacts`, `AffinityRows`, `AffinityHolder`, `Health`, `MediaCounts`, `Header`, `ZoneCapacity`, and its enums `HashKind`, `SignalSource`, `OverloadRegime` | yes | code outside the crate starts a record from its `Default`, or a `Candidate` from `Candidate::ready`, and assigns the fields it knows. It never matches one whole, and a `match` on one of the enums keeps an arm for the rest |
| the parts of a decision, of a lease and of a profile | `Target`, `Pair`, `Trace`, `Shed`, `Queue`, `LiftedVeto`, `DemandFacts`, `Lease`, `LeaseKind`, `Dispatch`, `WeightedScorer`, `Profile`, `Binding`, `PairPools`, `EncoderFacts`, `EncodePool`, `EncodeItems` | yes | code outside the crate starts a record from its `Default` or from its call (`Target::of`, `Pair::of`, `PairPools::of`, `WeightedScorer::of`, `Profile::of`, `Binding::of`) and assigns the fields it knows. A `Trace`, a `Shed` and a `Queue` are answers: a caller reads one and builds none. It never matches one whole, and a `match` on `LiftedVeto` or `LeaseKind` keeps an arm for the rest |
| what no caller builds or matches whole | `Pick`, `Overlap`, and every other public enum or record that no program outside the crate builds or matches whole | yes | a caller reads one by its fields, and starts one from its `Default` or its call where it has one. A `match` on one keeps an arm for the rest, and a pattern on one ends with `..` |
| what a caller builds or matches whole | the lists a request lends and its `Demands`, the settings, a held decision as its guard lends it, a lease as a store holds it, a binding's sessions rule, a pool's word, a store's checkpoint | not yet | each takes its mark with a call to start from, and its module or its own declaration says so |
| the enums closed by design | `Decision`, `ShedReason`, `Side`, `Phase`, `Role`, `Said`, `Pairing`, `Head`, and the client's `Answer` | no | a caller answers every case of them, and a new case must stop its build |
| the records closed by their shape | a record of one value: `WorkerId`, `LeaseId`, `GatewayId`, `TieBand`. A record with no field, as the stage `Best` | no | its one value, or its name, is the whole record |
| a record closed by its own doc | `DispatchCounts` | no | a further number goes on `CreditedDispatches` |
| an enum with no case | `DiscoverySettings` in a build without discovery | no | it has no value, and a caller may match it with no arm |

The crate's lint lines hold the table. `clippy::exhaustive_enums` and `clippy::exhaustive_structs` are
on at the crate's root. A public enum, or a public record whose fields are all public, fails them
unless it carries the mark or says why it does not: at its declaration where it stays closed, and
where it waits for its call at its module's head or at its own declaration. The store's modules that
leave the crate, and its index of
cache events, take no mark: each says why at its head. A module that says so at its head quiets its
lint for every type in it, so a mark inside such a module is held by review until the module's last
waiting type is marked. No lint reads a case's own named fields: beyond the events above, such a case
carries no mark yet.

## 4. The data the engine reads

Rule 2 says the engine keeps no copy of worker data. This section says where the data lives: in
structures of the worker crate and of the kv-index crate. The Grand Central crate adds structures of its
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
new list and a new `G` for each model the worker serves, and for each model it leaves, and readers
get the new value whole. A published list is never changed.

The gateway's `G` holds the model's workers and its lists by role and protocol (prefill workers
over gRPC, regular workers over HTTP, and so on). The controller's `G` holds its lists by role
alone, with the workers of every protocol in one list.

The controller's holder lends each role's list with its cell and with what its rows state of each
worker (`Objects::lent`).

Beside a pair's two lists it hands their pair classes (`Objects::pair_classes`): each worker's
class by its place in the list as lent, under the store's own pairing rule over what the rows
state. On a wire that carries one runtime's request, a pair with a worker whose runtime is
unknown goes behind every other pair: the rule states a prefill's partners in those other pairs
as its first partners. So a controller makes a pair on one known runtime ahead of a pair with
such a worker, as a gateway does on that wire (section 9.2). The classes hold a model's workers
of both wires in one table, so a pair of the HTTP wire is a first pair too: no order stands
between the wires, and a frame decided over both has no twin in a gateway's own placement,
which places each request on one wire. A row whose facts change gets a new worker object, so
the registry builds a new list and a new `G` for each model the worker serves, and for each model
it leaves. The classes are
derived once for a `G` and a pairing mode, at the first ask, and stand with it.

A controller knows no admission window of a gateway. So it reads a pair's admission open
(`OpenAdmission`), as the rows its store builds state it. The gateway that asks is to gate
capacity when it commits **[planned]**: no gateway calls a controller yet (section 11).

All of it stands in a build with the `worker-host` and `in-process` features, as a controller's
is. A controller's decision of a pair, at its prefill and at its decode call, reads it (section
10.1).

Beside its encoder list and its prefill list the holder hands the two columns an encode request's
decision reads (`Objects::encode_columns`). They are what the encode side reads of each encoder
and the encode class each prefill names, by a worker's place in the list as lent. The classes are
the engine's encode rule (`EncodeClasses::derive`) over what the rows state. The columns stand in
the same build. A controller's decision of an encode request that carries its items reads them
(section 10.1).

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
record, and, on a host that follows a controller, by its worker host, which lays each status it
is told. A controller's standby is told by its leader's log. A gateway's watcher
(`service/watcher.rs`) is told by its watch of the controller's workers, and no gateway runs it
on main (section 11). The liveness flag
is also written on the word of the stream a worker pushes on (section 4.3), of a passed probe
and of a request's own token stream.

A decision reads a worker's routing state in one read: `Worker::routing_state()` gives ready,
circuit breaker, stalled, overloaded, the in-flight count and the count of requests processed.

### 4.3 The load reports **[built]** (`smg-worker`)

```rust
// crates/worker/src/load/state.rs
pub struct LoadSnapshot {
    entries: HashMap<Arc<str>, LoadEntry>, // worker URL -> its last load report and when it was taken
    kept: FxHashMap<Arc<str>, LoadEntry>,  // worker URL -> the last report of one whose load call fails
}
pub type LoadReceiver = watch::Receiver<Arc<LoadSnapshot>>; // how a reader follows new snapshots
```

The load monitor polls each worker (every 10 seconds by the gateway's default) and takes the load
records that workers push. A worker whose pushed records are fresh is not polled. A new snapshot
is published after each poll, and for pushed records at most once in 20 ms. A snapshot is never
changed after it is published. Where a worker's load call fails, the next snapshot holds no entry
of it, and its last report is kept beside the entries, for `LoadSnapshot::report_or_kept` alone:
where it states a running or waiting request, and no further than the first poll that finds its
sample 30 seconds old (decision 12).

**Who opens the stream.** A worker pushes on a stream that its host opens to it. The worker
crate's collector connects to the worker and subscribes, and the worker answers on that stream
(`crates/worker/src/kv_events/`). It sends its cache events with the load records that ride
them, or its load records alone, as its host asks. A gateway asks a worker for its cache events
where what places the worker reads them. Otherwise it asks for the load records alone, and opens
no stream where its load monitor wants no loads of the worker. A controller asks a worker for
its cache events where the worker's registration asks for them, or, for a worker its own
discovery found, where the document in force at the find reads cache events for the worker's
first model. Otherwise it asks for the load records alone, on every replica: its load monitor
wants the loads of every worker the replica dials. It asks for an engine's cache events on one
stream: a data-parallel engine's rank 0 streams every rank's events. For each rank above 0 it
opens a stream of load records alone, and takes that rank's records from it. The collector
hands the load records to the load monitor and the cache events to the index of cache events
(section 4.5). Only a worker dialed over gRPC has such a stream; the monitor polls the others.
A worker dials nobody for this, so it needs no address of a gateway or of a controller.

The host that opens the stream is the process that holds the workers' data (the table of
section 1): the gateway in classic and with the engine in the gateway, the controller with the
engine in a controller (section 10.2). So a worker's data takes one way in every mode. Only the
process that opens the stream changes. A controller's collector hands its load monitor the load
records while the replica leads, and a standby drops them (section 10.2).

### 4.4 The in-flight counters **[built]** (`smg-worker`)

A route creates one `WorkerLoadGuard` for the worker it chose. The guard adds one to the worker's
in-flight counter, and to its routing key's counter. It lives as long as the request is on that
worker. At its drop
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
(67,108,864 for a model by default: tokens, or for text its characters plus 1024 for each node
a worker stands on) by an eviction pass every 120 seconds.
Between two passes it grows past the bound.

A request's prompt is stored under its **cache partition** (cache salt, extra value, adapter), so
that two tenants' prompts never match. The gateway derives the partition once for each request.

**The engine today [built].** It reads record A through two small traits that the host
implements over its own index (`CacheIndexes`, `CacheIndex`, in `engine/in_process/overlap.rs`).
For what it sent itself it keeps a record of its own: a hash table behind one lock for each model
(`tracking/prefix_predictor.rs`), not a tree. That record has three faults, each seen on a cluster
(section 13.3):

1. It holds at most the first 256 blocks of a prompt (4,096 tokens by default). A match needs a
   0.3 share of the prompt, so past 13,653 tokens a prompt stops matching.
2. Its write waits for no lock. On the cluster a write that met the lock held was lost, so writes
   were lost when many requests arrived at once. The deciding thread now keeps what such a write
   has left and writes it on behind a later pick of its own. Until that stands no other thread
   reads it, and a thread keeps eight: what newer ones push out is lost.
3. For a request that has token ids, a profile that reads cache events neither reads nor writes
   it for a worker the index holds a block for. A profile that reads the record alone
   (`source: predictor`) reads and writes it for every worker of its list. So does any profile
   for a request given as text, which asks no index. One limit follows: a profile that reads
   the record alone knows its own sends to a worker with a block, and none that a profile
   reading cache events placed there over the same pool. A pair's decode side and a decode
   request asked alone can be two such profiles. Classic keeps one policy for a model and one
   for each side of a pair, so it has no twin for this limit.

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
- **[built]** A request placed by the one call (`InProcess::place`) hands its entry too. Its
  caller states the entry with the prompt (`Prompt::entry`), and the call's own maker asks it
  where a decision reads the tree.
- **[built]** The tree is written for the worker the request goes to. The tree's walk asks the
  request's maker which target of the pick its caller sends to (`RequestMaker::sends_to`), best
  first, and stores the prompt for the first one: for no worker where the caller sends to none.
  The one call answers by the read it makes of a target itself: ready, and admitted by the
  route's rule. So a call that decides again stores the prompt once, for the worker it sends
  to, and a call shed under the rule's name stores nothing. A maker that does not answer is
  stored for the pick's first target. A pick that a session's pin made ahead of the walk stores
  nothing where a worker's leave was told in between: its worker may be the one that left.
- **[built]** The one call takes a target by the answer the tree's walk was given on it: a
  worker's state is read, and its route's rule asked, once a decision. So the tree names the
  worker the prompt went to. A worker that stops being ready behind that read is still sent to,
  as under classic, which also stores its selection ahead of the send. The hash table is written
  by the same answer: for the first target its caller sends to, and for none where it sends to
  none.
- **[built]** A request given as text is decided the same way in the string tree its host lends,
  where the host names the request's entry by its text (`RequestMaker::text_key`). The tree is
  asked for whatever the model's index holds: an index speaks of token blocks. A match is
  counted as the bytes of text it holds, in the request's windows. **[planned]** The gateway
  lends its string trees.
- **[planned]** The hash table leaves the single decision. Until then a request with token ids
  of a model with an index of cache events reads the index, and the hash table for the
  workers the index holds no block for, as today. A request given as text asks no index: it
  reads the hash table for every worker of its list, and its pick is written there for the
  worker it goes to, whatever the index holds for that worker. So does a request with token
  ids under a profile that reads the hash table alone (`source: predictor`).
- **[built]** A prefill-decode pair decided over worker lists reads the host's index of cache
  events and the hash table for each side, by the side's own list: the index, where the request
  has token ids, for the workers it holds a block for, and the hash table for the others. A
  request given as text reads the hash table for every worker of each side's list. So does a
  side whose own profile reads the hash table alone, whichever profile a request of its phase
  asked alone resolves to. Behind
  its pick it writes
  the first target its caller sends to into the hash table, a side at a time, where the index
  does not speak for that side's worker: it speaks for none on a request given as text, and for
  none to a side whose profile reads the hash table alone. An encode
  request's first decision does the same for its pair. It writes neither side where its caller
  cannot send to an encoder the decision placed (`RequestMaker::sends_to_encoder`). Classic keeps
  trees for each side. **[planned]** The engine reads the host's trees for each side.

A decision of the engine publishes no insert to other gateways.

**What the workers send and what they do not.** Only record A comes from the workers: the cache
events they push fill the index and nothing else. Records B and C are the deciding process's own
account of the prompts it sent, written by its policy at each decision. No worker pushes them and
no worker's stream carries them. In classic the gateway's policy keeps them. With the engine in
the gateway the gateway keeps them and lends them to the engine. On main it lends the token tree
alone, to a single request, under a document that reads the prefixes sent to each worker (section
9.2). A single request with token ids is decided inside that tree, by classic's own key. That
holds where its model has no index of cache events and its profile reads sent prefixes and names
the holders' rows. A request given as text names no entry and still reads the hash table, for
every worker of its list whatever the index holds. A
request with token ids of a model that has an index still reads the index,
and the hash table for the workers the index holds no block for:
for every worker under a profile that reads the hash table alone.
A prefill-decode pair is lent no tree: each of its sides
reads the hash table, and the index where its model has one, by the side's own list, worker by
worker as a single request does. An encode
request's pair reads them the same way. A controller is to
keep trees of its own **[planned]**, and section 11 says what its contract must hold for that. On
main a controller's engine reads its hash table, and the index of cache events that its worker
host keeps (section 10.1).

One change was made outside the Grand Central crate first **[built]**: classic's trees moved out of
its policy object into one store type of the kv-index crate (`ModelTrees`), with no behaviour
change. Classic's policy object keeps its trees in that type. The gateway keeps the token trees
it lends the engine in the same type **[built]**.

### 4.6 Who writes and who reads

| Structure | Written by | Read by classic | Read by the engine |
|---|---|---|---|
| a model's worker list | the registry, at each membership change | yes | yes, the same list |
| the worker object's state | health checker, load monitor, load guard (on a controller, the leases on file), outcomes | yes | yes, at each decision, and once more for each worker of a target the placement call tests behind the pick |
| `LoadSnapshot` | the load monitor | `least_load`, `cache_aware`, `power_of_two` | yes, the latest snapshot; a controller's deciding thread gives it with each request, where the document keeps no report |
| the index of cache events | the workers' event streams | `cache_aware` | yes, through `CacheIndex`; a controller lends its engines the index in its worker host's KV tables (`worker_host/kv_read.rs`) |
| the two trees | the deciding policy: classic's, or the engine inside the tree's walk | `cache_aware` | the host's own trees, lent: the token tree **[built]**, the string tree **[built]** in the engine, its lending by the gateway **[planned]** |

## 5. The engine

The engine is in `engine/`.

### 5.1 The engine object **[built]**

A gateway has one engine object. It is built once, from the policy document in force at its start,
and holds the document's stages and what they remember. A later document is put in force on the same
object (`InProcess::apply`): the engine holds its document as one value and replaces it whole,
and a decision reads one document from its entry to its end. The session table and the record of
sent prompts stand across the call. So do the stages of a profile whose name and text did not
change, with what they remember, under the rules that follow. A new document that changes `view` or
`signals` builds every profile anew. Nothing is kept unless both documents were validated over one
stage table: the stock one, or one table of the host's own that took no stage between the two
validations.
A profile is built anew too where one of its stages, built again, says otherwise of itself than the
standing one: another name, other words of a request's life to hear, or anything else the engine
reads of a stage when it builds a profile. A factory of the host's own may build such a stage.
A count of sends stands where a profile that reads it is kept: a profile built new beside it reads
the same count, and a send is counted once. A count none of whose readers is kept is built new, and
takes over the sends of a standing one (`Engine::succeed`). Profiles whose
stages read a count under the same `sent_ttl_ms` and `sent_report_delay_ms` read one count, and a
document that states neither, as the one built from the routing settings does, puts them all under
one. It holds no worker of its own: the registry that its host hands it (section 8.1) holds them. A
controller builds a new engine object for each document it applies. That object keeps the stages of
an unchanged profile from the object before it, by the same rules (section 10.2).

A word of a lease issued under an earlier document is told by one rule, in a gateway's own process
and on a controller: the stages kept since the lease's document hear it, every count of sends in
force takes a dispatch, and a stage built since hears nothing. The session table hears it where
the table stood at the lease's issue. A gateway's engine keeps one table for its life. A
controller's table is the new engine object's own: it starts empty and hears nothing of an earlier
lease, whose first decision claimed in the table that left (section 10.2).

An entry of the engine reads the document through a reference its own thread keeps. The engine
holds the number of the document in force in one atomic word, written behind the document at an
apply. A thread keeps one counted reference of the document it read last, and an entry compares
that document's number with the engine's. Where they are one, the entry takes a plain count on the
thread's reference: no atomic exchange, and nothing allocated. Otherwise it reads the engine's cell
and the thread keeps what it read: at a thread's first entry, at its first entry behind a new
document, and at each entry of a thread that asks two engines by turns. So a replaced document
lives until its last decision ends, its last handle is let go, and each thread that read it last
asks an engine again or ends: at most one replaced document for each thread. The stages that the
document in force did not keep are dropped on the thread that lets the last reference go: as a
rule at that thread's next entry of an engine (a decision, a lease's word or a read of the
document), or at its end. A drop that panics there unwinds out of the entry. The process aborts
where the entry is made while the thread already unwinds, as a request's guard dropped with a
report owed speaks for its lease, and at a thread's end.

```rust
// engine/in_process.rs
pub struct InProcess {
    document: ArcSwap<InForce>, // the validated document (profiles, bindings, stage objects and their state), its
                          // number, what is read of it once (which lease events it takes) and its
                          // shadow counts
    applied: AtomicU64,   // the number of the document in force: an entry compares it with the number of
                          // the document its thread read last
    clock: DecisionClock, // one clock for every decision and every load report's age
    leases: OwnWord,      // the last lease id taken, by a pair or in a thread's run: one atomic counter
    sessions: Arc<SessionTable>, // the pins of every key a sessions rule placed: one table for the
                          // engine's life
    affinity: Affinity,   // cache affinity: the host's index of cache events, and the engine's own
                          // record of the prefixes it sent, one for each model
    epochs: AtomicU64,    // how many list values were made
    held: Arc<HeldFaults>, // the panics the engine held, three counts: decisions that panicked, lease events
                          // and membership changes that panicked inside a stage, and steps of the engine's
                          // own task that panicked (the upkeep, the registry's follow, a telling of load
                          // reports). A controller's store hands every engine it builds one set of them
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
- the id the host knows the worker by, where the host keeps ids of its own: one for every worker
  of its lists or for none, and no id twice in a list. A debug build checks each list on its own
  where it fills the cell;
- what the worker states for a request's hard demands, which stands in a column beside the
  records.

`PublishedList` is a list, its cell and its facts as one value, and lends the three together. A
host builds one with the list at hand. One built with facts reads its workers there. So a host
whose registry makes a generation's value under its own guard builds it behind the guard, at the
list's first lending, in a once-cell of that value, and a lending reads that one cell. The
controller's holder keeps each role's list so.

The fill makes the list's records.

```rust
// engine/in_process.rs
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
live half is read from the worker object once for each decision. A thread keeps the rows its
decision of a single request wrote whole for a list. Where its next such decision would copy from
the same rows, and reads the same session counts, it writes the live half over them and copies
nothing. The load report comes from the
load snapshot, a kept report included (decision 12), and is as old as its sample. A snapshot's
reports are converted once for a list and cached. Until the list
holds them, a record written alone takes its report by one read of the snapshot, which looks
among the kept reports too where it holds no entry of the worker: a drawn
worker's, a worker's that a routing key names, and the held prefill's of a decode decided again.

The field `engine` numbers the list's data-parallel engines: records that state one number are
ranks of one engine. `InProcess` writes the numbers once for a list, where it finds the list's
engines. A default record leaves 0 on every record, which would read as one engine. So the field
tells engines apart only where the records' maker numbered them. `InProcess` does, in process
and in a controller. A host that writes its own records numbers them itself, as the crate's
example does (`crates/grand_central/examples/in_process/main.rs`). The number holds within
one generation of one list and is no id beyond it.

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
    pub decode_ring_positions: Option<&'a [u64]>, // a pair's decode side's own places, where its two sides read the ring by two words
    // six more fields: `affinity`, `named`, `signal_cells`, `session_counted`,
    // `session_key_from_rid` and `model_fingerprint`, which 5.8 removes
}

pub struct Constraints<'a> {
    pub excluded: &'a [WorkerId], // workers this host just failed to reach
    pub wire_modes: WireModes,    // the protocols this host can dial for the request
    pub overload_lifted: bool,    // the overload veto does not apply to this decision
    pub decode_later: bool,       // a pair's decode will be chosen after the prefill has run
    pub counted_on_decode: bool,  // the host counts the request on its first decode at a decode asked again
    pub subset_hint: Option<u32>, // decide over at most this many workers
    pub demands: Demands,         // what the request needs of its worker
    pub stalled: &'a [u32],       // positions the host itself holds back
    pub held: &'a [u32],          // those of `stalled` that no rule may lift
    pub places: Option<&'a [u32]>, // the places of the list the request may take; `None`: every place
}
```

What is not built in the engine that stays: a request that states a subset hint, or protocols
that not every worker of its list serves is refused. The refusal names the input.
A pair's demand is kept a side at a time: each side of its two lists is held to the
workers that meet it as that side's role is held to it, and a side it leaves nobody sheds the pair
under that demand's stage, the prefills' first.
An encode request's demand is kept the same way for its pair, a side at a time
(`EncodeLists::decide`), and by the encode pass for its encoders. An
encoder takes the media of a request that carries items. So the encoders are held to the runtime,
to vision and to the references, and each side of the pair as a pair's side is, less vision and
the references (`Demands::beside_encoders`). Beside the pair an encoder is held to no window and to
no way of deciding a constraint (`Demands::beside_pair`): it runs neither the prompt nor the
completion. An encode request that stands alone has no pair beside it, and its one list is held to
the whole demand. A request with no item takes no encoder,
and its pair's two lists are held as a pair's own. A side the demand leaves nobody sheds the
request under that demand's stage, the prefills' first. No encoder is cut ahead of the encode
pass. The pass keeps the encoders' part of the demand in its one test of an encoder, a part at a
time, against what each encoder states beside its record (`EncodePool::demands`): the runtime is
the rest of the request's wire, the references and their pipeline stand with the items'
references, and vision with the request's pixels. An encoder that fails a part is no member of
its class. So what the encoders lack is named in the pass's own order, as classic names it,
whether a request states a fact as a demand or as its own: an encoder that would take the
request but for its images (`encode-vision`), a pushing encoder that takes the media
(`ec-compatible`), the media (`encode-media`), the encoders (`encoders`). No encoder is shed
under a demand's own stage. Three answers for the encoders stand ahead of a side's stage: a gap
of their membership, encoders none of which is open, and a saturated fleet. A
host's places for the encoders stand beside a demand as beside none: the pass reads every
encoder for what the model takes, and the places for what the request may reach. The
encoders' `places` answer waits behind a side's stage, as it waits behind a gap of the pair's
membership.
A request with items that demands vision or a way of processing its
references leaves that part of the demand to its encoders. A decode asked again takes no items
and would hold the decode to the whole demand, so the decode of such a request is not asked
again: the entry declines one that states the later decode under `decode-later`, and a placement
books its decode with its prefill.
A pair's encode items are decided over a host's three lists, with the pair's lease issued
(`InProcess::decide_encode_at`). The trait has no method for them: the one it had took no
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
request is shed under `places`: it is held off every worker that takes it. The places are its
host's, or the ones its demand leaves. No gap of the membership
says that, and the gaps stay read over every worker. A gap of the pairs' membership the encode pass
reads (`ec-compatible`, `decode-vision`) is answered ahead of every list's `places`: no other place
closes it. What the pair's own decision answers stays
behind the pass, under places as under none. Places that name no worker of a list are no such
case. Such places of a side shed the request as they shed its pair alone, and such places of the
encoders shed it as a list with no open encoder does.
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
    pub survivors: Survivors,              // how many workers passed the filters, or none behind a draw
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

The host calls `InProcess::decide_at(maker, clock)`. `maker` is the host's object for the
request (section 8). The call runs on the caller's thread.

The engine waits on no lock of its own, but for these. Under a document that keeps a sample of its
decisions (decision 7), a decision that is kept takes the one lock of the sample's ring for its
push. A decision that finds a list no lookup table for cache affinity (5.2) reads one, and a table
stands for a second before one decision reads it anew. Such a read takes the lock of the engine's
map of the records of sent prefixes, once at most. A stage that keeps state takes its own short
lock (one shard of the session table or the record of sent prefixes, among them). A decision that
meets the one build of a list's value, or of a part of it, waits for it (5.2).

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
   | round robin | It takes the next turn of the list's counter and takes the worker it lands on if that worker is eligible. Where it is not, the engine reads the list once and the rotation picks among the eligible workers. A thread's first turns of a list are single steps of the counter, 64 for each worker; past them the thread takes 61 turns at once (67 where 61 divides the list's size, one where 67 does too) and plays them in order, for four lists at a time. |
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
   workers named before the pass, unless it is named no row, as one that shows its rows to a
   plugin is: such a profile is handed a column that says of each worker what it holds. After the
   pass the prompt is recorded for the first target the request's caller sends to, and for none
   where it sends to none (section 4.5).
7. **Records.** One `Candidate` for each worker is written into the thread's buffer. Over a list
   of 16 workers or more, a profile whose picker draws first (`random`, `round-robin`,
   `passthrough`, power of two, `best` with a sample) has only the drawn workers' records written.
   A request whose session key holds no pin is decided on a draw too, where its sessions rule
   leaves a new key to a profile whose picker takes turns and whose filters are one test of a
   worker's record. The worker the list's counter points at has its record written first, with no
   turn taken. Where the filters keep that worker, the workers the draw then takes have theirs
   written, and so has a worker that another request claimed for the key meanwhile. Where they
   refuse it, or the worker the draw's own turn lands on, every worker's record is written
   (decision 15).
   For a request held to places no worker is drawn first: every worker's record is written.
   Under a binding that names a shadow every worker's record is written, for the two decisions.
8. **Filters.** Each filter removes records. If a filter leaves nothing, the binding's rule for a
   list wholly over its thresholds may still pick among the workers the overload filter removed.
   Next the rule may pick among the workers their liveness flag keeps out, whether or not the
   binding sheds. A binding that sheds keeps the overload filter in that pick, unless the host
   lifted the overload veto for the request. The rule lifts nothing harder: a veto the host
   holds (`Constraints::held`), an open breaker, a worker that is not ready or is draining.
   Otherwise the decision is a refusal that names the filter: `Saturated` for the overload filter,
   `NoEndpoints` for any other. An encode request's encoders enter no such rule. Where the
   overload filter leaves none of them, the refusal is `Saturated` under a binding that sheds
   and `NoEndpoints` under `encoders` under one that steers. Classic answers a client the
   same way: its overload shed with the shed option, and no available worker without it.
   A single request with a routing key keeps its key's worker over
   the overload veto, beside free workers too, where its binding does not shed, the plain ring
   alone places its profile and every worker it may take is a single worker. A sessions rule adds
   conditions of its own. Over workers of another role the veto stands, and the key passes on to
   the next worker its ring reaches.
9. **Scorers and the picker.** Each scorer scores the survivors. The scores are combined by the
   document's weights. The picker orders up to four targets. No target is a refusal,
   `PolicyDeclined`, naming the picker.
10. **The lease.** If the decision is a pick that has a lease (section 6), the engine issues it and
    tells the stages, unless a sessions rule keeps the pick from them.
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
A stage that panics at a word of a lease, its issue among them, or at a change of membership is
held and counted apart: the request keeps its decision and its lease, and a registration, a
removal or an apply goes on. A gateway shows both counts and the count of its upkeep's panics
at each scrape of its metrics endpoint, as the series `smg_policy_faults_total` by `kind`:
`decision`, `event` and `upkeep`. A controller shows the same series on its metrics page,
counted over every engine its store built, so a count outlasts the document it rose under. Its
`event` is a change of membership alone: a controller holds a stage's panic at a word of a
lease under catches of its own, and counts it in series of its own. Where such a panic unwinds
the decision that told the word, as at the issue of a single's or a pair's lease, the request
is answered with no decision.

A request whose host hands no list (`WorkerLists::None`) is decided over no worker: it is shed
with no endpoint under the stage `none`, and its trace names no profile. A controller hands none
for a model no worker serves, and for a gateway that dials no worker of the model's list.

What one decision writes:

- the thread's own buffer;
- on the short path under `round-robin`, one atomic add on the list's counter;
- one atomic add for the lease id of a pair, an encode request's pair too, or for each run of 64
  ids a thread takes for single requests, where a pick has a lease (section 6). A thread takes a
  new run too where it issues a single request's lease under another document than its run was
  taken under, and the ids left of its old run name no lease;
- a session pin, where a sessions rule decides;
- the sample's ring, for a decision that is kept (decision 7);
- the record of sent prefixes, where a cache-affinity profile decides;
- once for each new load snapshot and list, the converted load reports, which allocates.

On the short path a decision allocates only for what is kept for later decisions, among it:

- the calling thread's reference of the document in force, where the thread kept another document
  or none;
- a list's value, where the decision asks the list ahead of any task of the engine (section 5.2);
- the model's name, which a list's value keeps beside the profile one pick decides;
- the calling thread's list of the eligible workers, where the engine reads the list once (step 1);
- the sample's ring, at the first decision a document keeps (decision 7).

Under round robin not every decision writes the list's
counter: past a thread's first turns of a list, 64 for each worker, the thread takes 61 turns of the
counter in one write (67 where 61 divides the list's size, one where 67 does too), and between two
such writes a pick whose turn the filters keep writes the thread's own run alone. A thread keeps
such runs for four lists; a fifth writes the counter on every pick until one of the four places
goes unused.
A draw that looked up load reports counts them toward its
list's conversion where it takes the list's claim, which it tries and never waits for.

### 5.7 One engine **[partly]**

Main has one engine, `InProcess`, behind the trait `DecisionEngine`
(`engine/decision_engine.rs`). It decides over its host's worker objects, read in place. It keeps
no lease: the host hands the lease back with each event. This document also calls it the kept
engine, its name while a second engine stood beside it.

A second engine stood beside it: the store-backed engine, also called the old engine here. It
decided over records copied into the policy store and kept its leases there. Its type is deleted,
a step of stage 4 (section 16). The store's copies of the worker data that only it read are
deleted in the steps behind it (sections 10.2 and 10.3). A controller decides a pair at its
prefill, its decode call and an encode request that carries its items over worker lists too
(decision 5).

The types the two engines shared stay where they were moved ahead of the deletion: `Request`,
`DecideOutcome`, `DecisionRefused` and `DecodeRefused` in the trait's file, and
`Said` (section 6) beside `Lease`. `GatewayId` is still defined in the store.

### 5.8 What changes in the engine **[partly]**

- It honours what it once refused: an encode request's pair held to some workers of its two
  lists, and its encoders held to some of theirs (`EncodeLists::decide` decides the request
  among the places its host states for each of the three lists, section 5.4, which is
  built); an encode request's demands, its pair's kept a side at a time as a pair's are kept
  and its encoders' by the encode pass, in classic's order of refusals (section 5.4), which is
  built.
- The request fields that only the old engine's host fills leave with it. `model_fingerprint`
  has yet to go.
  Another, `session_counted`, stays, which is built: by it a host says that somebody counts a
  key's requests on each worker, and a sessions rule with a cap asks nothing more of it. Where
  that count stands on a worker object is the request's maker's to say
  (`RequestMaker::session_counted_under`, a provided method): under the key's text for a host
  whose own load guard counts, and under the session's number for a controller, which counts
  by the leases it holds (10.2).
  `Constraints::pins`, the session pins a host kept outside the engine, is removed with
  its trait `PinSource`: no host lent any. A session's worker outlives a document on the same
  engine object, which is built. The call that puts a later document in force there shares
  the engine's table with the new document's engine (`Engine::share_sessions`, section 5.1).
  A controller builds a new engine object for each document, and its pins start empty
  (section 10.2).
- The trait's method that the kept engine answers with a constant, `decide_encode_pair`, is built
  or removed. It is removed, and the guard's maker of that name with it, which is done: a host
  asks the engine's own entry with its three lists (`InProcess::decide_encode_at`) and holds
  the answer (`DecisionGuard::holding`).
- The record of sent prefixes leaves the single decision. The two trees a request is then
  decided in are the host's own, lent (section 4.5).

`signal_cells` stays, and it is built for a single request: where the request's profile reads a
named signal, the engine fills the field itself, from the reports of the load snapshot the
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
exists only where the document takes a lease event. Where a stage or a count of sends takes one,
every pick has a lease. Where the session table alone takes one, a pick the table is told nothing
of has none, a request with no session key among them. The short path is the exception: its pick
has a lease wherever the document takes a lease event (section 5.6). A pair's pick always has one.
On a controller every pick it answers has one on file: the leases on file are its count of
requests in flight (section 4.2).

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
                                         (a controller releases one on its file 5 s after issue;
                                          a pair's decode side behind its prefill, 5 s after
                                          that prefill's end)
```

A controller's deadline is the time to name a lease's first dispatch. A pair's decode side that
was issued with its prefill is sent when that prefill has answered. So the file holds it while
the prefill is dispatched, as its booking stands in a gateway's own process, and it has a whole
deadline from the file's first pass behind the prefill's end. A side a decode call issued keeps
the call's own deadline.

The kept engine stores nothing about a lease. The host keeps what it was handed and hands it back
with each event (`DecisionEngine::say`). The crate has a guard for that, `DecisionGuard`
(`tracking/decision_guard.rs`): it holds one decision's lease, has `dispatched`, `completed` and
`released`, and releases at its drop. A controller keeps its leases in a file. It lends a lease
back from that file in the guard's form, so one rule tells a lease's events under either host
(`lease_events::single`, `lease_events::pair`). The host says who hears. Behind a new document the
stages that document kept of the ones that stood at the issue hear a lease's events, and no stage
it built new: in a gateway's own process and on a controller alike. Every count of sends in force
takes the dispatch (section 5.1). A file meets events no guard says. An end is told once for each
side of a pair that is still live. A release of a decode side behind its dispatch names the worker
that side went to. It states the side's claim only where the side claimed that worker: behind a
dispatch on another worker a release takes back no claim.

What an event does:

| Event | What the stages do with it |
|---|---|
| Dispatched | the session's pin follows the worker the request went to; every count of sends takes it, whichever profile placed the request; scorers and pickers that asked hear it. A pick a sessions rule keeps from the profile moves the pin alone. A pin at its rule's cap moved already, to the worker under the cap that the request which met the cap was placed on: where that request was decided, or for a pair's prefill side where the pair formed (decision 13). The dispatch finds the pin there, or fronts the later target the request went to. A request placed on another worker at the cap moves no pin |
| Completed | scorers and pickers that asked hear the outcome (status, time to first token, tokens in and out, duration) |
| Released | the same stages hear the reason. Under a sessions rule a commit that refused every target takes back the pin its decision claimed, on each side of a pair. A pair's later decode, issued no lease, gives its claim back with its prefill, or with a release that names it alone: a controller's file tells that word, and a request's guard says none for a later decode its host gives up by name |

A count of sends holds a send until one of four ends: a load report of the worker sampled at or
after the send (later by `sent_report_delay_ms`, where the profile sets it), the age `sent_ttl_ms`
where the profile sets it, a newer send once the worker's rank holds 8,192, or a membership
change the count is told. The changes that end a worker's sends are its removal, its replacement
under another id and, where a controller's store tells it, a change of its data-parallel width.
A list of the ids that stand
(section 8.1) ends the sends of each worker it leaves out.

A sessions rule places some requests alone, behind the profile's filters and with no scorer or
picker of the profile asked: it follows a pin of the request's key, or its own assignment places
the key. The rule says whether such a request is the profile's own (`rule_placed`). Under `told`,
the default, it is: the stages hear its issue and its dispatch,
every count of sends takes it, and its prompt is recorded as behind any pick. Under
`untold` it is kept from the profile: no stage hears its issue or its dispatch, no count of sends
takes it, and neither the engine's record of sent prefixes nor a lent token tree takes its prompt.
The session's pin still follows its dispatch, and its completion and its release are told as any
request's, so a stage matches an end by the lease's id. The routing settings write `untold` for
the sticky override's rule alone: a gateway's sticky map keeps a request it places from the
gateway's policy. The engine keeps the word on the pick, and one function reads it
(`lease_events::stages_hear`) wherever it is asked, among them a single's lease words, the record's
two writes, a pair's two writes behind its pick, the tree's store, and the hand-over of a pin
followed by one row. Such a pin that is the profile's own
goes through the lookup that names a request's holders with its answer made, so its prompt is
recorded as a pin's the whole decision follows. A controller asks the same function where it files
a pick's lease, and keeps the answer on the lease's record (`LeaseRecord::untold`): no stage hears
that issue, and the dispatch it tells off the file moves the session's pin and reaches no stage
and no count of sends. Its decision is the engine's own, so its record of sent prefixes follows the
word too.

The session table takes no new key once it holds 4,194,304 keys. A request of such a key is
placed with no pin, until keys leave the table.

Upkeep, about once a second (`sweep`): expired session keys are removed, and the record of sent
prefixes gets a bounded step of clean-up. A record leaves the engine's map once no list's table
holds it and it holds no worker's place.

Under the follow of a host's registry a record no table holds leaves with its places too, once no
worker that stands states its model as its own: the follow asks the registry behind its own step
of the upkeep, and a record a table took during the ask stays.

## 7. The ten built-in policies **[built]**

Each classic policy name has a preset in the crate. Every preset starts with the filters
`healthy`, `not-overloaded` and `not-tried`. In a pair, the prefill side adds `has-capacity` and
the decode side adds `has-room`.

| Policy | Scorers | Picker | What it remembers between decisions |
|---|---|---|---|
| `random` | none | `random` | nothing: it draws from the decision's seed, and the one pick from the calling thread's own draw sequence |
| `passthrough` | none | `passthrough` | nothing: the first survivor |
| `round_robin` | none | `round-robin` | one counter for each worker list, which the one pick and a first attempt's whole decision both turn, at every list size; in a fixed table, one rotation for each set of surviving workers, which a refused turn, a retry and a request held to places turn, and one for each pool whose draw is handed no list counter |
| `power_of_two` | `least-load` | `power-of-two` | a count of sends (section 6) |
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
   `InProcess::start(spec, WorkerData { registry, loads, cache_indexes })`. It builds the
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
   host knows a fact of its workers that their objects do not hold. `PublishedList` keeps the
   three as one value (section 5.2).
4. **It hands its worker lists by model, once**: `engine.set_lists(lists)`, beside the start,
   which does not take them yet. The host implements `ModelLists` over its registry. The engine's
   task fills a list's cell behind the registry's word of the membership change. A later call is
   ignored.
5. **It hands its load monitor's channel again where it builds another monitor**:
   `engine.set_loads(channel)`. The engine keeps the channel under any document. Its task keeps no
   snapshot where the document reads no load report and no named signal, and none of its
   policies keeps a report.
6. **It puts a later document in force on the running engine**: `engine.apply(spec)`, refused
   where the document's generation is not above the standing one. The next decision runs under it,
   and a decision under way ends under the one it began under. The session table and the record of
   sent prompts stand, and so do the policies of a profile whose name and text did not change, under
   the rules of section 5.1. The policies built new are told the workers that stand before the
   call returns.

A caller that builds its engine with `InProcess::new(spec)`, as the crate's own cases and
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
| `Replaced` | nothing where the host states the same id of the new object and that object serves every model of the old one; the models it left where it serves one of them no more; a leave of the old id and a join of the new one where the host states another id; nothing of an object the task does not hold |
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
    fn shadow_ring<'w>(&'w self, spec: &'w AppliedSpec, shadow: u16) -> Option<(&'w dyn LentRing, &'w [u64])>; // the ring it lends its binding's shadow profile and its places on it, by that profile's word
    fn plain(&self) -> Option<Plain>;                 // what a plain request states, if it is plain
    fn prompt_chars(&self) -> u32;                    // the length of its text in characters, asked for a plain request's lease
    fn prompt_bytes(&self) -> u32;                    // the same length in bytes
    fn session_counted_under(&self) -> SessionCountedUnder; // where a worker object counts its session key's requests
    fn workers(&self) -> WorkerLists<'_>;             // the worker list, a pair's two lists, or an encode request's three
    fn loads(&self) -> Option<&Arc<LoadSnapshot>>;    // a request's snapshot of the host's own; without one the engine reads its own
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>>; // the prompt's entry in the host's token tree, where the host lends one
    fn text_key(&self) -> Option<TextKey<'_>>;        // the entry of a prompt the host holds as text in its string tree, where the host lends one
    fn sends_to(&self, target: &Target) -> bool;      // whether its caller sends to this target of a pick: the prompt is stored for the first such
    fn sends_to_encoder(&self, position: u32) -> bool; // whether its caller sends to this encoder of an encode request's pick: the prompt is stored for no target behind one it does not
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
13,856 lines (tests included) for it, and a branch in every route. For a single request, a pair
and an encode request, `place` of section 8.2 now derives the key, hashes the prompt where a
decision reads its hashes, and tries the targets, and its
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
into and asks the ring where each stands (`LentRing::head_position`): the routing value where a
header handed the request one, else the prompt's heads. Where a document names the request's own
id as a source, the value it hands is no head: it names the session alone, as classic hashes a
header's key and never the id's. A
ring whose lender places no head answers nothing: such a request is lent no ring, the lender is
asked about no further head, and the stage's own ring places the request. A request that has no
head states no place and is lent no ring either.
The items of an encode request are placed on that one ring too, each by its own text, where the
encode profile's ring stage reads it.

A pair is lent the one ring for its two lists, and where its two sides read it by one word, one
list of places is both sides'. Where they read it by two forms, or cut a prompt at two depths, each
side is stated what a single request of its own profile is stated: the request's facts carry the
decode side's list beside the prefill side's (`RequestFacts::decode_ring_positions`), and each
side's stage walks the lent ring from its own places, as each classic policy walks the model's
ring from its own. A side whose own profile reads the ring by two words is stated nothing, as a
single request of that profile is, and its stage places by its own ring.

### 8.2 The contract as designed **[partly]**

Everything in the list above that is the same for every host moves into the crate. A host does
three things. The names below may change; the contract may not.

**1. At start, hand over what it already has, once.**

```rust
// engine/in_process/start.rs
let (engine, task) = InProcess::start(spec, WorkerData {
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
upkeep, tells no policy a load report and fills no cell ahead of a request. By itself it tells
the policies no worker either, with one exception: a later document put in force has the policies
it builds new told the workers that stand (section 8.1, step 6).

The start takes no worker lists. Each request gives its own. The engine's task asks only the
lists a host hands it beside the start, to fill their cells (the fifth job below).

The engine's background task does what a host did by hand. Five of its jobs are built
(`InProcess::run`):

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
(`Lifecycle::keeps_loads`). No stock policy keeps one: the job serves a policy a host registers.
At each load snapshot the task takes, it tells such a policy the
reports that are new in it, for the workers it follows in the host's registry, and only then keeps
the snapshot at hand. A report is new where its sample instant is not the one the task told last
of its worker. The channel keeps one snapshot, so of two reports of one worker the task took no
snapshot between, the earlier is told to no policy. A told report states its named values too, one
cell a signal the document declares and in the document's order, under a document a stage of which
keeps a signal's column: a stage reads its column's cell (`SignalColumn::told`) for the time of the
call and copies out what it keeps. A report that is not the whole of its worker object states
none, as a list's cells hold none of it, and neither plugin view carries the values yet.

The fifth job fills a model's cells ahead of its requests, for a host that handed its lists
(`InProcess::set_lists`). The task asks the host for lists (`ModelLists::lists_of`):

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

These faults are held and counted with the task's faults (`InProcess::upkeep_faults`):

- a panic of the host's own code where a worker names its models, which costs that worker's
  fills and, where the worker was replaced under its id, its record of sent prefixes and the
  stages' word of the models it left;
- a list refused for its facts, which costs its own fill alone;
- a panic of the host's own code where it hands a model's lists, which costs the lists of that
  model not yet handed.

The policies are told the word all the same.

The host keeps one empty cell beside each worker list it publishes. That is one field in the
host's `Generation` type. The engine fills the cell at the list's first ask: the task's where
the host handed its lists (the fifth job above), and the first decision's otherwise.

The cell is built. A request's `WorkerLists` name each list as a `LentList`, and the decision
that needs a list asks its cell. The start does not take the lists' trait yet. A host hands its
lists by `InProcess::set_lists`, and the field of the start's data comes with the change in
which the gateway hands its own lists.

**2. For each request, one call.**

```rust
// engine/in_process/place.rs, behind the feature `in-process`: built for a single request, a pair and an encode request
pub struct PlaceRequest<'a> {
    pub model: &'a str,
    pub lists: WorkerLists<'a>,       // one list, a pair's two lists with their pair classes, or an encode request's three
    pub prompt: Prompt<'a>,           // token ids or text, and the seeds of its cache partition
    pub headers: &'a dyn HeaderRead,  // the engine reads the headers the document names
    pub request_id: Option<&'a str>,
    pub explain_id: Option<&'a str>,  // its host's name for it: what a kept decision is found by where it states no id
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
    pub counted_on_decode: bool,      // the caller counts the request on its first decode at that ask
    pub ring: Option<&'a dyn LentRing>, // the hash ring the caller keeps of the model's workers
    pub admits: Option<WorkerRule<'a>>, // the route's own rule, asked of the pick
}

pub type WorkerRule<'a> = &'a dyn Fn(&dyn Worker) -> bool;

impl InProcess {
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

Inside `place` the crate derives the routing key, hashes the prompt where a decision reads its hashes, decides, and tries the
targets best first: health and circuit breaker are read again, and the route's rule is asked of
the picked worker. If none holds, it decides again over what is left. The host sees a worker or a
reason.

Built so far, for a single request, a pair over its two lists and an encode request over its three: the key, the hashes, the decision, and its targets tried
best first by health, circuit breaker and the route's rule. Where none holds, a lease the pick was issued is given up at once and the decision is made again, told to leave out the workers that could not be sent to, three decisions a call at most; then the request has no worker
(`NoWorkers`).

A pair's target holds where both its workers can be sent to.

A profile's `not-tried` filter leaves those workers out while another worker remains, and a
profile that lists no such filter decides over the whole list again, or over the workers the rule takes once a call is held to them. Of a pair's target they are
the sides that could not be sent to: a side that held stays a candidate. A worker the rule
refuses is left out like one that is not ready.

Behind the first decision of a single request or of a pair none of whose targets holds, whether the rule refused them
or they were not ready, a rule the request states is asked of every worker of the list once, ready or not, and the later
decisions are held to the workers it takes. A pair's rule is asked of each of its two lists, among the places its host states for that side, and its later decisions are held to the workers it takes on each. A pair's first decision that sheds the request as saturated has the rule asked the same way: it judged the overload over the whole lists, and the next decision judges it over the workers the rule takes. So the rule's one worker of a long list is found, one
over its overload thresholds is answered by the binding's rule for a list wholly over them, and
one an earlier attempt tried is handed again where no other is left. A request whose rule takes
no worker of a list is shed with no endpoint, under `admits`. A single request's later decision
is given what the call has left of the request's budget, and a request with none left when a
decision is due is shed under `budget`. A pair's decisions are each given the whole budget.

**[built]** A single request can name its worker.

- **Where it is named.** A binding's `routing_key` block names a target source
  (`KeySourceSpec::target`): a header whose number names the one worker the request goes to. The
  preset of `consistent_hashing` names `x-smg-target-worker` as that source, as classic's policy
  of that name reads it, and the routing settings write the same source for that policy alone.
- **How the number counts.** The one call reads the header once, ahead of its decision. The
  number counts, from 0 and in the list's order, the workers that are available by the worker's
  own test and that the route's rule takes, as classic counts them (`select_pass` in
  `model_gateway/src/routers/common/placement.rs`; `target_among` in
  `model_gateway/src/policies/consistent_hashing.rs`). A worker that fails a hard demand of the
  request takes no number: classic counts over the pool it narrowed to a pinned request's wire
  (`Candidates::of` in `placement.rs`).
- **What the call does with it.** The call's one decision is held to the place of the worker
  the number reaches (`Constraints::places`), and the request goes to that worker or to none.
  A retry is a new call and counts again by the same tests: a worker an earlier attempt tried
  takes a number where it passes them, and is handed again where the number reaches it.
- **When the request is shed.** Where some worker of the list can be sent to, the request is
  shed with no endpoint, under `target-worker`, in each of these cases: the text is no number,
  the number reaches no worker, or the worker it reaches cannot take the request after all.
- **What it costs under a document that names none.** Whether a binding names a target source
  is decided when the document is applied: a request under a document that names none pays one
  test of that word, and no header of it is read for a target.

Where no worker of the list can be sent to, there is none to count. Classic answers such a
request by its default for workers that are all vetoed, as it answers any request, and lifts no
veto for its routing
value (`overload_fallback` in `placement.rs`; `routes_by_session_key` in
`consistent_hashing.rs`). The engine places it as a
request that names no worker and carries no routing value, by the binding's rule for workers
that are all vetoed.

**[built]** A pair can name its worker too, a side at a time. Classic reads the header once a
side, each side's policy over its own workers (`pair_pass` in `placement.rs`), and a side it
holds is narrowed by no admission filter and by no retry (`strict_target`,
`admission_prefilters`). A pair's binding names one list of sources for its two sides, so the
target source is the side's whose profile is the hash ring keyed by no prefix
(`Engine::names_worker_on`): classic's policy of that name alone reads the header. The call
counts the prefill's number as classic counts its candidates: the prefills that can be sent to
and have a partner that can, at the places the caller states for each side, and those with such a
partner among their first partners alone where there is one. A worker that fails a hard demand
its side's role is held to takes no number on that side: classic narrows both pools of a pinned
pair to its wire ahead of the count. A place its host masks takes no number either, as one that
fails its own test. The mask (`PairLists::prefill_stalled`, `PairLists::decode_stalled`) states a
veto the host's worker objects do not carry, and classic's carry every veto its count reads.
Where the decode's side is the ring's too, its number counts
those partners of the prefill reached, and the pair's one decision is held to the two places
(`PairLists::prefill_places`, `PairLists::decode_places`). Where it is not, the decision is held
to the prefill's place and the decode is its own profile's pick. A named prefill at the admission
limit is waited for (`Unplaced::PrefillsFull`), as classic waits for it. A named decode with no
free room is taken, because the `has-room` filter is soft (classic,
`retain_decodes_with_a_room`). A retry is handed the pair again. A number that reaches no worker
on a side sheds the pair under `target-worker`, as classic refuses it. Where no decode can be
sent to, a decode over its thresholds alone counts as a partner, and the decode goes by load, as
classic pairs such a pool. Under a binding that sheds a pool over its thresholds on every worker,
no decode counts so, as under classic's shedding. Where no prefill is counted, the pair is placed
as one that names no worker, and a pool its binding sheds is shed as for any pair. A pair held to
the two workers it names books its decode with its prefill under every document, as classic picks
a decode once, with its prefill: a decode asked again behind the prefill is refused as one that
never stood chosen (`DecodeRefused::NotChosen`). Classic's side of every clause is `pair_pass`,
and the cases name its lines (`place/tests/pair/named.rs`).

**[built]** Where the decode's side alone is the plain hash ring, the prefill is its own
profile's pick, known inside a decision alone, and classic counts the decode's number behind
that pick (`pair_pass`; its case
`an_explicit_target_under_consistent_hashing_keeps_its_full_decode_partner` in `placement.rs`).
The call lends the places of the decodes that can be sent to, counted as a pair named on both
sides counts them (a hard demand a decode is held to, a place its host masks), and states the
number to each of its decisions (`Constraints::target_worker`). A decision counts it among the
partners at those places of the prefill it picks, its first partners where the pair is made
among those, and its decode side keeps that one decode through every pass. It books that decode
with its prefill under every document, as a pair held on both sides does. A number that reaches
no partner sheds the pair under `target-worker`, and so does a decode the decision cannot name:
the partners of no other prefill are asked, as classic refuses there. The pair's default for a
pool its vetoes refuse reads no number, as classic's reads no header, and a decode pool all over
its thresholds states none. The call states none either where no prefill opens the pair: none
that can be sent to, read as a pair named on both sides reads its prefills, has a partner among
the decodes counted. Classic reads no header there: it answers as for a pair that names no
worker, and pairs by load a pool whose every worker is vetoed (`relaxed_pair`). So does the
engine, whose default pairs over the caller's own places. A prefill over its thresholds alone
opens the pair too where its profile weighs load itself and its binding steers, as classic
lifts the veto for such a policy with its shedding off. A decision that states a number beside
no places of its decode side is declined under `target-worker`, and so is an encode request's
that states one. A controller's served decision is lent the same decodes and stated the number
its pair's frame carries (section 10.1).

One thing differs from classic there, by rule. **[planned]** The engine reads no encode
request's header for a target yet. One limit is a supplied policy spec's own: a pair's target
source holds a side only where that side's profile is the plain hash ring, because the spec has
no word for a side.

An encode request is placed over its three lists as a pair is over its two, each decision given the whole budget. Where it carries items its call makes three decisions at most, and the later ones are held to the workers the rule takes on each of its pair's two lists, as a pair's are: behind the first decision none of whose targets holds or whose encoder cannot be sent to, and behind a first decision that sheds the request as saturated over a list of its pair. The rule is asked of no encoder, and the encoders' list is decided as its caller lent it: a shed made over the encoders stands, by the decision's own word of it, and no rule is asked behind it. One
decision names its pair and each item's encoder, and the pair's targets are tried as a pair's
are. The placement names each item's encoder beside the pair (`Placement::encoders`). An item has
one encoder: where it cannot be sent to, the decision is given up and made again with that
encoder left out, and each side of the pair that could not be sent to either. An encoder is read by its health and its circuit breaker alone: the route's
rule is asked of the pair's two workers. Three lists with no item are placed as their pair. A
request whose decode is held to the workers that take images books its decode with its prefill,
whatever its caller states of asking the decode again: a decode asked again reads no image fact.
So does a request with items that demands vision or a way of processing its references: its
encoders meet that part of the demand, and a decode asked again would hold the decode to it.
That decode never stands chosen.

A retry stays on its wire by its demand alone. The host gives one number to each runtime and
transport it dials. It writes that number as each worker's fact and as the retry's demand, and
it states no wire modes for a list it dials whole. The engine compares the two numbers and
reads no more of them. So a pool on two transports is handed as one list, and no host keeps a
list by wire. An encode request's retry is held the same way, a list at a time. Asked again for
its pair alone it states no item and is held as a pair is. Asked again with its items, its
encoders are held to that number too.

Two things keep the short path of section 5.6 as cheap as it is today. The route's rule is asked
of the one picked worker, and of a whole list only behind a decision that leaves its call no placement: one none of whose picks can be sent to, an encode request's whose encoder cannot be, and the first of a pair's or of an encode request's that sheds the request as saturated over a list of the pair; a rule that is the same for every request of a
route belongs to the list itself. And a plain request touches neither the load snapshot nor the
other fields of the request, with one exception in a gateway's own process. Where a stage needs
lease events, the lease states the text's bytes, and its characters where a stage reads a size.
Counting the characters is the one pass over a plain request's text.

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
    // a pair's decode stands chosen: decode_again may pick it again
    pub fn decode_chosen(&self) -> bool;
    // a pair's decode decided again behind its prefill's answer, the request stated again
    pub fn decode_again(&mut self, request: &PlaceRequest<'_>) -> Result<(), Unplaced>;
    // dropped with a side not ended: the engine hears a release before the side went out,
    // and an end of class cancelled after it
}
```

What a placement holds, and so what these calls do, depends on the request:

- **A single request.** Its placement holds a lease only where its pick has one (section 6). With
  no lease, `sent` and `ended` tell the engine nothing. `decode_again` refuses a single request with
  a lease or without one: it has no decode side.
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
   again, as its first call stated it. A caller that counts the request on its first decode
   across this call states `counted_on_decode` beside it, and the decision reads that decode
   without the request: one request less in flight, and one less of its key where a capped
   sessions rule reads that count.
3. The engine makes the request anew from what the caller stated, as it makes a first call's,
   and decides under the binding of the model stated there. Every fact of the decode's decision
   is that request's: the pair's two lists as they stand then, the request's size, its routing
   value and the workers it tried among them. The engine decides the decode on the records of
   that instant, behind the prefill that was sent. Of the prefills it writes that one's record
   alone and reads no other prefill's state. Where the list names no place of that prefill,
   it writes every prefill's.
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
    // a pair's decode side: its decode stands chosen and may be decided again
    pub fn decode_chosen(&self) -> bool;
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
allocation a side. The document asked is the one the placement was decided under, whichever is
in force when its host splits it.

A pair's two sides share the pair's one placement behind a lock of that one request. The
crate keeps the order of a pair's words itself: a prefill side let go while the decode still
stands chosen holds its word back until the decode side speaks, and says it then as it stood.
A prefill that went out says its end, with the outcome reported or with the instant of its own
drop. One that never went out is released. So a host that sends a pair's decode and no prefill
orders nothing by hand.

A host that keeps sides asks the decode again on the pair's decode side, once the prefill
side's send was said. It asks first whether the decode still stands chosen (`decode_chosen`),
and makes no call where it does not: the call would be refused by name, with no decision made.
It may let the prefill side go before that call or behind it. A prefill's end that waits is said
behind the decode the call took. Behind a call that took none, it still waits. A side that is
not a pair's decode side is refused by name, with no decision made.

Whatever a request lends that second decision (its rule, its headers, its ring, its lists'
admission and their workers' own calls) is asked under the pair's lock. None of it may send or
drop a side of the pair, or report through or drop a handle on a side's end: it would wait on a
lock it is asked under and never return.

Nothing else is asked of a host that makes the one call (`place`). In particular it never names
a policy, never looks inside a decision, never keeps a lease and never runs a timer for the
engine.

### 8.3 What is reused and what is added

The design adds no second mechanism where one exists.

| Need | Reused, as it exists | Added |
|---|---|---|
| a model's worker lists | the registry's published lists (`Generation`) | one field beside each list: the engine's value |
| a worker's health, breaker, overload, in-flight count | the worker object, read in place | one bit at a pair's decode asked again: that the caller counts the request on that decode already (section 8.2) |
| load reports | the load monitor's `LoadSnapshot` | the last report of a worker whose load call fails, kept beside the snapshot's entries, and the one read that gives it (decision 12) |
| which worker holds a prompt, by cache events | the kv-index crate's index, through two traits | the gateway's two readers of its index for those traits (section 9.2) |
| which prefill pairs with which decode | the gateway's pair index (`PdPairIndex`) | the pair classes read from it, one file (section 9.2) |
| which worker was sent a prompt | the host's own trees, of the kv-index crate's two tree types | one more question on the trait the host already answers for cache events |
| a request's lifetime on a worker | the load guard every route keeps | one optional field (decision 4) |
| the place where a worker is chosen | classic's four placement functions | one call in each |
| the rank a request is sent with | the worker object's own rank, where its registration states one | nothing: no rank choice runs behind the pick of a worker registered rank by rank, and the classic rank option is refused at the start beside the engine (section 9.2 names the two places a rank is still chosen) |
| the lease's reporting | `DecisionGuard` | `Placement`, which holds it |
| proof against classic | classic's recorders and their sheets | replays through the engine |

## 9. The gateway as a host **[partly]**

After stage 0 the gateway did not call the engine. This section is the one change that brought it
back, behind an option that is off by default (section 9.3).

### 9.1 Where classic chooses a worker

Every classic route ends in one of four functions:

| Function | Used by |
|---|---|
| `placement::select_single` | HTTP typed requests; the token-count route |
| `placement::select_single_of` | the gRPC regular stage; transcription |
| `placement::select_pair` | the HTTP and gRPC prefill-decode routes |
| the encode stage's own selection | the gRPC encode-prefill-decode stage |

Three are in `model_gateway/src/routers/common/placement.rs`. They take what a classic policy
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
        the four placement functions
              │
              ├─ engine off ──> the classic policy object                 (unchanged)
              │
              └─ engine on  ──> engine.place(request) ──> Placement
              ▼
        the load guard for the chosen worker   (exists today; lives while the request is on it)
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

  The start makes the Grand Central crate's one start call as the last step of the context's build. The document is checked first against the table of stages the gateway's host hands the start, or the stock table where it hands none, and a refused document ends the start before the gateway listens. The engine's task follows the registry's events and tells the document's stages a worker that joined and a worker that was removed. With the engine on, the gateway builds its policy registry with no classic policy object: a stand-in that places nothing stands in its default and encode slots, and a worker's registration and a router's build add no policy.

- **Each request.** Three of the four functions ask the engine. They fill `PlaceRequest` from the inputs
  they already take. A route names the answer's type and makes its load guard from the answer, in the places it made one before.

  Where the engine places no worker for a request, the gateway names the client's answer itself,
  by classic's own judgement of the pool. The engine's word for the refusal goes to a log line
  and no further: at `debug`, or at `warn` where the engine refused the decision itself. One word
  is acted on, a pair's prefills all at their admission limit: the route waits for a slot, as
  under classic.

  The fourth, the encode stage's own selection, takes its pair from `select_pair` and refuses a request that carries an item for the encode workers by name **[planned]**: the gateway does not yet hand the engine an encode request's three lists.

  Where the document asks a pair's decode again, the two pair routes make one more call between the prefill's answer and the decode's dispatch (section 8.2).

  A retry held to its wire is placed over the pool's whole list and stays on the wire by its demand (section 8.2) **[built]**. The gateway gives each runtime and transport it dials one number. It states that number of each worker where a list is first lent to the engine, and a retry states its wire's number as its demand, as does a first attempt its route pins to a wire. A request pinned to no wire states none. The route's own rule for a worker still holds beside the demand. An encode request that states a demand is held to it a list at a time (section 5.4). The gateway hands the engine no such request yet: its encode entry waits until the two refusals of section 5.4 are named as classic names them **[planned]**.

- **The request's end.** The placement stays with the request's load guard, which every route
  already keeps for as long as the request is on its worker, streamed or not (decision 4).

  The engine hears the dispatch where the guard is made, or at the send for a guard a route makes
  ahead of its send, and the end where the guard is dropped. It hears a guard dropped before its
  send as a release, abandoned. Two guards are made ahead of their sends, both a pair's: the
  decode's, behind the selection, and the prefill's, where its load is acquired (with prefill
  admission, under the admission's lock). The prefill's is a plain guard under the route's key.
  A route says the prefill's send where the prefill's request goes out, a batch once for all its
  sub-requests. There the guard takes the engine's side, and counts under the routing value the
  engine names from there on.

  Classic's `least_load` credits the worker it picks among several with the request's work at the
  pick, until a report of that worker sampled behind the pick, and takes no pick back (`decide` in
  `model_gateway/src/policies/least_load.rs`). The engine's counts of sends take a send at its
  dispatch. So a pair that a route refuses between its selection and its send stays in classic's
  count and enters none of the engine's. The maintainer decided that difference on 2026-11-24
  (decision 16): under `least_load` the engine's count stays, and a request that is never sent is
  not counted.

  The regular HTTP router's typed route and its streamed pass-through tell the engine that end's
  outcome, through the handle of section 8.2: the status the route records on the worker, and its
  class by the gateway's one mapping. An answer read whole is reported through the answer the
  route keeps, where the route records its status. A streamed answer is reported by its relay,
  through the handle the route asked of the guard: the head's status at the stream's end, or the
  status of a cut behind a good head. A client that left is reported by nobody, and so is a
  client's own body fault on the pass-through. These routes state no time to first token and no
  tokens.

  A route tells every answer of its workers or none. The other routes tell no outcome yet
  **[planned]**: the transcription route, the gRPC stages, the pair routes. There, and for an end
  nobody reported, a stage hears the end of a request that was sent as a cancelled completion,
  with the time since its dispatch and no other measure. So a plugin stage that learns from
  outcomes reads those requests as cancelled.

- **Membership.** The gateway's value for a model's workers gets one cell for each list, which
  the engine fills.

  The pair classes a gateway hands with a pair's two lists are derived under the pairing of the document in force: beside a supplied document the registry takes that document's `plans.pd_pair.pairing` at its build, and no request reads it.

  On the wire that carries one runtime's request the classes also state each prefill's first partners: its partners on its own known runtime (`PartnerClasses::number_first`). The engine makes a pair among the prefills with a first partner open and those partners while there is one, so a pair on one known runtime goes ahead of a pair with a worker whose runtime is unknown, as under classic. The other wire states no first partner, and a fleet whose every partner would go first states none either.
- **The rank.** A data-parallel worker registered rank by rank (every registration reads the worker's width and registers one worker a rank; an HTTP worker whose load route does not hold a row for every rank stays one worker, unless its host keeps a load source of its own for it) is the shape the engine serves in full. Each rank is then a worker object, and a record of its own in its list (`Candidate::dp_rank`, section 5.3). The record reads its own rank's named values (section 5.8). A stage that keeps load reports is told each rank's own report, under that rank (`Signal::Load`). `Candidate::engine` is the same number on every rank of one such worker. The request is sent with the rank of the record the engine picked: where a record is one rank, no rank choice runs behind the pick. A rank is chosen behind the pick in two places, each by classic's own code, with the engine on as under classic, and each inside a worker registered once for all its ranks: on the HTTP pair route, under a connector that holds both sides of a pair to one rank, the route's own turn names the rank both sides are sent to; and under the gateway's `--grouped-dp-sticky hash`, the gRPC dispatch pins a single request that has a sticky key, and where the runtime allows it a pair's prefill side, to that key's rank. That option is refused beside a supplied document.

  The gateway's rank option (`dp_minimum_tokens_scheduler`) is refused at the start beside the engine, under the settings' own document and under a supplied one. A worker object that stands for several ranks states no rank, and outside those two places its request is sent with none. A route that can name no rank to its worker, the multipart transcription, sends with none to the engine of the record picked, a rank's record included: the engine places the request among its ranks, and the record's count holds it.

The change also wrote again three small things that stage 0 took out with the earlier
integration: the gateway's two readers of its own index of cache events for the engine (about
70 lines on main today), the pair classes read from the gateway's pair index (one file of 404
lines), and the cell beside each published list.

The bar for that change:

1. It adds no file of policy logic to the gateway and no branch on a policy's name.
2. No route gets an engine branch. The call sites are the four functions, and one call in the two
   pair routes between a prefill's answer and its decode's dispatch, where the document asks the
   decode again. A route that tells an outcome makes one more call where it records the status
   on the worker: on the answer it keeps, or on a handle it asked of the guard for its stream's
   reader. A classic request makes the same call.
3. With the engine off, a classic decision pays what rule 5 names and no more, counted in the
   gateway's decision bench.

   With the engine off, neither pair route makes the one call of the second point. Each route asks the pair's first answer once, where it decides how the pair is sent, and builds no ask for a decode the classic policies picked. A classic pair sent prefill first still pays that test, the holder of its decode ahead of the send, and on the HTTP route the cell a moved decode is written to. The decision bench runs no route's attempt, so it does not count that cost. It was counted once by a program built for the count, under Callgrind, on the lines the second point added.

   With the engine off, an outcome's report through the answer returns at its first test, of the answer's engine side. A streamed answer pays the route's ask of its guard and its reader's test of an empty handle. A pair route's relay is handed an empty handle and pays its test. The decision bench's `--end` flag counts each against a cell that reports nothing.
4. With the engine on, every classic test of the gateway passes under the settings' own document.

### 9.3 The options

| Option | Meaning |
|---|---|
| `--grand-central on\|off` | who decides, for the whole process; default `off` |
| `--policy-spec FILE` | run this document in place of the one the routing flags mean |
| `--print-policy-spec` | print the document the routing flags mean, and exit |

They came back with the change (decision 9). No released version has them.

In a gateway whose every router is a third-party provider's, `on` changes nothing: no engine
starts, the classic policies stay, and the start says so in one log line. A document supplied
beside such a mode is refused at the start, by the mode's name.

## 10. The controller **[partly]**

The controller is the binary `smg-grand-central`. This section fixes the direction. The
switch is made in steps: 10.1 says what is on main.

### 10.1 On main

It starts from a policy document (a watched file, or a folder it keeps). It serves gRPC on three
listeners (gateways, operators, standbys) and HTTP
on an admin listener (probes, version, metrics). It elects a leader by a leader lease, held in the
process, in a file, or in a Kubernetes Lease object. That lease is not the lease of section 6.

Its data stands in an in-memory store (`store/memory.rs`), in the worker crate's registry of worker
objects and its load monitor, in the worker host's tables and in the engine's own tables:

| Data | Filled by | Note |
|---|---|---|
| worker records | the `RegisterWorker`, `PatchWorker`, `DeregisterWorker` calls, and the leader's own discovery | discovery is behind a feature the image leaves out; a record it writes asks for the worker's cache events where the document in force at the find reads them for the worker's first model, and a document applied later does not write that word again |
| worker objects | the write of each worker record, in one step with it: a call's or the discovery's on the leader, the applier's on a standby | a second form of the same workers, of the worker crate's type; each states its registration's overload thresholds over one default, which its start states (8 queued requests and 0.8 usage by its flags' defaults), an external record's over none, as every record's is while the document in force switches the default off, and the load monitor judges by them while the replica leads; an object's status leads its record's, and a single request's decision reads the workers here |
| health | the worker crate's health checker over the worker objects, on the leader | copied into the records |
| loads | the worker crate's load monitor over the worker objects, while the replica leads: a poll every 10 seconds, and the load records on the workers' streams | a single request's decision is given the monitor's snapshot where the document reads reports and keeps none; the store holds no load table |
| cache index | a follower of each worker's cache events | an index in the worker host's KV tables (`worker_host::KvTables`), which the holder of the worker objects is handed; a decision reads it through the engine's two index traits (`worker_host/kv_read.rs`), and the engine's own record of sent prompts for a worker the index holds no block for, and for every worker on a request given as text or under a profile that reads the record alone (`source: predictor`) |
| leases | decisions | not replicated to a standby; a lease that counts for its session key steps that key's count on its worker's object, which a controller's in-flight cap reads (10.2); a single request's session pins are the engine's own table, one for each document |

The gRPC service (`crates/grand_central/contracts/grand_central.proto`):

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
| `StepDown`, `WatchRegime` | | not served |

A `Decide` request of the single phase is decided on one of a fixed set of threads (the model
picks the thread) by `InProcess::decide_without_lease`, over the list of the model's workers
that the worker objects' registry publishes. A pair's request at its prefill is decided there
too, over the registry's two lists of its model. Its answer carries a lease a side, or the
prefill side's alone where its frame stated the later decode. A decode call that names a lease
goes to the thread that keeps that lease's file. One that names the
pair's lease is decided behind the prefill that lease holds, by
`InProcess::decide_decode_without_lease`, and its pick is issued under that lease. A prefill
request, or a decode request that names no lease, stands alone where its plan pairs no side of
its phase: it is decided as a single request is, over the registry's list of its own phase's
workers, and its lease is a single's. The token count of a prefill-decode mode (section 3.3) is
not decided this way: it is a single request over the prefill workers, the wire cannot say so,
and a controller reads the single workers' list for it, which such a fleet does not have.

An encode request states its items on a frame of the encode phase (`DecideRequest.encode`), with
a key or with none, and 256 keys at most. A longer list is refused as out of range, and so is a
frame of another phase that carries items: no frame with items is decided without them. Such a
frame is its pair's one request. It is decided at the prefill phase, for both sides of the pair
and for each item's encoder, by `InProcess::decide_encode_without_lease` over the registry's
three lists of its model, with the holder's pair classes and its encode columns. Its pick is
issued as a pair's is, with no lease for an encoder, and its answer names each item's encoder
beside the decision (`DecideAnswer.encode`). The answers come in a pair's order, and a request
with a key takes its encoders' list as a third list in it. A decode asked again names its lease
and states no items. A frame of the encode phase that does not state `DecideRequest.encode`, and a
decode request that names no lease under a pair plan, are answered `Busy`. By the contract the
gateway decides such a request locally
(`crates/grand_central/contracts/grand_central.proto`) **[planned]**: no gateway calls a
controller yet.

The crate's own client states the items. A handle asked over a link takes an encode request at the
prefill phase, as the engine's entry takes it in process, with its items
(`DecisionGuard::asked_encode`), and its frame states the encode phase. A session key in its hash
form crosses as the session's value under the encode phase. The controller reads the key back under
that phase, so the pair's session is the one the request has in process. The items' one key
crosses where the host states one, with a form or with none: one with no form still places the
items as one, with no affinity, as in process. The handle is a pair's,
and it reads each item's encoder from the answer, in the keys' order. The link refuses a frame whose
items are out of range by the controller's own check and sends nothing, so no such frame ends its
stream.

A frame states whether the request's own id handed its session key. Where it says so, a
controller's prefix ring places that request by its prompt, as a gateway's own process does
(section 8.1). The frame of a host older than that field says nothing, and under it the id's key
places as any other.

A frame may state the number its request names its worker by (`DecideRequest.named_worker`): a
single request's frame, one that stands alone at a side's phase, and a pair's. Its host reads the
number under the target source of its binding (section 8.2) and states it. A text that is no
number crosses as the widest number, which reaches no worker. Under a document whose binding for
the frame names that source, the controller counts as the one call does, by the same function.
For a frame decided over one list it counts from 0 and in the order of its own list, over the
workers the asking gateway can be sent to. Those are the places the gateway may take by its reach
and its wires, that none of its standing vetoes holds back, and whose worker object is available
by its own test. The frame's one decision is held to the worker the number reaches. The request
goes to that worker or is shed with no endpoint, under `target-worker`: where the number reaches
no worker, or that worker cannot take the request after all. Where the gateway can be sent to no
worker of the list, the request is decided as one that names none, and carries neither its
session key nor its client's key. A frame that states no number is decided as it was, and the
frame of a host older than the field states none. No encode request's frame is read for the
number.

A pair's frame states one number for its two sides, as one header gives a gateway's own process
one (section 8.2). The controller counts it as the one call counts a pair's, by the same
functions, over the two lists as the asking gateway may take them: each side's places by its
reach and its wires, the places its standing vetoes hold back as that side's mask, and each
worker object's own test. The frame's one decision is held to the places the count reaches: the
prefill's and the decode's where both sides' profiles are the plain hash ring, and the prefill's
alone where the decode's is not. Where the decode's side alone is the ring's, the decision is
lent the decodes the gateway can be sent to and counts the number itself, behind the prefill it
picks (`Constraints::target_worker`). A number that reaches no worker on a side sheds the pair
with no endpoint, under `target-worker`, and so does a named worker that cannot take the request
after all. Where no prefill is counted, or none opens the pair, the frame is decided as one that
names none, and keeps its keys. Where every decode is over its thresholds and the binding steers
such a pool, the decode goes by load and its number is not read. A pair's frame that states no
number pays one test of it.

A frame that states its later decode keeps that word (`DecideRequest.decode_later`). Its decode
side is filed chosen, with the one decode the held decision named: it books nothing and waits
under no dispatch deadline. A decode call behind it (`DecideRequest.decode_of`) is refused as
one whose decode never stood chosen where the call's frame names its worker and the decode side
of the binding reads the number, and the gateway sends the decode the first decision named. The
call is judged by the policy spec in force at the call, whatever spec decided the first frame. A
call whose frame leaves the number out is decided again as any pair's decode.

Six things differ there from a gateway's own process (section 8.2). The first four are open
under rule 6. The last two stand only behind a frame that states its later decode, which the
crate's own client states for no request.

- **The route's rule is not asked.** It is the gateway's own code and no frame states it yet, so
  the number counts a worker that rule refuses. It ends when the rule crosses as section 11 plans
  it, and the count passes over the workers it refuses **[planned]**.
- **The order is the controller's list.** The number counts in the order the workers registered
  with the controller (10.2), and a gateway's own process counts in the order of its own list. It
  ends when a gateway under a controller follows the controller's list and keeps its order
  (section 11) **[planned]**.
- **No hard demand crosses.** A frame states none in this protocol version, and a host decides a
  request that states one by itself. So the controller's count leaves no worker out for a demand.
  It ends when a frame states a request's demands **[planned]**.
- **A named prefill at its admission limit is not waited for.** A gateway's own process waits for
  the prefill a pair names where it stands at that gateway's admission limit (section 8.2). A
  controller keeps no gate: its count takes the prefill, and the gateway finds the limit. It ends
  when a controller keeps a pair's admission **[planned]**.
- **A later decode is booked at its dispatch.** A gateway's own process books the decode a pair
  names with its prefill. A controller files it chosen, as the contract has a later decode, and
  books it where a dispatch names it.
- **A decode call is refused whatever the first decision counted.** Where the first decision
  held no decode to the number (no prefill was counted or opened the pair, or every decode stood
  over its thresholds), a gateway's own process asks the decode again as any pair's. A controller
  refuses the call, and the gateway sends the first decision's decode. This is a named limit and
  no cure is planned: the lease on file keeps no word of what its first decision counted.

A gateway that decides by itself for lack of an answer (decision 6) counts over its own list by
its own rule, so it can reach another worker than the controller's count reaches.

### 10.2 As designed **[partly]**

Rule 2 and rule 3 apply to the controller as they do to the gateway: one set of worker data, in
the worker crate's structures, and the one engine.

| Data | On main | As designed |
|---|---|---|
| workers | records in the store, and worker objects kept in step with them | worker objects in a worker-crate registry; beside each, the code that keeps the objects holds what a registration stated and the object does not hold as stated (10.3); the membership log that watchers and standbys read stays |
| lists | a single request's decision reads the registry's published list of the model's single workers, a pair's, at its prefill and at its decode call, the lists of its two sides, an encode request's those two and its encoders', and a prefill or a decode request that stands alone the list of its own phase's workers | the registry's published lists by role and protocol, with the engine's value beside each (decision 10) |
| health | the worker crate's health checker over the worker objects, on the leader; each outcome moves the object, the store's record is told the status that stands, and a decision reads the object's | the same checker; a decision reads the object's own status; a gateway's veto on a worker stays in the store |
| loads | the worker crate's load monitor over the worker objects, while the replica leads, fed by its polls and by the load records on the workers' streams: a decision is given its snapshot where the document reads reports and keeps none, and reads the overload verdict on the object; the crate's own poller runs no more, and the store holds no load table | the worker crate's load monitor, publishing `LoadSnapshot` (decision 10). Every worker is judged by the thresholds its object states, so the loads of every worker the controller dials are read: no new cost against classic as it runs by default, whose gateway hands each worker of an engine the same two defaults and an external worker none at its registration, as a controller's objects state them. A gateway started with `--disable-load-monitoring` polls by need, and this monitor polls every group |
| cache events | an index in the worker host's KV tables, which a decision reads through the engine's two index traits | the kv-index crate's index, filled by the worker crate's collector |
| sessions | the engine's session table for a single request, one for each document | the engine's session table |
| leases | the store's lease tables, in memory, one for each deciding thread | stay: a controller must remember what it issued to gateways |
| decisions | a single request, a pair at its prefill, its decode call, an encode request that carries its items, and a prefill or a decode request that stands alone: `InProcess` over the registry's lists; a frame of the encode phase that does not state `DecideRequest.encode`, and a decode request that names no lease under a pair plan: none, the request is answered `Busy` | `InProcess::decide_without_lease` over the registry's lists, and `decide_encode_without_lease` over an encode request's three |
| discovery | the discovery crate through its host interface, registering and removing through the worker crate's job queue while the replica leads; behind a feature the image leaves out; a worker's API key is refused at the start, since no replica keeps one | the same in the image, with a worker's API key; how a replica comes by the key of a worker found before it led is open, as a gateway's way to a key is (section 11) |

**Who sends a worker's data.** Nothing changes for a worker. It pushes its load records and its
cache events on the stream its host opens (section 4.3), and here the host is the controller. The
controller runs the worker crate's collector and load monitor over its own worker objects, as a
gateway does in classic. A gateway under a controller opens no such stream (section 11). On main
the controller already opens the workers' streams with that collector, on every replica, and
runs that load monitor while it leads. The load records on the streams are handed to that
monitor while the replica leads, and a standby drops them. A gRPC worker that is not indexed
streams its load records alone: the leader reads its load from that stream, and polls it only
while its records are not fresh. A worker that refuses that stream has none, and the leader polls
it at every interval. A standby holds that stream too, so a take finds it open.

The contract has no call that would push the same data to a controller.

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
prompts start empty. The stages of a profile whose name and text did not change do not start
empty, under the rules of section 5.1: the new object keeps them from the one before it, with what
they hold, and tells the members that stand to the stages it built new alone. A lease issued under
the earlier document is told to the stages kept, and writes nothing to the new session pins. That
engine runs no task of its own. The store tells it each word of its
membership, so the record forgets a worker that leaves, is replaced, serves a model no more or
takes another role. The
engine's upkeep runs at each pass of the presence sweep, a tenth of the grace period: once a
second, the period of the engine's own task in a gateway's process. A sessions rule's cap reads a
key's requests on the worker object too: the store steps that count for the lease of each keyed
request on file, a single's on its worker and a pair's on each side's worker, with a sessions rule
or with none, under the session's number written as text. The object's count of keys with a
request in flight follows the same leases. A
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
| the worker record | the store's table of records (`store/memory.rs`) | the code that keeps the objects, among its reads: the one record behind a write the store refuses, every record where a cut is laid whole, a record's status where a leader's log tells it, a host's own follow, and a first copy, for the status the log told of each record; and the store itself, for its published copy and the membership log's event |
| a copy of the record's facts | beside the worker objects (`worker_host/objects.rs`) | the code that keeps the objects, to tell a record that changed under its object |
| the worker object | the worker-crate registry | the health checker, the follower of cache events, a decision, and a watcher's, a standby's and an operator's first copy |
| the remainder: 17 of the 24 facts and one bit of an eighteenth, with the record's id and revision | beside each worker object (`worker_host/remainder.rs`) | a decision, for the record's id, whether the worker takes images, its wire modes and from where it is reached. A pair's classes read the worker's engine kind and its pairing descriptor too, and for a record that states no descriptor the pairing facts its worker object derives of its labels. What the worker states for a request's hard demands is lent with the list, and no frame states a demand yet. The follower of cache events asks here whether a worker's events are indexed: its signals stand here and its models on the object (`worker_host/kv_feed.rs:191`, through `Objects::row_of`) |
| the published copy: of a record its id alone, beside its slot in the counts and in a gateway's vetoes by pool; with them each pool's index by model and role, the generation and the clock | the store | a pick's booking and a gateway's vetoes; every other loader reads the generation; no row of a record stands in it and no request is decided over it |
| the membership log's event | `service/replication/log.rs` | a watcher behind its first copy; a standby |

A leader's write of a record is decided over the worker object since step 1 below, and the
table follows it. The object leads a status too since step 3, and the table is told it. The
object leads a first copy's facts since the last part of step 2, and the log its statuses. After the switch the object leads: a worker's data
stands once, as its worker object with its remainder beside it, and the log still tells it. The
remainder is the copy that stays. The table of records and the second copy of the facts end
with step 4 below. The published copy holds no row of a record, and a status change makes no
publication. It ends when the two readers of what it holds of a worker, a pick's booking and
a gateway's vetoes, read the worker objects: a later step **[planned]**.

**What stands beside an object [built].** `RowRemainder` (`worker_host/remainder.rs`) holds
what a registration stated and the worker object does not hold as stated. The code that keeps
the objects writes one with every object it builds. `Objects::row_of` gives a worker's
whole record back from the object and its remainder, with no read of the store's table.
A decision reads the remainder's facts, among them the record's id, whether the worker takes
images, its wire modes and from where it is reached. A pair's classes read the worker's engine
kind and its pairing descriptor too. A record that states no descriptor is read by what its
worker object offers a partner: the pairing facts the object derives of the record's labels and
engine kind, as a gateway reads them of its own worker. The controller's door lays no
descriptor, since its log holds none, so a controller's workers pair by their labels; its
reading of a worker's spec lays the pairing protocol and the KV connector the spec holds
among those labels. The controller's registration calls decide a write over a
whole record read back (step 1 below). Of the 24 facts a worker's record holds:

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
| the health policy | the checks in whole seconds, each threshold at least one, off where the controller cannot dial the worker | a part of a second, a threshold of none, and the policy of a worker built with its checks off |
| the overload thresholds | where a program asks for them, as the controller's does, an engine's worker's own over the program's defaults, and its own alone for an external worker or while the document in force switches the defaults off | which of them an engine's worker stated |
| the drain window | the window in force | a window stated equal to the object's default, which reads back as none |

- Ten are not carried to the worker object: the incarnation, the zone, the pairing facts, the
  encode offer, vision, the admission window, the priority, from where the worker is reached,
  how its signals arrive and the demand facts. Three of them have a place of the object's own
  that a record does not fill: the object's priority is unsigned where a record's is signed,
  the object derives its pairing facts when it is built, and its admission window too, from
  a label. The remainder holds the ten, with the record's id and revision, who registered
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
   on a standby, by one body. The membership log is told the record the write handed that
   body, ahead of the table's write, and the event of such a write tells no record read from
   the table. The revision an event tells is the keeper's, of the write that made the record.
   Two things are the table's until step 4: a status event comes with no record and tells the
   table's, and the record a write takes out is the one the table held. In a controller the
   code that keeps the objects alone writes a worker object's own facts, so the record a change
   in place writes to the table is the object's record read back whole, less its status: the
   table keeps its own, and the object's is told behind the write. A second writer of an
   object's facts in a controller would break that, as a gateway's refresh of a worker's models
   is one in a gateway. The locks in their
   order: the door's turn, the keeper's turn, the store's membership lock, and no lock of the
   store over a write of the registry. The store takes a decided change only where the record
   it was decided over still stands at the revision the keeper read, tested under the
   membership lock: a write that landed between the two is never written over, and the keeper
   reads that record again before it answers. A first copy is cut from the objects in the
   keeper's turn, the last part of step 2. Since step 3 a status is the object's
   word: a drain and a connect signal write the object and tell the table, a key's next life
   starts in the status its worker object stood in, and a call's answer waits on no follow. A
   change the store took and the registry did not get is laid where the store next refuses a
   write of the record: the keeper reads that record again before it answers. A record with no
   object at all is reached so by a registration under its key alone. A change that moves a fact the object holds builds the object again: the
   new object takes the standing one's place under its id, with its running state, as the code
   that keeps the objects does today for a record that changed.
2. A standby applies the leader's events to its own worker objects **[built]**. A call reads the
   record that stands from the worker objects **[built]**. A gateway's watcher hands its
   watch's events to its own worker objects **[built]**. A watcher's, a standby's and an
   operator's first copy is cut from the objects and their remainders, with the status the
   log last told of each record **[built]**.
3. No health outcome is judged over a record. It moves the worker object, whose status leads,
   and the record is told the status that stands, for the log and a first copy **[built]**. A
   connect signal and a drain are written the same way **[built]**. The store's own machine
   for a health outcome is deleted **[built]**. A decision reads the object's status, as a single
   request's does on main.
4. The table of records, the second copy of the facts, the two conversions between a record
   and an object, and the task that follows the table are deleted. The task is deleted
   **[built]**: the calls a record is written by are public, a host of the worker objects
   writes through them, and a served controller hands the code that keeps its objects. A
   record written through the store's own entries gets its object in three places: where the
   store refuses the next write under its key, where a cut is laid whole, and where its host
   reads the store's log. No task lays it. No turn that takes the controller's own store starts with
   a read of its table **[built]**: a write call decides over the object that stands, a call's read
   and a first copy read the objects, and a standby's applier lays its leader's event. A
   turn of the keeper's own reads no log of the store's changes **[built]**: a refused write
   reads its own record, a cut laid whole reads the store's records once, then the status the
   table holds for each, and the call that lays a kept credential again reads the objects. A
   host's follow still reads the store's log.

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
place keeps the instant it was admitted at. A status the leader tells is written to the table and
laid on the object in the same turn. A restore is the store's restore and the read of its
records in one turn of that code. A first copy is cut from the objects, by the last
part of step 2.

**A call reads the worker objects [built].** A registration call reads the record that stands
before it writes: the record under an address, the record under an id, and an engine's records.
It reads them back from the worker objects (the second part of step 2), in the turn of the code
that keeps them. So a call's test and the
write behind it read the record from one copy, and a call that waited behind another writer reads what that
writer left. A record read this way carries the status its worker object states (step 3).
An engine's records are found by the list of ranks kept beside the objects,
where the store passed over every record under the membership lock: that read of the table is
deleted.

**A first copy from the worker objects [built].** What a watcher starts from, what a standby
is restored from and what an operator reads are cut from the worker objects and their
remainders (the last part of step 2), in the turn of the code that keeps them. Every writer
of a record's facts holds that turn from its decision to the log's telling. The store's own
entries for a registration, a patch and a removal take no such turn: no product code calls
them, and they end with step 4. A copy reads the log's place and the store's epoch in one
hold, and no record of the table. So a copy holds
each record the log told at the copy's place, and none it did not, beside every writer that holds the turn. A
record the store took and the registry did not get is in no copy until it is laid (step 1):
one written through the store's own entries, once its host has had that code follow the
store. A copy's statuses are the
ones the log last told at its place, and not the objects': an object can stand one tell
ahead of the log. The store keeps what the log last told of each record's status, beside its
epoch and its tail. A copy reads the log's place, the epoch and those statuses in one hold
of the membership lock, which it holds for one status a record and for no record. A copy
takes no object's lock. Three writes move the log's place outside the keeper's turn, and a
copy stays whole beside each: a status tell changes no fact, and a generation's apply and a
term's start change no record. The told status has five readers: the next status event, for
the status a record moved from, the event of a record changed in place, which carries it, a
first copy, a tell the store refused, whose object takes it back, and a standby's applier or
a watcher, which lays it on the object. No decision reads it. An operator's read of a few
records is read the same way, and the store's own read for it is deleted. The store's own
cut of its table has no caller in the product and ends with step 4.

**A watcher's events through the worker objects [built].** A gateway's watcher hands each
event of a controller's worker watch, and the watch's first copy at its last page, to the
code that keeps the gateway's worker objects (the third part of step 2), as a standby's
applier does. So the object of a record stands when the event that told it returns, and the
host of the watcher follows nothing behind the watch. A first copy keeps the object of a
record that stands through it. The store's two entries for a watcher, a told change and a
first copy, have no caller in the product but that code.

**An object's status leads [built].** On the replica that leads, a status is written on the
worker object and the store is told it (the first part of step 3). A probe's outcome goes to
the worker registry's own health machine, which moves the object. The code that keeps the
objects then tells the store the status that stands, under the object's own lock, through one
entry that asks the writer's lead under the membership lock. A move the store takes no word
of is taken back in that same turn of the object's lock: the object takes the status its
record holds or, where no record stands for it, the status it left. A registration call's
connect signal and its drain go through
that entry too: the store is told first, and the object moves where the store took the
status. The locks in their order: the object's own, then the store's membership lock. No
holder of the store's lock takes an object's, and no decision takes one. From the table to an
object there are two ways, both of a told status. A standby's applier and a gateway's watcher
lay each status their leader tells, the status of each record it tells, and each status of a
first copy. A tell the store refuses gives the object the status its record holds. The
code that keeps a
controller's objects copies no status from its own store's tail: with that copy, two moves in
a row could end with the object at the older one. The store judges no health outcome: its
own machine for one is deleted.

A lease keeps naming its worker by the record's id, never by a place in a list: a gateway
reports on a lease later and across membership changes.

## 11. A gateway under a controller **[planned]**

One more option, `--grand-central-address HOST:PORT`, puts a gateway under a controller. Then
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
   │── Decide: request facts ─────────────────────>│  InProcess::decide_without_lease over its own lists
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
- The gateway's side is the same four functions of section 9, made asynchronous: an answer from a
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
  it. Nothing counts it toward that worker's health yet **[planned]**. In process the regular HTTP
  router's typed and pass-through routes tell the gateway's own engine the outcome, and its other
  routes do not yet (section 9.2).
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
writes the sheet. The sheets are in `crates/grand_central/tests/classic/sheets/`. A crate
test replays each line through the crate and compares.

On main: 20 sheets, 512 lines.

| | Sheets | Lines |
|---|---|---|
| replayed and passing | 19 | 507 |
| recorded, replayed by nothing (`constraint_decider`) | 1 | 5 |
| replayed through the engine that stays (`InProcess`) | **6**, and 31 lines of `prefix_hash` | **140** |
| replayed lines held to an answer of the crate's own, not classic's | | 7 |

A replay of all sixteen decision sheets through `InProcess` over worker objects is in work and not on main yet: on main the three hard-demand sheets (`retry_wire`, `media_refs`, `context_window`; 26 lines), `sticky_rank` (8 lines), `encode_pair` (52 lines), `pair_named` (23 lines) and 31 lines of `prefix_hash` replay through it. Measured
with it locally on 2026-11-22, on main (`096c0f5ea`): of the 408 lines that recorded a decision
then (397 of the 438 of point 1 below, and the 11 of a rank sheet that has left with its rule), the engine answers 286 as classic recorded and 28 differently, and 94 cannot
be asked yet. Of the 94, 25 are declined because the request states a hard demand, 16 need a
word a request cannot state, 52 are encode requests and 1 is refused at validation. Runs with
one input left off the worker objects (the requests in flight, the load reports, the status,
the ring) lose lines, so the replay can fail.

Five things follow, and they are the first work of section 16.

1. **Only the three hard-demand sheets, `sticky_rank`, `encode_pair`, `pair_named` and 31 lines of `prefix_hash` (140 lines) go through the engine that stays** in main's own tests. The other replays of a decision sheet
   call the decision function over hand-built candidates. Sixteen sheets
   (438 lines) record a decision, and each has to replay through `InProcess` over worker
   objects. The other four (74 lines: `overload` 25, `pair_index` 23, `minimum_tokens` 18,
   `sticky_rank` 8) record an answer of the host, which the engine is never asked for. It
   reads the overload verdict that the host's load monitor wrote on the worker object. It is
   handed each prefill's partners with a pair's two lists. It names the rank a worker
   registered, never one of the ranks behind a worker that stands for all of them. In a host's
   own process those answers are classic's own code, which stays, so the 74 lines hold there
   with no replay. For the two rank sheets that rests on a rule of the gateway's change
   (section 9.2). On main `pair_index` replays through the pairing rule a controller derives its pair classes by (`PairClasses::derive`), `sticky_rank` replays over rank objects through `InProcess`, and `overload` (25 lines) replays as recorded through the worker crate's load monitor, built as a host builds it, where the worker is one object for all its ranks. Eleven of its lines answer differently where each rank is a worker object, by the one rule stage 4's line names (section 16). `minimum_tokens` (18 lines) replays as recorded through the rank cache of that load monitor, where the worker is one object for all its ranks or the object of its one rank. No test replays it through the engine's pick among rank objects, which stage 4's line counts (section 16).
2. **23 lines were held to the crate's own answer** (`minimum_tokens` 10, `overload` 6,
   `sticky_rank` 4, `prefix_hash` 2, `manual` 1). The 16 of `minimum_tokens` and `overload` were held by the old engine's store. The 6 of `overload` replay as classic recorded now, through the host's load monitor (point 1). The 10 of `minimum_tokens` replay as classic recorded now too, through that monitor's rank cache (point 1); three of them are rules of the scorer and the picker, which the engine shares where it picks among rank objects. The 4 of `sticky_rank` are held the same through `InProcess`, by the ring and the picker. The
   other three are the engine's. The 2 of `prefix_hash` differ from classic there too (a
   key folds whole blocks). One of them turns in the crate on a ring its caller lends
   (decision 3). The other measures which tokens are in a prompt's head, which is then the
   caller's recipe: it is proven at stage 3, by a gateway test of classic's own function. The
   1 of `manual` replays as classic recorded now: a sessions rule runs on the two sides of a
   pair, each side by its own session and under its own cap.
   One answer under a cap differs from classic's, by decision 13 (section 15): the engine moves a
   prefill side's pin where the pair forms, and classic where it selects the prefill. Each
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
   candidates. Through `InProcess::place` over worker objects, with the ring lent and no place
   stated, the 31 hold their recorded answers too: the engine cuts each request into its heads
   and the line's ring answers for each
   (`the_classic_prefix_hash_rows_replay_through_the_one_call_with_no_place_stated`). The other
   16 measure which tokens are in a prompt's key (15 lines) or how
   2,000 prompts spread over eight workers (1 line); the second held line is one of the 15.
   With a lent ring a request's place is the host's to state, so these 16 test the host's
   recipe. In the crate they stay on the crate's own ring with its own hashes, which a caller
   that lends no ring still runs: 15 hold classic's shares, as on main, and one is held apart.
   A gateway runs that ring for some requests, among them a shadow profile whose live profile
   reads the ring too, by another word: such a shadow is lent no ring (`Engine::unlent`). A
   single binding's shadow beside a live profile that reads no ring is lent it, and stated its
   own place.
   Classic's key recipe and its ring lookup are functions that classic and the lender both call
   (`PrefixHashPolicy::ring_key` and `HashRing::walk_from` in `model_gateway/src/policies/`), and
   gateway tests place the policy's own requests through classic and through the engine, worker
   for worker (`model_gateway/src/routers/common/placement.rs`) **[built]**. A gateway test that
   puts all 47 lines through both is not written **[planned]**.

Hosted CI's first test line of the crate enables the feature `client`. It takes `controller`,
which takes the feature the engine is behind, so that line runs the engine's own tests.
Decision 11 says how this was closed.

### 13.2 Speed

Classic's cost is counted in the gateway's decision bench: instructions for one decision, for
each classic policy at 8, 64 and 512 workers. That count is the mark the engine has to meet.

The crate's bench of the engine over worker objects is `crates/grand_central/benches/in_process.rs`
**[built]**: a cell is one document over a host's worker lists, with the request its callers make.
Every change to a policy shows its cell before and after, with classic's count beside it.

### 13.3 On a cluster

The engine has run beside classic on a cluster twice: 72 engines, 1,024 clients, single requests
in conversations of several turns. "Same worker" is the share of a conversation's next turns that
went to the worker of the turn before.

**Through the gateway's option, 2026-11-23.** One gateway ran with `--grand-central` off and
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
| A binding's overload switch (`overload.enabled`) under worker objects | **[built]** | |
| Bindings by tenant, priority, header, prompt size, labels; a session id issued by the engine (`issue_session_id`) | **[planned]** | in the schema; a document that writes one is refused as not in this build |
| `canary`, `override: allowed`, `admission` | **[planned]** | in the schema; a document that writes one is refused, and the error names the field |
| `shadow`, `explain.sample` | **[partly]** | the kept engine runs a single request's shadow, but not one that reads affinity rows of its own; it runs a pair's shadows by side at the pair's first decision, and the decode side's at a decode decided again, in a host's own process; a gateway takes a document that names them, and no served controller asks them, and `explain.sample` is kept by the served controller and, for a single request and a pair's first decision, by the engine in a gateway's process, where the gateway answers a record at `GET /policy/explain/{request_id}` (decision 7) |
| The ten presets and the conversion from routing settings | **[built]** | |
| A worker's `policy` label names its model's policy | **[partly]** | section 3.3: built: the builder of the settings' document takes a label and binds its model, and a registration that states a label is answered with a warning; planned: the gateway hands the builder its labels, and a document put in force keeps the stages of a profile that did not change |
| The stage interfaces and a builder's own stages | **[built]** | |
| The engine over worker lists (`InProcess`) | **[partly]** | refuses a subset hint and mixed protocols; of the recorded lines only the three hard-demand sheets, `sticky_rank`, `encode_pair` and `pair_named` (109), and 31 lines of `prefix_hash`, replay through it |
| `round_robin`, `random`, `passthrough`, `least_load`, `power_of_two` on that engine | **[partly]** | no sheet over a healthy pool for four of them |
| `cache_aware` on that engine | **[partly]** | a single request whose maker names its entry by token ids (`RequestMaker::token_key`) is decided in the host's token tree, where the host lends it, the model has no index of cache events and the profile reads sent prefixes and names the holders' rows, as the preset does: the record's first two faults do not reach it. Every other request the engine names the holders of reads the engine's own record, with its three faults, the third being that a request with token ids reads it for no worker the index holds a block for, unless its profile reads the record alone: a request whose caller names no entry is one. A request its host holds as text is decided the same way in a lent string tree (`RequestMaker::text_key`), with or without an index. No gateway lends its string tree yet |
| `consistent_hashing`, `prefix_hash` on that engine | **[partly]** | placed on the ring the host lends (decision 3): through `place` the plain ring at the place the engine reads with the request's keys, and `prefix_hash` at the places the host's ring answers for the request's heads, where it answers for a head; that both are classic's is proven at stage 3; a host that lends none is placed on the crate's ring, which may be another worker than classic's |
| `manual` and sticky sessions on that engine | **[partly]** | their sheets replay through the decision function, not through the engine |
| `bucket` on that engine | **[partly]** | the same |
| Prefill-decode pairs on that engine | **[partly]** | the sheet of classic's pair choice (`pair_select`) replays through the pair's decision function, not through the engine; one of its 40 lines is held to an answer of the crate's own; the 8 lines behind the prefill admission's bound (`pair_bound`) and the 10 lines of a pair on one known runtime (`pair_runtime`) replay through that function as recorded, the 10 also over the classes a controller's pairing rule derives; the 23 lines of a pair that names its worker (`pair_named`) replay through the engine's one call for a host in process |
| Encode-prefill-decode on that engine | **[partly]** | `InProcess::decide_encode_at` decides it over a host's three lists and issues the pair's lease, and `place` places it; the entry declines a request held to a wire that a worker of its pair's two lists is not dialed on, places its items on the encoders dialed on a wire it states, declines a request with items where the list holds encoders and none is, and declines one that states the later decode where its decode is held to the workers that take images, or where it carries items and demands vision or a way of processing its references: a decode asked again takes no items |
| The lease and its events; the guard | **[built]** | |
| The host contract of section 8.2 (`place`, `Placement`) | **[partly]** | `Placement` is built, with its call for a pair's decode decided again, and `place` for a single request, for a pair and for an encode request, with the route's rule and the decision made again, three decisions a call at most, and the later decisions of a single request, of a pair and of an encode request held to the workers the route's rule takes; an encode request's header for a target is not read |
| The crate's bench of the kept engine | **[built]** | |
| The gateway as a host | **[partly]** | section 9, behind `--grand-central on`, which is off by default: single requests and pairs on the HTTP and gRPC routes, a pair's decode asked again, a host's table of stages at the start; an encode request that carries an item is refused by name until the gateway hands the engine its three lists; every classic test of the gateway with the engine on (the bar's fourth point) is not met yet |
| The controller: service, election, replication, document apply and roll back | **[partly]** | a session never pushes the document and refuses a gateway's replay; a standby serves neither the worker list nor the snapshot (section 10.1) |
| The controller on the one engine and the worker crate's structures | **[partly]** | a single request, a pair at its prefill, its decode call and an encode request that carries its items are decided by `InProcess` over the registry's lists; a decision reads no load report under a document that holds a stage that keeps reports, as no stock stage does (section 10.2) |
| The controller finding its own workers | **[partly]** | the image leaves the feature out; a worker's API key is refused; no flag sets a registration's timing |
| A gateway under a controller | **[planned]** | section 11 |
| User stages in WebAssembly and over gRPC | **[partly]** | library code that no document reaches |

**When it is ready.** Three tests, all of them: requests are decided over the real data (the
worker crate's registry, loads and cache records, not a copy); the speed is classic's, by count;
and it has served traffic on a cluster beside classic.

## 15. Decisions

Each is the maintainer's. Work inside the crate goes on where a decision is not needed for it.

All seventeen at a glance. An answer can be as short as the numbers with their letters.

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
| 11 | The crate in CI and in the published set | closed inside the crate: the feature `controller` takes the engine's feature; (a) out of the published set until stage 3, and as it went (b): the crate stayed in it | CI's run of the kept engine's tests; stage 0 |
| 12 | The last report of a worker whose load call fails | (a) the engine alone reads it, about 40 seconds at most at the default poll | rule 6, for the documents whose stages read load reports |
| 13 | Where a capped sessions rule moves a pair's prefill pin | (a) where the pair forms | rule 6, for a pair under a sessions rule that states a cap |
| 14 | Round robin's order across deciding threads | recorded, the answer is open: (a) a thread takes a list's turns in runs | rule 6, for `round_robin` where several threads decide |
| 15 | The workers a session's first request reads | recorded, the answer is open: (a) the workers its turn asks about | rule 6, for a sessions rule that leaves a new key to round robin |
| 16 | `least_load`'s count of a request that is never sent | decided 2026-11-24: (a) the engine counts a send at its dispatch | rule 6, for `least_load` where a route refuses a request behind its selection |
| 17 | The pick among workers that tie at the fewest requests | decided 2026-11-25: (a) the engine draws among them | rule 6, for `prefix_hash` where it places a request by load |

### Decision 1: the name of a cache partition

*Needed for:* the host's two trees, which the engine reads for `cache_aware` (section 4.5). A
request's entry in a tree must be named as classic names it.

| Option | Cost |
|---|---|
| (a) The host hands the entry already named, with the length of the partition's marker at its start. The engine takes it as it takes the prompt's hash seeds today. | About 70 lines in the crate, and about 15 lines (an estimate) in the gateway's one integration, calling classic's own two naming functions, one for each tree. The naming is then the host's to keep: a host that hands a plain entry for a partitioned request lets two tenants' prompts match. |
| (b) Classic's naming (336 lines) and its two functions move into the kv-index crate. | A move across 12 files, 10 of them in the gateway. No behaviour changes. |

Under (a) the engine also states for itself which of the three records a request reads (the rule
of section 4.5), about 12 lines. Classic has that rule inline, so there is no function to call.
Classic's `cache_aware` sheet has lines for all three records (12, 13 and 3) and holds the
engine's rule equal to classic's. The other way is one function in the kv-index crate that
classic calls too: about 90 lines, and it touches the gateway.

**Recommendation: (a).** Nothing is copied and nothing moves. Work on it is on main, in the crate
and in the gateway: the gateway hands a tokenized request's entry already named, by classic's own
naming function, and lends its token tree.

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
| (a) The host lends the ring it keeps; the engine asks it which workers a routing key reaches. | Crate only: one deletion and four changes. The crate's ring beside a worker list is deleted (1,728 lines removed, 230 added, 17 files). The four changes are about 580, 270, 450 and 550 changed lines, about 370 of them product code and 1,050 proof. The host lends its ring through one trait (`LentRing`) and states where a request stands on it. For such a request the crate then reads no ring of its own, so the request is placed on classic's own ring. The price in proof: 16 of the 47 recorded `prefix_hash` lines measure which tokens are in a key (15 lines) or how prompts spread over workers (1 line). In the crate they keep replaying on the crate's own ring. One of the 15 is held apart (section 13.1). The other 14 and the spread line hold classic's shares, as on main. For a gateway they are proven at stage 3 (section 13.1). On main classic's key recipe and its ring lookup are functions that classic and the lender both call. A controller has no classic ring to be lent: until (b) it keeps its own ring and places a routing key on another worker than a gateway would. |
| (b) Classic's ring (one file, 508 lines) moves to a crate both sides use. | A pure move, about 15 lines changed beside it. The gateway, the engine and the controller then build one ring type. |
| (c) The crate's ring is changed to classic's hashes. | About 125 lines. Classic's recipe written twice; a gateway would keep two rings for a model. |

**Recommendation: (a) now, (b) with the controller's switch.** Not (c). Work on (a) is on main in
the crate and in the gateway: the gateway lends the ring it keeps, for a key's own text and for a
request's heads.

### Decision 4: where a placement stays during a request

*Needed for:* the gateway's integration (section 9).

| Option | Cost |
|---|---|
| (a) Inside the load guard: the worker crate's `WorkerLoadGuard` gets one optional field. | One small addition to the worker crate. No route keeps the placement itself: every route already keeps the guard for as long as the request is on its worker. A route that tells an outcome keeps a handle on the request's end as well, in its answer or with its stream's reader (section 9.2). |
| (b) Each route keeps the placement beside its guard. | No change to the worker crate. Every route changes, which is what made the earlier integration wide. |

**Recommendation: (a).** Taken with the integration, not before.

### Decision 5: how the controller comes onto the one engine

| Option | Cost |
|---|---|
| (a) Whole: the kept engine is finished and proven in the crate, the controller's parts are added beside the old engine, one small change switches the deciding threads, one deletion follows. | Nothing of the new path runs in a controller until the switch. The types both engines share move out of the old engine's file first (section 5.7). |
| (b) Document by document, behind a test at apply. | A test at apply, two load pollers for a time, a temporary call, and refactors inside the old engine that are deleted afterwards. |

**Recommendation: (a).**

**As taken.** The deciding threads were switched for a single request first (section 10.1), and
the store-backed engine's type was deleted behind that step. A pair at its prefill, its decode
call and an encode request that carries its items are decided by the one engine over worker lists
too.

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
rows of its own. It runs a pair's shadows by side at the pair's first decision, and the decode
side's at a decode decided again, where a host decides in its own process. It counts them by
side, a decode decided again apart. A side its binding's sessions rule or its picker's
assignment places has no shadow decision yet: it is counted as skipped. A host reads the shadow
counts in one call of the engine (`InProcess::shadow_counts`). A gateway shows them at each
scrape of its metrics endpoint, as the series `smg_policy_shadow_decisions_total`. A gateway
takes a document that names a pair's shadows, and a served controller asks no side's
shadow. The explain sample is kept by the served controller, and by the engine in a gateway's
process for a single request and a pair's first decision. There an encode request decided with
its items, a decode decided again and a pair's shadows are in no record. The gateway answers a kept record at `GET /policy/explain/{request_id}`, in the JSON the crate writes one in. It states no id of its own yet, so a request whose body states none is in no record. A document
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
`--grand-central on|off`, `--policy-spec FILE`, `--print-policy-spec`. A gateway under a
controller (section 11) brings `--grand-central-address` and three TLS options.

**Recommendation:** the first three with section 9's change, the others with section 11's.

### Decision 10: what the controller needs in the worker crate

*Needed for:* section 10.2. Not needed before the controller's switch.

| Need | Option recommended | Cost |
|---|---|---|
| A model's lists by role and protocol, with the engine's value beside each | one type in the worker crate that the gateway and the controller both use | 279 lines added; the gateway moves onto it at its integration |
| A load monitor that a controller can stop when it stops leading, and start again | one public `stop()` on the worker crate's monitor | about 110 lines with its test |
| The facts of a worker's record that a worker object does not hold as the record states them, among them zone, the encode offer, vision, the admission window, the demand facts and pairing | the controller keeps them beside the object (10.3, built); a fact moves to a field of the worker crate's object when that crate has one that holds it as stated, checked against the crate first | 245 lines of code for what is kept beside the object |

One more point under this decision. A document can write an `overload` block for each binding, and
a worker may state thresholds of its own. A worker object holds one set. **Recommendation:** one
set for each worker, the worker's own values over one default of the program, as the classic
policies have it. An external worker takes its own values alone, over no default. Classic
registers it with no threshold at all. What that leaves of a binding's block:

- **Its two numbers [built].** A block names none. A document that names one is refused when it
  is applied, by the number's path, and the refusal says where a number is stated: a worker's
  registration, or its program's default. The two that a gateway's routing settings state are
  the defaults its worker objects are built with where the switch below is on, and the crate's
  translation of those settings writes neither onto a binding. A gateway states them on its
  command line, beside a supplied document too. A standalone controller states them on its own
  command line, by the gateway's two flags, and its worker objects are built with them: the
  flags' defaults are 8 queued requests and 0.8 usage.
- **Its shed word [built].** A binding keeps whether a request is shed when every worker is
  vetoed. A decision reads it.
- **Its switch [built].** Whether the program's default vetoes at all is one word for the
  bindings a host runs, because the default is the program's: one for every worker object that
  takes one. The host reads it where it
  builds that default. A gateway that runs a supplied document reads it at its start, and
  refuses a document whose bindings disagree on it, or on the shed word: its routes read one
  shed word for the process. It also refuses one whose block is on where the gateway states
  neither number. With the switch off the gateway builds its worker objects with no default. A
  binding with no block says the defaults. A standalone controller reads it at each document
  that takes force: its document door states the word on the holder of its worker objects
  behind each generation's entry, under the door's lock, and the objects the word changes are
  built again before the entry returns. Its default is off where every binding switches it
  off. Bindings that disagree keep it on, and a controller refuses no document for them. At a
  standby's restore the snapshot's rows are built ahead of its generation: where that
  generation moves the word, a row it changes is built twice. A worker's own thresholds veto
  either way.

### Decision 11: the crate in CI and in the published set

| Question | Fact | Options | Recommendation |
|---|---|---|---|
| How does CI run the kept engine's own tests? | The kept engine compiles only under the feature `in-process`. The feature `controller` takes it. The first of CI's four test lines of the crate in `scripts/ci_rust_tests.sh` enables `client`, which takes `controller`, so that line runs the cases behind `in-process` and the test target `caller`. | (a) The crate's default build includes the feature: one line in the crate's manifest. Measured: two of CI's four lines then run the 82 cases, in under a second more. But a default build of the crate goes from 147 to 272 packages, the controller's image and three bench workflows then build the engine in, and two READMEs become wrong, one of them outside the crate. (b) CI's first test line names the feature: one line in the script, outside the crate. That line builds the same 364 packages with it and without it, and nothing else builds differently. (c) The crate's test-only dependency on itself names the feature: inside the crate, and no product build changes. But every build that takes the crate's test-only dependencies then builds the engine too: CI's two lines that test one feature alone, the lint of `operator` alone, and the three bench workflows that (a) names. Not measured. | Closed a fourth way, inside the crate: `controller` takes `in-process` and `worker-host`, the two features the controller's switch (section 10.2) builds on. A build with `controller` or `client` then compiles the kept engine and the worker objects, and a controller built with `controller` alone keeps worker objects as the stock binary does. No line outside the crate changed. |
| Is the crate published before it is ready? | The earlier integration added the crate to the published set: four lines in two release workflows and the release version check. | (a) Stage 0's change takes the four lines out, and they return with the gateway's integration (stage 3). (b) They stay. | (a). As it went: (b). No change took the lines out, and the gateway's integration is on main. |

### Decision 12: the last report of a worker whose load call fails

*Needed for:* rule 6 (section 1), for the documents whose stages read load reports.

A worker's load call can fail while the worker goes on serving. The worker crate's load monitor
drops the worker's report at that poll. Classic then scores the worker as one with no report:
`least_load` starts it from the score of its best reporting peer. So a worker that had stated a
full queue can be drawn again at the first failed call.

The monitor keeps such a report beside its entries where the report states a running or waiting
request. The kept report ends when the worker's load call answers again, when a load record of
the worker arrives on its event stream, and when the worker leaves or is no longer ready. It ends
no later than the first poll that finds its sample 30 seconds old: at the default poll of 10
seconds, about 40 seconds after the sample. One read gives it, `LoadSnapshot::report_or_kept`,
and the engine's rows and signal cells take that read. Classic's reads, `LoadSnapshot::report`
and `LoadSnapshot::get`, give no kept report.

So while a report is kept, the engine and classic can answer differently under the same routing
flags. The engine charges the worker its last report and every request sent to it since that
report's sample. Classic scores it as a worker with no report.

| Option | Cost |
|---|---|
| (a) The engine alone reads the kept report. | Classic's answers do not move. With the option on, a worker whose load call fails while it serves is charged its last report for up to about 40 seconds at the default poll, and can be under-used that long. In that window the engine differs from classic, by this decision. |
| (b) Classic's reads take the kept report too. | The two agree. Classic's answers change for every policy that reads a report, and each of those reads pays a second look-up where a worker has no entry. |
| (c) No stage reads it. | The two agree, and classic's answers do not move. A worker that stated a full queue is scored from its peers at the first failed load call, and can be drawn again. |

**Recommendation: (a).** A failed call is not a worker that never reported. The kept report is
bounded by its own sample, and classic stays as recorded.

**As taken.** (a) is on main: the monitor keeps the report, the engine's rows and signal cells take
it, and classic's reads are as they were.

### Decision 13: where a capped sessions rule moves a pair's prefill pin

*Needed for:* rule 6 (section 1), for a pair under a sessions rule that states a cap.

A key whose pin is at its rule's cap is placed again, and the pin moves where the request lands on
another worker under the cap. For a single request and for a pair's decode side, classic and the
engine both write that move where the side is decided. They differ on a pair's prefill side.

Classic selects a pair's prefill first, and its sticky map moves the pin at that selection
(`select_sticky` in `model_gateway/src/policies/registry.rs`). The pair can still fail behind it, as
where the decode's policy declines (`pair_pass` in `model_gateway/src/routers/common/placement.rs`).
The move stays, on a prefill that took no request.

The engine's prefill side places the request again and moves no pin. A prefill whose partners
leave no decode is passed over. The pin moves where the pair forms, to the prefill the pair formed
on and ahead of any dispatch (`Engine::decide_pair` in `engine/framework.rs`). A decision that forms
no pair moves none.

So behind a pair that fails past its prefill's selection, the key's next request finds the pin
moved under classic and where it stood under the engine.

| Option | Cost |
|---|---|
| (a) The engine moves the pin where the pair forms. | Behind a pair that fails past its prefill's selection the engine differs from classic on the key's next request, by this decision. Classic's answers do not move. |
| (b) The engine moves the pin at its prefill side's run, as classic does. | The key's next request finds the pin as classic's does. A prefill that is passed over then holds the pin while the decision runs. A request of the key decided in that time follows it there, and the pin is taken back from under that request or stays on a prefill the pair never formed on. |

**Recommendation: (a).** A pin says where a session's requests went. A move for a pair that never
formed names a prefill that took none.

**As taken.** (a) is on main: the prefill side's run moves no pin, and the pair's decision moves it
where the pair forms.

### Decision 14: round robin's order across deciding threads

*Needed for:* rule 6 (section 1), for `round_robin` where several threads decide.

Classic's round robin takes one step of its list's counter at every pick
(`model_gateway/src/policies/round_robin.rs`). Requests decided on any threads then take the list's
workers in one order. Where every worker can take a request, the picks of two workers differ by one
at most.

The engine's round robin takes single steps too for a thread's first turns of a list, 64 for each
worker. Past them the thread takes a run of the counter's turns in one write, 61 as a rule, and
plays them in order (section 5.6; `row_of` in `policies/round_robin.rs`). Between two such writes a
pick whose turn the filters keep writes no word that the deciding threads share.

So where several threads decide, the engine's picks follow no one order: inside a run a thread
takes the list's workers one after the other. Where the filters keep every worker and the counter
has not wrapped, the picks of two workers differ by one at most, and by one more for each run that
is not played out: the run a deciding thread holds, and each run that was left behind. Where one
thread alone has decided over a list, it takes the list's workers in the order single steps give,
for as long as the list keeps its place among the thread's four. A list made anew where a dropped
list of the same size stood first plays the turns the thread had left of that one, fewer than one
run.

| Option | Cost |
|---|---|
| (a) A thread takes a list's turns in runs. | Where several threads decide, the order of the picks differs from classic's, and the even share is bounded as above and not exact. |
| (b) Every pick takes one step of the counter, as classic does. | The two agree. Every pick writes one word that all deciding threads share, and a decision slows as deciding threads are added, by the count of the change that took the runs. |

**Recorded, not decided.** (a) is on main. The maintainer's answer is open.

### Decision 15: the workers a session's first request reads

*Needed for:* rule 6 (section 1), for a sessions rule that leaves a new key to a round robin
profile.

A routing read of a worker moves its open circuit breaker to half-open once the breaker's timeout
has passed (`CircuitBreaker::state` in the worker crate). So the workers a request reads decide
when a tripped worker is tried again.

For a key with no pin, classic's sticky map first lists the workers that can take the request, and
reads every worker of the list for it (`select_sticky` in `model_gateway/src/policies/registry.rs`).
Then its policy picks.

The engine reads fewer workers where it draws for such a request. Over a list of 16 workers or
more it reads the worker the list's counter points at, with no turn taken. Where the filters keep
that worker, it reads the workers the draw then takes and a worker that another request claimed
for the key meanwhile (`Engine::decide_claimed` in `engine/framework.rs`). Where they keep the
worker the draw's own turn lands on too, the decision's cost does not grow with the list. In the
other cases, a worker the filters refuse at either of those two reads among them, the engine
reads every worker, as classic does.

So a tripped worker whose timeout has passed turns half-open at any session's first request under
classic, and under the engine at the first request that asks about that worker. An answer can
differ. Behind a session's first request, a failure of an older request on such a worker opens its
breaker again under classic and changes nothing under the engine. A request pinned to that worker
is then placed on another worker by classic and on its pin by the engine.

| Option | Cost |
|---|---|
| (a) The engine reads the workers its turn asks about. | A tripped worker is tried again at another request than under classic, by this decision. |
| (b) The engine reads every worker, as classic does. | The two agree. The decision grows with the list: every worker's record is written and filtered before the profile takes its turn. |

**Recorded, not decided.** (a) is on main. The maintainer's answer is open.

### Decision 16: `least_load`'s count of a request that is never sent

*Needed for:* rule 6 (section 1), for `least_load` where a route refuses a request behind its
selection.

Classic's `least_load` credits the worker it picks among several with the request's work at the
pick (`decide` in `model_gateway/src/policies/least_load.rs`). The credit stays until a load report
of that worker sampled behind the pick (a worker keeps its newest 8,192), and classic takes no pick
back. So a request that is selected and never sent weighs on its worker until that report.

The engine's counts of sends take a send at its dispatch (section 6). A request that is never sent
enters none of them.

So behind a request that a route refuses between its selection and its send, classic's next picks
count that request on its worker and the engine's do not (section 9.2). An answer can differ until
the worker's next report.

| Option | Cost |
|---|---|
| (a) The engine counts a send at its dispatch. | Behind a request that is never sent the engine's picks can differ from classic's until the worker's next report, by this decision. Classic's answers do not move. |
| (b) The engine credits a pick where it is decided, as classic does. | The two agree. A request that is never sent then weighs on its worker in the engine too, until the worker's next report. |

**Decided 2026-11-24: (a).** The engine's count stays, and a request that is never sent is not
counted.

### Decision 17: the pick among workers that tie at the fewest requests

*Needed for:* rule 6 (section 1), for `prefix_hash` where it places a request by load.

Classic's `prefix_hash` places some requests by load, among them a request with nothing to hash: no
routing key, no tokens and no text. For that request it takes the worker with the fewest requests in
flight among those it may use, and of several at that count the first in the order of the list it is
handed (`least_loaded_healthy` in `model_gateway/src/policies/prefix_hash.rs`). Classic's own
`least_load` draws among the workers that tie, so that an idle fleet's requests spread
(`TIE_EPSILON_SECS` in `model_gateway/src/policies/least_load.rs`).

The engine's profile for `prefix_hash` adds the weighted scores of `consistent-hash` and
`least-requests` (section 7). For such a request `consistent-hash` scores every worker alike, so
`least-requests` orders them, and the picker `best` draws among the workers at the best score, from
the decision's seed.

So where several workers tie at the fewest requests in flight, classic takes the first of them and
the engine draws one. That holds where the policy itself places: for a single request, and for each
side of a pair.

| Option | Cost |
|---|---|
| (a) The engine draws among the workers that tie. | Where several workers tie, the engine's pick can differ from classic's, by this decision. Classic's answers do not move. |
| (b) The engine takes the first of them in its list's order, as classic does. | The two agree. Each request that finds its workers at one count then goes to the first of them. |

**Decided 2026-11-25: (a).** The engine's draw stays.

## 16. The order of work

The crate first (rule 7). Each stage ends with a test, and the next does not start before it.

| Stage | Work | Ends when |
|---|---|---|
| 0 | The earlier gateway integration is taken out. | The gateway routes by classic alone; every classic test and bench cell is as before. |
| 1 | **The policies, proven in the crate.** Each classic policy decided by `InProcess` over worker objects: its sheet replays through that engine, and its bench cell meets classic's count. | Every line of the 16 sheets that record a decision (438 of section 13.1's 512) replays through the kept engine over worker objects. Each difference from classic is fixed or accepted as a named difference. Under decision 3 (a) the 16 `prefix_hash` lines that measure a key's tokens or a spread of prompts replay there with no ring lent, and their proof for a gateway is stage 3's. |
| 2 | **The host contract.** `place`, `Placement`, the engine's own upkeep and membership events (section 8.2). | A test host in the crate serves every plan through `place` alone. |
| 3 | **The gateway, one change** (section 9), off by default. | The four points of section 9.2's bar; under decision 3 (a), the 47 `prefix_hash` lines and the 12 ring lines of `all_vetoed` read as recorded, count for count, through the gateway's own entry with the engine on and the recorder's own workers and prompts; a cluster run beside classic. |
| 4 | **The controller on the one engine** (section 10.2); the old engine is deleted. | The controller's own tests pass on `InProcess`; the store-backed engine is gone; each of the 74 lines of the four host sheets (section 13.1) is answered by the host's code that the controller takes from the worker crate, or by a rule of the crate's with its differences named. Differences are named so far on two of the sheets. The `sticky_rank` lines that the crate decides apart answer by the ring and the picker, where the gateway sets no rank (section 13.1). 11 `overload` lines answer by one rule: where ranks are worker objects, a threshold that judged an engine's summed queue judges each rank's own queue, and a usage threshold each rank's own usage in place of the mean (two ranks that queue 4 requests each under a threshold of 8 are vetoed as one worker and not as two). The 18 `minimum_tokens` lines are answered as recorded by the worker crate's rank cache behind that monitor, which no code of a controller asks. They are not answers of the engine: it serves no rank option (`refuse_unsupported` in `document/routing_settings.rs` refuses classic's at the start), and where ranks are worker objects it picks among them by its profile. That pick was measured once, on 2026-11-23 over `55c35100d`, outside the tests: under a resident-work profile with a picker of two it names the recorded ranks on 5 of the 18 lines and other ranks on 13, by four rules. On 6 lines classic finds no rank on a worker its cache holds nothing of, and the engine places on a ready object: one that has no report (`with_no_survivor_reporting_resident_tokens_the_in_flight_count_decides`) or, where the answer states no token capacity, one it scores by its resident tokens. On 3 a rank that the answer omits or reports with no token capacity is no rank to classic. The engine scores an omitted rank from its best reporting peer (`a_worker_without_resident_tokens_scores_from_the_best_reporting_peer`) and a rank with no token capacity by its resident tokens. On 3 the engine reads resident tokens and draws among equal ranks, where classic reads used tokens and takes the lowest rank (`equal_resident_work_ties_and_a_token_apart_does_not`). On 1 classic keeps a worker's last answer and its bookings behind a failed poll, and the monitor publishes no report of that worker (`publish_polled` in the worker crate's load monitor). It keeps none for the engine either: the line's report states no running or waiting request (decision 12). Three of the lines move with how the caller counts a pick in flight and with the picker, and the count stays at 5 or 6. |
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
new recorders (2), the controller's parts (10), shadow and explain on the
kept engine (7).
