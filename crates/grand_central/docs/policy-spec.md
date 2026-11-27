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

[The stages page](stages.md) lists every stage with its settings.

### The tally of sends

A profile can keep a *tally of sends*: what the host sent each worker since that worker's last load report. A stage reads the tally through `StageBuild::dispatch_credit`.

A send leaves the tally at the first of these events:

- A load report of the worker, sampled at or after the send. Where the profile sets `sent_report_delay_ms`, the sample must come at least that long after the send.
- The worker is removed, joins again, is replaced or changes its number of ranks.
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

A host can read these settings for an index of its own. The gateway does: at each sweep it trims each tree of a model toward `max_size`, each tree by itself. Writes take a tree past it until the next sweep. At a `sweep_s` of 4294967295 it starts no sweep, and a tree under that entry is held to no size. The engine's record keeps its bound at any cadence.

The crate's engine reads `kind` and `max_size` alone, for a pool whose profile reads sent prefixes:

- Under `tree`, the engine bounds the model's record of sent prefixes at `max_size` divided by the block size. The record holds one placement for each block a worker was sent.
- At the defaults that is 4,194,304 placements. The record takes about 55 bytes a block, so about 230 MB a model.
- Under `hash`, the pool keeps no record.
- The entry read is the one for the model the request names, under the id by which the first worker of its list lists that model. A model that workers list behind their primary one keeps its own record under its own entry.
- A record read for a model behind a first worker's primary one is bounded by the smallest block size any list asked with, for as long as it lives.
- An id and an alias of one model share one record. A name no model of that worker answers to reads the record and the entry of the worker's primary model id.

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

See also [Signals a decision can read](embedding.md#signals-a-decision-can-read).

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

A binding holds more than its match. The sections below cover `overload`, `sessions`, `routing_key`, `shadow` and the reserved fields. A host's own calls for keys and shadows stand on [the embedding page](embedding.md#keys-and-shadows-a-hosts-calls).

### Overload

A binding's `overload` block switches the default overload thresholds on or off. It also says what a decision does when the overload veto refuses every usable worker.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Whether the host's default thresholds veto a worker. The switch does not touch a worker's own thresholds. |
| `shed_when_all_vetoed` | `false` | `true` sheds a request where the overload veto refuses every usable worker. |

A binding with no block says the defaults.

**No threshold in a spec.** The block names no threshold. A spec that names a number under `waiting_requests` or `token_usage` is refused when it is applied, by the field's path.

**Where a worker's thresholds come from.**

- A worker's registration states its thresholds.
- A threshold the registration leaves out takes its program's default. That is 8 queued requests and 0.8 usage, in a gateway or a standalone controller that sets no others.
- A worker of the `external` runtime takes no default when it registers. A standalone controller judges it by its registration's thresholds alone, and a gateway registers it with no threshold.
- A gateway's update of a worker's properties builds the worker again with the gateway's default, an external worker too.

**What the load monitor judges.**

- A host's load monitor judges two values over a worker's ranks: the sum of queued requests, and the mean active KV usage.
- A rank whose report carries no split of active usage counts its total usage there.
- Reaching either limit sets the veto.
- A worker whose default moves reads not overloaded until a load report judges it again.

**One switch for a process.** A host builds its workers with one default set of thresholds, so a spec's bindings say the switch once.

| Host | What it reads | What it refuses |
|---|---|---|
| A gateway that runs the spec in its own process | The switch and the shed word, from the spec it starts with. | At its start, a spec whose bindings differ on either word. |
| A standalone controller | The switch, on its worker objects, at each spec that takes force. | A spec whose bindings differ on the switch, by the path of the first binding that differs from the first. |

- With `enabled: false` a gateway builds its workers with no default threshold, whatever its two threshold options say.
- A gateway whose two thresholds are both set to none has no default threshold. Its start refuses a spec whose block is on, which would switch nothing on, and takes one whose bindings switch the block off.
- A controller refuses bindings that differ at its start, at an apply, a dry run and a rollback, and where a standby is told a spec by its leader.
- Where every binding says `enabled: false`, a controller's workers take no default threshold.
- A standalone controller's decision reads the verdict its load monitor writes on the worker. Its overload series count that monitor's verdicts.

**When the overload veto refuses every usable worker.** `shed_when_all_vetoed` controls the fallback of the `single` plan, the fallback of a pair's decode pool, and what a plain ring does.

- A worker under its thresholds that only the liveness check doubts is not refused by this veto.
- For a decode pool that a pair uses, the binding selected at prefill supplies the overload policy.
- That binding supplies an encode pool's shed word too, where its plan is `epd`.

### Sessions

`bindings[].sessions` configures sticky placement. The rule pins a key to a worker, and a later request of the key follows that pin where it can. Among what sends it elsewhere: the rule's cap, a filter that refuses the worker for that request, and a target entry that names another worker.

- The rule runs under `single`. It also runs on the two sides of `pd-pair` and of the pair in `epd`, each side by its own session.
- Under `epd` it places no encode item.
- A picker that places by session assignment runs on the pair's two sides of `epd`, and is refused on its encode side.
- The stock controller refuses a rule that states `rule_placed: untold` under `pd-pair` and under `epd`.

| `sessions` field | Default | Meaning |
|---|---|---|
| `sticky_header` | Required without `routing_key` | A header that holds the key. It must be listed in `view.headers`. |
| `headers` | Empty: the sticky header alone | The headers looked up in order. A list that is not empty contains the sticky header. Each name is unique and listed in `view.headers`. |
| `assignment` | `delegate` | How a key seen for the first time picks its worker. |
| `ttl_s` | 14400 | Seconds an idle key stays pinned. Above zero. 4294967295 means no expiry. |
| `sweep_s` | 120 | Seconds between sweeps of idle keys. Above zero. With a finite `ttl_s` the sweep is finite too. |
| `inflight_cap` | 2 | Requests of one key a worker holds before the next is placed again. Above zero. 65535 means no cap. |
| `recovery` | `original` | Which pin leads once both of a key's pins are usable again. |
| `rule_placed` | `told` | Whether a request the rule placed alone is the profile's own. |
| `issue_session_id` | `false` | `true` is refused. |

**The assignment.**

- `delegate` uses the profile.
- `random` draws a worker.
- `min_load` uses the host's counts of requests in flight.
- `min_group` uses the counts of active sessions.
- With the `manual` picker, `delegate` uses `min_load`. A request with no key follows the same assignment and takes no pin.

**The cap.**

- At the cap the rule tries a fresh placement. A pin moves only to another worker below the cap.
- The cap reads the key's counts. A host that decides over its own rows feeds them.
- In a gateway's own process the cap reads no count for a key from a header that `view.headers` does not list. Such a key meets no cap there.
- The standalone controller holds them from its leases, on each worker's object.

**Recovery.** A key keeps two pins at most. `recovery` says where a worker newly pinned for the key goes.

- Under `original` it goes behind the pins the key holds.
- Under `latest` it goes ahead of them.
- Where a pin moves at the cap, its new worker goes ahead under either word.

**A request the rule placed alone.** Under `told` the request is the profile's own: its issue, its dispatch, its completion and its release are told as any request's are. Under `untold` no stage hears its issue or its dispatch, and no count of sends takes it. A completion and a release are told the same way under either word.

**How the stock controller counts a key.**

- It counts a key's requests by the leases it holds. A request counts on the worker its lease books, from its decision to its end or to that worker's removal.
- Decisions made at the same moment for one key each read the count before the others' leases are on file. So a burst of them can pass the cap.
- A request placed with its key counts whatever rule placed it. A pair counts on each side's worker, from that side's issue.
- So a cap that a change sets counts the requests that stood before it.
- A lease a host files directly counts for its key under any rule: through `LeasesStore::issue`, `issue_pair` or `restore`.

**When a worker leaves.** These rules hold where a store tells the session table of its workers, as a standalone controller's does.

- A worker that leaves membership is remembered for the absence's time, and its pins can stand that long. That time is the longest `ttl_s` among the spec's session rules.
- A rule with no expiry counts as four hours, and so does a spec with no rule.
- Meanwhile a pin on that worker reads as not routable. The key's other pin serves where it can, and else the rule places the request.
- A worker registered again inside that time as the worker it was takes back the pins that still stand, under its new id: the same role, models and wires. `recovery` then rules as for two usable pins.
- Another worker under the same registration key takes none.
- A worker that stays away is forgotten at the absence's end: by the store at its tick, and by the engine's session table at the engine's upkeep.
- A gateway's engine runs that upkeep once a second. A controller runs it at each pass of its presence sweep.
- The table and the store each keep 4,096 absent workers at most.
- A standby takes the same steps at the writes it follows, and counts the absence by its own clock. So does a gateway whose store watches a controller.

**When the table is full.**

- A session table is full at 4,194,304 keys.
- A full table records no new key: the rule places each request of such a key with no pin. The keys the table holds keep their pins.
- The table records keys again once keys leave it: at the sweep after their `ttl_s` ends, or with a worker the table forgets.
- A gateway's own engine tells its table of no worker that left, so there keys leave at their end alone.
- So a table full of keys whose rule has no expiry stays full.

### The request's key

A binding names the sources of its requests' key under `routing_key`. The host reads them in order, and the first that holds a value is the key. A ring places by that key, and a sessions rule pins it.

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
| `sources` | Required | One to eight entries, read in order. Up to four client entries and one target entry can stand beside them. |
| `max_bytes`, on an entry | 128 | The most bytes of a value the source hands, from 1 to 8192. |
| `client`, on an entry | `false` | The header tells one client from another, not one session. |
| `target`, on an entry | `false` | The header names a worker by a number, and hands no key. |
| `strip_lineage` | `false` | Take the turn and retry suffixes off a key: `conv_t2_r1` keys as `conv`. |

**An entry.**

- An entry is `header: <name>`, or `rid: true` for the request's own id, where its host hands the engine one.
- A value longer than `max_bytes` is passed over. So is an empty one, and one that is not UTF-8.
- A header is measured against `max_bytes` as sent. The request's own id is measured after `strip_lineage` took its suffixes off, where that field is on.
- A suffix's counter has any number of digits.
- In a gateway's own process a ring's prefix form takes a key from a header alone. It does not read a key from the request's id.

**A client entry.**

- Its value places a request that has no session key on a plain ring, and does nothing else. It lifts no overload veto and pins no session, and the engine counts it under no key.
- An entry marked `client` is read apart, for a request the other entries give no key.
- The first client entry whose header the request carries is read whole.
- A value gives no key where it is empty, or holds a byte other than a tab or printable ASCII.
- `client` is refused beside `rid` and `max_bytes`. A block of client entries alone is refused.
- A header may be named both ways, once without `client` and once with it.

**A target entry.**

- The header hands no key. Its value is a number that counts the workers that can take the request when it arrives, from 0 and in the order of the request's list.
- For this count a worker can take the request where it is ready and under no veto. It also meets the hard demands its role is held to, and passes its host's rule for the route.
- So a number can name another worker, or none, while a worker ahead of it is down.
- The number is decimal digits, 4294967295 at most, with one `+` ahead of them or none.
- A header that is absent or empty names no worker. Neither does a value that holds a byte other than a tab or printable ASCII.
- A request that names no worker is placed as any other.
- A block holds one target entry at most. `rid: true`, `client: true` and any `max_bytes` are refused on it.
- A block with no plain entry is refused, so a target entry stands beside one of the one to eight.
- It is read for a request a program places in its own process: a single request, and each side of a pair whose profile scores by `consistent-hash` with no `prefix_tokens`.
- The number is not read for a request placed with an encode side, under `epd`.
- A standalone controller counts the workers of its own lists, by its own tests of a worker. The asking gateway's rule for the route is not asked there.
- The asking host reads the header and states the number in the request's frame. The count there also passes over a worker that fails a hard demand its role is held to.

**A single request that names a worker.**

- Where no worker at all can take it, it is placed as a request that names no worker and carries no key.
- Otherwise the worker its number reaches is the only one it can go to.
- Where the value is no number, or fewer workers can take the request, it is shed under `target-worker`.

**A pair that names a worker.**

- A pair states one number, and each side that reads it counts by it.
- The prefill's number counts the prefills that can take the request and pair with a decode that can. In a host's own process a prefill at its admission limit still counts.
- The decode's number counts the decodes that can take the request and pair with the pair's prefill, whether the request names that prefill or not.
- Outside HTTP-only workers, a pair on one known runtime goes ahead of a pair with a worker of unknown runtime. While a pair ahead can take the request, the two numbers count no pair behind.
- A side that does not read the number is placed as in any pair.
- Where no prefill is counted, the pair is placed as one that names no worker. Unlike a single request it keeps its key.
- Otherwise a number that reaches no worker on a side sheds the pair under `target-worker`, and so does a value that is no number. Where the decode's side alone reads the number, a pair can first wait for a prefill at its admission limit.
- Where no decode can take the request and the binding does not shed, a decode that its overload thresholds alone hold back counts as a partner. The decode's number is then not read.

**What a source may be.**

- A source's header need not be in `view.headers`. Its value enters a view only where the header is listed there.
- A credential header is refused: `authorization`, `proxy-authorization`, `set-cookie`, `x-api-key`, `api-key`, `x-goog-api-key`, and each header its host denies.
- A key from a header that `view.headers` does not list, `cookie` for one, leaves the host as its fingerprint alone. So no stage, plugin or record of the engine sees its text.

**A binding names its sources once.**

- With `routing_key`, a sessions rule names neither `sticky_header` nor `headers`: a name there is refused.
- Without it, the rule's own headers are the sources.
- A binding with neither gives its requests no key.

### A ring that holds a key across the overload veto

A plain ring under `single` can keep a key on its worker while that worker is over its thresholds. All of these must hold:

- The profile's only scorer is `consistent-hash` with no `prefix_tokens` and no `bounded_load`.
- Its picker is a bare `best`.
- The binding steers: its `shed_when_all_vetoed` is false.
- Every worker the request may take is a regular worker. With a worker of another role among them, the overload veto stands for the key too.
- A sessions rule, where the binding has one, delegates. Its cap is off, or the host holds the key's counts.

The standalone controller holds those counts from its leases. A host that decides over its own rows sets `session_counted`.

Three more rules hold for a ring:

- With shedding enabled, a vetoed owner passes the request to a usable pin or ring successor, if one exists.
- With a bounded prefix ring, an owner strictly above both load bounds is passed over. The mean includes the incoming request.
- Where boundaries supply a shallower key, it is tried before the profile's keyless scoring path.

### A shadow profile

A `single` binding can name a second profile, as in `shadow: next`.

- Under `pd-pair` and `epd` the field is a map of profiles by side: `prefill`, `decode` or both. A side the map leaves out is not shadowed, and the encode side takes none.
- Request handling runs the live profile, and only its decision serves the request.
- The engine over worker lists decides the shadow beside a live decision of that binding. It does so on the caller's thread, ahead of the decision's commit.
- The shadow reads the rows, the counts and the session pins the live decision read. So a `single` binding's twin of the live profile agrees where its stages keep no state of their own.
- A shadow profile's rotation takes its turn only where the shadow is asked.
- A shadow issues no lease and tells no stage.
- A served controller asks the shadow of a single request's binding behind its live decision.
- A pair's sides are shadowed where a host's own process decides the pair over its two lists and issues its lease. A pair a controller serves shadows no side.
- A request placed with an encode side shadows no side at its first decision.
- Use native stages, and a profile that no binding uses live.

**What the engine counts.** The engine counts what each shadow said, by the binding's position in its spec and one of four kinds:

| Kind | Meaning |
|---|---|
| `agreed` | The same first target, or the same outcome without a pick. |
| `other-target` | Both picked, and the first targets differ. |
| `other-outcome` | One picked, shed or queued where the other did something else. |
| `skipped` | The shadow was due and not asked. |

- The count is a position's: a binding past the sixteenth counts with the sixteenth.
- A gateway that runs the spec in its own process shows the counts on its metrics endpoint, as `smg_policy_shadow_decisions_total`.
- Each scrape reads the engine once, and no request does.
- A binding that names a shadow has its series from the first scrape: the four kinds, for each side it is counted by.
- A single request's decision whose lease fails to issue counts no shadow.
- [Gateway routing metrics](../REFERENCE.md#gateway-routing-metrics) has the series.

**When a shadow is skipped.** These rules are a `single` binding's:

- A shadow is asked over the rows its own profile would be handed.
- Where that profile would be named cached-block rows of its own, the shadow is skipped and counted. So `cache_aware` is not shadowed behind `least_load` or `random`.
- The same holds where both profiles are named such rows and their samples differ.
- A pool keeps one warm-up rule and one sent-prefix record at most.
- Behind a live profile that is named rows of its own, a shadow of another warm-up rule or another record is skipped and counted.
- The engine reads this from the spec alone: the bindings' model patterns, phase lists and order.
- A shadow profile that reads no cached blocks is asked over the whole pool, as it decides live.

A pair's side has rules of its own:

- A side can be skipped and counted too. Among the cases: a pair its profiles shed, and a keyed pair under a sessions rule.
- A pair's decode shadow is asked over the partners of the live pair's prefill.

**Budget and faults.**

- The engine over worker lists skips the shadow of a request whose stated budget is under 1 ms. The request keeps the budget for its answer.
- In a host's own process, a request that states no budget skips no shadow on that account.
- `Engine::shadow`, which a host calls itself, has no such floor.
- The engine over worker lists catches a shadow that panics. The live decision stands as made, and the engine's fault count rises by one.
- A shadowed request waits for one more decision for each shadow asked. A spec that names no shadow decides as before.

### Reserved fields

The validator checks each reserved field for shape and refuses its active use.

| Field | What is refused | Its shape |
|---|---|---|
| `match.tenant` | Any value. | A tenant id. |
| `match.priority`, `match.prompt_tokens` | Any bounds. | `min` and `max`. |
| `match.header` | Any matcher. | `name` and `values`. |
| `match.labels` | A list that is not empty. | Each label once. |
| `canary` | Any block. | A `profile`, and exactly one of `percent` (1 to 100), `tenants` or `header`. |
| `override` | `allowed`. | `locked` is the default. |
| `admission.enabled` | `true`. | |
| `sessions.issue_session_id` | `true`. | |

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

`plans.pd-pair.decode` says when the decode side is picked: `with-prefill` (the default) or `after-prefill`. See [A decode chosen after its prefill](embedding.md#a-decode-chosen-after-its-prefill).

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

A spec's life is in the reference:

- [Updating a policy spec](../REFERENCE.md#updating-a-document)
- [Decisions and refusals](../REFERENCE.md#decisions-and-refusals)
- [The presets](../REFERENCE.md#presets), each as a whole spec
