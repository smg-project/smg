# Policy controller

The policy controller decides which worker serves a request. It is this crate: a library that a host process embeds, and a binary, `smg-policy-controller`, that runs the same code as a service of its own. Every decision follows one **policy document**, a short YAML or JSON file that names the stages a decision runs through. The code calls the document a spec (`PolicySpec`), so "spec file" and "policy document" mean the same thing. The gateway does not route through the controller yet; that integration is in work.

**Do I have to write one?** Not for the gateway: nothing in this crate changes its routing flags. The stock controller does need a document. Its start names a file or a store, and each classic policy name is a [preset](#the-ten-presets) to copy as that file.

This page explains the document.

1. [The smallest document](#the-smallest-document)
2. [Filter, score, pick](#filter-score-pick)
3. [The ten presets](#the-ten-presets)
4. [Bindings and plans](#bindings-and-plans)
5. [Session keys and keyed policies](#session-keys-and-keyed-policies)
6. [Loading and changing a document](#loading-and-changing-a-document)
7. [A stage of your own](#a-stage-of-your-own)
8. [What is not built yet](#what-is-not-built-yet)

## The smallest document

```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [healthy]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    profile: default
```

| Line | Meaning |
|---|---|
| `apiVersion: smg.policy/v1` | The version of the format. Any other value is refused. |
| `profiles:` | The profiles, by name. A **profile** is one way to choose a worker. |
| `default:` | The name of this profile. The name is yours to choose. |
| `filter: [healthy]` | Drop the workers that are not healthy. |
| `score: [least-load]` | Rank the rest. The shortest expected wait scores highest. |
| `pick: best` | Take the worker with the highest score. |
| `bindings:` | Which requests use which profile. |
| `match: { model: "*" }` | This binding matches every model. |
| `profile: default` | A matching request uses the profile `default`. |

Four more rules:

- `filter` and `score` may be left out. `pick` is required.
- With one profile, `bindings` may be left out. The profile then serves every request. With two profiles or more, the bindings are required.
- A field the format does not know is refused by name. Nothing is ignored.
- The document has no `kind` line. A document that writes one is refused.

## Filter, score, pick

A decision starts with the **pool**: the workers that serve the request's model. The profile then runs three stages.

**1. Filter.** Each filter drops workers from the pool.

- A **hard** filter is final. Hard filters run first, in the order written. If one leaves no worker, the decision stops and the request is shed. The answer names the filter that emptied the pool.
- A **soft** filter steps aside when it would leave no worker. The pool goes on as it was before that filter.
- `not-overloaded` has a rule of its own. When every usable worker is over its thresholds, the request still goes to one of them. Set `overload.shed_when_all_vetoed: true` on the binding to shed it instead.

**2. Score.** Each scorer gives every remaining worker a score from 0 to 1. Higher is better. The profile's score for a worker is the weighted average of those scores.

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

- A scorer written as a bare name has weight 1. Here `least-load` counts for three quarters of the score and `least-requests` for one quarter.
- A weight is above zero.
- A scorer's other settings go beside its weight, in the same braces.

**3. Pick.** One picker turns the scores into the answer: an ordered list of up to four workers. The first is the pick. The others are alternates.

- A profile has exactly one picker.
- Workers whose scores are equal, within a small band, tie. `best` draws among them at random.
- A decision that ends with no worker is a **shed**. It names the stage that left it empty.

### Filters

Filters take no settings.

<!-- filters: begin -->
| Name | Kind | Keeps a worker when |
|---|---|---|
| `healthy` | hard | It is ready, its circuit breaker is closed, and it is not stalled or draining. |
| `not-overloaded` | hard | It is under the overload thresholds: 8 queued requests and a KV-cache usage of 0.8. The binding's `overload` block changes the two numbers with `waiting_requests` and `token_usage`, and a worker's own registration overrides them. |
| `has-capacity` | hard | Its prefill capacity window has a free slot. |
| `has-room` | soft | Its decode running window has a free room. |
| `not-tried` | soft | No earlier attempt of this request tried it, and the host has not excluded it. |
<!-- filters: end -->

### Scorers

Every scorer takes `weight`, 1 by default. The table lists the other settings, each with its default.

<!-- scorers: begin -->
| Name | Scores highest | Settings |
|---|---|---|
| `least-load` | The shortest expected wait: the token work on the worker over its drain rate, plus a penalty for KV-cache pressure. | `kv_pressure_weight` (0.15), `mean_prefill_tokens` (1024), `default_throughput` (2000 tokens a second) |
| `least-requests` | The fewest requests in flight. | none |
| `fewest-sessions` | The fewest session keys with a request in flight. | none |
| `session-affinity` | The worker that owns the request's session key. It scores 1 and every other worker 0. | none |
| `prefix-affinity` | The workers whose cache already holds the longest prefix of the request. | `threshold` (0.3), `source` (`index`), `spill.abs` (64), `spill.rel` (1.5), `spill.prefix_margin_tokens` (0) |
| `kv-pressure` | Nobody. It sets the point at which the profile's affinity scorers give way to load. | `spread` (1, off), `ceiling` (1, off) |
| `warmup-slice` | Workers whose cache is still young or thin, for a share of the requests that no cache holds. | `secs` (60), `share` (0.25), `blocks` (1024), `thin_ratio` (0.5), `divert_every` (8), `miss_share` (0.3) |
| `bucket` | For a prefill decision, the worker that owns the range of prompt sizes the request falls in. | `abs_threshold` (64), `rel_threshold` (1.5), `adjust_interval_s` (5) |
| `consistent-hash` | The owner of the request's key on a hash ring, then the ring's next workers. | `prefix_tokens` (unset), `bounded_load` (unset; written, its `factor` is 1.25 and its `abs_margin` 10) |
| `resident-work` | The least work in residence: the context tokens a worker holds, plus `request_tokens` for each request it holds. | `request_tokens` (required) |
<!-- scorers: end -->

### Pickers

<!-- pickers: begin -->
| Name | How it picks | Settings |
|---|---|---|
| `best` | The highest score. Ties are drawn at random. The next best follow as alternates. | `sample` (optional, 1 to 64), written `best: { sample: 16 }` |
| `round-robin` | The workers the filters kept take turns. | none |
| `random` | A random worker, then other random workers as alternates. | none |
| `power-of-two` | Two random workers. The one with the better score wins. It needs a scorer. | none |
| `count-first` | The fewest requests in flight. The score decides between equal counts. | none |
| `power-of-two-count-first` | Two random workers. The one with fewer requests in flight wins, and the score decides equal counts. It needs a scorer. | none |
| `passthrough` | The first worker the filters kept, in the order the host lists them. | none |
| `manual` | It leaves the choice to the binding's sessions rule (see [session keys](#session-keys-and-keyed-policies)). It takes no scorer. | none |
| `best-with-temperature` | Not in this build. A document that names it is refused. | `temperature` |
<!-- pickers: end -->

`sample` is for a host that names the workers holding the request's prefix and the workers warming up. `best` then decides a large pool from those workers plus a random sample of that many more. A pool is large when it holds at least 16 workers and more than twice the sample. Any other pool is read whole. The stock controller's store names those workers itself.

## The ten presets

A **preset** is a ready-made document inside the crate, one for each classic policy name. `PolicySpec::preset("least_load")` builds one.

- Each preset is one profile, bound to every model.
- All ten start with the same three filters.
- A preset's name is written with underscores, as the classic policy names are. A stage's name is written with hyphens. So the preset `round_robin` picks with `round-robin`.
- A preset also writes `generation: 1`. The documents below leave that line out; [loading](#loading-and-changing-a-document) says why.

### `random`

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

A random worker each time. It reads no load data, so use it when the workers report none.

### `round_robin`

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

The workers take turns. It reads no load data either. Use it when every worker should get the same count of requests.

### `passthrough`

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

Always the first usable worker in the host's list. Use it in front of a single worker.

### `power_of_two`

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

Two random workers are compared by expected wait, and the shorter wait wins. Use it when the workers report load and a decision should look at two workers, not at the whole pool.

### `least_load`

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

Every worker is scored by expected wait, and the shortest wait wins. Use it when the workers report load and cache reuse does not matter.

### `consistent_hashing`

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
```

A key always lands on the same worker while that worker is in the pool. Use it when requests carry a routing key.

### `prefix_hash`

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
```

Requests that share the first 256 tokens of a prompt land on the same worker, unless that worker is too busy. Use it to reuse prompt caches when the engines publish no cache events.

### `manual`

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

Sticky sessions: a key stays on the first worker it was placed on. Use it when clients send the `x-smg-routing-key` header and a session must stay on one worker.

### `cache_aware`

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

Workers whose cache already holds the request's prefix come first, and the shortest expected wait decides among them. Use it when the engines publish cache events. The stock controller does not take cache events yet: there no worker is seen as holding a prefix, and this preset places by warm-up and load. See [what is not built yet](#what-is-not-built-yet).

### `bucket`

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

Prefill requests of a similar prompt size land on the same worker. Use it for the prefill side of a prefill and decode pair. The stock controller refuses this preset today; see [what is not built yet](#what-is-not-built-yet).

## Bindings and plans

### Bindings

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

- Bindings are read from the top. The first match wins.
- `match.model` is an exact model id, `prefix*`, `*suffix` or `*`.
- `match.phase` lists the phases the binding covers: `single`, `prefill`, `decode`, `encode`. Left out, it covers every phase.
- A request that no binding matches is shed. The answer names the stage `binding`.

### Plans

A **plan** says how many workers one request needs.

| `plan` | One decision picks | The binding names |
|---|---|---|
| `single` (the default) | One worker. | `profile` |
| `pd-pair` | A prefill worker, then a decode worker that can pair with it. | `profiles.prefill`, `profiles.decode` |
| `epd` | An encode worker for each encode item, then the pair. | `profiles.encode`, `profiles.prefill`, `profiles.decode` |

The library decides all three plans. The stock controller applies a document of any plan, and its `Decide` call decides `single` requests alone today. It answers a request for a side of a pair `Busy` with no reason: the gateway is to decide that request itself. See [what is not built yet](#what-is-not-built-yet).

A prefill and decode pair:

```yaml
apiVersion: smg.policy/v1
profiles:
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
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

- The prefill profile runs first, over the prefill pool.
- The decode profile then runs over the decode workers that can pair with the chosen prefill.
- If no decode worker is left, the next best prefill is tried.
- Both workers are chosen in one decision.

Encode with a pair:

```yaml
apiVersion: smg.policy/v1
profiles:
  encode:
    filter: [healthy, not-overloaded, not-tried]
    score: [consistent-hash]
    pick: best
  prefill:
    filter: [healthy, not-overloaded, has-capacity, not-tried]
    score: [least-load]
    pick: best
  decode:
    filter: [healthy, not-overloaded, has-room, not-tried]
    score: [least-load]
    pick: best
bindings:
  - match: { model: "*" }
    plan: epd
    profiles: { encode: encode, prefill: prefill, decode: decode }
```

- The encode profile runs once for each encode item of the request, such as an image.
- A request with images only gets an encode worker that has a vision encoder.
- Here the encode profile is the hash ring, keyed by the item's content hash. The same item lands on the same encode worker.

### Pairing modes

`plans.pd-pair.pairing` says how strictly a prefill worker and a decode worker must agree before they pair. A worker's registration can carry the facts a pair is compared on: its runtime, a protocol name the operator sets, its KV transport, facts about its KV layout, and its engine version.

| Mode | It refuses a pair when |
|---|---|
| `off` | Never for a fact of the KV handoff: none is compared. |
| `lenient` (the default) | Both workers report a fact and the two differ: the runtime, the transport, a KV layout fact or the protocol name. A fact that only one of them reports refuses nothing. |
| `strict` | As `lenient`. Also when only one of the two reports a fact, when neither names its runtime or its transport, and when both report a version and the two differ. |

- A protocol name that both workers share settles the pair. Their transport and layout are not compared.
- Two workers that name different protocols never pair, except under `off`.
- One rule holds in every mode: on any wire but HTTP, two workers whose runtimes are known and differ do not pair.
- The controller works out which workers can pair when a worker registers or changes, and when a new document changes the mode.

## Session keys and keyed policies

A **session key** is a value a request carries so that related requests land on the same worker. The host reads it from a request header.

Four presets place a request by a key. They differ in what the key is, and in what happens when the key's worker is gone.

| Preset | The key | When the key's worker is gone |
|---|---|---|
| `manual` | The value of the sticky header, `x-smg-routing-key`. | The rule places the key again, on a random worker here, and remembers that worker too. When the first worker is back, the key returns to it. |
| `consistent_hashing` | The request's routing key. Under `epd`, an encode item's content hash. | The next worker on the ring takes the key. The keys of other workers do not move. |
| `prefix_hash` | The routing key. Without one, the hash of the prompt's first 256 tokens. | The next worker on the ring takes the key. |
| `bucket` | The size of the prompt, for a prefill decision. | The nearest usable worker below it in the order of sizes takes the request, else the nearest above. |

A request with no key is not an error:

- `manual` places it like a new key and does not remember it.
- `consistent_hashing` draws a worker at random.
- `prefix_hash` takes the worker with the fewest requests in flight. It does the same when the key's worker is too busy: more than 1.25 times the mean of requests in flight, and more than 10 above it.

The `manual` preset is a **sessions rule** and nothing else. A rule sits on a binding, and it can sit on any profile under the `single` plan.

| Field | Default | Meaning |
|---|---|---|
| `sticky_header` | required | The header whose value is the key. It must be listed in `view.headers`. |
| `assignment` | `delegate` | How a new key gets its worker: `random`, `min_load` (the fewest running requests), `min_group` (the fewest active keys) or `delegate` (the profile's own stages). |
| `ttl_s` | 14400 | Seconds an idle key stays remembered. |
| `sweep_s` | 120 | Seconds between two sweeps of idle keys. |
| `inflight_cap` | 2 | Requests of one key a worker carries before the next one goes elsewhere. `65535` means no cap. |
| `recovery` | `original` | Which worker leads when both of a key's workers are usable: `original`, the first, or `latest`. |

One more rule for the hash ring. Under `consistent_hashing`, a key stays on its worker while that worker is over the overload thresholds. `overload.shed_when_all_vetoed: true` on the binding moves the key to the ring's next worker instead.

## Loading and changing a document

The stock controller takes its document from one of two sources. A start names exactly one.

The binary needs the crate's `controller` feature. Without it the binary is not built.

```sh
cargo build --release -p smg-policy-controller --features controller
```

**A file.**

```sh
smg-policy-controller --controller-id controller-0 --spec-file policy.yaml
```

- The file is YAML, or JSON under a `.json` name. It holds at most 128 KiB.
- It is read at the start. A file that cannot be read, or a document that is refused, stops the start.
- It is then watched, by a look each second. A changed file is applied in place as the next generation. Nothing restarts.
- A changed file that is refused is rejected whole. The running document stays.
- With a standby, the leader's file decides. The standby follows each document its leader applies and runs it under the leader's number. Each time it takes the lead it reads its own file again, as a start does: a document other than the running one is applied as the next generation, and the running one changes nothing. The read comes within about two seconds of the take, and the replica runs the document it followed until then.
- Under a shared lease a replica checks its file at the start and applies it when it leads. Two replicas that start over two versions of the file still pair up: the second follows the first, and runs its own file's document once it leads. So a rollout of a new file takes effect as the new replicas take over.

**A store.** `--spec-store <directory>` takes the place of the file. Documents then arrive through the apply call, `ApplyPolicySpec`, in its enforce mode.

- Each document is saved in the directory before it runs, one file for each generation. The last 8 stay on file.
- `RollbackPolicySpec` runs one of them again, as the next generation.
- `GetPolicySpec` reads the running document, under either source.
- The calls are gRPC, on the controller's control listener. [`contracts/policy_controller.proto`](contracts/policy_controller.proto) defines them. The repository ships no command-line client for them yet.

**A dry run.** `ApplyPolicySpec` in its dry-run mode checks a document and applies nothing. It says whether the document would run. If not, it gives the one issue that refuses it: the field's path and the reason. It works under either source.

**The Helm chart.** `policyController.spec` holds the document as a map in the values. The chart mounts it as a file and passes `--spec-file`. See the [chart's README](../../deploy/helm/smg/README.md#policy-controller).

**The `generation` line.** A document may write `generation: <number>`.

- Leave it out, and the controller counts. The first document runs as generation 1, and each change as the next number.
- Write it, and you raise it at every change. A changed document under the number that runs is rejected.

### How a refusal reads

Every refusal names the field and says why. A start on this file is refused:

<!-- refused -->
```yaml
apiVersion: smg.policy/v1
profiles:
  default:
    filter: [ready]
    pick: random
```

```text
start refused error=the spec file policy.yaml: profile "default": unknown filter "ready"
```

A running controller whose file changes to the next one keeps its document and logs one warning:

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
policy spec rejected: the running generation stays generation=1 error=profiles.default.pick: the best-with-temperature picker is not in this build
```

On the admin listener's `/metrics`, `smg_policy_spec_generation` is the generation that runs, and `smg_policy_spec_apply_total` counts each change as applied or rejected.

## A stage of your own

```rust
fn stages() -> Result<Catalog, CatalogError> {
    let mut stages = Catalog::builtin();
    stages.scorer("pack", || Box::new(Pack))?;
    Ok(stages)
}

fn main() -> Result<ExitCode, CatalogError> {
    Ok(controller::run(stages()?))
}
```

A stage is a Rust type that implements `Filter`, `Scorer` or `Picker`. You add it to the catalog under a new name and hand the catalog to `controller::run`: the result is the stock controller with your stage in it. A document then names your stage as it names a built-in, here `score: [pack]`.

- The code above is from [`examples/policy-controller`](../../examples/policy-controller), the guide with the scorer, its tests and its document. That page also lists the data the stock controller feeds a stage today.
- [`policyspec.schema.json`](policyspec.schema.json) is the JSON schema of the document, generated from the crate's types. It lists every field with its description.
- [`contracts`](contracts) holds the wire contracts: the controller's calls and the plugin interfaces.

## What is not built yet

- A plan that waits for the prefill worker's answer before it picks the decode worker. `pd-pair` and `epd` pick every worker in one decision.
- Pair decisions from the stock controller. The library decides `pd-pair` and `epd` (`Engine::decide_pair`, `Engine::decide_encode_pair`). The stock controller's `Decide` call does not decide a side of a pair yet: it answers `Busy` with no reason.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- `shadow`, `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd`.
- The `best-with-temperature` picker, and three settings of `prefix-affinity`: `decay`, `accounting_ttl_ms` and `spill.uncached_credit`.
- The `bucket` scorer in the stock controller. The library runs it. The stock controller refuses a document that names it, because it does not yet report every request's end to a stage.
- Cache events in the stock controller. Its `PushKvEvents` call is not served, and nothing else feeds its cache index. So `prefix-affinity` sees no worker as holding a prefix there. The library places by cache for a host that feeds the index.
- Two more inputs in the stock controller. A worker's row has a circuit-breaker bit that nothing feeds yet, so `healthy` reads it as closed. A gateway's own breaker still counts: its breaker veto on its session takes the worker out of that gateway's decisions. A key's count of requests in flight is not fed, so `inflight_cap` passes no worker over.
- A plugin named in the document. The format has no field for one.
- Six calls of the contract: `Explain`, `GetSnapshot`, `PushLoad`, `PushKvEvents`, `StepDown` and `WatchRegime`. The stock controller answers each as not served.
- The audit mode of the apply call.
- The gateway's use of the controller.
