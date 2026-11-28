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

What feeds each signal, and the views a host supplies, stand on [the embedding page](docs/embedding.md#signals-a-decision-can-read).

## Run it

The standalone service stands on [a page of its own](docs/service.md): its flags, its worker discovery, the calls it serves and the client a host calls it with. A host that decides in its own process reads [Embed the engine](docs/embedding.md).

## Extend it

A Rust stage, a WebAssembly component and a gRPC plugin stand on [a page of their own](docs/extending.md).

### Register a Rust stage

[Register a Rust stage](docs/extending.md#register-a-rust-stage) on that page shows how a host registers one. [A stage with parameters](docs/extending.md#a-stage-with-parameters) shows a scorer with typed parameters.

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

**A running engine.** `InProcess::apply` puts a document in force on an engine that stays. A worker list that stands across N documents keeps N values in its cell, and a read under the newest passes N - 1 of them, until the host publishes the list again. A dispatch said behind a new document for a decision made under an earlier one is counted once by every tally of sends in force, whether the tally stands since that document or was built since. The session table hears it. A pair's decode asked again behind a new document tells its issue to the stages kept since the pair's own document that name it. A header's name and a profile's name in a document are kept for the life of the process. `Members::standing` and `Members::facts_of` run under the lock a change of membership is told under: neither may call `forget`, `joined` or `apply`. `InProcess::spec` counts a reference at each call, and `DecisionEngine::with_spec` borrows the document for the call.

**What a new document keeps.** A profile keeps its stages where its name and its block are the same in both documents, and so are the document's `signals` and its `view`. One difference of the `view` keeps them too: another header list. No stage is built from that list. A stage that reads a header by a place its host wrote is kept as it stands: it reads that place of the new list.

A profile is built anew too where one of its stages, built again, says otherwise of itself than the standing one. That is another name, other words of a request's life to hear, or anything else the engine reads of a stage when it builds a profile. The words to hear and `Lifecycle::keeps_loads` are compared as the two stages answer at the apply, and as each answered at its own build. A factory of the host's own may build such a stage: build the same stage from the same text.

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
| `smg_router_requests_total`, `smg_router_request_errors_total`, `smg_router_request_duration_seconds`, `smg_router_tokens_total`, `smg_router_ttft_seconds`, `smg_router_dispatch_wait_seconds`, `smg_router_dispatch_wait_after_body_seconds`, `smg_router_tpot_seconds`, `smg_http_responses_total`, `smg_http_request_duration_seconds`, `smg_http_inflight_request_age_count`, `smg_http_connections_active` | Same | Kept. Request and connection metrics. |
| `smg_admission_inflight`, `smg_admission_queue_depth`, `smg_admission_queue_rejected_total`, `smg_scheduler_admit_total`, `smg_scheduler_inflight`, `smg_scheduler_queue_depth` | Same | Kept. Admission and scheduling. |
| `smg_engine_running_requests`, `smg_engine_waiting_requests`, `smg_engine_token_usage`, `smg_engine_cache_hit_rate`, `smg_worker_health`, `smg_worker_cb_state`, `smg_worker_health_checks_total`, `smg_worker_retries_total`, `smg_worker_retries_exhausted_total` | Same | Kept. Worker reports and resilience. |
| `smg_kv_event_lag_seconds`, `smg_kv_event_gaps_total`, `smg_kv_event_resyncs_total`, `smg_kv_event_missed_batches_total`, `smg_kv_event_stream_errors_total`, `smg_kv_event_degraded_ranks`, `smg_kv_index_entries` | Same | Kept. Gateway cache event and index metrics. |
| `smg_symphony_outputs_total`, `smg_symphony_malformed_total`, `smg_symphony_fallback_total`, `smg_symphony_refused_total`, `smg_symphony_calls_suppressed_total`, `smg_tokenizer_cache_lookups_total` | Same | Kept. Output parsing and tokenization. |
| `smg_tokio_worker_busy_ratio`, `smg_tokio_event_loop_stalls_total`, `smg_tokio_event_loop_delay_seconds`, `smg_tokio_alive_tasks`, `smg_tokio_global_queue_depth`, `smg_log_dropped_lines` | Same | Kept. Runtime and logging. |
| `smg_allocator_resident_bytes`, `smg_allocator_allocated_bytes` | Same | Kept. Available when the gateway runs with jemalloc statistics. |
| `smg_pd_prefill_duration_seconds`, `smg_pd_kv_transfer_duration_seconds`, `smg_pd_decode_ttft_seconds`, `smg_pd_ttft_seconds`, `smg_pd_rendezvous_failures_total`, `smg_pd_bootstrap_failures_total`, `smg_pd_kv_transfer_failures_total`, `smg_pd_kv_connector_mode_total` | Same | Kept. Prefill and decode dispatch. |

## Presets

`PolicySpec::preset("least_load")` builds a complete document with generation 1. Each preset has one profile, the same three filters and a binding for every model. The copies below leave the number to the source. `PolicySpec::power_of_two` also accepts `ExpectedWait`, `Requests`, `LeastRequests` or `ResidentTokens { request_tokens }`. The resident variant ranks the full set by resident work. `pair_presets` and `epd_presets` compose side profiles from preset names and add `has-capacity` to the prefill side and `has-room` to the decode side. The standalone controller places a pair at its prefill, and reads both bits open. [Preset constructors](src/document/spec/presets.rs)

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

A worker's row, its bounds, the door that writes it and the worker jobs of a host with a local store stand on [a page of their own](docs/worker-rows.md).

## What is not built yet

- A client that marks a request as asked again. The controller serves the mark. The crate's own client does not set it yet.
- A client that asks a served controller for a pair's later decode or for its decode call. The controller serves both. The crate's own client states neither yet.
- A choice by the media a worker can process, outside `epd`. No built-in filter reads it.
- A binding that matches on a tenant, a priority, a header, a prompt size or labels. The fields are in the schema. A document that writes one is refused as not in this build.
- A pair's shadows at an encode request's first decision, beside a side its sessions rule or its picker's assignment places and through a served controller, and plugin stages in a shadow profile.
- `canary`, `override: allowed` and `admission` on a binding, and `issue_session_id` in a sessions rule.
- A sessions rule under `pd-pair` or `epd` that states `rule_placed: untold`, through the stock controller. Its file of leases keeps no word of a pair's side. In a host's own process a rule runs on both sides of a pair under either plan, or under `pd-pair` on the one side `sides` names.
- A list of `sides` in a sessions rule under `epd`. The field is the `pd-pair` plan's, and under `epd` a rule covers the pair's two sides.
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
