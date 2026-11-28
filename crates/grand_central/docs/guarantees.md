# Guarantees and limits

This page says what a decision books and what the service refuses. It also covers a gateway whose stream ends, and how a policy spec is updated.

[Run the service](service.md) lists the calls. [Worker rows and jobs](worker-rows.md) has a row's bounds and the probes. [Embed the engine](embedding.md) covers a host that decides in its own process.

## Decisions and refusals

### A pick and its lease

- A pick's lease books its first target at issue. The caller can dispatch an alternate and report which target it took.
- Report completion or release so the lease can end. The service gives up a side that no dispatch names in time: [When a gateway loses its stream](#when-a-gateway-loses-its-stream) has the rule.
- In a host's own process a single pick may come with no lease: [Embed the engine](embedding.md#decide-in-your-own-process) has the rule. No pass ends a lease there: it ends at the host's word or at its guard's drop.
- A repeat marked `again` ends an earlier lease still waiting for dispatch. A decode call's repeat follows a rule of its own: [When a gateway loses its stream](#when-a-gateway-loses-its-stream) has it.
- The service checks the leadership term and budget as it makes the decision. At send it checks the term again, and that the lease's dispatch span is not spent.
- A decision takes its engine and the asking gateway's vetoes from one published world. It reads the worker list the worker objects' registry publishes.
- A new generation builds fresh stages for each profile it does not keep. [What a new spec keeps](#what-a-new-spec-keeps) has the rule.
- On a controller and under `InProcess`, a lease's later words go as a rule to the stages the spec in force kept from the lease's own spec. [What a stateful stage hears](extending.md#what-a-stateful-stage-hears) says which words a stage is told.
- Under those two hosts a stage built since hears nothing of that lease, and as a rule a stage not kept hears no more.

### When no worker is left

- A single profile's empty pass can rerun filters with the overload veto lifted, then with liveness doubts lifted. As a rule it favors fewer in-flight requests on fallback.
- `shed_when_all_vetoed: true` keeps overload in force and can still lift liveness alone.
- Draining, not-ready and breaker holds stay refused by the health check.
- In a host's own process a pair's first decision takes both halves of that fallback, over its prefills and its decodes. A decode asked again behind its prefill takes both too.
- The encoders of an `epd` decision shed without it. Under the stock filters, for want of an open encoder they shed as saturated where the overload veto left none open under `shed_when_all_vetoed: true`. Else that shed names a missing endpoint, under `encoders` or a narrower name.

**What is a shed.**

- Unmatched bindings, no endpoints, an empty filter pass and a declined picker are sheds. So are incompatible pair or encode workers in a host's own process.
- The built-ins emit no queue tickets.
- Hosts decide how to map sheds to request status and retry behavior.

**What a shed's answer names.**

- The answer says the reason and a stage or constraint: as a rule the one that stopped the decision. The next four points are a pair's exceptions.
- A pair's shed as saturated names the overload veto, whichever stage ended its pass.
- A pair's shed with no endpoint, behind a pass the overload veto ended, is named by one worker of the list judged. That is the first worker with room, ready and under its thresholds, else the first ready one, else the list's first.
- The shed names the first hard filter that refuses that worker, with the overload veto and the admission limit set aside. Where no filter refuses it, or the list judged holds no worker, the shed names `partnered`.
- A pair that names its decode worker and sheds for want of it, with no endpoint or as saturated, answers with no endpoint at `target-worker`.
- A host that holds a controller's answer in a handle (`DecisionGuard::asked`) reads some of these names as `controller`: [Holding a decision](service.md#holding-a-decision) has the rule.

### Limits of the service

| Limit | Value |
|---|---|
| An incoming `Decide` frame, unknown fields included | 64 KiB |
| A request ID | 1 to 256 bytes |
| The headers of a request | 256, at positions 0 to 255 |
| The credits of a decide stream | 128 |
| The decide streams of a gateway | 256 |
| The wait for `Hello` | 5 seconds |
| A gateway's grace after a disconnect | 10 seconds |
| The gateways in the table | 4,096 |
| The vetoes of a gateway | 4,096, and 8 kinds a worker |
| The watches of a principal at a replica | 4 |
| A watch page | 128 rows and 2 MiB of row data |
| The model IDs of a watch or a snapshot | 128 |
| The snapshots that wait | 32 |
| The watched events kept for replication | 4,096 |
| The snapshot restores a leader holds at once | 4 |

**What three of the numbers bound.**

- A gateway's decide streams are counted at each replica.
- A gateway's grace runs up to 20 seconds while a session it replaced still runs. A new leadership term starts it anew.
- A watch page's first row passes at any size.

### The decide stream

- `Decide` requires a request ID of 1 to 256 bytes.
- A phase left unstated, or stated as 0, reads as single, and a hash kind as token. A number the contract does not list is refused. A frame that names a decode lease is read as a decode, whichever listed phase it states.
- A nonzero session hash must decode to a nonzero fingerprint for its model and the phase its frame states. Zero means no key.
- A frame is also refused for text hashes with no block size, over 256 headers, a header position over 255, or `decode_of: 0`.
- Encode items are refused outside the encode phase, beside a decode lease, or over 256 keys. A demand is refused for an unknown references number, or for a decoding value over 255. No demand is refused for the phase of its frame.
- A frame these checks refuse ends the stream with `INVALID_ARGUMENT`, and a frame over 64 KiB with `RESOURCE_EXHAUSTED`. A request answered `Busy` leaves its stream open.
- The client cuts tail hashes to fit. It returns an error when cutting them cannot make the message fit.
- `Welcome` states a stream's credits and a gateway's bound of decide streams. A request spends one credit, and a credit frame replenishes it. At zero credit a request is answered `Busy` and spends none.
- The caller owns any local fallback.

**The reasons of `Busy`.**

- Five are a request's own: `queue_depth`, `no_credit`, `budget_gone_on_arrival`, `not_leader` and `unspecified`.
- The five cover a full shard queue, exhausted stream credit, expired budgets or dispatch span, and a fenced replica.
- They also cover cases such as raw prompt text, an unsupported side, a decider fault or exit, spent lease IDs or a spec still awaited.
- Four more answer a decode call its lease refuses: `decode_no_lease`, `decode_not_chosen`, `decode_prefill_not_held` and `decode_stands`.

### A gateway's session

- The session expects `Hello` as its first frame, with a matching principal and protocol 1 or 2 in range.
- A gateway can keep one current session.
- A `Hello` veto count must match the list. A `Hello` is refused with over 4,096 vetoes listed, spent ones among them, with a veto of kind 0, or with over 8 live kinds on one worker.
- A later veto frame ends the session where its kind is 0, or where it would stand a ninth kind on a worker or a gateway's 4,097th veto.
- A veto on a worker with no row is dropped. One with time left ends the session all the same where the gateway holds 4,096 live vetoes: the ceiling is judged first.
- A lifecycle frame ends the session where a number names nothing: an unset or unknown side, transport class or release reason. An unset side of `Released` means every side.
- So does a target index over 255, a status over 65,535, a release that states neither a lease id nor a ticket, and a second `Hello`.
- Old terms and replaced sessions can end calls.
- The reference client applies its term floor to decision and session answers. It waits up to five seconds per operation step, with a separately streamed watch.

### Workers registered over gRPC

- A registration states 64 ranks at most. One that states no count of ranks stands for one rank.
- A new row must supply its role and its wire, HTTP or gRPC. A replace or an upsert over a standing row may leave either out, and keeps the row's.
- The service refuses a host-local wire for a row that every gateway reaches, as a new row is. It refuses a nonempty topology, and any nonempty pairing descriptor field but the bootstrap port.
- [A worker's row](worker-rows.md#a-workers-row) has the bounds of a row's facts.
- A scrape needs an interval and uses the registered runtime's parser.
- The stock controller's load monitor reads an HTTP worker by the controller's own load interval, whatever the row's scrape interval.
- Probes default to three failures, two successes, a 60-second interval and a five-second timeout.
- A refused write can answer `ALREADY_EXISTS` (an add, the default mode, over a standing row) or `NOT_FOUND` (a replace of no row, an ID that names none). It can answer `PERMISSION_DENIED` (another maker's row) or `FAILED_PRECONDITION` (a stale revision).
- A replace or an upsert that is taken over a draining row is the URL's next incarnation, whatever its facts. A drain ends with the registration it named, and only a patch or a second drain keeps it.

### Watches and snapshots

- `WatchWorkers` can require a fresh snapshot when history no longer matches. It ends a stalled watch when polling resumes after 30 seconds.
- `GetSnapshot` can cap the worker rows returned. A cap of 0 lifts the count alone: the rows' bytes stay bound, but for a first row, which passes at any size. It takes one snapshot at a time.
- It refuses a number that names no slice as `INVALID_ARGUMENT`, 0 among them. It refuses a slice this build does not serve as `UNIMPLEMENTED`: it serves the workers and the spec alone. An empty list asks for every slice, and is refused the same way.
- It refuses a model list over 128 IDs, and an ID that is empty, over 256 bytes, or holds a blank or a control character.
- It refuses identifiers in clear as `UNIMPLEMENTED`, and the spec slice too where the controller has no spec source and holds no spec. These checks run in the order of these points, and the first refusal answers.
- `Explain` refuses a decision ID or an incomplete key, and returns `NOT_FOUND` on a miss.

### Replication

- A shared lease keeps a ring of watched events for replication. Load reports are not replicated.
- History gaps require a snapshot.
- The leader can release a stalled restore after 60 seconds.

### Statuses

- A call on the wrong scope returns `PERMISSION_DENIED`.
- A scoped call without a principal returns `UNAUTHENTICATED`.
- Leader-only operations can return `FAILED_PRECONDITION` with leader details, or `UNAVAILABLE` during a term transition.
- Unserved calls return `UNIMPLEMENTED` after admission.

## When a gateway loses its stream

A gateway asks the stock controller for decisions on a stream. A pick carries a **lease** that books its worker.

- As a rule a side that no dispatch names within five seconds is given up.
- A pair's decode side that was issued with its prefill is held while that prefill is dispatched. Once the prefill has ended, the side has at least five seconds to be dispatched.

A stream can end with requests unanswered. Two rules keep one lease standing for one request.

- **The stream's end.** A decide stream that ends gives up the lease of every decision it had not sent yet. No gateway could use them.
- **A request asked again.** A gateway whose stream ended sends each unanswered request again and marks it `again` (`DecideRequest.again` in [`contracts`](../contracts)). The controller ends the lease of the earlier ask at once, where it still waits for a dispatch, and decides the request anew.

What a gateway must hold to:

- The mark is for a request whose stream has ended, and the gateway reads nothing more from that stream. On a stream that is still open the earlier answer can still arrive, and it would carry a lease the mark has ended.
- A request sent twice without the mark is two requests. Each pick among them carries a lease of its own.
- A repeat that the stream answers `Busy` before a decider takes it ends nothing. The next ask of that request carries the mark again.
- The earlier lease is found by a 64-bit fingerprint of the request's id, among that gateway's own leases.

**A decode call differs in three points.**

- As a rule the pair's lease it names stays. An answer that was not sent gives back that call's issue where it still stands. A repeat with the mark takes back the issue that stands on the decode side, where a decode call issued it and no dispatch named it.
- While the pair's lease stands, a repeat without the mark is refused `decode_stands` once a call's issue is on the side and not given back. A side given back takes the next call by a first call's rules, until its 255 issues are spent.
- No fingerprint is searched.

Two counters tell what happened, on the admin listener's `/metrics`:

| Counter | What it counts |
|---|---|
| `smg_grand_central_decide_asked_again_total` | Requests whose `again` mark was acted on, by `earlier_lease`: `ended` where a waiting lease or a decode call's issue was ended, `none` where none waited. |
| `smg_grand_central_decide_answers_dropped_total` | A decider's answers that no stream took, by `cause`: `stream_gone` and `queue_full`. |

- `stream_gone` counts an answer whose stream had ended before the answer reached its queue, or with the answer still in that queue.
- After a stream's end a repeat can read `none`: the stream gave the lease up, or the first ask was never decided. That is no error.
- A marked request counts under neither label where its mark is not read. Among the cases: a frame the checks refuse, a replica that does not lead, raw prompt text, and a `Busy` answered before a decider takes it.

## Updating a spec

### From a file

- Use a path that leads to a regular file of at most 128 KiB. The reader follows symlinks, and a `.json` extension of the path given selects JSON.
- The controller checks the file at startup.
- The watch polls after one second and reads a changed identity after two matching looks.
- A rejected update leaves the running generation in place. It is logged and counted once per settled file state for each take of the lead.
- Unchanged specs count as neither apply nor rejection.
- A fast same-size edit can escape detection if the file identity does not move. An update that stays unstable can be rejected.

### From a spec store

- `--spec-store` uses an existing directory with a lock and support for hard links. It runs with a process-local lease.
- Startup loads the newest file, or starts with no spec in an empty store. An empty store can report ready and shed requests until the first apply.
- Enforce persists before publishing, and retains the newest eight files normally.
- A competing start waits up to the shutdown grace for the lock.
- As a rule startup runs the newest file under the number in its name. It does so also where the host's store already runs that spec under a lower number. Where the newest file does not run, the spec store is not bound, and no older file is tried.

**A generation the store refuses behind its write.**

- If the file cannot be removed, the apply fails as `not persisted` and says both.
- The store tries to remove that file again before its next write.
- A process that starts before that write reads it as a generation on file, and runs it if it is the newest.

### Generation numbers

- A zero or omitted number starts at 1 and increments on change. A file can claim a higher number. Behind the largest number a `u64` holds, a change that states none is refused.
- Enforce through the store takes the next number.
- Changed content under the standing number, a stale number, or an out-of-order store number is refused.
- An apply of a spec identical to the standing one, with the standing or automatic number, puts no new generation in force.
- The digest is a `sha256:` digest of the parsed content in its canonical form, with the generation line zeroed.

### The three calls

- `ApplyPolicySpec` requires a mode. Use `yaml` or `json` for the spec's format.
- `DRY_RUN` on either source checks the candidate under that source's rules. `ENFORCE` writes only under a store.
- Both can take the current `expected_digest` as a guard.
- A verdict can include before and after needs, or one issue with a field path when one applies.
- `AUDIT` is unserved.
- `GetPolicySpec` returns this replica's standing spec, or a generation by number if held. It returns `NOT_FOUND` when none is held.
- A controller with no spec source and no spec held serves none of the three calls: each returns `UNIMPLEMENTED`. A replica built with no source serves them once it follows a leader's spec: `GetPolicySpec` at once, the other two where it leads.
- `RollbackPolicySpec` under a store reapplies a named retained generation using a new number. A rollback to the spec that stands changes nothing.
- Enforce and rollback under a file return `read_only` before the spec is validated.
- Each dry-run and write queue has room for up to 32 waiting.
- Both use the same 128 KiB cap as the file, with a separate bound on decoded YAML expansion.

### In a host's process

A host that embeds the store takes the same door with the crate's `operator` feature.

- `document::operator::DocumentManager::new(store, catalog, denied_headers)` makes it.
- `apply` takes the host's spec as bytes, YAML or JSON. It checks the spec over the host's catalog and denied headers, and runs it.
- A changed spec that leaves `generation` out runs as the next generation, by the rule of [Generation numbers](#generation-numbers). One that states a higher number runs under that number.
- `dry_run` checks a spec and applies nothing.
- A refused apply counts once in `apply_counts`, and a dry run counts nothing.
- `document::operator::MAX_DOCUMENT_BYTES` is the 128 KiB bound, and `Format::named` reads `yaml` or `json`.
- `standing` reads the generation that runs, and `apply_counts` the applies. `status` can read that generation too.
- `document` hands the running spec's bytes where the manager holds them. It takes the number 0 or the running one, and answers `NotHeld` for any other.

**Errors and rows a host can print.**

- `EnforceRefused` and `NotHeld` are errors a host can print. `NotHeld` holds what was asked, what is held, what stands and what a spec store has on file.
- Printed, `NotHeld` says what was asked, then what the spec store has on file. Where no store is bound it says what is held and what stands.
- `document::operator::snapshot::worker_rows_page` gives the worker rows an operator reads.
- As a rule an address is told without its userinfo, and so is a provider's key that has a scheme. Either reads as `<unparsable address>` where an `@` stands in its path, its query or its fragment.
- A provider's key with no scheme that holds an `@` reads as `<unparsable address>` too.

**No spec store is bound in process.**

- `writable` says so. There `roll_back` answers with the refusal `not persisted: no spec store is bound`, and so does `enforce` once its `expected_digest` guard has passed.
- The spec store itself builds on Linux and macOS.
- A host that starts from a file names the file's form with `document::operator::Format::of`: a `.json` extension selects JSON. It reads the bytes with `Format::read`, the read an apply passes too.

### Under a shared lease

- A standby follows the leader's spec if replication is configured and working.
- It validates its own file at startup, and watches without applying it while following.
- On promotion it may apply its file after it settles. A file already observed running since the last change is skipped.
- An unnumbered spec the standby never saw run can take effect on promotion even if its file is older. An explicit generation keeps a stale number from taking over.
- A new leader awaiting its first spec can return `Busy`.

**A file lease on slow storage.**

- A file lease's renewal is a synced write: the file is synced and renamed, and the directory's sync is best effort. With the default timings, a slow renewal can stop decisions, close sessions or lose the lease.
- For slower storage the lease flags can give more time.

```sh
--lease-duration-ms 15000 --lease-renew-deadline-ms 10000 --lease-renew-period-ms 2000
```

### Sends across specs

A profile's tally of sends holds the host's sends to each row that still count (`StageBuild::dispatch_credit`). A read applies its own spec's limits. It lets go the sends made `sent_report_delay_ms` before the row's report sample or longer, and those made `sent_ttl_ms` before the read or longer, where the spec sets it. When a send arrives with 8,192 or more held on a row, the tally lets one of the row's sends go, or none: as a rule the oldest. A read at a request clock of zero (`RequestFacts::now_micros`) applies no `sent_ttl_ms`.

- The tally carries over to the next spec, when both have a stage that reads it.
- A `least-load` in the new spec keeps the dispatches held under the standing one. So its estimate does not restart at a new spec.
- The profiles of a spec that set the same `sent_ttl_ms` and `sent_report_delay_ms` read one tally.
- That tally carries over to the tally under the same two settings in the next spec, whichever profiles read it there. A profile under a new name keeps the sends where its settings stand.
- Under the same two settings the two generations share the entries that stand at the carry. A send that one read releases has left the other too.

**A tally under two settings that no standing tally has.**

- It shares none. It takes a copy of the sends the standing tallies hold: for each rank of a worker, those of the tally that holds the most.
- So a changed `sent_ttl_ms` applies from the next read.
- A send that every standing tally has released is in no copy. A tally starts empty where no tally stands.

**Who makes the carry.**

- A refused spec takes nothing.
- `MemoryStore::apply` makes the carry before it publishes, and `InProcess::apply` before its new spec decides.
- A host with a store of its own calls `Engine::succeed` where it swaps engines.
- A dispatch can name a worker that a served controller's worker objects no longer hold at that rank. As a rule it opens no entry in the tally: one that races the removal can.
- Where the rank's entry still stands, such a dispatch is counted there, until the removal's word drops the entry.

### A running engine

`InProcess::apply` puts a spec in force on an engine that stays.

- A worker list's cell keeps one value for each spec that asked the list, until the host publishes the list again. A read passes the values made ahead of its spec's: as a rule the newest spec's read passes the older ones.
- A dispatch can be said behind a new spec for a decision made under an earlier one. Every tally of sends in force counts it once, whether the tally stands since that spec or was built since.
- A dispatch told while an apply is under way can miss a tally that apply builds new.
- No tally counts the dispatch of a pick that a sessions rule keeps from the profile (`rule_placed: untold`).
- The session table can still hear that dispatch.
- A pair's decode asked again behind a new spec tells its issue only to stages the spec in force kept from the pair's own spec. Among them it tells those that hear an issue. It tells none where a sessions rule keeps the side from the profile.
- A profile's name in a spec is kept for the life of the process, and so is the name of a header that a key source reads.
- `Members::standing` and `Members::facts_of` can run under the lock a change of membership is told under: `InProcess::apply` calls them so. Neither may call `forget`, `joined` or `apply`.
- `InProcess::spec` counts a reference at each call, and `DecisionEngine::with_spec` borrows the spec for the call.

### What a new spec keeps

`InProcess::apply` keeps what it can of the spec it replaces.

- A profile keeps its stages where its name and its block are the same in both specs, and so are the spec's `signals` and its `view`.
- One difference of the `view` keeps them too: another header list. No stage is built from that list.
- A stage that reads a header by a place its host wrote is kept as it stands: it reads that place of the new list.

**A profile built anew.**

- A profile is built anew too where one of its stages, built again, says otherwise of itself than the standing one.
- That is another name, other words of a request's life to hear, or anything else the engine reads of a stage when it builds a profile.
- The words to hear and `Lifecycle::keeps_loads` are compared as the two stages answer at the apply, and as each answered at its own build.
- A factory of the host's own may build such a stage: build the same stage from the same text.

**The tally of sends.**

- A tally of sends stands where a profile that reads it is kept. A profile built new beside it reads the same tally, and a send is counted once.
- A tally none of whose readers is kept is built new, and carries over as [Sends across specs](#sends-across-specs) says.
- Profiles under the same `sent_ttl_ms` and `sent_report_delay_ms` read one tally. A spec that states neither puts them all under one.

**Which stage table.**

- The keep stands between two specs validated over the stock stage table. It also stands over one `StageRegistry` of the host's own as it stood at both validations.
- A host that builds its table anew for each spec keeps nothing.
- A stage that a validation built for a kept profile is let go at the call.

### A replaced spec

A spec that `InProcess::apply` replaces lives until three things end.

- Its last decision ends, and its last `SpecHandle` is let go.
- Each thread that read it last at an entry of an engine makes another entry or ends.

**Where a stage is dropped.**

- The stages the new spec did not keep are dropped by their last holder. That can be a handle to an older spec that shared the profile, or a host's own clone of a profile (`Engine::profiles`).
- As a rule that is a thread that asks the engine. It drops them at its next entry of an engine or at its end.
- An entry is a decision that takes no `SpecHandle`, a lease's word, `DecisionEngine::with_spec`, or a read such as `InProcess::explained`. `InProcess::spec` is none: its handle is a counted reference of its own. `InProcess::apply` and a word of membership are none either.
- A stage whose drop blocks holds that call meanwhile.
- A drop that panics there unwinds out of that call.
- It aborts the process where the call is made while the thread already unwinds. A `DecisionGuard` dropped with a report owed speaks for its lease there.
- It aborts the process at a thread's end too.

### What the validator refuses

The validator refuses these, among others.

- Unknown fields of the spec, a non-null `kind` and a wrong API version.
- In a spec's text, a map key that is a sequence or a map, and a key written twice.
- An empty profile name, a header name that is not lowercase, and a binding's model that is empty, holds two stars, or holds its star in the middle.
- A binding's range with neither `min` nor `max`, or with `min` above `max`.
- A binding that names no profile or an unknown one, or states its single profile both as `profile` and under `profiles`, and a plan's side left unnamed.
- No bindings at all in a spec of more than one profile. A spec of one profile and no binding binds it to everything. Two bindings may match one request: the first then wins.
- Unsafe headers, unknown stages, unsupported active fields and stage factory errors.

The one exception is the parameters of a stage you register to take them: its factory reads them, and an unknown field passes unless the factory refuses it. [A stage with parameters](extending.md#a-stage-with-parameters) shows a type that denies them. A refusal can be about the whole spec. The validator messages in these two examples also appear in startup or apply errors.

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
