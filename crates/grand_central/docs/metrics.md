# Explain and metrics

Grand Central can keep a sample of its decisions to be read later, and its standalone service exports metrics. This page covers both. It also lists the plugins' stats and the gateway's routing metrics.

The [policy spec's page](policy-spec.md) has the `explain` field. The [service's page](service.md#the-calls) has the `Explain` call.

## Explain

Set `explain: { sample: N }` to keep a sample of picks and sheds. Zero turns it off. One keeps every pick or shed that is counted.

**On the service.**

- Each decider counts the picks and sheds it answers and keeps every Nth.
- Each decider allocates its 1024-slot ring on its first sampled write and overwrites its oldest records.
- With sampling on, a decision takes its decider's ring lock only when it keeps a record. That write can wait for an `Explain` read of the ring.
- At R counted decisions a second on that decider, a full ring covers about `1024 × N / R` seconds.
- A new generation restarts counting. Older samples can still be read under the same term.
- A term change restarts counting too, and makes old samples unavailable.

**In a host's own process.**

- Each calling thread counts its own picks and sheds and keeps every Nth. Its count starts over each time the thread counts under another engine or a later spec.
- The engine counts a request only where an ID names it: the request's own, one its host states for it, or the value of a header its host names.
- A sample is of a single request's decision or of a pair's first decision. An encode request decided with its items is not sampled, and neither is a decode decided again.
- The spec in force holds one ring of 1,024 records, which every thread shares. A later spec starts with an empty ring, and `InProcess::explained` reads the ring of the one in force.

### The lookup

- The remote lookup uses `gateway_id` and `request_id`.
- For `gateway_id`, use the gateway's subject on the gateway listener. `Explain` hashes it the same way as the decide stream.
- In a host's own process `InProcess::explained` finds a record by its request's ID alone.

### What a record names

- A record's names are the plan, the profile, and that profile's picker, filters and scorers in their declared order. `DecisionRecord::names` reads them from the engine it is given.
- A shed made ahead of every profile names the plan alone. A subset hint of zero on a pool with members is one example.
- In process, `Trace::made_under()` gives the profile's position, or `None` for such a shed.
- On the wire, read `trace.no_profile` first. When it is true, the decision was made ahead of every profile and `trace.profile` is 0.
- The name fields are empty for unmatched decisions and records from older generations.

### What a record holds

- The `Explain` call's record echoes the lookup key and gives the decision and its trace without a lease.
- It keeps up to three top scores.
- The control call keeps the record's original binding position and trace. `binding` is 0 when no binding matched or when the first binding matched.
- The record leaves the decision ID, full candidate list, view digest, plugin projections and lifecycle outcome unset.
- It omits request facts, headers and session keys.

## The service's metrics

The admin `/metrics` endpoint exports the following series. A spec source adds the two spec series.

| Metric | What it measures |
|---|---|
| `smg_grand_central_state` | One gauge per `state` (`starting`, `leader`, `syncing`, `follower`, `draining`). The current state is 1. |
| `smg_grand_central_start_time_seconds` | Startup time since the Unix epoch. |
| `smg_grand_central_replication_synced` | 1 for a synced standby. |
| `smg_grand_central_replication_lag_seconds` | Seconds since the standby last applied a frame and stood at the head its leader stated, as a rule. 0 when leading or before such a frame. |
| `smg_grand_central_replication_restores_asked_total` | Standby's restore requests. |
| `smg_grand_central_load_polls_total` | The load monitor's poll decisions for workers on a connection, by `mode` (`poll`, `fallback`, `suppressed`). |
| `smg_grand_central_load_poll_failures_total` | Failures of a poll over a connection, by `reason`. |
| `smg_grand_central_load_scrapes_total` | Load reads over HTTP that brought a report: the native load route or the `/metrics` gauges. The ranks of one engine share one read. |
| `smg_grand_central_load_scrape_failures_total` | Load reads over HTTP that brought none, by `reason`. The poll and scrape label values are `transport`, `answer`, `deadline`, `no_load`, `unsupported`, and `client`. |
| `smg_grand_central_load_rows_total` | Rank load rows of polled answers that the load monitor published. |
| `smg_grand_central_load_values_dropped_total` | Named values that the list bounds removed from the load monitor's polled answers. |
| `smg_policy_spec_generation` | Standing spec generation, as a gauge's float: exact up to 2^53. |
| `smg_policy_spec_apply_total` | Applies by `result` (`applied` or `rejected`). |
| `smg_grand_central_decide_busy_total` | Busy answers by wire `reason`. |
| `smg_grand_central_decide_no_decision_total` | Unspecified Busy by `cause` (`fault`, `pair_side`, `prompt_text`, `decider_gone`, `lease_ids_spent`, or `document_awaited`). |
| `smg_grand_central_decide_hashes_cut_total` | Requests that reached a decision with a hash list longer than the generation reads. |
| `smg_grand_central_decide_answers_dropped_total` | Dropped answers by `cause` (`stream_gone` or `queue_full`). |
| `smg_grand_central_decide_asked_again_total` | Requests whose `again` mark the decider acted on, by `earlier_lease` (`none` or `ended`). |
| `smg_grand_central_decide_candidates_undialed_total` | Candidates removed by the asking gateway's wire constraints. |
| `smg_grand_central_decide_budget_outlasted_total` | Decisions cut at the final decision check because the budget or dispatch span was too short. |
| `smg_grand_central_members_outside_list` | Members not classified by a configured list, by `list` (`engines` or `zones`). |
| `smg_grand_central_overload_flips_total` | Overload verdicts of a worker that changed, either way. The monitor takes every verdict off and counts each one when it stops or rebuilds its groups. |
| `smg_grand_central_lifecycle_faults_total` | Panics contained while applying dispatch, completion, release, dispatch expiry or a decode call's issue. |
| `smg_policy_faults_total` | Panics the engine held and counted, by `kind` (`decision`, `event` or `upkeep`). |

**`smg_policy_faults_total`.** This is the family an embedded gateway publishes.

- It is read from the store at each scrape and counted over every engine the store built. So it never starts again at a new policy spec.
- `decision` is a decision that panicked, a shadow's beside a live decision among them. A live decision's panic also counts as `fault` in `smg_grand_central_decide_no_decision_total`.
- `event` is a stage that panicked at a change of membership.
- A stage's panic at a lease's word is held by the controller itself. It counts as `fault` where it unwinds the decision that told the word. That is the issue of a single request's or a pair's lease, or the release of a lease the decision gives up.
- At every other word it counts in `smg_grand_central_lifecycle_faults_total`, a decode call's issue among them.
- `upkeep` is a run of the engine's upkeep that panicked at the controller's sweep.

## Plugin stats

Plugin stats are available through the host APIs. Declared counters are separate from these host stats.

| Kind | What its stats count |
|---|---|
| Both kinds | Answers, declines, exhausted decision budgets, breaker bypass and openings, stalls and observed events. |
| WebAssembly, also | Traps, fuel use and exhaustion, invalid replies, skips of a request the packed form cannot carry, instantiations, lost events and unread timing. |
| gRPC, also | Plugin Busy, invalid replies, overlarge pools, cold or refused frames, misses and their attribution, connections, stream failures, late answers and dropped observe events. |

[WebAssembly stats](../src/plugins/wasm/plugin.rs) · [gRPC stats](../src/plugins/grpc/plugin.rs)

## Gateway routing metrics

The table below sets the classic gateway's metrics beside those of a gateway that embeds the engine.

- `Same` carries the classic names over.
- The last column also records families whose embedded writers are pending.
- Four names in it are planned, and no code writes them yet: `smg_policy_lease_expired_total{kind}`, `smg_policy_leases{kind,state}`, `smg_policy_fallback_total{reason}` and `smg_policy_deadline_misses_total`.
- The standalone controller already publishes `smg_policy_spec_generation` and `smg_policy_spec_apply_total{result}` when it has a spec source.
- The shipped dashboard uses the classic names.
- The standalone controller keeps the `smg_grand_central_` prefix for its process metrics, including role, replication and load polling.

| Classic gateway | Embedded gateway | Class and plan |
| --- | --- | --- |
| `smg_worker_selection_total{policy}` | Same | Kept. Under the engine the `policy` label can hold the profile's name. |
| `smg_worker_overload_shed_total`, `smg_worker_overload_fallback_total`, `smg_worker_liveness_fallback_total` | Same | Kept. The embedded writers will use the decision's outcome and lifted veto. |
| `smg_pd_admission_waits_total`, `smg_pd_admission_sheds_total`, `smg_pd_admission_window`, `smg_pd_prefill_admission_inflight`, `smg_pd_prefill_admission_queued`, `smg_pd_prefill_admission_rejections_total`, `smg_pd_prefill_admission_wait_seconds` | Same | Kept. The gateway owns admission. |
| `smg_routing_key_source_total{source}` | — | Written by the classic sticky override alone, with `header` or `rid` as the source. |
| `smg_cache_aware_policy_branch_total{branch}`, `smg_manual_policy_branch_total{branch}`, `smg_consistent_hashing_policy_branch_total{branch}`, `smg_prefix_hash_policy_branch_total{branch}` | — | — |
| `smg_policy_inflight_reconciled_total{policy}` | — | Planned rename to `smg_policy_lease_expired_total{kind}`. No code writes the new name yet. |
| `smg_cache_aware_match_ratio` | Same | Kept. [The match ratio](#the-match-ratio) has the rules. |
| `smg_kv_index_lookup_seconds` | — | Written by the classic `cache_aware` policy alone, at its index lookup. |
| `smg_cache_placement_entries` | Same | Kept under a hash index: each sweep of the placement maps the gateway lends the engine writes it, as the classic policy's placement sweep does. |
| `smg_cache_tree_tokens`, `smg_cache_tree_chars`, `smg_cache_tree_tenants` | Same | Kept for the trees the gateway lends the engine. See [The tree sizes](#the-tree-sizes). |
| `smg_manual_policy_cache_entries` | — | Ends with the classic manual cache. |
| `smg_cache_aware_prefix_margin_held_total` | — | Open. An embedded successor requires the scorer to report whether the margin held without an extra read. Otherwise the counter ends with classic routing. |
| — | `smg_policy_shadow_decisions_total{binding,side,kind}` | New with the engine: shadow decisions beside live ones. See [Shadow decisions](#shadow-decisions). |
| — | `smg_policy_faults_total{kind}` | New with the engine: panics the engine held and counted. See [Faults of the engine](#faults-of-the-engine). |
| — | — | Planned additions. Embedded writers are pending for `smg_policy_leases{kind,state}`, `smg_policy_spec_generation`, and `smg_policy_spec_apply_total{result}`. |
| — | — | Planned renames of the standalone controller's `smg_grand_central_decide_busy_total` to `smg_policy_fallback_total{reason}` for fallback and of its `smg_grand_central_decide_budget_outlasted_total` to `smg_policy_deadline_misses_total` for decision deadlines. No code writes the new names yet. |
| `smg_router_requests_total`, `smg_router_request_errors_total`, `smg_router_request_duration_seconds`, `smg_router_tokens_total`, `smg_router_ttft_seconds`, `smg_router_dispatch_wait_seconds`, `smg_router_dispatch_wait_after_body_seconds`, `smg_router_tpot_seconds`, `smg_http_responses_total`, `smg_http_request_duration_seconds`, `smg_http_inflight_request_age_count`, `smg_http_connections_active` | Same | Kept. Request and HTTP metrics. |
| `smg_admission_inflight`, `smg_admission_queue_depth`, `smg_admission_queue_rejected_total`, `smg_scheduler_admit_total`, `smg_scheduler_inflight`, `smg_scheduler_queue_depth` | Same | Kept. Admission and scheduling. |
| `smg_engine_running_requests`, `smg_engine_waiting_requests`, `smg_engine_token_usage`, `smg_engine_cache_hit_rate`, `smg_worker_health`, `smg_worker_cb_state`, `smg_worker_health_checks_total`, `smg_worker_retries_total`, `smg_worker_retries_exhausted_total` | Same | Kept. Worker reports and resilience. |
| `smg_kv_event_lag_seconds`, `smg_kv_event_gaps_total`, `smg_kv_event_resyncs_total`, `smg_kv_event_missed_batches_total`, `smg_kv_event_stream_errors_total`, `smg_kv_event_degraded_ranks`, `smg_kv_index_entries` | Same | Kept. Gateway cache event and index metrics. |
| `smg_symphony_outputs_total`, `smg_symphony_malformed_total`, `smg_symphony_fallback_total`, `smg_symphony_refused_total`, `smg_symphony_calls_suppressed_total`, `smg_tokenizer_cache_lookups_total` | Same | Kept. Output parsing and tokenization. |
| `smg_tokio_worker_busy_ratio`, `smg_tokio_event_loop_stalls_total`, `smg_tokio_event_loop_delay_seconds`, `smg_tokio_alive_tasks`, `smg_tokio_global_queue_depth`, `smg_log_dropped_lines` | Same | Kept. Runtime and logging. |
| `smg_allocator_resident_bytes`, `smg_allocator_allocated_bytes` | Same | Kept. Available when the gateway runs with jemalloc statistics. |
| `smg_pd_prefill_duration_seconds`, `smg_pd_kv_transfer_duration_seconds`, `smg_pd_decode_ttft_seconds`, `smg_pd_ttft_seconds`, `smg_pd_rendezvous_failures_total`, `smg_pd_bootstrap_failures_total`, `smg_pd_kv_transfer_failures_total`, `smg_pd_kv_connector_mode_total` | Same | Kept. Prefill and decode dispatch. |

### The match ratio

Under the engine `smg_cache_aware_match_ratio` is observed from the overlap a pick states.

- A host observes the match ratio from the overlap a single pick of `InProcess` states of the chosen worker (`Pick::overlap`).
- The gateway observes it from the placement the one call answers (`Placement::overlap`): once at most for each placement, a single request's or a pair's. A placement that states none observes nothing.
- It does so where the placement's worker is the one its pick's first target names.
- A pair's pick states what its prefill holds and nothing of its decode. So a pair's placement observes once at most, where a classic pair can observe each `cache_aware` side.
- An encode request decided with its items states none yet, so it observes nothing.
- Under a hash index a single request decided inside the placement map the gateway lends states no share. So the gateway observes nothing for it, as a classic hash decision observes none.
- A pair under a hash index is lent no map. Where its prefill's profile reads cached blocks, its pick states its share by what the prefill's column holds. That is the engine's record of sent prefixes or the model's index of cache events. A classic hash pair observes none.
- Where a lent tree's match names the chosen worker to the decision, the pick states the tree's own two numbers, so the gateway divides what a classic tree decision divides.

These still differ there, among others:

- The count is the chosen worker's, a miss where it is no holder of the match. Classic observes the match whoever it chose.
- Under `best` with a sample, where each filter and scorer returns true from `row_wise`, a decision is named that many of the match's holders at most, drawn among them. A pick of a holder left out states a miss.
- Past 65,535 tokens or characters both counts are halved together, as often as it takes the request's to fit 16 bits. A count past 4,294,967,295 is cut to that number first.

### The tree sizes

`smg_cache_tree_tokens`, `smg_cache_tree_chars` and `smg_cache_tree_tenants` are kept for the trees the gateway lends the engine.

- Their size pass writes a token tree's tokens, a string tree's size and each tree's workers.
- Under an `index` entry of the longest cadence no pass runs, and none of the three is written for its trees.

### Shadow decisions

`smg_policy_shadow_decisions_total{binding,side,kind}` counts shadow decisions beside live ones. It is read from the engine at each scrape.

- `binding` is the binding's position in the policy spec from 0. `15` stands for every binding from the sixteenth on.
- `side` is `single`, `prefill`, `decode` or `decode-again`.
- `kind` is `agreed`, `other-target`, `other-outcome` or `skipped`.
- A shadow that panicked is in the engine's fault count and not here.
- A classic gateway publishes no such family.

### Faults of the engine

`smg_policy_faults_total{kind}` counts panics the engine held and counted. It is read from the engine at each scrape.

- `kind` is `decision`, `event` or `upkeep`. Each kind has its series from the first scrape.
- `decision` is a decision that panicked, a shadow's beside a live decision among them (`DecisionEngine::faults`).
- `event` is a stage that panicked at a lease event or a change of membership (`InProcess::unheard`).
- `upkeep` is a step of the engine's upkeep that panicked: its sweep, the registry's follow, a load report or a fill of worker lists (`InProcess::upkeep_faults`).
- A classic gateway publishes no such family.
- The standalone controller publishes it on its own metrics page, as [its table of series](#the-services-metrics) says.
