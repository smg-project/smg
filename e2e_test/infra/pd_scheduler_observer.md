# Scheduler observer prototype

This test-only prototype is not enabled by a fixture, CLI argument, workflow or
environment change. It produces partial observations, never an acceptance result.
The CPU tests use explicit contract stubs, not a vLLM installation or running engine.

The source contract is the installed vLLM 0.31.0 ROCm image
`sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f`.
Its scheduler interface is not stable. The two entry classes preserve the actual
`Scheduler` and `AsyncScheduler` parents. `observer_class_for` mirrors the native
default choice only after `VllmConfig` has resolved `async_scheduling` to a bool;
it does not resolve auto configuration or support a different custom scheduler.
The constructor marks a class/config mismatch unknown without changing that config.

Each hook delegates once, returns the native result and propagates native errors.
`finish_requests_return` records only the returned requests' actual statuses.
An input abort signal, empty return, natural length completion or RUNNING result
is not converted into FINISHED_ABORTED. The original internal request ID is kept
verbatim. `abort_immediately` distinguishes synthetic cleanup requests; observing
this flag does not make them cancellation evidence. An actual ABORTED status alone
does not establish client-disconnect causality or exclude teardown.

The trace records admission returns, first scheduling, request token counts,
native free-method returns, registration membership and deferred block/fence
snapshots. Computed tokens can include in-flight work. A request may leave
`self.requests` while its block references remain deferred. Queue removal is
not a transfer ACK, proof that a block is globally unreferenced, physical HBM
release, or destruction of remote peer registrations. `transfer_ack` and
`peer_memory_released` remain null. Block IDs can be reused and shared; these raw
events do not establish a complete block ownership history.

Only primitive snapshots are retained. The producer does not consume ID
iterators, native event queues, connector state or output streams. Observation
errors latch `trace.unknown` and stop emission while native calls continue.
Every written event says `observation: partial`. A valid prefix of a truncated
file is not a complete trace; this prototype has no final completeness handshake
or external acceptance reader.

For a future explicitly configured worker, the JSONL sink needs an existing
`SMG_PD_TRACE_DIR` and explicit `SMG_PD_TRACE_RUN`, `ATTEMPT`, `WORKER`, `ROLE`
variables (each prefixed with `SMG_PD_TRACE_`; role is prefill or decode).
Files are unique per scheduler process/instance. No default directory is created.
Synchronous trace I/O has not been measured against real scheduling races.

Before fixture integration: verify the exact source hashes and child import path,
the original resolved scheduler class/config, per-worker trace completeness and
failure visibility, and the real client-to-P/D internal ID/retry relationship.
GPU validation still needs actual abort versus ignored-abort/natural-completion
negative controls before teardown, phase evidence for prefill cancellation and
connector-specific ACK/release evidence. No HTTP skip is removed by this prototype.
