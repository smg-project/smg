# Shared AMD CI scheduling

All times are UTC. The shared machines are `amd-mi300x-1` and `amd-mi300x-2`.
Router retains daily 05:00-06:30. SMG PD retains Monday-Saturday 09:45;
on Sunday non-AMD rows remain at 09:45 and AMD rows are dispatched at 22:35.
The Sunday AMD dispatch retains its agreed 22:35 time even though the benchmark
moves to Monday. It queues behind any work that still owns the lease; changing
the PD schedule is a separate decision.
A delayed schedule is classified by its triggering cron, not the current weekday.
Manual engine selection remains unchanged.

The AMD benchmark runs Monday 07:00 in four ordered waves: single-worker vLLM,
single-worker SGLang, multi-worker vLLM, then multi-worker SGLang. Each wave has
two model rows, retaining the original eight GPU jobs, models, iterations and
HTTP/gRPC tests with a six-hour job timeout. A manual runtime selection excludes
unselected engine waves before runner allocation; per-model filters still run
inside each selected job.

Monday 07:17 AMD BFCL (when `SMG_RUN_AMD_BFCL` and automatic AMD selection are
enabled), Monday 09:45 AMD PD and ordinary AMD jobs share the same machines and
lease with the benchmark. BFCL currently declares five AMD model rows with
360-minute job caps; it can add substantial queue contention. Deferred Sunday
AMD work can also reach Monday. These waves do
not reserve hosts or guarantee queue ordering, a new 24-hour queue allowance,
or completion on Monday. Work can continue in later legal windows after actual
Router cleanup and lease admission. The longest retained job in three Sunday
runs was 4h09m11s; that observation does not bound future runtimes or guarantee
that the whole eight-job round finishes at 22:30.

## Host enforcement is a separate deployment gate

Cron changes alone do not protect PR/manual jobs or preparation. Every eligible
job must be mapped to a trusted full-lifetime bound before a Listener claims it.
Admission must be checked before claiming work, after waiting for the shared
lease and immediately before starting. A denial keeps work queued. Checking
only the first job step or sleeping in JOB_STARTED starts the timeout too early.

The proposed classified bounds are PR/chat 90 minutes, PD 200 minutes and
benchmark 390 minutes, including preparation, cleanup and margin. Their latest
starts before 05:00 are 03:30, 01:40 and 22:30 the previous day. These require
protected workflow/profile binding and actual bound enforcement; they are not
made effective by this document or the cron patch. Unknown eligible work must
remain queued.

A conservative mixed-label entry instead reserves the largest eligible bound
for every claim: a 360-minute job and 390-minute complete lifetime, including
enabled BFCL work. Its latest start is 22:30. The proposed 90/200-minute profiles
cannot be substituted for that mixed entry without classifying its consumers.
Monday benchmark waves work with this entry and the existing runner labels;
separate runner-label management is not a prerequisite for this schedule.

Every enabled AMD PR/manual job follows the same Router reservation. The hosted
configuration check prints effective boolean gate values and runs the holder
fault controls using isolated temporary locks on Ubuntu; it uses no GPU.
Do not enable AMD PR jobs solely to test scheduling. Repository and organization
configuration may both affect the resolved value.

## Existing lease and logging

Retain `/run/lock/amd-ci-gpu.lock` on each host, including its inode. Acquire it
before Docker cleanup, process termination, GPU checks/use, and retain custody
through the full Worker lifetime and cleanup. Router's local/remote holder is
unchanged; node2 does not need a new Buildkite agent.

`scripts/ci/amd_ci_hold_host_lease.sh` is the existing SMG PID/start-time holder
with best-effort, bounded syslog events: `waiting`, `acquired`, `wait-timeout`,
`released` under tag `smg-amd-ci-lease`. Logging children close the holder's lock
FD first. A release event describes this holder; it does not assert that no next
owner has acquired the lock or that every GPU allocation is healthy.

Install only after checking the current holder hash and lock identity. Back up
the existing file and atomically rename a root-owned 0755 replacement; never
truncate a running script or recreate/delete the shared lock. Do not restart an
active Worker or modify Router. Read back both hashes and journal events.

This PR does not install a pre-Listener admission implementation, configure a
protected runner cohort, waive an existing runtime HOLD, or change repository
variables. Full activation requires that separate owner/admin integration and
controlled runtime evidence. The logging update alone is not full reservation
enforcement.
