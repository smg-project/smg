# Shared AMD CI scheduling

All times are UTC. The shared machines are `amd-mi300x-1` and `amd-mi300x-2`.
Router retains daily 05:00-06:30. SMG PD retains Monday-Saturday 09:45;
on Sunday non-AMD rows remain at 09:45 and AMD rows are dispatched at 22:35.
The Sunday AMD dispatch queues behind any benchmark that still owns the lease.
A delayed schedule is classified by its triggering cron, not the current weekday.
Manual engine selection remains unchanged.

The AMD benchmark remains Sunday 07:00. Its two GPU jobs retain all eight matrix
rows and all models/iterations, with a six-hour job timeout. The previous 24-hour
limit could not fit into any daily 22.5-hour interval outside the Router window.
The longest retained job in three Sunday runs was 4h09m11s; the new limit is not
a guarantee that the whole eight-job round finishes at 22:30.

## Host enforcement is a separate deployment gate

Cron changes alone do not protect PR/manual jobs or preparation. Every eligible
job must be mapped to a trusted full-lifetime bound before a Listener claims it.
Admission must be checked before claiming work, after waiting for the shared
lease and immediately before starting. A denial keeps work queued. Checking
only the first job step or sleeping in JOB_STARTED starts the timeout too early.

The proposed complete bounds are PR/chat 90 minutes, PD 200 minutes and benchmark
390 minutes, including preparation, cleanup and margin. Corresponding latest
starts before 05:00 are 03:30, 01:40 and 22:30 the previous day. These require
protected workflow/profile binding and actual bound enforcement; they are not
made effective by this document or the cron patch. Unknown eligible work must
remain queued. Do not substitute one global benchmark cutoff for all profiles.

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
