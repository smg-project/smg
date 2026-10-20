# Slurm recordings

Real output of the commands the Slurm provider runs, recorded from a one-node
Slurm in a throwaway container per release: Ubuntu 22.04 (21.08.5), Ubuntu
24.04 (23.11.4) and Debian trixie (24.11.5), each with the distribution's
`slurm-wlm` and `munge` packages. The JSON was re-serialized compactly; its
content is as Slurm printed it, except that `squeue-21.08.json` keeps only the
first job.

The cluster is `smgtest`, and the jobs are run as `root` (the user `squeue` ran
as) unless noted:

| Job (23.11) | Job (24.11) | Name | Notes |
| --- | --- | --- | --- |
| 1 | 1 | `smg-worker` | |
| 2 | 2 | `smg-worker` | account `acct-b`, partition `gpu` |
| 3, 7 | 3, 8 | `smg-prefill` | the two tasks of array job 3 |
| 4 | 4 | `smg-decode` | requeued once: `restart_cnt` 1, a later `start_time` |
| 5 | 5 | `unrelated` | a name no mapping lists |
| 8 | 6 | `smg-worker` | submitted by `alice` |
| 6 | 7 | `smg-worker` | pending, held with `--begin` |

Every running job ran on node `slurmnode`. The node records come from
`scontrol --json show nodes slurmnode,gpu01,gpu02,gpu03`:

| Node | `NodeAddr` | `NodeHostname` |
| --- | --- | --- |
| `slurmnode` | `127.0.0.1` | `slurmnode` |
| `gpu01` | `10.0.0.11` | `gpu01.cluster` |
| `gpu02` | unset (reported as `gpu02.cluster`) | `gpu02.cluster` |
| `gpu03` | unset (reported as `gpu03`) | unset (reported as `gpu03`) |

What the recordings showed, and the provider relies on:

- `squeue --json` ignores filters on 21.08 and, except `--user`, on 23.11;
  24.11 applies them all. The provider filters every answer again.
- `meta.client.user` names the user `squeue` ran as (23.11 and later).
- Each array task has its own `job_id`. A requeue keeps the `job_id` and moves
  `restart_cnt` and `start_time`.
- 21.08 has no `scontrol --json`, and leaves `user_name` empty.
- `scontrol --json show nodes` answers an unknown node with exit 0 and an empty
  list, and `scontrol show hostnames` answers a malformed hostlist with exit 0
  and nothing on stdout.
