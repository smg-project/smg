#!/bin/sh
# Start one container of the test cluster as `controller` or `node`, then
# keep it up. The test waits for every node to be idle before it submits.
set -eu
mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /var/log/smg /var/log/jobs
# Job output, written on the batch node by whichever user the job runs as.
chmod 1777 /var/log/jobs
chown -R munge:munge /run/munge /var/lib/munge /var/log/munge
su -s /bin/sh munge -c munged
case "${1:-}" in
controller)
    slurmctld
    ;;
node)
    mkdir -p /sys/fs/cgroup/system.slice
    slurmd
    ;;
*)
    echo "usage: slurm-role controller|node" >&2
    exit 2
    ;;
esac
exec sleep infinity
