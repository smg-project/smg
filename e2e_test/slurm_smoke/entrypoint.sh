#!/bin/sh
# Start munge, the controller and the node daemon, then keep the container up.
# The test waits for the node to be idle before it submits anything.
set -eu
mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /var/log/smg
if [ ! -s /etc/munge/munge.key ]; then
    dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null
fi
chown -R munge:munge /etc/munge /run/munge /var/lib/munge /var/log/munge
chmod 0400 /etc/munge/munge.key
su -s /bin/sh munge -c munged
slurmctld
slurmd
exec sleep infinity
