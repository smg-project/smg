# A Slurm 23.11 node, the release Ubuntu 24.04 ships, with the workspace smg
# and mock-worker binaries. One image runs every container of the test
# cluster: `controller` (munge, slurmctld, and the gateway the test starts)
# or `node` (munge, slurmd). The build context is a staging directory holding
# this directory's files and the two binaries, built on the runner, which the
# workflow pins to ubuntu-24.04 so the binaries' glibc matches this image's.
#
# The munge key is baked in so every container shares it. It authenticates
# nothing outside the test cluster.
FROM docker.io/library/ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        slurm-wlm munge openssl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* \
    && dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key && chmod 0400 /etc/munge/munge.key \
    && useradd -m alice
COPY slurm.conf cgroup.conf /etc/slurm/
COPY entrypoint.sh /usr/local/bin/slurm-role
COPY smg mock-worker /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/slurm-role"]
