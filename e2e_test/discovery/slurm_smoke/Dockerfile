# A one-node Slurm (munge, slurmctld and one slurmd) beside the workspace smg
# and mock-worker binaries, for the real-Slurm smoke. The build context is a
# staging directory holding this directory's files and the two binaries, built
# on the runner (ubuntu-latest), whose glibc is older than trixie's.
#
# Debian trixie ships Slurm 24.11, whose slurmd runs without systemd or a
# privileged container once its cgroup plugin is off (cgroup.conf). The base
# image is pinned; the test checks the Slurm release it runs.
FROM docker.io/library/debian:trixie-slim@sha256:a29215f6a35e51e22adffa17f89e9d2ef06214e64a2bad10d765c46aea49f11f
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        slurm-wlm munge openssl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
COPY slurm.conf cgroup.conf /etc/slurm/
COPY entrypoint.sh /usr/local/bin/slurm-node
COPY smg mock-worker /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/slurm-node"]
