FROM fra.ocir.io/idqj093njucb/action-runner@sha256:fbf5e9cbeaf6ea03501654934805a712bd1011ea0a66694828d69b8c35f5d595
USER root
ARG RUNNER_VERSION=2.337.0
ARG RUNNER_SHA256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
# Preserve the existing container startup scripts, hooks, and system tooling.
# Replace the complete runner runtime and bundled Node runtimes together.
RUN set -eux; \
    curl --fail --location --retry 3 \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
      --output /tmp/runner-update.tar.gz; \
    echo "${RUNNER_SHA256}  /tmp/runner-update.tar.gz" | sha256sum --check -; \
    mkdir /tmp/runner-update; \
    tar -xzf /tmp/runner-update.tar.gz -C /tmp/runner-update; \
    test -x /tmp/runner-update/bin/Runner.Listener; \
    rm -rf /home/runner/bin /home/runner/externals; \
    mv /tmp/runner-update/bin /home/runner/bin; \
    mv /tmp/runner-update/externals /home/runner/externals; \
    chown -R runner:docker /home/runner/bin /home/runner/externals; \
    rm -rf /tmp/runner-update /tmp/runner-update.tar.gz
LABEL io.smg.actions-runner.version="${RUNNER_VERSION}"
USER runner
WORKDIR /home/runner
