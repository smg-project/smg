"""What the servicers read a scheduler's health from: its pushes.

The two health answers the SGLang and TokenSpeed servicers give rest on one
signal, the time of the scheduler's last push to the servicer
(``last_receive_tstamp``), and this module holds the two rules over it.

The readiness probe (the standard gRPC health service) answers from a serving
flag and one liveness rule: a scheduler that has pushed nothing for longer
than :data:`STUCK_SCHEDULER_THRESHOLD_SECS` while requests are pending is
stuck, and the service reports NOT_SERVING until it speaks again or its
pending requests drain (every waiting client gone). An idle scheduler
(nothing pending) is quiet, not stuck. :class:`StallReport` is that rule with
its reporting: probes arrive on their interval for as long as the stall
holds, so the stall is said once when it starts and again only if it clears
and comes back, and the recovery is said once; repeating it every probe
buries whatever else the scheduler is trying to report. Each servicer keeps
one report and logs through its own logger.

The deep health RPC (``HealthCheck``) submits a one-token probe request and
waits for its answer with :func:`await_probe`: healthy as soon as the
scheduler pushes anything after the probe started (a busy scheduler gets to
the probe when its turn comes; any push meanwhile says the scheduler is
alive), unhealthy the moment the probe reports a failure of its own (its
request could not be submitted, or its stream raised or ended empty: there
is nothing left to wait for), and unhealthy at the RPC's deadline with
neither. The wait is one
function so that both servicers answer the same case the same way; each
keeps its own deadline, its own messages and the release of what its probe
held.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

# Seconds of scheduler silence, with requests pending, before the readiness
# probe reports NOT_SERVING. More conservative than the servicers' own deep
# health RPC (20 s by default): the probe decides whether traffic keeps coming.
STUCK_SCHEDULER_THRESHOLD_SECS = 30.0

# Seconds between the deep health RPC's reads of the scheduler while its probe
# is out: a push, or a probe that fails at once, is answered at most this late.
PROBE_STEP_SECS = 0.5


async def await_probe(
    probe: asyncio.Future[bool],
    *,
    started: float,
    last_receive: Callable[[], float],
    timeout: float,
    step: float = PROBE_STEP_SECS,
) -> bool | None:
    """The deep health RPC's answer for ``probe``, the task driving a one-token
    request submitted at ``started`` (``time.time()``) that resolves to whether
    the request produced an output.

    Read every ``step`` seconds: ``True`` once ``last_receive()``, the time of
    the scheduler's last push, is later than ``started``, or the probe is done
    with an output; ``False`` once the probe is done without one (the engine
    refused or raised on the request, or its stream ended empty); ``None``
    once ``timeout`` seconds have passed since ``started`` with neither. The
    push is read before the verdict: while the probe waits its turn at a busy
    scheduler, every push is the scheduler saying it is alive, and the probe's
    own output arrives as one. The caller owns the probe: it cancels one still
    pending at the deadline and releases what the request held.
    """
    while time.time() - started < timeout:
        await asyncio.sleep(step)
        if last_receive() > started:
            return True
        if probe.done():
            return bool(probe.result())
    return None


class StallReport:
    """One scheduler's stall state as the readiness probe sees it."""

    def __init__(
        self,
        log: logging.Logger,
        *,
        threshold_secs: float = STUCK_SCHEDULER_THRESHOLD_SECS,
    ) -> None:
        self._log = log
        self._threshold_secs = threshold_secs
        self._reported = False

    def stuck(self, last_receive_tstamp: float, pending: int) -> bool:
        """Whether the scheduler is stuck now: silent for longer than the
        threshold with ``pending`` requests waiting on it. Logs the stall when
        it starts and the recovery when it ends."""
        silence = time.time() - last_receive_tstamp
        if silence > self._threshold_secs and pending > 0:
            if not self._reported:
                self._reported = True
                self._log.warning(
                    "Scheduler not responsive: %.1fs since last receive, %d pending requests",
                    silence,
                    pending,
                )
            return True
        if self._reported:
            self._reported = False
            self._log.info("Scheduler is responsive again")
        return False
