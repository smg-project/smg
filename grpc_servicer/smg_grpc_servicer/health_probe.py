"""What the readiness probe says about a scheduler that went quiet.

The SGLang and TokenSpeed servicers answer the standard gRPC health probe
from a serving flag and one liveness rule: a scheduler that has pushed
nothing for longer than :data:`STUCK_SCHEDULER_THRESHOLD_SECS` while
requests are pending is stuck, and the service reports NOT_SERVING until it
speaks again or its pending requests drain (every waiting client gone). An
idle scheduler (nothing pending) is quiet, not stuck.
:class:`StallReport` is that rule with its reporting: probes arrive on
their interval for as long as the stall holds, so the stall is said once
when it starts and again only if it clears and comes back, and the
recovery is said once; repeating it every probe buries whatever else the
scheduler is trying to report. Each servicer keeps one report and logs
through its own logger.
"""

from __future__ import annotations

import logging
import time

# Seconds of scheduler silence, with requests pending, before the readiness
# probe reports NOT_SERVING. More conservative than the servicers' own deep
# health RPC (20 s by default): the probe decides whether traffic keeps coming.
STUCK_SCHEDULER_THRESHOLD_SECS = 30.0


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
