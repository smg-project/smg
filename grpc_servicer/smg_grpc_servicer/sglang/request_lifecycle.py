"""Thread-safe lifecycle transitions for SGLang gRPC requests."""

import threading
from collections.abc import Awaitable, Callable
from typing import Protocol


class RequestState(Protocol):
    """State fields used by request lifecycle transitions."""

    request_id: str
    finished: bool
    stream_finished: bool


class RequestLifecycle:
    """Coordinate request registration, completion, abort, and cleanup."""

    def __init__(self, states: dict[str, RequestState]):
        self._states = states
        self._abort_claims: dict[str, list[RequestState]] = {}
        self._lock = threading.Lock()

    def register(self, state: RequestState) -> None:
        """Register a request, preserving the manager's overwrite semantics."""
        with self._lock:
            self._states[state.request_id] = state

    def get(self, request_id: str) -> RequestState | None:
        """Return the currently registered state for a request ID."""
        with self._lock:
            return self._states.get(request_id)

    def claim_abort(self, request_id: str) -> RequestState | None:
        """Atomically claim an active request for abort exactly once."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.finished:
                return None

            state.finished = True
            state.stream_finished = True
            self._abort_claims.setdefault(request_id, []).append(state)
            return state

    async def abort_if_active(
        self, request_id: str, forward: Callable[[], Awaitable[None]]
    ) -> bool:
        """Forward an abort only after atomically claiming an active request."""
        state = self.claim_abort(request_id)
        if state is None:
            return False

        try:
            await forward()
        except BaseException:
            self._rollback_abort(request_id, state)
            raise
        return True

    def _rollback_abort(self, request_id: str, state: RequestState) -> None:
        """Release a failed abort claim if it still belongs to this state."""
        with self._lock:
            claims = self._abort_claims.get(request_id)
            if claims is None:
                return

            claim_index = next(
                (index for index, claim in enumerate(claims) if claim is state),
                None,
            )
            if claim_index is None:
                return

            claims.pop(claim_index)
            if not claims:
                self._abort_claims.pop(request_id)

            if self._states.get(request_id) is state:
                state.finished = False
                state.stream_finished = False

    def finish(
        self,
        request_id: str,
        state: RequestState,
        *,
        stream_finished: bool = False,
    ) -> bool:
        """Atomically finish a request if it is still current and active."""
        with self._lock:
            if self._states.get(request_id) is not state or state.finished:
                return False

            state.finished = True
            if stream_finished:
                state.stream_finished = True
            return True

    def accept_scheduler_abort(self, request_id: str) -> RequestState | None:
        """Accept a scheduler abort for an active or locally aborted request."""
        with self._lock:
            claims = self._abort_claims.get(request_id)
            if claims:
                state = claims.pop(0)
                if not claims:
                    self._abort_claims.pop(request_id)
            else:
                state = self._states.get(request_id)
                if state is None or state.finished:
                    return None

            state.finished = True
            state.stream_finished = True
            return state

    def remove(self, request_id: str, state: RequestState | None = None) -> RequestState | None:
        """Remove a request unless the ID has since been reused by another state."""
        with self._lock:
            current = self._states.get(request_id)
            if current is None or (state is not None and current is not state):
                return None
            return self._states.pop(request_id)
