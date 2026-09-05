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
        self._abort_claims: dict[str, RequestState] = {}
        self._lock = threading.Lock()

    def register(self, state: RequestState) -> None:
        """Register a request, preserving the manager's overwrite semantics."""
        with self._lock:
            self._states[state.request_id] = state
            self._abort_claims.pop(state.request_id, None)

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
            self._abort_claims[request_id] = state
            return state

    async def abort_if_active(
        self, request_id: str, forward: Callable[[], Awaitable[None]]
    ) -> bool:
        """Forward an abort only after atomically claiming an active request."""
        if self.claim_abort(request_id) is None:
            return False

        await forward()
        return True

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
            state = self._states.get(request_id)
            local_claim = self._abort_claims.get(request_id) is state
            if state is None or (state.finished and not local_claim):
                return None

            state.finished = True
            state.stream_finished = True
            if local_claim:
                self._abort_claims.pop(request_id, None)
            return state

    def remove(self, request_id: str, state: RequestState | None = None) -> RequestState | None:
        """Remove a request unless the ID has since been reused by another state."""
        with self._lock:
            current = self._states.get(request_id)
            if current is None or (state is not None and current is not state):
                return None
            removed = self._states.pop(request_id)
            if self._abort_claims.get(request_id) is removed:
                self._abort_claims.pop(request_id, None)
            return removed
