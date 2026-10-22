"""Correlate an aborted request's native immediate/deferred reference releases."""

from collections import Counter

from .pd_local_kv_state import block_snapshot, capture, held_blocks, released, require
from .pd_scheduler_trace import nonnegative_int


class LocalKVTracker:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.pending = {}
        self.sequence = 0
        self.unknown = False

    def before_free(self, request):
        if self.unknown or request.status.name != "FINISHED_ABORTED" or request.abort_immediately:
            return None
        require(len(self.pending) < 16, "too many pending local releases")
        return capture(self.scheduler, request)

    def after_free(self, state):
        if state is None:
            return []
        self.sequence += 1
        ticket = self.sequence
        scheduler = self.scheduler
        require(not held_blocks(scheduler, state["request_id"]), "request retains block mapping")
        added = [entry for entry in scheduler.deferred_frees if id(entry) not in state["queue_ids"]]
        if not added:
            return [released(scheduler, state, "immediate", ticket)]
        require(len(added) == 1, "ambiguous deferred entry")
        entry = added[0]
        fence, blocks = entry
        require(
            Counter(map(id, blocks)) == Counter(map(id, state["blocks"])),
            "deferred ownership differs",
        )
        require(
            nonnegative_int(fence) > nonnegative_int(scheduler.processed_step_seq),
            "invalid deferred fence",
        )
        after = block_snapshot(state["pool"], state["blocks"])
        require(after == state["before"], "references changed while being deferred")
        self.pending[ticket] = (state, entry)
        return [
            {
                "request_id": state["request_id"],
                "pool_identity": f"{id(state['pool']):x}",
                "phase": "deferred",
                "ticket": ticket,
                "fence": fence,
                "processed_step_seq": scheduler.processed_step_seq,
                "before": state["before"],
                "after": after,
                "request_holds_after": [],
                "held_by": "deferred_queue",
                "request_references_released": False,
                "exclusive_pages_reusable": [],
                "transfer_ack": None,
                "peer_memory_released": None,
            }
        ]

    def before_drain(self):
        if self.unknown:
            return []
        queue = list(self.scheduler.deferred_frees)
        require(len(queue) <= 32, "too many deferred entries")
        observations = []
        for ticket, (state, entry) in self.pending.items():
            require(any(item is entry for item in queue), "pending entry disappeared before drain")
            target = set(map(id, state["blocks"]))
            require(
                not any(
                    item is not entry and target.intersection(map(id, item[1])) for item in queue
                ),
                "overlapping deferred owners",
            )
            before = block_snapshot(state["pool"], state["blocks"])
            observations.append((ticket, dict(state, before=before), entry))
        return observations

    def after_drain(self, observations):
        records = []
        queue = list(self.scheduler.deferred_frees)
        for ticket, state, entry in observations:
            if any(item is entry for item in queue):
                require(
                    block_snapshot(state["pool"], state["blocks"]) == state["before"],
                    "pending references changed during drain",
                )
                continue
            require(
                nonnegative_int(entry[0]) <= self.scheduler.processed_step_seq,
                "fence was not reached",
            )
            records.append(released(self.scheduler, state, "drained", ticket, entry[0]))
            del self.pending[ticket]
        return records
