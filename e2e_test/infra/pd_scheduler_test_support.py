"""Native lifecycle contract shared by the observer CPU tests."""

import json
from enum import IntEnum, auto
from types import SimpleNamespace

from infra.pd_scheduler_observer import SchedulerObserverMixin
from infra.pd_scheduler_trace import SchedulerTrace


class Status(IntEnum):
    RUNNING = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()


def request(request_id="chatcmpl-nonce-0123abcd"):
    return SimpleNamespace(
        request_id=request_id,
        status=Status.RUNNING,
        abort_immediately=False,
        num_computed_tokens=12,
        num_output_tokens=2,
        num_prompt_tokens=10,
        num_output_placeholders=0,
        last_sched_seq=3,
        max_tokens=100,
    )


class NativeContract:
    """Small stub of the pinned methods' lifecycle, not an engine simulator."""

    def __init__(self, config, marker=None):
        self.scheduler_config = config
        self.marker = marker
        self.requests = {}
        self.calls = [("init", marker)]
        self.ignore_abort = False
        self.exception = None
        self.finished = []
        self.free_result = ({"transfer": "untouched"}, None)
        self.freed = []

    def add_request(self, req):
        self.calls.append(("add", req.request_id))
        self.requests[req.request_id] = req

    def finish_requests(self, ids, status):
        self.calls.append(("finish", ids, status))
        if self.exception:
            raise self.exception
        self.finished = []
        if self.ignore_abort:
            return self.finished
        selected = [ids] if isinstance(ids, str) else list(ids or self.requests)
        for request_id in selected:
            req = self.requests.get(request_id)
            if req is not None and req.status == Status.RUNNING:
                req.status = status
                self._free_request(req)
                self.finished.append(req)
        return self.finished

    def _free_request(self, req, delay_free_blocks=False):
        self.calls.append(("free_request", req.request_id, delay_free_blocks))
        if not delay_free_blocks:
            self._free_blocks(req)
        return self.free_result

    def _free_blocks(self, req):
        self.calls.append(("free_blocks", req.request_id))
        self._free_request_blocks(req)
        del self.requests[req.request_id]

    def _free_request_blocks(self, req):
        self.calls.append(("request_blocks", req.request_id))
        self.freed.append(req.request_id)


def observed(base=NativeContract, asynchronous=False, sink=None):
    records = []
    trace = SchedulerTrace(
        sink or records.append,
        {
            "run": "cpu-contract",
            "attempt": "1",
            "worker": "decode-0",
            "role": "decode",
            "pid": 123,
            "instance": "contract-stub",
        },
    )

    class Observed(SchedulerObserverMixin, base):
        _observer_expected_async = asynchronous

        def _make_observer_trace(self):
            return trace

    instance = Observed(SimpleNamespace(async_scheduling=asynchronous), marker=object())
    return instance, trace, records


def events(records, name):
    return [event for line in records if (event := json.loads(line))["event"] == name]


def actual_aborted_ids(records):
    return [
        item["request"]["request_id"]
        for event in events(records, "finish_requests_return")
        for item in event["requests"]
        if item["request"]["status"] == "FINISHED_ABORTED"
    ]
