"""Real client-close consumer wiring with native-method CPU observations."""

import json

import httpx
from infra import pd_abort_probe as abort
from infra import pd_probe as probe
from infra.pd_local_kv_compat import SOURCE_SHA256
from infra.pd_local_kv_test_support import observed

pytest_plugins = ["infra.test_pd_abort_probe"]


def test_existing_abort_entry_waits_for_post_fence_reusable_page(monkeypatch, pending):
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "1")
    rows = next(rows for rows in pending.records.values() if rows[0]["role"] == "decode")
    terminal = rows.pop()
    for records in pending.records.values():
        records[0]["source_contract"]["local_kv_sources"] = dict(SOURCE_SHA256)
    before = pending.write()
    scheduler, req, raw = observed(deferred=True)
    held = scheduler.owners.pop(req.request_id)
    scheduler.requests.pop(req.request_id)
    req.request_id = pending.internal
    scheduler.owners[req.request_id] = held
    scheduler.requests[req.request_id] = req
    scheduler._free_blocks(req)
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    events = [json.loads(line) for line in raw if json.loads(line)["event"] == "local_kv_return"]
    context = {
        key: rows[0][key]
        for key in (
            "run",
            "attempt",
            "worker",
            "role",
            "worker_instance",
            "instance",
            "pid",
            "schema",
            "observation",
        )
    }
    # Re-envelope native-method fields with the fabricated fleet fixture identity.
    rows.append(dict(events[0], **context, seq=4))
    terminal["seq"] = 5
    rows.append(terminal)
    prefix_without_release = pending.write()
    rows.append(dict(events[1], **context, seq=6))
    complete = pending.write()
    prefixes, reads = iter((before, prefix_without_release, complete)), []

    def read(directory):
        assert directory == pending.directory
        reads.append(True)
        return next(prefixes)

    monkeypatch.setattr(probe, "read_prefix", read)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)

    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"1"},"finish_reason":null}]}\n\n'
            raise AssertionError("client consumed the complete response")

        def close(self):
            self.closed = True

    stream = Stream()

    def handle(request):
        assert request.headers["x-request-id"] == pending.identity.request_id
        assert json.loads(request.content)["max_tokens"] == abort.BUDGET
        return httpx.Response(200, stream=stream)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(abort.httpx, "stream", client.stream)
        state = {}
        abort.run_decode_abort(
            "http://cpu.invalid",
            "unchanged-model",
            {"trace_dir": pending.directory},
            state,
            "/pinned/vllm",
            pending.workers,
            pending.identity,
        )
    assert stream.closed and len(reads) == 3
    assert state["decode_abort"]["finish_seq"] == 5
    assert state["local_kv"]["release_seq"] == 6
    assert state["local_kv"]["exclusive_pages_reusable"] == [1]
    assert state["local_kv"]["accepted"] is False
