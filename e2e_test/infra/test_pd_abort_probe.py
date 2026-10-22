"""CPU consumer and real HTTP-client close controls; traces are fabricated."""

import json

import httpx
import pytest
from infra import pd_abort_probe as abort
from infra import pd_probe as probe

pytest_plugins = ["infra.test_pd_probe_contract"]


@pytest.fixture
def pending(sample):
    rows = next(rows for rows in sample.records.values() if rows[0]["role"] == "decode")
    rows[1]["request"]["max_tokens"] = rows[2]["requests"][0]["request"]["max_tokens"] = (
        abort.BUDGET
    )
    sample.before = sample.write()
    sample.internal = rows[1]["request"]["request_id"]
    snapshot = dict(rows[1]["request"], status="FINISHED_ABORTED", num_output_tokens=7)
    sample.terminal = dict(
        rows[0], seq=4, event="finish_requests_return", requests=[dict(request=snapshot)]
    )
    rows.append(sample.terminal)
    return sample


def validate(sample):
    return abort.validate_abort(
        sample.write(),
        sample.workers,
        sample.identity,
        "/pinned/vllm",
        sample.before,
        sample.internal,
    )


def test_post_close_native_abort_consumer(pending):
    assert validate(pending) == dict(
        internal_request_id=pending.internal, finish_seq=4, output_tokens=7
    )


def test_native_cleanup_prefix_retries_until_finish_return(monkeypatch, pending):
    rows = next(rows for rows in pending.records.values() if rows[0]["role"] == "decode")
    terminal = rows.pop()
    cleanup = dict(
        terminal, event="free_request_return", request=terminal["requests"][0]["request"]
    )
    del cleanup["requests"]
    rows.append(cleanup)
    partial = pending.write()
    terminal["seq"] = 5
    rows.append(terminal)
    complete = pending.write()
    prefixes, reads = iter((partial, complete)), []

    def read(directory):
        assert directory == pending.directory
        reads.append(True)
        return next(prefixes)

    monkeypatch.setattr(probe, "read_prefix", read)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    result = probe.observe_request(
        pending.directory,
        pending.workers,
        pending.identity,
        {},
        "/pinned/vllm",
        validator=lambda files, workers, identity, root: abort.validate_abort(
            files, workers, identity, root, pending.before, pending.internal
        ),
    )
    assert result is complete and len(reads) == 2


def test_duplicate_native_finish_is_rejected(pending):
    rows = next(rows for rows in pending.records.values() if rows[0]["role"] == "decode")
    rows.append(dict(pending.terminal, seq=5))
    with pytest.raises(AssertionError, match="repeated native finish"):
        validate(pending)


@pytest.mark.parametrize(
    "fault",
    [
        "length",
        "error",
        "reaper",
        "missing",
        "budget",
        "zero",
        "late",
        "identity",
        "before_close",
        "rewritten",
        "trace",
    ],
)
def test_terminal_evidence_is_required(pending, fault):
    row = pending.terminal
    snapshot = row["requests"][0]["request"]
    if fault in ("length", "error"):
        snapshot["status"] = "FINISHED_LENGTH_CAPPED" if fault == "length" else "FINISHED_ERROR"
    elif fault == "reaper":
        row["event"] = "free_blocks_return"
        row["request"] = row.pop("requests")[0]["request"]
    elif fault == "missing":
        next(rows for rows in pending.records.values() if rows[0]["role"] == "decode").pop()
    elif fault in ("budget", "zero", "late"):
        snapshot["max_tokens" if fault == "budget" else "num_output_tokens"] = {
            "budget": 8,
            "zero": 0,
            "late": abort.BUDGET,
        }[fault]
    elif fault == "identity":
        snapshot["request_id"] = "other-request"
    elif fault == "before_close":
        pending.before = pending.write()
    elif fault == "trace":
        name = next(name for name, rows in pending.records.items() if rows[0]["role"] == "decode")
        del pending.records[name]
        (pending.directory / name).unlink()
    else:
        next(rows for rows in pending.records.values() if rows[0]["role"] == "decode")[1][
            "request"
        ]["num_prompt_tokens"] = 1
    with pytest.raises((AssertionError, ValueError)):
        validate(pending)


@pytest.mark.parametrize("text", [True, False])
def test_real_http_client_closes_unconsumed_stream(monkeypatch, pending, text):
    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            try:
                yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                if text:
                    yield b'data: {"choices":[{"delta":{"content":"1"},"finish_reason":null}]}\n\n'
                else:
                    yield b"data: [DONE]\n\n"
                raise AssertionError("client consumed the whole stream")
            finally:
                self.close()

        def close(self):
            self.closed = True

    stream, sent, polls = Stream(), {}, []

    def handle(request):
        sent.update(json.loads(request.content))
        assert request.headers["x-request-id"] == pending.identity.request_id
        return httpx.Response(200, stream=stream)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(abort.httpx, "stream", client.stream)

    def observe(directory, workers, identity, state, root, validator=None):
        polls.append(validator)
        assert stream.closed is (validator is not None)
        files = pending.before if validator is None else pending.write()
        state["files"] = files
        if validator:
            validator(files, workers, identity, root)
        return files

    monkeypatch.setattr(abort, "observe_request", observe)
    state = {}
    args = (
        "http://cpu",
        "model",
        {"trace_dir": pending.directory},
        state,
        "/pinned/vllm",
        pending.workers,
        pending.identity,
    )
    if text:
        abort.run_decode_abort(*args)
        assert (
            state["decode_abort"]["client_closed"] and state["decode_abort"]["output_tokens"] == 7
        )
        assert len(polls) == 2 and sent["ignore_eos"] and sent["max_tokens"] == abort.BUDGET
    else:
        with pytest.raises(AssertionError, match="without generated text"):
            abort.run_decode_abort(*args)
        assert not polls and "decode_abort" not in state
    assert stream.closed
    client.close()
