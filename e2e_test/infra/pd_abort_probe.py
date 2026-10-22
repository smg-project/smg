"""One controlled client close; scheduler termination is not KV-release proof."""

import hashlib

import httpx

from .pd_metrics import wait_for_generated_text
from .pd_probe import observe_request
from .pd_probe_contract import validate_prefix

BUDGET = 1024


def decode_trace(files, workers, identity, root):
    validate_prefix(files, workers, identity, root)
    name, (_, records) = next(
        (name, value) for name, value in files.items() if value[1][0]["role"] == "decode"
    )
    admission = next(
        row
        for row in records
        if row["event"] == "add_request_return"
        and identity.request_id in row["request"]["request_id"]
    )
    return name, records, admission["request"]["request_id"]


def target_snapshots(records, internal):
    for record in records:
        entries = record.get("requests", [record])
        for entry in entries:
            snapshot = entry.get("request", {})
            if snapshot.get("request_id") == internal:
                yield record, snapshot


def validate_abort(files, workers, identity, root, before, internal):
    name, records, current = decode_trace(files, workers, identity, root)
    assert current == internal, "decode request identity changed"
    assert files.keys() == before.keys() and all(
        files[key][0].startswith(value[0]) for key, value in before.items()
    ), "trace prefix changed after close"
    terminals = [
        (record, snapshot)
        for record, snapshot in target_snapshots(records, internal)
        if snapshot.get("status", "").startswith("FINISHED_")
    ]
    if not terminals:
        raise ValueError("decode abort not observed")
    assert all(snapshot["status"] == "FINISHED_ABORTED" for _, snapshot in terminals), (
        "decode ended without an abort"
    )
    finished = [(r, s) for r, s in terminals if r["event"] == "finish_requests_return"]
    assert len(finished) == 1, "missing or repeated native finish return"
    record, snapshot = finished[0]
    assert record["seq"] > before[name][1][-1]["seq"], "abort preceded pre-close checkpoint"
    assert snapshot["max_tokens"] == BUDGET and snapshot["abort_immediately"] is False
    output = snapshot["num_output_tokens"]
    assert type(output) is int and 0 < output <= BUDGET // 4, "abort too near generation budget"
    return dict(internal_request_id=internal, finish_seq=record["seq"], output_tokens=output)


def run_decode_abort(base_url, model, context, state, root, workers, identity):
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        headers=identity.headers,
        json=dict(
            model=model,
            messages=[{"role": "user", "content": "Count upwards, one number per line."}],
            temperature=0,
            max_tokens=BUDGET,
            ignore_eos=True,
            n=1,
            stream=True,
        ),
        timeout=60,
    ) as response:
        response.raise_for_status()
        assert response.status_code == 200
        lines = response.iter_lines()
        wait_for_generated_text(lines)
        assert not response.is_closed, "stream closed before the client action"
        before = observe_request(context["trace_dir"], workers, identity, state, root)
        _, records, internal = decode_trace(before, workers, identity, root)
        assert not any(
            snapshot["status"].startswith("FINISHED_")
            for _, snapshot in target_snapshots(records, internal)
        ), "decode finished before client close"
        state["decode_abort"] = dict(
            internal_request_id=internal,
            budget=BUDGET,
            before_close_sha256={
                key: hashlib.sha256(raw).hexdigest() for key, (raw, _) in before.items()
            },
        )
        response.close()
        assert response.is_closed, "client stream did not close"
    state["decode_abort"]["client_closed"] = True

    def check(files, workers, identity, expected_root):
        state["decode_abort"].update(
            validate_abort(files, workers, identity, expected_root, before, internal)
        )

    observe_request(context["trace_dir"], workers, identity, state, root, validator=check)
