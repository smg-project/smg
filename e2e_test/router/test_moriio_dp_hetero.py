"""Opt-in MI355X candidate regression: P DP2/TP2 -> D DP2/TP1, WRITE/XGMI.

Run only inside an owned eight-GPU allocation with a validated engine carrying
vLLM #61726 and #64574. This does not remove the existing MI300X/Llama guards.
Set E2E_MORIIO_DP_HETERO=1, E2E_RUNTIME=vllm, E2E_GPU_TIER=8,
E2E_VLLM_KV_BACKEND=moriio, E2E_VLLM_MORIIO_MODE=write and
E2E_VLLM_MORIIO_BACKEND=xgmi. E2E_MORIIO_GATEWAY_COMMAND optionally supplies
a JSON argv, e.g. ["/work/smg-release-venv/bin/python","-I","-m","smg.cli","launch"].
E2E_MORIIO_MODEL_PATH may name an existing absolute model snapshot, avoiding downloads.
"""

import inspect
import json
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from infra import Gateway, WorkerType, start_workers, stop_workers
from infra.constants import (
    get_gpu_offset,
    get_runtime,
    vllm_kv_backend,
    vllm_moriio_backend,
    vllm_moriio_mode,
)
from infra.model_specs import MODEL_SPECS, get_model_spec

MODEL = "Qwen/Qwen3-30B-A3B"
pytestmark = [
    pytest.mark.engine("vllm"),
    pytest.mark.gpu(8),
    pytest.mark.e2e,
    pytest.mark.model(MODEL),
    pytest.mark.skipif(
        os.environ.get("E2E_MORIIO_DP_HETERO") != "1",
        reason="explicit MI355X candidate opt-in required",
    ),
]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        sample = re.fullmatch(
            r"(vllm:prompt_tokens_total|vllm:prompt_tokens_by_source_total)\{([^}]*)\}\s+(\S+)",
            line,
        )
        if not sample:
            continue
        name, labels, raw = sample.groups()
        label_name = "engine" if name.endswith(":prompt_tokens_total") else "source"
        label = re.search(rf'(?:^|,){label_name}="([^"]+)"', labels)
        if not label:
            continue
        key = f"{label_name}={label[1]}"
        if label_name == "engine":
            assert label[1] in ("0", "1"), f"unexpected DP engine {label[1]}"
        value = float(raw)
        assert math.isfinite(value) and value >= 0, f"invalid counter {key}={value}"
        values[key] = values.get(key, 0) + value
    required = {
        "engine=0",
        "engine=1",
        "source=external_kv_transfer",
        "source=local_compute",
        "source=local_cache_hit",
    }
    assert required <= values.keys(), f"missing counter series: {required - values.keys()}"
    return {key: values[key] for key in required}


def metric_delta(before, after):
    assert before.keys() == after.keys(), "counter set changed"
    result = {key: after[key] - before[key] for key in before}
    assert all(math.isfinite(value) and value >= 0 for value in result.values()), result
    return result


def actual_layout(text, tp):
    initial, groups, models, connectors, assigned = {}, {}, set(), {}, {}
    for line in _ANSI.sub("", text).splitlines():
        prefix = re.match(r"^\((Worker(?:_\w+)*) pid=(\d+)\).*?\[([^]]+)\]\s+(.*)$", line)
        if not prefix:
            continue
        name, pid, source, body = prefix.groups()
        init = re.fullmatch(
            r"world_size=(\d+) rank=(\d+) local_rank=(\d+) distributed_init_method=(\S+) backend=(\S+)",
            body,
        )
        if init and source.startswith("parallel_state.py:"):
            assert pid not in initial, "duplicate worker initialization"
            initial[pid] = init.groups()
        group = re.fullmatch(r"Worker_DP(\d+)(?:_TP(\d+))?_EP(\d+)", name)
        if group:
            assert pid in initial, "group has no prior world initialization"
            dp, tensor, ep = group.groups()
            assert (tensor is None) == (tp == 1), "TP suffix disagrees with group size"
            row = (int(dp), int(tensor or 0), int(ep))
            assert pid not in groups or groups[pid] == row, "conflicting worker groups"
            groups[pid] = row
            if source.startswith("model_runner.py:") and body.startswith("Model loading took "):
                models.add(pid)
            connector = re.fullmatch(
                r"Creating v1 connector with name: MoRIIOConnector and engine_id: (\S+)_dp(\d+)",
                body,
            )
            if source.startswith("factory.py:") and connector:
                assert int(connector[2]) == row[0]
                assert pid not in connectors or connectors[pid] == connector[1]
                connectors[pid] = connector[1]
        assignment = re.search(
            r"rank (\d+) in world size (\d+) is assigned as DP rank (\d+),.*?, TP rank (\d+),.*?, EP rank (\d+),",
            body,
        )
        if source.startswith("parallel_state.py:") and assignment:
            assert pid not in assigned
            assigned[pid] = tuple(map(int, assignment.groups()))
    assert len(initial) == 2 * tp and set(initial) == set(groups) == models == set(connectors)
    assert {int(row[1]) for row in initial.values()} == set(range(2 * tp))
    assert len({row[3] for row in initial.values()}) == len(set(connectors.values())) == 1
    for pid, (world, rank, local, _, backend) in initial.items():
        rank = int(rank)
        assert int(world) == 2 * tp and int(local) == rank and backend == "nccl"
        assert groups[pid] == (rank // tp, rank % tp, rank)
        if pid in assigned:
            assert assigned[pid] == (rank, 2 * tp, *groups[pid])
    assert any(int(initial[pid][1]) == 0 for pid in assigned), "missing rank-zero cross-check"
    return {pid: {"rank": int(initial[pid][1]), "dp_tp_ep": groups[pid]} for pid in initial}


def _metrics(worker):
    response = httpx.get(f"{worker.http_url}/metrics", timeout=5)
    response.raise_for_status()
    return parse_metrics(response.text)


def _snapshot(workers):
    return {role: _metrics(worker) for role, worker in workers.items()}


def _settle(workers, before, expected):
    deadline = time.monotonic() + 15
    while True:
        delta = {
            role: metric_delta(before[role], after) for role, after in _snapshot(workers).items()
        }
        assert (
            delta["decode"]["source=local_compute"]
            == delta["decode"]["source=local_cache_hit"]
            == 0
        ), delta
        if (
            all(sum(row[f"engine={rank}"] for rank in (0, 1)) == expected for row in delta.values())
            and delta["decode"]["source=external_kv_transfer"] == expected
        ):
            return delta
        assert time.monotonic() < deadline, f"metrics did not settle: {delta} expected={expected}"
        time.sleep(0.25)


def _request(url, model, index, nonce):
    prompt = (
        f"Ledger request {index} nonce {nonce}. Continue the ledger.\n"
        + "".join(
            f"Ledger line {n:04d} for {nonce[:8]}: account {n * 7919 % 10007} "
            f"moved {n * 4999 % 9973} units to account {n * 31 % 997}.\n"
            for n in range(85)
        )
        + "Ledger line"
    )
    body = {"model": model, "prompt": prompt, "max_tokens": 32, "temperature": 0, "stream": False}
    row = {"index": index, "nonce": nonce, "request": body}
    try:
        response = httpx.post(f"{url}/v1/completions", json=body, timeout=120)
        row.update(status=response.status_code, response=response.json())
    except (httpx.HTTPError, ValueError) as error:
        row.update(status=None, error=repr(error))
    return row


def _usage(row):
    assert row["status"] == 200, row
    value = row["response"]["usage"]["prompt_tokens"]
    assert type(value) is int and value >= 1000, value
    return value


def test_dp2_heterogeneous_write_xgmi(tmp_path, monkeypatch):
    assert get_runtime() == "vllm" and vllm_kv_backend() == "moriio"
    assert vllm_moriio_mode() == "write" and vllm_moriio_backend() == "xgmi"
    assert get_gpu_offset() == 0, "candidate requires its own full-node GPU slice"
    model_path = os.environ.get("E2E_MORIIO_MODEL_PATH")
    if model_path:
        assert Path(model_path).is_absolute() and (Path(model_path) / "config.json").is_file(), (
            model_path
        )
        monkeypatch.setitem(MODEL_SPECS, MODEL, {**get_model_spec(MODEL), "model": model_path})
    # Check installed code capability, never infer support from a version string.
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
            get_port_offset,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
            compute_write_tp_mapping,
        )

        assert callable(compute_write_tp_mapping)
        assert (
            inspect.signature(get_port_offset).parameters["tp_size"].default
            is inspect.Parameter.empty
        )
    except (ImportError, KeyError, AssertionError) as error:
        pytest.fail(
            f"candidate requires actual vLLM #61726 mapping and #64574 mandatory TP stride: {error}"
        )
    import torch

    assert torch.version.hip and torch.cuda.device_count() == 8
    assert all("gfx950" in torch.cuda.get_device_properties(i).gcnArchName for i in range(8)), (
        "MI355X/gfx950 allocation required"
    )
    command = (
        json.loads(os.environ["E2E_MORIIO_GATEWAY_COMMAND"])
        if os.environ.get("E2E_MORIIO_GATEWAY_COMMAND")
        else None
    )
    gateway = Gateway(launch_command=command)
    directory = Path(os.environ.get("E2E_LOG_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    report, workers = {"serial": [], "concurrent": []}, {}
    common = [
        "--distributed-executor-backend",
        "mp",
        "--enforce-eager",
        "--dtype",
        "bfloat16",
        "--block-size",
        "16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        ".25",
        "--max-num-seqs",
        "16",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
    ]
    try:
        for role, tp, offset in (("prefill", 2, 0), ("decode", 1, 4)):
            workers[role] = start_workers(
                MODEL,
                engine="vllm",
                worker_type=WorkerType.PREFILL if role == "prefill" else WorkerType.DECODE,
                tp=tp,
                dp_size=2,
                gpu_offset=offset,
                wait_ready=False,
                timeout=650,
                log_dir=str(directory),
                extra_engine_args=common,
            )[0]
        for role, worker in workers.items():
            worker.wait_ready(650)
            report[f"{role}_layout"] = actual_layout(worker.read_log(), worker.tp)
        gateway.start(
            prefill_workers=[],
            decode_workers=[],
            extra_args=["--prefill-policy", "round_robin", "--decode-policy", "round_robin"],
            log_dir=str(directory),
        )
        for role, worker in workers.items():
            ok, detail = gateway.add_worker(
                worker.base_url, worker_type=role, spec=worker.moriio_registration()
            )
            assert ok, detail
        deadline = time.monotonic() + 60
        while True:
            registered = gateway.list_workers(strict=True)
            if len(registered) == 2 and all(worker.status == "healthy" for worker in registered):
                break
            assert time.monotonic() < deadline, registered
            time.sleep(0.5)
        model = get_model_spec(MODEL)["model"]
        ranks = []
        for index in range(4):
            before = _snapshot(workers)
            row = _request(gateway.base_url, model, index, uuid.uuid4().hex)
            report["serial"].append(row)
            row["metrics_delta"] = _settle(workers, before, _usage(row))
            active = {
                role: [rank for rank in (0, 1) if delta[f"engine={rank}"] > 0]
                for role, delta in row["metrics_delta"].items()
            }
            assert active["prefill"] == active["decode"] and len(active["decode"]) == 1, active
            ranks.extend(active["decode"])
            ref_before = _metrics(workers["prefill"])
            reference = _request(workers["prefill"].base_url, model, index, row["nonce"])
            row["reference"] = reference
            assert _usage(reference) == _usage(row)
            assert (
                reference["response"]["choices"][0]["text"] == row["response"]["choices"][0]["text"]
            )
            deadline = time.monotonic() + 15
            while True:
                reference_delta = metric_delta(ref_before, _metrics(workers["prefill"]))
                if sum(reference_delta[f"engine={rank}"] for rank in (0, 1)) == _usage(reference):
                    break
                assert time.monotonic() < deadline, "reference metrics did not settle"
                time.sleep(0.25)
        assert set(ranks) == {0, 1}, ranks
        before = _snapshot(workers)
        nonces = [uuid.uuid4().hex for _ in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            report["concurrent"] = list(
                pool.map(lambda item: _request(gateway.base_url, model, *item), enumerate(nonces))
            )
        rows = report["concurrent"]
        total = sum(_usage(row) for row in rows)
        assert len({row["nonce"] for row in rows}) == 8
        for row in rows:
            text = row["response"]["choices"][0]["text"]
            assert row["nonce"][:8] in text
            assert not any(other["nonce"][:8] in text for other in rows if other is not row)
        delta = report["concurrent_metrics_delta"] = _settle(workers, before, total)
        assert all(
            delta["prefill"][f"engine={rank}"] == delta["decode"][f"engine={rank}"] > 0
            for rank in (0, 1)
        ), delta
        report["passed"] = True
    finally:
        (directory / "moriio-dp-hetero.json").write_text(json.dumps(report, indent=2) + "\n")
        gateway.shutdown()
        stop_workers(list(workers.values()))
