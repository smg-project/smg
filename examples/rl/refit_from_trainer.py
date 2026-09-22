#!/usr/bin/env python3
"""Refit every selected engine over NCCL from a trainer process, through SMG.

    python examples/rl/refit_from_trainer.py --smg http://127.0.0.1:30000 \
        --model-path /ckpt/step-42 --weight-version 42 --selector engine=tokenspeed

This is the refit path TokenSpeed implements, and the one slime drives: the
trainer holds rank 0 of a NCCL process group that every engine rank joins, then
pushes each weight with `dist.broadcast(..., src=0)` while the engines receive
them in the same order and feed them to the model's `load_weights`. TokenSpeed
answers HTTP 501 to `update_weights_from_disk`, so `refit_from_disk.py` is for
SGLang only.

The sequence, in order:

    pause_generation                 fan-out (`smg.rl.paused`)
    init_weights_update_group        per worker -- each needs its own rank_offset
    update_weights_from_distributed  fan-out, once per --chunk parameters
    destroy_weights_update_group     fan-out
    continue_generation              fan-out (`smg.rl.paused`)

then one `/generate` through SMG to confirm an engine reports the new
`meta_info.weight_version`.

Unlike the other examples this one runs where a trainer runs: it needs `torch`
built with CUDA and `transformers`, and it loads --model-path onto --device to
stand in for a training loop's live policy weights.

Both halves of a step block on each other: `init_weights_update_group` does not
return until the group forms, and `update_weights_from_distributed` does not
return until every weight in the batch has been received. So each is issued
from a thread while the trainer's own rendezvous and broadcasts run on the main
thread. An engine that dies mid-batch leaves the broadcast waiting on NCCL's
own watchdog, not on --timeout.
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import socket
import sys
import threading
import traceback
import urllib.request
from typing import Any

import torch
import torch.distributed as dist
import transformers
from smg.rl import RL, CallResult, FanoutResult, Worker, paused
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    rendezvous,
)
from transformers import AutoModelForCausalLM

# torch 2.6 renamed `_new_process_group_helper`'s backend-options keyword.
_PG_OPTIONS_KW = (
    "backend_options"
    if "backend_options" in inspect.signature(_new_process_group_helper).parameters
    else "pg_options"
)


def rank_layout(tp_sizes: list[int]) -> tuple[int, list[int]]:
    """Rank assignment for a trainer at rank 0 followed by each engine's ranks.

    Returns `(world_size, rank_offsets)`. Engine `k` occupies ranks
    `rank_offsets[k] .. rank_offsets[k] + tp_sizes[k] - 1` -- the engine adds
    its own tensor-parallel rank to the offset -- and the trainer keeps rank 0,
    so `world_size == 1 + sum(tp_sizes)`.
    """
    offsets: list[int] = []
    next_rank = 1
    for tp in tp_sizes:
        if tp < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp}")
        offsets.append(next_rank)
        next_rank += tp
    return next_rank, offsets


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _load_policy(model_path: str, device: torch.device) -> Any:
    """The trainer's policy weights, standing in for a live training model."""
    major = transformers.__version__.split(".")[0]
    kwarg = "dtype" if major.isdigit() and int(major) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_path, **{kwarg: torch.bfloat16})
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _join_trainer_group(
    master_address: str,
    master_port: int,
    world_size: int,
    group_name: str,
    timeout: float,
) -> Any:
    """Build the trainer's rank-0 side of the engines' weight-update group.

    Each engine joins a non-default group keyed by `PrefixStore(group_name, ...)`
    over torch's TCP store, through the same private helper torch's own
    `init_process_group` uses. The trainer has to build its side identically: a
    plain `init_process_group` prefixes the store differently and the first
    broadcast would deadlock instead of matching up.

    Rank 0 hosts the store, and the constructor does not return until every
    rank has checked in, so the engines' HTTP calls must already be in flight.
    """
    delta = datetime.timedelta(seconds=timeout)
    store, rank, world_size = next(
        rendezvous(f"tcp://{master_address}:{master_port}", 0, world_size, timeout=delta)
    )
    store.set_timeout(delta)
    store = PrefixStore(group_name, store)
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend("nccl"),
        store,
        group_name=group_name,
        timeout=delta,
        **{_PG_OPTIONS_KW: None},
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def _print_fanout(label: str, res: FanoutResult) -> None:
    print(f"{label}: {res.succeeded}/{res.total} ok")
    for wid, r in sorted(res.results.items()):
        print(f"  {wid}: HTTP {r.status} in {r.latency_ms} ms -> {json.dumps(r.body)[:120]}")
    for f in res.failed:
        print(f"  {f.worker_id}: FAILED {f.error}: {f.message}", file=sys.stderr)


def _print_calls(label: str, results: dict[str, CallResult]) -> None:
    print(f"{label}: {len(results)}/{len(results)} ok")
    for wid, r in sorted(results.items()):
        print(f"  {wid}: HTTP {r.status} in {r.latency_ms} ms -> {json.dumps(r.body)[:120]}")


def _call_failed(result: CallResult) -> bool:
    body = result.body if isinstance(result.body, dict) else {}
    return result.status != 200 or body.get("success") is False


def _fanout_into(
    box: dict[str, Any], rl: RL, path: str, body: Any, selector: str, timeout: float
) -> None:
    """Run one fan-out on a thread, parking its result or its error in `box`."""
    try:
        box["result"] = rl.fanout(
            path, body, selector=selector, timeout=timeout, allow_partial=True
        )
    except Exception as e:  # noqa: BLE001 - reported by the thread that joins
        box["error"] = e


def _await_fanout(box: dict[str, Any], thread: threading.Thread, label: str) -> FanoutResult:
    if thread.is_alive():
        raise TimeoutError(f"{label} did not return within --timeout")
    if "error" in box:
        raise box["error"]
    res: FanoutResult = box["result"]
    _print_fanout(label, res)
    if res.failed:
        detail = ", ".join(f"{f.worker_id} ({f.error}: {f.message})" for f in res.failed)
        raise RuntimeError(f"{label} failed on {detail}")
    return res


def _init_groups(
    rl: RL,
    targets: list[Worker],
    offsets: list[int],
    world_size: int,
    master_address: str,
    master_port: int,
    group_name: str,
    timeout: float,
) -> Any:
    """Join every engine and the trainer into one weight-update group."""
    results: dict[str, CallResult] = {}
    errors: dict[str, BaseException] = {}

    def _call(worker: Worker, rank_offset: int) -> None:
        try:
            results[worker.id] = rl.call(
                worker.id,
                "init_weights_update_group",
                {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                },
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001 - reported once the threads join
            errors[worker.id] = e

    threads = []
    for worker, rank_offset in zip(targets, offsets, strict=True):
        thread = threading.Thread(target=_call, args=(worker, rank_offset), daemon=True)
        thread.start()
        threads.append(thread)
    try:
        pg = _join_trainer_group(master_address, master_port, world_size, group_name, timeout)
    finally:
        for thread in threads:
            thread.join(timeout)

    problems = [f"{wid}: {err}" for wid, err in sorted(errors.items())]
    problems += [
        f"{wid}: HTTP {r.status} {json.dumps(r.body)[:200]}"
        for wid, r in sorted(results.items())
        if _call_failed(r)
    ]
    if problems:
        raise RuntimeError("init_weights_update_group failed: " + "; ".join(problems))
    _print_calls("init_weights_update_group", results)
    return pg


def _stream_weights(
    rl: RL,
    pg: Any,
    params: list[tuple[str, torch.Tensor]],
    *,
    selector: str,
    group_name: str,
    weight_version: str,
    chunk: int,
    timeout: float,
) -> None:
    """Broadcast every parameter, --chunk at a time, into the engines."""
    total = len(params)
    for start in range(0, total, chunk):
        batch = params[start : start + chunk]
        stop = start + len(batch)
        last = stop >= total
        body: dict[str, Any] = {
            "names": [name for name, _ in batch],
            "dtypes": [str(t.dtype).removeprefix("torch.") for _, t in batch],
            "shapes": [list(t.shape) for _, t in batch],
            "group_name": group_name,
            "flush_cache": last,
        }
        # Only the last batch stamps the version: a refit is not complete, and
        # must not be advertised as complete, until every weight has landed.
        if last:
            body["weight_version"] = weight_version
        box: dict[str, Any] = {}
        thread = threading.Thread(
            target=_fanout_into,
            args=(box, rl, "update_weights_from_distributed", body, selector, timeout),
            daemon=True,
        )
        thread.start()
        for _, tensor in batch:
            dist.broadcast(tensor.contiguous(), src=0, group=pg)
        thread.join(timeout)
        _await_fanout(box, thread, f"update_weights_from_distributed[{start}:{stop}/{total}]")


def _teardown(rl: RL, pg: Any, selector: str, group_name: str, timeout: float) -> list[str]:
    """Drop the group on the engines and on the trainer, together."""
    box: dict[str, Any] = {}
    thread = threading.Thread(
        target=_fanout_into,
        args=(
            box,
            rl,
            "destroy_weights_update_group",
            {"group_name": group_name},
            selector,
            timeout,
        ),
        daemon=True,
    )
    thread.start()
    problems: list[str] = []
    try:
        if pg is not None:
            dist.destroy_process_group(pg)
    except Exception as e:  # noqa: BLE001 - a teardown problem must not mask the refit's
        problems.append(f"trainer-side destroy failed: {e}")
    thread.join(timeout)
    try:
        _await_fanout(box, thread, "destroy_weights_update_group")
    except Exception as e:  # noqa: BLE001 - same
        problems.append(str(e))
    return problems


def _generate_weight_version(smg: str, model: str, api_key: str | None, timeout: float) -> str:
    body: dict[str, Any] = {"text": "1+1=", "sampling_params": {"max_new_tokens": 4}}
    if model:
        body["model"] = model
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{smg}/generate", data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    first = out[0] if isinstance(out, list) else out
    return str(first.get("meta_info", {}).get("weight_version"))


def _refit(rl: RL, args: argparse.Namespace) -> str:
    """Pause, refit over NCCL, resume. Returns the model name to generate with."""
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"loading {args.model_path} onto {device} ...")
    model = _load_policy(args.model_path, device)
    params = [(name, p.detach()) for name, p in model.named_parameters()]
    print(f"{len(params)} parameter(s) to broadcast, {args.chunk} per call")

    by_id = {w.id: w for w in rl.workers()}
    targets: list[Worker] = []
    with paused(rl, args.selector) as pause:
        _print_fanout("pause_generation", pause)
        missing = sorted(set(pause.results) - set(by_id))
        if missing:
            raise RuntimeError(f"workers vanished between discovery and pause: {missing}")
        targets = [by_id[wid] for wid in sorted(pause.results)]
        # A TokenSpeed engine launched without an explicit parallelism flag
        # reports no tp_size, and a wrong tp_size lays the ranks out wrong and
        # deadlocks the group. Assume 1 unless --tp-size says otherwise.
        fallback = args.tp_size or 1
        for worker in targets:
            if worker.tp_size is None:
                print(f"  {worker.id}: no tp_size reported, using {fallback}", file=sys.stderr)
        world_size, offsets = rank_layout([w.tp_size or fallback for w in targets])
        layout = ", ".join(f"{w.id}@{o}+{w.tp_size or fallback}" for w, o in zip(targets, offsets))
        print(f"world_size={world_size} (trainer at rank 0), {layout}")

        pg = None
        try:
            pg = _init_groups(
                rl,
                targets,
                offsets,
                world_size,
                args.master_address,
                args.master_port,
                args.group_name,
                args.timeout,
            )
            _stream_weights(
                rl,
                pg,
                params,
                selector=args.selector,
                group_name=args.group_name,
                weight_version=args.weight_version,
                chunk=args.chunk,
                timeout=args.timeout,
            )
            torch.cuda.synchronize(device)
        finally:
            for problem in _teardown(rl, pg, args.selector, args.group_name, args.timeout):
                print(f"WARN {problem}", file=sys.stderr)
    print("continue_generation ... resumed")
    return args.model or (targets[0].model_id if targets else "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smg", required=True, help="SMG base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument(
        "--model-path", required=True, help="HF checkpoint the trainer loads and broadcasts"
    )
    ap.add_argument("--weight-version", required=True, help="version string to stamp, e.g. 42")
    ap.add_argument("--selector", default="engine=tokenspeed", help="which workers to refit")
    ap.add_argument("--device", default="cuda:0", help="trainer device holding the weights")
    ap.add_argument("--master-address", default="127.0.0.1", help="host the engines dial back")
    ap.add_argument("--master-port", type=int, default=None, help="default: a free port")
    ap.add_argument("--group-name", default="smg_refit", help="weight-update group name")
    ap.add_argument("--chunk", type=int, default=64, help="parameters per broadcast call")
    ap.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="tp_size to assume for workers that report none (default 1)",
    )
    ap.add_argument("--model", default=None, help="model name for the trailing /generate")
    ap.add_argument("--api-key", default=None, help="SMG control-plane key, if configured")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    if args.chunk < 1:
        ap.error("--chunk must be >= 1")
    if args.tp_size is not None and args.tp_size < 1:
        ap.error("--tp-size must be >= 1")
    if args.master_port is None:
        args.master_port = _free_port()
    if not torch.cuda.is_available():
        print(
            "this example needs a CUDA build of torch: the engines receive over NCCL",
            file=sys.stderr,
        )
        return 1

    rl = RL(args.smg, api_key=args.api_key, timeout=args.timeout)
    try:
        model = _refit(rl, args)
    except Exception:
        traceback.print_exc()
        print("refit failed", file=sys.stderr)
        return 1

    got = _generate_weight_version(args.smg, model, args.api_key, args.timeout)
    print(f"/generate reported weight_version={got}")
    if got != str(args.weight_version):
        print("MISMATCH: engine did not report the new version", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
