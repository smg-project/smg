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
stand in for a training loop's live policy weights. The engines dial the
trainer back on --master-address, which must be this host as the engines see
it; the default only works when they share a host with the trainer.

Both halves of a step block on each other: `init_weights_update_group` does not
return until the group forms, and `update_weights_from_distributed` does not
return until every weight in the batch has been received. So each is issued
from a thread while the trainer's own rendezvous and broadcasts run on the main
thread.

An HTTP-level failure is cleaned up: the group is destroyed and the engines
resumed on the way out. An engine that stops participating in the collective is
not. The trainer's main thread is inside NCCL by then, so torch's watchdog
takes the process down without raising into Python, no cleanup runs, and the
fleet stays paused. Un-pause it by hand:

    python -c 'from smg.rl import RL; RL("http://127.0.0.1:30000").fanout(
        "continue_generation", {}, selector="engine=tokenspeed")'
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import socket
import sys
import threading
import time
import traceback
import urllib.parse
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
    # Racy by construction: the port is free when we look and handed to the
    # engines afterwards. Pass --master-port to pin it on a busy host.
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    name = host.strip("[]").lower()
    return name in {"localhost", "::1", "0.0.0.0"} or name.startswith("127.")


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
    except Exception as e:  # reported by the thread that joins
        box["error"] = e


def _fanout_problems(res: FanoutResult) -> dict[str, str]:
    """Worker id -> why its leg of the fan-out failed, empty when all passed.

    A non-2xx lands in both `failed` and `results`; `failed` carries the better
    message, so it wins. The `results` pass also catches a 200 that carries
    `success: false`, which SMG counts as a success.
    """
    bad = {
        wid: f"HTTP {r.status} {json.dumps(r.body)[:120]}"
        for wid, r in res.results.items()
        if _call_failed(r)
    }
    bad.update({f.worker_id: f"{f.error}: {f.message}" for f in res.failed})
    return bad


def _await_fanout(
    box: dict[str, Any],
    thread: threading.Thread,
    label: str,
    expected: set[str] | None = None,
) -> FanoutResult:
    if thread.is_alive():
        raise TimeoutError(f"{label} did not return within --timeout")
    if "error" in box:
        raise box["error"]
    res: FanoutResult = box["result"]
    _print_fanout(label, res)

    # Every fan-out re-resolves --selector, so the fleet can drift underneath a
    # rank layout that was pinned at pause time. A worker that joined has no
    # rank; one that left leaves the collective short.
    if expected is not None:
        reached = set(res.results) | {f.worker_id for f in res.failed}
        if reached != expected:
            raise RuntimeError(
                f"{label} reached a different worker set than the rank layout covers: "
                f"joined {sorted(reached - expected)}, left {sorted(expected - reached)}"
            )

    bad = _fanout_problems(res)
    if bad:
        detail = ", ".join(f"{wid} ({why})" for wid, why in sorted(bad.items()))
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
        except Exception as e:  # reported once the threads join
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
    expected: set[str],
    group_name: str,
    weight_version: str,
    chunk: int,
    timeout: float,
    progress: dict[str, int],
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
        # Weights have left the trainer: from here a failure leaves the engines
        # holding a mix of old and new tensors.
        progress["chunks"] += 1
        thread.join(timeout)
        _await_fanout(
            box, thread, f"update_weights_from_distributed[{start}:{stop}/{total}]", expected
        )


def _teardown(
    rl: RL,
    pg: Any,
    selector: str,
    expected: set[str],
    group_name: str,
    timeout: float,
    *,
    flush_cache: bool,
) -> list[str]:
    """Drop the group on the engines and on the trainer, together.

    `flush_cache` is for the failure path: a refit that stopped part-way leaves
    a KV cache computed under weights that are no longer loaded, and only the
    last chunk would have flushed it. Best effort -- a flush that fails is
    reported like any other teardown problem, never raised.
    """
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
    except Exception as e:  # a teardown problem must not mask the refit's
        problems.append(f"trainer-side destroy failed: {e}")
    thread.join(timeout)
    try:
        _await_fanout(box, thread, "destroy_weights_update_group", expected)
    except Exception as e:  # same
        problems.append(str(e))

    if flush_cache:
        try:
            res = rl.fanout(
                "flush_cache", {}, selector=selector, timeout=timeout, allow_partial=True
            )
            _print_fanout("flush_cache", res)
            bad = _fanout_problems(res)
            if bad:
                detail = ", ".join(f"{wid} ({why})" for wid, why in sorted(bad.items()))
                problems.append(f"flush_cache failed on {detail}")
        except Exception as e:  # same
            problems.append(f"flush_cache failed: {e}")
    return problems


def _generate_weight_version(
    smg: str, model: str, api_key: str | None, timeout: float, attempts: int = 5
) -> str:
    """The version an engine reports, retried: this fires right after resume.

    Every attempt and every pause between them draws on one `--timeout` budget,
    so retrying buys resilience against a hiccup right after `continue_generation`
    without multiplying how long the script can sit here.
    """
    body: dict[str, Any] = {"text": "1+1=", "sampling_params": {"max_new_tokens": 4}}
    if model:
        body["model"] = model
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            req = urllib.request.Request(
                f"{smg}/generate", data=json.dumps(body).encode(), headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=remaining) as resp:
                out = json.loads(resp.read())
            first = out[0] if isinstance(out, list) else out
            meta = first.get("meta_info") if isinstance(first, dict) else None
            return str((meta or {}).get("weight_version"))
        except Exception as e:
            last = e
        if attempt + 1 >= attempts:
            break
        nap = min(2.0, deadline - time.monotonic())
        if nap <= 0:
            break
        print(f"  /generate attempt {attempt + 1} failed ({last}), retrying", file=sys.stderr)
        time.sleep(nap)
    raise RuntimeError(f"/generate did not answer within {timeout:g}s: {last}") from last


def _refit(rl: RL, args: argparse.Namespace) -> tuple[str, list[str]]:
    """Pause, refit over NCCL, resume.

    Returns the model name to generate with and the teardown problems, which
    the caller turns into a non-zero exit even when the refit itself worked.
    """
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"loading {args.model_path} onto {device} ...")
    model = _load_policy(args.model_path, device)
    params = [(name, p.detach()) for name, p in model.named_parameters()]
    print(f"{len(params)} parameter(s) to broadcast, {args.chunk} per call")

    by_id = {w.id: w for w in rl.workers()}
    targets: list[Worker] = []
    problems: list[str] = []
    with paused(rl, args.selector) as pause:
        _print_fanout("pause_generation", pause)
        unknown = sorted(set(pause.results) - set(by_id))
        if unknown:
            raise RuntimeError(
                f"workers registered between discovery and pause, so their topology is "
                f"unknown: {unknown}. Re-run."
            )
        targets = [by_id[wid] for wid in sorted(pause.results)]
        expected = {w.id for w in targets}
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
        progress = {"chunks": 0}
        complete = False
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
                expected=expected,
                group_name=args.group_name,
                weight_version=args.weight_version,
                chunk=args.chunk,
                timeout=args.timeout,
                progress=progress,
            )
            torch.cuda.synchronize(device)
            complete = True
        finally:
            partial = progress["chunks"] > 0 and not complete
            problems = _teardown(
                rl,
                pg,
                args.selector,
                expected,
                args.group_name,
                args.timeout,
                flush_cache=partial,
            )
            if partial:
                print(
                    f"WARN the refit stopped after {progress['chunks']} chunk(s) had already "
                    f"been broadcast: {', '.join(sorted(expected))} are serving a model that "
                    "is part old weights and part new. Refit them again or restart them "
                    "before they take traffic.",
                    file=sys.stderr,
                )
            for problem in problems:
                print(f"WARN {problem}", file=sys.stderr)
    print("continue_generation ... resumed")
    return args.model or (targets[0].model_id if targets else ""), problems


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
    ap.add_argument(
        "--master-address",
        default="127.0.0.1",
        help="this host's address as the engines see it; they dial it to join the group",
    )
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
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds for each control call, for the group rendezvous, and for the "
        "whole trailing /generate check including its retries",
    )
    args = ap.parse_args()
    if args.chunk < 1:
        ap.error("--chunk must be >= 1")
    if args.tp_size is not None and args.tp_size < 1:
        ap.error("--tp-size must be >= 1")
    smg_host = urllib.parse.urlparse(args.smg).hostname
    if not _is_loopback(smg_host) and _is_loopback(args.master_address):
        ap.error(
            f"--smg points at {smg_host}, so the engines are very unlikely to share this "
            f"host, but --master-address is {args.master_address}: every engine would dial "
            "its own loopback, none would reach the trainer's store, and the rendezvous "
            "would stall for --timeout. Pass --master-address <this host as the engines "
            "see it>."
        )
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
        model, problems = _refit(rl, args)
    except Exception:
        traceback.print_exc()
        print("refit failed", file=sys.stderr)
        return 1

    try:
        got = _generate_weight_version(args.smg, model, args.api_key, args.timeout)
    except Exception:
        traceback.print_exc()
        print("could not verify the refit through /generate", file=sys.stderr)
        return 1
    print(f"/generate reported weight_version={got}")
    if got != str(args.weight_version):
        print("MISMATCH: engine did not report the new version", file=sys.stderr)
        return 1
    if problems:
        print(
            "the refit landed but teardown did not finish cleanly; see the WARN lines above",
            file=sys.stderr,
        )
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
