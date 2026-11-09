#!/usr/bin/env python3
"""Run the official BFCL benchmark against two "arms" and diff the scores.

This is **Track B** of the parser-verification proposal: a live, end-to-end A/B
that holds the model + engine + checkpoint + sampling fixed and varies only the
*frontend* (chat template + tokenization + tool/reasoning parsing):

  * baseline = pure vLLM  (vLLM's own parsers)
  * candidate = SMG -> vLLM gRPC  (SMG's Rust parsers)

Both expose an identical OpenAI ``/v1`` endpoint. We point the **official**
``bfcl`` CLI (FC mode, so the server's parsed ``tool_calls`` are what gets
scored) at each arm via ``LOCAL_SERVER_ENDPOINT`` / ``LOCAL_SERVER_PORT`` +
``--skip-server-setup``, run ``generate`` then ``evaluate`` into a per-arm
``BFCL_PROJECT_ROOT``, parse the per-category accuracy, and emit a comparison
table. Any score delta is attributable to the frontend — that is the number
that persuades an engine to adopt SMG's parsing layer.

The arms must already be serving (see ``launch_arm.sh``); this driver does not
launch them. Example::

    python run_ab.py \\
        --baseline   vllm=http://127.0.0.1:31199 \\
        --candidate  smg=http://127.0.0.1:31200 \\
        --bfcl-model Qwen/Qwen3-4B-Instruct-2507-FC \\
        --categories simple_python,multiple,parallel,irrelevance \\
        --bfcl /home/keyang/bfcl-env/bin/bfcl \\
        --project-root /home/keyang/bfcl_ab \\
        --out /tmp/bfcl_ab.md --json-out /tmp/bfcl_ab.json

Exit codes: ``0`` clean, ``1`` score regression beyond ``--tolerance``, ``2`` an
arm or the driver did not complete, including a failure to write its reports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INCOMPLETE = 2


RESULT_FILE_LIMIT = 64 * 1024 * 1024
RESULT_TOTAL_LIMIT = 128 * 1024 * 1024
RESULT_CLASSES = (
    "non_error_result",
    "http_503_no_available_workers",
    "other_inference_error",
)


def unknown_result_diagnostics(reason: str) -> dict:
    return {"status": "unknown", "reason": reason, "counts": None, "categories": {}}


def result_paths(project_root: Path, model: str, categories: list[str]) -> tuple[dict, str | None]:
    """Find only this BFCL model's exact category files, without following links."""
    paths = {category: [] for category in categories}
    model_dir = model.replace("/", "_")
    if model_dir in ("", ".", "..") or "\\" in model_dir:
        return paths, "unsupported_model_directory"
    result_root = project_root / "result" / model_dir
    try:
        if any(path.is_symlink() for path in (project_root, project_root / "result", result_root)):
            return paths, "linked_result_directory"
        patterns = {
            category: re.compile(rf"(?:BFCL_v\d+_)?{re.escape(category)}_result\.json")
            for category in categories
        }
        for index, path in enumerate(result_root.rglob("*")):
            if index >= 4096:
                return paths, "result_directory_limit"
            if path.is_symlink():
                return paths, "linked_result_payload"
            if path.is_file():
                for category, pattern in patterns.items():
                    if pattern.fullmatch(path.name):
                        paths[category].append(path)
    except OSError:
        return paths, "unreadable_result_directory"
    return paths, None


def classify_result(value) -> str | None:
    """Classify explicit status prefixes, with finite depth/work for result data."""
    pending = [(value, 0)]
    result = "non_error_result"
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if depth > 64 or visited > 10000:
            return None
        if isinstance(current, list):
            if len(current) + len(pending) > 10000:
                return None
            pending.extend((item, depth + 1) for item in current)
            continue
        if isinstance(current, dict) or current is None:
            # BFCL FC uses function->arguments dictionaries and nullable model
            # replies. Arguments are opaque model output, not inference errors.
            continue
        if not isinstance(current, str):
            return None
        if not current.startswith("Error during inference:"):
            continue
        status = re.match(
            r"Error during inference:\s*(?:Error code:\s*|HTTP\s+)([0-9]{3})(?![0-9])", current
        )
        if status and status[1] == "503" and "no_available_workers" in current:
            result = "http_503_no_available_workers"
        elif result == "non_error_result":
            result = "other_inference_error"
    return result


def collect_result_diagnostics(
    project_root: Path,
    model: str,
    categories: list[str],
    before: dict,
    before_error: str | None,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Read new generation outputs only. Missing/ambiguous evidence stays unknown.

    A pre-existing category file is never borrowed, even if generation rewrites
    it. The digest and ordered ID/line inventory describe observed results, not
    proof that a model served those cases or that the official scorer is correct.
    """
    if not categories or len(categories) != len(set(categories)):
        return unknown_result_diagnostics("empty_or_duplicate_categories")
    if before_error:
        return unknown_result_diagnostics(before_error)
    paths, error = result_paths(project_root, model, categories)
    if error:
        return unknown_result_diagnostics(error)
    diagnostic = {"status": "observed", "model": model, "counts": None, "categories": {}}
    remaining = RESULT_TOTAL_LIMIT
    for category in categories:
        row = {"status": "unknown", "counts": None}
        diagnostic["categories"][category] = row
        if before.get(category):
            row["reason"] = "preexisting_result_files"
            continue
        if len(paths[category]) != 1:
            row["reason"] = "missing_or_ambiguous_result_files"
            continue
        path = paths[category][0]
        try:
            if not hasattr(os, "O_NOFOLLOW"):
                raise ValueError("nofollow_unavailable")
            with os.fdopen(
                os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
            ) as source:
                initial = os.fstat(source.fileno())
                if not stat.S_ISREG(initial.st_mode) or initial.st_size > min(
                    remaining, RESULT_FILE_LIMIT
                ):
                    raise ValueError("result_size_or_type_limit")
                body = source.read(min(remaining, RESULT_FILE_LIMIT) + 1)
                final = os.fstat(source.fileno())
            current = path.stat(follow_symlinks=False)
            identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(
                getattr(initial, key) != getattr(value, key)
                for key in identity_fields
                for value in (final, current)
            ):
                raise ValueError("result_changed_during_read")
            if len(body) != initial.st_size:
                raise ValueError("result_size_changed")
            remaining -= len(body)
            row["source"] = {
                "path": str(path.relative_to(project_root)),
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            counts = dict.fromkeys(RESULT_CLASSES, 0)
            inventory, seen, duplicates, invalid = [], set(), [], False
            for line_number, line in enumerate(body.splitlines(), 1):
                if not line.strip():
                    continue
                if len(line) > 1024 * 1024 or len(inventory) >= 10000:
                    raise ValueError("result_record_limit")
                try:
                    record = json.loads(line)
                    case_id = record.get("id") if isinstance(record, dict) else None
                    kind = (
                        classify_result(record["result"])
                        if isinstance(record, dict) and "result" in record
                        else None
                    )
                    if (
                        not isinstance(case_id, str)
                        or len(case_id) > 256
                        or not case_id.startswith(category + "_")
                    ):
                        raise ValueError("invalid_case_id")
                    inventory.append(
                        {"id": case_id, "line": line_number, "classification": kind or "unknown"}
                    )
                    if case_id in seen:
                        duplicates.append(case_id)
                    seen.add(case_id)
                    if kind is None:
                        invalid = True
                    else:
                        counts[kind] += 1
                except (ValueError, UnicodeError, RecursionError):
                    invalid = True
            row.update(id_inventory=inventory, unique_ids=len(seen), duplicate_ids=duplicates)
            if not inventory or invalid or duplicates:
                row["reason"] = "empty_invalid_or_duplicate_records"
            elif expected_counts is not None and len(inventory) != expected_counts.get(category):
                row["scored_count"] = expected_counts.get(category)
                row["reason"] = "result_and_score_counts_differ"
            else:
                row.update(status="observed", counts=counts)
        except (OSError, ValueError) as exc:
            row["reason"] = str(exc) if isinstance(exc, ValueError) else "unreadable_result_file"
    if any(row["status"] != "observed" for row in diagnostic["categories"].values()):
        diagnostic.update(status="unknown", reason="one_or_more_categories_unknown")
    else:
        diagnostic["counts"] = {
            kind: sum(row["counts"][kind] for row in diagnostic["categories"].values())
            for kind in RESULT_CLASSES
        }
    return diagnostic


@dataclass
class Arm:
    """One side of the A/B: a name and an already-serving OpenAI base URL."""

    name: str
    base_url: str
    host: str = ""
    port: str = ""
    project_root: Path = field(default_factory=Path)
    scores: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)  # per-category test-case count
    timed_out: list[str] = field(default_factory=list)  # stages killed by --run-timeout
    result_diagnostics: dict = field(
        default_factory=lambda: unknown_result_diagnostics("not_collected")
    )


# These persisted markers are never waivable by --allow-incomplete. They survive
# save/load and sequential --diff without inventing a quality-eligible result.
LIFECYCLE_FAILURES = frozenset(("absolute_deadline", "cleanup_failed"))


class WorkDeadlineExpired(RuntimeError):
    pass


class OwnedCleanupFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class LifecycleDeadline:
    """Controller-supplied UTC pair, translated once to a shared monotonic anchor.

    This only bounds this driver's work. Admission before preparation, native
    server custody and physical release remain the caller/controller's job.
    """

    work: float
    cleanup: float

    @classmethod
    def from_epochs(cls, work: float | None, cleanup: float | None):
        if work is None and cleanup is None:
            return None
        if (
            type(work) not in (int, float)
            or type(cleanup) not in (int, float)
            or not math.isfinite(work)
            or not math.isfinite(cleanup)
            or not 0 < work < cleanup
        ):
            raise ValueError(
                "work/cleanup deadlines require finite positive UTC epochs, work < cleanup"
            )
        monotonic, wall = time.monotonic(), time.time()
        result = cls(monotonic + (work - wall), monotonic + (cleanup - wall))
        if not math.isfinite(result.work) or not math.isfinite(result.cleanup):
            raise ValueError("deadline conversion overflow")
        return result

    def remaining(self) -> float:
        return self.work - time.monotonic()


def lifecycle_failed(arm: Arm) -> bool:
    return bool(LIFECYCLE_FAILURES.intersection(arm.timed_out))


def check_work(arm: Arm, deadline: LifecycleDeadline | None) -> bool:
    if deadline is not None and deadline.remaining() <= 0:
        if "absolute_deadline" not in arm.timed_out:
            arm.timed_out.append("absolute_deadline")
        arm.result_diagnostics = unknown_result_diagnostics("absolute_deadline")
        return False
    return not lifecycle_failed(arm)


def parse_arm(spec: str, project_root: Path) -> Arm:
    """Parse a ``name=url`` arm spec and derive host/port + a per-arm BFCL root."""
    if "=" not in spec:
        raise ValueError(f"arm spec must be name=url, got: {spec!r}")
    name, url = spec.split("=", 1)
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 80)
    # urlparse strips the brackets of an IPv6 literal; URLs want them back.
    if ":" in host:
        host = f"[{host}]"
    return Arm(
        name=name,
        base_url=url,
        host=host,
        port=port,
        project_root=project_root / name,
    )


def run_bfcl(
    arm: Arm,
    *,
    bfcl: str,
    model: str,
    categories: list[str],
    num_threads: int,
    temperature: float,
    skip_generate: bool,
    run_timeout: int = 0,
    deadline: LifecycleDeadline | None = None,
) -> None:
    """Run ``bfcl generate`` + ``bfcl evaluate`` for one arm, in its own root.

    ``run_timeout`` (0 = off) mirrors tau2's ``--run-timeout``. No BFCL analogue of
    ``--request-timeout`` exists: the FC handler's ``openai.OpenAI`` client reads
    ``api_key``/``base_url`` from the environment but not ``timeout``, and ``bfcl
    generate`` exposes no passthrough, so 600s x 2 retries is the per-call ceiling.
    """
    if not check_work(arm, deadline):
        return

    def run_stage(cmd, env, label):
        try:
            # Leave the legacy call signature unchanged for existing callers.
            kwargs = {"deadline": deadline} if deadline is not None else {}
            return _run(cmd, env, label, timeout=run_timeout, **kwargs)
        except WorkDeadlineExpired:
            if "absolute_deadline" not in arm.timed_out:
                arm.timed_out.append("absolute_deadline")
        except OwnedCleanupFailed:
            arm.timed_out.append("cleanup_failed")
        arm.result_diagnostics = unknown_result_diagnostics("lifecycle_incomplete")
        return None

    arm.project_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(arm.project_root)
    # OpenAICompletionsHandler (the FC handler we register the models with) talks to
    # this arm's OpenAI-compatible endpoint via OPENAI_BASE_URL. LOCAL_SERVER_* is
    # kept for any model bfcl ships with a local OSS handler instead.
    env["OPENAI_BASE_URL"] = f"http://{arm.host}:{arm.port}/v1"
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env["LOCAL_SERVER_ENDPOINT"] = arm.host
    env["LOCAL_SERVER_PORT"] = arm.port
    # NOTE: we do NOT force HF_HUB_OFFLINE. With the model cached, bfcl runs fine
    # online (~7 req/s in practice); forcing offline would only hide a genuinely
    # missing cache. Export HF_HUB_OFFLINE=1 yourself for air-gapped runs.

    cats = ",".join(categories)
    before, before_error = (
        result_paths(arm.project_root, model, categories) if not skip_generate else ({}, None)
    )
    if not skip_generate:
        generated = run_stage(
            [
                bfcl,
                "generate",
                "--model",
                model,
                "--test-category",
                cats,
                "--skip-server-setup",
                "--backend",
                "vllm",
                "--temperature",
                str(temperature),
                "--num-threads",
                str(num_threads),
                "--allow-overwrite",
            ],
            env,
            f"[{arm.name}] generate",
        )
        if generated is None:
            return
        if not generated:
            arm.timed_out.append("generate")
    if not check_work(arm, deadline):
        return
    # Evaluate even after a killed generate: whatever completed is already on disk.
    # Each evaluate scores into a new directory under score/ (BFCL resolves
    # --score-dir against BFCL_PROJECT_ROOT), and only that directory is parsed:
    # a reused project root keeps score files from earlier runs, such as another
    # model, an older BFCL version, or a category this evaluate did not rescore.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    score_dir = Path("score") / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
    evaluated = run_stage(
        [
            bfcl,
            "evaluate",
            "--model",
            model,
            "--test-category",
            cats,
            "--score-dir",
            str(score_dir),
        ],
        env,
        f"[{arm.name}] evaluate",
    )
    if evaluated is None:
        return
    if not evaluated:
        arm.timed_out.append("evaluate")
    if not check_work(arm, deadline):
        return
    arm.scores, arm.counts = parse_scores(arm.project_root / score_dir, model, categories)
    if skip_generate or "generate" in arm.timed_out:
        arm.result_diagnostics = unknown_result_diagnostics(
            "generation_skipped" if skip_generate else "generation_timed_out"
        )
    else:
        arm.result_diagnostics = collect_result_diagnostics(
            arm.project_root, model, categories, before, before_error, arm.counts
        )

    check_work(arm, deadline)


def _run(
    cmd: list[str],
    env: dict[str, str],
    label: str,
    timeout: int = 0,
    deadline: LifecycleDeadline | None = None,
) -> bool:
    """Run one bfcl subcommand; return False if it hit ``timeout`` (0 = no cap)."""
    # Anchor the stage cap before process creation too: neither budget may be
    # refreshed by time spent inside Popen (which needs external supervision).
    stage_deadline = time.monotonic() + timeout if deadline is not None and timeout else None
    remaining = deadline.remaining() if deadline is not None else None
    if remaining is not None and remaining <= 0:
        raise WorkDeadlineExpired(label)
    wait_timeout = (
        min(timeout, remaining)
        if timeout and remaining is not None
        else (remaining if remaining is not None else (timeout or None))
    )
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    if deadline is not None:
        remaining = deadline.remaining()
        if remaining <= 0:
            raise WorkDeadlineExpired(label)
        stage_remaining = (
            stage_deadline - time.monotonic() if stage_deadline is not None else remaining
        )
        if stage_remaining <= 0:
            return False
        wait_timeout = min(stage_remaining, remaining)
    # Own session so a timeout can SIGKILL the whole group: bfcl generate fans out
    # over a thread pool and can sit blocked on a hung HTTP read.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    if deadline is not None:
        remaining = deadline.remaining()
        stage_remaining = (
            stage_deadline - time.monotonic() if stage_deadline is not None else remaining
        )
        if remaining <= 0 or stage_remaining <= 0:
            _killpg(proc, deadline.cleanup)
            if deadline.remaining() <= 0:
                raise WorkDeadlineExpired(label)
            return False
        wait_timeout = min(stage_remaining, remaining)
    try:
        returncode = proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        _killpg(proc, deadline.cleanup if deadline is not None else None)
        if deadline is not None and deadline.remaining() <= 0:
            raise WorkDeadlineExpired(label)
        print(f"WARNING: {label} timed out after {timeout}s; killed", file=sys.stderr)
        return False
    if deadline is not None and deadline.remaining() <= 0:
        raise WorkDeadlineExpired(label)
    if returncode != 0:
        print(f"WARNING: {label} exited {returncode}", file=sys.stderr)
    return True


def _killpg(proc: subprocess.Popen, cleanup_deadline: float | None = None) -> None:
    # Popen created this exact child's session/group; do not resolve a pid from
    # external metadata. A finite fallback also bounds legacy timeout cleanup.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise OwnedCleanupFailed("cannot signal owned BFCL process group") from exc
    remaining = 5.0 if cleanup_deadline is None else max(0.0, cleanup_deadline - time.monotonic())
    try:
        proc.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise OwnedCleanupFailed("owned BFCL process did not reap within cleanup budget") from exc


def parse_scores(
    score_root: Path, model: str, categories: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Extract per-category accuracy and test-case count from one evaluate's scores.

    ``score_root`` is the ``--score-dir`` of a single ``bfcl evaluate``. BFCL writes
    ``<score_root>/<sanitized-model>/<category>_score.json`` whose FIRST line is a
    summary dict ``{"accuracy", "correct_count", "total_count"}``. We glob for the
    model dir (sanitization differs across versions) and read each category's
    summary. ``total_count`` is what weights the weighted overall.
    """
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    if not score_root.is_dir():
        print(f"WARNING: no score dir at {score_root}", file=sys.stderr)
        return scores, counts
    for cat in categories:
        summary = _find_category_summary(score_root, cat)
        if summary is not None:
            scores[cat] = summary[0]
            counts[cat] = summary[1]
    return scores, counts


def _find_category_summary(score_root: Path, category: str) -> tuple[float, int] | None:
    # BFCL nests scores as <model>/<section>/BFCL_v4_<category>_score.json, and
    # the BFCL_v4_ prefix varies by version. The category must be the whole name
    # after that prefix: "*multiple_score.json" alone also matches
    # parallel_multiple, live_multiple and live_parallel_multiple, and which of
    # them rglob yields first depends on the filesystem.
    exact = re.compile(rf"(?:BFCL_v\d+_)?{re.escape(category)}_score\.json")
    for path in sorted(score_root.rglob(f"*{category}_score.json")):
        if not exact.fullmatch(path.name):
            continue
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            summary = json.loads(first)
        except (OSError, ValueError, IndexError):
            continue
        acc = next((summary[k] for k in ("accuracy", "acc", "score") if k in summary), None)
        if acc is not None:
            return float(acc), int(summary.get("total_count", 0))
    return None


def incompleteness(arm: Arm, categories: list[str]) -> str | None:
    """Why this arm can't be compared (timed out / unscored categories), else None."""
    missing = [c for c in categories if c not in arm.scores]
    if not missing and not arm.timed_out:
        return None
    reasons = []
    if missing:
        reasons.append(f"no score for {len(missing)}/{len(categories)} categories")
    if arm.timed_out:
        reasons.append(f"timed out during: {', '.join(arm.timed_out)}")
    if missing:
        shown = ", ".join(missing[:8]) + (f", +{len(missing) - 8} more" if len(missing) > 8 else "")
        reasons.append(f"unscored: {shown}")
    return "; ".join(reasons)


def build_report(baseline: Arm, candidate: Arm, categories: list[str]) -> tuple[str, dict]:
    """Build a markdown comparison table + a JSON blob; candidate minus baseline."""
    rows: list[dict] = []
    for cat in categories:
        b = baseline.scores.get(cat)
        c = candidate.scores.get(cat)
        delta = (c - b) if (b is not None and c is not None) else None
        # Per-category count is the same on both arms (same test set); fall back
        # across arms in case one didn't score the category.
        n = baseline.counts.get(cat) or candidate.counts.get(cat)
        rows.append({"category": cat, "baseline": b, "candidate": c, "delta": delta, "count": n})

    def unweighted(key: str) -> float | None:
        # macro average: mean of per-category accuracies (BFCL calculate_unweighted_accuracy)
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    def weighted(key: str) -> float | None:
        # micro average: weighted by each category's test-case count
        # (BFCL calculate_weighted_accuracy = total correct / total cases)
        num = sum(r[key] * r["count"] for r in rows if r[key] is not None and r["count"])
        den = sum(r["count"] for r in rows if r[key] is not None and r["count"])
        return num / den if den else None

    def delta_of(fn) -> float | None:
        b, c = fn("baseline"), fn("candidate")
        return (c - b) if (b is not None and c is not None) else None

    b_overall, c_overall = unweighted("baseline"), unweighted("candidate")
    overall_delta = delta_of(unweighted)
    b_weighted, c_weighted = weighted("baseline"), weighted("candidate")
    weighted_delta = delta_of(weighted)

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.2f}"

    def fmt_d(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:+.2f}"

    lines = [
        f"# BFCL A/B — {candidate.name} (candidate) vs {baseline.name} (baseline)",
        "",
        f"| category | n | {baseline.name} | {candidate.name} | Δ (cand−base) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        n = "—" if r["count"] is None else str(r["count"])
        lines.append(
            f"| {r['category']} | {n} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | {fmt_d(r['delta'])} |"
        )
    n_total = sum(r["count"] for r in rows if r["count"]) or "—"

    def overall_row(label: str, b: float | None, c: float | None, d: float | None) -> str:
        return f"| **{label}** | {n_total} | **{fmt(b)}** | **{fmt(c)}** | **{fmt_d(d)}** |"

    lines.append(overall_row("overall (unweighted)", b_overall, c_overall, overall_delta))
    lines.append(overall_row("overall (weighted by n)", b_weighted, c_weighted, weighted_delta))
    lines.append("")
    incomplete = {a.name: incompleteness(a, categories) for a in (baseline, candidate)}
    incomplete = {k: v for k, v in incomplete.items() if v}
    for name, reason in incomplete.items():
        lines.append(f"> ⚠️ **Incomplete arm `{name}`** — {reason}.")
    if incomplete:
        lines.append("")
    lines.append(
        "_Scores are % accuracy (official BFCL, FC mode). Same model, engine, "
        "checkpoint and sampling on both arms — the only difference is the "
        "frontend, so Δ is attributable to the tokenization+parsing layer. "
        "Unweighted = mean of category accuracies (macro); weighted = by test-case "
        "count n (micro = total correct / total cases), matching BFCL's "
        "calculate_unweighted/weighted_accuracy._"
    )

    lines.append("")
    lines.append(
        "Infrastructure-result diagnostics (separate from scores and gates; non-error is not served-quality proof):"
    )
    for arm in (baseline, candidate):
        diagnostic = arm.result_diagnostics
        detail = (
            json.dumps(diagnostic.get("counts"), sort_keys=True)
            if diagnostic.get("status") == "observed"
            else "UNKNOWN"
        )
        lines.append(f"- `{arm.name}`: {detail}.")

    payload = {
        "baseline": {
            "name": baseline.name,
            "base_url": baseline.base_url,
            "scores": baseline.scores,
            "counts": baseline.counts,
            "timed_out": baseline.timed_out,
            "result_diagnostics": baseline.result_diagnostics,
        },
        "candidate": {
            "name": candidate.name,
            "base_url": candidate.base_url,
            "scores": candidate.scores,
            "counts": candidate.counts,
            "timed_out": candidate.timed_out,
            "result_diagnostics": candidate.result_diagnostics,
        },
        "incomplete": incomplete,
        "per_category": rows,
        "overall": {"baseline": b_overall, "candidate": c_overall, "delta": overall_delta},
        "overall_weighted": {
            "baseline": b_weighted,
            "candidate": c_weighted,
            "delta": weighted_delta,
        },
    }
    return "\n".join(lines), payload


def save_scores(arm: Arm, path: Path) -> None:
    """Persist one arm's per-category scores so a later --diff can compare them.

    Used by the sequential mode: when a model needs the whole node (TP=8) the two
    arms cannot run at once, so each is scored on its own and diffed afterwards.
    """
    try:
        relative_source = str(arm.project_root.resolve().relative_to(path.parent.resolve()))
    except (OSError, ValueError):
        relative_source = None
    path.write_text(
        json.dumps(
            {
                "name": arm.name,
                "base_url": arm.base_url,
                "scores": arm.scores,
                "counts": arm.counts,
                "timed_out": arm.timed_out,
                "result_diagnostics": arm.result_diagnostics,
                "result_source_root": relative_source,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def saved_result_diagnostics(data: dict, path: Path) -> tuple[dict, Path]:
    """Revalidate saved diagnostics from contained source, otherwise UNKNOWN.

    A score file may be moved or its source cleaned up. Scores remain usable;
    metadata never turns those missing bindings into zero infrastructure errors.
    """
    unknown = unknown_result_diagnostics("saved_diagnostics_not_source_verified")
    try:
        cached = data.get("result_diagnostics")
        relative_root = data.get("result_source_root")
        if not isinstance(cached, dict) or cached.get("status") not in ("observed", "unknown"):
            return unknown, Path()
        if not isinstance(relative_root, str):
            return unknown, Path()
        relative_root = Path(relative_root)
        if relative_root.is_absolute() or ".." in relative_root.parts:
            return unknown, Path()
        root = path.parent / relative_root
        current = root
        while current != path.parent:
            if current.is_symlink():
                return unknown, Path()
            current = current.parent
        if not root.resolve().is_relative_to(path.parent.resolve()):
            return unknown, Path()
        model, rows = cached.get("model"), cached.get("categories")
        scores, counts = data.get("scores"), data.get("counts")
        if not isinstance(model, str) or not model or len(model) > 256:
            return unknown, Path()
        if (
            not isinstance(rows, dict)
            or not rows
            or not isinstance(scores, dict)
            or not isinstance(counts, dict)
        ):
            return unknown, Path()
        if set(rows) != set(scores) or set(rows) != set(counts):
            return unknown, Path()
        if not all(
            isinstance(key, str) and type(counts[key]) is int and counts[key] > 0 for key in rows
        ):
            return unknown, Path()
        fresh = collect_result_diagnostics(root, model, list(rows), {}, None, counts)
        # Canonical JSON equality rejects booleans masquerading as numeric0 and
        # unknown/extra schema fields as well as stale digests, IDs or counters.
        if json.dumps(cached, sort_keys=True, allow_nan=False) != json.dumps(
            fresh, sort_keys=True, allow_nan=False
        ):
            return unknown, Path()
        return fresh, root
    except (OSError, ValueError, TypeError, RecursionError):
        return unknown, Path()


def load_scores(path: Path) -> Arm:
    """Rebuild scores; optional diagnostics require their actual contained source."""
    data = json.loads(path.read_text(encoding="utf-8"))
    diagnostic, root = saved_result_diagnostics(data, path)
    return Arm(
        name=data["name"],
        base_url=data.get("base_url", ""),
        project_root=root,
        scores=data["scores"],
        counts=data.get("counts", {}),
        timed_out=data.get("timed_out", []),
        result_diagnostics=diagnostic,
    )


def write_report_and_gate(
    baseline: Arm,
    candidate: Arm,
    categories: list[str],
    args: argparse.Namespace,
    deadline: LifecycleDeadline | None = None,
) -> int:
    """Emit the markdown + JSON comparison and apply the completeness/regression gates."""
    check_work(baseline, deadline)
    check_work(candidate, deadline)
    report_md, payload = build_report(baseline, candidate, categories)
    if not all([check_work(arm, deadline) for arm in (baseline, candidate)]):
        report_md, payload = build_report(baseline, candidate, categories)
    print("\n" + report_md)
    if args.out:
        args.out.write_text(report_md + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    exit_code = EXIT_OK
    overall = payload["overall"]
    if overall["delta"] is not None and overall["delta"] < -args.tolerance:
        print(
            f"\nREGRESSION: {candidate.name} overall is {overall['delta'] * 100:.2f}pp "
            f"below {baseline.name} (tolerance {args.tolerance * 100:.2f}pp)",
            file=sys.stderr,
        )
        exit_code = EXIT_REGRESSION

    # An incomplete arm outranks a regression: there is no trustworthy delta to gate on.
    for name, reason in payload["incomplete"].items():
        print(f"\nINCOMPLETE: arm {name} — {reason}", file=sys.stderr)
    check_work(baseline, deadline)
    check_work(candidate, deadline)
    if any(lifecycle_failed(arm) for arm in (baseline, candidate)):
        # A write itself may have consumed the last work budget. Persist the
        # failed verdict too; blocked I/O still needs the caller's supervisor.
        report_md, payload = build_report(baseline, candidate, categories)
        if args.out:
            args.out.write_text(report_md + "\n", encoding="utf-8")
        if args.json_out:
            args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        exit_code = EXIT_INCOMPLETE
    elif payload["incomplete"] and not args.allow_incomplete:
        exit_code = EXIT_INCOMPLETE
    return exit_code


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Concurrent mode (both arms serving at once): pass both.
    p.add_argument("--baseline", help="name=base_url, e.g. vllm=http://127.0.0.1:31199")
    p.add_argument("--candidate", help="name=base_url, e.g. smg=http://127.0.0.1:31200")
    # Sequential mode (one arm owns the whole node, TP=8): score one live arm now,
    # then later diff the two saved score files.
    p.add_argument("--score-arm", help="name=base_url of the single live arm to score")
    p.add_argument("--scores-out", type=Path, help="write this arm's scores JSON here")
    p.add_argument("--diff-baseline", type=Path, help="baseline scores JSON (from --scores-out)")
    p.add_argument("--diff-candidate", type=Path, help="candidate scores JSON (from --scores-out)")
    p.add_argument(
        "--bfcl-model",
        help="BFCL model handler name, e.g. Qwen/Qwen3-4B-Instruct-2507-FC",
    )
    p.add_argument(
        "--categories",
        default="simple_python",
        help="comma-separated BFCL test categories",
    )
    p.add_argument("--bfcl", default="bfcl", help="path to the bfcl executable")
    p.add_argument("--project-root", default="/tmp/bfcl_ab", type=Path)
    p.add_argument("--num-threads", default=16, type=int)
    p.add_argument("--temperature", default=0.001, type=float)
    p.add_argument(
        "--tolerance",
        default=0.02,
        type=float,
        help="max allowed candidate-below-baseline overall drop",
    )
    p.add_argument(
        "--run-timeout",
        default=0,
        type=int,
        help="seconds before a bfcl generate/evaluate is killed (0 = no cap)",
    )
    p.add_argument(
        "--work-deadline-epoch",
        type=float,
        help="admitted absolute UTC work cutoff; requires --cleanup-deadline-epoch",
    )
    p.add_argument(
        "--cleanup-deadline-epoch",
        type=float,
        help="absolute UTC cleanup cutoff, strictly later than work cutoff",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report a timed-out or unscored arm without failing (exit 2 otherwise)",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse existing generation results, only evaluate",
    )
    p.add_argument("--out", type=Path, help="write the markdown report here")
    p.add_argument("--json-out", type=Path, help="write the JSON report here")
    args = p.parse_args()

    try:
        deadline = LifecycleDeadline.from_epochs(
            args.work_deadline_epoch, args.cleanup_deadline_epoch
        )
    except ValueError as exc:
        p.error(str(exc))
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Mode: diff two previously-saved score files (sequential mode, final step).
    if args.diff_baseline or args.diff_candidate:
        if not (args.diff_baseline and args.diff_candidate):
            p.error("--diff-baseline and --diff-candidate must be given together")
        return write_report_and_gate(
            load_scores(args.diff_baseline),
            load_scores(args.diff_candidate),
            categories,
            args,
            deadline,
        )

    # Mode: score a single live arm and persist its scores (sequential mode, per arm).
    if args.score_arm:
        if not (args.bfcl_model and args.scores_out):
            p.error("--score-arm requires --bfcl-model and --scores-out")
        arm = parse_arm(args.score_arm, args.project_root)
        run_bfcl(
            arm,
            bfcl=args.bfcl,
            model=args.bfcl_model,
            categories=categories,
            num_threads=args.num_threads,
            temperature=args.temperature,
            skip_generate=args.skip_generate,
            run_timeout=args.run_timeout,
            deadline=deadline,
        )
        check_work(arm, deadline)
        save_scores(arm, args.scores_out)
        check_work(arm, deadline)
        if lifecycle_failed(arm):
            # If saving crossed the work cutoff, do not leave eligible cached
            # scores for a later --diff. Preserve the nonwaivable marker.
            save_scores(arm, args.scores_out)
        print(f"[{arm.name}] scores -> {args.scores_out}: {arm.scores}")
        if arm.timed_out:
            print(f"WARNING: [{arm.name}] timed out during: {', '.join(arm.timed_out)}")
        # Legacy stage incompleteness is persisted for the final --diff gate.
        # Lifecycle failures return 2 now, including unsettled legacy cleanup.
        # Callers must retain partial evidence and stop before more native work.
        return EXIT_INCOMPLETE if lifecycle_failed(arm) else EXIT_OK

    # Mode: concurrent A/B — both arms serve at once on opposite GPU halves, so
    # score them in PARALLEL (separate servers, separate project_roots, no
    # contention) — roughly halves wall-clock vs scoring one then the other.
    if not (args.baseline and args.candidate and args.bfcl_model):
        p.error("concurrent mode requires --baseline, --candidate and --bfcl-model")
    baseline = parse_arm(args.baseline, args.project_root)
    candidate = parse_arm(args.candidate, args.project_root)

    def score(arm: Arm) -> None:
        run_bfcl(
            arm,
            bfcl=args.bfcl,
            model=args.bfcl_model,
            categories=categories,
            num_threads=args.num_threads,
            temperature=args.temperature,
            skip_generate=args.skip_generate,
            run_timeout=args.run_timeout,
            deadline=deadline,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # list() forces both futures to complete and re-raises any exception.
        list(ex.map(score, (baseline, candidate)))
    return write_report_and_gate(baseline, candidate, categories, args, deadline)


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        # Exit 1 is reserved for a completed comparison's score regression.
        exit_code = EXIT_INCOMPLETE
        with contextlib.suppress(Exception):
            traceback.print_exc()
    raise SystemExit(exit_code)
