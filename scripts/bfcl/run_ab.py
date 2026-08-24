#!/usr/bin/env python3
"""Run the official BFCL benchmark against serving arms and report the scores.

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

The normal mode compares two arms. ``--report-arm`` renders absolute scores for
a saved single arm when the serving engine has no comparable pure-vLLM path.

Exit codes: ``0`` clean, ``1`` score regression beyond ``--tolerance``, ``2`` an
arm did not finish (timed out, missed a category, or processed zero cases).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INCOMPLETE = 2


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
    failed: dict[str, str] = field(default_factory=dict)  # nonzero benchmark stages


def parse_arm(spec: str, project_root: Path) -> Arm:
    """Parse a ``name=url`` arm spec and derive host/port + a per-arm BFCL root."""
    if "=" not in spec:
        raise ValueError(f"arm spec must be name=url, got: {spec!r}")
    name, url = spec.split("=", 1)
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 80)
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
) -> None:
    """Run ``bfcl generate`` + ``bfcl evaluate`` for one arm, in its own root.

    ``run_timeout`` (0 = off) mirrors tau2's ``--run-timeout``. No BFCL analogue of
    ``--request-timeout`` exists: the FC handler's ``openai.OpenAI`` client reads
    ``api_key``/``base_url`` from the environment but not ``timeout``, and ``bfcl
    generate`` exposes no passthrough, so 600s x 2 retries is the per-call ceiling.
    """
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
    if not skip_generate:
        generate_rc = _run(
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
            timeout=run_timeout,
        )
        if generate_rc is None:
            arm.timed_out.append("generate")
        elif generate_rc != 0:
            arm.failed["generate"] = f"exited {generate_rc}"
    # Evaluate even after a killed generate: whatever completed is already on disk.
    evaluate_rc = _run(
        [bfcl, "evaluate", "--model", model, "--test-category", cats],
        env,
        f"[{arm.name}] evaluate",
        timeout=run_timeout,
    )
    if evaluate_rc is None:
        arm.timed_out.append("evaluate")
    elif evaluate_rc != 0:
        arm.failed["evaluate"] = f"exited {evaluate_rc}"
    arm.scores, arm.counts = parse_scores(arm.project_root, model, categories)


def _run(cmd: list[str], env: dict[str, str], label: str, timeout: int = 0) -> int | None:
    """Run one bfcl subcommand; return its status, or None after a timeout."""
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    # Own session so a timeout can SIGKILL the whole group: bfcl generate fans out
    # over a thread pool and can sit blocked on a hung HTTP read.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=timeout or None)
    except subprocess.TimeoutExpired:
        _killpg(proc)
        print(f"WARNING: {label} timed out after {timeout}s; killed", file=sys.stderr)
        return None
    if returncode != 0:
        print(f"WARNING: {label} exited {returncode}", file=sys.stderr)
    return returncode


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def parse_scores(
    project_root: Path, model: str, categories: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Extract per-category accuracy and test-case count from BFCL's score output.

    BFCL writes ``<root>/score/<sanitized-model>/<category>_score.json`` whose
    FIRST line is a summary dict ``{"accuracy", "correct_count", "total_count"}``.
    We glob for the model dir (sanitization differs across versions) and read each
    category's summary. ``total_count`` is what weights the weighted overall.
    """
    score_root = project_root / "score"
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
    # BFCL nests scores as <model>/<section>/BFCL_v4_<category>_score.json, so
    # match a trailing-wildcard pattern (the BFCL_v4_ prefix varies by version).
    for path in score_root.rglob(f"*{category}_score.json"):
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
    zero_count = [c for c in categories if c in arm.scores and arm.counts.get(c, 0) <= 0]
    if not missing and not zero_count and not arm.timed_out and not arm.failed:
        return None
    reasons = []
    if missing:
        shown = ", ".join(missing[:8]) + (f", +{len(missing) - 8} more" if len(missing) > 8 else "")
        reasons.append(f"no score for {len(missing)}/{len(categories)} categories: {shown}")
    if arm.timed_out:
        reasons.append(f"timed out during: {', '.join(arm.timed_out)}")
    if arm.failed:
        reasons.append(
            "failed/partial: "
            + ", ".join(f"{stage} ({failure})" for stage, failure in arm.failed.items())
        )
    if zero_count:
        reasons.append(f"zero cases: {', '.join(zero_count)}")
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

    payload = {
        "baseline": {
            "name": baseline.name,
            "base_url": baseline.base_url,
            "scores": baseline.scores,
            "counts": baseline.counts,
            "timed_out": baseline.timed_out,
            "failed": baseline.failed,
        },
        "candidate": {
            "name": candidate.name,
            "base_url": candidate.base_url,
            "scores": candidate.scores,
            "counts": candidate.counts,
            "timed_out": candidate.timed_out,
            "failed": candidate.failed,
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


def build_single_report(arm: Arm, categories: list[str]) -> tuple[str, dict]:
    """Build an absolute BFCL report for one arm with no comparable baseline."""
    rows = [
        {
            "category": category,
            "accuracy": arm.scores.get(category),
            "count": arm.counts.get(category),
        }
        for category in categories
    ]
    numeric = [row["accuracy"] for row in rows if row["accuracy"] is not None]
    overall = sum(numeric) / len(numeric) if numeric else None
    weighted_num = sum(
        row["accuracy"] * row["count"]
        for row in rows
        if row["accuracy"] is not None and row["count"]
    )
    weighted_den = sum(row["count"] for row in rows if row["accuracy"] is not None and row["count"])
    overall_weighted = weighted_num / weighted_den if weighted_den else None

    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.2f}"

    lines = [
        f"# BFCL — {arm.name} (single arm)",
        "",
        "| category | n | accuracy |",
        "|---|---:|---:|",
    ]
    for row in rows:
        count = "—" if row["count"] is None else str(row["count"])
        lines.append(f"| {row['category']} | {count} | {fmt(row['accuracy'])} |")
    total = sum(row["count"] for row in rows if row["count"]) or "—"
    lines.extend(
        [
            f"| **overall (unweighted)** | {total} | **{fmt(overall)}** |",
            f"| **overall (weighted by n)** | {total} | **{fmt(overall_weighted)}** |",
            "",
        ]
    )
    incomplete = incompleteness(arm, categories)
    if incomplete:
        lines.extend([f"> ⚠️ **Incomplete arm `{arm.name}`** — {incomplete}.", ""])
    lines.append(
        "_Absolute official BFCL FC-mode scores. This is a single-arm run because "
        "the comparison engine cannot serve this checkpoint; no frontend parity "
        "claim or A/B delta is implied._"
    )
    payload = {
        "mode": "single_arm",
        "arm": {
            "name": arm.name,
            "base_url": arm.base_url,
            "scores": arm.scores,
            "counts": arm.counts,
            "timed_out": arm.timed_out,
            "failed": arm.failed,
        },
        "incomplete": incomplete,
        "per_category": rows,
        "overall": overall,
        "overall_weighted": overall_weighted,
    }
    return "\n".join(lines), payload


def save_scores(arm: Arm, path: Path) -> None:
    """Persist one arm's per-category scores so a later --diff can compare them.

    Used by the sequential mode: when a model needs the whole node (TP=8) the two
    arms cannot run at once, so each is scored on its own and diffed afterwards.
    """
    path.write_text(
        json.dumps(
            {
                "name": arm.name,
                "base_url": arm.base_url,
                "scores": arm.scores,
                "counts": arm.counts,
                "timed_out": arm.timed_out,
                "failed": arm.failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_scores(path: Path) -> Arm:
    """Rebuild an Arm from a file written by save_scores."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Arm(
        name=data["name"],
        base_url=data.get("base_url", ""),
        scores=data["scores"],
        counts=data.get("counts", {}),
        timed_out=data.get("timed_out", []),
        failed=data.get("failed", {}),
    )


def write_report_and_gate(
    baseline: Arm, candidate: Arm, categories: list[str], args: argparse.Namespace
) -> int:
    """Emit the markdown + JSON comparison and apply the completeness/regression gates."""
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
    if payload["incomplete"] and not args.allow_incomplete:
        exit_code = EXIT_INCOMPLETE
    return exit_code


def write_single_report_and_gate(arm: Arm, categories: list[str], args: argparse.Namespace) -> int:
    """Emit one arm's absolute report and reject incomplete benchmark output."""
    report_md, payload = build_single_report(arm, categories)
    print("\n" + report_md)
    if args.out:
        args.out.write_text(report_md + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if payload["incomplete"] and not args.allow_incomplete:
        print(f"\nINCOMPLETE: arm {arm.name} — {payload['incomplete']}", file=sys.stderr)
        return EXIT_INCOMPLETE
    return EXIT_OK


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
    p.add_argument(
        "--report-arm",
        type=Path,
        help="render and gate one saved arm's absolute scores (from --scores-out)",
    )
    p.add_argument("--diff-baseline", type=Path, help="baseline scores JSON (from --scores-out)")
    p.add_argument("--diff-candidate", type=Path, help="candidate scores JSON (from --scores-out)")
    p.add_argument(
        "--bfcl-model",
        help="BFCL model handler name, e.g. Qwen/Qwen3-4B-Instruct-2507-FC",
    )
    p.add_argument(
        "--categories", default="simple_python", help="comma-separated BFCL test categories"
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

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Mode: report one previously-scored arm when no comparable baseline exists.
    if args.report_arm:
        return write_single_report_and_gate(load_scores(args.report_arm), categories, args)

    # Mode: diff two previously-saved score files (sequential mode, final step).
    if args.diff_baseline or args.diff_candidate:
        if not (args.diff_baseline and args.diff_candidate):
            p.error("--diff-baseline and --diff-candidate must be given together")
        return write_report_and_gate(
            load_scores(args.diff_baseline), load_scores(args.diff_candidate), categories, args
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
        )
        save_scores(arm, args.scores_out)
        print(f"[{arm.name}] scores -> {args.scores_out}: {arm.scores}")
        if arm.timed_out:
            print(f"WARNING: [{arm.name}] timed out during: {', '.join(arm.timed_out)}")
        # Always 0: the gate runs once, on the --diff step that sees both arms.
        return EXIT_OK

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
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # list() forces both futures to complete and re-raises any exception.
        list(ex.map(score, (baseline, candidate)))
    return write_report_and_gate(baseline, candidate, categories, args)


if __name__ == "__main__":
    raise SystemExit(main())
