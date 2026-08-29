#!/usr/bin/env python3
"""Run τ²-bench against two serving "arms" and diff pass^k. Track B (multi-turn).

baseline = pure vLLM; candidate = SMG -> vLLM gRPC. Both expose an identical
OpenAI /v1 endpoint; the official `tau2` CLI points --agent-llm at each arm and
--user-llm at a FIXED gpt-5.2, so any score delta is attributable to the
frontend (tokenization + tool/reasoning parsing). Arms must already be serving
(see launch_arms.sh); this driver does not launch them.

Exit codes: 0 clean, 1 score regression beyond --tolerance, 2 an arm did not
finish (timed out or scored nothing), so there is nothing to compare.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INCOMPLETE = 2


@dataclass
class Arm:
    name: str
    base_url: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)  # domains killed by --run-timeout


def passk(num_success: int, num_trials: int, k: int) -> float:
    """tau-bench pass^k unbiased estimator: C(c,k)/C(n,k); 0 if k>n or n==0."""
    if num_trials == 0 or k > num_trials:
        return 0.0
    if k <= 0:
        return 1.0
    return comb(num_success, k) / comb(num_trials, k)


def load_results(raw: dict) -> list[dict]:
    """Flatten τ²-bench results.json to [{task_id, reward}] (one record per trial).

    Validated schema (tau2-bench 1.0.0, recon d8e915f): the top-level Results
    object has a `simulations` list; each SimulationRun carries `task_id` and
    `reward_info.reward` (0.0/1.0). (`simulation_index[].reward` mirrors it.)
    """
    out: list[dict] = []
    for s in raw["simulations"]:
        out.append({"task_id": str(s["task_id"]), "reward": float(s["reward_info"]["reward"])})
    return out


def domain_scores(results: list[dict], k: int) -> dict[str, float]:
    """pass1 = mean reward over all trials; passk = mean over tasks of C(c,k)/C(n,k).

    Also reports the sample sizes behind those numbers: n_tasks (pass^k denominator)
    and n_sims (total trials = pass^1 denominator), so the report can show how much
    data backs each cell (and expose arm asymmetry when a domain is missing).
    """
    by_task: dict[str, list[float]] = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    all_rewards = [x for xs in by_task.values() for x in xs]
    pass1 = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    per_task = [passk(sum(1 for x in xs if x >= 1.0), len(xs), k) for xs in by_task.values()]
    passk_val = sum(per_task) / len(per_task) if per_task else 0.0
    return {"pass1": pass1, "passk": passk_val, "n_tasks": len(by_task), "n_sims": len(all_rewards)}


def run_tau2(
    arm: Arm,
    *,
    tau2: str,
    agent_model: str,
    domain: str,
    num_trials: int,
    num_tasks: int,
    max_concurrency: int,
    user_llm: str,
    data_dir: Path,
    request_timeout: int = 0,
    run_timeout: int = 0,
) -> None:
    """Run `tau2 run` for one arm+domain, then read back its results.json.

    Validated routing (tau2-bench 1.0.0): the agent uses LiteLLM's OpenAI provider
    with a per-call `api_base` pointing at this arm (via --agent-llm-args); the
    user uses the fixed gpt-5.2. Results land at
    <data_dir>/simulations/<save_to>/results.json.

    Two fail-fast bounds keep one pathological task from consuming the whole CI
    budget (a single Qwen3.6-27B airline sim once hung ~5h until the 6h job limit):
    `request_timeout` caps each LiteLLM call so a degenerate generation errors out
    instead of streaming for tens of minutes; `run_timeout` caps the whole domain
    subprocess so a wedged `tau2 run` is killed and its domain left unscored (the
    report renders "—") rather than blocking forever. Both are opt-in (0 = off).
    """
    save_to = f"ab_{arm.name}_{domain}"
    agent_llm_args: dict[str, object] = {
        "api_base": arm.base_url.rstrip("/") + "/v1",
        "api_key": "smg-local",
        "temperature": 0.0,
    }
    if request_timeout > 0:
        # LiteLLM completion kwarg: hard per-request wall-clock cap.
        agent_llm_args["timeout"] = request_timeout
    agent_args = json.dumps(agent_llm_args)
    user_args = json.dumps({"temperature": 0.0})
    cmd = [
        tau2,
        "run",
        "--domain",
        domain,
        "--agent-llm",
        f"openai/{agent_model}",
        "--agent-llm-args",
        agent_args,
        "--user-llm",
        user_llm,
        "--user-llm-args",
        user_args,
        "--num-trials",
        str(num_trials),
        "--save-to",
        save_to,
    ]
    if num_tasks > 0:
        cmd += ["--num-tasks", str(num_tasks)]
    if max_concurrency > 0:
        cmd += ["--max-concurrency", str(max_concurrency)]
    print(f"\n=== [{arm.name}/{domain}] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # Pin tau2's write dir to the same dir we read from, so results land where we
    # look for them regardless of any inherited $TAU2_DATA_DIR or whether tau2 is
    # installed editable vs into site-packages.
    env["TAU2_DATA_DIR"] = str(data_dir)
    try:
        proc = subprocess.run(cmd, env=env, check=False, timeout=(run_timeout or None))
    except subprocess.TimeoutExpired:
        # tau2 is a single in-process asyncio run (no forked workers), so run() has
        # already SIGKILLed it. Leave the domain unscored and move on; the servers
        # are torn down by the workflow's own cleanup/nuke_gpus trap.
        print(
            f"WARNING: [{arm.name}/{domain}] timed out after {run_timeout}s; killed",
            file=sys.stderr,
        )
        arm.timed_out.append(domain)
        return
    if proc.returncode != 0:
        print(f"WARNING: [{arm.name}/{domain}] exited {proc.returncode}", file=sys.stderr)
    results_json = data_dir / "simulations" / save_to / "results.json"
    try:
        arm.scores[domain] = domain_scores(
            load_results(json.loads(results_json.read_text())), k=num_trials
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        # tau2 may have died before writing results (OOM, engine death, API auth).
        # Skip this domain rather than aborting — build_report renders "—" for a
        # missing domain, so one failure doesn't discard the rest of the run.
        print(f"WARNING: [{arm.name}/{domain}] no usable results ({e})", file=sys.stderr)


def incompleteness(arm: Arm, domains: list[str]) -> str | None:
    """Why this arm can't be compared (timed out / unscored domains), else None."""
    missing = [d for d in domains if d not in arm.scores]
    if not missing and not arm.timed_out:
        return None
    reason = f"no score for {len(missing)}/{len(domains)} domains"
    if arm.timed_out:
        reason += f"; timed out: {', '.join(arm.timed_out)}"
    if missing:
        reason += f"; unscored: {', '.join(missing)}"
    return reason


def build_report(baseline: Arm, candidate: Arm, domains: list[str], k: int):
    """Markdown + JSON; candidate − baseline; overall = unweighted mean."""

    def cell(x):
        return "—" if x is None else f"{x * 100:.2f}"

    def dcell(x):
        return "—" if x is None else f"{x * 100:+.2f}"

    rows, agg = [], {"pass1": {"b": [], "c": []}, "passk": {"b": [], "c": []}}
    nsum = {"b": 0, "c": 0}
    for d in domains:
        b, c = baseline.scores.get(d, {}), candidate.scores.get(d, {})
        row = {
            "domain": d,
            "n": {
                "baseline": {"tasks": b.get("n_tasks"), "sims": b.get("n_sims")},
                "candidate": {"tasks": c.get("n_tasks"), "sims": c.get("n_sims")},
            },
        }
        nsum["b"] += b.get("n_sims") or 0
        nsum["c"] += c.get("n_sims") or 0
        for m in ("pass1", "passk"):
            bv, cv = b.get(m), c.get(m)
            row[m] = {
                "baseline": bv,
                "candidate": cv,
                "delta": (cv - bv) if (bv is not None and cv is not None) else None,
            }
            if bv is not None:
                agg[m]["b"].append(bv)
            if cv is not None:
                agg[m]["c"].append(cv)
        rows.append(row)

    overall = {}
    for m in ("pass1", "passk"):
        bo = sum(agg[m]["b"]) / len(agg[m]["b"]) if agg[m]["b"] else None
        co = sum(agg[m]["c"]) / len(agg[m]["c"]) if agg[m]["c"] else None
        overall[m] = {
            "baseline": bo,
            "candidate": co,
            "delta": (co - bo) if (bo is not None and co is not None) else None,
        }

    # At k=1, pass^k == pass^1, so show only the pass^1 columns (a duplicated
    # triple is confusing). At k>1 show both pass^1 and pass^k.
    metrics = ["pass1"] if k == 1 else ["pass1", "passk"]
    mlabel = {"pass1": "pass^1", "passk": f"pass^{k}"}

    def triple(d, bold=False):
        b, c, dl = cell(d["baseline"]), cell(d["candidate"]), dcell(d["delta"])
        return f"**{b}** | **{c}** | **{dl}**" if bold else f"{b} | {c} | {dl}"

    def ncol(n_b, n_c, bold=False):
        s = f"{n_b or '—'}/{n_c or '—'}"
        return f"**{s}**" if bold else s

    header = " | ".join(
        f"{baseline.name} {mlabel[m]} | {candidate.name} {mlabel[m]} | Δ" for m in metrics
    )
    # No "τ²-bench A/B" title here — the workflow summary step owns that heading; this
    # caption only adds what it lacks (which arm is which). Avoids a stacked duplicate.
    lines = [
        f"**{candidate.name}** (candidate) vs **{baseline.name}** (baseline)",
        "",
        f"| domain | N ({baseline.name}/{candidate.name}) | {header} |",
        "|---" * (2 + 3 * len(metrics)) + "|",
    ]
    for r in rows:
        n = r["n"]
        lines.append(
            f"| {r['domain']} | {ncol(n['baseline']['sims'], n['candidate']['sims'])} | "
            + " | ".join(triple(r[m]) for m in metrics)
            + " |"
        )
    lines.append(
        f"| **overall** | {ncol(nsum['b'], nsum['c'], bold=True)} | "
        + " | ".join(triple(overall[m], bold=True) for m in metrics)
        + " |"
    )
    lines.append("")
    incomplete = {a.name: incompleteness(a, domains) for a in (baseline, candidate)}
    incomplete = {n: r for n, r in incomplete.items() if r}
    for name, reason in incomplete.items():
        lines.append(f"> ⚠️ **Incomplete arm `{name}`** — {reason}.")
    lines += [
        "",
        f"_N = simulations per arm (baseline/candidate) = tasks × {k} trials. "
        "Same model · engine · checkpoint · sampling · user-sim (gpt-5.2) on both arms "
        "— only the frontend differs, so Δ is the parsing layer._",
    ]
    overall_out = dict(overall)
    overall_out["n"] = {"baseline": nsum["b"], "candidate": nsum["c"]}
    payload = {
        "baseline": {
            "name": baseline.name,
            "scores": baseline.scores,
            "timed_out": baseline.timed_out,
        },
        "candidate": {
            "name": candidate.name,
            "scores": candidate.scores,
            "timed_out": candidate.timed_out,
        },
        "incomplete": incomplete,
        "per_domain": rows,
        "overall": overall_out,
    }
    return "\n".join(lines), payload


def score_arm(
    arm: Arm,
    *,
    tau2: str,
    agent_model: str,
    domains: list[str],
    num_trials: int,
    num_tasks: int,
    max_concurrency: int,
    user_llm: str,
    data_dir: Path,
    request_timeout: int = 0,
    run_timeout: int = 0,
) -> None:
    """Run tau2 for every domain against one already-serving arm, filling arm.scores."""
    for domain in domains:
        run_tau2(
            arm,
            tau2=tau2,
            agent_model=agent_model,
            domain=domain,
            num_trials=num_trials,
            num_tasks=num_tasks,
            max_concurrency=max_concurrency,
            user_llm=user_llm,
            data_dir=data_dir,
            request_timeout=request_timeout,
            run_timeout=run_timeout,
        )


def save_scores(arm: Arm, path: Path) -> None:
    """Persist one arm's per-domain scores so a later --diff can compare them.

    Sequential mode: a whole-node model (TP=8) can't run both arms at once, so
    each arm is scored on its own and the two score files are diffed afterwards.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": arm.name,
                "base_url": arm.base_url,
                "scores": arm.scores,
                "timed_out": arm.timed_out,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_scores(path: Path) -> Arm:
    """Rebuild an Arm (name + per-domain scores) from a file written by save_scores."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Arm(
        name=data["name"],
        base_url=data.get("base_url", ""),
        scores=data["scores"],
        timed_out=data.get("timed_out", []),
    )


def write_report_and_gate(
    baseline: Arm, candidate: Arm, domains: list[str], k: int, args: argparse.Namespace
) -> int:
    """Emit the markdown + JSON comparison and apply the completeness/regression gates."""
    report_md, payload = build_report(baseline, candidate, domains, k)
    print("\n" + report_md)
    if args.out:
        args.out.write_text(report_md + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    exit_code = EXIT_OK
    delta = payload["overall"]["passk"]["delta"]
    if delta is not None and delta < -args.tolerance:
        print(
            f"\nREGRESSION: {candidate.name} pass^{k} {delta * 100:.2f}pp "
            f"below {baseline.name} (tol {args.tolerance * 100:.2f}pp)",
            file=sys.stderr,
        )
        exit_code = EXIT_REGRESSION

    # An incomplete arm outranks a regression: there is no trustworthy delta to gate on.
    for name, reason in payload["incomplete"].items():
        print(f"\nINCOMPLETE: arm {name} — {reason}", file=sys.stderr)
    if payload["incomplete"] and not args.allow_incomplete:
        exit_code = EXIT_INCOMPLETE
    return exit_code


def _parse_arm(spec: str) -> Arm:
    name, url = spec.split("=", 1)
    return Arm(name=name, base_url=url)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Concurrent mode (both arms serving at once on opposite GPU halves): pass both.
    p.add_argument("--baseline", help="name=base_url (concurrent mode)")
    p.add_argument("--candidate", help="name=base_url (concurrent mode)")
    # Sequential mode (one arm owns the whole node, TP=8): score one live arm now,
    # then later diff the two saved score files.
    p.add_argument("--score-arm", help="name=base_url of the single live arm to score")
    p.add_argument("--scores-out", type=Path, help="write this arm's per-domain scores JSON here")
    p.add_argument("--diff-baseline", type=Path, help="baseline scores JSON (from --scores-out)")
    p.add_argument("--diff-candidate", type=Path, help="candidate scores JSON (from --scores-out)")
    p.add_argument("--domains", default="retail,airline,telecom")
    p.add_argument("--num-trials", type=int, default=2)
    p.add_argument("--num-tasks", type=int, default=0, help="0 = all tasks")
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="tau2 --max-concurrency (concurrent simulations per arm; 0 = tau2 default)",
    )
    p.add_argument(
        "--request-timeout",
        type=int,
        default=0,
        help="per-LiteLLM-request timeout in seconds injected into the agent (0 = off)",
    )
    p.add_argument(
        "--run-timeout",
        type=int,
        default=0,
        help="per-domain `tau2 run` wall-clock cap in seconds; killed if exceeded (0 = off)",
    )
    p.add_argument(
        "--agent-model",
        default="Qwen/Qwen3.8-27B",
        help="served model name on both arms (used as openai/<name>)",
    )
    p.add_argument("--user-llm", default="gpt-5.2", help="fixed user-sim model")
    p.add_argument("--tau2", default="tau2", help="path to the tau2 executable")
    p.add_argument(
        "--data-dir",
        type=Path,
        help="tau2 DATA_DIR (results written/read under <data-dir>/simulations)",
    )
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report a timed-out or unscored arm without failing (exit 2 otherwise)",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    # Mode: diff two previously-saved score files (sequential mode, final step).
    if args.diff_baseline or args.diff_candidate:
        if not (args.diff_baseline and args.diff_candidate):
            p.error("--diff-baseline and --diff-candidate must be given together")
        baseline = load_scores(args.diff_baseline)
        candidate = load_scores(args.diff_candidate)
        # Requested domains are part of the expected set, so one that neither arm
        # scored still shows up as missing instead of silently vanishing.
        diff_domains = sorted(set(domains) | set(baseline.scores) | set(candidate.scores))
        return write_report_and_gate(baseline, candidate, diff_domains, args.num_trials, args)

    # Mode: score a single live arm and persist its scores (sequential mode, per arm).
    if args.score_arm:
        if not (args.scores_out and args.data_dir):
            p.error("--score-arm requires --scores-out and --data-dir")
        arm = _parse_arm(args.score_arm)
        score_arm(
            arm,
            tau2=args.tau2,
            agent_model=args.agent_model,
            domains=domains,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            max_concurrency=args.max_concurrency,
            user_llm=args.user_llm,
            data_dir=args.data_dir,
            request_timeout=args.request_timeout,
            run_timeout=args.run_timeout,
        )
        save_scores(arm, args.scores_out)
        print(f"[{arm.name}] scores -> {args.scores_out}: {arm.scores}")
        if arm.timed_out:
            print(f"WARNING: [{arm.name}] timed out on: {', '.join(arm.timed_out)}")
        # Always 0: the gate runs once, on the --diff step that sees both arms.
        return EXIT_OK

    # Mode: concurrent A/B — arms serve on opposite GPU halves, so score them in
    # PARALLEL (separate servers, separate save_to dirs) to roughly halve wall-clock.
    if not (args.baseline and args.candidate and args.data_dir):
        p.error("concurrent mode requires --baseline, --candidate and --data-dir")
    baseline = _parse_arm(args.baseline)
    candidate = _parse_arm(args.candidate)

    def score(arm: Arm) -> None:
        score_arm(
            arm,
            tau2=args.tau2,
            agent_model=args.agent_model,
            domains=domains,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            max_concurrency=args.max_concurrency,
            user_llm=args.user_llm,
            data_dir=args.data_dir,
            request_timeout=args.request_timeout,
            run_timeout=args.run_timeout,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # list() forces both futures to complete and re-raises any exception.
        list(ex.map(score, (baseline, candidate)))
    return write_report_and_gate(baseline, candidate, domains, args.num_trials, args)


if __name__ == "__main__":
    raise SystemExit(main())
