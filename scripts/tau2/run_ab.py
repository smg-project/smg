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
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guarded_cli as budget_guard

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INCOMPLETE = 2


@dataclass
class Arm:
    name: str
    base_url: str
    scores: dict[str, dict[str, object]] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)  # domains killed by --run-timeout
    invocations: dict[str, dict[str, object]] = field(default_factory=dict)
    budget: dict | None = None


def passk(num_success: int, num_trials: int, k: int) -> float:
    """tau-bench pass^k unbiased estimator: C(c,k)/C(n,k); 0 if k>n or n==0."""
    if num_trials == 0 or k > num_trials:
        return 0.0
    if k <= 0:
        return 1.0
    return comb(num_success, k) / comb(num_trials, k)


def load_results(raw: dict) -> list[dict]:
    """Flatten τ²-bench results.json to task/trial identity and reward records.

    Validated schema (tau2-bench 1.0.0, recon d8e915f): the top-level Results
    object has a `simulations` list; each SimulationRun carries `task_id` and
    `reward_info.reward` (0.0/1.0). (`simulation_index[].reward` mirrors it.)
    """
    out: list[dict] = []
    for s in raw["simulations"]:
        out.append(
            {
                "task_id": str(s["task_id"]),
                "trial": s.get("trial"),
                "reward": float(s["reward_info"]["reward"]),
            }
        )
    return out


def domain_scores(results: list[dict], k: int) -> dict[str, object]:
    """pass1 = mean reward over all trials; passk = mean over tasks of C(c,k)/C(n,k).

    Also reports the sample sizes behind those numbers: n_tasks (pass^k denominator)
    and n_sims (total trials = pass^1 denominator), so the report can show how much
    data backs each cell (and expose arm asymmetry when a domain is missing).
    """
    if not results:
        raise ValueError("no simulations were executed")
    by_task: dict[str, list[float]] = {}
    task_trials: dict[str, list[int | None]] = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
        task_trials.setdefault(r["task_id"], []).append(r.get("trial"))
    all_rewards = [x for xs in by_task.values() for x in xs]
    pass1 = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    per_task = [passk(sum(1 for x in xs if x >= 1.0), len(xs), k) for xs in by_task.values()]
    passk_val = sum(per_task) / len(per_task) if per_task else 0.0
    return {
        "pass1": pass1,
        "passk": passk_val,
        "n_tasks": len(by_task),
        "n_sims": len(all_rewards),
        "task_trials": task_trials,
    }


def prepare_expected_tasks(domains: list[str], num_tasks: int, num_trials: int, seed: int) -> dict:
    """Offline selection using the caller's pinned tau2 install, before any LLM call."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import _load_run_tasks

    selected = {}
    for domain in domains:
        config = TextRunConfig(
            domain=domain,
            agent="llm_agent",
            task_split_name="base",
            num_tasks=num_tasks or None,
            num_trials=num_trials,
            seed=seed,
        )
        ids = [str(task.id) for task in _load_run_tasks(config)]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"empty or duplicate expected tasks for {domain}")
        selected[domain] = ids
    return {"schema_version": 1, "domains": selected, "num_trials": num_trials, "seed": seed}


def read_expected_tasks(path: Path, domains: list[str], k: int, seed: int) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("expected task manifest must be an object")
    if data.get("schema_version") != 1 or data.get("num_trials") != k or data.get("seed") != seed:
        raise ValueError("expected task manifest schema/trials/seed mismatch")
    selected = data.get("domains")
    if not isinstance(selected, dict) or set(selected) != set(domains):
        raise ValueError("expected task manifest domains mismatch")
    for domain, ids in selected.items():
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(task, str) or not task for task in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"invalid expected task IDs for {domain}")
    return selected


def coverage_problem(scores: dict, expected: list[str] | None, k: int) -> str | None:
    if not expected:
        return "missing expected task manifest"
    manifest = scores.get("task_trials")
    if not isinstance(manifest, dict) or not manifest:
        return "missing task/trial identity manifest"
    if set(manifest) != set(expected):
        return "task IDs differ from expected selection"
    for task, trials in manifest.items():
        if (
            not isinstance(trials, list)
            or any(type(trial) is not int for trial in trials)
            or sorted(trials) != list(range(k))
        ):
            return f"missing, duplicate or unexpected trials for task {task}"
    if scores.get("n_tasks") != len(manifest) or scores.get("n_sims") != len(manifest) * k:
        return "sample counts disagree with task/trial identity manifest"
    return None


def prepare_budget(args=None):
    def value(name):
        return (
            getattr(args, name, None)
            if args is not None
            else os.environ.get("TAU2_" + name.upper())
        )

    policy_path, contexts_path, ledger_path = (
        value(k) for k in ("budget_policy", "budget_contexts", "budget_ledger")
    )
    contexts_sha = value("budget_contexts_sha256")
    if not all((policy_path, contexts_path, ledger_path, contexts_sha)):
        raise budget_guard.BudgetError(
            "Explicit paid policy, independently verified contexts and ONE ledger required"
        )
    policy, _ = budget_guard.load_policy(policy_path, contexts_path, contexts_sha)
    if value("budget_scope") != policy.scope_id:
        raise budget_guard.BudgetError("Explicit approved run/job scope must match policy")
    # Explicit guarded_cli --initialize is the sole creator, before either arm.
    budget_guard.Ledger(ledger_path, policy, create=False)
    return dict(
        policy_path=str(Path(policy_path).resolve()),
        contexts_path=str(Path(contexts_path).resolve()),
        contexts_sha256=contexts_sha,
        ledger_path=str(Path(ledger_path).resolve()),
        scope_id=policy.scope_id,
    )


def final_budget(baseline, candidate):
    if not baseline.budget or baseline.budget != candidate.budget:
        raise budget_guard.BudgetError(
            "Both saved arms must bind the same policy/scope/shared ledger"
        )
    state = budget_guard.read_descriptor(baseline.budget).audit()
    if state["halted"] or state["pending"] or state["unknown"]:
        raise budget_guard.BudgetError(
            "Shared paid budget halted, pending or unknown after all arms completed"
        )
    return state


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
    expected_task_ids: list[str] | None = None,
    seed: int = 300,
    budget_descriptor: dict | None = None,
    tau2_python: str | None = None,
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
    # A failed/new run must never resume or score a previous invocation's files.
    invocation_id = uuid.uuid4().hex
    save_to = f"ab_{invocation_id}"
    results_json = data_dir / "simulations" / save_to / "results.json"
    invocation = {
        "invocation_id": invocation_id,
        "arm": arm.name,
        "domain": domain,
        "results_path": str(results_json),
        "status": "running",
        "exit_code": None,
    }
    arm.invocations[domain] = invocation
    arm.scores.pop(domain, None)
    arm.timed_out = [d for d in arm.timed_out if d != domain]

    def fail(kind: str, stage: str, error: Exception | None = None) -> None:
        invocation.update(status="failed", failure_kind=kind, failure_stage=stage)
        if error is not None:
            invocation["exception_type"] = type(error).__name__
        print(
            f"WARNING: [{arm.name}/{domain}] invocation {invocation_id}: {kind} ({stage})",
            file=sys.stderr,
        )

    if os.path.lexists(results_json.parent):
        fail("output_exists", "preflight")
        return

    if budget_descriptor is None:
        fail("missing_budget", "budget")
        return
    arm.budget = budget_descriptor
    try:
        if budget_guard.read_descriptor(budget_descriptor).summary()["halted"]:
            raise budget_guard.BudgetError("Shared paid budget is halted")
    except (OSError, ValueError, KeyError) as error:
        fail("invalid_budget", "budget", error)
        return

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
    interpreter = tau2_python or str(Path(tau2).parent / "python")
    cmd = [
        interpreter,
        "-I",
        str(Path(__file__).with_name("guarded_cli.py")),
        "--policy",
        budget_descriptor["policy_path"],
        "--contexts",
        budget_descriptor["contexts_path"],
        "--contexts-sha256",
        budget_descriptor["contexts_sha256"],
        "--ledger",
        budget_descriptor["ledger_path"],
        "--scope",
        budget_descriptor["scope_id"],
        "--arm",
        arm.name,
        "--domain",
        domain,
        "--invocation",
        invocation_id,
        "--local-model",
        agent_model,
        "--local-url",
        arm.base_url,
        "--",
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
    cmd += ["--agent", "llm_agent", "--task-split-name", "base", "--seed", str(seed)]
    if expected_task_ids:
        cmd += ["--task-ids", *expected_task_ids]
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
    except subprocess.TimeoutExpired as error:
        # subprocess.run has killed/reaped its child. Even a partial result from
        # that invocation cannot establish a completed simulation domain.
        arm.timed_out.append(domain)
        fail("timeout", "cli", error)
        return
    except OSError as error:
        fail("launch_error", "cli", error)
        return
    invocation["exit_code"] = proc.returncode
    if proc.returncode != 0:
        # Do not interpret a generic process exit as an API/provider failure or
        # accept partial files the failed command happened to write.
        fail("nonzero_exit", "cli")
        return
    try:
        # B may legitimately be in flight while A returns; only A's requests are
        # required settled here. The GLOBAL pending/halt gate runs after join/diff.
        own = budget_guard.read_descriptor(budget_descriptor).audit(invocation_id)
        if own["pending"] or own["unknown"]:
            fail("invocation_budget_incomplete", "budget")
            return
        if results_json.is_symlink() or results_json.parent.is_symlink():
            raise ValueError("results must belong to the fresh invocation directory")
        body = results_json.read_bytes()
        scores = domain_scores(load_results(json.loads(body)), k=num_trials)
        invocation["results_sha256"] = hashlib.sha256(body).hexdigest()
        if issue := coverage_problem(scores, expected_task_ids, num_trials):
            fail("incomplete_results", "results")
            invocation["coverage_problem"] = issue
            return
        arm.scores[domain] = scores
        invocation["status"] = "completed"
    except FileNotFoundError as error:
        fail("missing_results", "results", error)
    except (OSError, KeyError, ValueError, TypeError) as error:
        fail("invalid_results", "results", error)


def incompleteness(
    arm: Arm, domains: list[str], expected_tasks: dict | None = None, k: int = 1
) -> str | None:
    """Why this arm can't be compared (timed out / unscored domains), else None."""
    missing = [d for d in domains if d not in arm.scores]
    # Saved score files (sequential mode) can predate the raw-results guard.
    # A zero score is valid, but it needs positive evidence of executed tasks.
    no_samples = [
        d
        for d in domains
        if d in arm.scores
        and any(
            type(arm.scores[d].get(key)) is not int or arm.scores[d][key] <= 0
            for key in ("n_tasks", "n_sims")
        )
    ]
    coverage = {
        d: issue
        for d in domains
        if d in arm.scores
        if (issue := coverage_problem(arm.scores[d], (expected_tasks or {}).get(d), k))
    }
    invocation_failures = {}
    for domain in domains:
        invocation = arm.invocations.get(domain)
        if invocation is None:
            continue  # Legacy score files do not gain new invocation provenance.
        if not isinstance(invocation, dict) or (
            invocation.get("arm"),
            invocation.get("domain"),
        ) != (arm.name, domain):
            invocation_failures[domain] = "invalid invocation identity"
        elif invocation.get("status") != "completed" or invocation.get("exit_code") != 0:
            invocation_failures[domain] = str(
                invocation.get("failure_kind", "unfinished invocation")
            )
    if (
        not missing
        and not arm.timed_out
        and not no_samples
        and not coverage
        and not invocation_failures
    ):
        return None
    reason = f"no score for {len(missing)}/{len(domains)} domains"
    if arm.timed_out:
        reason += f"; timed out: {', '.join(arm.timed_out)}"
    if missing:
        reason += f"; unscored: {', '.join(missing)}"
    if no_samples:
        reason += f"; no valid samples: {', '.join(no_samples)}"
    if coverage:
        reason += "; coverage: " + ", ".join(f"{d}: {issue}" for d, issue in coverage.items())
    if invocation_failures:
        reason += "; invocation failure: " + ", ".join(
            f"{d}: {issue}" for d, issue in invocation_failures.items()
        )
    return reason


def build_report(
    baseline: Arm, candidate: Arm, domains: list[str], k: int, expected_tasks: dict | None = None
):
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
    incomplete = {
        a.name: incompleteness(a, domains, expected_tasks, k) for a in (baseline, candidate)
    }
    incomplete = {n: r for n, r in incomplete.items() if r}
    for name, reason in incomplete.items():
        lines.append(f"> ⚠️ **Incomplete arm `{name}`** — {reason}.")
    lines += [
        "",
        f"_N = simulations per arm (baseline/candidate) = tasks × {k} trials. "
        "Same model · engine · checkpoint · sampling · user-sim (gpt-5.2) on both arms "
        "— a delta is comparable only when neither arm is marked incomplete._",
    ]
    overall_out = dict(overall)
    overall_out["n"] = {"baseline": nsum["b"], "candidate": nsum["c"]}
    payload = {
        "baseline": {
            "name": baseline.name,
            "scores": baseline.scores,
            "timed_out": baseline.timed_out,
            "invocations": baseline.invocations,
            "budget": baseline.budget,
        },
        "candidate": {
            "name": candidate.name,
            "scores": candidate.scores,
            "timed_out": candidate.timed_out,
            "invocations": candidate.invocations,
            "budget": candidate.budget,
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
    expected_tasks: dict | None = None,
    seed: int = 300,
    budget_descriptor: dict | None = None,
    tau2_python: str | None = None,
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
            expected_task_ids=(expected_tasks or {}).get(domain),
            seed=seed,
            budget_descriptor=budget_descriptor,
            tau2_python=tau2_python,
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
                "invocations": arm.invocations,
                "budget": arm.budget,
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
        invocations=data.get("invocations", {}),
        budget=data.get("budget"),
    )


def write_report_and_gate(
    baseline: Arm, candidate: Arm, domains: list[str], k: int, args: argparse.Namespace
) -> int:
    """Emit the markdown + JSON comparison and apply the completeness/regression gates."""
    report_md, payload = build_report(
        baseline, candidate, domains, k, getattr(args, "expected_task_ids", None)
    )
    budget_failed = False
    if getattr(args, "enforce_budget", False):
        try:
            payload["budget"] = final_budget(baseline, candidate)
        except (OSError, ValueError, KeyError) as error:
            budget_failed = True
            payload["incomplete"]["budget"] = str(error)
            report_md += f"\n\n> Incomplete paid budget: {error}."
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
    # Quality diagnostics cannot waive failed mandatory paid-budget acceptance.
    if budget_failed or (payload["incomplete"] and not args.allow_incomplete):
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
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--expected-tasks", type=Path, help="preselected domain -> task IDs manifest")
    p.add_argument(
        "--prepare-expected-tasks",
        type=Path,
        help="write offline selection manifest using the pinned tau2 interpreter, then exit",
    )
    p.add_argument("--tau2-python", help="exact pinned tau2 venv interpreter for the guarded child")
    for name in (
        "budget-policy",
        "budget-contexts",
        "budget-contexts-sha256",
        "budget-ledger",
        "budget-scope",
    ):
        p.add_argument(
            "--" + name, default=os.environ.get("TAU2_" + name.upper().replace("-", "_"))
        )
    args = p.parse_args()
    args.enforce_budget = True

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    if (
        args.num_trials <= 0
        or args.num_tasks < 0
        or not domains
        or len(set(domains)) != len(domains)
    ):
        p.error("positive trials, nonnegative task cap and unique nonempty domains are required")
    if args.prepare_expected_tasks:
        if not args.data_dir:
            p.error("manifest preparation requires --data-dir")
        os.environ["TAU2_DATA_DIR"] = str(args.data_dir.resolve())
        data = prepare_expected_tasks(domains, args.num_tasks, args.num_trials, args.seed)
        args.prepare_expected_tasks.write_text(json.dumps(data, indent=2) + "\n")
        return EXIT_OK
    try:
        args.expected_task_ids = (
            read_expected_tasks(args.expected_tasks, domains, args.num_trials, args.seed)
            if args.expected_tasks
            else None
        )
    except (OSError, ValueError, TypeError) as error:
        p.error(f"invalid expected task coverage: {error}")
    if not args.expected_task_ids and not (args.diff_baseline or args.diff_candidate):
        p.error("--expected-tasks is required before any paid simulation")
    if args.expected_task_ids and args.num_tasks > 0:
        if any(len(ids) > args.num_tasks for ids in args.expected_task_ids.values()):
            p.error("task cap would truncate the expected task manifest")

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

    try:
        args.budget_descriptor = prepare_budget(args)
    except (OSError, ValueError, KeyError) as error:
        p.error(f"invalid paid budget: {error}")

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
            expected_tasks=args.expected_task_ids,
            seed=args.seed,
            budget_descriptor=args.budget_descriptor,
            tau2_python=args.tau2_python,
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
            expected_tasks=args.expected_task_ids,
            seed=args.seed,
            budget_descriptor=args.budget_descriptor,
            tau2_python=args.tau2_python,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # list() forces both futures to complete and re-raises any exception.
        list(ex.map(score, (baseline, candidate)))
    return write_report_and_gate(baseline, candidate, domains, args.num_trials, args)


if __name__ == "__main__":
    raise SystemExit(main())
