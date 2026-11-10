import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tau2_run_ab", Path(__file__).resolve().parents[1] / "tau2" / "run_ab.py"
)
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules["tau2_run_ab"] = run_ab
SPEC.loader.exec_module(run_ab)


@pytest.fixture(autouse=True)
def synthetic_budget_inputs(monkeypatch, tmp_path):
    contexts = {
        "schema_version": 1,
        "models": {
            model: {
                "context_tokens": 10,
                "response_models": [model],
                "source_url": "https://developers.openai.com/offline-fixture",
            }
            for model in run_ab.budget_guard.MODELS.values()
        },
    }
    context_path = tmp_path / "provider-contexts.json"
    context_path.write_text(json.dumps(contexts))
    digest = hashlib.sha256(context_path.read_bytes()).hexdigest()
    policy_path = tmp_path / "budget-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "scope_id": "offline-scorer",
                "cap_nusd": 100000,
                "calls": 10,
                "calls_per_arm": 10,
                "models": {
                    "openai/" + model: {
                        "input_per_million": "1",
                        "output_per_million": "1",
                        "input_limit": 10,
                        "output_limit": 2,
                        "service_tier": "default",
                    }
                    for model in run_ab.budget_guard.MODELS.values()
                },
            }
        )
    )
    for key, value in dict(
        BUDGET_SCOPE="offline-scorer",
        BUDGET_POLICY=policy_path,
        BUDGET_CONTEXTS=context_path,
        BUDGET_CONTEXTS_SHA256=digest,
        BUDGET_LEDGER=tmp_path / "shared-budget.db",
    ).items():
        monkeypatch.setenv("TAU2_" + key, str(value))
    policy, _ = run_ab.budget_guard.load_policy(policy_path, context_path, digest)
    run_ab.budget_guard.Ledger(tmp_path / "shared-budget.db", policy)


def test_passk_all_pass():
    # 2 successes out of 2 trials, k=2 -> 1.0
    assert run_ab.passk(2, 2, 2) == 1.0


def test_passk_one_fail():
    # 1 success out of 2 trials, k=2 -> C(1,2)/C(2,2) = 0
    assert run_ab.passk(1, 2, 2) == 0.0


def test_passk_pass1_is_success_rate():
    # k=1 -> c/n
    assert run_ab.passk(1, 2, 1) == 0.5


def test_passk_k_gt_trials_is_zero():
    assert run_ab.passk(1, 1, 2) == 0.0


def test_domain_scores_groups_by_task():
    results = [
        {"task_id": "t1", "reward": 1.0},
        {"task_id": "t1", "reward": 1.0},
        {"task_id": "t2", "reward": 1.0},
        {"task_id": "t2", "reward": 0.0},
    ]
    scores = run_ab.domain_scores(results, k=2)
    assert scores["pass1"] == 0.75  # (1+1+1+0)/4
    assert scores["passk"] == 0.5  # t1 passes^2, t2 does not -> 0.5


def test_build_report_delta_and_overall():
    base = run_ab.Arm(name="vllm", base_url="u")
    cand = run_ab.Arm(name="smg", base_url="u")
    base.scores = {"retail": {"pass1": 0.80, "passk": 0.60}}
    cand.scores = {"retail": {"pass1": 0.82, "passk": 0.62}}
    md, payload = run_ab.build_report(base, cand, ["retail"], k=2)
    assert "retail" in md and "overall" in md
    assert payload["overall"]["passk"]["delta"] == pytest.approx(0.02)


def test_domain_scores_reports_sample_counts():
    results = [
        {"task_id": "t1", "reward": 1.0},
        {"task_id": "t1", "reward": 0.0},
        {"task_id": "t2", "reward": 1.0},
    ]
    scores = run_ab.domain_scores(results, k=2)
    assert scores["n_tasks"] == 2
    assert scores["n_sims"] == 3


def test_build_report_shows_sample_counts_and_no_duplicate_title():
    base = run_ab.Arm(name="vllm", base_url="u")
    cand = run_ab.Arm(name="smg", base_url="u")
    base.scores = {
        "retail": {"pass1": 0.80, "passk": 0.60, "n_tasks": 114, "n_sims": 228},
        "airline": {"pass1": 0.74, "passk": 0.64, "n_tasks": 50, "n_sims": 100},
    }
    cand.scores = {"retail": {"pass1": 0.72, "passk": 0.58, "n_tasks": 114, "n_sims": 228}}
    md, payload = run_ab.build_report(base, cand, ["retail", "airline"], k=2)
    # per-domain sample counts render per arm (baseline/candidate), "—" when missing
    assert "228/228" in md  # retail: both arms scored
    assert "100/—" in md  # airline: candidate arm missing -> visibly asymmetric
    # de-duped: build_report no longer emits its own "τ²-bench A/B" heading (the
    # workflow summary step owns that title); arms are still identified.
    assert "τ²-bench A/B" not in md
    assert "candidate" in md and "baseline" in md
    # JSON carries the counts, and overall N sums only over domains present per arm
    assert payload["per_domain"][0]["n"]["baseline"]["sims"] == 228
    assert payload["overall"]["n"]["baseline"] == 328  # retail 228 + airline 100
    assert payload["overall"]["n"]["candidate"] == 228  # retail only


def test_build_report_dedups_columns_at_k1():
    base = run_ab.Arm(name="vllm", base_url="u")
    cand = run_ab.Arm(name="smg", base_url="u")
    base.scores = {"retail": {"pass1": 0.6667, "passk": 0.6667}}
    cand.scores = {"retail": {"pass1": 1.0, "passk": 1.0}}
    # k=1: pass^k == pass^1, so only the pass^1 triple renders (no duplicate).
    md1, _ = run_ab.build_report(base, cand, ["retail"], k=1)
    assert "pass^2" not in md1
    assert md1.count("pass^1") == 2  # once per arm in the header, not four times
    # k>1: both pass^1 and pass^k columns render.
    md2, _ = run_ab.build_report(base, cand, ["retail"], k=2)
    assert "pass^2" in md2


def test_domain_scores_on_real_fixture():
    fixture = Path(__file__).resolve().parent / "fixtures" / "tau2_results_sample.json"
    raw = json.loads(fixture.read_text())
    results = run_ab.load_results(raw)  # schema validated live against tau2 1.0.0 output
    # Fixture: 3 tasks x 2 trials, mixed pass/fail -> pass1 = 3/6, and at k=2 one
    # task passes both trials -> pass^2 = 1/3. Pinned so a grouping/schema regression
    # (e.g. swapping pass1/passk, or reading the wrong reward key) actually fails.
    scores = run_ab.domain_scores(results, k=2)
    assert scores["pass1"] == pytest.approx(0.5)
    assert scores["passk"] == pytest.approx(1 / 3)


def _run_tau2(arm, tmp_path, **overrides):
    """Invoke run_tau2 with sensible defaults; overrides tune the knobs under test."""
    kwargs = dict(
        tau2="tau2",
        agent_model="Qwen/Qwen3.6-27B",
        domain="airline",
        num_trials=1,
        num_tasks=0,
        max_concurrency=0,
        user_llm="gpt-5.2",
        data_dir=tmp_path,
        request_timeout=0,
        run_timeout=0,
        budget_descriptor=run_ab.prepare_budget(),
    )
    kwargs.update(overrides)
    run_ab.run_tau2(arm, **kwargs)


def test_run_tau2_injects_request_timeout_into_agent_args(monkeypatch, tmp_path):
    # A per-request litellm timeout caps a single generate() so a degenerate task
    # (Qwen3.6-27B airline/44.1 ran 77 min on one request) fails fast instead of
    # dragging the whole leg into the 6h job limit.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="retail", request_timeout=300, run_timeout=5400)

    i = captured["cmd"].index("--agent-llm-args")
    agent_args = json.loads(captured["cmd"][i + 1])
    assert agent_args["timeout"] == 300
    assert captured["timeout"] == 5400  # per-domain subprocess wall-clock cap


def test_run_tau2_omits_timeout_when_disabled(monkeypatch, tmp_path):
    # request_timeout/run_timeout == 0 preserves the pre-fix behavior for ad-hoc runs.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="retail", request_timeout=0, run_timeout=0)

    i = captured["cmd"].index("--agent-llm-args")
    agent_args = json.loads(captured["cmd"][i + 1])
    assert "timeout" not in agent_args
    assert captured["timeout"] is None


def test_run_tau2_survives_domain_timeout(monkeypatch, tmp_path):
    # A hung tau2 must NOT propagate or block: the domain is left unscored (renders
    # "—") so one wedged simulation can't consume the entire CI budget.
    def fake_run(cmd, **kwargs):
        raise run_ab.subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="airline", request_timeout=300, run_timeout=5400)

    assert "airline" not in arm.scores


def test_save_load_scores_roundtrip(tmp_path):
    # Sequential mode persists one arm's per-domain scores, then --diff reloads them.
    arm = run_ab.Arm(
        name="vllm", base_url="http://x:1", scores={"retail": {"pass1": 0.7, "passk": 0.5}}
    )
    path = tmp_path / "vllm.json"
    arm.budget = run_ab.prepare_budget()
    run_ab.save_scores(arm, path)
    back = run_ab.load_scores(path)
    assert back.name == "vllm"
    assert back.scores == {"retail": {"pass1": 0.7, "passk": 0.5}}


def test_domain_scores_rejects_no_executed_simulations():
    with pytest.raises(ValueError, match="no simulations"):
        run_ab.domain_scores([], k=2)


@pytest.mark.parametrize("raw", [{}, {"simulations": []}])
def test_concurrent_empty_arms_are_incomplete(monkeypatch, tmp_path, raw):
    # Exercise the real concurrent scorer/report gate, replacing only the paid CLI.
    def fake_run(command, **kwargs):
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(raw))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    report = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://unused.invalid",
            "--candidate",
            "smg=http://unused.invalid",
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            "--data-dir",
            str(tmp_path),
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    result = json.loads(report.read_text())
    assert set(result["incomplete"]) == {"vllm", "smg"}
    assert result["overall"]["passk"]["delta"] is None


@pytest.mark.parametrize("counts", [{}, {"n_tasks": 0, "n_sims": 0}])
def test_sequential_empty_saved_scores_are_incomplete(monkeypatch, tmp_path, counts):
    # Legacy saved 0/0 reports must not become a successful equal-score comparison.
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(
            name,
            "",
            scores={
                "retail": {
                    "pass1": 0.0,
                    "passk": 0.0,
                    **counts,
                }
            },
        )
        path = tmp_path / f"{name}.json"
        arm.budget = run_ab.prepare_budget()
        run_ab.save_scores(arm, path)
        files.append(path)
    report = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    assert set(json.loads(report.read_text())["incomplete"]) == {"vllm", "smg"}


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_executed_zero_rewards_remain_a_valid_comparison(monkeypatch, tmp_path, allow_incomplete):
    scores = run_ab.domain_scores(
        [
            {"task_id": "failed-task", "trial": 0, "reward": 0.0},
            {"task_id": "failed-task", "trial": 1, "reward": 0.0},
        ],
        k=2,
    )
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(name, "", scores={"retail": scores})
        path = tmp_path / f"{name}.json"
        arm.budget = run_ab.prepare_budget()
        run_ab.save_scores(arm, path)
        files.append(path)
    report = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            *(["--allow-incomplete"] if allow_incomplete else []),
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["failed-task"])),
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_OK
    payload = json.loads(report.read_text())
    assert payload["incomplete"] == {}
    assert payload["overall"]["passk"]["delta"] == 0.0
    assert payload["overall"]["n"] == {"baseline": 2, "candidate": 2}


def _expected_manifest(tmp_path, ids):
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "domains": {"retail": ids},
                "num_trials": 2,
                "seed": 300,
            }
        )
    )
    return path


def _rows(ids=("task",), trials=(0, 1), reward=0.0):
    return [{"task_id": task, "trial": trial, "reward": reward} for task in ids for trial in trials]


def _coverage_gate(tmp_path, baseline_rows, candidate_rows, expected):
    baseline = run_ab.Arm("vllm", "", scores={"retail": run_ab.domain_scores(baseline_rows, 2)})
    candidate = run_ab.Arm("smg", "", scores={"retail": run_ab.domain_scores(candidate_rows, 2)})
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "comparison.json",
        tolerance=0.02,
        allow_incomplete=False,
        expected_task_ids={"retail": expected},
    )
    return run_ab.write_report_and_gate(baseline, candidate, ["retail"], 2, args)


@pytest.mark.parametrize(
    "candidate",
    [
        _rows(ids=("different-task",)),
        _rows(trials=(0,)),
        _rows(trials=(0, 0)),
        _rows(trials=(0, 1, 2)),
        _rows(trials=(0, None)),
    ],
)
def test_positive_scores_with_incomplete_identity_fail(tmp_path, candidate):
    assert _coverage_gate(tmp_path, _rows(), candidate, ["task"]) == run_ab.EXIT_INCOMPLETE


def test_matching_partial_arms_still_fail_expected_dataset_coverage(tmp_path):
    assert (
        _coverage_gate(tmp_path, _rows(), _rows(), ["task", "missing-task"])
        == run_ab.EXIT_INCOMPLETE
    )


def test_exact_task_trials_with_zero_rewards_remain_valid(tmp_path):
    assert _coverage_gate(tmp_path, _rows(), _rows(), ["task"]) == run_ab.EXIT_OK


def test_legacy_positive_saved_scores_cannot_prove_identity(monkeypatch, tmp_path):
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(
            name,
            "",
            scores={
                "retail": {
                    "pass1": 0.0,
                    "passk": 0.0,
                    "n_tasks": 1,
                    "n_sims": 2,
                }
            },
        )
        path = tmp_path / f"{name}.json"
        arm.budget = run_ab.prepare_budget()
        run_ab.save_scores(arm, path)
        files.append(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE


def test_missing_expected_manifest_denies_before_any_simulation(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("paid CLI must not run without expected coverage")

    monkeypatch.setattr(run_ab.subprocess, "run", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://unused.invalid",
            "--candidate",
            "smg=http://unused.invalid",
            "--domains",
            "retail",
            "--data-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_ab.main()
    assert error.value.code == run_ab.EXIT_INCOMPLETE


def test_both_live_arms_bound_to_same_selected_ids_and_seed(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        save_to = command[command.index("--save-to") + 1]
        folder = tmp_path / "simulations" / save_to
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(
            json.dumps(
                {
                    "simulations": [
                        {"task_id": "task", "trial": trial, "reward_info": {"reward": 0.0}}
                        for trial in (0, 1)
                    ]
                }
            )
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://unused.invalid",
            "--candidate",
            "smg=http://unused.invalid",
            "--domains",
            "retail",
            "--data-dir",
            str(tmp_path),
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_OK
    assert len(calls) == 2
    for command in calls:
        assert command[command.index("--task-ids") + 1] == "task"
        assert command[command.index("--seed") + 1] == "300"
        assert command[command.index("--task-split-name") + 1] == "base"


@pytest.mark.parametrize("cap, selected", [(0, ["z", "b", "a"]), (2, ["z"]), (20, ["z", "b", "a"])])
def test_manifest_uses_actual_pinned_loader_selection_without_guessing_counts(
    monkeypatch, tmp_path, cap, selected
):
    # The upstream loader applies ordered split/cap then agent filtering. Its
    # independent selected IDs include a filtered cap=2 case and cap>dataset.
    def fake_loader(config):
        assert config.domain == "retail" and config.task_split_name == "base"
        assert config.agent == "llm_agent" and config.num_tasks == (cap or None)
        assert config.seed == 300 and config.num_trials == 2
        return [types.SimpleNamespace(id=task) for task in selected]

    simulation = types.ModuleType("tau2.data_model.simulation")
    simulation.TextRunConfig = types.SimpleNamespace
    batch = types.ModuleType("tau2.runner.batch")
    batch._load_run_tasks = fake_loader
    monkeypatch.setitem(sys.modules, "tau2.data_model.simulation", simulation)
    monkeypatch.setitem(sys.modules, "tau2.runner.batch", batch)
    output = tmp_path / "expected.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--prepare-expected-tasks",
            str(output),
            "--domains",
            "retail",
            "--num-tasks",
            str(cap),
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_OK
    assert run_ab.read_expected_tasks(output, ["retail"], 2, 300) == {"retail": selected}


def test_sequential_persists_task_identity_and_rejects_different_tasks(monkeypatch, tmp_path):
    files = []
    for name, ids in (("vllm", ("task",)), ("smg", ("different-task",))):
        arm = run_ab.Arm(name, "", scores={"retail": run_ab.domain_scores(_rows(ids=ids), 2)})
        path = tmp_path / f"{name}.json"
        arm.budget = run_ab.prepare_budget()
        run_ab.save_scores(arm, path)
        files.append(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE


def test_missing_dataset_stops_manifest_preparation_before_paid_calls(monkeypatch, tmp_path):
    def missing_dataset(config):
        raise FileNotFoundError("pinned task dataset is absent")

    simulation = types.ModuleType("tau2.data_model.simulation")
    simulation.TextRunConfig = types.SimpleNamespace
    batch = types.ModuleType("tau2.runner.batch")
    batch._load_run_tasks = missing_dataset
    monkeypatch.setitem(sys.modules, "tau2.data_model.simulation", simulation)
    monkeypatch.setitem(sys.modules, "tau2.runner.batch", batch)

    def unexpected(*args, **kwargs):
        pytest.fail("no simulation may run while selecting expected tasks")

    monkeypatch.setattr(run_ab.subprocess, "run", unexpected)
    path = tmp_path / "expected.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--prepare-expected-tasks",
            str(path),
            "--domains",
            "retail",
            "--data-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileNotFoundError, match="dataset is absent"):
        run_ab.main()
    assert not path.exists()


def _raw_trials(trials=(0, 1), reward=0.0):
    return {
        "simulations": [
            {"task_id": "task", "trial": trial, "reward_info": {"reward": reward}}
            for trial in trials
        ]
    }


@pytest.mark.parametrize(
    "returncode, current, timed_out, failure_kind",
    [
        (7, None, False, "nonzero_exit"),
        (7, _raw_trials((0,)), False, "nonzero_exit"),
        (0, None, False, "missing_results"),
        (0, _raw_trials((0,)), False, "incomplete_results"),
        (0, {}, False, "invalid_results"),
        (0, _raw_trials(), True, "timeout"),
    ],
)
def test_invocation_failure_cannot_score_exact_matching_old_results(
    monkeypatch, tmp_path, returncode, current, timed_out, failure_kind
):
    old = tmp_path / "simulations" / "ab_vllm_retail" / "results.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(_raw_trials()))
    old_bytes = old.read_bytes()
    scores = run_ab.domain_scores(run_ab.load_results(_raw_trials()), k=2)
    arm = run_ab.Arm("vllm", "http://unused.invalid", scores={"retail": scores})
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        save_to = command[command.index("--save-to") + 1]
        if current is not None:
            folder = tmp_path / "simulations" / save_to
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "results.json").write_text(json.dumps(current))
        if timed_out:
            raise run_ab.subprocess.TimeoutExpired(command, kwargs.get("timeout"))
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    _run_tau2(
        arm,
        tmp_path,
        domain="retail",
        num_trials=2,
        expected_task_ids=["task"],
        run_timeout=10,
    )
    assert "retail" not in arm.scores
    invocation = arm.invocations["retail"]
    assert invocation["arm"] == "vllm" and invocation["domain"] == "retail"
    assert invocation["status"] == "failed" and invocation["failure_kind"] == failure_kind
    assert invocation["exit_code"] == (None if timed_out else returncode)
    assert "api" not in invocation["failure_kind"]  # CLI exit alone does not identify its cause.
    assert old.read_bytes() == old_bytes
    assert len(commands) == 1
    if timed_out:
        assert arm.timed_out == ["retail"]

    # The independent expected manifest is exactly the same as the OLD file.
    # Current task/trial coverage alone must not make this a valid comparison.
    candidate = run_ab.Arm("smg", "", scores={"retail": scores})
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "comparison.json",
        tolerance=0.02,
        allow_incomplete=False,
        expected_task_ids={"retail": ["task"]},
    )
    assert (
        run_ab.write_report_and_gate(arm, candidate, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE
    )
    payload = json.loads(args.json_out.read_text())
    assert payload["baseline"]["invocations"]["retail"] == invocation
    assert failure_kind in payload["incomplete"]["vllm"]


def test_invocation_fresh_zero_results_preserved_and_each_output_unique(monkeypatch, tmp_path):
    old = tmp_path / "simulations" / "ab_vllm_retail" / "results.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(_raw_trials(reward=1.0)))
    outputs = []

    def fake_run(command, **kwargs):
        output = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        assert output != old.parent and not output.exists()
        outputs.append(output)
        output.mkdir(parents=True)
        (output / "results.json").write_text(json.dumps(_raw_trials()))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    for _ in range(2):
        _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
        assert arm.scores["retail"]["pass1"] == arm.scores["retail"]["passk"] == 0.0
        assert arm.invocations["retail"]["status"] == "completed"
        assert arm.invocations["retail"]["exit_code"] == 0
        assert Path(arm.invocations["retail"]["results_path"]) == outputs[-1] / "results.json"
    assert outputs[0] != outputs[1]
    assert json.loads(old.read_text()) == _raw_trials(reward=1.0)


def test_invocation_failure_provenance_survives_saved_diff(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return types.SimpleNamespace(returncode=13)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    # Even an old matching score coexisting with explicit new failure must not pass.
    arm.scores["retail"] = run_ab.domain_scores(run_ab.load_results(_raw_trials()), 2)
    path = tmp_path / "baseline.json"
    arm.budget = run_ab.prepare_budget()
    run_ab.save_scores(arm, path)
    restored = run_ab.load_scores(path)
    assert restored.invocations == arm.invocations
    candidate = run_ab.Arm("smg", "", scores=arm.scores)
    candidate_path = tmp_path / "candidate.json"
    candidate.budget = arm.budget
    run_ab.save_scores(candidate, candidate_path)
    report = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--diff-baseline",
            str(path),
            "--diff-candidate",
            str(candidate_path),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    payload = json.loads(report.read_text())
    assert payload["baseline"]["invocations"] == arm.invocations
    assert "nonzero_exit" in payload["incomplete"]["vllm"]


def test_invocation_existing_namespace_denies_before_cli(monkeypatch, tmp_path):
    nonce = "1" * 32
    folder = tmp_path / "simulations" / f"ab_{nonce}"
    folder.mkdir(parents=True)
    result = folder / "results.json"
    result.write_text(json.dumps(_raw_trials()))
    monkeypatch.setattr(run_ab.uuid, "uuid4", lambda: types.SimpleNamespace(hex=nonce))

    def unexpected(*args, **kwargs):
        pytest.fail("must not resume or overwrite an existing output namespace")

    monkeypatch.setattr(run_ab.subprocess, "run", unexpected)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert not arm.scores
    assert arm.invocations["retail"]["failure_kind"] == "output_exists"
    assert arm.invocations["retail"]["failure_stage"] == "preflight"
    assert json.loads(result.read_text()) == _raw_trials()


def test_invocation_launch_error_records_exception_without_provider_guess(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("missing tau2 executable")

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert not arm.scores
    failure = arm.invocations["retail"]
    assert failure["failure_kind"] == "launch_error"
    assert failure["failure_stage"] == "cli"
    assert failure["exception_type"] == "FileNotFoundError"
    assert failure["exit_code"] is None


def test_invocation_symlink_to_old_result_cannot_prove_freshness(monkeypatch, tmp_path):
    old = tmp_path / "old-results.json"
    old.write_text(json.dumps(_raw_trials()))

    def fake_run(command, **kwargs):
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        (folder / "results.json").symlink_to(old)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert not arm.scores
    assert arm.invocations["retail"]["failure_kind"] == "invalid_results"
    assert json.loads(old.read_text()) == _raw_trials()


def test_invocation_does_not_fail_on_other_domains_legitimate_pending(monkeypatch, tmp_path):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    ledger.reserve(
        "other-domain",
        "smg",
        "openai/gpt-5.2",
        10,
        2,
        num_retries=0,
        output_limit_verified=True,
        metadata={"invocation_id": "other"},
    )

    def fake_cli(command, **kwargs):
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(_raw_trials()))
        assert "-I" in command and command[2].endswith("guarded_cli.py")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_cli)
    arm = run_ab.Arm("vllm", "http://127.0.0.1:8000")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert arm.scores["retail"]["pass1"] == 0.0
    candidate = run_ab.Arm("smg", "", scores=arm.scores, budget=descriptor)
    with pytest.raises(run_ab.budget_guard.BudgetError, match="pending"):
        run_ab.final_budget(arm, candidate)
    ledger.finish("other-domain", 1, 1)
    assert run_ab.final_budget(arm, candidate)["pending"] == 0


@pytest.mark.parametrize("allow_incomplete", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "deleted",
        "policy_changed",
        "scope_mismatch",
        "halted",
        "unknown",
        "pending",
        "missing_descriptor",
    ],
)
def test_saved_diff_reopens_existing_budget_and_never_resets(
    monkeypatch, tmp_path, change, allow_incomplete
):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(
            name,
            "",
            scores={"retail": run_ab.domain_scores(run_ab.load_results(_raw_trials()), 2)},
            budget=dict(descriptor),
        )
        if change == "scope_mismatch" and name == "smg":
            arm.budget["scope_id"] = "other-scope"
        if change == "missing_descriptor" and name == "smg":
            arm.budget = None
        path = tmp_path / (name + ".json")
        run_ab.save_scores(arm, path)
        files.append(path)
    if change == "deleted":
        Path(descriptor["ledger_path"]).unlink()
    elif change == "policy_changed":
        path = Path(descriptor["policy_path"])
        data = json.loads(path.read_text())
        data["cap_nusd"] += 1
        path.write_text(json.dumps(data))
    elif change == "halted":
        ledger.halt()
    elif change == "pending":
        ledger.reserve(
            "pending", "smg", "openai/gpt-5.2", 10, 2, num_retries=0, output_limit_verified=True
        )
    elif change == "unknown":
        ledger.reserve(
            "unknown", "vllm", "openai/gpt-5.2", 10, 2, num_retries=0, output_limit_verified=True
        )
        with pytest.raises(run_ab.budget_guard.BudgetError):
            ledger.finish("unknown")
    report = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            *(["--allow-incomplete"] if allow_incomplete else []),
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    assert "budget" in json.loads(report.read_text())["incomplete"]
    if change == "deleted":
        assert not Path(descriptor["ledger_path"]).exists()


def test_missing_paid_policy_denies_before_cli(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU2_BUDGET_POLICY")

    def unexpected(*args, **kwargs):
        pytest.fail("missing approved policy must never launch paid CLI")

    monkeypatch.setattr(run_ab.subprocess, "run", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://127.0.0.1:8000",
            "--candidate",
            "smg=http://127.0.0.1:8001",
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            "--data-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_ab.main()
    assert error.value.code == run_ab.EXIT_INCOMPLETE


def test_sequential_live_arm_cannot_recreate_deleted_shared_ledger(tmp_path):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    ledger.reserve(
        "first-arm", "vllm", "openai/gpt-5.2", 10, 2, num_retries=0, output_limit_verified=True
    )
    Path(descriptor["ledger_path"]).unlink()
    with pytest.raises(run_ab.budget_guard.BudgetError, match="Existing shared ledger"):
        run_ab.prepare_budget()
    assert not Path(descriptor["ledger_path"]).exists()


@pytest.mark.parametrize("state", ["halted", "pending", "deleted"])
@pytest.mark.parametrize("regression", [False, True])
def test_concurrent_budget_errors_are_hard_gates_with_allow_incomplete(
    monkeypatch, tmp_path, state, regression
):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)

    def score(arm, **kwargs):
        assert kwargs["budget_descriptor"] == descriptor
        arm.budget = dict(descriptor)
        arm.scores["retail"] = run_ab.domain_scores(
            _rows(reward=1.0 if regression and arm.name == "vllm" else 0.0), 2
        )
        if arm.name == "smg":
            if state == "halted":
                ledger.halt()
            elif state == "pending":
                ledger.reserve(
                    "pending",
                    "smg",
                    "openai/gpt-5.2",
                    10,
                    2,
                    num_retries=0,
                    output_limit_verified=True,
                )
            else:
                Path(descriptor["ledger_path"]).unlink()

    monkeypatch.setattr(run_ab, "score_arm", score)
    report = tmp_path / "concurrent.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://127.0.0.1:8000",
            "--candidate",
            "smg=http://127.0.0.1:8001",
            "--domains",
            "retail",
            "--data-dir",
            str(tmp_path),
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            "--allow-incomplete",
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    payload = json.loads(report.read_text())
    assert set(payload["incomplete"]) == {"budget"}
    assert payload["overall"]["passk"]["delta"] == (-1.0 if regression else 0.0)
    if state == "deleted":
        assert not Path(descriptor["ledger_path"]).exists()


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_valid_budget_keeps_quality_only_diagnostic_flag(monkeypatch, tmp_path, allow_incomplete):
    descriptor = run_ab.prepare_budget()
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(name, "", budget=dict(descriptor))
        if name == "vllm":
            arm.scores["retail"] = run_ab.domain_scores(_rows(), 2)
        path = tmp_path / (name + ".json")
        run_ab.save_scores(arm, path)
        files.append(path)
    report = tmp_path / "quality-only.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            "--expected-tasks",
            str(_expected_manifest(tmp_path, ["task"])),
            *(["--allow-incomplete"] if allow_incomplete else []),
            "--json-out",
            str(report),
        ],
    )
    assert run_ab.main() == (run_ab.EXIT_OK if allow_incomplete else run_ab.EXIT_INCOMPLETE)
    payload = json.loads(report.read_text())
    assert payload["incomplete"] and "budget" not in payload["incomplete"]
    assert payload["budget"]["halted"] is False
