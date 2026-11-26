import hashlib
import importlib.util
import json
import random
import sys
import threading
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
            {"task_id": "failed-task", "trial": 0, "seed": 626729, "reward": 0.0},
            {"task_id": "failed-task", "trial": 1, "seed": 373753, "reward": 0.0},
        ],
        k=2,
    )
    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(name, "", scores={"retail": scores})
        path = tmp_path / f"{name}.json"
        arm.budget = run_ab.prepare_budget()
        _attach_provenance(tmp_path, arm, _rows(ids=("failed-task",)))
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


def _fixture_trial_seed(trial):
    # Independent reproduction of pinned batch.py's seeded randint sequence.
    # Invalid trial fixtures remain invalid; do not manufacture their identity.
    if type(trial) is not int or trial < 0:
        return None
    rng = random.Random(300)
    return [rng.randint(0, 1000000) for _ in range(trial + 1)][-1]


def _rows(ids=("task",), trials=(0, 1), reward=0.0):
    return [
        {"task_id": task, "trial": trial, "seed": _fixture_trial_seed(trial), "reward": reward}
        for task in ids
        for trial in trials
    ]


def _attach_provenance(tmp_path, arm, rows, configured_seed=300):
    identity = run_ab.uuid.uuid4().hex
    result = tmp_path / ("ab_" + identity) / "results.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": row["task_id"],
                        "trial": row["trial"],
                        "seed": row["seed"],
                        "reward_info": {"reward": row["reward"]},
                    }
                    for row in rows
                ]
            }
        )
    )
    arm.invocations["retail"] = dict(
        invocation_id=identity,
        arm=arm.name,
        domain="retail",
        configured_seed=configured_seed,
        trial_seeds=run_ab.trial_seeds(configured_seed, 2),
        status="completed",
        exit_code=0,
        results_path=str(result),
        results_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
    )
    arm.budget = arm.budget or run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(arm.budget)
    ledger.begin_invocation(identity, arm.name, "retail")
    ledger.complete_invocation(
        identity, arm.name, "retail", arm.invocations["retail"]["results_sha256"]
    )


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
                        {
                            "task_id": "task",
                            "trial": trial,
                            "seed": _fixture_trial_seed(trial),
                            "reward_info": {"reward": 0.0},
                        }
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
            {
                "task_id": "task",
                "trial": trial,
                "seed": _fixture_trial_seed(trial),
                "reward_info": {"reward": reward},
            }
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
    both_completed = threading.Barrier(2, timeout=5)

    def score(arm, **kwargs):
        assert kwargs["budget_descriptor"] == descriptor
        arm.budget = dict(descriptor)
        arm.scores["retail"] = run_ab.domain_scores(
            _rows(reward=1.0 if regression and arm.name == "vllm" else 0.0), 2
        )
        _attach_provenance(
            tmp_path, arm, _rows(reward=1.0 if regression and arm.name == "vllm" else 0.0)
        )
        # This test injects a final-gate fault after both real completions.
        both_completed.wait()
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
    assert set(payload["incomplete"]) == (
        {"budget", "vllm", "smg"} if state == "deleted" else {"budget"}
    )
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
            _attach_provenance(tmp_path, arm, _rows())
        else:
            # Quality-only means an actual registered child that returned zero
            # without scores, not an arm with no invocation evidence at all.
            with monkeypatch.context() as patch:
                patch.setattr(
                    run_ab.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0)
                )
                _run_tau2(
                    arm,
                    tmp_path,
                    domain="retail",
                    num_trials=2,
                    expected_task_ids=["task"],
                    budget_descriptor=descriptor,
                )
            assert arm.invocations["retail"]["failure_kind"] == "missing_results"
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


def _infra_placeholder(trial=0):
    # Pinned run_with_retry(max_retries=0) writes this schema, not a zero reward.
    return {
        "task_id": "task",
        "trial": trial,
        "seed": 300 + trial,
        "termination_reason": "infrastructure_error",
        "reward_info": None,
        "info": {
            "error_type": "TimeoutError",
            "error": "sensitive provider detail must not enter reports",
            "error_traceback": "private raw traceback",
            "failed_after_attempts": 1,
        },
    }


def test_known_infra_null_preserves_unscored_and_scored_identities():
    zero = {"task_id": "task", "trial": 1, "seed": 301, "reward_info": {"reward": 0.0}}
    with pytest.raises(run_ab.InfrastructureResults) as failure:
        run_ab.load_results({"simulations": [_infra_placeholder(), zero]})
    assert failure.value.unscored == [
        {
            "task_id": "task",
            "trial": 0,
            "seed": 300,
            "termination_reason": "infrastructure_error",
            "error_type": "TimeoutError",
        }
    ]
    assert failure.value.scored == [{"task_id": "task", "trial": 1, "seed": 301, "reward": 0.0}]
    assert "sensitive" not in str(failure.value.unscored)
    assert "traceback" not in str(failure.value.unscored)


@pytest.mark.parametrize(
    "row",
    [
        {"task_id": "task", "trial": 0, "reward_info": None},
        {"task_id": "task", "trial": 0, "termination_reason": "unknown", "reward_info": None},
        {"task_id": "task", "trial": 0},
        {**_infra_placeholder(), "reward_info": {"reward": 0.0}},
        {**_infra_placeholder(), "trial": None},
        {**_infra_placeholder(), "task_id": None},
        {**_infra_placeholder(), "seed": True},
        {"task_id": "task", "trial": 0, "reward_info": {"reward": True}},
        {"task_id": "task", "trial": 0, "reward_info": {"reward": float("nan")}},
    ],
)
def test_unknown_or_contradictory_null_contract_is_not_normalized(row):
    with pytest.raises(ValueError) as failure:
        run_ab.load_results({"simulations": [_infra_placeholder(), row]})
    assert not isinstance(failure.value, run_ab.InfrastructureResults)


def test_genuine_zero_reward_still_loads_without_infra_relabeling():
    assert run_ab.load_results(_raw_trials((0,), reward=0.0)) == [
        {"task_id": "task", "trial": 0, "seed": 626729, "reward": 0.0}
    ]


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_infra_results_report_identities_rawhash_and_force_incomplete(
    monkeypatch, tmp_path, allow_incomplete
):
    raw = _raw_trials((1,), reward=0.0)
    raw["simulations"].insert(0, _infra_placeholder())
    written = []

    def fake_run(command, **kwargs):
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        body = json.dumps(raw).encode()
        (folder / "results.json").write_bytes(body)
        written.append(body)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    invocation = arm.invocations["retail"]
    assert not arm.scores
    assert invocation["failure_kind"] == "infrastructure_error_results"
    assert invocation["results_sha256"] == hashlib.sha256(written[0]).hexdigest()
    assert invocation["n_infrastructure_errors"] == 1
    assert invocation["unscored_cases"][0]["trial"] == 0
    assert invocation["unscored_cases"][0]["seed"] == 300
    assert invocation["scored_case_diagnostics"] == [
        {"task_id": "task", "trial": 1, "seed": 373753, "reward": 0.0}
    ]
    candidate = run_ab.Arm("smg", "", scores={"retail": run_ab.domain_scores(_rows(), 2)})
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "infra-comparison.json",
        tolerance=0.02,
        allow_incomplete=allow_incomplete,
        expected_task_ids={"retail": ["task"]},
    )
    assert (
        run_ab.write_report_and_gate(arm, candidate, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE
    )
    payload = json.loads(args.json_out.read_text())
    assert payload["overall"]["passk"]["delta"] is None
    assert payload["baseline"]["invocations"]["retail"] == invocation
    assert "infrastructure_error_results" in payload["incomplete"]["vllm"]


@pytest.mark.parametrize("gate", ["nonzero", "pending", "unknown", "halted"])
def test_infra_placeholder_does_not_bypass_existing_paid_or_process_gate(
    monkeypatch, tmp_path, gate
):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    if gate in ("pending", "unknown"):
        proof = types.SimpleNamespace(
            summary=ledger.summary,
            begin_invocation=ledger.begin_invocation,
            halt=ledger.halt,
            audit=lambda invocation: {
                "pending": int(gate == "pending"),
                "unknown": int(gate == "unknown"),
            },
        )
        monkeypatch.setattr(run_ab.budget_guard, "read_descriptor", lambda _: proof)
    elif gate == "halted":
        ledger.halt()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps({"simulations": [_infra_placeholder()]}))
        return types.SimpleNamespace(returncode=2 if gate == "nonzero" else 0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", budget_descriptor=descriptor)
    invocation = arm.invocations["retail"]
    expected = (
        "invalid_budget"
        if gate == "halted"
        else "nonzero_exit"
        if gate == "nonzero"
        else "invocation_budget_incomplete"
    )
    assert invocation["failure_kind"] == expected
    assert "unscored_cases" not in invocation and "results_sha256" not in invocation
    assert not arm.scores
    assert len(calls) == (0 if gate == "halted" else 1)


@pytest.mark.parametrize("infra_first", [True, False])
def test_parent98_regression_mixed_invalid_never_waives_unscored(
    monkeypatch, tmp_path, infra_first
):
    rows = [
        _infra_placeholder(),
        {"task_id": "bad", "trial": 1, "seed": 301, "reward_info": {"reward": True}},
    ]
    if not infra_first:
        rows.reverse()
    raw = {"simulations": rows}

    def fake_run(command, **kwargs):
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(raw))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        expected_task_ids={"retail": ["task"]},
    )
    assert run_ab.write_report_and_gate(arm, run_ab.Arm("smg", ""), ["retail"], 2, args) == 2
    invocation = arm.invocations["retail"]
    assert invocation["failure_kind"] == "invalid_results"
    assert invocation["unscored_cases"][0]["task_id"] == "task"
    assert invocation["invalid_rows"] == [
        {"row_index": 1 if infra_first else 0, "kind": "malformed_reward"}
    ]
    assert not arm.scores


def test_parent98_regression_saved_scores_reject_different_trial_seed(tmp_path):
    import random

    def rows(configured_seed):
        rng = random.Random(configured_seed)
        return [
            {"task_id": "task", "trial": trial, "seed": rng.randint(0, 1000000), "reward": 0.0}
            for trial in range(2)
        ]

    assert _coverage_gate(tmp_path, rows(300), rows(301), ["task"]) == 2
    assert _coverage_gate(tmp_path, rows(300), rows(300), ["task"]) == 0


@pytest.mark.parametrize("allow_incomplete", [False, True])
@pytest.mark.parametrize("candidate_seed", [300, 301])
def test_parent98_seed_identity_survives_saved_diff(
    monkeypatch, tmp_path, candidate_seed, allow_incomplete
):
    files = []
    for name, configured_seed in (("vllm", 300), ("smg", candidate_seed)):
        rng = random.Random(configured_seed)
        rows = [
            {"task_id": "task", "trial": i, "seed": rng.randint(0, 1000000), "reward": 0.0}
            for i in range(2)
        ]
        arm = run_ab.Arm(name, "", scores={"retail": run_ab.domain_scores(rows, 2)})
        arm.budget = run_ab.prepare_budget()
        _attach_provenance(tmp_path, arm, rows, configured_seed)
        path = tmp_path / f"{name}-seed-scores.json"
        run_ab.save_scores(arm, path)
        files.append(path)
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
            "--seed",
            "300",
        ],
    )
    assert run_ab.main() == (0 if candidate_seed == 300 else 2)


def test_pinned_trial_seed_sequence_and_missing_saved_identity(tmp_path):
    assert run_ab.trial_seeds(300, 2) == [626729, 373753]
    assert run_ab.trial_seeds(301, 2) == [418104, 656453]
    scores = run_ab.domain_scores(_rows(), 2)
    del scores["task_trial_seeds"]
    assert (
        run_ab.coverage_problem(scores, ["task"], 2) == "missing task/trial/seed identity manifest"
    )


@pytest.mark.parametrize(
    "bad_row",
    [
        None,
        {},
        {"task_id": "bad", "reward_info": None},
        {"task_id": "bad", "reward_info": {"reward": 10**400}},
    ],
)
@pytest.mark.parametrize("infra_first", [False, True])
def test_invalid_sibling_diagnostics_keep_every_infra_obligation(bad_row, infra_first):
    rows = [_infra_placeholder(), bad_row]
    if not infra_first:
        rows.reverse()
    with pytest.raises(run_ab.InvalidResults) as error:
        run_ab.load_results({"simulations": rows})
    assert len(error.value.unscored) == 1
    assert len(error.value.invalid_rows) == 1
    assert not error.value.scored
    assert "10**400" not in str(error.value)


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_parent98_regression_missing_expected_manifest_is_fatal_cli(tmp_path, allow_incomplete):
    # Exercise the real saved --diff CLI process, including its shell exit code.
    # Both score files have otherwise valid complete task/trial/seed identities.
    import os
    import subprocess

    files = []
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(name, "", scores={"retail": run_ab.domain_scores(_rows(), 2)})
        arm.budget = run_ab.prepare_budget()
        path = tmp_path / f"{name}-without-manifest.json"
        run_ab.save_scores(arm, path)
        files.append(path)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(run_ab.__file__).resolve()),
            "--diff-baseline",
            str(files[0]),
            "--diff-candidate",
            str(files[1]),
            "--domains",
            "retail",
            *(["--allow-incomplete"] if allow_incomplete else []),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == run_ab.EXIT_INCOMPLETE, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "missing expected task manifest" in result.stderr


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_guarded_nonzero_is_never_waived(monkeypatch, tmp_path, allow_incomplete):
    monkeypatch.setattr(
        run_ab.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=2)
    )
    arms = [run_ab.Arm(name, "http://unused.invalid") for name in ("vllm", "smg")]
    for arm in arms:
        _run_tau2(arm, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=allow_incomplete,
        expected_task_ids={"retail": ["task"]},
    )
    assert run_ab.write_report_and_gate(*arms, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE


def test_duplicate_display_names_do_not_erase_missing_baseline(tmp_path):
    baseline = run_ab.Arm("same", "")
    candidate = run_ab.Arm("same", "", scores={"retail": run_ab.domain_scores(_rows(), 2)})
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=False,
        expected_task_ids={"retail": ["task"]},
    )
    assert (
        run_ab.write_report_and_gate(baseline, candidate, ["retail"], 2, args)
        == run_ab.EXIT_INCOMPLETE
    )


def _provenant_arm(tmp_path, name):
    arm = run_ab.Arm(
        name,
        "",
        scores={"retail": run_ab.domain_scores(_rows(), 2)},
        budget=run_ab.prepare_budget(),
    )
    _attach_provenance(tmp_path, arm, _rows())
    return arm


@pytest.mark.parametrize(
    "change", ["missing", "id", "digest", "configured_seed", "changed_raw", "score"]
)
def test_saved_provenance_is_required_even_for_quality_diagnostics(tmp_path, change):
    a, b = (_provenant_arm(tmp_path, name) for name in ("vllm", "smg"))
    record = a.invocations["retail"]
    if change == "missing":
        a.invocations.clear()
    elif change == "id":
        record["invocation_id"] = "other"
    elif change == "digest":
        record["results_sha256"] = "0" * 64
    elif change == "configured_seed":
        record["configured_seed"] = 301
    elif change == "changed_raw":
        Path(record["results_path"]).write_text("changed")
    else:
        a.scores["retail"]["pass1"] = 1.0
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE


def test_empty_saved_diff_requires_expected_manifest_with_allow_incomplete(tmp_path):
    descriptor = run_ab.prepare_budget()
    a, b = (run_ab.Arm(n, "", budget=descriptor) for n in ("vllm", "smg"))
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids=None,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE


def test_wrong_seed_retained_beside_malformed_sibling_is_hard_failure(monkeypatch, tmp_path):
    def fake(command, **kwargs):
        result = tmp_path / "simulations" / command[command.index("--save-to") + 1] / "results.json"
        result.parent.mkdir(parents=True)
        data = _raw_trials()
        data["simulations"][0]["seed"] = 301
        data["simulations"][1]["reward_info"]["reward"] = True
        result.write_text(json.dumps(data))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake)
    a = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(a, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert a.invocations["retail"]["seed_identity_invalid"] is True
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        expected_task_ids={"retail": ["task"]},
    )
    assert (
        run_ab.write_report_and_gate(a, run_ab.Arm("smg", ""), ["retail"], 2, args)
        == run_ab.EXIT_INCOMPLETE
    )


def test_bare_tau2_selects_its_path_environment_interpreter(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(run_ab.shutil, "which", lambda name: "/qualified/bin/tau2")

    def fake(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(returncode=2)

    monkeypatch.setattr(run_ab.subprocess, "run", fake)
    _run_tau2(
        run_ab.Arm("vllm", "http://unused.invalid"),
        tmp_path,
        domain="retail",
        num_trials=2,
        expected_task_ids=["task"],
        tau2="tau2",
    )
    assert commands[0][0] == "/qualified/bin/python"


def test_saved_arms_cannot_relabel_one_independent_completion(tmp_path):
    a, b = (_provenant_arm(tmp_path, name) for name in ("vllm", "smg"))
    b.invocations["retail"] = dict(a.invocations["retail"], arm="smg")
    b.scores = dict(a.scores)
    output = tmp_path / "report.md"
    args = types.SimpleNamespace(
        out=output,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == run_ab.EXIT_INCOMPLETE
    assert "inconsistent invocation provenance" in output.read_text()


def test_result_digest_and_scores_use_one_buffer(tmp_path, monkeypatch):
    a, b = (_provenant_arm(tmp_path, name) for name in ("vllm", "smg"))
    target = Path(a.invocations["retail"]["results_path"])
    original = Path.read_bytes
    reads = []

    def changing(path):
        if path == target:
            reads.append(True)
            if len(reads) > 1:
                return b"changed between reads"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changing)
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=False,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == run_ab.EXIT_OK
    assert len(reads) == 1


@pytest.mark.parametrize("own_pending", [True, False])
def test_timeout_halts_only_its_own_unknown_paid_work(tmp_path, monkeypatch, own_pending):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)

    def timeout(cmd, **kwargs):
        identity = cmd[cmd.index("--invocation") + 1] if own_pending else "healthy-sibling"
        ledger.reserve(
            "paid",
            "vllm",
            "openai/gpt-5.2",
            1,
            1,
            num_retries=0,
            output_limit_verified=True,
            metadata={"invocation_id": identity, "domain": "airline"},
        )
        raise run_ab.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(run_ab.subprocess, "run", timeout)
    arm = run_ab.Arm("vllm", "http://127.0.0.1:1")
    _run_tau2(arm, tmp_path, budget_descriptor=descriptor)
    assert ledger.summary()["halted"] is own_pending
    assert arm.invocations["airline"]["failure_kind"] == "timeout"


@pytest.mark.parametrize("with_sidecar", [True, False])
def test_actual_nightly_staging_keeps_fail_stop(tmp_path, monkeypatch, with_sidecar):
    import os
    import subprocess

    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/nightly-tau2.yml"
    lines = workflow.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "- name: Stage results for upload" in line)
    run = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "run: |")
    end = next(i for i in range(run + 1, len(lines)) if lines[i].startswith("      - name:"))
    script = "\n".join(line[10:] for line in lines[run + 1 : end])
    ledger = tmp_path / "ledger.db"
    ledger.write_bytes(b"owned ledger fixture")
    sidecar = tmp_path / "ledger.db.fail-stop"
    if with_sidecar:
        sidecar.write_bytes(b"")
    monkeypatch.setenv("TAU2_BUDGET_LEDGER", str(ledger))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    subprocess.run(
        ["/usr/bin/bash", "-c", script], cwd=tmp_path, env=os.environ.copy(), check=True, timeout=10
    )
    output = tmp_path / "tau2_artifact"
    assert (output / "ledger.db").read_bytes() == ledger.read_bytes()
    assert (output / sidecar.name).exists() is with_sidecar


def test_current_unhalted_completion_refusal_cannot_be_waived(tmp_path, monkeypatch):
    candidate = _provenant_arm(tmp_path, "smg")
    descriptor = candidate.budget
    ledger = run_ab.budget_guard.read_descriptor(descriptor)

    def successful_child(command, **kwargs):
        destination = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        destination.mkdir(parents=True)
        (destination / "results.json").write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "task_id": row["task_id"],
                            "trial": row["trial"],
                            "seed": row["seed"],
                            "reward_info": {"reward": row["reward"]},
                        }
                        for row in _rows()
                    ]
                }
            )
        )
        return types.SimpleNamespace(returncode=0)

    def refused(self, *args):
        raise run_ab.budget_guard.BudgetError("owned unhalted completion refusal")

    monkeypatch.setattr(run_ab.subprocess, "run", successful_child)
    monkeypatch.setattr(run_ab.budget_guard.Ledger, "complete_invocation", refused)
    baseline = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(
        baseline,
        tmp_path,
        domain="retail",
        num_trials=2,
        expected_task_ids=["task"],
        budget_descriptor=descriptor,
    )
    assert not ledger.summary()["halted"]
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "current.json",
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert (
        run_ab.write_report_and_gate(baseline, candidate, ["retail"], 2, args)
        == run_ab.EXIT_INCOMPLETE
    )
    assert baseline.invocations["retail"]["failure_kind"] == "ledger_refused"
    assert baseline.invocations["retail"]["failure_stage"] == "budget"


@pytest.mark.parametrize("mode", ["own", "foreign", "audit_error"])
def test_current_nonzero_child_audits_only_owned_paid_work(tmp_path, monkeypatch, mode):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)

    def killed_child(command, **kwargs):
        identity = command[command.index("--invocation") + 1] if mode == "own" else "sibling"
        ledger.reserve(
            "paid",
            "vllm",
            "openai/gpt-5.2",
            1,
            1,
            num_retries=0,
            output_limit_verified=True,
            metadata={"invocation_id": identity, "domain": "airline"},
        )
        return types.SimpleNamespace(returncode=-9)

    if mode == "audit_error":
        original_audit = run_ab.budget_guard.Ledger.audit

        def unavailable(self, invocation_id=None):
            if invocation_id is not None:
                raise run_ab.budget_guard.BudgetError("owned audit unavailable")
            return original_audit(self, invocation_id)

        monkeypatch.setattr(run_ab.budget_guard.Ledger, "audit", unavailable)
    monkeypatch.setattr(run_ab.subprocess, "run", killed_child)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, budget_descriptor=descriptor)
    assert arm.invocations["airline"]["failure_kind"] == "nonzero_exit"
    assert ledger.summary()["halted"] is (mode != "foreign")


@pytest.mark.parametrize("reuse", [True, False])
def test_current_same_arm_uuid_reuse_has_independent_control(tmp_path, reuse):
    a, b = (_provenant_arm(tmp_path, "vllm") for _ in range(2))
    if reuse:
        b.invocations["retail"] = dict(a.invocations["retail"])
        b.scores = dict(a.scores)
    args = types.SimpleNamespace(
        out=tmp_path / "same-arm.md",
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == (
        run_ab.EXIT_INCOMPLETE if reuse else run_ab.EXIT_OK
    )
    assert ("inconsistent invocation provenance" in args.out.read_text()) is reuse


def test_current_staging_copies_marker_after_policy_copy_error(tmp_path, monkeypatch):
    import os
    import subprocess

    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/nightly-tau2.yml"
    lines = workflow.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "- name: Stage results for upload" in line)
    run = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "run: |")
    end = next(i for i in range(run + 1, len(lines)) if lines[i].startswith("      - name:"))
    script = "\n".join(line[10:] for line in lines[run + 1 : end])
    ledger, policy = tmp_path / "ledger.db", tmp_path / "policy.json"
    ledger.write_bytes(b"owned ledger")
    policy.write_bytes(b"owned policy")
    sidecar = tmp_path / "ledger.db.fail-stop"
    sidecar.write_bytes(b"owned marker")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("TAU2_BUDGET_LEDGER", str(ledger))
    monkeypatch.setenv("TAU2_BUDGET_POLICY", str(policy))
    monkeypatch.setenv("FAIL_COPY", str(policy))
    # Actual extracted workflow, actual copies except one explicit owned fault.
    prefix = 'cp() { if [ "$2" = "$FAIL_COPY" ]; then return 23; fi; /usr/bin/cp "$@"; };\n'
    child = subprocess.run(
        ["/usr/bin/bash", "-c", prefix + script],
        cwd=tmp_path,
        env=os.environ.copy(),
        check=False,
        timeout=10,
    )
    assert (tmp_path / "tau2_artifact" / sidecar.name).read_bytes() == b""
    assert child.returncode != 0


@pytest.mark.parametrize("startup", ["begin_refused", "missing_descriptor"])
def test_startup_budget_rejection_is_not_quality_diagnostic(tmp_path, monkeypatch, startup):
    candidate = _provenant_arm(tmp_path, "smg")
    descriptor = candidate.budget
    # An Arm may retain prior valid budget metadata even if this call lacks it.
    # The missing-budget invocation itself must stay fatal, not rely on another gate.
    baseline = run_ab.Arm("vllm", "http://unused.invalid", budget=dict(descriptor))
    launched = []

    def forbidden_child(*args, **kwargs):
        launched.append(args)
        raise AssertionError("startup budget failure must not launch a child")

    original_begin = run_ab.budget_guard.Ledger.begin_invocation

    def begin(self, identity, arm, domain):
        if arm == "vllm":
            raise run_ab.budget_guard.BudgetError("owned unhalted startup refusal")
        return original_begin(self, identity, arm, domain)

    monkeypatch.setattr(run_ab.budget_guard.Ledger, "begin_invocation", begin)
    monkeypatch.setattr(run_ab.subprocess, "run", forbidden_child)
    _run_tau2(
        baseline,
        tmp_path,
        domain="retail",
        num_trials=2,
        expected_task_ids=["task"],
        budget_descriptor=None if startup == "missing_descriptor" else descriptor,
    )
    assert not launched
    assert baseline.invocations["retail"]["failure_kind"] == (
        "missing_budget" if startup == "missing_descriptor" else "invalid_budget"
    )
    audit = run_ab.budget_guard.read_descriptor(descriptor).audit()
    assert not audit["halted"] and audit["pending"] == audit["unknown"] == 0
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "startup.json",
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert (
        run_ab.write_report_and_gate(baseline, candidate, ["retail"], 2, args)
        == run_ab.EXIT_INCOMPLETE
    )


@pytest.mark.parametrize("mode", ["healthy", "malformed"])
def test_startup_quality_controls_remain_diagnostic(tmp_path, monkeypatch, mode):
    candidate = _provenant_arm(tmp_path, "smg")
    descriptor = candidate.budget
    baseline = run_ab.Arm("vllm", "http://unused.invalid")
    launched = []

    def child(command, **kwargs):
        launched.append(command)
        folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
        folder.mkdir(parents=True)
        result = {
            "simulations": [
                {
                    "task_id": r["task_id"],
                    "trial": r["trial"],
                    "seed": r["seed"],
                    "reward_info": {"reward": r["reward"]},
                }
                for r in _rows()
            ]
        }
        (folder / "results.json").write_text(
            "{broken" if mode == "malformed" else json.dumps(result)
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", child)
    _run_tau2(
        baseline,
        tmp_path,
        domain="retail",
        num_trials=2,
        expected_task_ids=["task"],
        budget_descriptor=descriptor,
    )
    assert len(launched) == 1
    assert baseline.invocations["retail"].get("failure_kind") == (
        "invalid_results" if mode == "malformed" else None
    )
    args = types.SimpleNamespace(
        out=None,
        json_out=tmp_path / "control.json",
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(baseline, candidate, ["retail"], 2, args) == run_ab.EXIT_OK


@pytest.mark.parametrize("budget_fault", ["descriptor", "audit"])
def test_material_postchild_budget_io_is_not_quality_waivable(tmp_path, monkeypatch, budget_fault):
    descriptor = run_ab.prepare_budget()
    original = run_ab.budget_guard.read_descriptor
    launched = []

    def child(cmd, **kwargs):
        launched.append(cmd)
        if budget_fault == "descriptor":
            Path(descriptor["contexts_path"]).unlink()
        else:

            def refuse_audit(*args):
                raise OSError("owned postchild audit IO fault")

            monkeypatch.setattr(ledger_class, "audit", refuse_audit)
        return types.SimpleNamespace(returncode=0)

    # guarded_cli imports Ledger from its private budget module.
    ledger_class = type(original(descriptor))
    monkeypatch.setattr(run_ab.subprocess, "run", child)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, budget_descriptor=descriptor)
    assert len(launched) == 1
    assert arm.invocations["airline"]["failure_kind"] == "budget_unavailable"
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"airline": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(arm, run_ab.Arm("smg", ""), ["airline"], 1, args) == 2


def test_material_unconfirmed_halt_quarantines_later_domain_and_paired_arm(tmp_path, monkeypatch):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    ledger_class = type(ledger)
    calls = []

    def child(cmd, **kwargs):
        calls.append(cmd)
        ledger.reserve(
            "owned-pending",
            "vllm",
            "openai/gpt-5.2",
            1,
            1,
            num_retries=0,
            output_limit_verified=True,
            metadata={"invocation_id": cmd[cmd.index("--invocation") + 1]},
        )
        return types.SimpleNamespace(returncode=9)

    def cannot_halt(self):
        raise OSError("both halt stores unavailable")

    monkeypatch.setattr(run_ab.subprocess, "run", child)
    monkeypatch.setattr(ledger_class, "halt", cannot_halt)
    arm = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(arm, tmp_path, budget_descriptor=descriptor)
    assert arm.invocations["airline"]["scope_quarantined"] is True
    _run_tau2(arm, tmp_path, domain="retail", budget_descriptor=descriptor)
    sibling = run_ab.Arm("smg", "http://unused.invalid")
    _run_tau2(sibling, tmp_path, budget_descriptor=descriptor)
    assert len(calls) == 1
    assert arm.invocations["retail"]["failure_kind"] == "scope_quarantined"
    assert sibling.invocations["airline"]["failure_kind"] == "scope_quarantined"
    assert ledger.summary()["halted"] is False  # no cross-process halt claim


@pytest.mark.parametrize("dangling", [True, False])
def test_material_staging_records_lexical_stop_without_target_contents(
    tmp_path, monkeypatch, dangling
):
    import os
    import subprocess

    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/nightly-tau2.yml"
    lines = workflow.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "- name: Stage results for upload" in line)
    run = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "run: |")
    end = next(i for i in range(run + 1, len(lines)) if lines[i].startswith("      - name:"))
    script = "\n".join(line[10:] for line in lines[run + 1 : end])
    target = tmp_path / "private-target"
    if not dangling:
        target.write_bytes(b"must not be uploaded")
    marker = tmp_path / "ledger.db.fail-stop"
    marker.symlink_to(target)
    monkeypatch.setenv("TAU2_BUDGET_LEDGER", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    child = subprocess.run(
        ["/usr/bin/bash", "-e", "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        check=False,
        timeout=10,
    )
    captured = tmp_path / "tau2_artifact" / marker.name
    assert child.returncode == 0
    assert captured.is_file() and not captured.is_symlink()
    assert captured.read_bytes() == b""


def test_material_healthy_foreign_work_does_not_quarantine_pair(tmp_path, monkeypatch):
    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    ledger.reserve(
        "foreign",
        "smg",
        "openai/gpt-5.2",
        1,
        1,
        num_retries=0,
        output_limit_verified=True,
        metadata={"invocation_id": "foreign"},
    )
    calls = []

    def child(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=9)

    def forbidden_halt(self):
        pytest.fail("healthy foreign-only work must not halt or quarantine")

    monkeypatch.setattr(run_ab.subprocess, "run", child)
    monkeypatch.setattr(type(ledger), "halt", forbidden_halt)
    for name in ("vllm", "smg"):
        arm = run_ab.Arm(name, "http://unused.invalid")
        _run_tau2(arm, tmp_path, budget_descriptor=descriptor)
        assert arm.invocations["airline"]["failure_kind"] == "nonzero_exit"
        assert "scope_quarantined" not in arm.invocations["airline"]
    assert len(calls) == 2
    assert ledger.summary()["halted"] is False


def _unified_pair(tmp_path, monkeypatch):
    b = _provenant_arm(tmp_path, "smg")
    monkeypatch.setattr(
        run_ab.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0)
    )
    a = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(a, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    assert a.invocations["retail"]["failure_kind"] == "missing_results"
    assert run_ab.final_budget(a, b)["pending"] == 0
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    return a, b, args


@pytest.mark.parametrize("change", ["missing", "empty", "identity", "quality_control"])
def test_unified_requested_unscored_outcome_is_not_synthesized(tmp_path, monkeypatch, change):
    a, b, args = _unified_pair(tmp_path, monkeypatch)
    if change == "missing":
        a.invocations.clear()
    elif change == "empty":
        a.invocations["retail"] = {}
    elif change == "identity":
        a.invocations["retail"]["invocation_id"] = "e" * 32
    assert run_ab.final_budget(a, b)["halted"] is False
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == (
        0 if change == "quality_control" else 2
    )


def test_unified_post_budget_io_oracle_has_independently_healthy_final_budget(
    tmp_path, monkeypatch
):
    b = _provenant_arm(tmp_path, "smg")
    real = run_ab.budget_guard.read_descriptor
    calls = []

    def transient(descriptor):
        calls.append(True)
        if len(calls) == 2:
            raise FileNotFoundError("only the postchild descriptor observation fails")
        return real(descriptor)

    monkeypatch.setattr(run_ab.budget_guard, "read_descriptor", transient)
    monkeypatch.setattr(
        run_ab.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0)
    )
    a = run_ab.Arm("vllm", "http://unused.invalid")
    _run_tau2(a, tmp_path, domain="retail", num_trials=2, expected_task_ids=["task"])
    monkeypatch.setattr(run_ab.budget_guard, "read_descriptor", real)
    assert a.invocations["retail"]["failure_kind"] == "budget_unavailable"
    assert run_ab.final_budget(a, b) == {
        "halted": False,
        "pending": 0,
        "unknown": 0,
        "requests": [],
    }
    args = types.SimpleNamespace(
        out=None,
        json_out=None,
        tolerance=0.02,
        allow_incomplete=True,
        enforce_budget=True,
        expected_task_ids={"retail": ["task"]},
        seed=300,
    )
    assert run_ab.write_report_and_gate(a, b, ["retail"], 2, args) == 2


def test_unified_stop_linearizes_before_blocked_halt_io(tmp_path, monkeypatch):
    import threading

    descriptor = run_ab.prepare_budget()
    ledger = run_ab.budget_guard.read_descriptor(descriptor)
    entered, release = threading.Event(), threading.Event()
    launched = []
    errors = []

    def child(cmd, **kwargs):
        arm = cmd[cmd.index("--arm") + 1]
        launched.append(arm)
        if arm == "vllm":
            ledger.reserve(
                "owned",
                arm,
                "openai/gpt-5.2",
                1,
                1,
                num_retries=0,
                output_limit_verified=True,
                metadata={"invocation_id": cmd[cmd.index("--invocation") + 1]},
            )
        return types.SimpleNamespace(returncode=9)

    def halt(self):
        entered.set()
        assert release.wait(5), "owned halt barrier was not released"
        raise OSError("ordered stores unavailable")

    monkeypatch.setattr(run_ab.subprocess, "run", child)
    monkeypatch.setattr(type(ledger), "halt", halt)

    def a():
        try:
            _run_tau2(
                run_ab.Arm("vllm", "http://unused.invalid"), tmp_path, budget_descriptor=descriptor
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=a)
    thread.start()
    try:
        assert entered.wait(5), "A did not reach halt IO"
        b = run_ab.Arm("smg", "http://unused.invalid")
        done = threading.Event()

        def attempt_b():
            try:
                _run_tau2(b, tmp_path, budget_descriptor=descriptor)
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        sibling = threading.Thread(target=attempt_b)
        sibling.start()
        assert done.wait(2), "admission lock was held across halt IO"
        sibling.join(2)
        assert launched == ["vllm"]
        assert b.invocations["airline"]["failure_kind"] == "scope_quarantined"
        # A separate reviewed scope/ledger still admits work while A is blocked.
        other = dict(descriptor)
        policy = json.loads(Path(descriptor["policy_path"]).read_text())
        policy["scope_id"] += "-independent"
        policy_path = tmp_path / "other-policy.json"
        policy_path.write_text(json.dumps(policy))
        other.update(
            policy_path=str(policy_path),
            ledger_path=str(tmp_path / "other.db"),
            scope_id=policy["scope_id"],
        )
        other_policy, _ = run_ab.budget_guard.load_policy(
            policy_path, other["contexts_path"], other["contexts_sha256"]
        )
        run_ab.budget_guard.Ledger(other["ledger_path"], other_policy)
        c = run_ab.Arm("smg", "http://unused.invalid")
        _run_tau2(c, tmp_path, budget_descriptor=other)
        assert launched == ["vllm", "smg"]
        assert c.invocations["airline"]["failure_kind"] == "nonzero_exit"
    finally:
        release.set()
        thread.join(5)
        if "sibling" in locals():
            sibling.join(5)
    assert not thread.is_alive() and not errors


def _unified_real_parent(tmp_path, mode, ablate=False):
    import os
    import subprocess
    import textwrap

    source = Path(run_ab.__file__).resolve()
    root = source.parents[2]
    bootstrap = tmp_path / "cli-bootstrap.py"
    bootstrap.write_text(
        textwrap.dedent("""\
        import importlib.util,sys,os,json,types
        from pathlib import Path
        source=Path(sys.argv[1]).resolve()
        sys.path.insert(0,str(source.parent))
        spec=importlib.util.spec_from_file_location('actual_run_ab',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        def child(cmd,**kwargs):
            arm=cmd[cmd.index('--arm')+1]
            with open(os.environ['OWN_LAUNCHES'],'a') as out:out.write(arm+'\\n')
            if os.environ['OWN_MODE']=='hard' and arm=='vllm':
                ledger=module.budget_guard.read_descriptor(module.prepare_budget())
                ledger.reserve('owned',arm,'openai/gpt-5.2',1,1,num_retries=0,output_limit_verified=True,metadata={'invocation_id':cmd[cmd.index('--invocation')+1]})
                return types.SimpleNamespace(returncode=9)
            return types.SimpleNamespace(returncode=0)
        def no_halt(self):raise OSError('owned halt fault only in A process')
        module.subprocess.run=child
        if (os.environ['OWN_MODE']=='hard' and '--score-arm' in sys.argv
            and sys.argv[sys.argv.index('--score-arm')+1].startswith('vllm=')):
            module.budget_guard.Ledger.halt=no_halt
        if os.environ.get('OWN_ABLATE')=='1':module._arm_safety_exit=lambda *a,**k:0
        sys.argv=sys.argv[1:]
        raise SystemExit(module.main())
        """)
    )
    workflow = (root / ".github/workflows/nightly-tau2.yml").read_text()
    start = workflow.index('            A_URL=$(TAU2_GPU="$GPU_A"')
    end = workflow.index("          else", start)
    actual = workflow[start:end]
    descriptor = run_ab.prepare_budget()
    manifest = tmp_path / "tau2_expected_tasks.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "num_trials": 2, "seed": 300, "domains": {"retail": ["task"]}}
        )
    )
    import shlex

    common = [
        "--domains",
        "retail",
        "--num-trials",
        "2",
        "--expected-tasks",
        str(manifest),
        "--data-dir",
        str(tmp_path),
        "--budget-policy",
        descriptor["policy_path"],
        "--budget-contexts",
        descriptor["contexts_path"],
        "--budget-contexts-sha256",
        descriptor["contexts_sha256"],
        "--budget-ledger",
        descriptor["ledger_path"],
        "--budget-scope",
        descriptor["scope_id"],
    ]
    prefix = (
        "set -euo pipefail\nCOMMON=("
        + shlex.join(common)
        + ")\nGPU_A=0; DOMAINS=retail; NUM_TRIALS=2\n"
    )
    prefix += 'python() { "$OWN_PYTHON" -I -B "$OWN_BOOTSTRAP" "$@"; };\n'
    prefix += 'bash() { printf "http://unused.invalid\\n"; }; gate() { "$@"; };\n'
    env = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "OWN_BOOTSTRAP": str(bootstrap),
        "OWN_PYTHON": sys.executable,
        "OWN_LAUNCHES": str(tmp_path / "launches"),
        "OWN_MODE": mode,
        "OWN_ABLATE": "1" if ablate else "0",
    }
    child = subprocess.run(
        ["/usr/bin/bash", "-c", prefix + actual],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    records = (tmp_path / "launches").read_text().splitlines()
    assert (tmp_path / "scores_vllm.json").is_file(), child.stderr
    return child, records


@pytest.mark.parametrize("mode", ["hard", "quality"])
def test_unified_actual_cli_parent_propagates_hard_arm_exit(tmp_path, monkeypatch, mode):
    run_ab.prepare_budget()
    child, records = _unified_real_parent(tmp_path, mode)
    assert child.returncode == 2
    assert records == (["vllm"] if mode == "hard" else ["vllm", "smg"]), child.stderr


@pytest.mark.parametrize("marker", ["none", "regular", "dangling", "live"])
def test_unified_staging_uses_resolved_ledger_basename(tmp_path, monkeypatch, marker):
    import os
    import subprocess

    root = Path(run_ab.__file__).resolve().parents[2]
    lines = (root / ".github/workflows/nightly-tau2.yml").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "- name: Stage results for upload" in line)
    run = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "run: |")
    end = next(i for i in range(run + 1, len(lines)) if lines[i].startswith("      - name:"))
    script = "\n".join(line[10:] for line in lines[run + 1 : end])
    target = tmp_path / "actual.db"
    target.write_bytes(b"owned database")
    alias = tmp_path / "alias.db"
    alias.symlink_to(target)
    sidecar = tmp_path / "actual.db.fail-stop"
    if marker == "regular":
        sidecar.write_bytes(b"not read")
    elif marker in {"live", "dangling"}:
        secret = tmp_path / "private-target"
        if marker == "live":
            secret.write_bytes(b"must not be uploaded")
        sidecar.symlink_to(secret)
    env = {**os.environ, "RUNNER_TEMP": str(tmp_path), "TAU2_BUDGET_LEDGER": str(alias)}
    child = subprocess.run(
        ["/usr/bin/bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    out = tmp_path / "tau2_artifact"
    assert (out / "actual.db").read_bytes() == b"owned database"
    assert not (out / "alias.db").exists()
    captured = out / "actual.db.fail-stop"
    assert captured.exists() is (marker != "none")
    if marker != "none":
        assert captured.is_file() and not captured.is_symlink() and captured.read_bytes() == b""


def test_unified_ablation_of_budget_predicate_rejects_the_isolated_oracle(tmp_path, monkeypatch):
    real = run_ab._hard_outcome
    monkeypatch.setattr(
        run_ab,
        "_hard_outcome",
        lambda record: (
            False
            if isinstance(record, dict) and record.get("failure_kind") == "budget_unavailable"
            else real(record)
        ),
    )
    with pytest.raises(AssertionError, match="0 == 2"):
        test_unified_post_budget_io_oracle_has_independently_healthy_final_budget(
            tmp_path, monkeypatch
        )


def test_unified_ablation_of_requested_record_guard_rejects_the_isolated_oracle(
    tmp_path, monkeypatch
):
    real = run_ab._invocation_problem
    monkeypatch.setattr(
        run_ab,
        "_invocation_problem",
        lambda arm, domain, *args: (
            None if domain not in arm.invocations else real(arm, domain, *args)
        ),
    )
    with pytest.raises(AssertionError, match="0 == 2"):
        test_unified_requested_unscored_outcome_is_not_synthesized(tmp_path, monkeypatch, "missing")


def test_unified_ablation_of_cli_handoff_rejects_the_parent_oracle(tmp_path):
    child, records = _unified_real_parent(tmp_path, "hard", ablate=True)
    assert child.returncode == 2
    assert records == ["vllm", "smg"]  # removing only handoff gate really launches B


def _saved_shape_pair_and_cli(tmp_path, mode="clean", allow_incomplete=True):
    import os

    arms = []
    files = []
    for name in ("vllm", "smg"):
        rows = _rows(reward=0.0 if mode == "regression" and name == "smg" else 1.0)
        arm = run_ab.Arm(name, "", scores={"retail": run_ab.domain_scores(rows, 2)})
        _attach_provenance(tmp_path, arm, rows)
        arms.append(arm)
        path = tmp_path / (name + "-saved.json")
        run_ab.save_scores(arm, path)
        files.append(path)
    assert (
        arms[0].invocations["retail"]["invocation_id"]
        != arms[1].invocations["retail"]["invocation_id"]
    )
    assert run_ab.final_budget(*arms) == {
        **run_ab.final_budget(*arms),
        "pending": 0,
        "unknown": 0,
        "halted": False,
    }
    root = Path(run_ab.__file__).resolve().parents[2]
    cli = [
        sys.executable,
        "-I",
        "-B",
        str(Path(run_ab.__file__).resolve()),
        "--diff-baseline",
        str(files[0]),
        "--diff-candidate",
        str(files[1]),
        "--domains",
        "retail",
        "--num-trials",
        "2",
        "--seed",
        "300",
        "--expected-tasks",
        str(_expected_manifest(tmp_path, ["task"])),
        "--json-out",
        str(tmp_path / "comparison.json"),
    ]
    if allow_incomplete:
        cli.append("--allow-incomplete")
    return arms, files, root, cli, os.environ.copy()


def _actual_nightly_saved_gate(root, cli, env):
    import subprocess

    workflow = (root / ".github/workflows/nightly-tau2.yml").read_text()
    start = workflow.index("          AB_RC=0\n          gate() {")
    end = workflow.index('          if [ "$ARM_MODE" = sequential ]; then', start)
    gate = "\n".join(line[10:] for line in workflow[start:end].splitlines())
    finish = '          [ "$AB_RC" -eq 0 ] || exit "$AB_RC"'
    assert workflow.count(finish) == 1
    # Execute the original gate and final status check. No replacement function.
    script = "set -euo pipefail\n" + gate + '\ngate "$@"\n' + finish[10:] + "\n"
    return subprocess.run(
        ["/usr/bin/bash", "-e", "-c", script, "owned-real-nightly-gate", *cli],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("bad", [None, [], 17], ids=["null", "list", "number"])
@pytest.mark.parametrize("arm_index", [0, 1], ids=["baseline", "candidate"])
@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_saved_domain_shape_is_hard_in_actual_cli_and_nightly_gate(
    tmp_path, bad, arm_index, allow_incomplete
):
    import subprocess

    arms, files, root, cli, env = _saved_shape_pair_and_cli(
        tmp_path, allow_incomplete=allow_incomplete
    )
    healthy = run_ab.final_budget(*arms)
    assert healthy["pending"] == healthy["unknown"] == 0 and healthy["halted"] is False
    # Prove the complete paired files are accepted before changing only one domain value.
    control = subprocess.run(
        cli, cwd=root, env=env, capture_output=True, text=True, timeout=10, check=False
    )
    assert control.returncode == 0, control.stderr
    original = json.loads(files[arm_index].read_text())
    mutated = dict(original, scores={**original["scores"], "retail": bad})
    files[arm_index].write_text(json.dumps(mutated))
    direct = subprocess.run(
        cli, cwd=root, env=env, capture_output=True, text=True, timeout=10, check=False
    )
    parent = _actual_nightly_saved_gate(root, cli, env)
    assert run_ab.final_budget(*arms) == healthy
    assert (direct.returncode, parent.returncode) == (2, 2), (
        direct.returncode,
        direct.stderr,
        parent.returncode,
        parent.stdout,
        parent.stderr,
    )
    assert "invalid saved comparison evidence" in direct.stderr
    assert "::error::" in parent.stdout and "::warning::" not in parent.stdout


@pytest.mark.parametrize("mode,expected", [("clean", 0), ("regression", 1)])
def test_saved_real_nightly_gate_keeps_clean_and_quality_regression_policy(
    tmp_path, mode, expected
):
    import subprocess

    arms, files, root, cli, env = _saved_shape_pair_and_cli(tmp_path, mode=mode)
    direct = subprocess.run(
        cli, cwd=root, env=env, capture_output=True, text=True, timeout=10, check=False
    )
    parent = _actual_nightly_saved_gate(root, cli, env)
    assert direct.returncode == expected, direct.stderr
    assert parent.returncode == 0, (parent.stdout, parent.stderr)
    assert ("::warning::" in parent.stdout) is (expected == 1)
    assert run_ab.final_budget(*arms)["halted"] is False


def _comparable_quality_driver_pair(tmp_path, monkeypatch, mode="regression"):
    import os

    def child(command, **kwargs):
        domain = command[command.index("--domain") + 1]
        name = command[command.index("--arm") + 1]
        if domain == "retail":
            folder = tmp_path / "simulations" / command[command.index("--save-to") + 1]
            folder.mkdir(parents=True)
            reward = 1.0 if name == "vllm" or mode == "healthy" else 0.0
            (folder / "results.json").write_text(
                json.dumps(
                    {
                        "simulations": [
                            {
                                "task_id": row["task_id"],
                                "trial": row["trial"],
                                "seed": row["seed"],
                                "reward_info": {"reward": row["reward"]},
                            }
                            for row in _rows(reward=reward)
                        ]
                    }
                )
            )
        return types.SimpleNamespace(
            returncode=9 if mode == "safety" and name == "vllm" and domain == "airline" else 0
        )

    arms = []
    with monkeypatch.context() as patch:
        patch.setattr(run_ab.subprocess, "run", child)
        for name in ("vllm", "smg"):
            arm = run_ab.Arm(name, "http://unused.invalid")
            for domain, task in (("retail", "task"), ("airline", "airline-task")):
                _run_tau2(arm, tmp_path, domain=domain, num_trials=2, expected_task_ids=[task])
            arms.append(arm)
    assert all("retail" in arm.scores and "airline" not in arm.scores for arm in arms)
    assert all(arm.invocations["retail"]["status"] == "completed" for arm in arms)
    healthy = run_ab.final_budget(*arms)
    assert healthy["pending"] == healthy["unknown"] == 0 and healthy["halted"] is False
    if mode == "invalid_provenance":
        arms[0].invocations["retail"]["invocation_id"] = "e" * 32
    paths = []
    for arm in arms:
        path = tmp_path / ("scores_" + arm.name + ".json")
        run_ab.save_scores(arm, path)
        paths.append(path)
    manifest = tmp_path / "expected-two-domains.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 300,
                "num_trials": 2,
                "domains": {"retail": ["task"], "airline": ["airline-task"]},
            }
        )
    )
    cli = [
        sys.executable,
        "-I",
        "-B",
        str(Path(run_ab.__file__).resolve()),
        "--diff-baseline",
        str(paths[0]),
        "--diff-candidate",
        str(paths[1]),
        "--domains",
        "retail,airline",
        "--num-trials",
        "2",
        "--seed",
        "300",
        "--expected-tasks",
        str(manifest),
        "--tolerance",
        "0.02",
        "--json-out",
        str(tmp_path / "quality-comparison.json"),
    ]
    if mode != "regression_strict":
        cli.append("--allow-incomplete")
    return arms, paths, cli, os.environ.copy(), healthy


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("regression", 1),
        ("healthy", 0),
        ("invalid_provenance", 2),
        ("safety", 2),
        ("regression_strict", 2),
    ],
)
def test_comparable_quality_waiver_preserves_regression_and_hard_gates(
    tmp_path, monkeypatch, mode, expected
):
    import subprocess

    arms, paths, cli, env, healthy = _comparable_quality_driver_pair(tmp_path, monkeypatch, mode)
    direct = subprocess.run(cli, env=env, capture_output=True, text=True, timeout=10, check=False)
    assert run_ab.final_budget(*arms) == healthy
    assert direct.returncode == expected, (direct.returncode, direct.stdout, direct.stderr)
    if mode == "regression":
        assert "REGRESSION:" in direct.stderr
        report = json.loads((tmp_path / "quality-comparison.json").read_text())
        assert report["overall"]["passk"]["delta"] == -1.0
        assert set(report["incomplete"]) == {"vllm", "smg"}
    elif mode == "healthy":
        assert "REGRESSION:" not in direct.stderr


def _literal_always_stage(root, tmp_path):
    import os
    import subprocess

    workflow = (root / ".github/workflows/nightly-tau2.yml").read_text()
    start = workflow.index("      - name: Stage results for upload")
    end = workflow.index("      - name: Upload results (summary + scores + transcripts)", start)
    block = workflow[start:end]
    assert "        if: always()" in block
    script = block.split("        run: |\n", 1)[1]
    script = "\n".join(line[10:] for line in script.splitlines())
    assert "          path: tau2_artifact/" in workflow[end:]
    env = {**os.environ, "RUNNER_TEMP": str(tmp_path)}
    return subprocess.run(
        ["/usr/bin/bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_failure_artifacts_keep_early_hard_arm_without_starting_pair(tmp_path):
    parent, launches = _unified_real_parent(tmp_path, "hard")
    assert parent.returncode == 2 and launches == ["vllm"]
    # The original sequential score-arm command is unguarded. Its hard2 happens
    # before any gate/diff call (the existing helper's gate stub is unreachable).
    score = tmp_path / "scores_vllm.json"
    before = score.read_bytes()
    failed = json.loads(before)["invocations"]["retail"]
    assert failed["status"] == "failed" and failed["halt_error"]
    assert "scope_stop_state" in failed
    assert not (tmp_path / "scores_smg.json").exists()
    assert not (tmp_path / "tau2_ab.json").exists()
    stage = _literal_always_stage(Path(run_ab.__file__).resolve().parents[2], tmp_path)
    assert stage.returncode == 0, stage.stderr
    assert (tmp_path / "tau2_artifact/scores_vllm.json").read_bytes() == before
    assert not (tmp_path / "tau2_artifact/scores_smg.json").exists()


def test_failure_artifacts_keep_both_actual_arm_outcome_files(tmp_path, monkeypatch):
    arms, files, cli, env, healthy = _comparable_quality_driver_pair(
        tmp_path, monkeypatch, "healthy"
    )
    assert all(arm.invocations["airline"]["failure_kind"] == "missing_results" for arm in arms)
    before = {path.name: path.read_bytes() for path in files}
    stage = _literal_always_stage(Path(run_ab.__file__).resolve().parents[2], tmp_path)
    assert stage.returncode == 0, stage.stderr
    assert all(
        (tmp_path / "tau2_artifact" / name).read_bytes() == raw for name, raw in before.items()
    )
    assert run_ab.final_budget(*arms) == healthy
