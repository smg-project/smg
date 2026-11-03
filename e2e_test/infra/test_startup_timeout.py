"""Unit tests for the default startup wait and the ``E2E_STARTUP_TIMEOUT``
floor on startup waits."""

import ast
import importlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from infra import worker as worker_module
from infra.constants import (
    DEFAULT_STARTUP_TIMEOUT,
    ENGINE_START_SECS,
    ENV_STARTUP_TIMEOUT,
    SERVICER_WARMUP_BUDGET_SECS,
    effective_startup_timeout,
    get_startup_timeout,
)
from infra.model_specs import MODEL_SPECS
from infra.startup_profile_test_support import set_profile_lane
from infra.worker import worker_startup_timeout

_SERVICER_WARMUP = (
    Path(__file__).resolve().parents[2] / "grpc_servicer" / "smg_grpc_servicer" / "warmup.py"
)


def _servicer_default_budget_secs() -> float:
    """``DEFAULT_BUDGET_SECS`` as the servicer's warm-up module assigns it, read
    off its source so the harness test needs none of the servicer's imports."""
    for node in ast.parse(_SERVICER_WARMUP.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_BUDGET_SECS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"DEFAULT_BUDGET_SECS is not assigned at the top of {_SERVICER_WARMUP}")


def test_default_bound_is_the_engine_start_plus_the_servicer_warmup_budget():
    """The servicer holds SERVING until its planned warm-up ran, a pass bounded
    by its default budget; the harness waits for the engine's start plus that
    budget. The budget half is pinned to the servicer's own constant so the two
    cannot drift apart unnoticed."""
    assert DEFAULT_STARTUP_TIMEOUT == ENGINE_START_SECS + SERVICER_WARMUP_BUDGET_SECS
    if not _SERVICER_WARMUP.is_file():
        pytest.skip(f"{_SERVICER_WARMUP} is not in this tree")
    assert SERVICER_WARMUP_BUDGET_SECS == _servicer_default_budget_secs()


def test_a_spec_that_raises_the_bound_keeps_the_warm_up_budget():
    """A spec's ``startup_timeout`` replaces the default, so a value sized for
    the engine's start alone, at or under the default, would lose the warm-up
    headroom: every spec that sets one raises the default. That the budget is
    part of the value is said at the spec itself, as
    ``X + SERVICER_WARMUP_BUDGET_SECS``; no bound here can tell."""
    overrides = {
        name: spec["startup_timeout"]
        for name, spec in MODEL_SPECS.items()
        if "startup_timeout" in spec
    }
    assert overrides
    for name, bound in overrides.items():
        assert bound > DEFAULT_STARTUP_TIMEOUT, name


def test_unset_env_leaves_the_per_model_bound(monkeypatch):
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    assert get_startup_timeout() is None
    assert effective_startup_timeout(DEFAULT_STARTUP_TIMEOUT) == DEFAULT_STARTUP_TIMEOUT
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, " ")
    assert effective_startup_timeout(600) == 600


def test_env_is_a_floor_not_a_cap(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, " 1800 ")
    assert get_startup_timeout() == 1800
    assert effective_startup_timeout(DEFAULT_STARTUP_TIMEOUT) == 1800
    assert effective_startup_timeout(3600) == 3600


def test_non_positive_values_are_rejected(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "0")
    with pytest.raises(ValueError, match=ENV_STARTUP_TIMEOUT):
        get_startup_timeout()


def test_a_workers_bound_is_its_specs_timeout_or_the_default(monkeypatch):
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    specs = {"slow": {"startup_timeout": 900}, "plain": {}}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: specs[model_id])
    assert worker_startup_timeout("slow") == 900
    assert worker_startup_timeout("plain") == DEFAULT_STARTUP_TIMEOUT
    assert worker_startup_timeout("plain", timeout=450) == 450
    # the spec's own bound wins over a caller's default
    assert worker_startup_timeout("slow", timeout=450) == 900


def test_a_workers_bound_takes_the_env_floor(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1800")
    specs = {"slow": {"startup_timeout": 900}, "slower": {"startup_timeout": 3600}}
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: specs[model_id])
    assert worker_startup_timeout("slow") == 1800
    assert worker_startup_timeout("slower") == 3600


_ZMQ_DP2_MODELS = (
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
)


def test_zmq_dp2_all_models_can_run_sequentially_in_one_profile(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    assert [worker_startup_timeout(model) for model in _ZMQ_DP2_MODELS] == [300] * 8


def test_4gpu_both_models_keep_distinct_frozen_bounds_in_one_process(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-4gpu-chat")
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1200
    assert worker_startup_timeout("openai/gpt-oss-120b") == 600
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1200


@pytest.mark.parametrize(
    "profile,floor",
    [
        ("vllm-zmq-dp2-chat", ""),
        ("vllm-zmq-dp2-chat", " "),
        ("vllm-zmq-dp2-chat", "1"),
        ("vllm-zmq-dp2-chat", "300"),
        ("vllm-4gpu-chat", ""),
        ("vllm-4gpu-chat", "60"),
    ],
)
def test_a_compatible_floor_cannot_extend_a_profile(monkeypatch, profile, floor):
    set_profile_lane(monkeypatch, profile, floor=floor)
    model, bound = (
        (_ZMQ_DP2_MODELS[0], 300)
        if profile == "vllm-zmq-dp2-chat"
        else ("openai/gpt-oss-120b", 600)
    )
    assert worker_startup_timeout(model, timeout=9999) == bound


@pytest.mark.parametrize("floor", ["301", "1201", "601", "0", "-1", "bad", "1.5"])
def test_zmq_dp2_rejects_extended_or_invalid_floor(monkeypatch, floor):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, floor)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ZMQ_DP2_MODELS[0])


def test_4gpu_rejects_floor61_before_its_shorter_gateway_wait(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-4gpu-chat")
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "61")
    with pytest.raises(ValueError, match="frozen wait"):
        worker_startup_timeout("openai/gpt-oss-120b")


@pytest.mark.parametrize(
    "key,value",
    [
        ("E2E_STARTUP_PROFILE", "unsupported-profile"),
        ("E2E_ENGINE", "sglang"),
        ("E2E_RUNTIME", "sglang"),
        ("E2E_GPU_VENDOR", "nvidia"),
        ("E2E_GPU_TIER", "4"),
        ("E2E_CONNECTION_MODE", "grpc"),
        ("E2E_ZMQ_ENGINE_COUNT", "1"),
        ("E2E_MIN_SELECTED", "1"),
    ],
)
def test_unknown_or_mismatched_profile_never_falls_back(monkeypatch, key, value):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ZMQ_DP2_MODELS[0])


def test_unknown_profile_model_is_rejected_even_with_an_ordinary_spec(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 1})
    with pytest.raises(ValueError, match="no frozen startup contract"):
        worker_startup_timeout("different-model")


def test_spec_tp_override_is_rejected_without_changing_the_model_spec(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 2})
    with pytest.raises(ValueError, match="requires TP1"):
        worker_startup_timeout(_ZMQ_DP2_MODELS[0])


@pytest.mark.parametrize("new_value", ["", "vllm-4gpu-chat"])
def test_selected_profile_cannot_change_in_process(monkeypatch, new_value):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    assert worker_startup_timeout(_ZMQ_DP2_MODELS[0]) == 300
    monkeypatch.setenv("E2E_STARTUP_PROFILE", new_value)
    with pytest.raises(ValueError):
        worker_startup_timeout(_ZMQ_DP2_MODELS[0])


def test_profile_floor_cannot_change_after_a_model_was_resolved(monkeypatch):
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    assert worker_startup_timeout(_ZMQ_DP2_MODELS[0]) == 300
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1")
    with pytest.raises(ValueError, match="context changed"):
        worker_startup_timeout(_ZMQ_DP2_MODELS[1])


def test_blank_profile_retains_current_model_and_floor_semantics(monkeypatch):
    from infra import constants as constants_module

    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.setenv("E2E_STARTUP_PROFILE", " ")
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    assert worker_startup_timeout(_ZMQ_DP2_MODELS[0]) == 600
    assert worker_startup_timeout("meta-llama/Llama-4-Scout-17B-16E-Instruct") == 1500
    assert worker_startup_timeout("openai/gpt-oss-120b") == 900
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1800")
    assert worker_startup_timeout("openai/gpt-oss-120b") == 1800


def test_declared_profile_literals_preserve_models_lane_and_limits():
    from infra.constants import _STARTUP_PROFILES, ConnectionMode

    zmq = _STARTUP_PROFILES["vllm-zmq-dp2-chat"]
    four = _STARTUP_PROFILES["vllm-4gpu-chat"]
    assert dict(zmq.models) == {model: (300, 1) for model in _ZMQ_DP2_MODELS}
    assert dict(four.models) == {
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": (1200, 4),
        "openai/gpt-oss-120b": (600, 4),
    }
    assert dict(zmq.lane) == {
        "E2E_ENGINE": "vllm",
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_GPU_TIER": "1",
        "E2E_CONNECTION_MODE": "zmq",
        "E2E_ZMQ_ENGINE_COUNT": "2",
        "E2E_MIN_SELECTED": "202",
    }
    assert dict(four.lane) == {
        "E2E_ENGINE": "vllm",
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_GPU_TIER": "4",
        "E2E_CONNECTION_MODE": "",
        "E2E_ZMQ_ENGINE_COUNT": "",
        "E2E_MIN_SELECTED": "32",
    }
    assert (zmq.max_floor, zmq.modes, zmq.gpu_multiplier) == (300, {ConnectionMode.ZMQ}, 2)
    assert (four.max_floor, four.modes, four.gpu_multiplier) == (
        60,
        {ConnectionMode.HTTP, ConnectionMode.GRPC},
        1,
    )
    assert zmq.test_dir == four.test_dir == "e2e_test/chat_completions"


@pytest.mark.parametrize("model", _ZMQ_DP2_MODELS)
def test_zmq_gateway_helper_enforces_real_model_load_bound(monkeypatch, model):
    setup = importlib.import_module("fixtures.setup_backend")
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, model, 60) == 300
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, model, 300) == 300
    with pytest.raises(ValueError, match="exceeds frozen wait 300"):
        setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, model, 301)


@pytest.mark.parametrize("mode", ["grpc", "http"])
def test_4gpu_gateway_helper_keeps_short_wait_after_long_worker(monkeypatch, mode):
    setup = importlib.import_module("fixtures.setup_backend")
    set_profile_lane(monkeypatch, "vllm-4gpu-chat")
    for model, worker_bound in [
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 1200),
        ("openai/gpt-oss-120b", 600),
    ]:
        assert worker_startup_timeout(model) == worker_bound
        assert setup._gateway_readiness_timeout(setup.ConnectionMode(mode), model, 60) == 60
        with pytest.raises(ValueError, match="exceeds frozen wait 60"):
            setup._gateway_readiness_timeout(setup.ConnectionMode(mode), model, 61)


def test_ordinary_gateway_helper_preserves_model_max_and_environment_floor(monkeypatch):
    setup = importlib.import_module("fixtures.setup_backend")
    monkeypatch.delenv("E2E_STARTUP_PROFILE", raising=False)
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.setattr(setup, "get_model_spec", lambda _: {"startup_timeout": 600})
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, "model", 60) == 600
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, "model", 900) == 900
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.GRPC, "model", 60) == 60
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "1800")
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, "model", 900) == 1800
    assert setup._gateway_readiness_timeout(setup.ConnectionMode.HTTP, "model", 60) == 1800


def test_zmq_local_fixture_passes_profile_bound_to_actual_gateway_start(monkeypatch):
    setup = importlib.import_module("fixtures.setup_backend")
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    calls = []
    workers = [SimpleNamespace(base_url="zmq://test-only")]
    gateway = SimpleNamespace(
        base_url="http://test-only",
        shutdown=lambda: calls.append("shutdown"),
        start=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(setup, "_start_workers_tracked", lambda **kwargs: workers)
    monkeypatch.setattr(setup, "set_class_workers", lambda _: None)
    monkeypatch.setattr(setup, "clear_class_workers", lambda: None)
    monkeypatch.setattr(setup, "stop_workers", lambda ws: calls.append(("stop", ws)))
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: object())
    generator = setup._setup_local(
        _ZMQ_DP2_MODELS[0],
        "model-path",
        "vllm",
        setup.ConnectionMode.ZMQ,
        {},
        dict(setup._GW_DEFAULTS),
        gateway,
        "vllm",
        None,
    )
    next(generator)
    assert calls[0]["timeout"] == 300
    assert calls[0]["workers"] is workers
    assert calls[0]["backend"] == "vllm"
    generator.close()
    assert calls[1:] == ["shutdown", ("stop", workers)]


@pytest.mark.parametrize("profile", ["vllm-zmq-dp2-chat", "vllm-4gpu-chat"])
def test_real_pytest_config_binds_chat_selection_without_test_env(monkeypatch, profile):
    from infra.constants import get_startup_profile

    config = set_profile_lane(monkeypatch, profile)
    assert "TEST_DIRS" not in os.environ and "TEST_FILTER" not in os.environ
    assert len(config.args) == 1 and Path(config.args[0]).name == "chat_completions"
    assert get_startup_profile() == profile


@pytest.mark.parametrize(
    "restriction",
    [
        "keyword",
        "markexpr",
        "deselect",
        "ignore",
        "ignore_glob",
        "lf",
        "stepwise",
        "nodeid",
        "different-directory",
        "multiple-directories",
    ],
)
def test_real_pytest_config_rejects_restricted_selection_before_setup(monkeypatch, restriction):
    chat = Path(__file__).resolve().parents[1] / "chat_completions"
    args = [str(chat)]
    options = {
        "keyword": ["-k", "subset"],
        "markexpr": ["-m", "e2e"],
        "deselect": ["--deselect", str(chat / "test_one.py") + "::test_one"],
        "ignore": ["--ignore", str(chat / "test_one.py")],
        "ignore_glob": ["--ignore-glob", "*one*"],
        "lf": ["--lf"],
        "stepwise": ["--sw"],
    }
    if restriction == "nodeid":
        args = [str(chat) + "::selected"]
    elif restriction == "different-directory":
        args = [str(Path(__file__).parent)]
    elif restriction == "multiple-directories":
        args += [str(Path(__file__).parent)]
    else:
        args += options[restriction]
    with pytest.raises(ValueError, match="pytest|chat selection"):
        set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat", args=args)


@pytest.mark.parametrize("fault", ["missing-proof", "changed-keyword", "changed-directory"])
def test_direct_profile_rejects_absent_or_changed_selection_before_preparation(monkeypatch, fault):
    from infra import constants as constants_module

    config = set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    if fault == "missing-proof":
        monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_SELECTION", None)
    elif fault == "changed-keyword":
        config.option.keyword = "subset"
    else:
        config.args[:] = [str(Path(__file__).parent)]
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: pytest.fail("model prepared"))
    monkeypatch.setattr(worker_module, "get_engine_port", lambda: pytest.fail("port prepared"))
    with pytest.raises(ValueError, match="pytest"):
        worker_startup_timeout(_ZMQ_DP2_MODELS[0])
    with pytest.raises(ValueError, match="pytest"):
        worker_module.start_workers(
            _ZMQ_DP2_MODELS[0], engine="vllm", mode=constants_module.ConnectionMode.ZMQ
        )


def test_collection_finish_checks_actual_paths_and_declared_minimum(monkeypatch):
    hooks = importlib.import_module("fixtures.hooks")
    config = set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    reporter = config.pluginmanager.get_plugin("e2e_selection_reporter")
    chat = Path(__file__).resolve().parents[1] / "chat_completions"
    item = SimpleNamespace(path=chat / "test_chat.py", nodeid="chat::selected")
    session = SimpleNamespace(config=config, items=[item] * 202)
    reporter.pytest_collection_finish(session)
    session.items = [item] * 201
    with pytest.raises(pytest.exit.Exception, match="floor of 202"):
        reporter.pytest_collection_finish(session)
    session.items = [SimpleNamespace(path=Path(__file__), nodeid="infra::wrong")] * 202
    with pytest.raises(ValueError, match="outside"):
        reporter.pytest_collection_finish(session)
    assert hooks._SelectionReporter is type(reporter)


@pytest.mark.parametrize(
    "entry,profile,mode,base",
    [
        ("local", "vllm-zmq-dp2-chat", "zmq", 301),
        ("class", "vllm-zmq-dp2-chat", "zmq", 301),
        ("pooled", "vllm-zmq-dp2-chat", "zmq", 301),
        ("local", "vllm-4gpu-chat", "http", 61),
        ("class", "vllm-4gpu-chat", "http", 61),
    ],
)
def test_gateway_contract_rejection_precedes_model_ports_and_worker_acquisition(
    monkeypatch, entry, profile, mode, base
):
    setup = importlib.import_module("fixtures.setup_backend")
    set_profile_lane(monkeypatch, profile)
    model = _ZMQ_DP2_MODELS[0] if mode == "zmq" else "openai/gpt-oss-120b"
    events = []

    def forbidden(*args, **kwargs):
        events.append("preparation")
        pytest.fail("invalid gateway contract reached model/port/worker preparation")

    monkeypatch.delenv(setup.ENV_SKIP_BACKEND_SETUP, raising=False)
    monkeypatch.setattr(setup, "get_model_spec", forbidden)
    monkeypatch.setattr(worker_module, "get_model_spec", forbidden)
    monkeypatch.setattr(worker_module, "get_engine_port", forbidden)
    monkeypatch.setattr(setup, "_start_workers_tracked", forbidden)
    monkeypatch.setattr(setup, "get_pool", lambda: SimpleNamespace(acquire_zmq=forbidden))
    monkeypatch.setattr(setup, "get_runtime", lambda: "vllm")
    config = {**setup._GW_DEFAULTS, "timeout": base}
    node = SimpleNamespace(stash={})
    if entry == "local":
        generator = setup._setup_local(
            model,
            "model-path",
            "vllm",
            setup.ConnectionMode(mode),
            {},
            config,
            SimpleNamespace(shutdown=forbidden),
            "vllm",
            None,
        )
    elif entry == "pooled":
        generator = setup._setup_pooled_zmq(
            node, model, "model-path", "vllm", {}, config, "vllm", None
        )
    else:
        monkeypatch.setattr(setup, "resolve_class_marker", lambda *args: None)
        monkeypatch.setattr(setup, "model_id_for_engine", lambda *args: model)
        monkeypatch.setattr(setup, "get_marker_kwargs", lambda *args, **kwargs: config)
        request = SimpleNamespace(param="http", node=node)
        generator = setup.setup_backend.__wrapped__(request)
    with pytest.raises(ValueError, match="exceeds frozen wait"):
        next(generator)
    assert events == []


@pytest.mark.parametrize("changed", [None, "", " ", "vllm-4gpu-chat", "unsupported-profile"])
def test_real_gateway_rejects_cleared_or_changed_pinned_profile_before_model(monkeypatch, changed):
    setup = importlib.import_module("fixtures.setup_backend")
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    if changed is None:
        monkeypatch.delenv("E2E_STARTUP_PROFILE")
    else:
        monkeypatch.setenv("E2E_STARTUP_PROFILE", changed)
    events = []

    def forbidden(*args, **kwargs):
        events.append("model lookup")
        pytest.fail("changed profile reached the ordinary model timeout")

    monkeypatch.setattr(setup, "get_model_spec", forbidden)
    monkeypatch.setattr(worker_module, "get_model_spec", forbidden)
    with pytest.raises(
        ValueError, match="context changed|validated pytest|declared startup profile"
    ):
        setup._gateway_readiness_timeout(setup.ConnectionMode.ZMQ, _ZMQ_DP2_MODELS[0], 60)
    assert events == []


def test_profile_ci_budgets_include_startups_and_prior_test_allowance():
    from infra.constants import _STARTUP_PROFILES

    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/pr-test-rust.yml"
    jobs = dict(
        re.findall(
            r"^  ([a-z0-9_-]+):\n(.*?)(?=^  [a-z0-9_-]+:\n|\Z)",
            workflow.read_text(),
            re.M | re.S,
        )
    )
    for profile, job, test_allowance, job_gap in [
        ("vllm-zmq-dp2-chat", "e2e-2gpu-chat-zmq-dp-vllm-amd", 40, 6),
        ("vllm-4gpu-chat", "e2e-4gpu-chat-vllm-amd", 25, 5),
    ]:
        startup_seconds = sum(
            bound for bound, _tp in _STARTUP_PROFILES[profile].models.values()
        )
        step_match = re.search(r"^      test_timeout: (\d+)$", jobs[job], re.M)
        job_match = re.search(r"^      timeout: (\d+)$", jobs[job], re.M)
        assert step_match is not None and job_match is not None
        step_minutes = int(step_match.group(1))
        job_minutes = int(job_match.group(1))
        assert step_minutes >= (startup_seconds + 59) // 60 + test_allowance
        assert job_minutes >= step_minutes + job_gap


def test_absent_profile_definition_rejects_before_model_lookup(monkeypatch):
    from infra.constants import ConnectionMode

    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    monkeypatch.setattr(worker_module, "startup_profile_definition", lambda: None)

    def forbidden_lookup(model_id):
        pytest.fail("model lookup must follow profile-definition validation")

    monkeypatch.setattr(worker_module, "get_model_spec", forbidden_lookup)
    with pytest.raises(ValueError, match="no startup profile definition"):
        worker_module.validate_startup_profile_worker(
            _ZMQ_DP2_MODELS[0], engine="vllm", mode=ConnectionMode.ZMQ
        )
