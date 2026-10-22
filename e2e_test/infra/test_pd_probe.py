"""CPU-only entry controls; native source guard and backend boundary are explicit stubs."""

import ast
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
import pytest_rerunfailures
from infra import pd_probe as probe
from infra.constants import WorkerType

pytest_plugins = ["pytester"]


def probe_class():
    source = (Path(__file__).parents[1] / "router/test_pd_topologies.py").read_text()
    return next(
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == probe.PROBE
    )


@pytest.fixture
def configured(monkeypatch, tmp_path):
    env = dict(
        GITHUB_EVENT_NAME="workflow_dispatch",
        GITHUB_REPOSITORY="smg-project/smg",
        GITHUB_RUN_ID="123",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_SHA="a" * 40,
        RUNNER_NAME="cpu",
        E2E_JOB_MARKER="e2e-123-1-cpu",
        E2E_RUNTIME="vllm",
        E2E_GPU_TIER="4",
        E2E_VLLM_KV_BACKEND="moriio",
        E2E_LOG_DIR=str(tmp_path / "logs"),
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", raising=False)
    monkeypatch.setitem(sys.modules, "torch", NS(version=NS(hip="CPU-stub")))
    monkeypatch.setattr(probe, "require_installed_contract", lambda: {"root": "/cpu-stub/vllm"})
    markers, options = [], {"keyword": probe.PROBE}
    request = NS(
        config=NS(getoption=lambda name, default=None: options.get(name, default)),
        node=NS(add_marker=lambda marker, append: markers.append((marker, append))),
    )
    request.session = NS(items=[NS(getparent=lambda _: request.node)])
    monkeypatch.setattr(pytest_rerunfailures, "get_reruns_count", lambda _: 0)
    return request, options, markers, tmp_path / "logs"


@pytest.mark.parametrize("keyword", ["", "ObserverProbe", probe.PROBE + " or other"])
def test_default_or_combined_selection_does_nothing(configured, keyword):
    request, options, markers, directory = configured
    options["keyword"] = keyword
    with pytest.raises(pytest.skip.Exception):
        probe.explicit_context(request)
    assert not directory.exists() and not markers


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_EVENT_NAME", "schedule"),
        ("GITHUB_RUN_ID", ""),
        ("GITHUB_RUN_ATTEMPT", "0"),
        ("GITHUB_SHA", "invalid"),
        ("GITHUB_REPOSITORY", "foreign/repo"),
        ("E2E_JOB_MARKER", "e2e-123-2-cpu"),
        ("RUNNER_NAME", ""),
        ("E2E_RUNTIME", "sglang"),
        ("E2E_GPU_TIER", "2"),
        ("E2E_LOG_DIR", ""),
        ("E2E_VLLM_KV_BACKEND", "unknown"),
        ("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1"),
    ],
)
def test_invalid_identity_fails_before_preparation(configured, monkeypatch, key, value):
    request, _, markers, directory = configured
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        probe.explicit_context(request)
    assert not directory.exists() and not markers


@pytest.mark.parametrize("options", [{"force_reruns": 2}, {"reruns_mode": "append"}])
def test_retry_override_is_rejected(configured, options):
    request, actual, markers, directory = configured
    actual.update(options)
    with pytest.raises(ValueError):
        probe.explicit_context(request)
    assert not directory.exists() and not markers


def test_explicit_context_precreates_directory_and_local_marker(configured):
    request, _, markers, directory = configured
    context, root = probe.explicit_context(request)
    assert context["trace_dir"].is_dir() and context["trace_dir"].is_relative_to(directory)
    assert context["trace_dir"].is_absolute() and root == "/cpu-stub/vllm"
    assert markers[0][1] is False and markers[0][0].kwargs["scheduler_observation"] is context


@pytest.mark.parametrize("keyword,errors,skipped", [("", 0, 1), (probe.PROBE, 1, 0)])
@pytest.mark.parametrize("selector", [probe.PROBE, probe.ABORT_PROBE])
def test_actual_class_autouse_precedes_real_fixture_and_disables_reruns(
    configured, pytester, monkeypatch, keyword, errors, skipped, selector
):
    search_path = [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in search_path if path))
    block = ast.unparse(probe_class())
    if selector == probe.ABORT_PROBE:
        source = (Path(__file__).parents[1] / "router/test_pd_topologies.py").read_text()
        block += "\n" + ast.unparse(
            next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == selector
            )
        )
    keyword = selector if keyword else ""
    pytester.makeini("[pytest]\nmarkers=\n engine\n gpu\n e2e\n model\n gateway\n workers\n")
    testfile = pytester.makepyfile(
        "import os, pytest, httpx\n_MODEL='cpu'\n_LOG_DIR='cpu'\n_GATEWAY_ARGS=[]\n" + block
    )
    pytester.makeconftest("""
import importlib, sys
from pathlib import Path
from types import SimpleNamespace as NS
from infra import pd_probe
backend = importlib.import_module('fixtures.setup_backend')
setup_backend = backend.setup_backend
sys.modules['torch'] = NS(version=NS(hip='CPU-stub'))
pd_probe.require_installed_contract = lambda: {'root': '/cpu-stub/vllm'}
backend.get_model_spec = lambda _: {'model': '/cpu-model', 'tp': 1}
def gateway():
    Path('gateway').touch()
    return NS(shutdown=lambda: None)
backend.Gateway = gateway
def cpu_boundary(model_id, model_path, engine, mode, workers_config, *args):
    ctx = workers_config['scheduler_observation']
    assert ctx['trace_dir'].is_dir() and ctx['run'] == '123'
    with Path('starts').open('a') as stream:
        stream.write('one\\n')
    try:
        raise RuntimeError('controlled CPU setup failure')
        yield
    finally:
        Path('stopped').touch()
backend._setup_pd = cpu_boundary
""")
    wrapper_retry = ["--reruns", "2", "--reruns-delay", "5"]
    result = pytester.runpytest_subprocess(
        "-p",
        "rerunfailures",
        *wrapper_retry,
        "-s",
        "-q",
        "-W",
        "error",
        "-k",
        keyword,
        str(testfile) + "::" + selector,
    )
    result.assert_outcomes(errors=errors, skipped=skipped)
    if errors:
        assert (pytester.path / "starts").read_text() == "one\n"
        assert (pytester.path / "stopped").exists()
        result.stdout.fnmatch_lines(["SMG_PD_PROBE_BEGIN", "SMG_PD_PROBE_END"])
        lines = result.stdout.lines
        header = json.loads(lines[lines.index("SMG_PD_PROBE_BEGIN") + 1])
        method = (
            "test_decode_client_close"
            if selector == probe.ABORT_PROBE
            else "test_real_startup_and_admission"
        )
        assert len(header["nodeids"]) == 1 and f"::{method}[" in header["nodeids"][0]
    else:
        assert not (pytester.path / "starts").exists() and not configured[3].exists()
        assert not (pytester.path / "gateway").exists()


@pytest.mark.parametrize("fault", ["rocm", "source"])
def test_unknown_build_or_source_fails_before_directory(configured, monkeypatch, fault):
    if fault == "rocm":
        monkeypatch.setitem(sys.modules, "torch", NS(version=NS(hip=None)))
    else:
        monkeypatch.setattr(
            probe,
            "require_installed_contract",
            lambda: (_ for _ in ()).throw(RuntimeError("unknown source")),
        )
    with pytest.raises((ValueError, RuntimeError)):
        probe.explicit_context(configured[0])
    assert not configured[3].exists() and not configured[2]


@pytest.mark.parametrize("settle", [False, True])
def test_snapshot_wait_is_bounded_and_preserves_raw_bytes(monkeypatch, settle):
    files, calls, state = {"prefix": (b"original\n", [])}, [], {}
    clock = iter([0, 0 if settle else 31])

    def validate(*args):
        calls.append("validate")
        if len(calls) == 1:
            raise ValueError("missing or unfinished evidence")

    monkeypatch.setattr(probe, "read_prefix", lambda _: files)
    monkeypatch.setattr(probe, "validate_prefix", validate)
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(probe.time, "sleep", lambda delay: None)
    if settle:
        assert probe.observe_request(None, [], None, state, "/stub") is files and len(calls) == 2
    else:
        with pytest.raises(ValueError, match="30 seconds"):
            probe.observe_request(None, [], None, state, "/stub")
    assert state["files"] is files


def test_actual_request_nonce_and_worker_context_are_exportable(monkeypatch, tmp_path):
    node = probe_class()

    namespace = dict(globals(), os=os, re=re, _MODEL="cpu", _LOG_DIR="cpu", _GATEWAY_ARGS=[])
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-probe-class", "exec"), namespace)
    context, state, sent = dict(run="123", attempt="1", trace_dir=tmp_path), {}, {}
    workers = [
        NS(
            worker_type=role,
            gpu_ids=[index],
            observer_instance=str(index + 1) * 32,
            scheduler_observation=NS(
                run="123", attempt="1-fixture-" + "a" * 32, worker=role.value, trace_dir=tmp_path
            ),
        )
        for index, role in enumerate((WorkerType.PREFILL, WorkerType.DECODE))
    ]

    def post(url, **kwargs):
        sent.update(kwargs)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(
        probe, "observe_request", lambda d, w, i, s, r: sent.update(observed=i.request_id) or {}
    )
    cls = namespace[probe.PROBE]
    cls._probe = context, state, "/cpu-stub/vllm"
    gateway = NS(base_url="http://cpu", prefill_workers=workers[:1], decode_workers=workers[1:])
    cls().test_real_startup_and_admission(("pd_http", "cpu", None, gateway))
    assert state["request_id"] == sent["headers"]["x-request-id"] == sent["observed"]
    assert [entry["worker_instance"] for entry in state["workers"]] == [
        w.observer_instance for w in workers
    ]
    assert sent["json"]["max_tokens"] == 8 and sent["json"]["n"] == 1 and state["error"] is None


@pytest.mark.parametrize("policy", [None, False, "0", 2])
def test_actual_effective_retry_policy_must_be_integer_zero(configured, monkeypatch, policy):
    monkeypatch.setattr(pytest_rerunfailures, "get_reruns_count", lambda _: policy)
    with pytest.raises(ValueError, match="effective probe rerun"):
        probe.explicit_context(configured[0])
    assert not configured[3].exists() and not configured[2]


@pytest.mark.parametrize("count", [0, 2])
def test_only_one_actual_class_item_may_enable_observation(configured, count):
    configured[0].session.items *= count
    with pytest.raises(ValueError, match="exactly one"):
        probe.explicit_context(configured[0])
    assert not configured[3].exists() and not configured[2]
