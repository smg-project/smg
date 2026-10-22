"""CPU plugin/child discovery and selector contracts, not a real vLLM startup."""

import hashlib
import io
import json
import os
import subprocess
import sys
from functools import wraps
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import smg_pd_observer as plugin
from infra import pd_scheduler_compat as compat


def context(directory):
    return {
        f"SMG_PD_TRACE_{key}": value
        for key, value in {
            "RUN": "cpu",
            "ATTEMPT": "1",
            "WORKER": "p-0",
            "ROLE": "prefill",
            "DIR": str(directory),
        }.items()
    }


@pytest.mark.parametrize("allowlist", [None, "foo,bar", "", f"foo,{plugin.PLUGIN},bar"])
def test_child_env_preserves_default_plugins_allowlist_and_pythonpath(tmp_path, allowlist):
    base = {**context(tmp_path), "PYTHONPATH": "keep::order"}
    if allowlist is not None:
        base["VLLM_PLUGINS"] = allowlist
    assert plugin.child_environment(base) == base
    env = plugin.child_environment(base, enabled=True)
    assert env["PYTHONPATH"].endswith(os.pathsep + base["PYTHONPATH"])
    assert plugin.child_environment(env, enabled=True) == env
    if allowlist is None:
        assert "VLLM_PLUGINS" not in env
    else:
        expected = (
            allowlist
            if plugin.PLUGIN in allowlist.split(",")
            else (allowlist + ("," if allowlist else "") + plugin.PLUGIN)
        )
        assert env["VLLM_PLUGINS"] == expected
    assert "SMG_PD_OBSERVE" not in base


def test_real_child_discovers_metadata_and_disabled_callback_imports_no_backend(tmp_path):
    env = plugin.child_environment({**os.environ, **context(tmp_path)}, enabled=True)
    env.pop("SMG_PD_OBSERVE")
    code = """
import importlib.metadata, json, sys
points = [p for p in importlib.metadata.entry_points(group='vllm.general_plugins')
          if p.name == 'smg_pd_scheduler_observer']
assert len(points) == 1
points[0].load()()
assert not any(n.split('.')[0] in ('infra', 'vllm', 'torch') for n in sys.modules)
print(json.dumps({'entrypoint': points[0].value, 'backend_loaded': False}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert json.loads(result.stdout)["entrypoint"] == "smg_pd_observer:register"


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_child_opt_in_requires_explicit_bool(value):
    with pytest.raises(ValueError, match="explicit bool"):
        plugin.child_environment({}, enabled=value)


def test_enabled_missing_context_fails_before_loading_runtime(monkeypatch):
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    monkeypatch.delenv("SMG_PD_TRACE_RUN", raising=False)
    monkeypatch.setattr(plugin, "_runtime", lambda: pytest.fail("must not load runtime"))
    with pytest.raises(RuntimeError, match="missing explicit"):
        plugin.register()


def test_conditional_loader_hash_and_origins_are_checked(monkeypatch, tmp_path):
    monkeypatch.setattr(compat, "require_installed_contract", lambda: {"root": str(tmp_path)})
    for name, relative in (
        ("vllm.plugins", "plugins/__init__.py"),
        ("vllm.config.scheduler", "config/scheduler.py"),
    ):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / relative)
        module.__spec__ = ModuleSpec(name, None, origin=module.__file__)
        module.SchedulerConfig = object()
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "plugins/__init__.py"
    path.parent.mkdir()
    path.write_bytes(b"CPU loader fixture")
    monkeypatch.setattr(plugin, "LOADER_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert plugin._runtime()[1] == tmp_path
    path.write_bytes(b"changed loader")
    with pytest.raises(RuntimeError, match="unknown installed plugin-loader source"):
        plugin._runtime()
    sys.modules["vllm.plugins"].__spec__.origin = "/foreign/plugins.py"
    with pytest.raises(RuntimeError, match="unknown observer adapter module origin"):
        plugin._runtime()


def test_native_getter_once_only_final_constructor_maps_and_errors_keep_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    native, observed = object(), object()
    calls, choices = [], []
    error = RuntimeError("native getter")

    def original(config):
        calls.append(config)
        if config.fail:
            raise error
        return native

    def choose(value, config):
        choices.append((value, config))
        return observed

    monkeypatch.setattr(plugin, "_choose", choose)
    select = plugin._selector(original, tmp_path)
    namespace = {"select": select}
    exec(
        compile(
            "class EngineCore:\n def __init__(self, config):\n  self.selected = select(config)\n",
            str(tmp_path / "v1/engine/core.py"),
            "exec",
        ),
        namespace,
    )
    core = ModuleType("vllm.v1.engine.core")
    core.EngineCore = namespace["EngineCore"]
    monkeypatch.setitem(sys.modules, core.__name__, core)
    config = SimpleNamespace(fail=False, async_scheduling=True)
    assert select(config) is native and choices == []
    assert plugin.last_unknown is not None
    assert core.EngineCore(config).selected is observed
    assert choices == [(native, config)] and len(calls) == 2
    config.fail = True
    with pytest.raises(RuntimeError) as caught:
        core.EngineCore(config)
    assert caught.value is error and len(calls) == 3
    config.fail = False
    monkeypatch.delenv("SMG_PD_OBSERVE")
    assert core.EngineCore(config).selected is native and len(choices) == 1
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    old = core.EngineCore.__init__
    core.EngineCore.__init__ = lambda self, cfg: old(self, cfg)
    with pytest.raises(RuntimeError, match="unknown engine constructor origin"):
        core.EngineCore(config)


def test_plugin_registration_is_idempotent_and_rejects_foreign_getter(monkeypatch, tmp_path):
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    for key, value in context(tmp_path).items():
        monkeypatch.setenv(key, value)
    namespace = {"__name__": "vllm.config.scheduler"}
    exec(
        compile(
            "class Config:\n def get_scheduler_cls(self): return None\n",
            str(tmp_path / "config/scheduler.py"),
            "exec",
        ),
        namespace,
    )
    config_type = namespace["Config"]
    original = config_type.get_scheduler_cls
    monkeypatch.setattr(plugin, "_runtime", lambda: (config_type, tmp_path))
    plugin.register()
    installed = config_type.get_scheduler_cls
    assert installed.__smg_pd_original__ is original
    plugin.register()
    assert config_type.get_scheduler_cls is installed
    config_type.get_scheduler_cls = wraps(installed)(lambda self: installed(self))
    with pytest.raises(RuntimeError, match="unknown observer selector wrapper"):
        plugin.register()
    config_type.get_scheduler_cls = lambda self: None
    with pytest.raises(RuntimeError, match="unknown original scheduler selector"):
        plugin.register()


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_writer_failure_does_not_replace_native_result(monkeypatch, tmp_path, missing):
    monkeypatch.setenv("SMG_PD_OBSERVE", "1")
    native = object()
    closed = io.StringIO()
    closed.close()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", None if missing else closed)
        assert plugin._selector(lambda config: native, tmp_path)(object()) is native
    assert plugin.last_unknown is not None
