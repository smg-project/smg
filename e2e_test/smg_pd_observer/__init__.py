"""Opt-in test plugin. Importing this module never imports infra or a backend."""

import hashlib
import importlib
import os
import sys
from functools import wraps
from pathlib import Path

PLUGIN = "smg_pd_scheduler_observer"
# Official db9527a source; equality with the installed image is not yet verified.
LOADER_SHA256 = "06942d64e526c2c7407265e1903d625c4d2118d8d7fc39c616b1cc6aa6fc2c8f"
last_unknown = None


def _context(env):
    if any(not env.get(f"SMG_PD_TRACE_{key}") for key in ("RUN", "ATTEMPT", "WORKER")):
        raise RuntimeError("missing explicit observer context")
    if env.get("SMG_PD_TRACE_ROLE") not in ("prefill", "decode"):
        raise RuntimeError("unknown observer role")
    directory = Path(env.get("SMG_PD_TRACE_DIR", ""))
    if not directory.is_absolute() or not directory.is_dir():
        raise RuntimeError("observer trace directory must exist and be absolute")


def child_environment(base, *, enabled=False):
    env = dict(base)
    if type(enabled) is not bool:
        raise ValueError("observer opt-in must be an explicit bool")
    if not enabled:
        return env
    _context(env)
    root = Path(__file__).resolve().parents[1]
    prefix = os.pathsep.join((str(root), str(root / "pd_observer_plugins")))
    previous = env.get("PYTHONPATH")
    if previous != prefix and not (previous and previous.startswith(prefix + os.pathsep)):
        env["PYTHONPATH"] = prefix + (os.pathsep + previous if previous is not None else "")
    if "VLLM_PLUGINS" in env and PLUGIN not in env["VLLM_PLUGINS"].split(","):
        env["VLLM_PLUGINS"] += ("," if env["VLLM_PLUGINS"] else "") + PLUGIN
    env["SMG_PD_OBSERVE"] = "1"
    return env


def _runtime():
    from infra.pd_scheduler_compat import require_installed_contract

    contract = require_installed_contract()
    root = Path(contract["root"])
    config = importlib.import_module("vllm.config.scheduler")
    for name, relative in (
        ("vllm.plugins", "plugins/__init__.py"),
        ("vllm.config.scheduler", "config/scheduler.py"),
    ):
        module = sys.modules.get(name)
        paths = (
            getattr(module, "__file__", None),
            getattr(getattr(module, "__spec__", None), "origin", None),
        )
        if any(path is None or Path(path).resolve() != root / relative for path in paths):
            raise RuntimeError(f"unknown observer adapter module origin: {name}")
    try:
        loader_hash = hashlib.sha256((root / "plugins/__init__.py").read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("unknown installed plugin-loader source") from exc
    if loader_hash != LOADER_SHA256:
        raise RuntimeError("unknown installed plugin-loader source")
    return config.SchedulerConfig, root


def _choose(native, config):
    from infra.pd_vllm_scheduler import observer_class_for_native

    return observer_class_for_native(native, config)


def _unknown(reason):
    global last_unknown
    last_unknown = reason
    try:
        sys.stderr.write(f"SMG_PD_OBSERVER_UNKNOWN: {reason}\n")
    except Exception:
        pass


def _selector(original, root):
    @wraps(original)
    def select(config):
        native = original(config)
        if os.environ.get("SMG_PD_OBSERVE") != "1":
            return native
        caller = sys._getframe(1).f_code
        core = sys.modules.get("vllm.v1.engine.core")
        init = getattr(getattr(core, "EngineCore", None), "__init__", None)
        code = getattr(init, "__code__", None)
        if code is not None and Path(code.co_filename).resolve() != root / "v1/engine/core.py":
            raise RuntimeError("unknown engine constructor origin")
        if caller is not code:
            if Path(caller.co_filename).resolve() == root / "v1/engine/core.py":
                raise RuntimeError("unknown replaced engine constructor")
            _unknown("selector call outside verified EngineCore constructor")
            return native
        return _choose(native, config)

    select.__smg_pd_original__ = original
    return select


def register():
    if os.environ.get("SMG_PD_OBSERVE") != "1":
        return
    _context(os.environ)
    config_type, root = _runtime()
    original = config_type.get_scheduler_cls
    if getattr(original, "__smg_pd_original__", None) is not None:
        if original.__globals__ is not globals():
            raise RuntimeError("unknown observer selector wrapper")
        return
    if (
        original.__module__ != "vllm.config.scheduler"
        or Path(original.__code__.co_filename).resolve() != root / "config/scheduler.py"
    ):
        raise RuntimeError("unknown original scheduler selector")
    config_type.get_scheduler_cls = _selector(original, root)
