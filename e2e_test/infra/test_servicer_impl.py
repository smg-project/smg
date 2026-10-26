"""Which gRPC servicer implementation a lane's workers run: the lane's own
setting when it has one, else the servicer package's default read from the
package's own switch, never a belief about what unset means."""

from __future__ import annotations

import sys
import types

import pytest
from infra import constants
from infra.constants import (
    ENV_VLLM_MM_PROCESSOR,
    ENV_VLLM_SERVICER_IMPL,
    SERVICER_IMPL_VARIABLES,
    get_servicer_impl,
    get_servicer_impl_default,
    get_servicer_impl_setting,
    get_vllm_mm_processor,
)

ENGINES = sorted(SERVICER_IMPL_VARIABLES)


def _package_default(monkeypatch, impl: str | None) -> None:
    """Pin what the servicer package's switch answers for a launch that names
    nothing (``impl``), or make the package absent (None)."""
    switch = None if impl is None else (lambda variable, environ=None, **_: (impl, "default"))
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: switch)


def _install_fake_servicer_package(monkeypatch, default: str) -> list[tuple[str, dict]]:
    """A stand-in ``smg_grpc_servicer`` whose ``rust_lifecycle.servicer_switch``
    answers ``default`` for an empty environment, recording every call."""
    calls: list[tuple[str, dict]] = []

    def servicer_switch(env_name, args=None, environ=None, **_):
        calls.append((env_name, dict(environ or {})))
        value = (environ or {}).get(env_name)
        return (value or default), ("env" if value else "default")

    package = types.ModuleType("smg_grpc_servicer")
    package.__path__ = []  # a package, so the submodule import resolves through sys.modules
    lifecycle = types.ModuleType("smg_grpc_servicer.rust_lifecycle")
    monkeypatch.setattr(lifecycle, "servicer_switch", servicer_switch, raising=False)
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer", package)
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer.rust_lifecycle", lifecycle)
    return calls


@pytest.fixture(autouse=True)
def _no_lane_setting(monkeypatch):
    for setting, _variable in SERVICER_IMPL_VARIABLES.values():
        monkeypatch.delenv(setting, raising=False)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("raw", ["", "  ", "rust", "Python", " RUST "])
def test_the_lane_setting_is_read_trimmed_and_lowercased_or_none(monkeypatch, engine, raw):
    setting, _variable = SERVICER_IMPL_VARIABLES[engine]
    monkeypatch.setenv(setting, raw)
    expected = raw.strip().lower() or None
    assert get_servicer_impl_setting(engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unrecognized_lane_setting_raises_naming_the_variable(monkeypatch, engine):
    setting, _variable = SERVICER_IMPL_VARIABLES[engine]
    monkeypatch.setenv(setting, "go")
    with pytest.raises(ValueError, match=setting):
        get_servicer_impl_setting(engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("default", ["python", "rust"])
def test_the_package_default_is_read_from_the_packages_own_switch(monkeypatch, engine, default):
    """The real import path: the package's switch is asked about an empty
    environment under the engine's own variable name, so whatever the package
    ships as its default is what the harness believes."""
    calls = _install_fake_servicer_package(monkeypatch, default)
    _setting, variable = SERVICER_IMPL_VARIABLES[engine]
    assert get_servicer_impl_default(engine) == default
    assert get_servicer_impl(engine) == default
    assert calls == [(variable, {}), (variable, {})]


@pytest.mark.parametrize("engine", ENGINES)
def test_an_absent_package_decides_nothing(monkeypatch, engine):
    monkeypatch.setitem(sys.modules, "smg_grpc_servicer", None)  # import raises ImportError
    monkeypatch.delitem(sys.modules, "smg_grpc_servicer.rust_lifecycle", raising=False)
    assert get_servicer_impl_default(engine) is None
    assert get_servicer_impl(engine) is None


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("package_default", ["python", "rust", None])
@pytest.mark.parametrize("lane_setting", ["python", "rust"])
def test_the_lane_setting_wins_over_the_package_default(
    monkeypatch, engine, package_default, lane_setting
):
    _package_default(monkeypatch, package_default)
    setting, _variable = SERVICER_IMPL_VARIABLES[engine]
    monkeypatch.setenv(setting, lane_setting)
    assert get_servicer_impl(engine) == lane_setting


@pytest.mark.parametrize("package_default", ["python", "rust", None])
def test_the_effective_implementation_is_the_package_default_when_the_lane_sets_nothing(
    monkeypatch, package_default
):
    _package_default(monkeypatch, package_default)
    for engine in ENGINES:
        assert get_servicer_impl(engine) == package_default


def test_the_smg_media_processor_needs_the_rust_servicer_by_setting_or_by_default(monkeypatch):
    """``smg``'s pipeline runs inside the Rust servicer: the check reads the
    implementation the workers will run, so a lane on the package's Rust
    default passes without naming the servicer, and a lane that runs the
    Python servicer (explicitly, or by a Python default) is refused."""
    monkeypatch.setenv(ENV_VLLM_MM_PROCESSOR, "smg")
    _package_default(monkeypatch, "rust")
    assert get_vllm_mm_processor() == "smg"
    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "python")
    with pytest.raises(ValueError, match="Rust servicer"):
        get_vllm_mm_processor()
    monkeypatch.delenv(ENV_VLLM_SERVICER_IMPL)
    _package_default(monkeypatch, "python")
    with pytest.raises(ValueError, match="Rust servicer"):
        get_vllm_mm_processor()
    monkeypatch.setenv(ENV_VLLM_SERVICER_IMPL, "rust")
    assert get_vllm_mm_processor() == "smg"
