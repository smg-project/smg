"""The scheduler pin the installed TokenSpeed runtime nightly implies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_tokenspeed_scheduler_pin.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("ci_tokenspeed_scheduler_pin", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_tokenspeed_scheduler_pin"] = module
    spec.loader.exec_module(module)
    return module


NIGHTLY = [
    "tokenspeed-kernel==0.1.4.post20261009",
    "tokenspeed-mooncake>=0.3.13.post20260929",
    "tokenspeed-scheduler>=0.1.26",
    "tokenspeed-smg==1.10.1.post20260920",
]


def test_lower_bound_becomes_the_pin(mod):
    assert mod.scheduler_pin(NIGHTLY) == "tokenspeed-scheduler==0.1.26"


def test_exact_requirement_is_kept(mod):
    assert mod.scheduler_pin(["tokenspeed-scheduler==0.1.29"]) == "tokenspeed-scheduler==0.1.29"


def test_lower_bound_wins_over_an_upper_bound(mod):
    assert mod.scheduler_pin(["tokenspeed-scheduler >=0.1.27, <0.2"]) == (
        "tokenspeed-scheduler==0.1.27"
    )


def test_name_spelling_and_markers_do_not_matter(mod):
    assert mod.scheduler_pin(['Tokenspeed_Scheduler>=0.1.28; python_version >= "3.10"']) == (
        "tokenspeed-scheduler==0.1.28"
    )


@pytest.mark.parametrize(
    "requirements",
    [
        [],
        None,
        ["tokenspeed-kernel==0.1.4.post20261009"],
        ["tokenspeed-scheduler"],
        ["tokenspeed-scheduler<0.2"],
        ["tokenspeed-scheduler~=0.1.26"],
    ],
)
def test_no_lower_bound_means_no_pin(mod, requirements):
    assert mod.scheduler_pin(requirements) is None


def test_main_prints_the_pin(mod, monkeypatch, capsys):
    monkeypatch.setattr(mod.metadata, "version", lambda _: "0.1.1.post20261009")
    monkeypatch.setattr(mod.metadata, "requires", lambda name: NIGHTLY)
    assert mod.main() == 0
    assert capsys.readouterr().out == "tokenspeed-scheduler==0.1.26\n"


def test_main_prints_nothing_without_a_bound(mod, monkeypatch, capsys):
    monkeypatch.setattr(mod.metadata, "version", lambda _: "0.1.1.post20261009")
    monkeypatch.setattr(mod.metadata, "requires", lambda name: ["tokenspeed-scheduler"])
    assert mod.main() == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "no lower bound" in err


def test_main_fails_without_the_runtime(mod, monkeypatch, capsys):
    def missing(name):
        raise mod.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(mod.metadata, "requires", missing)
    assert mod.main() == 1
    assert capsys.readouterr().out == ""


def test_storage_nightly_uses_known_pair_without_changing_old_nightlies(mod):
    requirements = ["tokenspeed-scheduler>=0.1.41"]
    assert mod.scheduler_pin(requirements, "0.1.1.post20261023") == "tokenspeed-scheduler==0.1.42"
    assert mod.scheduler_pin(requirements, "0.1.1.post20261021") == "tokenspeed-scheduler==0.1.41"
    assert mod.scheduler_pin(requirements, "0.1.1.post20261024") == "tokenspeed-scheduler==0.1.41"


def test_known_pair_cannot_override_declared_constraints(mod):
    assert mod.scheduler_pin(["tokenspeed-scheduler==0.1.41"], "0.1.1.post20261023") == (
        "tokenspeed-scheduler==0.1.41"
    )
    with pytest.raises(ValueError, match="conflicts"):
        mod.scheduler_pin(["tokenspeed-scheduler>=0.1.41,<0.1.42"], "0.1.1.post20261023")


def test_storage_nightly_rejects_wrong_version_with_all_symbols_present(mod, monkeypatch):
    monkeypatch.setattr(mod.metadata, "version", lambda _: "0.1.41")
    module = SimpleNamespace(
        Scheduler=object(), StorageConfig=object(), StoragePressure=object(), StorageKind=object()
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda _: module)
    with pytest.raises(RuntimeError, match="requires the CI scheduler pair 0.1.42"):
        mod.check_scheduler("0.1.1.post20261023")


def test_storage_nightly_rejects_missing_storage_config_at_correct_version(mod, monkeypatch):
    monkeypatch.setattr(mod.metadata, "version", lambda _: "0.1.42")
    module = SimpleNamespace(Scheduler=object(), StoragePressure=object(), StorageKind=object())
    monkeypatch.setattr(mod.importlib, "import_module", lambda _: module)
    with pytest.raises(ImportError, match="lacks StorageConfig"):
        mod.check_scheduler("0.1.1.post20261023")


def test_storage_nightly_accepts_correct_version_with_all_symbols_present(mod, monkeypatch):
    monkeypatch.setattr(mod.metadata, "version", lambda _: "0.1.42")
    module = SimpleNamespace(
        Scheduler=object(), StorageConfig=object(), StoragePressure=object(), StorageKind=object()
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda _: module)
    assert mod.check_scheduler("0.1.1.post20261023") is None


def test_old_nightly_checks_its_scheduler_without_requiring_storage(mod, monkeypatch):
    monkeypatch.setattr(
        mod.importlib, "import_module", lambda _: SimpleNamespace(Scheduler=object())
    )
    mod.check_scheduler("0.1.1.post20261009")
