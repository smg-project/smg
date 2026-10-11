"""The scheduler pin the installed TokenSpeed runtime nightly implies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    monkeypatch.setattr(mod.metadata, "requires", lambda name: NIGHTLY)
    assert mod.main() == 0
    assert capsys.readouterr().out == "tokenspeed-scheduler==0.1.26\n"


def test_main_prints_nothing_without_a_bound(mod, monkeypatch, capsys):
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
