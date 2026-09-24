"""The Helm chart's router arguments must be options the launcher accepts.

The gateway image's entrypoint is ``python3 -m smg.launch_router``, so every
``--flag`` the chart's ``smg.routerArgs`` template emits has to be an exact
option of the ``RouterArgs`` parser. argparse accepts unambiguous
abbreviations by default, which let a few chart flags through by accident;
this check uses exact names.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
from smg.router_args import RouterArgs

HELPERS = (
    Path(__file__).resolve().parents[3] / "deploy" / "helm" / "smg" / "templates" / "_helpers.tpl"
)


def router_args_flags() -> set[str]:
    text = HELPERS.read_text()
    start = text.index('{{- define "smg.routerArgs" -}}')
    end = text.index("{{- define", start + 1)
    return set(re.findall(r'"(--[a-z0-9-]+)"', text[start:end]))


@pytest.mark.skipif(not HELPERS.exists(), reason="chart not present in this checkout")
def test_chart_router_flags_are_launcher_options():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    RouterArgs.add_cli_args(parser)
    known = set(parser._option_string_actions)
    flags = router_args_flags()
    assert flags, "no flags found in smg.routerArgs"
    unknown = sorted(flags - known)
    assert not unknown, f"the chart passes flags the launcher does not accept: {unknown}"
