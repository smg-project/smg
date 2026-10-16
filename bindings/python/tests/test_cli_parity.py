"""The launcher's options are the Rust CLI's flags, pinned against the CLI's source.

``RouterArgs.add_cli_args`` mirrors ``struct CliArgs`` in ``model_gateway/src/main.rs``:
one option per flag, the same spelling and the same second spellings (plus
``--prefill`` and ``--encode``, which the CLI reads ahead of clap). These tests read
the clap struct from the source tree, so a flag added on one side without the
other fails here instead of at a launch line; on an installed wheel without the
source they skip. The exception lists below carry the reason for every divergence
left, and an entry that no longer holds fails too, so the lists cannot rot.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from smg.router_args import RouterArgs

MAIN_RS = Path(__file__).resolve().parents[3] / "model_gateway" / "src" / "main.rs"

# Flags of the Rust CLI the launcher does not take, with the reason.
CLI_ONLY = {
    "--disable-audit-logging": "the launcher's switch is --control-plane-audit-enabled",
    "--engine-metrics": "the binding's Router has no parameter for it yet",
    "--jwt-role-claim": "the binding's JWT config has no role-claim field yet",
    "--kv-indexer-max-entries": "deprecated and ignored by the CLI; the launcher never had it",
    "--kv-indexer-ttl-secs": "deprecated and ignored by the CLI; the launcher never had it",
    "--pd-pairing-mode": "the binding's Router has no parameter for it yet",
    "--runtime-worker-threads": "the binding's Router has no parameter for it yet",
    "--webrtc-bind-addr": "the binding's Router has no parameter for it yet",
    "--webrtc-stun-server": "the binding's Router has no parameter for it yet",
}

# Options of the launcher with no Rust CLI flag of that spelling, with the reason.
LAUNCHER_ONLY = {
    "--bucket-adjust-interval-secs": "a Router parameter the Rust CLI leaves at its default",
    "--ca-cert-paths": "worker mTLS: a Router parameter the Rust CLI does not expose",
    "--client-cert-path": "worker mTLS: a Router parameter the Rust CLI does not expose",
    "--client-key-path": "worker mTLS: a Router parameter the Rust CLI does not expose",
    "--control-plane-audit-enabled": "the launcher's switch; the CLI's is --disable-audit-logging",
    "--eviction-interval-secs": "the launcher's first spelling of --eviction-interval",
    "--no-remove-unhealthy-workers": "argparse's negative form; the CLI takes =false",
    "--no-worker-auto-recovery": "argparse's negative form; the CLI takes =false",
    "--oracle-connect-descriptor": "the launcher's first spelling of --oracle-dsn",
    "--oracle-username": "the launcher's first spelling of --oracle-user",
    "--postgres-pool-max": "the launcher's first spelling of --postgres-pool-max-size",
    "--redis-pool-max": "the launcher's first spelling of --redis-pool-max-size",
}

# Flags on both sides whose literal defaults differ, with the reason.
DEFAULT_DIFFERS = {
    "--eviction-interval": "the launcher's 60 s predates the CLI's 120 s; kept until reconciled",
}


@dataclass
class RustCli:
    flags: dict[str, object] = field(default_factory=dict)  # primary spelling -> default
    aliases: dict[str, str] = field(default_factory=dict)  # second spelling -> primary
    pre_clap: set[str] = field(default_factory=set)  # read from argv ahead of clap

    @property
    def spellings(self) -> set[str]:
        return set(self.flags) | set(self.aliases) | self.pre_clap


def literal_default(attribute: str) -> object:
    """The clap default as a Python value, or None when there is none or it is
    not a literal (an expression the launcher cannot be compared with)."""
    quoted = re.search(r'\bdefault_value\s*=\s*"([^"]*)"', attribute)
    if quoted:
        return quoted.group(1)
    # The capture stops at the next argument, or at the attribute's own
    # closing `)]` when `default_value_t` is its last argument.
    typed = re.search(r"\bdefault_value_t\s*=\s*([^,)\]]+)", attribute)
    if not typed:
        return None
    raw = typed.group(1).strip().replace("_", "")
    raw = re.sub(r"(?<=\d)(u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|isize|f32|f64)$", "", raw)
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return None


def parse_rust_cli(text: str) -> RustCli:
    """The flags of ``struct CliArgs``: each ``#[arg(...)]`` attribute and the field
    it annotates, the long flag being the explicit ``long = ".."`` or the field
    name with dashes, the second spellings its ``alias``/``visible_alias``."""
    lines = text.splitlines()
    start = lines.index("struct CliArgs {")
    end = lines.index("}", start)
    cli = RustCli(pre_clap=set(re.findall(r'args\[i\] == "(--[a-z-]+)"', text)))
    i = start + 1
    while i < end:
        if not lines[i].lstrip().startswith("#[arg("):
            i += 1
            continue
        attribute = lines[i].strip()
        depth = attribute.count("(") - attribute.count(")")
        while depth > 0 or not attribute.endswith("]"):
            i += 1
            attribute += " " + lines[i].strip()
            depth += lines[i].count("(") - lines[i].count(")")
        i += 1
        while not lines[i].strip() or lines[i].lstrip().startswith(("#", "/")):
            i += 1
        declared = re.match(r"\s*(?:pub\s+)?([a-z0-9_]+)\s*:", lines[i])
        assert declared is not None, f"no field after the attribute ending at line {i}"
        name = declared.group(1)
        explicit = re.search(r'\blong\s*=\s*"([^"]+)"', attribute)
        flag = "--" + (explicit.group(1) if explicit else name.replace("_", "-"))
        cli.flags[flag] = literal_default(attribute)
        for alias in re.findall(r'\b(?:visible_)?alias\s*=\s*"([^"]+)"', attribute):
            cli.aliases["--" + alias] = flag
        i += 1
    return cli


def launcher_options(use_router_prefix: bool = False) -> dict[str, argparse.Action]:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    RouterArgs.add_cli_args(parser, use_router_prefix=use_router_prefix)
    return {
        option: action
        for option, action in parser._option_string_actions.items()
        if option.startswith("--")
    }


@pytest.fixture(scope="module")
def rust() -> RustCli:
    if not MAIN_RS.exists():
        pytest.skip("the Rust CLI's source is not in this checkout")
    cli = parse_rust_cli(MAIN_RS.read_text())
    assert len(cli.flags) > 150, "struct CliArgs parsed short: the parser needs updating"
    assert cli.pre_clap == {"--prefill", "--encode"}, (
        "the CLI's pre-clap flags changed: update the parser"
    )
    return cli


@pytest.fixture(scope="module")
def launcher() -> dict[str, argparse.Action]:
    return launcher_options()


def test_every_cli_flag_is_a_launcher_option(rust, launcher):
    missing = sorted(set(rust.flags) - set(launcher) - set(CLI_ONLY))
    assert not missing, (
        "Rust CLI flags the launcher does not take; add the option to"
        f" RouterArgs.add_cli_args, or list it in CLI_ONLY with the reason: {missing}"
    )


def test_every_cli_second_spelling_is_a_launcher_option(rust, launcher):
    missing = sorted(alias for alias in rust.aliases if alias not in launcher)
    assert not missing, (
        "second spellings of the Rust CLI the launcher does not take; add each as a"
        f" further option string of its option: {missing}"
    )


def test_every_launcher_option_is_a_cli_flag(rust, launcher):
    extra = sorted(set(launcher) - rust.spellings - set(LAUNCHER_ONLY))
    assert not extra, (
        "launcher options with no Rust CLI flag of that spelling; add the flag to the"
        f" CLI, or list the option in LAUNCHER_ONLY with the reason: {extra}"
    )


def test_exception_lists_are_current(rust, launcher):
    stale = [flag for flag in CLI_ONLY if flag not in rust.flags or flag in launcher]
    stale += [
        option for option in LAUNCHER_ONLY if option not in launcher or option in rust.spellings
    ]
    assert not stale, f"exceptions that no longer hold; drop them: {stale}"


def test_shared_flags_share_their_literal_defaults(rust, launcher):
    differs = {}
    for flag, cli_default in rust.flags.items():
        if cli_default is None or flag not in launcher:
            continue
        launcher_default = launcher[flag].default
        if launcher_default is None or isinstance(launcher_default, list):
            continue  # resolved later (a tri-state, the dataclass, the Rust core)
        if isinstance(cli_default, bool) or isinstance(launcher_default, bool):
            same = cli_default is launcher_default
        elif isinstance(cli_default, (int, float)) and isinstance(launcher_default, (int, float)):
            same = abs(float(cli_default) - float(launcher_default)) < 1e-9
        else:
            same = str(cli_default) == str(launcher_default)
        if not same:
            differs[flag] = (cli_default, launcher_default)
    unexpected = {flag: pair for flag, pair in differs.items() if flag not in DEFAULT_DIFFERS}
    assert not unexpected, f"defaults that differ from the Rust CLI's (CLI, launcher): {unexpected}"
    stale = [flag for flag in DEFAULT_DIFFERS if flag not in differs]
    assert not stale, f"defaults that no longer differ; drop them: {stale}"


def test_every_option_has_its_router_prefixed_spelling():
    plain = launcher_options()
    prefixed = launcher_options(use_router_prefix=True)
    missing = []
    for option in plain:
        if option in ("--host", "--port"):
            continue  # inherited from the serving command when embedded
        expected = (
            "--no-router-" + option[5:] if option.startswith("--no-") else "--router-" + option[2:]
        )
        if expected not in prefixed and option not in prefixed:
            missing.append(option)
    assert not missing, f"options without a --router- spelling: {missing}"
