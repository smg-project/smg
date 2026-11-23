"""The launcher's options are the Rust CLI's flags, pinned against the CLI's source.

``RouterArgs.add_cli_args`` mirrors the Rust arguments in ``model_gateway/src/cli/``:
one option per flag, with the same spelling and aliases. These tests read the
clap structs and the pre-clap flags (``--prefill`` and ``--encode``) from the
source tree, so a flag added on one side without the other fails here instead of at
a launch line. Comparisons using the ``rust`` fixture skip when the gateway source
tree is absent. The exception lists below give the reason for each difference, and
each listed difference still has to occur.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from smg.router_args import DOCUMENT_REPLACES, RouterArgs

ROOT = Path(__file__).resolve().parents[3]
GATEWAY_SRC = ROOT / "model_gateway" / "src"
CLI_DIR = GATEWAY_SRC / "cli"
CLI_MODULE = CLI_DIR / "mod.rs"
NATIVE_MODULE = ROOT / "bindings" / "python" / "src" / "lib.rs"


def cli_source() -> str:
    """The command-line module and the source files beside it."""
    return "\n".join(p.read_text() for p in sorted(CLI_DIR.glob("*.rs")))


def test_the_executable_takes_the_stock_allocator_in_one_line():
    if not GATEWAY_SRC.is_dir():
        pytest.skip("the Rust CLI's source is not in this checkout")
    executable = (GATEWAY_SRC / "main.rs").read_text()
    library = (GATEWAY_SRC / "lib.rs").read_text()
    cli = cli_source()
    registration = "register_jemalloc_as_global_allocator();"

    assert "smg::stock_allocator!();" in executable
    assert "smg::main_from_args()" in executable
    # No call registers the allocator: the library finds out which one serves.
    assert registration not in executable
    assert "pub fn main_from_args()" in library
    assert registration not in re.sub(r"(?m)^\s*//.*$", "", library)
    assert registration not in cli
    assert "#[global_allocator]" not in cli


# Flags of the Rust CLI the launcher does not take, with the reason.
CLI_ONLY = {
    "--disable-audit-logging": "the launcher's switch is --control-plane-audit-enabled",
    "--print-policy-spec": "the binary prints its policy document and exits; the launcher does not",
}

# Options of the launcher with no Rust CLI flag of that spelling, with the reason.
LAUNCHER_ONLY = {
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
DEFAULT_DIFFERS: dict[str, str] = {}


@dataclass
class RustCli:
    flags: dict[str, object] = field(default_factory=dict)  # primary spelling -> default
    aliases: dict[str, str] = field(default_factory=dict)  # second spelling -> primary
    pre_clap: set[str] = field(default_factory=set)  # read from argv ahead of clap
    attributes: int = 0  # #[arg(...)] attributes read, flags and skipped fields alike

    @property
    def spellings(self) -> set[str]:
        return set(self.flags) | set(self.aliases) | self.pre_clap


# Where a named CLI default is declared when the CLI sources do not declare it
# themselves: the config types and server modules, and the worker settings whose
# constants the config types re-export.
CONST_SOURCES = (
    GATEWAY_SRC / "config" / "types.rs",
    GATEWAY_SRC / "server.rs",
    ROOT / "crates" / "worker" / "src" / "settings.rs",
)


@dataclass(frozen=True)
class UnreadDefault:
    """A clap default the reader could not turn into a value: an expression, or a
    name no source declares as a constant. The default tests report it by name
    instead of skipping the flag in silence."""

    raw: str


def named_constant(name: str, *sources: str) -> str | None:
    """The literal of ``pub const <name>: T = <literal>;`` in the first of ``sources``
    that declares it, or None when none does."""
    pattern = re.compile(rf"\bconst\s+{re.escape(name)}\s*:[^=;]+=\s*([^;]+);")
    for text in sources:
        declared = pattern.search(text)
        if declared:
            return declared.group(1).strip()
    return None


def literal_value(raw: str) -> object:
    """A Rust literal as a Python value (a bool, an int or a float, the digit
    separators and the type suffix dropped), or None when the text is not one."""
    raw = raw.replace("_", "")
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


def literal_default(attribute: str, *sources: str) -> object:
    """The clap default as a Python value: None when the attribute carries none, the
    value of a literal or of the constant a name stands for in ``sources``, and an
    ``UnreadDefault`` for an expression the reader cannot compare."""
    quoted = re.search(r'\bdefault_value\s*=\s*"([^"]*)"', attribute)
    if quoted:
        return quoted.group(1)
    # The capture stops at the next argument, or at the attribute's own
    # closing `)]` when `default_value_t` is its last argument.
    typed = re.search(r"\bdefault_value_t\s*=\s*([^,)\]]+)", attribute)
    if not typed:
        return None
    raw = typed.group(1).strip()
    declared = named_constant(raw, *sources) if re.fullmatch(r"[A-Z][A-Z0-9_]*", raw) else None
    value = literal_value(raw if declared is None else declared)
    return UnreadDefault(raw) if value is None else value


def arg_attributes(text: str) -> int:
    """How many ``#[arg(...)]`` attributes a source holds, wherever they are."""
    return len(re.findall(r"^\s*#\[arg\(", text, re.MULTILINE))


def parse_rust_cli(
    text: str, struct_name: str = "CliArgs", constant_sources: tuple[str, ...] = ()
) -> RustCli:
    """The flags of ``struct CliArgs``: each ``#[arg(...)]`` attribute and the field
    it annotates, recursively including ``#[command(flatten)]`` argument groups.
    The long flag is the explicit ``long = ".."`` or the field
    name with dashes, the second spellings its ``alias``/``visible_alias``.
    A default naming a constant is read from ``text`` and then from
    ``constant_sources``. ``attributes`` counts every attribute read, so held
    against ``arg_attributes(text)`` it shows a group the reader did not reach."""
    lines = text.splitlines()
    header = re.compile(rf"^(\s*)(?:pub(?:\([^)]*\))?\s+)?struct\s+{struct_name}\s*\{{\s*$")
    starts = [(n, m.group(1)) for n, line in enumerate(lines) if (m := header.match(line))]
    assert len(starts) == 1, f"struct {struct_name}: {len(starts)} definitions found, one expected"
    start, indent = starts[0]
    end = lines.index(indent + "}", start)  # the struct's own brace, at its indentation
    cli = RustCli(pre_clap=set(re.findall(r'args\[i\] == "(--[a-z-]+)"', text)))
    i = start + 1
    while i < end:
        command = re.match(r"#\[(?:command|clap)\(", lines[i].strip()) is not None
        if not command and not lines[i].lstrip().startswith("#[arg("):
            i += 1
            continue
        attribute = lines[i].strip()
        depth = attribute.count("(") - attribute.count(")")
        while depth > 0 or not attribute.endswith("]"):
            i += 1
            attribute += " " + lines[i].strip()
            depth += lines[i].count("(") - lines[i].count(")")
        i += 1
        flattened = command and re.search(r"[(,]\s*flatten\s*[,)]", attribute) is not None
        if command and not flattened:
            continue
        while not lines[i].strip() or lines[i].lstrip().startswith(("#", "/")):
            i += 1
        declared = re.match(
            r"\s*(?:pub(?:\([^)]*\))?\s+)?([a-z0-9_]+)\s*:\s*(?:(?:Option|Box)<)?([A-Za-z0-9_]+)",
            lines[i],
        )
        assert declared is not None, f"no field after the attribute ending at line {i}"
        if flattened:
            group = parse_rust_cli(text, declared.group(2), constant_sources)
            cli.flags.update(group.flags)
            cli.aliases.update(group.aliases)
            cli.attributes += group.attributes
            i += 1
            continue
        cli.attributes += 1
        if re.fullmatch(r"#\[arg\(\s*skip\s*(=.*)?\)\]", attribute):
            # A field clap never reads from the command line (a record filled
            # after parsing): no flag to mirror.
            i += 1
            continue
        name = declared.group(1)
        explicit = re.search(r'\blong\s*=\s*"([^"]+)"', attribute)
        flag = "--" + (explicit.group(1) if explicit else name.replace("_", "-"))
        cli.flags[flag] = literal_default(attribute, text, *constant_sources)
        for alias in re.findall(r'\b(?:visible_)?alias\s*=\s*"([^"]+)"', attribute):
            cli.aliases["--" + alias] = flag
        i += 1
    return cli


@pytest.mark.parametrize(
    "flatten",
    [
        "#[command(flatten)]",
        '#[command(flatten, next_help_heading = "Worker")]',
        "#[clap(flatten)]",
        '#[command(\n        flatten,\n        next_help_heading = "Worker"\n    )]',
        '#[command(next_help_heading = "Worker", flatten)]',
        '#[command(flatten,\n        next_help_heading = "Worker"\n    )]',
    ],
)
def test_parse_rust_cli_follows_only_reachable_flattened_groups(flatten):
    """Flattened groups retain flags, defaults and aliases, without unrelated structs."""
    source = """
struct CliArgs {
    #[arg(long, default_value_t = 3)]
    retries: usize,
    #[command(flatten)]
    worker: WorkerArgs,
}
struct WorkerArgs {
    #[arg(long = "worker-url", visible_alias = "url")]
    endpoint: String,
    #[command(flatten)]
    nested: NestedArgs,
}
struct NestedArgs {
    #[arg(long, default_value_t = false)]
    enabled: bool,
    #[arg(skip)]
    internal: usize,
}
struct Unrelated {
    #[arg(long)]
    ignored: String,
}
"""
    cli = parse_rust_cli(source.replace("#[command(flatten)]", flatten))
    assert cli.flags == {"--retries": 3, "--worker-url": None, "--enabled": False}
    assert cli.aliases == {"--url": "--worker-url"}


def test_parse_rust_cli_accounts_for_every_attribute_it_reads():
    """Visibility and an Option/Box wrapper do not hide a group; the attribute count
    covers flags and skipped fields of every group reached, so a group left out
    shows as a shortfall against the source's attribute count."""
    source = """
pub(crate) struct CliArgs {
    #[arg(long)]
    retries: usize,
    #[command(flatten)]
    pub worker: Option<WorkerArgs>,
}
pub struct WorkerArgs {
    #[arg(long = "worker-url")]
    endpoint: String,
    #[arg(skip)]
    internal: usize,
}
struct Unrelated {
    #[arg(long)]
    ignored: String,
}
"""
    cli = parse_rust_cli(source)
    assert cli.flags == {"--retries": None, "--worker-url": None}
    assert cli.attributes == 3
    assert arg_attributes(source) == 4


def test_parse_rust_cli_ends_a_group_at_its_own_brace():
    """A group declared inside a module ends at its own closing brace, so the next
    struct of that module does not pass its flags off as the group's."""
    cli = parse_rust_cli("""
struct CliArgs {
    #[command(flatten)]
    workers: WorkerArgs,
}
mod groups {
    pub struct WorkerArgs {
        #[arg(long)]
        a: String,
    }
    pub struct Other {
        #[arg(long)]
        b: String,
    }
}
""")
    assert cli.flags == {"--a": None}
    assert cli.attributes == 1


def test_parse_rust_cli_names_a_group_it_cannot_find():
    source = "struct CliArgs {\n    #[command(flatten)]\n    workers: MissingArgs,\n}\n"
    with pytest.raises(AssertionError, match="struct MissingArgs"):
        parse_rust_cli(source)


def test_parse_rust_cli_reads_a_named_default_from_the_text_it_was_given():
    """A ``default_value_t`` naming a constant reads the constant's literal from the
    source under the reader, not from the checkout."""
    source = """
pub const RETRIES: usize = 3;
struct CliArgs {
    #[arg(long, default_value_t = RETRIES)]
    retries: usize,
}
"""
    assert parse_rust_cli(source).flags == {"--retries": 3}


def test_parse_rust_cli_reads_a_named_default_from_a_constant_source():
    """A constant another module declares is read from the sources handed in, and
    reported by name when none declares it."""
    source = (
        "struct CliArgs {\n    #[arg(long, default_value_t = RETRIES)]\n    retries: usize,\n}\n"
    )
    types = "pub const RETRIES: usize = 3;\n"
    assert parse_rust_cli(source, constant_sources=(types,)).flags == {"--retries": 3}
    assert parse_rust_cli(source).flags == {"--retries": UnreadDefault("RETRIES")}


def test_parse_rust_cli_reports_an_expression_default_as_unread():
    source = """
struct CliArgs {
    #[arg(long, default_value_t = Policy::RoundRobin)]
    policy: Policy,
}
"""
    assert parse_rust_cli(source).flags == {"--policy": UnreadDefault("Policy::RoundRobin")}


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
    if not GATEWAY_SRC.is_dir():
        pytest.skip("the Rust CLI's source is not in this checkout")
    text = cli_source()
    cli = parse_rust_cli(text, constant_sources=tuple(p.read_text() for p in CONST_SOURCES))
    assert cli.attributes == arg_attributes(text), (
        f"{CLI_DIR}: {arg_attributes(text)} #[arg] attributes, {cli.attributes} reached from"
        " struct CliArgs; a group it flattens is declared in a shape parse_rust_cli does not"
        " follow (or the rest belong to a struct outside the launch CLI: teach the reader)"
    )
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
    unread = []
    for flag, cli_default in rust.flags.items():
        if cli_default is None or flag not in launcher:
            continue
        launcher_default = launcher[flag].default
        if launcher_default is None or isinstance(launcher_default, list):
            continue  # resolved later (a tri-state, the dataclass, the Rust core)
        if isinstance(cli_default, UnreadDefault):
            unread.append(f"{flag} = {cli_default.raw}")  # a CLI default the reader skipped
            continue
        if isinstance(cli_default, bool) or isinstance(launcher_default, bool):
            same = cli_default is launcher_default
        elif isinstance(cli_default, (int, float)) and isinstance(launcher_default, (int, float)):
            same = abs(float(cli_default) - float(launcher_default)) < 1e-9
        else:
            same = str(cli_default) == str(launcher_default)
        if not same:
            differs[flag] = (cli_default, launcher_default)
    assert not unread, (
        f"CLI defaults that literal_default cannot read, so the test would skip them: {unread}"
    )
    unexpected = {flag: pair for flag, pair in differs.items() if flag not in DEFAULT_DIFFERS}
    assert not unexpected, f"defaults that differ from the Rust CLI's (CLI, launcher): {unexpected}"
    stale = [flag for flag in DEFAULT_DIFFERS if flag not in differs]
    assert not stale, f"defaults that no longer differ; drop them: {stale}"


# Constructor parameters whose literal default differs from the CLI flag of the
# same name, with the reason.
CONSTRUCTOR_DEFAULT_DIFFERS: dict[str, str] = {}


def signature_literal(raw: str) -> object:
    """A default of the ``#[pyo3(signature)]`` block as a Python value, or None when
    it is not a literal; an enum variant reads as the CLI's spelling of it. A
    ``Some(...)`` wrapper, an integer type suffix, a product or a power of integers
    and a ``vec![...]`` read as the value they stand for."""
    if raw in ("true", "false"):
        return raw == "true"
    quoted = re.fullmatch(r'String::from\("([^"]*)"\)', raw)
    if quoted:
        return quoted.group(1)
    wrapped = re.fullmatch(r"Some\((.*)\)", raw)
    if wrapped:
        return signature_literal(wrapped.group(1).strip())
    vector = re.fullmatch(r"vec!\[(.*)\]", raw)
    if vector:  # a list default reads as the CLI's comma-joined spelling of it
        elements = [signature_literal(e.strip()) for e in vector.group(1).split(",") if e.strip()]
        return None if any(e is None for e in elements) else ",".join(str(e) for e in elements)
    power = re.fullmatch(r"(\d+)(?:[ui](?:8|16|32|64|128|size))?\.pow\((\d+)\)", raw)
    if power:
        return int(power.group(1)) ** int(power.group(2))
    variant = re.fullmatch(r"[A-Za-z]+::([A-Z][A-Za-z0-9]*)", raw)
    if variant:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", variant.group(1)).lower()
    number = raw.replace("_", "")
    number = re.sub(r"(?<=\d)(u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|isize|f32|f64)$", "", number)
    factors = [factor.strip() for factor in number.split("*")]
    if len(factors) > 1 and all(re.fullmatch(r"\d+", factor) for factor in factors):
        product = 1
        for factor in factors:
            product *= int(factor)
        return product
    try:
        return int(number)
    except ValueError:
        pass
    try:
        return float(number)
    except ValueError:
        return None


def constructor_defaults() -> dict[str, tuple[str, object]]:
    """The parameters of the Rust ``Router`` constructor with their defaults as the
    Rust text and the literal it reads as (None when it is not a literal), read
    from the ``#[pyo3(signature)]`` block of ``lib.rs``."""
    text = (ROOT / "bindings" / "python" / "src" / "lib.rs").read_text()
    start = text.index("#[pyo3(signature = (", text.index("impl Router {"))
    block = text[start : text.index("))]", start)]
    defaults: dict[str, tuple[str, object]] = {}
    for line in block.splitlines()[1:]:
        entry = line.split("//", 1)[0].strip().rstrip(",")
        if not entry or entry == "*":
            continue
        name, _, raw = entry.partition("=")
        raw = raw.strip()
        defaults[name.strip()] = (raw, signature_literal(raw) if raw else None)
    return defaults


def test_the_constructor_defaults_are_the_cli_defaults(rust):
    """A direct ``Router(...)`` call and the binary start from the same values: the
    constructor's literal defaults equal those of the CLI flags of the same names."""
    defaults = constructor_defaults()
    assert len(defaults) > 100, "the signature block was not read whole"
    differs = {}
    unreadable = []
    for name, (raw, default) in defaults.items():
        spelling = f"--{name.replace('_', '-')}"
        cli_default = rust.flags.get(rust.aliases.get(spelling, spelling))
        if cli_default is None or raw in ("", "None"):
            continue
        if isinstance(cli_default, UnreadDefault):
            unreadable.append(f"{name}: the CLI default {cli_default.raw}")  # the CLI side skipped
            continue
        if default is None:
            unreadable.append(f"{name} = {raw}")  # a constructor literal the reader skipped
            continue
        if isinstance(default, bool) or isinstance(cli_default, bool):
            same = default is cli_default
        elif isinstance(default, (int, float)) and isinstance(cli_default, (int, float)):
            same = abs(float(default) - float(cli_default)) < 1e-9
        else:
            same = str(default) == str(cli_default)
        if not same:
            differs[name] = (cli_default, default)
    assert not unreadable, (
        f"constructor defaults with a literal that neither reader can read on one side, "
        f"so the test would skip them: {unreadable}"
    )
    unexpected = {n: pair for n, pair in differs.items() if n not in CONSTRUCTOR_DEFAULT_DIFFERS}
    assert not unexpected, (
        f"constructor defaults that differ from the CLI's (CLI, constructor): {unexpected}"
    )
    stale = [name for name in CONSTRUCTOR_DEFAULT_DIFFERS if name not in differs]
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


def test_the_launcher_backend_choices_are_the_cli_backends(launcher):
    """``--backend`` takes the same names on both sides, in the CLI's order, so a
    provider route the binary runs can be chosen from Python and the chart."""
    enum = CLI_MODULE.read_text().split("enum Backend {", 1)[1].split("}", 1)[0]
    assert launcher["--backend"].choices == re.findall(r'#\[value\(name = "([a-z]+)"\)\]', enum)


def test_the_launcher_encode_policy_choices_are_the_cli_choices(launcher):
    """``--encode-policy`` takes the same three names on both sides; the binding
    refuses the other policies from ``start()`` with those names in its message."""
    attribute = re.search(r"#\[arg\(([^\n]*)\)\]\n\s*encode_policy: Option<String>,", cli_source())
    assert attribute, "the CLI's --encode-policy attribute moved"
    cli_choices = re.findall(r'"([a-z_]+)"', attribute.group(1).split("value_parser", 1)[1])
    assert cli_choices == ["random", "round_robin", "consistent_hashing"]
    assert list(launcher["--encode-policy"].choices) == cli_choices


def test_the_launcher_policy_controller_choices_are_the_cli_choices(launcher):
    attribute = re.search(
        r"value_parser\s*=\s*\[([^\]]+)\][^\]]*\)\]\s*policy_controller: String,",
        cli_source(),
    )
    assert attribute, "the CLI's --policy-controller choices are missing"
    assert list(launcher["--policy-controller"].choices) == re.findall(
        r'"([a-z]+)"', attribute.group(1)
    )


def test_the_launchers_leave_policy_unset_and_describe_the_same_default(rust, launcher):
    option = launcher["--policy"]
    assert rust.flags["--policy"] is option.default is None
    assert constructor_defaults()["policy"] == ("None", None)

    declaration = re.search(
        r"(?P<help>(?:^[ \t]*/// [^\n]+\n)+)"
        r"\s*#\[arg\((?P<attribute>[^\n]+)\)\]\n\s*policy: Option<String>,",
        CLI_MODULE.read_text(),
        re.MULTILINE,
    )
    assert declaration, "the CLI's --policy declaration moved"
    choices = re.search(r"value_parser\s*=\s*\[([^\]]+)\]", declaration["attribute"])
    assert choices, "the CLI's --policy choices are missing"
    assert sorted(option.choices) == sorted(re.findall(r'"([a-z_]+)"', choices[1]))
    cli_help = " ".join(line.split("/// ", 1)[1] for line in declaration["help"].splitlines())
    default_help = (
        "Defaults to cache_aware."
        " A policy document replaces the routing flags, and --policy is refused beside one."
    )
    assert cli_help.endswith(default_help)
    assert option.help.endswith(default_help)


def test_the_launchers_list_the_same_options_a_document_replaces():
    block = re.search(
        r"const DOCUMENT_REPLACES: \[&str; (\d+)\] = \[(.*?)\];", CLI_MODULE.read_text(), re.S
    )
    assert block, "the CLI's list of the flags a document replaces moved"
    cli = re.findall(r'"([a-z_0-9]+)"', block[2])
    assert len(cli) == int(block[1])
    # The launcher spells the interval with its unit and folds the disable
    # flag into worker_overload_protection.
    cli.remove("disable_worker_overload_protection")
    cli[cli.index("eviction_interval")] = "eviction_interval_secs"
    assert list(DOCUMENT_REPLACES) == cli


def test_the_native_start_lists_the_same_options_a_document_replaces():
    # The native start refuses by its own copy of the list: a third home.
    block = re.search(r"let moved: Vec<&str> = moved!\[(.*?)\]", NATIVE_MODULE.read_text(), re.S)
    assert block, "the native start's list of the settings a document replaces moved"
    assert re.findall(r"[a-z_0-9]+", block[1]) == list(DOCUMENT_REPLACES)


@pytest.mark.parametrize("use_router_prefix", [False, True])
@pytest.mark.parametrize("settle", [None, 0, 5, 30, 120])
def test_shutdown_settle_option(use_router_prefix, settle):
    """Both launcher spellings preserve an omitted window and explicit zero or seconds."""
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=use_router_prefix)
    prefix = "router-" if use_router_prefix else ""
    flags = [] if settle is None else [f"--{prefix}shutdown-settle-secs", str(settle)]
    args = RouterArgs.from_cli_args(parser.parse_args(flags), use_router_prefix=use_router_prefix)
    assert args.shutdown_settle_secs == settle
    assert args.shutdown_grace_period_secs == 180


def test_the_eviction_interval_default_is_one_value_on_every_surface(rust, launcher):
    """The CLI, the launcher, the Python Router and the chart evict the cache tree
    on one cadence. The launcher's 60 s predated the CLI's 120 s and was never
    followed; the three other surfaces agreed on 120."""
    lib_rs = ROOT / "bindings" / "python" / "src" / "lib.rs"
    chart_values = ROOT / "deploy" / "helm" / "smg" / "values.yaml"
    signature = re.search(r"\beviction_interval_secs = (\d+)", lib_rs.read_text())
    chart = re.search(r"^\s*evictionIntervalSecs:\s*(\d+)\s*$", chart_values.read_text(), re.M)
    assert signature and chart, "the Python Router's signature or the chart's value moved"
    surfaces = {
        "cli": rust.flags["--eviction-interval"],
        "launcher": launcher["--eviction-interval"].default,
        "python_router": int(signature.group(1)),
        "chart": int(chart.group(1)),
    }
    assert set(surfaces.values()) == {120}, surfaces


def test_a_selector_entry_is_refused_with_the_cli_text(rust):
    """A discovery selector entry without ``key=value`` is refused on both sides
    with the one reason the CLI prints, so one slip reads the same everywhere."""
    reason = "expected 'key=value'"
    assert reason in cli_source(), "the CLI's selector refusal changed: update the launcher's"
    refusal = f"Invalid value for field 'decode_selector': tier - {reason}"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        RouterArgs._parse_selector(["app=engine tier"], "decode_selector")
