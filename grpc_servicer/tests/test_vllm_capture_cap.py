"""The CUDA-graph capture cap a gRPC launch derives from the admission step
(``smg_grpc_servicer.vllm.capture_cap``): the derivation table, the engine's
grid, the opt-out, and the plugin's parse step that puts the value where the
launcher builds the engine from."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.machinery
import io
import json
import logging
import sys
import types

import pytest
from smg_grpc_servicer.vllm import capture_cap, launcher_switch, plugin, rust
from smg_grpc_servicer.vllm.capture_cap import (
    CAPTURE_CAP_ENV,
    DEFAULT_CEILING,
    apply_capture_cap,
    capture_sizes,
    ceiling_from_setting,
    derive_capture_cap,
    grid_floor,
)

# ---------------------------------------------------------------------------
# The engine's grid and the ceiling knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "floor"),
    [
        (0, 0),
        (1, 1),
        (3, 2),
        (7, 4),
        (8, 8),
        (100, 96),
        (248, 248),
        (255, 248),
        (256, 256),
        (300, 288),
        (512, 512),
        (1000, 992),
        (1024, 1024),
    ],
)
def test_the_grid_is_the_engines_capture_list(size, floor):
    """1, 2, 4, multiples of 8 up to 248, multiples of 16 from 256: the
    engine's own list rule, so a value on it is kept by the engine as it is."""
    assert grid_floor(size) == floor
    if floor:
        assert floor in capture_sizes(max(floor, 1))


def test_the_capture_list_sizes_say_what_a_cap_costs():
    assert capture_sizes(64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert len(capture_sizes(64)) == 11
    assert len(capture_sizes(512)) == 51
    assert len(capture_sizes(1024)) == 83
    assert capture_sizes(512)[-3:] == [480, 496, 512]


@pytest.mark.parametrize(
    ("setting", "ceiling"),
    [
        (None, DEFAULT_CEILING),
        ("", DEFAULT_CEILING),
        ("auto", DEFAULT_CEILING),
        (" Auto ", DEFAULT_CEILING),
        ("engine", None),
        ("ENGINE", None),
        ("1024", 1024),
        ("1000", 992),
        ("256", 256),
        ("1", 1),
    ],
)
def test_the_knob_takes_auto_engine_or_a_ceiling_on_the_grid(setting, ceiling):
    assert ceiling_from_setting(setting) == ceiling


@pytest.mark.parametrize("setting", ["0", "-5", "lots", "512 tokens", "1.5"])
def test_the_knob_refuses_anything_else_by_name(setting):
    with pytest.raises(ValueError) as error:
        ceiling_from_setting(setting)
    assert CAPTURE_CAP_ENV in str(error.value)
    assert repr(setting) in str(error.value)


# ---------------------------------------------------------------------------
# The derivation table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_num_seqs", "step", "setting", "cap"),
    [
        # The drill's setting: the engine's 64 becomes 512, the ceiling.
        (32, 16384, None, 512),
        # A step size under the ceiling is the cap itself: the engine captures
        # max_num_batched_tokens when it fits under the cap, grid or not.
        (32, 300, None, 300),
        (32, 100, None, 100),
        # A step the engine's default covers already: nothing to raise.
        (32, 64, None, None),
        (32, 48, None, None),
        # No step size: unbounded (the engine's own defaults are 2048 or more).
        (32, None, None, 512),
        # A window of 256 or more: the engine's default is the ceiling already.
        (256, 16384, None, None),
        (1024, 16384, None, None),
        (255, 16384, None, 512),
        # The opt-out, and the ceiling as a number.
        (32, 16384, "engine", None),
        (32, 16384, "auto", 512),
        (32, 16384, "1024", 1024),
        (32, 16384, "1000", 992),
        (32, 16384, "256", 256),
        (32, 16384, "48", None),
        (256, 16384, "1024", 1024),
        (32, 700, "1024", 700),
    ],
)
def test_the_cap_is_the_admission_step_under_the_ceiling_where_that_raises_the_default(
    max_num_seqs, step, setting, cap
):
    decision = derive_capture_cap(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=step, setting=setting
    )
    assert decision.cap == cap
    assert decision.derived is (cap is not None)
    if cap is not None:
        assert decision.source == "derived"
        assert str(cap) in decision.detail
        assert f"--max-num-seqs {max_num_seqs}" in decision.detail


def test_a_derived_cap_is_the_step_size_or_on_the_grid():
    """What the engine keeps as given: its grid, or its own step size."""
    for max_num_seqs in (1, 4, 32, 100, 255):
        for step in (None, 5, 37, 100, 300, 512, 777, 16384):
            for setting in (None, "1000", "4096"):
                decision = derive_capture_cap(
                    max_num_seqs=max_num_seqs, max_num_batched_tokens=step, setting=setting
                )
                if decision.cap is None:
                    continue
                assert decision.cap > min(2 * max_num_seqs, 512)
                assert decision.cap == step or grid_floor(decision.cap) == decision.cap


def test_no_window_leaves_the_engines_default():
    decision = derive_capture_cap(max_num_seqs=None, max_num_batched_tokens=16384)
    assert decision.cap is None
    assert decision.source == "window"


@pytest.mark.parametrize(
    ("max_num_seqs", "step", "setting", "cap", "mentions"),
    [
        # A window of 255: the engine's list ends 510 at 496, so a ceiling of
        # 500 (496 on the grid) raises nothing and 512 does.
        (255, 16384, "500", None, "default 496 (twice --max-num-seqs 255, on the capture grid)"),
        (255, 16384, "512", 512, "default was 496 (twice --max-num-seqs 255, on the capture grid)"),
        # The step under twice the window: the engine's own cap is the step.
        (
            255,
            500,
            None,
            None,
            "default 500 (twice --max-num-seqs 255, at most the admission step)",
        ),
        (
            255,
            500,
            "1024",
            None,
            "default 500 (twice --max-num-seqs 255, at most the admission step)",
        ),
        # A window of 33: 66 ends at 64 on the grid, and at the step under it.
        (33, 16384, None, 512, "default was 64 (twice --max-num-seqs 33, on the capture grid)"),
        (33, 65, None, None, "default 65 (twice --max-num-seqs 33, at most the admission step)"),
    ],
)
def test_the_engines_default_is_read_as_its_list_ends_it(
    max_num_seqs, step, setting, cap, mentions
):
    """The engine truncates a default that is not its list's last entry (to
    the grid), and appends the step size where that fits under the default
    (the step is the cap then): the derivation compares against that value."""
    decision = derive_capture_cap(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=step, setting=setting
    )
    assert decision.cap == cap
    assert mentions in decision.detail


PLATFORM_PAIR = "512 or 1024 by platform (--max-num-seqs unset"


@pytest.mark.parametrize(
    ("step", "setting", "cap", "mentions"),
    [
        (16384, "2048", 2048, f"default was {PLATFORM_PAIR}"),
        (16384, "1024", 1024, f"default was {PLATFORM_PAIR}"),
        (None, "2048", 2048, f"default was {PLATFORM_PAIR}"),
        # A ceiling under 1024 could lower the engine's own 1024 on the platform
        # whose default ceiling that is: kept out, and the line says so.
        (16384, "1000", None, "could lower it on the platform whose default ceiling is 1024"),
        (16384, "768", None, "could lower it on the platform whose default ceiling is 1024"),
        # The step under the default ceiling: the engine's own cap is the step.
        (300, "2048", None, "default 300 (--max-num-seqs unset"),
        (300, "2048", None, "at most the admission step) covers the admission step"),
        # The step between the two platform defaults: the 1024 one is the step.
        (700, "2048", 700, "512 or 700 by platform (--max-num-seqs unset"),
        (700, "2048", 700, "reach the default ceiling, at most the admission step)"),
        # A ceiling at or under the engine's default ceiling raises nothing.
        (16384, "512", None, f"default {PLATFORM_PAIR}"),
        (16384, "auto", None, f"default {PLATFORM_PAIR}"),
        (16384, None, None, f"default {PLATFORM_PAIR}"),
    ],
)
def test_a_recipes_ceiling_applies_when_the_window_is_left_to_the_engine(
    step, setting, cap, mentions
):
    """A recipe that leaves --max-num-seqs to the engine still gets the ceiling
    it set where that raises the engine's default on every platform (512, or
    1024 on one): a ceiling of 1024 or more applies; `auto` keeps the engine's
    default as before."""
    decision = derive_capture_cap(max_num_seqs=None, max_num_batched_tokens=step, setting=setting)
    assert decision.cap == cap
    assert mentions in decision.detail
    if cap is None:
        assert decision.source == "window"
    else:
        assert decision.source == "derived"
        assert str(cap) in decision.detail


def test_a_step_between_the_two_platform_defaults_is_named_as_the_step():
    """A window of 300 with a step of 550: 512, or 550 on the 1024 platform
    (the engine ends its list at the step there); a ceiling of 540 is kept and
    the line names the step, not the grid."""
    decision = derive_capture_cap(max_num_seqs=300, max_num_batched_tokens=550, setting="540")
    assert decision.cap is None
    assert "512 or 550 by platform (twice --max-num-seqs 300, at most the admission step" in (
        decision.detail
    )
    assert "on the capture grid" not in decision.detail


@pytest.mark.parametrize(
    ("max_num_seqs", "setting", "cap", "mentions"),
    [
        # A window of 512: the engine's default is 512, or 1024 on one platform.
        (512, "768", None, "512 or 1024 by platform (twice --max-num-seqs 512"),
        (512, "768", None, "could lower it on the platform whose default ceiling is 1024"),
        (512, "1024", 1024, "default was 512 or 1024 by platform"),
        (512, "2048", 2048, "default was 512 or 1024 by platform"),
        (512, None, None, "covers the admission step"),
        # A window of 300: 512, or 600 on the 1024 platform, which the engine's
        # list ends at 592; 768 raises both, and so does 600 (592 on the grid).
        (300, "768", 768, "default was 512 or 592 by platform (twice --max-num-seqs 300"),
        (300, "600", 592, "default was 512 or 592 by platform (twice --max-num-seqs 300, on the"),
        (300, "576", None, "could lower it on the platform whose default ceiling is 1024"),
        # A window of 511: 1022 ends at 1008 on the grid; 1008 raises 512 and equals it.
        (
            511,
            "1008",
            1008,
            "default was 512 or 1008 by platform (twice --max-num-seqs 511, on the",
        ),
        (
            511,
            "1000",
            None,
            "a cap of 992 (--max-num-batched-tokens 16384, ceiling 992) could lower",
        ),
        # A window of 256 or less: one default on every platform, as before.
        (256, "768", 768, "default was 512 (twice --max-num-seqs 256)"),
        (32, "768", 768, "default was 64 (twice --max-num-seqs 32)"),
    ],
)
def test_a_ceiling_never_lowers_the_engines_default_on_the_platform_with_the_higher_ceiling(
    max_num_seqs, setting, cap, mentions
):
    decision = derive_capture_cap(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=16384, setting=setting
    )
    assert decision.cap == cap
    assert mentions in decision.detail


@pytest.mark.parametrize(
    ("kwargs", "source", "mentions"),
    [
        ({"explicit_cap": 128}, "operator", "max_cudagraph_capture_size=128"),
        ({"explicit_sizes": [8, 1, 4, 2]}, "operator", "cudagraph_capture_sizes=[1, 2, 4, 8]"),
        ({"enforce_eager": True}, "eager", "--enforce-eager"),
        ({"cudagraph_mode": "NONE"}, "eager", "cudagraph_mode NONE"),
        ({"cudagraph_mode": 0}, "eager", "cudagraph_mode NONE"),
        ({"cudagraph_mode": types.SimpleNamespace(name="NONE")}, "eager", "cudagraph_mode NONE"),
        ({"speculative": True}, "speculative", "speculative decoding"),
    ],
)
def test_an_operator_value_and_the_settings_that_make_the_default_better_are_kept(
    kwargs, source, mentions
):
    decision = derive_capture_cap(max_num_seqs=32, max_num_batched_tokens=16384, **kwargs)
    assert decision.cap is None
    assert decision.source == source
    assert mentions in decision.detail


@pytest.mark.parametrize(
    "mode", [None, "FULL_AND_PIECEWISE", 3, types.SimpleNamespace(name="FULL")]
)
def test_a_cudagraph_mode_with_graphs_derives(mode):
    decision = derive_capture_cap(
        max_num_seqs=32, max_num_batched_tokens=16384, cudagraph_mode=mode
    )
    assert decision.cap == 512


def test_the_opt_out_wins_over_everything_else():
    decision = derive_capture_cap(
        max_num_seqs=32, max_num_batched_tokens=16384, explicit_cap=128, setting="engine"
    )
    assert decision.cap is None
    assert decision.source == "engine"


# ---------------------------------------------------------------------------
# The namespace: where the value is read and where it is put
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompilationConfig:
    """The engine's compilation config, as far as the derivation reads it."""

    cudagraph_mode: object = None
    cudagraph_capture_sizes: list | None = None
    max_cudagraph_capture_size: int | None = None


def _namespace(**values) -> argparse.Namespace:
    defaults = {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 16384,
        "max_cudagraph_capture_size": None,
        "cudagraph_capture_sizes": None,
        "compilation_config": CompilationConfig(),
        "enforce_eager": False,
        "speculative_config": None,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_a_derived_cap_goes_to_the_engines_top_level_flag():
    """The namespace carries the engine's own `--max-cudagraph-capture-size`:
    the derived cap is what that flag would have set, and the compilation
    config object (the parser's default, shared by every parse) is untouched."""
    parsed = _namespace()
    decision = apply_capture_cap(parsed, environ={})
    assert decision.cap == 512
    assert parsed.max_cudagraph_capture_size == 512
    assert parsed.compilation_config == CompilationConfig()


def test_a_derived_cap_goes_to_a_copy_of_the_compilation_config_without_the_flag():
    default = CompilationConfig()
    parsed = argparse.Namespace(
        max_num_seqs=32, max_num_batched_tokens=16384, compilation_config=default
    )
    decision = apply_capture_cap(parsed, environ={})
    assert decision.cap == 512
    assert parsed.compilation_config.max_cudagraph_capture_size == 512
    assert parsed.compilation_config is not default
    assert default.max_cudagraph_capture_size is None


def test_a_derived_cap_goes_into_a_dict_compilation_config():
    parsed = argparse.Namespace(
        max_num_seqs=32, max_num_batched_tokens=16384, compilation_config={}
    )
    assert apply_capture_cap(parsed, environ={}).cap == 512
    assert parsed.compilation_config == {"max_cudagraph_capture_size": 512}


def test_a_namespace_without_any_capture_field_gets_the_top_level_one():
    parsed = argparse.Namespace(max_num_seqs=32, max_num_batched_tokens=16384)
    assert apply_capture_cap(parsed, environ={}).cap == 512
    assert parsed.max_cudagraph_capture_size == 512


@pytest.mark.parametrize(
    "values",
    [
        {"max_cudagraph_capture_size": 128},
        {"cudagraph_capture_sizes": [1, 2, 4]},
        {"compilation_config": CompilationConfig(max_cudagraph_capture_size=128)},
        {"compilation_config": CompilationConfig(cudagraph_capture_sizes=[8])},
        {"compilation_config": {"max_cudagraph_capture_size": 128}},
        {"compilation_config": '{"max_cudagraph_capture_size": 128}'},
        {"compilation_config": CompilationConfig(cudagraph_mode="NONE")},
        {"enforce_eager": True},
        {"speculative_config": {"method": "ngram"}},
        {"max_num_seqs": None},
    ],
)
def test_the_operators_settings_on_the_namespace_are_kept(values):
    parsed = _namespace(**values)
    before = dict(vars(parsed))
    decision = apply_capture_cap(parsed, environ={})
    assert decision.cap is None
    assert vars(parsed) == before


def test_the_opt_out_is_read_from_the_environment():
    parsed = _namespace()
    decision = apply_capture_cap(parsed, environ={CAPTURE_CAP_ENV: "engine"})
    assert decision.cap is None
    assert parsed.max_cudagraph_capture_size is None
    assert apply_capture_cap(_namespace(), environ={CAPTURE_CAP_ENV: "1024"}).cap == 1024


def test_a_compilation_config_string_the_engine_would_refuse_is_left_to_it():
    parsed = _namespace(compilation_config="{not json")
    assert apply_capture_cap(parsed, environ={}).cap == 512
    assert parsed.compilation_config == "{not json"


# ---------------------------------------------------------------------------
# The plugin's parse step
# ---------------------------------------------------------------------------

OLD = launcher_switch.LAUNCHERS[-1].name


def _main_module(spec_name: str | None) -> types.ModuleType:
    main = types.ModuleType("__main__")
    main.__spec__ = None if spec_name is None else importlib.machinery.ModuleSpec(spec_name, None)
    return main


@pytest.fixture
def parser_class(monkeypatch):
    """vLLM's parser class as the plugin wraps it, in a process without vLLM;
    the media flags' handoff (it writes to the environment) and the launcher
    modules are not this file's. One line per decision, per process: the
    record of logged decisions starts empty for every test."""

    class FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    for name in ("vllm", "vllm.utils", "vllm.utils.argparse_utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.utils.argparse_utils"].FlexibleArgumentParser = FlexibleArgumentParser
    monkeypatch.setattr(plugin, "add_mm_arguments_to_grpc_parser", lambda parser: [])
    monkeypatch.setattr(plugin, "handoff_mm_flags", lambda parser, parsed: None)
    monkeypatch.setitem(sys.modules, "__main__", _main_module("vllm.entrypoints.cli.main"))
    for name in launcher_switch.LAUNCHER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv(CAPTURE_CAP_ENV, raising=False)
    monkeypatch.setattr(capture_cap, "_logged", set())
    plugin.register()
    return FlexibleArgumentParser


def _serve_parser(parser_class, with_top_level_flag: bool = True) -> argparse.ArgumentParser:
    """`vllm serve`'s parser, as far as the derivation reads it: the engine's
    flags with their types and defaults (the compilation config's default is
    one object, built when the flag is defined)."""
    parser = parser_class(prog="vllm serve")
    parser.add_argument("model", nargs="?")
    parser.add_argument("--grpc", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--speculative-config", type=json.loads, default=None)
    parser.add_argument(
        "--compilation-config",
        type=lambda text: CompilationConfig(**json.loads(text)),
        default=CompilationConfig(),
    )
    if with_top_level_flag:
        parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
        parser.add_argument("--cudagraph-capture-sizes", type=int, nargs="+", default=None)
    return parser


DRILL = ["m", "--max-num-seqs", "32", "--max-num-batched-tokens", "16384"]


def test_a_grpc_serve_parse_carries_the_derived_cap(parser_class, caplog):
    parser = _serve_parser(parser_class)
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = parser.parse_args(["--grpc", *DRILL])
    assert parsed.max_cudagraph_capture_size == 512
    assert parsed.compilation_config == CompilationConfig()
    lines = [
        record.getMessage() for record in caplog.records if "capture cap" in record.getMessage()
    ]
    assert len(lines) == 1
    assert "512, derived from the admission step (--max-num-batched-tokens 16384" in lines[0]
    assert "default was 64 (twice --max-num-seqs 32)" in lines[0]


def test_a_headless_serve_parse_carries_the_same_cap(parser_class):
    parsed = _serve_parser(parser_class).parse_args(["--headless", *DRILL])
    assert parsed.max_cudagraph_capture_size == 512


def test_an_http_serve_parse_keeps_the_engines_default(parser_class, caplog):
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(DRILL)
    assert parsed.max_cudagraph_capture_size is None
    assert not [record for record in caplog.records if "capture cap" in record.getMessage()]


def test_the_stock_launchers_own_parse_carries_the_derived_cap(parser_class, monkeypatch):
    """The stock gRPC launcher's parser defines no --grpc: the process's
    __main__ is the launcher module, and gRPC is all it serves."""
    monkeypatch.setitem(sys.modules, "__main__", _main_module(OLD))
    parser = parser_class(prog="grpc_server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
    parsed = parser.parse_args(DRILL[1:])
    assert parsed.max_cudagraph_capture_size == 512


@pytest.mark.parametrize(
    ("argv", "top_level", "config"),
    [
        (["--max-cudagraph-capture-size", "128"], 128, CompilationConfig()),
        (["--cudagraph-capture-sizes", "1", "8"], None, CompilationConfig()),
        (
            ["--compilation-config", '{"max_cudagraph_capture_size": 128}'],
            None,
            CompilationConfig(max_cudagraph_capture_size=128),
        ),
        (
            ["--compilation-config", '{"cudagraph_capture_sizes": [8, 16]}'],
            None,
            CompilationConfig(cudagraph_capture_sizes=[8, 16]),
        ),
        (["--enforce-eager"], None, CompilationConfig()),
        (["--speculative-config", '{"method": "ngram"}'], None, CompilationConfig()),
    ],
)
def test_an_operators_capture_setting_on_the_command_line_is_kept(
    parser_class, caplog, argv, top_level, config
):
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL, *argv])
    assert parsed.max_cudagraph_capture_size == top_level
    assert parsed.compilation_config == config
    lines = [
        record.getMessage() for record in caplog.records if "capture cap" in record.getMessage()
    ]
    assert len(lines) == 1
    assert "derived" not in lines[0]


def test_the_opt_out_variable_keeps_the_engines_default(parser_class, monkeypatch, caplog):
    monkeypatch.setenv(CAPTURE_CAP_ENV, "engine")
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
    assert parsed.max_cudagraph_capture_size is None
    lines = [
        record.getMessage() for record in caplog.records if "capture cap" in record.getMessage()
    ]
    assert lines == [
        "CUDA-graph capture cap: the engine's own default (SMG_VLLM_CUDAGRAPH_CAPTURE_CAP=engine: "
        "twice --max-num-seqs, at most 512)"
    ]


def test_a_ceiling_in_the_variable_bounds_the_derived_cap(parser_class, monkeypatch):
    monkeypatch.setenv(CAPTURE_CAP_ENV, "1000")
    parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
    assert parsed.max_cudagraph_capture_size == 992


def test_a_recipe_without_the_window_flag_gets_the_ceiling_it_set(
    parser_class, monkeypatch, caplog
):
    """The recipe's ceiling is applied with --max-num-seqs left to the engine,
    and the line says what the engine's default was bounded by."""
    monkeypatch.setenv(CAPTURE_CAP_ENV, "2048")
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(
            ["--grpc", "m", "--max-num-batched-tokens", "16384"]
        )
    assert parsed.max_cudagraph_capture_size == 2048
    lines = [
        record.getMessage() for record in caplog.records if "capture cap" in record.getMessage()
    ]
    assert len(lines) == 1
    assert (
        "2048, derived from the admission step (--max-num-batched-tokens 16384, ceiling 2048)"
        in lines[0]
    )
    assert "default was 512 or 1024 by platform (--max-num-seqs unset" in lines[0]


def test_a_bad_variable_value_is_a_usage_error_that_names_it(parser_class, monkeypatch, capsys):
    monkeypatch.setenv(CAPTURE_CAP_ENV, "lots")
    with pytest.raises(SystemExit) as exit_info:
        _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
    assert exit_info.value.code == 2
    assert f"{CAPTURE_CAP_ENV}='lots'" in capsys.readouterr().err


def test_a_second_parse_derives_again_from_a_fresh_namespace_and_logs_nothing_new(
    parser_class, caplog
):
    parser = _serve_parser(parser_class)
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        first = parser.parse_args(["--grpc", *DRILL])
        second = parser.parse_args(["--grpc", *DRILL])
        third = parser.parse_args(["--grpc", *DRILL[:-1], "300"])
    assert (first.max_cudagraph_capture_size, second.max_cudagraph_capture_size) == (512, 512)
    assert third.max_cudagraph_capture_size == 300
    assert parser.get_default("compilation_config") == CompilationConfig()
    lines = [
        record.getMessage() for record in caplog.records if "capture cap" in record.getMessage()
    ]
    assert len(lines) == 2


def test_without_the_top_level_flag_the_parsers_default_config_is_never_written(parser_class):
    parser = _serve_parser(parser_class, with_top_level_flag=False)
    parsed = parser.parse_args(["--grpc", *DRILL])
    assert parsed.compilation_config.max_cudagraph_capture_size == 512
    assert parser.get_default("compilation_config").max_cudagraph_capture_size is None
    again = parser.parse_args(["--grpc", *DRILL])
    assert again.compilation_config.max_cudagraph_capture_size == 512


# ---------------------------------------------------------------------------
# The line's way out: vLLM's handlers, configured before the parse (0.28
# through 0.30: at import) or after it (0.31: the launcher, after parse_args)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _no_handlers_anywhere():
    """A launcher process before vLLM configures logging: no handler on vLLM's
    logger, on the package's, or on the root logger they propagate to (where
    the test runner's own capture handler sits)."""
    loggers = (
        logging.getLogger(),
        logging.getLogger("vllm"),
        logging.getLogger("smg_grpc_servicer"),
    )
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in loggers]
    try:
        for lg in loggers:
            lg.handlers = []
            lg.setLevel(logging.WARNING if lg is loggers[0] else logging.NOTSET)
            lg.propagate = True
        yield
    finally:
        for lg, (handlers, level, propagate) in zip(loggers, saved, strict=True):
            lg.handlers = handlers
            lg.setLevel(level)
            lg.propagate = propagate


def _vllm_configures_logging(stream: io.StringIO) -> None:
    """What vLLM's configure_logging_from_args(args) does on 0.31, and its import
    did before: one handler on its own logger, a format with file and line."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(filename)s:%(lineno)d] %(message)s"))
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = [handler]
    vllm_logger.setLevel(logging.INFO)


def _the_launcher_runs_serve_grpc(parsed: argparse.Namespace, monkeypatch) -> None:
    """The launcher's ``uvloop.run(serve_grpc(args))`` after its parse, with the
    switch bound over ``serve_grpc`` and the Rust path a stand-in that returns."""

    async def serve_grpc(args):
        raise AssertionError("the Rust selection never reaches the launcher's own coroutine")

    async def serve_rust(args):
        return 0

    monkeypatch.setattr(rust, "serve_rust", serve_rust)
    parsed.servicer_impl = "rust"
    with pytest.raises(SystemExit):
        asyncio.run(launcher_switch.with_servicer_switch(serve_grpc)(parsed))


def test_the_decision_logged_at_a_parse_before_vllm_configures_logging_comes_out_once_it_has(
    parser_class, monkeypatch
):
    """vLLM 0.31: the launcher parses, then configures logging, then runs
    serve_grpc. The line logged at the parse has no handler then; it comes out
    at the switch's first act, through vLLM's handlers, under its own file."""
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _no_handlers_anywhere():
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
        assert parsed.max_cudagraph_capture_size == 512
        assert stream.getvalue() == ""
        _vllm_configures_logging(stream)
        _the_launcher_runs_serve_grpc(parsed, monkeypatch)
        lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("INFO [capture_cap.py:")
    assert (
        "CUDA-graph capture cap: 512, derived from the admission step (--max-num-batched-tokens 16384"
        in lines[0]
    )


def test_the_decision_prints_at_the_parse_where_vllm_configured_logging_before_it_and_never_twice(
    parser_class, monkeypatch
):
    """vLLM 0.28 through 0.30 configured logging at import: the line prints at
    the parse, as before, and the switch's first act replays nothing."""
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _no_handlers_anywhere():
        _vllm_configures_logging(stream)
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
        at_the_parse = stream.getvalue().splitlines()
        _the_launcher_runs_serve_grpc(parsed, monkeypatch)
        lines = stream.getvalue().splitlines()
    assert len(at_the_parse) == 1 and "512, derived" in at_the_parse[0]
    assert lines == at_the_parse


def test_the_python_path_ends_a_hold_nothing_released_before_the_launchers_coroutine(
    parser_class, monkeypatch, capsys
):
    """A launcher that configured no logging at all, Python servicer: the
    switch ends the hold before the launcher's own coroutine runs, so the
    package's warnings and errors reach stderr from there as without the hold
    (the held INFO line has nowhere to go, as before)."""
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    monkeypatch.setattr(rust, "require_python_media_mode", lambda args: None)
    monkeypatch.setattr(rust, "default_kv_events_config", lambda args: None)
    pkg_logger = logging.getLogger("smg_grpc_servicer")

    async def serve_grpc(args):
        logging.getLogger("smg_grpc_servicer.vllm.servicer").error("from the launcher's coroutine")
        return "python ran"

    with _no_handlers_anywhere():
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *DRILL])
        assert len(pkg_logger.handlers) == 1  # the hold, with the capture-cap line
        parsed.servicer_impl = "python"
        result = asyncio.run(launcher_switch.with_servicer_switch(serve_grpc)(parsed))
        assert result == "python ran"
        assert pkg_logger.handlers == []
    assert capsys.readouterr().err == "from the launcher's coroutine\n"
