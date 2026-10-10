"""Unit tests for the Rust request path of the vLLM servicer
(``smg_grpc_servicer.vllm.rust``).

Everything vLLM-shaped is lazy in the module, so these run without vLLM:
``serve_rust`` gets stub ``vllm`` modules, a fake Rust server class and a fake
engine launch; the lifecycle loop gets fake handles.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.vllm import rust


def _install(monkeypatch, name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        parent_module = sys.modules.get(parent) or _install(monkeypatch, parent)
        # Undone with the test: the parent is the real package where an engine is installed.
        monkeypatch.setattr(parent_module, child, module, raising=False)
    return module


def test_package_exposes_the_flag_and_the_rust_entry_points():
    import smg_grpc_servicer.vllm as pkg

    assert pkg.SERVICER_IMPL_ENV == "SMG_VLLM_SERVICER_IMPL"
    assert pkg.resolve_servicer_impl is rust.resolve_servicer_impl
    assert pkg.serve_rust is rust.serve_rust


def test_resolve_servicer_impl_prefers_the_launcher_flag_then_the_env():
    assert rust.resolve_servicer_impl(environ={}) == "python"
    assert rust.resolve_servicer_impl(environ={"SMG_VLLM_SERVICER_IMPL": " Rust "}) == "rust"
    args = argparse.Namespace(servicer_impl="rust")
    assert rust.resolve_servicer_impl(args, environ={}) == "rust"
    # An unset launcher flag falls through to the env.
    args = argparse.Namespace(servicer_impl=None)
    assert rust.resolve_servicer_impl(args, environ={"SMG_VLLM_SERVICER_IMPL": "rust"}) == "rust"
    with pytest.raises(ValueError, match="SMG_VLLM_SERVICER_IMPL"):
        rust.resolve_servicer_impl(environ={"SMG_VLLM_SERVICER_IMPL": "go"})


def test_a_launcher_flag_decision_is_written_back_for_the_python_guard():
    # `--servicer-impl python` with `rust` still exported: the hook picks
    # Python, and the servicer's guard must see that decision, not the env.
    environ = {"SMG_VLLM_SERVICER_IMPL": "rust"}
    args = argparse.Namespace(servicer_impl="python")
    assert rust.resolve_servicer_impl(args, environ=environ) == "python"
    assert environ["SMG_VLLM_SERVICER_IMPL"] == "python"
    rust.require_python_impl(environ=environ)
    # Without a flag in hand nothing is written.
    environ = {}
    assert rust.resolve_servicer_impl(environ=environ) == "python"
    assert environ == {}


def test_python_servicer_refuses_to_start_when_the_flag_asks_for_rust():
    rust.require_python_impl(environ={})
    rust.require_python_impl(environ={"SMG_VLLM_SERVICER_IMPL": "python"})
    with pytest.raises(RuntimeError, match="never ran"):
        rust.require_python_impl(environ={"SMG_VLLM_SERVICER_IMPL": "rust"})


def test_upstream_hook_is_detected_in_the_launcher_source(tmp_path):
    launcher = tmp_path / "vllm" / "entrypoints" / "launchers"
    launcher.mkdir(parents=True)
    launcher.joinpath("grpc_server.py").write_text("async def serve_grpc(args):\n    pass\n")
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is False
    launcher.joinpath("grpc_server.py").write_text(
        "from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust\n"
    )
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True
    # A moved launcher is found anywhere under entrypoints/.
    launcher.joinpath("grpc_server.py").unlink()
    (tmp_path / "vllm" / "entrypoints" / "serve_grpc.py").write_text("resolve_servicer_impl\n")
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "nowhere")) is False


def test_require_upstream_hook_names_the_fix(monkeypatch):
    monkeypatch.setattr(rust, "upstream_hook_installed", lambda: False)
    with pytest.raises(RuntimeError, match="consults this package's flag"):
        rust.require_upstream_hook()
    monkeypatch.setattr(rust, "upstream_hook_installed", lambda: True)
    rust.require_upstream_hook()


def test_launcher_switch_binds_serve_grpc_during_the_launchers_import(monkeypatch):
    """vLLM's launcher imports this package before it defines serve_grpc; that
    import hooks the module so the first attribute access after the import
    (the CLI's `from ... import serve_grpc`) binds the switch, in the module
    dict too, where the launcher's own main() reads it."""
    from smg_grpc_servicer.vllm import launcher_switch

    name = "vllm.entrypoints.launchers.grpc_server"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    assert launcher_switch.install_launcher_switch() == [name]
    calls = []

    async def serve_grpc(args):
        calls.append(args)
        return "python ran"

    module.__dict__["serve_grpc"] = serve_grpc  # the launcher's own `async def`, after our import
    switched = module.serve_grpc
    assert switched is not serve_grpc and module.__dict__["serve_grpc"] is switched
    assert switched.__name__ == "serve_grpc"
    python_args = argparse.Namespace(servicer_impl="python")
    assert asyncio.run(switched(python_args)) == "python ran" and calls == [python_args]
    rust_calls = []

    async def fake_serve_rust(args):
        rust_calls.append(args)
        return 7

    monkeypatch.setattr(rust, "serve_rust", fake_serve_rust)
    with pytest.raises(SystemExit) as exit_:
        asyncio.run(switched(argparse.Namespace(servicer_impl="rust")))
    assert exit_.value.code == 7 and len(rust_calls) == 1
    # Idempotent, and stable across accesses and re-installs.
    assert launcher_switch.install_launcher_switch() == [name]
    assert module.serve_grpc is switched
    # A launcher that imported this package after defining serve_grpc is rebound at once.
    late = types.ModuleType("vllm.entrypoints.grpc_server")
    late.__dict__["serve_grpc"] = serve_grpc
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.grpc_server", late)
    assert set(launcher_switch.install_launcher_switch()) == {name, "vllm.entrypoints.grpc_server"}
    assert late.__dict__["serve_grpc"] is not serve_grpc


def test_launcher_import_of_this_package_satisfies_the_hook_check(tmp_path):
    launcher = tmp_path / "vllm" / "entrypoints" / "launchers"
    launcher.mkdir(parents=True)
    path = launcher / "grpc_server.py"
    # Upstream's shape: the servicer imports inside a module-level try.
    path.write_text(
        "try:\n"
        "    import grpc\n"
        "    from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer\n"
        "    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer\n"
        "except ImportError as e:\n"
        "    raise ImportError('gRPC mode requires smg-grpc-servicer') from e\n"
        "async def serve_grpc(args):\n"
        "    pass\n"
    )
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True
    # Imported only inside serve_grpc, the switch would bind too late.
    path.write_text(
        "async def serve_grpc(args):\n"
        "    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer\n"
    )
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is False


def test_plugin_adds_the_servicer_impl_flag_to_grpc_parsers(monkeypatch):
    """vLLM loads this package's general plugin while it builds the serve
    parser; the flag joins any parser that defines --grpc, as it parses."""
    pytest.importorskip("vllm")
    from smg_grpc_servicer.vllm import plugin
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Recorded, so teardown undoes the value the parse step exports.
    monkeypatch.setenv(rust.SERVICER_IMPL_ENV, "python")
    plugin.register()
    plugin.register()  # idempotent: one wrap of the parse step
    parser = FlexibleArgumentParser(prog="vllm serve")
    parser.add_argument("--grpc", action="store_true")
    args = parser.parse_args(["--grpc", "--servicer-impl", "rust"])
    assert (args.grpc, args.servicer_impl) == (True, "rust")
    assert parser.parse_args(["--grpc"]).servicer_impl is None
    assert "--servicer-impl" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--grpc", "--servicer-impl", "go"])
    # The parsed flag decides ahead of the environment.
    assert rust.resolve_servicer_impl(args, environ={"SMG_VLLM_SERVICER_IMPL": "python"}) == "rust"
    # Parsers without --grpc (vllm bench, offline LLM args) are untouched.
    plain = FlexibleArgumentParser(prog="vllm bench")
    plain.add_argument("--model")
    assert not hasattr(plain.parse_args(["--model", "m"]), "servicer_impl")


def test_plugin_defines_the_mm_flags_on_grpc_launcher_parsers(monkeypatch):
    """The stock gRPC launcher's parser defines no --mm-* flag, so the media
    mode was selectable there through the environment only: the plugin adds
    the flags when the process's __main__ is that launcher module, and to a
    `vllm serve` parser (--grpc) that lacks them; other commands' parsers and
    a __main__ without a spec (a console script) are left alone."""
    from smg_grpc_servicer.vllm import plugin

    def launcher_parser():
        parser = argparse.ArgumentParser(prog="grpc_server")
        parser.add_argument("--host")
        parser.add_argument("--port", type=int, default=50051)
        parser.add_argument("--model")
        return parser

    def main_module(name):
        spec = None if name is None else types.SimpleNamespace(name=name)
        return types.SimpleNamespace(__spec__=spec)

    monkeypatch.setitem(sys.modules, "__main__", main_module("vllm.entrypoints.cli.main"))
    assert plugin.add_mm_arguments_to_grpc_parser(launcher_parser()) == []
    serve = launcher_parser()
    serve.add_argument("--grpc", action="store_true")
    assert plugin.add_mm_arguments_to_grpc_parser(serve)[0] == "--mm-processor"
    for name in ("vllm.entrypoints.grpc_server", "vllm.entrypoints.launchers.grpc_server"):
        monkeypatch.setitem(sys.modules, "__main__", main_module(name))
        parser = launcher_parser()
        assert plugin.add_mm_arguments_to_grpc_parser(parser)[0] == "--mm-processor"
        args = parser.parse_args(["--model", "m", "--mm-processor", "smg", "--mm-max-items", "2"])
        assert (args.mm_processor, args.mm_max_items) == ("smg", 2)
        assert plugin.add_mm_arguments_to_grpc_parser(parser) == []  # once
    monkeypatch.setitem(sys.modules, "__main__", main_module(None))
    assert plugin.add_mm_arguments_to_grpc_parser(launcher_parser()) == []


def test_plugin_hands_the_mm_flags_over_on_every_grpc_parse(monkeypatch):
    """The handoff to a servicer built without the namespace follows the
    parser serving gRPC, not this parse having defined the flags: a parser
    that has them already (a `vllm serve` with its own, a second parse of the
    same parser) hands its values over too, and the latest parse wins; a
    parser of another command hands nothing over."""
    from smg_grpc_servicer.vllm import mm_processor, plugin

    monkeypatch.setattr(mm_processor, "_launcher_settings", None)
    monkeypatch.setattr(mm_processor, "_environ_before_carry", None)
    monkeypatch.setenv("SMG_VLLM_MM_PROCESSOR", "off")
    monkeypatch.delenv("SMG_VLLM_MM_MAX_ITEMS", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "__main__",
        types.SimpleNamespace(__spec__=types.SimpleNamespace(name="vllm.entrypoints.grpc_server")),
    )
    parser = argparse.ArgumentParser(prog="grpc_server")
    parser.add_argument("--port", type=int, default=50051)
    assert plugin.add_mm_arguments_to_grpc_parser(parser)  # the flags are on it now
    assert plugin.add_mm_arguments_to_grpc_parser(parser) == []  # a second parse adds none...
    kept = plugin.handoff_mm_flags(
        parser, parser.parse_args(["--mm-processor", "inprocess", "--mm-max-items", "4"])
    )
    assert kept is mm_processor.launcher_settings()  # ...and still hands the values over
    assert kept.processor == "inprocess" and kept.resolve(env={}).source == "flag"
    assert os.environ["SMG_VLLM_MM_PROCESSOR"] == "inprocess"
    assert os.environ["SMG_VLLM_MM_MAX_ITEMS"] == "4"
    # The latest parse wins in the slot and in the environment: a flag the
    # first parse set and the second dropped is gone from both.
    again = plugin.handoff_mm_flags(parser, parser.parse_args(["--mm-processor", "redis"]))
    assert mm_processor.launcher_settings() is again and again.processor == "redis"
    assert "SMG_VLLM_MM_MAX_ITEMS" not in os.environ
    assert (
        again.resolve(env=os.environ).max_items
        == mm_processor.MmSettings().resolve(env={}).max_items
    )
    # A parser of another command: nothing kept, the slot untouched.
    other = argparse.ArgumentParser(prog="bench")
    other.add_argument("--model")
    assert plugin.handoff_mm_flags(other, other.parse_args(["--model", "m"])) is None
    assert mm_processor.launcher_settings() is again


def test_plugin_parses_the_mm_flags_and_exports_them_for_the_python_servicer(monkeypatch):
    """With vLLM installed: the stock launcher's parse step grows the flags and
    carries their values into the environment, where a servicer built without
    the namespace (upstream's launcher) reads them."""
    pytest.importorskip("vllm")
    from smg_grpc_servicer.vllm import mm_processor, plugin
    from smg_grpc_servicer.vllm.mm_processor import MmSettings
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # Recorded, so teardown undoes what the parse step exports...
    monkeypatch.setenv("SMG_VLLM_MM_PROCESSOR", "off")
    monkeypatch.setenv("SMG_VLLM_MM_MAX_ITEMS", "1")
    # ...and the settings it keeps for a servicer built without the namespace.
    monkeypatch.setattr(mm_processor, "_launcher_settings", None)
    monkeypatch.setitem(
        sys.modules,
        "__main__",
        types.SimpleNamespace(__spec__=types.SimpleNamespace(name="vllm.entrypoints.grpc_server")),
    )
    plugin.register()
    parser = FlexibleArgumentParser(prog="grpc_server")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args(["--mm-processor", "smg", "--mm-max-items", "4"])
    assert (args.mm_processor, args.mm_max_items) == ("smg", 4)
    assert os.environ["SMG_VLLM_MM_PROCESSOR"] == "smg"
    assert os.environ["SMG_VLLM_MM_MAX_ITEMS"] == "4"
    assert mm_processor.launcher_settings() == MmSettings.from_args(args)
    assert "--mm-processor" in parser.format_help()
    resolved = MmSettings.from_args(args).resolve(env={})
    assert (resolved.processor, resolved.max_items, resolved.source) == ("smg", 4, "flag")
    with pytest.raises(SystemExit):
        parser.parse_args(["--mm-processor", "sidecar"])


def test_plugin_entry_point_is_declared():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'smg-servicer = "smg_grpc_servicer.vllm.plugin:register"' in pyproject
    assert '[project.entry-points."vllm.general_plugins"]' in pyproject


def test_launcher_imported_after_this_package_is_switched_as_it_loads(tmp_path):
    """Something imported the servicer first (or the launcher imports it
    lazily): the meta-path finder binds the switch when the launcher loads,
    in a fresh interpreter so the real vLLM, if installed, stays out of it."""
    pkg = tmp_path / "vllm" / "entrypoints" / "launchers"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "vllm", tmp_path / "vllm" / "entrypoints", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "grpc_server.py").write_text("async def serve_grpc(args):\n    return 'python ran'\n")
    script = (
        "import sys\n"
        "from smg_grpc_servicer.vllm.launcher_switch import install_launcher_switch\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "assert install_launcher_switch() == []  # no launcher yet\n"
        "import vllm.entrypoints.launchers.grpc_server as launcher\n"
        "print(getattr(launcher.__dict__['serve_grpc'], '__smg_servicer_switch__', False))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "True"


def test_plugin_exports_the_parsed_flag_for_the_python_guard(monkeypatch):
    pytest.importorskip("vllm")
    from smg_grpc_servicer.vllm import plugin
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    plugin.register()
    monkeypatch.setenv(
        rust.SERVICER_IMPL_ENV, "python"
    )  # recorded, so teardown undoes the plugin's write
    parser = FlexibleArgumentParser(prog="vllm serve")
    parser.add_argument("--grpc", action="store_true")
    parser.parse_args(["--grpc", "--servicer-impl", "rust"])
    # A switch that never bound would start the Python servicer, whose guard
    # reads only the environment: the flag is there.
    assert os.environ[rust.SERVICER_IMPL_ENV] == "rust"
    with pytest.raises(RuntimeError, match="never ran"):
        rust.require_python_impl()
    assert set(plugin.SERVICER_IMPL_CHOICES) == set(rust.IMPLS)


def test_launcher_scan_ignores_type_checking_and_accepts_the_package_names(tmp_path):
    launcher = tmp_path / "vllm" / "entrypoints" / "launchers"
    launcher.mkdir(parents=True)
    path = launcher / "grpc_server.py"
    path.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer\n"
        "async def serve_grpc(args):\n"
        "    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer\n"
    )
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is False
    path.write_text("from smg_grpc_servicer.vllm import VllmEngineServicer, VllmHealthServicer\n")
    assert rust.upstream_hook_installed(vllm_root=str(tmp_path / "vllm")) is True


# ---------------------------------------------------------------------------
# Model / server info
# ---------------------------------------------------------------------------


def _config(**overrides):
    model_config = SimpleNamespace(
        model="org/m",
        served_model_name=["served-a", "served-b"],
        tokenizer="org/m-tok",
        runner_type="generate",
        is_multimodal_model=False,
        max_model_len=4096,
        get_vocab_size=lambda: 1024,
        hf_config=SimpleNamespace(
            model_type="qwen3", eos_token_id=[151645, 151643], pad_token_id=None, bos_token_id=1
        ),
        architectures=["Qwen3ForCausalLM"],
        try_get_generation_config=lambda: {"eos_token_id": [151643, 7], "temperature": 0.6},
        get_diff_sampling_param=lambda: {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": None,
            "max_new_tokens": 10,
        },
    )
    for key, value in overrides.items():
        setattr(model_config, key, value)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )


def test_smg_media_options_follow_the_engine_config(tmp_path, monkeypatch):
    from smg_grpc_servicer.vllm import mm_processor
    from smg_grpc_servicer.vllm.mm_processor import MmSettings

    # The loader's default frame count is vLLM's to tell; none installed here.
    monkeypatch.setattr(mm_processor, "vllm_default_video_frames", lambda: None)
    settings = MmSettings(processor="smg", max_inflight=3, max_items=2, max_item_bytes=10).resolve(
        env={}
    )
    config = _config(
        is_multimodal_model=True,
        dtype="torch.bfloat16",
        multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
    )
    assert rust.smg_media_options(config, settings, str(tmp_path)) == {
        "model_dir": str(tmp_path),
        "model_id": "org/m",
        "raw_pixels": True,
        "encoder_dtype": "bfloat16",
        "processor_kwargs_json": None,
        "max_inflight": 3,
        "max_items": 2,
        "engine_item_limits": None,
        "max_item_bytes": 10,
        "video_frame_budget": None,
        "video_loader_rule": None,
        "source": "flag",
    }
    # The engine's own per-prompt limits ride along for the pipeline to enforce.
    config.model_config.multimodal_config = SimpleNamespace(
        mm_device_do_normalize=True,
        get_limit_per_prompt=lambda modality: {"image": 8, "video": 2, "audio": 1}[modality],
    )
    assert rust.smg_media_options(config, settings, str(tmp_path))["engine_item_limits"] == {
        "image": 8,
        "video": 2,
    }
    # The engine's video frame budget rides along: its media kwargs when set
    # (a non-positive count meaning every frame), else the loader's default,
    # under the same SMG_VLLM_MM_MAX_VIDEO_FRAMES cap as the other processors.
    monkeypatch.setattr(mm_processor, "vllm_default_video_frames", lambda: 32)
    monkeypatch.setenv(mm_processor.ENV_MAX_VIDEO_FRAMES, "0")  # no cap
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 32
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"num_frames": 16}}
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 16
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"num_frames": -1}}
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 0
    # The cap bounds a set count, the loader's default and an unbounded count alike.
    monkeypatch.setenv(mm_processor.ENV_MAX_VIDEO_FRAMES, "8")
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 8
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"num_frames": 16}}
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 8
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"num_frames": 4}}
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 4
    config.model_config.multimodal_config.media_io_kwargs = {}
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 8
    # A loader of its own (video_backend in the kwargs or the environment) or a
    # frame rate above zero, which thins by duration: rules the pipeline cannot
    # follow go along by name, and the pipeline refuses them only for a family
    # that samples the way the loader does. The budget rides along regardless.

    def rule():
        return rust.smg_media_options(config, settings, str(tmp_path))["video_loader_rule"]

    config.model_config.multimodal_config.media_io_kwargs = {"video": {"video_backend": "x"}}
    assert rule() == "--media-io-kwargs video.video_backend=x"
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 8
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"video_backend": "opencv"}}
    assert rule() is None, "the default loader by name is no rule"
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"fps": 1}}
    assert rule() == "--media-io-kwargs video.fps=1"
    for no_cap in (-1, 0, "0"):
        config.model_config.multimodal_config.media_io_kwargs = {"video": {"fps": no_cap}}
        assert rule() is None, f"fps={no_cap!r} thins nothing"
    config.model_config.multimodal_config.media_io_kwargs = {}
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")
    assert rule() == "VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic"
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"fps": 2}}
    assert rule() == "VLLM_VIDEO_LOADER_BACKEND=opencv_dynamic; --media-io-kwargs video.fps=2"
    # vLLM takes the kwarg over the environment: a kwarg naming opencv silences
    # a fleet-wide VLLM_VIDEO_LOADER_BACKEND, and a foreign kwarg is the rule
    # whatever the environment says.
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"video_backend": "opencv"}}
    assert rule() is None
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    config.model_config.multimodal_config.media_io_kwargs = {"video": {"video_backend": "x"}}
    assert rule() == "--media-io-kwargs video.video_backend=x"
    config.model_config.multimodal_config.media_io_kwargs = {}
    assert rule() is None
    assert rust.smg_media_options(config, settings, str(tmp_path))["video_frame_budget"] == 8
    monkeypatch.delenv("VLLM_VIDEO_LOADER_BACKEND", raising=False)
    monkeypatch.delenv(mm_processor.ENV_MAX_VIDEO_FRAMES, raising=False)
    # A local model directory with its config is the pipeline's config source.
    (tmp_path / "config.json").write_text("{}")
    config.model_config.model = str(tmp_path)
    options = rust.smg_media_options(config, settings, None)
    assert options["model_dir"] == str(tmp_path) and options["model_id"] == str(tmp_path)
    # An engine that normalizes on the CPU takes normalized pixels in its dtype.
    config.model_config.multimodal_config = SimpleNamespace(mm_device_do_normalize=False)
    assert rust.smg_media_options(config, settings, None)["raw_pixels"] is False
    # The engine's processor kwargs ride along, less where its own processor runs.
    config.model_config.mm_processor_kwargs = {"max_pixels": 1000, "device": "cuda"}
    assert (
        rust.smg_media_options(config, settings, None)["processor_kwargs_json"]
        == '{"max_pixels": 1000}'
    )
    # A text model takes no media: the mode is ignored, as the Python processors ignore it.
    assert rust.smg_media_options(_config(), settings, None) is None


def test_model_info_advertises_device_side_normalization():
    assert rust.model_info_from_config(_config())["mm_device_do_normalize"] is False
    config = _config(
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
    )
    assert rust.model_info_from_config(config)["mm_device_do_normalize"] is True


def test_model_info_advertises_the_engines_item_limits():
    # A text model has none: the Router keeps its own caps.
    assert rust.model_info_from_config(_config())["mm_item_limits"] == ""
    config = _config(
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(
            mm_device_do_normalize=False,
            get_limit_per_prompt=lambda modality: {"image": 8, "video": 2, "audio": 1}[modality],
        ),
    )
    assert rust.model_info_from_config(config)["mm_item_limits"] == "image=8,video=2"


def test_model_info_carries_the_running_window():
    # The Rust servicer advertises the launcher's `--max-num-seqs` as the
    # Python servicer does; a config without one leaves the handshake's.
    assert rust.model_info_from_config(_config())["max_num_seqs"] == 0
    config = _config()
    config.scheduler_config = SimpleNamespace(max_num_seqs=64)
    assert rust.model_info_from_config(config)["max_num_seqs"] == 64


def test_model_info_mirrors_the_python_servicer(monkeypatch):
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", " nixl ")
    info = rust.model_info_from_config(_config())
    assert info["model_path"] == "org/m"
    assert info["served_model_name"] == "served-a"
    assert info["tokenizer_path"] == "org/m-tok"
    assert info["is_generation"] is True
    assert info["max_context_length"] == 4096
    assert info["vocab_size"] == 1024
    # A text model: vLLM's own multimodal check says no.
    assert info["supports_vision"] is False
    assert info["model_type"] == "qwen3"
    assert info["architectures"] == ["Qwen3ForCausalLM"]
    # Model config ids first (the primary EOS), generation-config extras after, deduped.
    assert info["eos_token_ids"] == [151645, 151643, 7]
    assert info["pad_token_id"] == 0
    assert info["bos_token_id"] == 1
    # None values and non-sampling keys are dropped, as the Python servicer does.
    assert json.loads(info["default_sampling_params_json"]) == {"temperature": 0.6, "top_p": 0.95}
    assert info["data_parallel_size"] == 2
    assert info["pairing_protocol"] == "nixl"
    # No KV connector configured: the PD identity is empty, as in Python.
    assert (info["kv_connector"], info["kv_role"], info["kv_engine_id"]) == ("", "", "")
    assert info["block_size"] == 0 and info["model_dtype"] == ""
    # No KV-event publisher configured, backend unset, and the host's shm id.
    assert info["kv_events_endpoint"] == "" and info["kv_events_topic"] == ""
    assert info["structured_outputs_backend"] == "auto"
    assert isinstance(info["shm_namespace_id"], str)
    # No pooler config on a generation model.
    assert info["pooler_use_activation"] is None and info["pooler_dimensions"] is None


def test_model_info_carries_the_pooler_config():
    config = _config(
        runner_type="pooling",
        pooler_config=SimpleNamespace(use_activation=False, dimensions=256, pooling_type="MEAN"),
    )
    info = rust.model_info_from_config(config)
    assert info["is_generation"] is False
    assert info["pooler_use_activation"] is False
    assert info["pooler_dimensions"] == 256
    config = _config(pooler_config=SimpleNamespace(use_activation=None, dimensions=None))
    info = rust.model_info_from_config(config)
    assert info["pooler_use_activation"] is None and info["pooler_dimensions"] is None


def test_model_info_reports_the_pd_identity_and_pairing_facts():
    config = _config(dtype="torch.bfloat16")
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="NixlConnector", kv_role="kv_producer", engine_id="eng-a"
    )
    config.cache_config = SimpleNamespace(cache_dtype="auto", block_size=16)
    config.attention_config = SimpleNamespace(backend=SimpleNamespace(name="FLASH_ATTN"))
    info = rust.model_info_from_config(config)
    assert info["kv_connector"] == "NixlConnector"
    assert info["kv_role"] == "kv_producer"
    assert info["kv_engine_id"] == "eng-a"
    # The same labels the Python servicer's GetServerInfo derives.
    assert info["kv_cache_dtype"] == "auto"
    assert info["block_size"] == 16
    assert info["attention_backend"] == "FLASH_ATTN"
    assert info["model_dtype"] == "torch.bfloat16"


def test_model_info_reports_vision_from_vllms_own_check():
    config = _config(supports_multimodal_inputs=True, is_multimodal_model=True)
    assert rust.model_info_from_config(config)["supports_vision"] is True
    # `--language-model-only` drops the encoder: vLLM says no, so do we.
    config = _config(supports_multimodal_inputs=False, is_multimodal_model=True)
    assert rust.model_info_from_config(config)["supports_vision"] is False


@pytest.mark.parametrize("modalities", [("image", "video"), ("image", "video", "audio")])
@pytest.mark.parametrize(
    "limits,expected", [({"video": 0}, True), ({"image": 0, "video": 0}, False)]
)
def test_model_info_respects_effective_multimodal_limits(monkeypatch, limits, expected, modalities):
    _install(
        monkeypatch,
        "vllm.multimodal",
        MULTIMODAL_REGISTRY=SimpleNamespace(
            get_processing_info=lambda mc: SimpleNamespace(
                supported_mm_limits=dict.fromkeys(modalities)
            )
        ),
    )
    config = _config(
        supports_multimodal_inputs=True,
        is_multimodal_model=True,
        multimodal_config=SimpleNamespace(
            enable_mm_embeds=True,
            language_model_only=False,
            get_limit_per_prompt=lambda modality: limits.get(modality, 999),
        ),
    )
    assert rust.model_info_from_config(config)["supports_vision"] is expected


def test_model_info_reports_kv_events_and_the_structured_backend():
    config = _config()
    config.structured_outputs_config = SimpleNamespace(backend="xgrammar")
    config.kv_events_config = SimpleNamespace(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:5557",
        replay_endpoint="tcp://*:5558",
        topic="kv",
    )
    info = rust.model_info_from_config(config)
    assert info["structured_outputs_backend"] == "xgrammar"
    assert info["kv_events_endpoint"] == "tcp://*:5557"
    assert info["kv_events_replay_endpoint"] == "tcp://*:5558"
    assert info["kv_events_topic"] == "kv"
    # A non-ZMQ publisher (or disabled events) leaves the relay off.
    config.kv_events_config.publisher = "null"
    assert rust.model_info_from_config(config)["kv_events_endpoint"] == ""
    config.kv_events_config.publisher = "zmq"
    config.kv_events_config.enable_kv_cache_events = False
    assert rust.model_info_from_config(config)["kv_events_endpoint"] == ""


def test_model_info_with_scalar_eos_and_no_generation_config():
    config = _config(
        hf_config=SimpleNamespace(
            model_type="llama", eos_token_id=2, pad_token_id=0, bos_token_id=1
        ),
        try_get_generation_config=lambda: None,
        get_diff_sampling_param=lambda: {},
        served_model_name=None,
    )
    info = rust.model_info_from_config(config)
    assert info["eos_token_ids"] == [2]
    assert info["served_model_name"] == "org/m"
    assert info["default_sampling_params_json"] == ""


# ---------------------------------------------------------------------------
# Headless engine launch
# ---------------------------------------------------------------------------


def _stub_frontend_args(monkeypatch):
    @dataclasses.dataclass
    class FrontendArgs:
        host: str | None = None
        port: int = 8000
        reasoning_parser_plugin: str = ""
        middleware: list = dataclasses.field(default_factory=list)

    _install(monkeypatch, "vllm")
    _install(monkeypatch, "vllm.entrypoints")
    _install(monkeypatch, "vllm.entrypoints.launchers")
    _install(monkeypatch, "vllm.entrypoints.launchers.cli_args", FrontendArgs=FrontendArgs)


def _stub_engine_exceptions(monkeypatch):
    """What `serve_rust`'s media bridge import needs from vLLM's exceptions."""
    _install(
        monkeypatch,
        "vllm.exceptions",
        VLLMNotFoundError=type("VLLMNotFoundError", (Exception,), {}),
        VLLMClientError=type("VLLMClientError", (Exception,), {}),
    )
    _install(monkeypatch, "vllm.v1")
    _install(monkeypatch, "vllm.v1.engine")
    _install(
        monkeypatch,
        "vllm.v1.engine.exceptions",
        EngineGenerateError=type("EngineGenerateError", (Exception,), {}),
    )


def test_headless_namespace_is_built_from_the_parsed_args_not_argv(monkeypatch):
    """The `python -m vllm.entrypoints.grpc_server` namespace: engine args
    stay as parsed, the frontend fields it lacks get defaults, and the
    data-parallel group is pinned to this host and the handshake port."""
    _stub_frontend_args(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "org/m", "--grpc", "--port", "50051"])
    args = argparse.Namespace(host="127.0.0.1", port=50051, model="org/m", max_model_len=4096)
    ns = rust.headless_namespace(args, handshake_port=24321, data_parallel_size=2)
    assert ns.model == "org/m" and ns.model_tag is None
    assert ns.max_model_len == 4096
    assert ns.headless is True and ns.grpc is False and ns.api_server_count == 0
    assert ns.data_parallel_size == 2 and ns.data_parallel_size_local == 2
    assert ns.data_parallel_address == "127.0.0.1"
    assert ns.data_parallel_rpc_port == 24321
    assert ns.data_parallel_start_rank is None
    assert ns.data_parallel_hybrid_lb is False and ns.data_parallel_external_lb is False
    # Frontend defaults filled for `run_headless`; parsed values untouched.
    assert ns.reasoning_parser_plugin == "" and ns.middleware == []
    assert ns.host == "127.0.0.1" and ns.port == 50051
    # The launcher's own namespace is not mutated.
    assert not hasattr(args, "headless")


def test_headless_namespace_from_the_vllm_serve_grpc_form(monkeypatch):
    """`vllm serve org/m --grpc`: the positional is already on `args.model`;
    `--grpc` is cleared so the child never re-enters the gRPC path, and every
    frontend field is already present."""
    _stub_frontend_args(monkeypatch)
    args = argparse.Namespace(
        model_tag="org/m",
        model="org/m",
        grpc=True,
        headless=False,
        api_server_count=None,
        host=None,
        port=8000,
        reasoning_parser_plugin="",
        middleware=[],
        tensor_parallel_size=2,
    )
    ns = rust.headless_namespace(args, handshake_port=24321, data_parallel_size=1)
    assert ns.grpc is False and ns.headless is True and ns.model_tag is None
    assert ns.model == "org/m" and ns.tensor_parallel_size == 2
    assert ns.api_server_count == 0
    assert ns.data_parallel_rpc_port == 24321 and ns.data_parallel_size_local == 1


def test_engine_process_is_popen_shaped():
    class Proc:
        pid = 4242
        exitcode = None
        events: list[str] = []

        def terminate(self):
            self.events.append("terminate")

        def kill(self):
            self.events.append("kill")

        def join(self, timeout=None):
            self.events.append(f"join:{timeout}")

    proc = Proc()
    engine = rust.EngineProcess(proc)
    assert engine.pid == 4242 and engine.poll() is None
    with pytest.raises(subprocess.TimeoutExpired):
        engine.wait(timeout=0.1)
    proc.exitcode = 0
    assert engine.wait() == 0 and engine.poll() == 0
    engine.terminate()
    engine.kill()
    assert proc.events == ["join:0.1", "join:None", "terminate", "kill"]


def test_launch_headless_engine_spawns_a_fresh_interpreter(monkeypatch):
    recorded: dict = {}

    class Process:
        def __init__(self, target, args, name):
            recorded["target"] = target
            recorded["args"] = args
            recorded["name"] = name
            self.pid = 7
            self.exitcode = None

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(
        rust.multiprocessing,
        "get_context",
        lambda method: recorded.setdefault("method", method) and SimpleNamespace(Process=Process),
    )
    ns = argparse.Namespace(model="org/m")
    engine = rust.launch_headless_engine(ns)
    assert recorded["method"] == "spawn"
    assert recorded["target"] is rust._run_headless
    assert recorded["args"] == (ns,)
    assert recorded["started"] is True
    assert engine.pid == 7


class _FakeCore:
    """A `multiprocessing`-shaped engine-core process the fake manager started."""

    def __init__(self, pid: int):
        self.pid = pid
        self.name = f"EngineCore_DP{pid}"
        self.exitcode: int | None = None
        self.events: list[str] = []

    def terminate(self) -> None:
        self.events.append("terminate")
        self.exitcode = -15

    def kill(self) -> None:
        self.events.append("kill")
        self.exitcode = -9

    def join(self, timeout=None) -> None:
        self.events.append(f"join:{timeout}")


class _FakeManager:
    """vLLM's `CoreEngineProcManager` as the launch sees it: the keyword
    arguments, the processes it started (in this process), one-shot shutdown."""

    instances: list[_FakeManager] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started_in = os.getpid()
        self.processes = [_FakeCore(1000 + i) for i in range(kwargs["local_engine_count"])]
        self.events: list[str] = []
        _FakeManager.instances.append(self)

    def shutdown(self, timeout=None) -> None:
        self.events.append(f"shutdown:{timeout}")
        for core in self.processes:
            if core.exitcode is None:
                core.terminate()


def _parallel(**overrides) -> SimpleNamespace:
    """The parallel config `create_engine_config(headless=True)` yields for the
    headless namespace: every engine local, pinned to the handshake port."""
    parallel = SimpleNamespace(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_rank=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_rpc_port=24321,
        data_parallel_backend="mp",
        node_rank_within_dp=0,
    )
    for key, value in overrides.items():
        setattr(parallel, key, value)
    return parallel


def _stub_engine_core_launch(monkeypatch, parallel: SimpleNamespace) -> dict:
    """Stub what `launch_engine_cores` imports from vLLM; returns the record
    of `create_engine_config` calls and the headless config it hands out."""
    recorded: dict = {"configs": []}
    headless_config = SimpleNamespace(parallel_config=parallel, shutdown_timeout=0)
    recorded["headless_config"] = headless_config

    class EngineArgs:
        disable_log_stats = False

        def __init__(self, args):
            self.args = args

        def create_engine_config(self, usage_context, headless=False):
            recorded["configs"].append((self.args, usage_context, headless))
            return headless_config if headless else _config()

    class AsyncEngineArgs:
        @staticmethod
        def from_cli_args(args):
            return EngineArgs(args)

    class Executor:
        @staticmethod
        def get_class(vllm_config):
            recorded["executor_config"] = vllm_config
            return Executor

    _stub_frontend_args(monkeypatch)
    _install(monkeypatch, "vllm.engine.arg_utils", AsyncEngineArgs=AsyncEngineArgs)
    _install(monkeypatch, "vllm.usage.usage_lib", UsageContext=SimpleNamespace(OPENAI_API_SERVER=1))
    _install(
        monkeypatch, "vllm.utils.network_utils", get_tcp_uri=lambda ip, port: f"tcp://{ip}:{port}"
    )
    _install(monkeypatch, "vllm.v1.engine.utils", CoreEngineProcManager=_FakeManager)
    _install(monkeypatch, "vllm.v1.executor", Executor=Executor)
    monkeypatch.setattr(_FakeManager, "instances", [])
    recorded["Executor"] = Executor
    return recorded


def test_launch_engine_cores_builds_the_manager_in_this_process_as_run_headless_does(monkeypatch):
    """smg-lab#43: the cores are started by `CoreEngineProcManager` from the
    servicer process, from the headless config of the same namespace, with
    the arguments vLLM's `run_headless` passes for the head node; no
    `multiprocessing` child runs the launcher."""
    recorded = _stub_engine_core_launch(monkeypatch, _parallel())

    def no_launcher_child(method):
        raise AssertionError(f"a {method} child was started for the engine launch")

    monkeypatch.setattr(rust.multiprocessing, "get_context", no_launcher_child)
    args = argparse.Namespace(model="org/m", host=None, port=50051, max_model_len=4096)
    ns = rust.headless_namespace(args, handshake_port=24321, data_parallel_size=2)

    engine = rust.launch_engine_cores(ns)

    assert isinstance(engine, rust_lifecycle.EngineProcessGroup)
    (manager,) = _FakeManager.instances
    assert manager.started_in == os.getpid()
    assert manager.kwargs == {
        "local_engine_count": 2,
        "start_index": 0,
        "local_start_index": 0,
        "vllm_config": recorded["headless_config"],
        "local_client": False,
        "handshake_address": "tcp://127.0.0.1:24321",
        "executor_class": recorded["Executor"],
        "log_stats": True,
    }
    assert recorded["executor_config"] is recorded["headless_config"]
    # The config is the headless one, from the launcher's namespace re-aimed
    # at the handshake port (not from argv).
    ((config_args, usage_context, headless),) = recorded["configs"]
    assert config_args is ns and usage_context == 1 and headless is True
    assert ns.headless is True and ns.data_parallel_rpc_port == 24321
    assert engine.pids == [1000, 1001] and engine.pid == 1000 and engine.poll() is None


@pytest.mark.parametrize(
    "overrides", [{"node_rank_within_dp": 1}, {"data_parallel_backend": "ray"}]
)
def test_launch_engine_cores_leaves_worker_ranks_and_ray_to_the_spawned_launcher(
    monkeypatch, overrides
):
    """The launcher alone serves a multi-node engine's worker ranks and the
    ray backend: those still go through the spawned `run_headless`, unchanged."""
    _stub_engine_core_launch(monkeypatch, _parallel(**overrides))
    launched: list[argparse.Namespace] = []
    fallback = FakeEngine()

    def fake_launch(ns):
        launched.append(ns)
        return fallback

    monkeypatch.setattr(rust, "launch_headless_engine", fake_launch)
    ns = rust.headless_namespace(
        argparse.Namespace(model="org/m", host=None, port=50051),
        handshake_port=24321,
        data_parallel_size=2,
    )
    assert rust.launch_engine_cores(ns) is fallback
    assert launched == [ns] and _FakeManager.instances == []


def test_serve_rust_starts_the_engine_cores_from_this_process(monkeypatch, tmp_path):
    """smg-lab#43 end to end: `serve_rust` builds the servicer's config, binds
    the server, then starts the cores in this process against the server's
    handshake port; the supervisor gets the cores' handle."""
    recorded = _stub_engine_core_launch(monkeypatch, _parallel())
    _stub_engine_exceptions(monkeypatch)
    _install(monkeypatch, "smg")
    _install(
        monkeypatch, "smg.servicer", VllmGrpcServer=FakeServer, init_servicer_tracing=lambda: None
    )
    monkeypatch.setattr(rust.multiprocessing, "get_context", lambda method: pytest.fail(method))
    monkeypatch.setattr(rust, "resolve_tokenizer_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(rust, "free_port", lambda: 24321)
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    supervised: dict = {}

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        supervised["server"], supervised["engine"] = server, engine
        return 0

    monkeypatch.setattr(rust, "supervise", fake_supervise)
    args = argparse.Namespace(
        model_tag="org/m", model="org/m", grpc=True, host=None, port=50051, max_model_len=4096
    )
    assert asyncio.run(rust.serve_rust(args)) == 0

    engine = supervised["engine"]
    assert isinstance(engine, rust_lifecycle.EngineProcessGroup)
    (manager,) = _FakeManager.instances
    assert manager.started_in == os.getpid()
    assert manager.kwargs["handshake_address"] == supervised["server"].kwargs["handshake_address"]
    assert manager.kwargs["local_engine_count"] == supervised["server"].kwargs["engine_count"] == 2
    # Two configs in this process: the servicer's own, then the headless one.
    assert [(call[1], call[2]) for call in recorded["configs"]] == [(1, False), (1, True)]
    assert recorded["configs"][0][0] is args and recorded["configs"][1][0].headless is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.address = "127.0.0.1:50051"
        self.engine_ready = False
        self.last_error = None
        self.running = True
        self.events: list[str] = []
        self.alive_reports = 0

    def note_engine_alive(self) -> None:
        self.alive_reports += 1

    def set_serving(self, serving: bool) -> None:
        self.events.append(f"serving:{serving}")

    def stop(self, timeout: float) -> None:
        self.events.append(f"stop:{timeout}")


class FakeEngine:
    def __init__(self, *_args, **_kwargs):
        self.returncode = None
        self.pid = 99
        self.events: list[str] = []

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.events.append("kill")

    def kill_descendants(self) -> int:
        return 0  # the straggler sweep finds nothing behind a fake


def _run_supervise(server, engine, *, before, drain_secs=0.0):
    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(
            rust.supervise(
                server,
                engine,
                drain_secs=drain_secs,
                stop_timeout=1.0,
                stop_event=stop,
                poll_secs=0.01,
            )
        )
        await asyncio.sleep(0.05)
        before(stop)
        return await task

    return asyncio.run(run())


def test_supervise_exits_nonzero_when_the_engine_dies():
    server, engine = FakeServer(), FakeEngine()

    def engine_dies(_stop):
        engine.returncode = 3

    assert _run_supervise(server, engine, before=engine_dies) == 1
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == []


def test_supervise_exits_nonzero_on_a_server_error_and_terminates_the_engine():
    server, engine = FakeServer(), FakeEngine()

    def server_fails(_stop):
        server.last_error = "engine connection failed: handshake timed out"

    assert _run_supervise(server, engine, before=server_fails) == 1
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == ["terminate"]


def test_supervise_drains_then_stops_on_a_signal():
    server, engine = FakeServer(), FakeEngine()
    server.engine_ready = True
    assert _run_supervise(server, engine, before=lambda stop: stop.set(), drain_secs=0.01) == 0
    # Draining is announced before the stop, with the engine still up for it.
    assert server.events == ["serving:False", "stop:1.0"]
    assert engine.events == ["terminate"]


def test_supervise_reports_the_engine_alive_while_its_handshake_runs():
    """Every poll that finds the engine process alive before the handshake
    completed is a sign of life for the server's startup bound; a connected
    engine needs no more reports."""
    server, engine = FakeServer(), FakeEngine()
    assert _run_supervise(server, engine, before=lambda stop: stop.set()) == 0
    assert server.alive_reports >= 1

    connected, engine = FakeServer(), FakeEngine()
    connected.engine_ready = True
    assert _run_supervise(connected, engine, before=lambda stop: stop.set()) == 0
    assert connected.alive_reports == 0

    class ServerWithoutReports(FakeServer):
        note_engine_alive = None  # an older binding: the supervisor does without

    assert _run_supervise(ServerWithoutReports(), FakeEngine(), before=lambda stop: stop.set()) == 0


def test_supervise_exits_nonzero_when_an_engine_core_dies_and_shuts_the_others_down():
    """One core's exit code ends the engine: the loop exits 1, the manager
    shuts the surviving cores down, nothing is terminated twice."""
    manager = _FakeManager(local_engine_count=2)
    server, engine = FakeServer(), rust_lifecycle.EngineProcessGroup(manager)

    def core_dies(_stop):
        manager.processes[1].exitcode = 3

    assert _run_supervise(server, engine, before=core_dies) == 1
    assert server.events == ["serving:False", "stop:1.0"]
    assert manager.events == ["shutdown:None"]
    assert [core.events for core in manager.processes] == [["terminate"], []]
    # The exit code is the one that ended the engine, not the sibling's.
    assert engine.poll() == 3 and engine.wait() == 3


def test_supervise_terminates_the_engine_cores_through_the_manager_on_a_signal():
    manager = _FakeManager(local_engine_count=2)
    server = FakeServer()
    server.engine_ready = True
    engine = rust_lifecycle.EngineProcessGroup(manager, shutdown_timeout=7.0)
    assert _run_supervise(server, engine, before=lambda stop: stop.set(), drain_secs=0.01) == 0
    assert server.events == ["serving:False", "stop:1.0"]
    assert manager.events == ["shutdown:7.0"]
    # Terminated once each through the manager, waited for, never killed.
    assert [_signals(core) for core in manager.processes] == [["terminate"], ["terminate"]]


def test_engine_process_group_is_popen_shaped():
    manager = _FakeManager(local_engine_count=2)
    engine = rust_lifecycle.EngineProcessGroup(manager)
    assert engine.pid == 1000 and engine.pids == [1000, 1001]
    assert engine.poll() is None and manager.events == []
    with pytest.raises(subprocess.TimeoutExpired):
        engine.wait(timeout=0.1)
    # `kill` reaches every core directly; the manager's one-shot shutdown is
    # not relied on for it, and nothing is left for it to do.
    engine.kill()
    assert [_signals(core) for core in manager.processes] == [["kill"], ["kill"]]
    assert engine.wait() == -9 and engine.poll() == -9 and manager.events == []
    assert engine.kill_descendants() == 0


def _signals(core: _FakeCore) -> list[str]:
    return [event for event in core.events if not event.startswith("join")]


def test_configure_logging_gives_the_package_a_handler(monkeypatch):
    import logging

    root = logging.getLogger()
    pkg = logging.getLogger("smg_grpc_servicer")
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(pkg, "handlers", [])
    monkeypatch.setattr(logging.getLogger("vllm"), "handlers", [])
    rust.configure_logging()
    assert root.handlers or pkg.handlers


# vLLM's own default for its ZMQ publisher's endpoint. `default_kv_events_config`
# leaves it to vLLM: absent from the JSON form, the dataclass default in the typed one.
KV_EVENTS_ENDPOINT = "tcp://*:5557"
# What vLLM sees of the configuration the Rust path applies: the publisher on.
KV_EVENTS_ON = (True, "zmq", KV_EVENTS_ENDPOINT)


def _kv_events(config):
    """``(enable_kv_cache_events, publisher, endpoint)`` as vLLM ends up seeing
    them, from either form ``default_kv_events_config`` returns: the JSON dict
    the launcher's parser takes when ``vllm.config`` is not importable (the
    unit-test job without an engine), or vLLM's own ``KVEventsConfig`` when it
    is (the engine-gated job, where the engine args hold the typed form)."""
    if isinstance(config, dict):
        config = SimpleNamespace(**{"endpoint": KV_EVENTS_ENDPOINT, **config})
    return config.enable_kv_cache_events, config.publisher, config.endpoint


def test_kv_event_publishing_is_on_by_default_under_the_rust_servicer(monkeypatch):
    """A launcher without --kv-events-config left vLLM publishing nothing and
    the router's cache-aware routing blind (smg-lab #1): the Rust path turns
    the ZMQ publisher on itself unless told otherwise."""
    # Nothing given: the typed form when vllm.config is importable, else the
    # JSON form the parser takes; what vLLM sees is the same.
    args = argparse.Namespace(model="org/m")
    applied = rust.default_kv_events_config(args, environ={})
    assert _kv_events(applied) == KV_EVENTS_ON
    assert args.kv_events_config is applied
    args = argparse.Namespace(model="org/m", kv_events_config=None)
    assert _kv_events(rust.default_kv_events_config(args, environ={})) == KV_EVENTS_ON
    # The opt-out.
    for value in ("0", "false", "No", " off "):
        args = argparse.Namespace(model="org/m", kv_events_config=None)
        assert rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: value}) is None
        assert args.kv_events_config is None
    args = argparse.Namespace(model="org/m")
    applied = rust.default_kv_events_config(args, environ={rust.KV_EVENTS_ENV: "1"})
    assert _kv_events(applied) == KV_EVENTS_ON
    # An explicit configuration is kept as given, off included.
    given = SimpleNamespace(enable_kv_cache_events=False, publisher="null")
    args = argparse.Namespace(model="org/m", kv_events_config=given)
    assert rust.default_kv_events_config(args, environ={}) is None
    assert args.kv_events_config is given

    # vLLM's own dataclass once vllm.config is importable, in vLLM 0.31's
    # shape: the typed form carries the same two settings and leaves the rest
    # (the endpoint among them) at vLLM's defaults.
    @dataclasses.dataclass
    class KVEventsConfig:
        enable_kv_cache_events: bool = False
        publisher: str | None = None
        endpoint: str = KV_EVENTS_ENDPOINT
        replay_endpoint: str | None = None
        buffer_steps: int = 10_000
        hwm: int = 100_000
        max_queue_size: int = 100_000
        topic: str = ""

        def __post_init__(self):
            if self.publisher is None:
                self.publisher = "zmq" if self.enable_kv_cache_events else "null"

    _install(monkeypatch, "vllm.config", KVEventsConfig=KVEventsConfig)
    args = argparse.Namespace(model="org/m")
    applied = rust.default_kv_events_config(args, environ={})
    assert isinstance(applied, KVEventsConfig) and _kv_events(applied) == KV_EVENTS_ON
    assert applied == KVEventsConfig(enable_kv_cache_events=True, publisher="zmq")
    assert args.kv_events_config is applied


def test_serve_rust_wires_the_server_the_engine_and_the_supervisor(monkeypatch, tmp_path):
    """The upstream hook hands `serve_rust` its parsed namespace: the Rust
    server binds the launcher's host/port, the engine cores are launched
    from that same namespace (plus the handshake), and both are supervised.
    argv is never consulted, so the `vllm serve <model> --grpc` form works."""
    recorded: dict = {}

    class AsyncEngineArgs:
        @staticmethod
        def from_cli_args(args):
            recorded["engine_args_from"] = args
            return SimpleNamespace(create_engine_config=lambda usage_context: _config())

    _stub_frontend_args(monkeypatch)
    _stub_engine_exceptions(monkeypatch)
    _install(monkeypatch, "vllm.engine.arg_utils", AsyncEngineArgs=AsyncEngineArgs)
    _install(monkeypatch, "vllm.usage.usage_lib", UsageContext=SimpleNamespace(OPENAI_API_SERVER=1))
    _install(monkeypatch, "smg")
    _install(
        monkeypatch,
        "smg.servicer",
        VllmGrpcServer=FakeServer,
        init_servicer_tracing=lambda: recorded.setdefault("tracing", True),
    )
    launched: list[argparse.Namespace] = []

    def fake_launch(ns):
        launched.append(ns)
        return FakeEngine()

    monkeypatch.setattr(rust, "launch_engine_cores", fake_launch)
    monkeypatch.setattr(rust, "resolve_tokenizer_dir", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(rust, "free_port", lambda: 24321)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "org/m", "--grpc", "--port", "50051"])
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setenv("SMG_VLLM_SERVICER_DRAIN_SECS", "0")

    async def fake_supervise(server, engine, *, drain_secs, **_kwargs):
        recorded["supervised"] = (server, engine, drain_secs)
        return 0

    monkeypatch.setattr(rust, "supervise", fake_supervise)

    args = argparse.Namespace(
        model_tag="org/m", model="org/m", grpc=True, host=None, port=50051, max_model_len=4096
    )
    assert asyncio.run(rust.serve_rust(args)) == 0

    server, engine, drain_secs = recorded["supervised"]
    assert isinstance(server, FakeServer) and isinstance(engine, FakeEngine)
    assert drain_secs == 0.0
    assert recorded["tracing"] is True

    # A launch that fails stops the already-bound server before propagating.
    servers: list[FakeServer] = []

    def recording_server(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]

    _install(
        monkeypatch,
        "smg.servicer",
        VllmGrpcServer=recording_server,
        init_servicer_tracing=lambda: None,
    )

    def failing_launch(ns):
        raise RuntimeError("engine would not start")

    monkeypatch.setattr(rust, "launch_engine_cores", failing_launch)
    with pytest.raises(RuntimeError, match="engine would not start"):
        asyncio.run(rust.serve_rust(args))
    assert servers[0].events == ["serving:False", "stop:5.0"]
    assert recorded["engine_args_from"] is args
    kwargs = server.kwargs
    # `vllm serve` leaves host unset; upstream binds every interface then.
    assert kwargs["bind_address"] == "0.0.0.0:50051"
    assert kwargs["ipc_base_url"] == f"ipc://{tmp_path}/servicer-{os.getpid()}"
    assert kwargs["handshake_address"] == "tcp://127.0.0.1:24321"
    assert kwargs["engine_count"] == 2
    assert kwargs["tokenizer_dir"] == str(tmp_path)
    assert kwargs["engine_startup_timeout_secs"] == rust_lifecycle.DEFAULT_STARTUP_TIMEOUT_SECS
    assert kwargs["engine_startup_ceiling_secs"] == rust_lifecycle.DEFAULT_STARTUP_CEILING_SECS
    assert kwargs["served_model_name"] == "served-a"
    assert kwargs["eos_token_ids"] == [151645, 151643, 7]
    assert kwargs["kv_connector"] == ""

    (ns,) = launched
    assert ns.model == "org/m" and ns.model_tag is None and ns.grpc is False
    assert ns.headless is True and ns.max_model_len == 4096
    assert ns.data_parallel_rpc_port == 24321
    assert ns.data_parallel_size == 2 and ns.data_parallel_size_local == 2
    # No --kv-events-config was given: the engine args and the engine cores
    # both see the publisher on, in whichever form vllm.config's presence
    # decides (the typed one in the engine-gated job).
    assert _kv_events(recorded["engine_args_from"].kv_events_config) == KV_EVENTS_ON
    assert _kv_events(ns.kv_events_config) == KV_EVENTS_ON
