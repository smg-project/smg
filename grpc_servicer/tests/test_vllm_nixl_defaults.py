"""The launch defaults a gRPC launch gives vLLM's NIXL connector in a
prefill/decode role (``smg_grpc_servicer.vllm.nixl_defaults``): where they
apply, what the operator's own values do, the shapes the parsed flag comes
in, and the plugin's parse step that puts them where the launcher builds the
engine from."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import sys
import types

import pytest
from smg_grpc_servicer.vllm import capture_cap, launcher_switch, nixl_defaults, plugin
from smg_grpc_servicer.vllm.nixl_defaults import (
    DEFAULT_ENGINE_TTL_SECS,
    UCX_IPC_CACHE_ENV,
    apply_nixl_defaults,
    nixl_pd_role,
)

NIXL_CONSUMER = {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}
NIXL_PRODUCER = {"kv_connector": "NixlConnector", "kv_role": "kv_producer"}


def _namespace(**fields) -> argparse.Namespace:
    return argparse.Namespace(**fields)


class _KvTransferConfig:
    """The engine's config object, as far as the defaults read and write it."""

    def __init__(self, **fields):
        self.kv_connector = fields.get("kv_connector")
        self.kv_role = fields.get("kv_role")
        self.kv_connector_extra_config = dict(fields.get("kv_connector_extra_config", {}))


# ---------------------------------------------------------------------------
# Where the defaults apply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "role"),
    [
        (NIXL_CONSUMER, "kv_consumer"),
        (NIXL_PRODUCER, "kv_producer"),
        ({"kv_connector": "NixlConnector", "kv_role": "kv_both"}, "kv_both"),
        (json.dumps(NIXL_CONSUMER), "kv_consumer"),
        (_KvTransferConfig(**NIXL_PRODUCER), "kv_producer"),
        ({"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"}, None),
        ({"kv_connector": "MoRIIOConnector", "kv_role": "kv_producer"}, None),
        ({"kv_connector": "NixlConnector"}, None),
        ({"kv_connector": "NixlConnector", "kv_role": "kv_store"}, None),
        # A wrapper is not read into: its children keep the engine's defaults.
        (
            {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {"connectors": [NIXL_CONSUMER]},
            },
            None,
        ),
        ("{not json", None),
        (None, None),
    ],
)
def test_the_defaults_apply_to_a_nixl_prefill_or_decode_side_only(config, role):
    assert nixl_pd_role(_namespace(kv_transfer_config=config)) == role
    assert nixl_pd_role(_namespace()) is None


# ---------------------------------------------------------------------------
# What a launch gets, and what the operator's values do
# ---------------------------------------------------------------------------


def test_a_nixl_decode_side_gets_both_defaults_and_the_log_names_them():
    parsed = _namespace(kv_transfer_config={**NIXL_CONSUMER, "kv_connector_extra_config": {"a": 1}})
    environ: dict[str, str] = {}

    defaults = apply_nixl_defaults(parsed, environ)

    assert environ == {UCX_IPC_CACHE_ENV: "n"}
    assert parsed.kv_transfer_config == {
        **NIXL_CONSUMER,
        "kv_connector_extra_config": {"a": 1, "engine_ttl": DEFAULT_ENGINE_TTL_SECS},
    }
    assert defaults is not None
    assert (defaults.role, defaults.ipc_cache, defaults.ipc_cache_source) == (
        "kv_consumer",
        "n",
        "default",
    )
    assert (defaults.engine_ttl, defaults.engine_ttl_source) == (DEFAULT_ENGINE_TTL_SECS, "default")
    assert defaults.detail == (
        f"NIXL kv_consumer: {UCX_IPC_CACHE_ENV}=n (default), "
        f"engine_ttl={DEFAULT_ENGINE_TTL_SECS} (default)"
    )


@pytest.mark.parametrize("ttl", [0, 0.0, 5, 3600.0])
def test_the_operators_own_values_are_kept_as_given(ttl):
    parsed = _namespace(
        kv_transfer_config={**NIXL_PRODUCER, "kv_connector_extra_config": {"engine_ttl": ttl}}
    )
    environ = {UCX_IPC_CACHE_ENV: "y"}

    defaults = apply_nixl_defaults(parsed, environ)

    assert environ == {UCX_IPC_CACHE_ENV: "y"}
    assert parsed.kv_transfer_config["kv_connector_extra_config"] == {"engine_ttl": ttl}
    assert defaults is not None
    assert (defaults.ipc_cache, defaults.ipc_cache_source) == ("y", "environment")
    assert (defaults.engine_ttl, defaults.engine_ttl_source) == (ttl, "flag")
    assert "engine_ttl=" + str(ttl) + " (flag)" in defaults.detail


def test_an_empty_variable_counts_as_unset():
    environ = {UCX_IPC_CACHE_ENV: ""}
    apply_nixl_defaults(_namespace(kv_transfer_config=dict(NIXL_CONSUMER)), environ)
    assert environ[UCX_IPC_CACHE_ENV] == "n"


def test_a_json_string_flag_is_written_back_as_a_json_string():
    parsed = _namespace(kv_transfer_config=json.dumps(NIXL_CONSUMER))
    apply_nixl_defaults(parsed, {})
    assert isinstance(parsed.kv_transfer_config, str)
    assert json.loads(parsed.kv_transfer_config) == {
        **NIXL_CONSUMER,
        "kv_connector_extra_config": {"engine_ttl": DEFAULT_ENGINE_TTL_SECS},
    }


def test_the_engines_config_object_is_copied_not_written():
    shared = _KvTransferConfig(**NIXL_PRODUCER, kv_connector_extra_config={"a": 1})
    parsed = _namespace(kv_transfer_config=shared)

    apply_nixl_defaults(parsed, {})

    assert parsed.kv_transfer_config is not shared
    assert parsed.kv_transfer_config.kv_connector_extra_config == {
        "a": 1,
        "engine_ttl": DEFAULT_ENGINE_TTL_SECS,
    }
    assert shared.kv_connector_extra_config == {"a": 1}


def test_another_launch_is_left_alone():
    environ: dict[str, str] = {}
    for config in (
        {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"},
        {"kv_connector": "MoRIIOConnector", "kv_role": "kv_producer"},
        None,
    ):
        parsed = _namespace(kv_transfer_config=config)
        assert apply_nixl_defaults(parsed, environ) is None
        assert parsed.kv_transfer_config == config
    assert environ == {}


# ---------------------------------------------------------------------------
# The plugin's parse step
# ---------------------------------------------------------------------------


def _main_module(spec_name: str | None) -> types.ModuleType:
    main = types.ModuleType("__main__")
    main.__spec__ = None if spec_name is None else importlib.machinery.ModuleSpec(spec_name, None)
    return main


@pytest.fixture
def parser_class(monkeypatch):
    """vLLM's parser class as the plugin wraps it, in a process without vLLM
    (the media flags' handoff and the launcher modules are not this file's);
    the environment variable unset, the logged decisions empty."""

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
    monkeypatch.delenv(UCX_IPC_CACHE_ENV, raising=False)
    monkeypatch.setattr(capture_cap, "_logged", set())
    monkeypatch.setattr(nixl_defaults, "_logged", set())
    plugin.register()
    return FlexibleArgumentParser


def _serve_parser(parser_class) -> argparse.ArgumentParser:
    """`vllm serve`'s parser, as far as the defaults read it: the flag is
    read as JSON, as the engine's parser reads it."""
    parser = parser_class(prog="vllm serve")
    parser.add_argument("model", nargs="?")
    parser.add_argument("--grpc", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--kv-transfer-config", type=json.loads, default=None)
    return parser


CONSUMER_FLAGS = ["m", "--kv-transfer-config", json.dumps(NIXL_CONSUMER)]


def test_a_grpc_serve_parse_of_a_nixl_decode_side_carries_the_defaults(parser_class, caplog):
    import os

    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(["--grpc", *CONSUMER_FLAGS])
    assert parsed.kv_transfer_config["kv_connector_extra_config"] == {
        "engine_ttl": DEFAULT_ENGINE_TTL_SECS
    }
    assert os.environ[UCX_IPC_CACHE_ENV] == "n"
    lines = [
        record.getMessage()
        for record in caplog.records
        if "KV transfer launch defaults" in record.getMessage()
    ]
    assert lines == [
        f"KV transfer launch defaults: NIXL kv_consumer: {UCX_IPC_CACHE_ENV}=n (default), "
        f"engine_ttl={DEFAULT_ENGINE_TTL_SECS} (default)"
    ]


def test_a_headless_serve_parse_carries_the_same_defaults(parser_class):
    parsed = _serve_parser(parser_class).parse_args(["--headless", *CONSUMER_FLAGS])
    assert parsed.kv_transfer_config["kv_connector_extra_config"] == {
        "engine_ttl": DEFAULT_ENGINE_TTL_SECS
    }


def test_an_http_serve_parse_keeps_the_engines_defaults(parser_class, caplog):
    import os

    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(CONSUMER_FLAGS)
    assert parsed.kv_transfer_config == NIXL_CONSUMER
    assert UCX_IPC_CACHE_ENV not in os.environ
    assert not [r for r in caplog.records if "KV transfer launch defaults" in r.getMessage()]


def test_a_grpc_serve_parse_without_a_kv_transfer_config_says_nothing(parser_class, caplog):
    import os

    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        parsed = _serve_parser(parser_class).parse_args(["--grpc", "m"])
    assert parsed.kv_transfer_config is None
    assert UCX_IPC_CACHE_ENV not in os.environ
    assert not [r for r in caplog.records if "KV transfer launch defaults" in r.getMessage()]


def test_a_second_parse_with_the_same_flags_logs_nothing_new(parser_class, caplog):
    with caplog.at_level(logging.INFO, logger="smg_grpc_servicer"):
        _serve_parser(parser_class).parse_args(["--grpc", *CONSUMER_FLAGS])
        _serve_parser(parser_class).parse_args(["--grpc", *CONSUMER_FLAGS])
    lines = [r for r in caplog.records if "KV transfer launch defaults" in r.getMessage()]
    assert len(lines) == 1
