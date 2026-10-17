"""The ``--servicer-impl`` registration the three engine plugins share, and
the weight the plugins may carry (they are loaded in every engine process)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PLUGINS = (
    "smg_grpc_servicer.vllm.plugin",
    "smg_grpc_servicer.sglang.plugin",
)


@pytest.mark.parametrize("module", PLUGINS)
def test_a_plugin_loads_without_the_lifecycle_module(module):
    """The plugins import the flag module (stdlib only), not the lifecycle
    module that acts on the choice: a fresh interpreter that imports a plugin
    has not loaded ``rust_lifecycle``."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (root, env.get("SMG_GRPC_PROTO_PATH", ""), env.get("PYTHONPATH", "")) if p
    )
    code = (
        f"import sys; import {module}; "
        "assert 'smg_grpc_servicer.servicer_impl_flag' in sys.modules; "
        "assert 'smg_grpc_servicer.rust_lifecycle' not in sys.modules, 'the lifecycle module is loaded'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


class TestServicerImplArgument:
    """The one registration of --servicer-impl the three engines share."""

    @staticmethod
    def _parser():
        import argparse

        return argparse.ArgumentParser()

    def test_the_flag_is_added_once_with_the_shared_choices(self):
        from smg_grpc_servicer.servicer_impl_flag import IMPLS, add_servicer_impl_argument

        parser = self._parser()
        assert add_servicer_impl_argument(parser, help="h") is True
        assert add_servicer_impl_argument(parser, help="h") is False
        assert parser.parse_args(["--servicer-impl", "rust"]).servicer_impl == "rust"
        assert parser.parse_args([]).servicer_impl is None
        action = parser._option_string_actions["--servicer-impl"]  # noqa: SLF001 — argparse's registry
        assert list(action.choices) == list(IMPLS)

    def test_a_parser_the_guard_refuses_is_left_alone(self):
        from smg_grpc_servicer.servicer_impl_flag import add_servicer_impl_argument

        parser = self._parser()
        assert add_servicer_impl_argument(parser, help="h", applies=lambda _p: False) is False
        assert "--servicer-impl" not in parser._option_string_actions  # noqa: SLF001 — argparse's registry

    def test_the_hook_runs_once_before_the_argument_and_the_action_is_honoured(self):
        import argparse

        from smg_grpc_servicer.servicer_impl_flag import add_servicer_impl_argument

        seen: list[str] = []

        class Remember(argparse._StoreAction):  # noqa: SLF001 — a store action, as the engines use
            def __call__(self, parser, namespace, values, option_string=None):
                seen.append(f"action:{values}")
                setattr(namespace, self.dest, values)

        parser = self._parser()
        added = add_servicer_impl_argument(
            parser, help="h", action=Remember, on_add=lambda: seen.append("hook")
        )
        assert added is True
        assert seen == ["hook"]
        assert parser.parse_args(["--servicer-impl", "python"]).servicer_impl == "python"
        assert seen == ["hook", "action:python"]
        assert (
            add_servicer_impl_argument(parser, help="h", on_add=lambda: seen.append("again"))
            is False
        )
        assert "again" not in seen
