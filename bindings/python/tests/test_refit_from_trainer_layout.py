"""The trainer-side refit example's rank layout (no gateway, no GPU, no engine).

`rank_layout` decides which ranks of the trainer's NCCL group each engine
occupies, which is the one piece of `examples/rl/refit_from_trainer.py` that is
pure arithmetic and the one an off-by-one silently deadlocks.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "examples" / "rl" / "refit_from_trainer.py"
SRC = str(REPO / "bindings" / "python" / "src")


def _smg_modules() -> list[str]:
    return [name for name in sys.modules if name == "smg" or name.startswith("smg.")]


@contextmanager
def _loaded_script() -> Iterator[Any]:
    """Import the example by path, with this checkout's `smg.rl` importable.

    The example imports `smg.rl`, torch and transformers at the top, the way a
    trainer would, and `smg` is usually already imported from wherever the
    package was installed -- a checkout that need not be this one, and whose
    compiled `smg_rs` extension is not in this source tree. So point `smg` at
    this checkout's sources for the import and put the interpreter back
    afterwards: leaving either `sys.path` or `sys.modules` shifted breaks every
    later test in the session that imports `smg.smg_rs`.
    """
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules[name] for name in _smg_modules()}
    try:
        sys.path.insert(0, SRC)
        for name in list(saved_modules):
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location("refit_from_trainer", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for name in _smg_modules():
            del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module")
def script() -> Iterator[Any]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    with _loaded_script() as module:
        yield module


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tp_sizes", "expected"),
    [
        ([], (1, [])),
        ([1], (2, [1])),
        ([1, 1], (3, [1, 2])),
        ([2, 4], (7, [1, 3])),
    ],
)
def test_rank_layout(script, tp_sizes, expected):
    assert script.rank_layout(tp_sizes) == expected


@pytest.mark.unit
def test_rank_layout_rejects_a_nonsense_tp_size(script):
    with pytest.raises(ValueError, match="tp_size must be >= 1"):
        script.rank_layout([1, 0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", True),
        ("10.0.1.52", False),
        ("trainer.internal", False),
        (None, False),
        ("", False),
    ],
)
def test_is_loopback(script, host, loopback):
    assert script._is_loopback(host) is loopback


@pytest.mark.unit
def test_loading_the_example_restores_the_interpreter():
    """Importing the example must not leave `smg` pointed at this source tree.

    Every other test file in this suite imports `smg.smg_rs` from inside a test
    body, and this source tree has no compiled extension, so a leaked
    `sys.path` entry or a purged `sys.modules` entry fails all of them.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    before_path = list(sys.path)
    before_modules = {name: sys.modules[name] for name in _smg_modules()}

    with _loaded_script() as module:
        assert module.rank_layout([1]) == (2, [1])
        assert sys.path[0] == SRC

    assert sys.path == before_path
    assert {name: sys.modules[name] for name in _smg_modules()} == before_modules
