"""The trainer-side refit example's rank layout (no gateway, no GPU, no engine).

`rank_layout` decides which ranks of the trainer's NCCL group each engine
occupies, which is the one piece of `examples/rl/refit_from_trainer.py` that is
pure arithmetic and the one an off-by-one silently deadlocks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "examples" / "rl" / "refit_from_trainer.py"


def _load_script():
    """Import the example by path, with this checkout's `smg.rl` importable.

    The example imports `smg.rl`, torch and transformers at the top, the way a
    trainer would. `smg` may already be imported from an unrelated install, so
    point it at this checkout's source tree first.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    src = str(REPO / "bindings" / "python" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
        for name in [m for m in sys.modules if m == "smg" or m.startswith("smg.")]:
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("refit_from_trainer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def test_rank_layout(tp_sizes, expected):
    assert _load_script().rank_layout(tp_sizes) == expected


@pytest.mark.unit
def test_rank_layout_rejects_a_nonsense_tp_size():
    with pytest.raises(ValueError, match="tp_size must be >= 1"):
        _load_script().rank_layout([1, 0])
