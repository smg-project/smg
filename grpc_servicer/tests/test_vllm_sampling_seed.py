# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exercise the vLLM servicer's sampling-params conversion with real protobufs.

The conversion body executes unchanged, extracted from the servicer source so
the GPU engine is never imported; the engine's parameter constructors are
replaced by capture objects. Missing protobuf dependencies fail collection
instead of skipping.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_proto.generated import vllm_engine_pb2 as proto


class Capture(dict):
    """The keyword arguments a constructor was given."""

    def __init__(self, **kwargs):
        super().__init__(kwargs)


@pytest.fixture(scope="module")
def convert():
    source = (
        Path(__file__).resolve().parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    ).read_text()
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sampling_params_from_proto"
    ]
    assert len(matches) == 1
    function = matches[0]
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    namespace = {
        "SamplingParams": Capture,
        "StructuredOutputsParams": Capture,
        "RequestOutputKind": SimpleNamespace(DELTA="delta", FINAL_ONLY="final_only"),
    }
    exec(compile(ast.fix_missing_locations(module), "vllm-servicer", "exec"), namespace)
    return namespace["_sampling_params_from_proto"]


def sampling(seed, n):
    params = proto.SamplingParams(temperature=1.0, max_tokens=256, n=n)
    if seed is not None:
        params.seed = seed
    return params


@pytest.mark.parametrize(
    ("seed", "engine_seed"),
    [
        (0, 0),
        (42, 42),
        (2**31 - 1, 2**31 - 1),
        (-1, 2**32 - 1),
        (-2, 2**32 - 2),
        (-(2**31), 2**31),
    ],
)
def test_seed_reaches_the_engine_as_the_unsigned_word(convert, seed, engine_seed):
    """The engine reads a seed of -1 as "no seed" and, for n > 1, derives each
    child's seed as the base plus the child's index, so a negative base in
    [-n, -2] left one child unseeded: the base reaches the engine in its
    unsigned form, never negative, and no child's seed can be -1."""
    result = convert(sampling(seed, n=2))
    assert result["seed"] == engine_seed
    assert result["n"] == 2
    assert all(result["seed"] + index != -1 for index in range(result["n"]))


def test_no_seed_stays_none(convert):
    result = convert(sampling(None, n=1))
    assert result["seed"] is None
