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

"""Exercise the servicer request-conversion bodies with real protobufs.

Extracting these two functions avoids loading the GPU engine. Their statements
execute unchanged; only the engine request constructor is replaced by a capture
object. Missing protobuf dependencies fail collection instead of skipping.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2 as proto


@pytest.fixture(scope="module")
def servicer():
    source = (
        Path(__file__).resolve().parents[1] / "smg_grpc_servicer/tokenspeed/servicer.py"
    ).read_text()
    tree = ast.parse(source)
    selected = []
    for name in ("_sampling_params_from_proto", "_build_generate_req"):
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(matches) == 1
        function = matches[0]
        function.decorator_list = []
        selected.append(function)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    namespace = {
        "json": json,
        "_engine_supports_dp_rank_pin": lambda: False,
        "_lazy_generate_req_input": lambda: SimpleNamespace,
    }
    exec(compile(ast.fix_missing_locations(module), "tokenspeed-servicer", "exec"), namespace)
    converter = namespace["_sampling_params_from_proto"]
    instance = SimpleNamespace(
        server_args=SimpleNamespace(reasoning_parser=None),
        _sampling_params_from_proto=converter,
    )
    return converter, namespace["_build_generate_req"], instance


def sampling(seed):
    params = proto.SamplingParams(temperature=1.0, max_new_tokens=256)
    if seed is not None:
        params.sampling_seed = seed
    return params


@pytest.mark.parametrize("seed", [None, 0, 42, 2**32 - 1, 2**64 - 1])
def test_optional_seed_preserved_in_converter(servicer, seed):
    convert, _, _ = servicer
    result = convert(sampling(seed))
    if seed is None:
        assert "seed" not in result
    else:
        assert result["seed"] == seed
    assert "sampling_seed" not in result
    assert result["temperature"] == 1.0
    assert result["max_new_tokens"] == 256


@pytest.mark.parametrize("seed", [None, 0, 42, 2**32 - 1, 2**64 - 1])
@pytest.mark.parametrize("choices", [1, 2])
def test_seed_reaches_engine_request(servicer, seed, choices):
    _, build, instance = servicer
    params = sampling(seed)
    params.n = choices
    request = proto.GenerateRequest(
        request_id="seed-contract",
        tokenized=proto.TokenizedInput(input_ids=[1, 2, 3]),
        sampling_params=params,
    )
    result = build(instance, request)
    if seed is None:
        assert "seed" not in result.sampling_params
    else:
        assert result.sampling_params["seed"] == seed
    assert result.input_ids == [1, 2, 3]
    assert result.rid == (
        "seed-contract" if choices == 1 else ["seed-contract-n0", "seed-contract-n1"]
    )
