# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file to
# you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import ctypes

import pytest

try:
    import torch
    import tvm_ffi  # noqa: F401
    from torch.utils import cpp_extension
    from tvm_ffi import libinfo
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]

if torch is None:
    _HAS_TORCH = False
    _HAS_GPU = False
    _HAS_DLPACK_EXCHANGE_API = False
else:
    _HAS_TORCH = True
    _HAS_GPU = bool(torch.cuda.is_available())
    _HAS_DLPACK_EXCHANGE_API = bool(hasattr(torch.Tensor, "__dlpack_c_exchange_api__"))


@pytest.mark.skipif(not _HAS_TORCH, reason="Requires torch")
@pytest.mark.skipif(not _HAS_GPU, reason="Requires GPU runtime")
@pytest.mark.skipif(not _HAS_DLPACK_EXCHANGE_API, reason="Requires __dlpack_c_exchange_api__")
def test_current_work_stream_matches_torch_stream() -> None:
    assert torch is not None
    api_attr = torch.Tensor.__dlpack_c_exchange_api__  # ty: ignore[unresolved-attribute]

    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_size_t
    pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    api_ptr = pythonapi.PyCapsule_GetPointer(api_attr, b"dlpack_exchange_api")
    assert api_ptr != 0

    source = r"""
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void assert_current_work_stream(int64_t api_ptr_int, bool is_hip, int64_t expected_stream) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);
        TORCH_CHECK(api != nullptr, "API pointer is NULL");
        TORCH_CHECK(api->current_work_stream != nullptr, "current_work_stream is NULL");

        void* stream_cuda = nullptr;
        int result_cuda = api->current_work_stream(kDLCUDA, 0, &stream_cuda);
        TORCH_CHECK(result_cuda == 0, "current_work_stream(kDLCUDA) failed");
        TORCH_CHECK(reinterpret_cast<int64_t>(stream_cuda) == expected_stream,
                    "kDLCUDA stream mismatch");

        if (is_hip) {
            void* stream_rocm = nullptr;
            int result_rocm = api->current_work_stream(kDLROCM, 0, &stream_rocm);
            TORCH_CHECK(result_rocm == 0, "current_work_stream(kDLROCM) failed");
            TORCH_CHECK(reinterpret_cast<int64_t>(stream_rocm) == expected_stream,
                        "kDLROCM stream mismatch");
        }
    }
    """

    include_paths = libinfo.include_paths()
    include_paths += cpp_extension.include_paths("cuda")

    mod = cpp_extension.load_inline(
        name="test_current_work_stream_gpu_ext",
        cpp_sources=[source],
        functions=["assert_current_work_stream"],
        with_cuda=torch.cuda.is_available(),
        extra_include_paths=include_paths,
    )

    device_id = torch.cuda.current_device()
    is_hip = torch.version.hip is not None
    stream = torch.cuda.Stream(device=device_id)
    with torch.cuda.stream(stream):
        expected_stream = int(stream.cuda_stream)
        mod.assert_current_work_stream(api_ptr, is_hip, expected_stream)
