"""Three prepared native controls; no model, retry, skip or benchmark."""

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path


def record(name, value):
    value["versions"] = {
        n: importlib.metadata.version(n)
        for n in ["torch", "sglang", "apache-tvm-ffi", "torch-c-dlpack-ext", "tilelang", "xgrammar"]
    }
    Path("diagnostic-results/native/" + name + ".json").write_text(
        json.dumps(value, indent=2) + "\n"
    )


def whole_vendor(name):
    path = Path(__file__).with_name("vendor") / (name + ".py")
    hashes = json.loads((path.parent / "hashes.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
    spec = importlib.util.spec_from_file_location("smg_native_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gfx942(torch):
    assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942")


def test_01_ffi_native_api_and_current_stream():
    import torch

    gfx942(torch)
    assert importlib.metadata.version("apache-tvm-ffi") == "0.1.12"
    assert importlib.metadata.version("torch-c-dlpack-ext") == "0.1.5"
    native = torch.Tensor.__dlpack_c_exchange_api__
    static = inspect.getattr_static(torch.Tensor, "__dlpack_c_exchange_api__")
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_size_t
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    native_pointer = get_pointer(native, b"dlpack_exchange_api")
    assert native_pointer != 0, "native DLPack capsule pointer is NULL"
    assert os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") == "0"
    original_cdll, original_run = ctypes.CDLL, subprocess.run
    attempts = []

    def refuse_addon(path, *args, **kwargs):
        if "libtorch_c_dlpack_addon" in str(path):
            attempts.append(str(path))
            raise AssertionError("unexpected addon load")
        return original_cdll(path, *args, **kwargs)

    def refuse_addon_build(args, *pos, **kwargs):
        if "_build_optional_torch_c_dlpack.py" in str(args):
            attempts.append(str(args))
            raise AssertionError("unexpected addon JIT")
        return original_run(args, *pos, **kwargs)

    ctypes.CDLL, subprocess.run = refuse_addon, refuse_addon_build
    try:
        import torch_c_dlpack_ext.core as addon
        import tvm_ffi  # noqa: F401
        from tvm_ffi import _optional_torch_c_dlpack as optional

        expected = {
            "apache-tvm-ffi": (optional, "tvm_ffi/_optional_torch_c_dlpack.py"),
            "torch-c-dlpack-ext": (addon, "torch_c_dlpack_ext/core.py"),
        }
        for package, (module, filename) in expected.items():
            origin = Path(importlib.metadata.distribution(package).locate_file(filename)).resolve()
            assert Path(module.__file__).resolve() == origin, "repair import origin differs"
        assert addon._lib is None and optional._LIB is None
        after = torch.Tensor.__dlpack_c_exchange_api__
        assert inspect.getattr_static(torch.Tensor, "__dlpack_c_exchange_api__") is static
        assert get_pointer(after, b"dlpack_exchange_api") == native_pointer
    finally:
        ctypes.CDLL, subprocess.run = original_cdll, original_run
    assert not attempts, "addon load/build attempted, even if caught by upstream"
    official = whole_vendor("current_work_stream_gpu")
    assert official._HAS_TORCH and official._HAS_GPU and official._HAS_DLPACK_EXCHANGE_API
    # Direct call preserves the complete official body and prevents marker skips.
    official.test_current_work_stream_matches_torch_stream()
    import tvm_ffi

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.arange(8, device="cuda", dtype=torch.int64)
        other = torch.utils.dlpack.from_dlpack(tvm_ffi.from_dlpack(tensor))
        assert other.data_ptr() == tensor.data_ptr() and other.device == tensor.device
        other[0] = 17
    stream.synchronize()
    assert tensor.tolist() == [17, 1, 2, 3, 4, 5, 6, 7]
    record(
        "ffi",
        {
            "native_API_preserved": True,
            "native_API_pointer": native_pointer,
            "native_API_type": type(native).__name__,
            "static_descriptor_type": type(static).__name__,
            "dynamic_object_identity_preserved": after is native,
            "addon_load_or_JIT_attempts": attempts,
            "nondefault_stream": int(stream.cuda_stream),
            "shared_GPU_storage": True,
        },
    )


def test_02_xgrammar_cpu_compile_and_match():
    import xgrammar as xgr

    assert importlib.metadata.version("xgrammar") == "0.2.7"
    tokenizer = xgr.TokenizerInfo(["a", "b"], xgr.VocabType.RAW, stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1, cache_enabled=False)
    compiled = compiler.compile_grammar(xgr.Grammar.from_ebnf('root ::= "a"'))
    rejected = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not rejected.accept_token(1)
    assert not rejected.is_terminated()
    accepted = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert accepted.accept_token(0)
    assert accepted.is_terminated()
    record("xgrammar", {"grammar": 'root ::= "a"', "accept_a": True, "reject_b": True})


def test_03_tilelang_gfx942_elementwise():
    import torch

    gfx942(torch)
    assert importlib.metadata.version("tilelang") == "0.1.7.post3+cuda.gita55a8230"
    official = whole_vendor("elementwise_add")
    kernel = official.elementwise_add(32, 32, 32, 32, "float32", "float32", 128)
    left = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(32, 32)
    right = torch.full_like(left, 3)
    left_before, right_before = left.clone(), right.clone()
    output = kernel(left, right)
    torch.cuda.synchronize()
    assert output.shape == left.shape and output.dtype == left.dtype
    torch.testing.assert_close(output, left + right, rtol=0, atol=0)
    assert torch.equal(left, left_before) and torch.equal(right, right_before)
    record("tilelang", {"shape": [32, 32], "gfx942": True, "exact_reference_match": True})
