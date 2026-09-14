"""The vLLM servicer package must import without vLLM (engine-free submodules)."""

import importlib
import logging

import pytest


def test_package_imports_without_vllm():
    pkg = importlib.import_module("smg_grpc_servicer.vllm")
    assert "VllmEngineServicer" in pkg.__all__
    # Engine-free submodules resolve through the package without loading vLLM.
    media_refs = importlib.import_module("smg_grpc_servicer.vllm.media_refs")
    assert media_refs.advertised_schemes("") == "http,https,data"


def test_servicer_attribute_loads_lazily():
    pkg = importlib.import_module("smg_grpc_servicer.vllm")
    try:
        import vllm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            pkg.VllmEngineServicer  # noqa: B018
    else:
        assert pkg.VllmEngineServicer.__name__ == "VllmEngineServicer"
    with pytest.raises(AttributeError):
        pkg.NoSuchServicer  # noqa: B018


def test_logging_attach_is_a_no_op_without_vllm_handlers():
    from smg_grpc_servicer.vllm import attach_vllm_logging

    pkg_logger = logging.getLogger("smg_grpc_servicer")
    before = (list(pkg_logger.handlers), pkg_logger.propagate)
    vllm_logger = logging.getLogger("vllm")
    saved = list(vllm_logger.handlers)
    vllm_logger.handlers = []
    try:
        attach_vllm_logging()
        assert (list(pkg_logger.handlers), pkg_logger.propagate) == before
    finally:
        vllm_logger.handlers = saved
