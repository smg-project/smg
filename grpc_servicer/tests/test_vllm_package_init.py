"""The vLLM servicer package must import without vLLM (engine-free submodules)."""

import contextlib
import importlib
import io
import logging
from types import SimpleNamespace

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


@contextlib.contextmanager
def _loggers_before_vllm_configures():
    """Both loggers as they are when the servicer module is imported under vLLM 0.31+."""
    loggers = (logging.getLogger("vllm"), logging.getLogger("smg_grpc_servicer"))
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in loggers]
    try:
        for lg in loggers:
            lg.handlers = []
            lg.setLevel(logging.NOTSET)
            lg.propagate = True
        yield
    finally:
        for lg, (handlers, level, propagate) in zip(loggers, saved, strict=True):
            lg.handlers = handlers
            lg.setLevel(level)
            lg.propagate = propagate


def _configure_like_vllm(stream):
    """What the launcher's configure_logging_from_args(args) does to the vllm logger."""
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = [logging.StreamHandler(stream)]
    vllm_logger.setLevel(logging.INFO)


def test_logging_attach_takes_effect_when_vllm_configures_after_the_first_call():
    from smg_grpc_servicer.vllm import attach_vllm_logging

    stream = io.StringIO()
    with _loggers_before_vllm_configures():
        attach_vllm_logging()
        _configure_like_vllm(stream)
        attach_vllm_logging()
        logging.getLogger("smg_grpc_servicer.vllm.servicer").info(
            "Generate request r1: media_refs=1"
        )
    assert "Generate request r1: media_refs=1" in stream.getvalue()


def test_servicer_attaches_logging_that_vllm_configured_after_its_import(monkeypatch):
    pytest.importorskip("vllm")
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

    monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)
    stream = io.StringIO()
    with _loggers_before_vllm_configures():
        _configure_like_vllm(stream)
        VllmEngineServicer(SimpleNamespace(), start_time=0.0)
    assert "VllmEngineServicer initialized" in stream.getvalue()
