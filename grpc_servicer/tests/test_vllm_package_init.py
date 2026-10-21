"""The vLLM servicer package must import without vLLM (engine-free submodules)."""

import contextlib
import importlib
import inspect
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


def test_logging_attach_leaves_the_package_alone_where_a_handler_prints_its_records_already():
    """No vLLM handlers, but a handler on the root logger (a launcher's own
    configuration; a test runner's capture): nothing attached, nothing held."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    pkg_logger = logging.getLogger("smg_grpc_servicer")
    with _nothing_configured():
        logging.getLogger().addHandler(logging.StreamHandler(io.StringIO()))
        assert attach_vllm_logging() is False
        assert (pkg_logger.handlers, pkg_logger.level, pkg_logger.propagate) == (
            [],
            logging.NOTSET,
            True,
        )


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


@contextlib.contextmanager
def _nothing_configured():
    """A launcher process before vLLM configures logging: no handler on vLLM's
    logger, on the package's, or on the root logger they propagate to (where
    the test runner's own capture handler sits), the root at its default level."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    with _loggers_before_vllm_configures():
        root.handlers = []
        root.setLevel(logging.WARNING)
        try:
            yield
        finally:
            root.handlers, level = saved
            root.setLevel(level)


_FORMAT = "%(levelname)s [%(filename)s:%(lineno)d] %(name)s %(message)s"


def _configure_like_vllm(stream, level=logging.INFO, fmt=_FORMAT):
    """What the launcher's configure_logging_from_args(args) does to the vllm
    logger: one handler at the configured level, a format with file and line."""
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.handlers = [handler]
    vllm_logger.setLevel(level)


def _log(name: str, level: int, message: str, *args) -> int:
    """A line as a module of the package logs it; returns the line number of
    the call (this helper's, the same for every line it logs), which a replay
    must keep: the replayed record is of the original call, not of the replay."""
    logging.getLogger(name).log(level, message, *args)
    return inspect.currentframe().f_lineno - 1


def _log_the_derivation() -> int:
    """The capture-cap line as the plugin's parse step logs it."""
    return _log(
        "smg_grpc_servicer.vllm.capture_cap",
        logging.INFO,
        "CUDA-graph capture cap: %s",
        "512, derived from the admission step",
    )


def test_records_logged_before_vllm_configures_are_held_and_replayed_once(monkeypatch):
    """vLLM 0.31 configures logging after the launcher's argument parse, during
    which the plugin's parse step logs the capture-cap decision: the record is
    held and comes out through vLLM's handlers at the first attach after they
    exist, once, in order with what else was held, under its own file and line."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    pkg_logger = logging.getLogger("smg_grpc_servicer")
    with _nothing_configured():
        assert attach_vllm_logging() is False
        assert pkg_logger.level == logging.INFO  # the bootstrap level: INFO records are made
        line = _log_the_derivation()
        _log("smg_grpc_servicer.vllm.mm_processor", logging.WARNING, "held %s", "too")
        assert stream.getvalue() == ""
        _configure_like_vllm(stream)  # the launcher's configure_logging_from_args(args)
        assert attach_vllm_logging() is True
        replayed = stream.getvalue().splitlines()
        assert attach_vllm_logging() is True
        _log("smg_grpc_servicer.vllm.rust", logging.INFO, "after")
        assert pkg_logger.propagate is False
    assert replayed == [
        f"INFO [test_vllm_package_init.py:{line}] smg_grpc_servicer.vllm.capture_cap "
        "CUDA-graph capture cap: 512, derived from the admission step",
        f"WARNING [test_vllm_package_init.py:{line}] smg_grpc_servicer.vllm.mm_processor held too",
    ]
    assert stream.getvalue().splitlines()[len(replayed) :] == [
        f"INFO [test_vllm_package_init.py:{line}] smg_grpc_servicer.vllm.rust after"
    ]


def test_a_replayed_record_is_made_by_the_record_factory_installed_since(monkeypatch):
    """vLLM installs a record factory with its logging configuration and its
    formatter prints what the factory adds: a record held from before cannot
    be handed to its handlers as it is."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    factory = logging.getLogRecordFactory()

    def with_process_name(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.vllm_process_name = "EngineCore"
        return record

    with _nothing_configured():
        attach_vllm_logging()
        _log_the_derivation()
        logging.setLogRecordFactory(with_process_name)
        try:
            _configure_like_vllm(stream, fmt="(%(vllm_process_name)s) %(message)s")
            attach_vllm_logging()
        finally:
            logging.setLogRecordFactory(factory)
    assert stream.getvalue() == (
        "(EngineCore) CUDA-graph capture cap: 512, derived from the admission step\n"
    )


def test_held_records_follow_the_level_vllm_configures(monkeypatch):
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        attach_vllm_logging()
        _log_the_derivation()
        logging.getLogger("smg_grpc_servicer.vllm.mm_processor").warning("kept")
        _configure_like_vllm(stream, level=logging.WARNING)
        attach_vllm_logging()
    assert [line.split(" ", 1)[0] for line in stream.getvalue().splitlines()] == ["WARNING"]


@pytest.mark.parametrize(
    ("setting", "level"),
    [(None, logging.INFO), ("DEBUG", logging.DEBUG), ("warning", logging.WARNING), ("loud", 20)],
)
def test_the_hold_is_at_vllms_bootstrap_level(monkeypatch, setting, level):
    """What vLLM's own handler starts at (VLLM_LOGGING_LEVEL, INFO when unset,
    INFO for a value that names no level): the records below it are not made."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    if setting is None:
        monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    else:
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", setting)
    stream = io.StringIO()
    pkg_logger = logging.getLogger("smg_grpc_servicer")
    with _nothing_configured():
        attach_vllm_logging()
        assert pkg_logger.level == level
        for name in ("debug", "info", "warning"):
            getattr(logging.getLogger("smg_grpc_servicer.vllm.capture_cap"), name)(name)
        _configure_like_vllm(stream, level=logging.DEBUG)
        attach_vllm_logging()
    held = [line.rsplit(" ", 1)[-1] for line in stream.getvalue().splitlines()]
    assert held == [
        name for name in ("debug", "info", "warning") if logging.getLevelName(name.upper()) >= level
    ]


def test_the_hold_is_bounded(monkeypatch):
    from smg_grpc_servicer.vllm import HELD_RECORDS, attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        attach_vllm_logging()
        for index in range(HELD_RECORDS + 5):
            logging.getLogger("smg_grpc_servicer.vllm.servicer").info("line %d", index)
        _configure_like_vllm(stream)
        attach_vllm_logging()
    lines = stream.getvalue().splitlines()
    assert len(lines) == HELD_RECORDS
    assert lines[0].endswith("line 0") and lines[-1].endswith(f"line {HELD_RECORDS - 1}")


def test_a_handler_configured_elsewhere_after_the_hold_takes_the_held_records(monkeypatch):
    """A launcher that configures a handler of its own (the root logger's)
    where vLLM configured none: the next attach ends the hold through it and
    puts the package logger back as it was, so its records go on as before."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    pkg_logger = logging.getLogger("smg_grpc_servicer")
    with _nothing_configured():
        attach_vllm_logging()
        line = _log_the_derivation()
        root = logging.getLogger()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        assert attach_vllm_logging() is False
        assert (pkg_logger.handlers, pkg_logger.level, pkg_logger.propagate) == (
            [],
            logging.NOTSET,
            True,
        )
        _log("smg_grpc_servicer.vllm.rust", logging.INFO, "after")
    assert stream.getvalue().splitlines() == [
        f"INFO [test_vllm_package_init.py:{line}] smg_grpc_servicer.vllm.capture_cap "
        "CUDA-graph capture cap: 512, derived from the admission step",
        f"INFO [test_vllm_package_init.py:{line}] smg_grpc_servicer.vllm.rust after",
    ]


def test_a_record_another_handler_prints_during_the_hold_is_not_held(monkeypatch):
    """A handler configured after the hold began (a launcher's own on the
    root logger) prints the package's records as they come; the release
    must not print them again."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        attach_vllm_logging()
        held_line = _log_the_derivation()
        root = logging.getLogger()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        _log("smg_grpc_servicer.vllm.servicer", logging.INFO, "printed as it comes")
        assert attach_vllm_logging() is False
    assert stream.getvalue().splitlines() == [
        f"INFO [test_vllm_package_init.py:{held_line}] smg_grpc_servicer.vllm.servicer "
        "printed as it comes",
        f"INFO [test_vllm_package_init.py:{held_line}] smg_grpc_servicer.vllm.capture_cap "
        "CUDA-graph capture cap: 512, derived from the admission step",
    ]


def test_a_mapping_argument_and_the_callers_extra_attributes_survive_the_replay(monkeypatch):
    from smg_grpc_servicer.vllm import attach_vllm_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        attach_vllm_logging()
        logging.getLogger("smg_grpc_servicer.vllm.servicer").info(
            "request %(id)s: %(n)d media items", {"id": "r1", "n": 2}, extra={"rank": 3}
        )
        _configure_like_vllm(stream, fmt="%(levelname)s rank=%(rank)d %(message)s")
        attach_vllm_logging()
    assert stream.getvalue() == "INFO rank=3 request r1: 2 media items\n"


def test_settle_logging_ends_a_hold_nothing_released_as_without_it(monkeypatch, capsys):
    """No handler ever appears (nothing configured logging): the hold ends when
    the launcher's moment to configure it is past; what was held and what comes
    after goes where the standard library sends it, its last-resort stderr
    handler for warnings and errors, nowhere for the rest, as before the hold."""
    from smg_grpc_servicer.vllm import attach_vllm_logging, settle_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    pkg_logger = logging.getLogger("smg_grpc_servicer")
    with _nothing_configured():
        attach_vllm_logging()
        _log_the_derivation()
        _log("smg_grpc_servicer.vllm.servicer", logging.ERROR, "held error")
        assert capsys.readouterr().err == ""
        assert settle_logging() is False
        assert (pkg_logger.handlers, pkg_logger.level) == ([], logging.NOTSET)
        released = capsys.readouterr().err
        logging.getLogger("smg_grpc_servicer.vllm.servicer").error("later error")
        logging.getLogger("smg_grpc_servicer.vllm.servicer").info("later info")
        later = capsys.readouterr().err
    assert released == "held error\n"
    assert later == "later error\n"


def test_settle_logging_attaches_where_vllm_configured_logging(monkeypatch):
    from smg_grpc_servicer.vllm import settle_logging

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        _configure_like_vllm(stream)
        assert settle_logging() is True
        _log("smg_grpc_servicer.vllm.servicer", logging.INFO, "through vllm")
    assert stream.getvalue().endswith("smg_grpc_servicer.vllm.servicer through vllm\n")


def test_release_held_records_counts_what_it_replayed(monkeypatch):
    from smg_grpc_servicer.vllm import attach_vllm_logging, has_handlers, release_held_records

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    stream = io.StringIO()
    with _nothing_configured():
        assert release_held_records() == 0
        attach_vllm_logging()
        assert has_handlers() is False  # the hold is not a handler that prints
        _log_the_derivation()
        _log_the_derivation()
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(stream))
        assert has_handlers() is True
        assert release_held_records() == 2
        assert release_held_records() == 0
    assert stream.getvalue().count("CUDA-graph capture cap") == 2


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
