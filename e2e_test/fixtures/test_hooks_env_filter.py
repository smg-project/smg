"""Unit tests for env-filter deselection reporting and the selection floor (no GPU).

Mirrors the ``_FakeItem`` harness in ``test_hooks_zmq_filter.py``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixtures import hooks

pytest_plugins = ["pytester"]


class _FakeItem:
    """Minimal stand-in for a pytest ``Item`` for the collection helpers.

    Exposes only what the env filter, ``_filter_zmq_items`` and
    ``_pool_sort_key`` touch: ``nodeid``, ``callspec.params``, ``cls`` (always
    None so marker resolution falls back to ``get_closest_marker``), and a
    name->marker mapping served through ``get_closest_marker``.
    """

    def __init__(self, nodeid, params=None, markers=None):
        self.nodeid = nodeid
        self.cls = None
        self.callspec = SimpleNamespace(params=params) if params is not None else None
        self._markers = markers or {}

    def get_closest_marker(self, name):
        return self._markers.get(name)


class _FakeConfig:
    """Fake ``pytest.Config`` recording ``pytest_deselected`` calls."""

    def __init__(self):
        self.stash = pytest.Stash()
        self.deselected = []
        self.hook = SimpleNamespace(pytest_deselected=self._record)

    def _record(self, items):
        self.deselected.extend(items)


def _item(nodeid, engine=None, vendor=None, gpu=None, setup_backend=None):
    markers = {}
    if engine is not None:
        markers["engine"] = pytest.mark.engine(*engine).mark
    if vendor is not None:
        markers["vendor"] = pytest.mark.vendor(*vendor).mark
    if gpu is not None:
        markers["gpu"] = pytest.mark.gpu(gpu).mark
    params = {"setup_backend": setup_backend} if setup_backend is not None else None
    return _FakeItem(nodeid, params=params, markers=markers)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate every test from the ambient lane environment."""
    for var in (
        "E2E_ENGINE",
        "E2E_VENDOR",
        "E2E_GPU_TIER",
        "E2E_CONNECTION_MODE",
        "E2E_RUNTIME",
        "E2E_ZMQ_ENGINE_COUNT",
        "E2E_MIN_SELECTED",
        "E2E_AMD_REQUIRE_PASSING_CALL",
    ):
        monkeypatch.delenv(var, raising=False)


def _run(config, items):
    hooks.pytest_collection_modifyitems(config, items)
    return items


def _summary(config):
    return hooks._SelectionReporter().pytest_report_collectionfinish(config)


# ---------------------------------------------------------------------------
# _filter_env_items
# ---------------------------------------------------------------------------


def test_env_filter_attributes_drops_to_first_rejecting_dimension():
    match = _item("t.py::test_a", engine=("sglang",), gpu=1)
    wrong_engine = _item("t.py::test_b", engine=("vllm",), gpu=1)
    no_marker = _item("t.py::test_c")
    wrong_tier = _item("t.py::test_d", engine=("sglang",), gpu=4)
    selected, deselected_by = hooks._filter_env_items(
        [match, wrong_engine, no_marker, wrong_tier],
        engine="sglang",
        vendor=None,
        gpu_tier="1",
    )
    assert selected == [match]
    assert deselected_by["engine"] == [wrong_engine, no_marker]
    assert deselected_by["vendor"] == []
    assert deselected_by["tier"] == [wrong_tier]


def test_env_filter_gpu_defaults_to_one_without_marker():
    unmarked = _item("t.py::test_a")
    selected, deselected_by = hooks._filter_env_items(
        [unmarked], engine=None, vendor=None, gpu_tier="1"
    )
    assert selected == [unmarked]
    assert deselected_by["tier"] == []


def test_env_filter_by_vendor():
    openai = _item("t.py::test_a", vendor=("openai",))
    xai = _item("t.py::test_b", vendor=("xai",))
    selected, deselected_by = hooks._filter_env_items(
        [openai, xai], engine=None, vendor="openai", gpu_tier=None
    )
    assert selected == [openai]
    assert deselected_by["vendor"] == [xai]


# ---------------------------------------------------------------------------
# pytest_collection_modifyitems: deselection reporting
# ---------------------------------------------------------------------------


def test_env_filtered_items_are_reported_deselected(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    unmarked = _item("t.py::test_c")
    config = _FakeConfig()
    items = _run(config, [match, wrong, unmarked])
    assert items == [match]
    assert config.deselected == [wrong, unmarked]


def test_no_env_filter_deselects_nothing():
    a = _item("t.py::test_a")
    b = _item("t.py::test_b")
    config = _FakeConfig()
    items = _run(config, [a, b])
    assert items == [a, b]
    assert config.deselected == []


def test_env_and_zmq_filters_both_report(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    grpc = _item("t.py::test_a[grpc]", engine=("sglang",), setup_backend="grpc")
    http = _item("t.py::test_a[http]", engine=("sglang",), setup_backend="http")
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    items = _run(config, [grpc, http, wrong])
    assert items == [grpc]
    assert set(config.deselected) == {http, wrong}


# ---------------------------------------------------------------------------
# Summary line
# ---------------------------------------------------------------------------


def test_summary_line_counts_and_prefix(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_GPU_TIER", "1")
    match = _item("t.py::test_a", engine=("sglang",), gpu=1)
    wrong_engine = _item("t.py::test_b", engine=("vllm",), gpu=1)
    unmarked = _item("t.py::test_c")
    wrong_tier = _item("t.py::test_d", engine=("sglang",), gpu=4)
    config = _FakeConfig()
    _run(config, [match, wrong_engine, unmarked, wrong_tier])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=sglang tier=1: selected 1 of 4 collected "
        "(2 deselected by engine, 1 by tier, 0 by zmq-dedup)"
    )


def test_summary_line_without_filters(monkeypatch):
    a = _item("t.py::test_a")
    config = _FakeConfig()
    _run(config, [a])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=- tier=-: selected 1 of 1 collected "
        "(0 deselected by engine, 0 by tier, 0 by zmq-dedup)"
    )


def test_summary_line_includes_vendor_and_zmq_counts(monkeypatch):
    monkeypatch.setenv("E2E_VENDOR", "openai")
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    kept = _item("t.py::test_a[grpc]", vendor=("openai",), setup_backend="grpc")
    twin = _item("t.py::test_a[http]", vendor=("openai",), setup_backend="http")
    other_vendor = _item("t.py::test_b", vendor=("xai",))
    config = _FakeConfig()
    _run(config, [kept, twin, other_vendor])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=- vendor=openai tier=-: selected 1 of 3 collected "
        "(0 deselected by engine, 1 by vendor, 0 by tier, 1 by zmq-dedup)"
    )


def test_summary_is_none_before_collection():
    config = _FakeConfig()
    assert _summary(config) is None


# ---------------------------------------------------------------------------
# E2E_MIN_SELECTED floor
# ---------------------------------------------------------------------------


def test_floor_unset_is_noop():
    hooks._enforce_selection_floor(0)  # must not raise


def test_floor_blank_is_noop(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "  ")
    hooks._enforce_selection_floor(0)  # must not raise


def test_floor_passes_at_and_above(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "2")
    hooks._enforce_selection_floor(2)
    hooks._enforce_selection_floor(3)


def test_floor_fails_below(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "5")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(1)
    msg = str(excinfo.value)
    assert "only 1 test selected" in msg  # singular for a one-test selection
    assert "floor of 5" in msg
    assert "E2E_ENGINE" in msg  # points at the env filter as the likely cause


def test_floor_message_pluralizes_above_one(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "5")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(2)
    assert "only 2 tests selected" in str(excinfo.value)


def test_floor_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "lots")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(100)
    assert "not an integer" in str(excinfo.value)


def test_floor_rejects_negative(monkeypatch):
    """A negative floor can never trip, so take it as a typo, not "no floor"."""
    monkeypatch.setenv("E2E_MIN_SELECTED", "-1")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(0)
    assert "must be non-negative" in str(excinfo.value)


def test_floor_zero_is_armed_but_unbreakable(monkeypatch):
    """Zero is a legal floor: explicitly armed, and no selection can fall under it."""
    monkeypatch.setenv("E2E_MIN_SELECTED", "0")
    hooks._enforce_selection_floor(0)


def test_floor_enforced_from_collection_hook(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_MIN_SELECTED", "2")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    with pytest.raises(pytest.exit.Exception):
        hooks.pytest_collection_modifyitems(config, [match, wrong])


def test_floor_satisfied_from_collection_hook(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_MIN_SELECTED", "1")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    items = _run(config, [match, wrong])
    assert items == [match]


def test_sglang_zmq_lane_keeps_multimodal_and_http_error_shapes_off_the_wire():
    mm = _FakeItem("e2e_test/chat_completions/test_multimodal.py::test_image[grpc]")
    shape = _FakeItem("e2e_test/chat_completions/test_error_shapes.py::test_json[http]")
    text = _FakeItem("e2e_test/chat_completions/test_basic.py::test_chat[grpc]")
    kept, deselected = hooks._filter_sglang_zmq_items([mm, shape, text])
    assert kept == [text] and deselected == [mm, shape]


def test_tokenspeed_rust_lane_keeps_prefill_decode_and_drops_encode_and_admin_ops():
    """The TokenSpeed Rust servicer serves the prefill/decode topologies and
    refuses the encode role: a Rust lane keeps the PD cases, drops the
    encode-prefill-decode ones, the admin operations the wire has no control
    message for, and a fleet of plain workers."""
    workers = lambda **kwargs: SimpleNamespace(kwargs=kwargs)  # noqa: E731
    pd = _FakeItem(
        "router/test_pd_topologies.py::T::t[1p1d]", params={"setup_backend": ("pd_grpc", (1, 1))}
    )
    pd_workers = _FakeItem(
        "router/test_pd_responses.py::t", markers={"workers": workers(prefill=1, decode=1)}
    )
    epd = _FakeItem(
        "chat_completions/test_epd_multimodal.py::T::t[1e1p1d]",
        params={"setup_backend": ("epd_grpc", (1, 1, 1))},
    )
    admin = _FakeItem("router/test_admin_ops.py::t", params={"setup_backend": "grpc"})
    fleet = _FakeItem("router/test_loads.py::t", markers={"workers": workers(count=2)})
    plain = _FakeItem("chat_completions/test_openai_server.py::t", params={"setup_backend": "grpc"})
    other = _FakeItem("router/test_other_wire.py::t", params={"setup_backend": "mesh_grpc"})
    kept, deselected = hooks._filter_tokenspeed_rust_items(
        [pd, pd_workers, epd, admin, fleet, plain, other]
    )
    assert [item.nodeid for item in kept] == [pd.nodeid, pd_workers.nodeid, plain.nodeid]
    assert [item.nodeid for item in deselected] == [
        epd.nodeid,
        admin.nodeid,
        fleet.nodeid,
        other.nodeid,
    ]


def test_sglang_rust_lane_drops_only_what_the_msgpack_wire_lacks():
    chat = _FakeItem(
        "e2e_test/chat_completions/test_basic.py::test_chat[grpc]", params={"setup_backend": "grpc"}
    )
    pd = _FakeItem(
        "e2e_test/chat_completions/test_pd.py::test_pd[pd_mooncake]",
        params={"setup_backend": "pd_mooncake"},
    )
    epd = _FakeItem(
        "e2e_test/chat_completions/test_epd.py::test_epd[1e1p1d]",
        params={"setup_backend": ("epd_grpc", (1, 1, 1))},
    )
    mm = _FakeItem(
        "e2e_test/chat_completions/test_multimodal.py::test_image[grpc]",
        params={"setup_backend": "grpc"},
    )
    admin = _FakeItem(
        "e2e_test/chat_completions/test_admin_ops.py::test_flush[grpc]",
        params={"setup_backend": "grpc"},
    )
    embed = _FakeItem(
        "e2e_test/embeddings/test_basic.py::test_embed[grpc]", params={"setup_backend": "grpc"}
    )
    kept, deselected = hooks._filter_sglang_rust_items([chat, pd, epd, mm, admin, embed])
    # Prefill/decode topologies, multimodal payloads, admin ops and embeddings
    # ride the wire (the bootstrap slots, the scheduler's processor output,
    # control calls, embedding requests); the EPD encode dispatch does not.
    assert kept == [chat, pd, mm, admin, embed] and deselected == [epd]


@pytest.mark.parametrize(
    "body, enabled, exit_code, gate_signal, xml_counts",
    [
        pytest.param(
            "import pytest\ndef test_case(): pytest.skip('optional')",
            False,
            0,
            False,
            (1, 0, 0),
            id="default-off",
        ),
        pytest.param(
            "import pytest\ndef test_case(): pytest.skip('optional')",
            True,
            1,
            True,
            (1, 0, 0),
            id="call-skip",
        ),
        pytest.param(
            "import pytest\n@pytest.fixture\ndef missing(): pytest.skip('unavailable')\n"
            "def test_case(missing): pass",
            True,
            1,
            True,
            (1, 0, 0),
            id="setup-skip",
        ),
        pytest.param("def test_case(): assert True", True, 0, False, (0, 0, 0), id="pass"),
        pytest.param(
            "import pytest\ndef test_a(): assert True\ndef test_b(): pytest.skip('optional')",
            True,
            0,
            False,
            (1, 0, 0),
            id="pass-and-skip",
        ),
        pytest.param("def test_case(): assert False", True, 1, False, (0, 1, 0), id="failure"),
        pytest.param(
            "import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError('setup')\n"
            "def test_case(broken): pass",
            True,
            1,
            False,
            (0, 0, 1),
            id="setup-error",
        ),
        pytest.param("value = 1", True, 5, False, (0, 0, 0), id="no-tests"),
        pytest.param(
            "def test_case(): raise KeyboardInterrupt()",
            True,
            2,
            False,
            (0, 0, 0),
            id="interrupted",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.xfail(reason='known', strict=True)\n"
            "def test_case(): assert True",
            True,
            1,
            False,
            (0, 1, 0),
            id="strict-xpass",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.xfail(reason='known')\ndef test_case(): assert False",
            True,
            1,
            True,
            (1, 0, 0),
            id="xfail",
        ),
        pytest.param(
            "import pytest\n@pytest.mark.xfail(reason='known')\ndef test_case(): assert True",
            True,
            0,
            False,
            (0, 0, 0),
            id="xpass",
        ),
    ],
)
def test_passing_call_real_sessions(
    pytester, monkeypatch, body, enabled, exit_code, gate_signal, xml_counts
):
    """Exercise the actual complete plugin in isolated pytest processes."""
    for key in list(os.environ):
        if key.startswith(("PYTEST_", "E2E_")):
            monkeypatch.delenv(key)
    hook = Path(__file__).with_name("hooks.py").resolve()
    monkeypatch.setenv("PYTHONPATH", str(hook.parent.parent))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if enabled:
        monkeypatch.setenv("E2E_AMD_REQUIRE_PASSING_CALL", "1")
    config = pytester.makeini("[pytest]\n")
    pytester.makeconftest(
        f"""
        from pathlib import Path
        pytest_plugins = ["fixtures.hooks"]
        def pytest_configure(config):
            from fixtures import hooks
            actual = Path(hooks.__file__).resolve()
            if actual != Path({str(hook)!r}):
                raise RuntimeError("wrong hooks module: " + str(actual))
            Path("loaded-hook.txt").write_text(str(actual))
        """
    )
    case = pytester.makepyfile(test_case=body)
    xml_path = pytester.path / "result.xml"
    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        str(config),
        "--confcutdir",
        str(pytester.path),
        "--junitxml",
        str(xml_path),
        str(case),
        timeout=10,
    )
    assert result.ret == exit_code
    assert ("AMD_E2E_NO_PASSING_CALL" in result.stdout.str() + result.stderr.str()) is gate_signal
    assert (pytester.path / "loaded-hook.txt").read_text() == str(hook)
    xml = ET.parse(xml_path).getroot()
    assert tuple(len(list(xml.iter(tag))) for tag in ("skipped", "failure", "error")) == xml_counts
