"""GetLoads must report the scheduler's configured running window.

``GetLoadReqOutput`` carries no admission bound, so the servicer left
``max_running_requests`` at zero on every TokenSpeed load report. A frontend
could not tell how many requests the worker will run at once — and on a
disaggregated decode worker that window is exactly the bound its prefill
peer's bootstrap deadline is racing.

Run with: pytest grpc_servicer/tests/test_tokenspeed_loads.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")


@pytest.fixture(scope="module")
def loads_mod():
    """Load loads.py by path: the tokenspeed package __init__ imports the engine."""
    path = Path(__file__).resolve().parent.parent / "smg_grpc_servicer" / "tokenspeed" / "loads.py"
    spec = importlib.util.spec_from_file_location("tokenspeed_loads_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunningWindow:
    def test_max_num_seqs_is_the_window(self, loads_mod):
        assert loads_mod.running_window(SimpleNamespace(max_num_seqs=16)) == 16

    def test_max_running_requests_spelling_is_accepted(self, loads_mod):
        assert loads_mod.running_window(SimpleNamespace(max_running_requests=64)) == 64

    def test_absent_window_reports_zero_rather_than_guessing(self, loads_mod):
        assert loads_mod.running_window(SimpleNamespace()) == 0
        assert loads_mod.running_window(SimpleNamespace(max_num_seqs=None)) == 0


class TestLoadConversion:
    def test_window_and_derived_counters_reach_the_protobuf(self, loads_mod):
        load_output = SimpleNamespace(dp_rank=1, num_reqs=5, num_waiting_reqs=2, num_pages=8)

        load = loads_mod.convert_load_to_protobuf(
            load_output,
            page_size=16,
            max_total_num_tokens=1024,
            max_running_requests=16,
        )

        assert load.max_running_requests == 16
        assert load.dp_rank == 1
        # num_reqs counts running + waiting; used tokens come from pages.
        assert load.num_running_reqs == 3
        assert load.num_waiting_reqs == 2
        assert load.num_total_reqs == 5
        assert load.num_used_tokens == 128
        assert load.max_total_num_tokens == 1024
        assert load.token_usage == pytest.approx(0.125)

    def test_unknown_capacity_reports_zero_usage_instead_of_dividing(self, loads_mod):
        load_output = SimpleNamespace(dp_rank=0, num_reqs=1, num_waiting_reqs=0, num_pages=4)

        load = loads_mod.convert_load_to_protobuf(
            load_output,
            page_size=1,
            max_total_num_tokens=0,
            max_running_requests=0,
        )

        assert load.token_usage == 0.0
        assert load.max_running_requests == 0
