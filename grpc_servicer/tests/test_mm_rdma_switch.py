"""The RDMA pixel lane switch is read with the gateway's spelling rule.

Run with: pytest grpc_servicer/tests/test_mm_rdma_switch.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module directly: the package __init__ pulls engine dependencies.
# Registered in sys.modules first, as its dataclasses resolve their module.
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "mm_rdma.py"
_spec = importlib.util.spec_from_file_location("mm_rdma", _MODULE_PATH)
mm_rdma = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mm_rdma
_spec.loader.exec_module(mm_rdma)


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " On ", "Yes"])
def test_every_on_spelling_the_gateway_reads_enables_the_lane(value):
    assert mm_rdma.rdma_pixel_lane_enabled({"SMG_MM_PIXEL_RDMA": value})


@pytest.mark.parametrize("value", ["0", "false", "off", "maybe", "", "  "])
def test_any_other_spelling_leaves_the_lane_off(value):
    assert not mm_rdma.rdma_pixel_lane_enabled({"SMG_MM_PIXEL_RDMA": value})


def test_an_unset_switch_leaves_the_lane_off():
    assert not mm_rdma.rdma_pixel_lane_enabled({})
