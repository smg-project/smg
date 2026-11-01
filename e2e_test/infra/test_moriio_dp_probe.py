"""CPU contracts for the opt-in candidate's independent evidence checks."""

import pytest
from router.test_moriio_dp_hetero import actual_layout, metric_delta, parse_metrics

_METRICS = """vllm:prompt_tokens_total{engine="0",model_name="qwen"} 10
vllm:prompt_tokens_total{engine="1",model_name="qwen"} 20
vllm:prompt_tokens_by_source_total{engine="0",source="external_kv_transfer"} 10
vllm:prompt_tokens_by_source_total{engine="1",source="external_kv_transfer"} 20
vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 0
vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 0
vllm:prompt_tokens_by_source_total{engine="0",source="local_cache_hit"} 0
"""


def test_metric_sources_are_independent_of_engine_label_order():
    counters = parse_metrics(_METRICS)
    assert counters == {
        "engine=0": 10,
        "engine=1": 20,
        "source=external_kv_transfer": 30,
        "source=local_compute": 0,
        "source=local_cache_hit": 0,
    }
    assert metric_delta(dict.fromkeys(counters, 0), counters) == counters
    with pytest.raises(AssertionError):
        metric_delta(counters, dict.fromkeys(counters, 0))


@pytest.mark.parametrize(
    "bad",
    [
        _METRICS.replace('engine="1"', 'engine="2"'),
        _METRICS.replace(" 20\n", " NaN\n"),
        "\n".join(line for line in _METRICS.splitlines() if "local_cache_hit" not in line),
    ],
)
def test_missing_nonfinite_or_wrong_engine_metrics_fail(bad):
    with pytest.raises(AssertionError):
        parse_metrics(bad)


_LAYOUT = "\n".join(
    [
        "(Worker pid=10) INFO [parallel_state.py:100] world_size=2 rank=0 local_rank=0 "
        "distributed_init_method=tcp://127.0.0.1:20000 backend=nccl",
        "(Worker pid=11) INFO [parallel_state.py:100] world_size=2 rank=1 local_rank=1 "
        "distributed_init_method=tcp://127.0.0.1:20000 backend=nccl",
        "(Worker_DP0_EP0 pid=10) INFO [model_runner.py:100] Model loading took 1 GiB",
        "(Worker_DP1_EP1 pid=11) INFO [model_runner.py:100] Model loading took 1 GiB",
        "(Worker_DP0_EP0 pid=10) INFO [factory.py:100] "
        "Creating v1 connector with name: MoRIIOConnector and engine_id: candidate_dp0",
        "(Worker_DP1_EP1 pid=11) INFO [factory.py:100] "
        "Creating v1 connector with name: MoRIIOConnector and engine_id: candidate_dp1",
        "(Worker_DP0_EP0 pid=10) INFO [parallel_state.py:100] rank 0 in world size 2 "
        "is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, DCP rank 0, EP rank 0,",
    ]
)


def test_actual_worker_world_group_and_connector_proofs_agree():
    assert actual_layout(_LAYOUT, 1) == {
        "10": {"rank": 0, "dp_tp_ep": (0, 0, 0)},
        "11": {"rank": 1, "dp_tp_ep": (1, 0, 1)},
    }


@pytest.mark.parametrize(
    "bad",
    [
        _LAYOUT.replace("candidate_dp1", "candidate_dp0"),
        _LAYOUT.replace("Worker_DP1_EP1", "Worker_DP0_EP1"),
        _LAYOUT.replace("world_size=2 rank=1", "world_size=4 rank=1"),
        "\n".join(line for line in _LAYOUT.splitlines() if "Model loading" not in line),
        "\n".join(line for line in _LAYOUT.splitlines() if "assigned as" not in line),
    ],
)
def test_partial_or_conflicting_worker_identity_fails(bad):
    with pytest.raises(AssertionError):
        actual_layout(bad, 1)
