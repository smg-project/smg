"""The independent prefill_cost policy reaches the native router unchanged."""

import argparse
from unittest.mock import patch

import pytest
from smg.router import Router, policy_from_str
from smg.router_args import RouterArgs
from smg.smg_rs import PolicyType


@pytest.mark.parametrize("prefixed", [False, True])
def test_prefill_cost_python_cli_roundtrips_to_the_native_constructor(prefixed):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=prefixed)
    prefix = "--router-" if prefixed else "--"
    parsed = parser.parse_args(
        [
            prefix + "policy",
            "prefill_cost",
            prefix + "worker-urls",
            "grpc://worker:8080",
            prefix + "prefill-cost-default-throughput",
            "8000",
            prefix + "prefill-cost-kv-pressure-weight",
            "0.25",
            prefix + "prefill-cost-mean-prefill-tokens",
            "2048",
        ]
    )
    args = RouterArgs.from_cli_args(parsed, use_router_prefix=prefixed)
    assert args.policy == "prefill_cost"
    assert args.prefill_cost_default_throughput == 8000.0
    assert args.prefill_cost_kv_pressure_weight == 0.25
    assert args.prefill_cost_mean_prefill_tokens == 2048
    assert Router.from_args(args) is not None


def test_prefill_cost_is_a_distinct_policy_and_forwards_its_own_tuning():
    args = RouterArgs(
        worker_urls=["grpc://worker:8080"],
        policy="prefill_cost",
        prefill_cost_default_throughput=8000.0,
        prefill_cost_kv_pressure_weight=0.25,
        prefill_cost_mean_prefill_tokens=2048,
    )
    assert policy_from_str("prefill_cost") == PolicyType.PrefillCost
    with patch("smg.router._Router") as constructor:
        Router.from_args(args)
    kwargs = constructor.call_args.kwargs
    assert kwargs["policy"] == PolicyType.PrefillCost
    assert kwargs["prefill_cost_default_throughput"] == 8000.0
    assert kwargs["prefill_cost_kv_pressure_weight"] == 0.25
    assert kwargs["prefill_cost_mean_prefill_tokens"] == 2048
    # This constructor stores configuration only: it starts no runtime or workers.
    assert Router.from_args(args) is not None
    assert RouterArgs().policy == "cache_aware"


@pytest.mark.parametrize("url", ["http://worker:8080", "ipc:///tmp/prefill-cost-worker"])
def test_prefill_cost_rejects_transports_without_lease_handoff(url):
    with pytest.raises(ValueError, match="regular native gRPC"):
        Router.from_args(RouterArgs(worker_urls=[url], policy="prefill_cost"))


def test_prefill_cost_rejects_disaggregated_mode():
    args = RouterArgs(
        worker_urls=[],
        pd_disaggregation=True,
        prefill_urls=[("grpc://prefill:8080", None)],
        decode_urls=["grpc://decode:8080"],
        policy="prefill_cost",
    )
    with pytest.raises(ValueError, match="regular native gRPC"):
        Router.from_args(args)
