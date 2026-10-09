"""Explicit contracts must survive both launcher spellings and reject bad configuration."""

import argparse

import pytest
from smg.router_args import RouterArgs


def parse(values, prefixed=False):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    RouterArgs.add_cli_args(parser, use_router_prefix=prefixed)
    return RouterArgs.from_cli_args(parser.parse_args(values), use_router_prefix=prefixed)


@pytest.mark.parametrize("prefixed", [False, True])
def test_model_profiles_accumulate_and_aliases_stay_separate(prefixed):
    flag = "--router-" if prefixed else "--"
    args = parse(
        [
            flag + "model-profile",
            "vllm-model=kimi_k3",
            flag + "model-profile",
            "another=deepseek_v4_1",
            flag + "model-alias",
            "ocid1.test=vllm-model",
        ],
        prefixed,
    )
    assert args.model_profiles == {"vllm-model": "kimi_k3", "another": "deepseek_v4_1"}
    assert args.model_aliases == {"ocid1.test": "vllm-model"}


@pytest.mark.parametrize("entry", ["no-equals", "=kimi_k3", "model=", "model=unsupported"])
def test_invalid_profile_configuration_is_rejected(entry):
    with pytest.raises((ValueError, SystemExit)):
        parse(["--model-profile", entry])


def test_conflicting_profiles_are_rejected():
    with pytest.raises(ValueError):
        parse(["--model-profile", "model=kimi", "--model-profile", "model=kimi_k3"])


def test_repeating_identical_profiles_is_idempotent():
    args = parse(["--model-profile", "model=kimi_k3", "--model-profile", "model=kimi_k3"])
    assert args.model_profiles == {"model": "kimi_k3"}
