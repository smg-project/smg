"""Unit tests for EPD command-builder logic (no GPU, no wheel required)."""

from __future__ import annotations

import pytest
from infra.constants import ConnectionMode, WorkerType
from infra.gateway import build_epd_mode_args
from infra.model_specs import get_model_spec
from infra.worker import Worker


def test_worker_type_encode_exists():
    assert WorkerType.ENCODE == "encode"
    assert WorkerType.ENCODE.value == "encode"


_TS_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # any tokenspeed-launchable spec at HEAD


def _ts_worker(worker_type, bootstrap_port=None):
    return Worker(
        model_id=_TS_MODEL,
        engine="tokenspeed",
        port=50104,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
        worker_type=worker_type,
        bootstrap_port=bootstrap_port,
        dist_init_addr="127.0.0.1:29500",
    )


@pytest.mark.parametrize(
    "worker_type,role",
    [(WorkerType.ENCODE, "encode"), (WorkerType.PREFILL, "prefill"), (WorkerType.DECODE, "decode")],
)
def test_tokenspeed_disagg_flags(worker_type, role):
    cmd = _ts_worker(worker_type, bootstrap_port=18995)._build_cmd()
    assert "--disaggregation-mode" in cmd
    assert cmd[cmd.index("--disaggregation-mode") + 1] == role
    assert "--disaggregation-transfer-backend" in cmd
    assert cmd[cmd.index("--disaggregation-transfer-backend") + 1] == "mooncake"
    assert "--dist-init-addr" in cmd
    assert cmd[cmd.index("--dist-init-addr") + 1] == "127.0.0.1:29500"
    assert "--skip-server-warmup" in cmd


def test_encode_and_prefill_carry_bootstrap_port():
    for wt in (WorkerType.ENCODE, WorkerType.PREFILL):
        cmd = _ts_worker(wt, bootstrap_port=18995)._build_cmd()
        assert "--disaggregation-bootstrap-port" in cmd
        assert cmd[cmd.index("--disaggregation-bootstrap-port") + 1] == "18995"


def test_prefill_is_eager_decode_and_prefill_cache():
    prefill = _ts_worker(WorkerType.PREFILL, bootstrap_port=1)._build_cmd()
    decode = _ts_worker(WorkerType.DECODE)._build_cmd()
    assert "--enforce-eager" in prefill
    assert "--enable-prefix-caching" in prefill
    assert "--enable-prefix-caching" in decode
    assert "--enforce-eager" not in decode


def test_regular_tokenspeed_worker_has_no_disagg_flags():
    cmd = _ts_worker(WorkerType.REGULAR)._build_cmd()
    assert "--disaggregation-mode" not in cmd


def _w(port, worker_type, bootstrap_port=None):
    return Worker(
        model_id=_TS_MODEL,
        engine="tokenspeed",
        port=port,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
        worker_type=worker_type,
        bootstrap_port=bootstrap_port,
    )


def test_build_epd_mode_args_basic():
    args = build_epd_mode_args(
        encode_workers=[_w(50104, WorkerType.ENCODE, 18995)],
        prefill_workers=[_w(50101, WorkerType.PREFILL, 19311)],
        decode_workers=[_w(50111, WorkerType.DECODE)],
    )
    assert args[0] == "--epd-disaggregation"
    assert "--encode" in args and "grpc://127.0.0.1:50104" in args
    assert "18995" in args  # encode bootstrap
    assert "--prefill" in args and "grpc://127.0.0.1:50101" in args
    assert "19311" in args  # prefill bootstrap
    assert "--decode" in args and "grpc://127.0.0.1:50111" in args
    assert args[args.index("--encode-policy") + 1] == "consistent_hashing"
    assert args[args.index("--multimodal-tensor-transport") + 1] == "inline"


def test_build_epd_mode_args_multi_encode_repeats_flag():
    args = build_epd_mode_args(
        encode_workers=[_w(50104, WorkerType.ENCODE, 1), _w(50105, WorkerType.ENCODE, 2)],
        prefill_workers=[_w(50101, WorkerType.PREFILL, 3)],
        decode_workers=[_w(50111, WorkerType.DECODE)],
    )
    assert args.count("--encode") == 2
    assert "grpc://127.0.0.1:50104" in args and "grpc://127.0.0.1:50105" in args


def test_qwen35_9b_spec_present_and_multimodal():
    spec = get_model_spec("Qwen/Qwen3.5-9B")
    assert spec["tp"] == 1
    assert "multimodal" in spec["features"]
    assert spec.get("skip_tier_download") is True
    ts_args = spec["tokenspeed_args"]
    assert "--attention-backend" in ts_args
    assert ts_args[ts_args.index("--attention-backend") + 1] == "fa3"
    # Hybrid-attention pool + paged-cache PD adjunct rejects layerwise
    # transfer; the spec must pin it off or PD/EPD workers die at startup.
    idx = ts_args.index("--disaggregation-layerwise-interval")
    assert ts_args[idx + 1] == "0"
