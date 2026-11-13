"""Write explicit managed-only configuration for a Qwen3-VL 1E/1P/1D deployment.

Run each generated worker configuration with its matching SMG managed launcher.
Engine weights and the native SGLang installation are not modified by this helper.
All latency values below are conservative cold priors, not benchmark results.
"""

import argparse
import json
from pathlib import Path


def configurations(
    model, model_path, revision, encoder, prefill, decode, embedding_backend="mooncake"
):
    """Keep deployment identity, native transport and resource bounds consistent across roles."""
    processor = "smg-qwen3-vl-f32-v1"
    common = dict(
        max_records=1024,
        max_active=8,
        max_work=1000000,
        max_context_bytes=2 * 1024**3,
        retention_ms=60000,
        max_duration_ms=120000,
        max_clock_error_ms=1000,
    )
    contract = dict(
        model_revision=revision,
        processor_contract=processor,
        embedding_dtype="bfloat16",
        kv_backend="mooncake",
        embedding_backend=embedding_backend,
        topology_contract="single-host-tp1",
        bootstrap_host=prefill,
        bootstrap_port=43002,
        tp_size=1,
    )
    e = dict(
        common,
        **contract,
        worker_id="managed-e",
        control_listen="0.0.0.0:43010",
        receive_endpoint=f"http://{encoder}:43000",
        max_cpu=2,
        max_artifact_bytes=64 * 1024**2,
        max_staging_bytes_per_request=32 * 1024**2,
        max_receivers=2,
        max_control_bytes=4 * 1024**2,
        max_images=32,
        drr_quantum=4096,
        aging_seconds=2,
        tenant_weights={},
        max_early_waiters=32,
        early_wait_seconds=5,
        load_interval_ms=100,
        max_http_connections=64,
        image_token_id=151655,
        processor_options={"model_dir": model_path, "model_id": model, "max_inflight": 2},
    )
    p = dict(common, worker_id="managed-p", role="prefill", contract=contract)
    d = dict(common, worker_id="managed-d", role="decode", contract=contract)
    gateway = dict(
        model=model,
        model_path=model_path,
        model_revision=revision,
        processor_contract=processor,
        pairs=[
            dict(
                prefill=f"http://{prefill}:43001",
                decode=f"http://{decode}:43003",
                prior=dict(
                    layout_ms=100,
                    manifest_delivery_ms=100,
                    decode_registration_ms=100,
                    bind_submit_ms=100,
                    native_pd_ms=5000,
                ),
            )
        ],
        encoders=[
            dict(
                endpoint=f"http://{encoder}:43010",
                prior=dict(
                    cpu_wait_ms=0,
                    preprocess_ms=2000,
                    gpu_available_ms=0,
                    encode_ms=3000,
                    receive_ready_ms=100,
                    send_available_ms=0,
                    transfer_startup_ms=100,
                    embedding_bytes=32 * 1024**2,
                    bytes_per_ms=100000,
                    verified_cache_saving_ms=0,
                    error_ms=1000,
                ),
            )
        ],
        max_attempts=64,
        max_records=1024,
        timeout_ms=120000,
        max_clock_error_ms=1000,
        probe_timeout_ms=3000,
        snapshot_max_age_ms=3000,
        max_images=32,
        max_pd_candidates=4,
        max_encoder_candidates=4,
    )
    return {"e": e, "p": p, "d": d, "gateway": gateway}


def main():
    """Require explicit hosts and weight identity; never bake environment-specific addresses in examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-path", "revision", "encoder", "prefill", "decode", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--embedding-backend", choices=("mooncake", "zmq_to_scheduler"), default="mooncake"
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations(
        args.model,
        args.model_path,
        args.revision,
        args.encoder,
        args.prefill,
        args.decode,
        args.embedding_backend,
    )
    for role, config in configs.items():
        path = output / f"{role}-config.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
