# isort: skip_file
# Preserve the reviewed SDK bootstrap import order.
"""Fixed original GPTOSS packed-slice probe; execute ONLY inside admitted GPU lease.

No model downloads/server/full-model load. This is not a resource controller.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL = "openai/gpt-oss-20b"
INPUT_SIZE = INTERMEDIATE = OUTPUT_SIZE = 64
ALPHA, CLAMP = 1.702, 7.0
REL_L2_LIMIT, SCALED_MAX_LIMIT = 0.02, 0.04
COUNTERFACTUAL_MIN_REL_L2 = 0.05
# Keep the alpha counterfactual distinguishable on the original biased slices.
STIMULUS_AMPLITUDE = 2.0
SUFFIXES = (
    "gate_up_proj_blocks",
    "gate_up_proj_scales",
    "gate_up_proj_bias",
    "down_proj_blocks",
    "down_proj_scales",
    "down_proj_bias",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_keys(weight_map):
    result = {}
    for suffix in SUFFIXES:
        matches = [key for key in weight_map if key.endswith(".layers.0.mlp.experts." + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one actual layer0 expert tensor: {suffix}")
        result[suffix] = matches[0]
    return result


def worker_checkpoint():
    # Evaluate only the actual harness resolver and its environment assignment.
    # Importing infra/model_specs as a package would initialize unrelated SDKs
    # in this -I entrypoint; these two original AST nodes need only stdlib os.
    source = Path(__file__).parents[1] / "e2e_test/infra/model_specs.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_resolve_model_path")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ROUTER_LOCAL_MODEL_PATH"
                for target in node.targets
            )
        )
    ]
    if len(nodes) != 2:
        raise ValueError("Actual worker model resolver contract changed")
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    selected = namespace["_resolve_model_path"](MODEL)
    ref = None
    checkpoint = Path(selected)
    if selected == MODEL:
        cache = Path(
            os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or "/models") / "hub"
        )
        repository = cache / "models--openai--gpt-oss-20b"
        reference = repository / "refs/main"
        # Diagnostic only: the actual worker opt-in selects the fixed revision,
        # not floating main, even when a later download has updated this ref.
        if reference.is_file() and reference.stat().st_size <= 128:
            ref = reference.read_text().strip()
        checkpoint = repository / "snapshots" / REVISION
    worker_path = str(checkpoint)
    return {
        "harness_model_path": selected,
        "worker_model_path": worker_path,
        "checkpoint_path": str(checkpoint),
        "resolver_source": str(source),
        "resolver_source_sha256": sha(source),
        "hub_main_ref": ref,
    }


def identity_validator(sdk, pins):
    # Reuse the exact pinned engine validator without importing GPU modules.
    source = sdk / "srt/layers/quantization/mxfp4.py"
    if sha(source) != pins["sdk_files"]["srt/layers/quantization/mxfp4.py"]:
        raise ValueError("Installed checkpoint validator source mismatch")
    names = {
        "_GPTOSS_REVISION",
        "_GPTOSS_METADATA_SHA256",
        "_original_shared_blob",
        "_validate_gptoss_local_checkpoint",
    }
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    if len(nodes) != 4:
        raise ValueError("Pinned engine checkpoint validator contract changed")
    namespace = {"Path": Path, "os": os, "hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_validate_gptoss_local_checkpoint"]


def _original_shared_blob(entry, repository, revision):
    """Recognize only snapshot -> original LFS entry -> same-Hub shared blob."""
    if repository is None:
        return False
    try:
        repository = Path(repository)
        snapshot = repository / "snapshots" / revision
        blobs = repository / "blobs"
        if (
            repository.name != "models--openai--gpt-oss-20b"
            or repository.resolve(strict=True) != repository
            or snapshot.resolve(strict=True) != snapshot
            or blobs.resolve(strict=True) != blobs
            or entry.parent != snapshot
            or entry.suffix != ".safetensors"
        ):
            return False
        first = Path(os.readlink(entry)).parts
        if (
            len(first) != 4
            or first[:3] != ("..", "..", "blobs")
            or len(first[3]) != 64
            or any(c not in "0123456789abcdef" for c in first[3])
        ):
            return False
        original = blobs / first[3]
        second = Path(os.readlink(original)).parts
        if (
            len(second) != 5
            or second[:3] != ("..", "..", "blobs")
            or len(second[4]) != 64
            or any(c not in "0123456789abcdef" for c in second[4])
            or second[3] != second[4][:2]
        ):
            return False
        shared = repository.parent / "blobs" / second[3] / second[4]
        return (
            shared.resolve(strict=True) == shared
            and shared.is_file()
            and entry.resolve(strict=True) == shared
        )
    except (OSError, ValueError):
        return False


def checkpoint_file(root, name, repository=None):
    p = Path(name)
    if p.name != name or p.suffix != ".safetensors":
        raise ValueError("Unsafe/non-safetensors checkpoint shard")
    entry = root / p
    actual = entry.resolve(strict=True)
    # A copy may read only its own shards or the same original HF blob store.
    if repository is None and root.parent.name == "snapshots" and root.name == REVISION:
        repository = root.parent.parent
    if not actual.is_file() or not (
        actual.is_relative_to(root)
        or (repository is not None and actual.is_relative_to(repository / "blobs"))
        or _original_shared_blob(entry, repository, REVISION)
    ):
        raise ValueError(
            f"Checkpoint shard escapes the selected checkpoint/original blobs: "
            f"{name}: {entry} -> {actual}"
        )
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required")
    args.output.mkdir(mode=0o700)
    record = {
        "status": "failed_or_incomplete",
        "model": MODEL,
        "revision": REVISION,
        "stimulus_amplitude": STIMULUS_AMPLITUDE,
        "contract": {
            "relative_l2": REL_L2_LIMIT,
            "scaled_max": SCALED_MAX_LIMIT,
            "counterfactual_min_relative_l2": COUNTERFACTUAL_MIN_REL_L2,
        },
        "source": {},
        "materials": {},
    }
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if os.environ.get(name) != "1":
                raise ValueError("Root must enforce offline cache")
        if (
            os.environ.get("SGLANG_USE_AITER") != "0"
            or os.environ.get("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK") != "1"
        ):
            raise ValueError("Exact per-model reviewed fallback environment required")
        package = importlib.util.find_spec("sglang")
        if package is None or package.origin is None:
            raise ValueError("Actual installed SDK required")
        sdk = Path(package.origin).parent
        pins = json.loads(
            Path(__file__).with_name("ci_sglang_mxfp4_numeric_inputs.json").read_text()
        )
        for relative, digest in pins["sdk_files"].items():
            actual = sdk / relative
            record["source"][relative] = {"path": str(actual), "sha256": sha(actual)}
            if record["source"][relative]["sha256"] != digest:
                raise ValueError(f"Installed SDK source mismatch: {relative}")
        selection = worker_checkpoint()
        identity = identity_validator(sdk, pins)(selection["checkpoint_path"])
        record["checkpoint"] = {**selection, **identity}
        root = Path(identity["resolved_path"])
        repository = Path(identity["repository"])
        import torch
        from safetensors import safe_open

        if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one Root-admitted visible HIP device required")
        prop = torch.cuda.get_device_properties(0)
        if prop.gcnArchName.split(":")[0] != "gfx942":
            raise ValueError("gfx942 required; never broaden visibility here")
        torch.cuda.set_device(0)
        record["device"] = {
            "visible_count": 1,
            "arch": prop.gcnArchName,
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
        }
        config_path = Path(identity["metadata"]["config.json"]["path"])
        cfg = json.loads(config_path.read_text())
        quant = cfg.get("quantization_config", {})
        if cfg.get("model_type") != "gpt_oss" or quant.get("quant_method") != "mxfp4":
            raise ValueError("Original GPTOSS MXFP4 metadata required")
        if (
            cfg.get("hidden_act", "silu") != "silu"
            or cfg.get("hidden_act_alpha", ALPHA) != ALPHA
            or cfg.get("swiglu_limit") != CLAMP
        ):
            raise ValueError("Original alpha/clamp/activation contract changed")
        index_path = Path(identity["metadata"]["model.safetensors.index.json"]["path"])
        if index_path.stat().st_size > 4 * 1024**2 or config_path.stat().st_size > 1024**2:
            raise ValueError("Unexpected checkpoint metadata size")
        index = json.loads(index_path.read_text())
        keys = checkpoint_keys(index["weight_map"])
        tensors = {}
        for suffix, key in keys.items():
            filename = index["weight_map"][key]
            path = checkpoint_file(root, filename, repository)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                view = handle.get_slice(key)
                shape = view.get_shape()
                if suffix == "gate_up_proj_blocks":
                    if (
                        len(shape) != 4
                        or shape[0] < 1
                        or shape[1] < 128
                        or shape[2] < 2
                        or shape[3] != 16
                    ):
                        raise ValueError("Unexpected original gate/up packed layout")
                    t = view[0:1, 0:128, 0:2, :]
                elif suffix == "down_proj_blocks":
                    if (
                        len(shape) != 4
                        or shape[0] < 1
                        or shape[1] < 64
                        or shape[2] < 2
                        or shape[3] != 16
                    ):
                        raise ValueError("Unexpected original down packed layout")
                    t = view[0:1, 0:64, 0:2, :]
                elif suffix.endswith("scales"):
                    rows = 128 if suffix.startswith("gate") else 64
                    if len(shape) != 3 or shape[0] < 1 or shape[1] < rows or shape[2] < 2:
                        raise ValueError("Unexpected original scale layout")
                    t = view[0:1, 0:rows, 0:2]
                else:
                    rows = 128 if suffix.startswith("gate") else 64
                    if len(shape) != 2 or shape[0] < 1 or shape[1] < rows:
                        raise ValueError("Unexpected original bias layout")
                    t = view[0:1, 0:rows]
            t = t.contiguous()
            if not torch.isfinite(t.float()).all():
                raise ValueError("Nonfinite material")
            tensors[suffix] = t
            raw = t.view(torch.uint8).numpy().tobytes()
            record["materials"][suffix] = {
                "key": key,
                "shard": filename,
                "resolved_path": str(path),
                "full_shape": shape,
                "slice_shape": list(t.shape),
                "dtype": str(t.dtype),
                "slice_sha256": hashlib.sha256(raw).hexdigest(),
                "slice_bytes": len(raw),
            }
        record["index_sha256"] = sha(index_path)
        record["config_sha256"] = sha(config_path)
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.runtime_context import publish, SpawnRanks
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        from sglang.srt.layers.moe.utils import initialize_moe_config

        server = ServerArgs(
            model_path=selection["worker_model_path"],
            device="cuda",
            dtype="bfloat16",
            tp_size=1,
            moe_runner_backend="triton",
            moe_a2a_backend="none",
            disable_cuda_graph=True,
            trust_remote_code=False,
        )
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        initialize_moe_config()
        group_started = False
        try:
            # Actual SDK public initialization, not fake get_parallel/get_exec globals.
            with tempfile.TemporaryDirectory(
                prefix="packed-numeric-", dir=args.output
            ) as rendezvous:
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method="file://" + str(Path(rendezvous) / "store"),
                    backend="nccl",
                    timeout=60,
                    moe_a2a_backend="none",
                )
                group_started = True
                initialize_model_parallel(backend="nccl", enable_symm_mem=False)
                execute(torch, tensors, record)
        finally:
            if group_started or torch.distributed.is_initialized():
                destroy_model_parallel()
                destroy_distributed_environment()
        record["status"] = "original_packed_slice_actual_generic_triton_numeric_pass"
    except Exception as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        path = args.output / "result.json"
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        path.chmod(0o600)


def execute(torch, material, record):
    from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerCore,
        TritonRunnerInput,
        TritonMoeQuantInfo,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    def decode(prefix):
        packed, scales = material[prefix + "_blocks"], material[prefix + "_scales"]
        if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise ValueError("Original uint8 packed bytes/scales required")
        actual = dequant_mxfp4(packed.cuda(), scales.cuda(), torch.bfloat16)
        # Independent scalar E2M1/UE8M0 reconstruction; never SDK decoder reused as oracle.
        values = []
        lut = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        for row, exponents in zip(packed[0].tolist(), scales[0].tolist()):
            out = []
            for block, exponent in zip(row, exponents):
                if exponent == 255:
                    raise ValueError("Invalid UE8M0 scale")
                for byte in block:
                    for nibble in (byte & 15, byte >> 4):
                        value = math.ldexp(lut[nibble & 7], exponent - 127)
                        out.append(-value if nibble & 8 else value)
            values.append(out)
        expected = torch.tensor([values], dtype=torch.float64).to(torch.bfloat16)
        if (
            not torch.isfinite(expected.float()).all()
            or not torch.equal(actual.cpu(), expected)
            or not torch.equal(torch.signbit(actual.cpu()), torch.signbit(expected))
        ):
            raise ValueError("Actual GPU decoder differs from original packed scalar oracle")
        return actual.contiguous(), expected[0].float()

    w13, ref13 = decode("gate_up_proj")
    w2, ref2 = decode("down_proj")
    b13 = material["gate_up_proj_bias"].to(torch.bfloat16)
    b2 = material["down_proj_bias"].to(torch.bfloat16)
    x = (
        torch.sin(torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64) * 0.173)
        * STIMULUS_AMPLITUDE
    ).to(torch.bfloat16)

    def oracle(counter=None, fp32=False):
        def rounded(t):
            return t if fp32 else t.to(torch.bfloat16).float()

        first = rounded(x.float() @ ref13.T + (0 if counter == "no_bias" else b13[0].float()))
        g, u = first[:, ::2], first[:, 1::2]
        if counter == "split_gate_up":
            g, u = first.chunk(2, dim=-1)
        g, u = g.clamp(max=CLAMP), u.clamp(-CLAMP, CLAMP)
        act = rounded(
            g
            * torch.sigmoid(g * (1.0 if counter == "alpha1" else ALPHA))
            * (u if counter == "no_up1" else u + 1)
        )
        return rounded(act @ ref2.T + (0 if counter == "no_bias" else b2[0].float()))

    expected = oracle()
    if not torch.isfinite(expected).all() or expected.norm().item() <= 1e-8:
        raise ValueError("Nonfinite/degenerate original material oracle")
    negatives = {}
    record["counterfactual_relative_l2"] = negatives
    for name in ("no_bias", "split_gate_up", "alpha1", "no_up1"):
        error = (oracle(name) - expected).norm().item() / expected.norm().item()
        negatives[name] = error
        if not math.isfinite(error) or error < COUNTERFACTUAL_MIN_REL_L2:
            raise ValueError(f"Inconclusive counterfactual sensitivity: {name}={error}")
    hidden = x.cuda().contiguous()
    weights = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    ids = torch.zeros((16, 1), device="cuda", dtype=torch.int32)
    prepared = _prepare_fused_moe_run(
        hidden,
        w13,
        w2,
        ids,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    config, down, down_tma, up_tma, sorted_ids, experts, post = prepared
    runner = TritonRunnerCore(
        MoeRunnerConfig(
            num_experts=1,
            num_local_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            gemm1_alpha=ALPHA,
            gemm1_clamp_limit=CLAMP,
            gate_up_interleaved=True,
            inplace=False,
        )
    )
    actual = runner.run(
        TritonRunnerInput(hidden, weights, ids, sorted_ids, experts, post),
        TritonMoeQuantInfo(w13, w2, b13=b13.cuda(), b2=b2.cuda()),
        {
            "config": config,
            "down_config": down,
            "down_moe_use_tma": down_tma,
            "up_moe_use_tma": up_tma,
        },
    ).hidden_states
    torch.cuda.synchronize()
    actual = actual.detach().cpu().float()
    if actual.shape != expected.shape:
        raise ValueError("Native output shape mismatch")
    error = actual - expected
    relative = error.norm().item() / expected.norm().item()
    scaled = error.abs().max().item() / expected.abs().max().item()
    record["numeric"] = {
        "actual_relative_l2": relative,
        "actual_scaled_max": scaled,
        "input_bf16_values": x.float().tolist(),
        "actual_output": actual.tolist(),
        "rounded_oracle_output": expected.tolist(),
        "fp32_oracle_output": oracle(fp32=True).tolist(),
        "counterfactual_relative_l2": negatives,
        "fp32_vs_rounded_relative_l2": (oracle(fp32=True) - expected).norm().item()
        / expected.norm().item(),
        "scope": (
            "one original expert bounded64x64 subnetwork; actual decoder+TritonRunnerCore; "
            "not full expert/model/graphs"
        ),
    }
    if not torch.isfinite(actual).all() or relative > REL_L2_LIMIT or scaled > SCALED_MAX_LIMIT:
        raise ValueError("Fixed predeclared numeric criterion failed")


if __name__ == "__main__":
    main()
