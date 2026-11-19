#!/usr/bin/env python3
r"""Generate independent Nemotron Transcribe fixtures by executing native vLLM.

Reference: vLLM a900ac810ecaedc0c2c5e9a8887e353cf1a5c350. Supply that checkout
and the public model config containing NeMo perception/pe_encoder_config.
Run from the crate directory:

    cd crates/multimodal
    python scripts/generate_nemotron_transcribe_fixtures.py \
        --vllm-source /path/to/vllm --model-config /path/to/config.json \
        --output tests/fixtures/nemotron_transcribe_native.json

Requires Git on PATH, Python 3.12, NumPy, torch, torchaudio and transformers.
Recorded versions:
NumPy 2.2.6, torch 2.13.0+cu130, torchaudio 2.11.0+cu130, transformers 5.14.1
(from nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0). No GPU or weights are needed.
Waveforms can differ with different numerical-library versions; provenance
records the versions and hashes of every executed native source file.

Original function bodies are AST-extracted and executed, rather than copying
preprocessing formulas. Processing classes lose only framework bases and unused
methods; context getters, list predicates and result containers are plumbing
stubs. The actual native config resolution, audio normalization, torchaudio
resampling, waveform padding and token counting execute unchanged. Six PCM16 WAV
inputs are written in the nemotron_transcribe/ directory beside the output JSON,
matching the Rust tests' fixture layout. The source checkout is executable
code and must be trusted. This covers CPU preprocessing, not model inference.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import subprocess
import types
import wave
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument(
        "--vllm-source", type=Path, required=True, help="Native vLLM git checkout root"
    )
    cli.add_argument(
        "--model-config", type=Path, required=True, help="Public NeMo model config.json"
    )
    cli.add_argument(
        "--output",
        type=Path,
        required=True,
        help=("Output fixture JSON; WAV files go in the adjacent nemotron_transcribe/ directory"),
    )
    return cli.parse_args(argv)


def extract_native(
    source: Path,
    relative: str,
    names: list[str] | dict[str, list[str]],
    namespace: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    """Execute the selected native bodies, removing only unused class plumbing."""
    path = source / relative
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.ClassDef) and isinstance(names, dict):
                node.bases = []
                node.body = [
                    member
                    for member in node.body
                    if isinstance(member, ast.FunctionDef) and member.name in names[node.name]
                ]
            nodes.append(node)
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f"Missing extraction in {path}: {names}")
    entry = selected.setdefault(
        relative,
        {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "symbols": []},
    )
    entry["symbols"].extend(names)
    future_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future_annotations, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)


def load_native_reference(vllm_source: Path, model_config: Path) -> types.SimpleNamespace:
    # Keep heavy dependencies local so importing this module and --help need only
    # the standard library. Native functions receive their original dependencies.
    import numpy as np
    import numpy.typing as npt
    import torch
    import torchaudio
    import transformers
    from transformers import BatchFeature

    # These imports are globals read by the extracted native function bodies.
    namespace = dict(
        __name__=__name__,
        contextlib=contextlib,
        math=math,
        os=os,
        Iterable=Iterable,
        Mapping=Mapping,
        dataclass=dataclass,
        Enum=Enum,
        lru_cache=lru_cache,
        Any=Any,
        Literal=Literal,
        cast=cast,
        np=np,
        npt=npt,
        torch=torch,
        torchaudio=torchaudio,
        transformers=transformers,
        BatchFeature=BatchFeature,
    )
    source = vllm_source / "vllm"
    selected: dict[str, Any] = {}

    def extract(relative: str, names: list[str] | dict[str, list[str]]) -> None:
        extract_native(source, relative, names, namespace, selected)

    extract(
        "transformers_utils/configs/nemotron3_5_transcribe.py",
        [
            "_encoder_config",
            "_preprocessor_config",
            "_checked_online",
            "resolve_audio_config",
        ],
    )
    namespace["_SUPPORTED_SPEAKER_FEATURE_MODES"] = ("thresholded", "continuous")
    raw = json.loads(model_config.read_text())
    config = namespace["resolve_audio_config"](raw["perception"], raw["pe_encoder_config"])
    extract("utils/torch_utils.py", ["set_default_torch_num_threads"])
    extract(
        "multimodal/audio.py",
        [
            "ChannelReduction",
            "AudioSpec",
            "normalize_audio",
            "_get_torchaudio_resampler",
            "resample_audio_torchaudio",
            "AudioResampler",
        ],
    )
    # Output container and list predicate are plumbing unrelated to DSP.
    namespace["is_list_of"] = lambda value, kind: (
        isinstance(value, list) and all(isinstance(item, kind) for item in value)
    )
    namespace["AudioProcessorItems"] = lambda value: value
    extract(
        "multimodal/parse.py",
        {
            "MultiModalDataParser": [
                "is_embeddings",
                "_get_audio_with_sr",
                "_parse_audio_data",
            ]
        },
    )
    extract("model_executor/models/nemotron3_5_transcribe.py", ["_num_audio_embeddings"])
    extract(
        "model_executor/models/nemotron3_5_transcribe.py",
        {
            "Nemotron3_5TranscribeProcessingInfo": [
                "get_min_num_samples",
                "get_num_audio_tokens",
            ],
            "Nemotron3_5TranscribeMultiModalProcessor": ["_call_hf_processor"],
        },
    )
    info = namespace["Nemotron3_5TranscribeProcessingInfo"]()
    info.get_hf_config = lambda: types.SimpleNamespace(audio_config=config)
    processor = namespace["Nemotron3_5TranscribeMultiModalProcessor"]()
    processor.info = info
    parser = namespace["MultiModalDataParser"]()
    parser.audio_resampler = namespace["AudioResampler"](
        target_sr=config["preprocessor"]["sample_rate"]
    )
    parser.target_channels = 1
    parser.expected_hidden_size = None
    parser.embedding_fields = {}
    return types.SimpleNamespace(
        np=np,
        config=config,
        selected=selected,
        info=info,
        processor=processor,
        parser=parser,
        versions={
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
        },
    )


def make_case(
    reference: types.SimpleNamespace,
    name: str,
    data: Any,
    rate: int = 16000,
    constant: bool = False,
) -> dict[str, Any]:
    np = reference.np
    data = np.asarray(data, dtype=np.float32)
    normalized = reference.parser._parse_audio_data((data, rate))
    result = reference.processor._call_hf_processor({"audio": normalized}, {})
    signal = result["audio_signal"][0]
    length = int(result["audio_signal_length"][0])
    case = {
        "name": name,
        "sample_rate": rate,
        "expected_samples": signal.tolist(),
        "expected_length": length,
        "expected_tokens": reference.info.get_num_audio_tokens(length),
        "expected_dtype": str(signal.dtype),
        "length_dtype": str(result["audio_signal_length"].dtype),
        "length_device": str(result["audio_signal_length"].device),
    }
    if constant:
        case.update(input_num_samples=len(data), constant_sample=float(data[0]))
        case.pop("expected_samples")
        case["expected_constant_samples"] = float(data[0])
        case["expected_zero_padding"] = max(0, length - len(data))
    else:
        case["input_samples"] = data.tolist()
    return case


def generate_cases(reference: types.SimpleNamespace, wav_dir: Path) -> list[dict[str, Any]]:
    np = reference.np
    cases = []
    lengths = [1, 159, 160, 161, 1279, 1280, 1281, 1439, 1440, 2559, 2560, 2561, 16000]
    for length in lengths:
        cases.append(
            make_case(
                reference,
                f"mono16k_constant_{length}",
                np.full(length, 0.25, dtype=np.float32),
                constant=True,
            )
        )
    cases.append(
        make_case(
            reference,
            "mono16k_signed_pcm",
            [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 32767 / 32768],
        )
    )
    cases.append(
        make_case(
            reference,
            "stereo16k_channels_time",
            [[-0.5, 0.25, 0.75, 0], [0.5, 0.75, -0.25, -1]],
        )
    )
    cases.append(
        make_case(
            reference,
            "stereo16k_time_channels",
            [[-0.5, 0.5], [0.25, 0.75], [0.75, -0.25], [0, -1]],
        )
    )
    for rate, length in [(8000, 97), (44100, 317), (48000, 481), (22050, 223)]:
        signal = np.sin(np.arange(length, dtype=np.float32) * np.float32(0.137)) * np.float32(0.5)
        cases.append(make_case(reference, f"mono{rate}_sine_{length}", signal, rate))
        cases.append(
            make_case(
                reference,
                f"stereo{rate}_sine_{length}",
                np.stack([signal, -signal * 0.5]),
                rate,
            )
        )
    # Quantize PCM16 WAVs first, so the oracle gets exactly the decoded samples.
    wav_inputs = [
        (16000, 1, 159),
        (16000, 1, 1440),
        (16000, 2, 161),
        (8000, 1, 97),
        (44100, 2, 317),
        (48000, 1, 481),
    ]
    for rate, channels, length in wav_inputs:
        base = (np.sin(np.arange(length, dtype=np.float32) * np.float32(0.137)) * 12000).astype(
            "<i2"
        )
        pcm = (
            base[:, None] if channels == 1 else np.stack([base, (-base // 2).astype("<i2")], axis=1)
        )
        name = f"pcm16wav_{rate}_{channels}ch_{length}"
        filename = name + ".wav"
        with wave.open(str(wav_dir / filename), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm.tobytes())
        decoded = (
            pcm[:, 0].astype(np.float32) / 32768
            if channels == 1
            else pcm.T.astype(np.float32) / 32768
        )
        case = make_case(reference, name, decoded, rate)
        case["input_wav"] = filename
        cases.append(case)
    return cases


def generate_batches(
    reference: types.SimpleNamespace, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # Existing complete cases provide the batch inputs, including stereo layout.
    # Distinct nonconstant clips pin order, minimum padding and variable lengths.
    case_names = [
        "pcm16wav_16000_1ch_159",
        "pcm16wav_16000_1ch_1440",
        "pcm16wav_16000_2ch_161",
    ]
    cases_by_name = {case["name"]: case for case in cases}
    inputs = [
        (
            reference.np.asarray(cases_by_name[name]["input_samples"], dtype=reference.np.float32),
            cases_by_name[name]["sample_rate"],
        )
        for name in case_names
    ]
    normalized = reference.parser._parse_audio_data(inputs)
    result = reference.processor._call_hf_processor({"audio": normalized}, {})
    signals = result["audio_signal"]
    lengths = result["audio_signal_length"]
    return [
        {
            "name": "three_variable_clips",
            "case_names": case_names,
            "input_lengths": [samples.shape[-1] for samples, _ in inputs],
            "expected_lengths": lengths.tolist(),
            "expected_shapes": [list(signal.shape) for signal in signals],
            "expected_tokens": [reference.info.get_num_audio_tokens(int(n)) for n in lengths],
            "expected_samples": [signal.tolist() for signal in signals],
            "expected_dtype": str(signals[0].dtype),
            "length_dtype": str(lengths.dtype),
            "length_device": str(lengths.device),
        }
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    native_commit = subprocess.check_output(
        ["git", "-C", str(args.vllm_source), "rev-parse", "HEAD"], text=True
    ).strip()
    reference = load_native_reference(args.vllm_source, args.model_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wav_dir = args.output.parent / "nemotron_transcribe"
    wav_dir.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(reference, wav_dir)
    batches = generate_batches(reference, cases)
    output = {
        "provenance": {
            "native_vllm_commit": native_commit,
            "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
            "executed_native_symbols": reference.selected,
            **reference.versions,
            "method": (
                "AST-extracted original function/class definitions compiled unchanged; "
                "class framework bases removed, HF config getter and parser output "
                "plumbing stubbed; no mathematical formulas or DSP reimplemented"
            ),
        },
        "config": reference.config,
        "cases": cases,
        "batches": batches,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "batches": [
                    {key: value for key, value in batch.items() if key != "expected_samples"}
                    for batch in batches
                ],
                "resolved_preprocessor": reference.config["preprocessor"],
                "asr_subsampling_factor": reference.config["asr_encoder"]["subsampling_factor"],
                "case_summary": [
                    {
                        key: value
                        for key, value in case.items()
                        if key in ["name", "expected_length", "expected_tokens"]
                    }
                    for case in cases
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
