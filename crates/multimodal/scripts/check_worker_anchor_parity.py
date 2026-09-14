#!/usr/bin/env python3
"""Check that the router's worker-mode media anchors are the tokens vLLM expands.

In worker mode the router sends unexpanded token ids with one placeholder
anchor per media item and the vLLM worker runs its own multimodal processor
over them. That only works when the anchor the smg registry renders is exactly
the single token vLLM's prompt updates target. This script derives the anchor
the way the registry does, builds the prompt the way the router does (prefix
tokens + [anchor id] + suffix tokens), runs vLLM's processor over it with dummy
media, and checks that the anchor alone became one placeholder range while the
surrounding tokens survived.

Usage:
    python crates/multimodal/scripts/check_worker_anchor_parity.py
    python crates/multimodal/scripts/check_worker_anchor_parity.py --model qwen3_vl
    python crates/multimodal/scripts/check_worker_anchor_parity.py \
        --model llama4 --model-id meta-llama/Llama-4-Scout-17B-16E-Instruct

Needs vllm (>= 0.20) and access to the model's config, tokenizer and processor
files; weights are not loaded.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

# One entry per ModelProcessorSpec that opts into worker_expandable(); the
# anchor rule mirrors the registry's placeholder_token_for().
MODELS = {
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        # Anchor = tokenizer token for the config's <modality>_token_id.
        "anchors": {
            "image": ("config_id", "image_token_id"),
            "video": ("config_id", "video_token_id"),
        },
    },
    "llama4": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # Anchor = the literal <|image|> token (config image_token_index when present).
        "anchors": {"image": ("literal", "<|image|>")},
    },
}

PREFIX_TEXT = "Describe what you see."
SUFFIX_TEXT = " Answer briefly."


@dataclass
class Outcome:
    model: str
    modality: str
    anchor: str
    anchor_id: int
    expanded: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def config_int(config, name: str) -> int | None:
    """Read an int field from the HF config, looking one level into sub-configs."""
    value = getattr(config, name, None)
    if isinstance(value, int):
        return value
    for attr in dir(config):
        if attr.endswith("_config"):
            sub = getattr(config, attr, None)
            value = getattr(sub, name, None)
            if isinstance(value, int):
                return value
    return None


def router_anchor(config, tokenizer, rule: tuple[str, str]) -> tuple[str, int]:
    kind, key = rule
    if kind == "config_id":
        token_id = config_int(config, key)
        if token_id is None:
            raise ValueError(f"config has no {key}")
        return tokenizer.convert_ids_to_tokens(token_id), token_id
    token_id = config_int(config, "image_token_index")
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(key)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {key} token")
    return key, token_id


def dummy_media(processor, modality: str) -> list:
    data = processor.dummy_inputs.get_dummy_mm_data(seq_len=256, mm_counts={modality: 1})
    items = data[modality]
    return list(items) if isinstance(items, (list, tuple)) else [items]


def check(model_key: str, model_id: str, trust_remote_code: bool) -> list[Outcome]:
    from transformers import AutoConfig
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_config = ModelConfig(
        model=model_id, tokenizer=model_id, trust_remote_code=trust_remote_code
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    tokenizer = processor.info.get_tokenizer()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    prefix = tokenizer.encode(PREFIX_TEXT, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX_TEXT, add_special_tokens=False)
    outcomes = []
    for modality, rule in MODELS[model_key]["anchors"].items():
        try:
            anchor, anchor_id = router_anchor(config, tokenizer, rule)
        except ValueError as e:
            outcomes.append(Outcome(model_key, modality, "?", -1, None, str(e)))
            continue
        prompt_ids = prefix + [anchor_id] + suffix
        try:
            result = processor.apply(prompt_ids, {modality: dummy_media(processor, modality)}, {})
        except Exception as e:  # noqa: BLE001 - any processor failure is a parity failure
            outcomes.append(
                Outcome(
                    model_key, modality, anchor, anchor_id, None, f"vLLM rejected the anchor: {e}"
                )
            )
            continue
        out_ids = list(result["prompt_token_ids"])
        ranges = list(result["mm_placeholders"].get(modality, []))
        error = None
        if len(ranges) != 1:
            error = f"expected one placeholder range, got {len(ranges)}"
        elif ranges[0].offset != len(prefix):
            error = f"placeholder starts at {ranges[0].offset}, anchor was at {len(prefix)}"
        elif out_ids[: len(prefix)] != prefix:
            error = "prefix tokens changed"
        elif out_ids[ranges[0].offset + ranges[0].length :] != suffix:
            error = "suffix tokens changed or the expansion overran the anchor"
        elif anchor_id not in out_ids[ranges[0].offset : ranges[0].offset + ranges[0].length]:
            error = "the expansion does not contain the anchor token"
        outcomes.append(
            Outcome(
                model_key, modality, anchor, anchor_id, ranges[0].length if ranges else None, error
            )
        )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        action="append",
        help="registry spec to check (default: all)",
    )
    parser.add_argument("--model-id", help="HF model id or local path (single --model only)")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    keys = args.model or sorted(MODELS)
    if args.model_id and len(keys) != 1:
        parser.error("--model-id needs exactly one --model")

    outcomes: list[Outcome] = []
    for key in keys:
        model_id = args.model_id or MODELS[key]["model_id"]
        print(f"== {key}: {model_id}")
        outcomes.extend(check(key, model_id, args.trust_remote_code))

    print(f"\n{'spec':10} {'modality':8} {'anchor':16} {'id':>8} {'expanded':>8}  result")
    for o in outcomes:
        verdict = "ok" if o.ok else f"FAIL: {o.error}"
        print(
            f"{o.model:10} {o.modality:8} {o.anchor:16} {o.anchor_id:>8} "
            f"{str(o.expanded):>8}  {verdict}"
        )
    failed = [o for o in outcomes if not o.ok]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
