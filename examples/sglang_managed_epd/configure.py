"""Write explicit managed-only configuration for a Qwen3-VL 1E/1P/1D deployment.

Run each generated worker configuration with its matching SMG managed launcher.
Engine weights and the native SGLang installation are not modified by this helper.
Native transport, dtype and TP settings come from the worker launch arguments.
"""

import argparse
import json
from pathlib import Path


def configurations(model, model_path, revision, encoder, prefill, decode):
    """Only endpoints and pinned model identity are deployment inputs; limits have bounded defaults."""
    workers = {
        role: {"model_revision": revision, "advertise_host": host}
        for role, host in (("e", encoder), ("p", prefill), ("d", decode))
    }
    workers["gateway"] = dict(
        model=model,
        model_path=model_path,
        model_revision=revision,
        encoders=[{"endpoint": f"http://{encoder}:31010"}],
        prefill_urls=[f"http://{prefill}:31001"],
        decode_urls=[f"http://{decode}:31003"],
    )
    return workers


def main():
    """Require explicit hosts and weight identity; never bake environment-specific addresses in examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-path", "revision", "encoder", "prefill", "decode", "output"):
        parser.add_argument("--" + name, required=True)
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
    )
    for role, config in configs.items():
        path = output / f"{role}-config.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
