# Standalone tokenizer binding

This package exposes SMG's Rust tokenizer without building the gateway or GPU
servicers. It is intended for CPU preprocessing and training integrations.

```bash
python -m pip install maturin
maturin develop --release --manifest-path bindings/tokenizer-python/Cargo.toml
# Opt into the existing SMG accelerated v1 backend:
maturin develop --release --manifest-path bindings/tokenizer-python/Cargo.toml --features tokenizers-v1
```

Run from the repository root so rustup uses the repository toolchain. Use
`maturin build --release` for a wheel; compile separately for Linux/aarch64.

```python
from smg_tokenizer import Tokenizer

tokenizer = Tokenizer("/path/to/pinned/checkpoint")
print(tokenizer.encode_path)
result = tokenizer.render([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
])
ids = result["input_ids"]
offsets = result["offsets"]  # UTF-8 byte ranges, or None
```

`render` is a complete-conversation API: no generation prompt or continuation.
It forwards tools and explicit template kwargs and executes deferred native
encoding jobs rather than re-encoding flattened text. It does not itself define
training loss masks. Callers must use the model's own training profile, especially
for K3's serving defaults. Media processing is outside this binding.

Offsets are supplied only when they can be verified against the exact flat HF
encoding. For accelerated encoders this currently requires a reference encode;
the cost is part of `render`, not `encode`. Deferred native encodings return no
offsets, even if their flattened prompt looks identical. The Python API releases
the GIL around loading, rendering and encoding, and loads only local artifacts.
