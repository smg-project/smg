# Ordinary SGLang provider data API

`ci_sglang_llama_provider_delivery.py` validates and copies provider JSON only.
It does not install packages, create a cohort, enable a workflow or qualify a runtime.

The CLI has three phases:

- `publish --input scripts/ci_data/sglang-llama-provider.json --checkout SHA --output PATH`
  checks the fixed checkout, ordinary schema/image, declared Source and reusable-workflow
  hashes, and exact tracked JSON before creating an exclusive read-only copy. It appends
  `provider_sha256` to `GITHUB_OUTPUT`.
- `bindings --input e2e-logs/ordinary-provider/provider.json --checkout SHA --sha256 HEX --artifact-id ID`
  validates the checkout, exact relative delivery path, digest/ID syntax and mutually
  exclusive ordinary/GPT-OSS selection. It does not assert an artifact API response.
- `read` takes the same bindings plus `--output REPORT`. It additionally checks the
  bounded regular single-link data, digest, Source contract and exact tracked JSON,
  then writes an exclusive report with `SDK_native_runtime_qualified: false`.

The producer and reader require the matching tracked provider packet, the ordinary
creator's `SOURCES` declaration and its declared files, and the reusable workflow.
These inputs are supplied by the later complete ordinary installation/caller component;
they are absent from this inactive API change. Missing or mismatched inputs refuse.
No current workflow calls this new helper. Default engine installation and NVIDIA
behavior are unchanged. A numeric artifact ID is a binding, not proof of same-run delivery;
the later caller must obtain it from its actual producer job and use the default same-run
artifact reader. This API itself makes no artifact network calls.

The 24 portable fixture checks cover real local Git identity, byte equality,
Source/workflow drift, bounded files, path and selection rejection, and exclusive outputs.
The separate actual-packet and complete-workflow integration checks remain with the full
caller activation component. They are not represented as passing on this inactive base.
