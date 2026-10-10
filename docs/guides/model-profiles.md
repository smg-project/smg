# Automatic Chat Completions profiles

SMG selects a model's Chat Completions contract from discovered worker metadata before normalization and validation. Kimi K3 is identified by `KimiK3ForConditionalGeneration` or Hugging Face model type `kimi_k3`, even when the engine registers as `vllm-model` and clients use an OCI endpoint alias. No additional profile argument is required:

```bash
smg --worker-urls grpc://engine:50051 \
  --model-alias 'ocid1.generativeaiendpoint.example=vllm-model' \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

Detection uses the requested model card, including registered aliases, rather than another card on the same worker. Precise K3 metadata takes precedence over name inference. After precise metadata, detection checks the canonical model name, then a single-model worker's discovered model path, then the configured alias's provider family. Shared architectures such as `KimiLinearForCausalLM` do not imply K3. Replicas with inconsistent resolved contracts return HTTP 503 (`model_profile_conflict`) before dispatch.

Compatibility: configured vendor-named aliases retain the provider-family fallback. An alias alone does not establish version-specific K3, GLM or DeepSeek rules; deployments with generic serving names must advertise the real model's architecture, type or path. A client-provided alias cannot override a recognized worker identity.

The resolved contract remains internal through model-name rewrites and request clones. It controls validation, defaults, dynamic tools and response tool-call IDs: K3 uses `<name>_<ordinal>`, other Kimi models use `functions.<name>:<ordinal>`, and other profiles use OpenAI-style IDs. Client JSON cannot override the internal selection, and internal metadata is never serialized to the backend.

Vendor contracts use the validated buffered HTTP path, including large request bodies. Parser configuration remains separate. These rules apply to Chat Completions; they do not configure weights, tokenizers or context limits.
