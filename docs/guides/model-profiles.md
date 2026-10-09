# Per-model Chat Completions profiles

Use an explicit profile when a model's public or backend serving name does not identify its protocol contract:

```bash
smg --worker-urls grpc://engine:50051 \
  --model-profile 'vllm-model=kimi_k3' \
  --model-alias 'ocid1.generativeaiendpoint.example=vllm-model' \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

The engine in this example registers as `vllm-model`. Clients may send that name or the registered OCI alias. SMG resolves the alias for contract selection before normalization can discard provider-specific message fields. The configured profile is carried internally through validation, dynamic-tool processing, and response handling. It is not a client parameter and is never serialized to the backend. Existing dispatch rules still translate self-hosted aliases to the engine's registered name; external-provider pinned model identifiers are preserved.

The Python launcher accepts the same repeatable flag. Programmatic `RouterArgs` and JSON/YAML `RouterConfig` use `model_profiles`, a mapping from canonical model IDs to profile values:

```yaml
model_profiles:
  vllm-model: kimi_k3
model_aliases:
  ocid1.generativeaiendpoint.example: vllm-model
```

| Value | Contract |
| --- | --- |
| `openai` | Core OpenAI-compatible rules, without vendor-specific additions |
| `kimi` | Shared Kimi/Moonshot rules; does not enable K3's pinned sampling/thinking rules |
| `kimi_k3` | Kimi rules plus K3 sampling defaults, constraints and thinking rules |
| `minimax` | Existing MiniMax contract |
| `zai` | Shared Zai/GLM contract |
| `zai_glm_5_3` | GLM 5.3 defaults and its always-enabled thinking contract |
| `deepseek_v4` | DeepSeek V4 rules and effort normalization |
| `deepseek_v4_1` | V4.1 rules, including native `xhigh` and integer reasoning budgets |

Profiles are keyed by the canonical registered model ID, not an alias. Matching is case-sensitive. An explicit value overrides name-based inference. If omitted, the gateway infers the contract from the registered canonical name. Repeated identical entries are accepted; conflicting entries, empty model IDs and unknown profiles are rejected.

These profiles apply to the existing **Chat Completions** contract. They do not change tokenizer selection, context length, model weights or parser configuration, and do not add provider-contract validation to other APIs. Keep the appropriate tool and reasoning parsers configured separately. Chat Completions tool-call IDs follow the resolved contract as well: `kimi_k3` uses `<name>_<ordinal>`, `kimi` uses `functions.<name>:<ordinal>`, and other profiles use OpenAI-style IDs. This also applies to inferred Moonshot names and opaque aliases. Requests subject to an explicit or inferred vendor contract use the validated buffered HTTP path, including large request bodies; OpenAI-baseline pass-through keeps its existing body-streaming behavior when no explicit profile is configured.
