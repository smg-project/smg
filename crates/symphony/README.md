# smg-symphony

One parser for everything a model says. A model's output stream interleaves content, reasoning and
tool calls in a vendor-specific format; this crate turns it into typed events through one object
with one method, `Parser::feed`, whose lifecycle is expressed as input: the prompt the engine was
given, each decoded delta, the end of the stream. Every byte of output lands in exactly one event,
including bytes that were dropped or could not be parsed.

Status: the public types are the contract, the adapters render them for the Chat Completions,
Responses and Messages APIs, and the engine runs a format table; the tables under `formats` are
the families recorded so far, and the models table below says which, and how far each is.

## Models

Symphony's target is every checkpoint group bellwether has a manifest for; a group is the
checkpoints that share a tokenizer and a template, recorded once by its primary. The table below
is written from `tests/readme.rs` and checked by `cargo test`, so it says what this branch holds,
and a stale table fails the build. It runs newest first: **Released** is the day the primary's
Hugging Face repository was created.

A group is **ready** when its table is on main, bellwether's `main` holds the group's
benchmark-scale set (the count in **bellwether set**), and a run of the fixture test
(`tests/bellwether_parse_fixtures.rs`) on main's code has replayed every case of that set at every
chunking with no difference. That run is local, and the pull request that marked the group ready
records it: SMG's CI runs the fixture test on every change to the crate against the bellwether
commit pinned in `.github/versions/bellwether.ref`, which holds the hand-written sets only, so CI
guards a ready group's set once the pin moves to a commit that holds it. **Replaying the set**
means the table is on main and bellwether's `main` holds the set, and that run has not finished.
**Awaiting fixtures** means the table is on main and bellwether has not recorded the group's set
yet; every such table was previewed against bellwether's scale run of 2026-10-06 with no
difference. **Pending** means no table yet.

**SMG today** is what the gateway takes for the group's primary before Symphony is wired in: the
`--tool-call-parser` and `--reasoning-parser` names its registries resolve from the model id, as
in `smg serve --model-path Qwen/Qwen3-8B --tool-call-parser qwen --reasoning-parser qwen3`; none
means passthrough, the output read as content. Symphony's tables carry one name each, the
**Table** column, and the Qwen3 table takes its call syntax as a second selector (JSON calls, or
the tagged calls of Qwen 3.5 and later and Qwen3-Coder); once the gateway selects a table, the
name and that selector are what to pass, and whether they replace the two flags or keep them as
aliases is an open question of the design.

<!-- models: begin -->
78 groups, 107 checkpoints, newest first.
25 ready, 27 replaying a recorded set, 0 on main awaiting fixtures, 26 pending.

| Released | Group | Model | Also | Table | bellwether set | Status | SMG today |
|---|---|---|---|---|---:|---|---|
| 2026-09-28 | [iquest-q1](https://github.com/smg-project/bellwether/tree/main/fixtures/iquest-q1) | IQuestLab/IQuest-Q1 |  | `iquest` | 54,193 | replaying the set | none |
| 2026-09-10 | [deepseek-v4.1-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v4.1-flash) | deepseek-ai/DeepSeek-V4.1-Flash |  | `deepseek_v4.1` | 54,418 | ready | `deepseek_v41`, `deepseek_v41` |
| 2026-09-06 | [minicpm5-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/minicpm5-2b) | openbmb/MiniCPM5-2B |  | — | — | pending | none |
| 2026-09-01 | [k2-horizon-36b](https://github.com/smg-project/bellwether/tree/main/fixtures/k2-horizon-36b) | IFM/K2-Horizon-36B |  | — | — | pending | none |
| 2026-08-27 | [hy4-preview](https://github.com/smg-project/bellwether/tree/main/fixtures/hy4-preview) | tencent/Hy4-preview |  | `hy4` | 54,195 | ready | `hy_v4`, `hy_v4` |
| 2026-08-27 | [qwen-drive-1.0-4b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen-drive-1.0-4b) | Qwen/Qwen-Drive-1.0-4B |  | `qwen3`, tagged calls | 54,179 | ready | `qwen`, `qwen3` |
| 2026-08-25 | [glm-5.3-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-5.3-flash) | zai-org/GLM-5.3-Flash |  | `glm` | 54,193 | ready | `glm47_moe`, `glm45` |
| 2026-08-24 | [qwen3.8-flash-next](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-flash-next) | Qwen/Qwen3.8-Flash-Next |  | `qwen3`, tagged calls | 54,193 | ready | `qwen_xml`, `qwen3` |
| 2026-08-09 | [dots3-note-prev](https://github.com/smg-project/bellwether/tree/main/fixtures/dots3-note-prev) | dots-studio/dots3-note-prev |  | — | — | pending | none |
| 2026-08-09 | [muse-glimmer-30b](https://github.com/smg-project/bellwether/tree/main/fixtures/muse-glimmer-30b) | meta-models/Muse-Glimmer-30B |  | — | — | pending | none |
| 2026-08-08 | [qwen3.8-2.4t-a95b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-2.4t-a95b) | Qwen/Qwen3.8-2.4T-A95B |  | `qwen3`, tagged calls | 54,183 | ready | `qwen_xml`, `qwen3` |
| 2026-08-05 | [qwen3.8-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.8-27b) | Qwen/Qwen3.8-27B |  | `qwen3`, tagged calls | 54,193 | ready | `qwen_xml`, `qwen3` |
| 2026-08-02 | [ling-3.0-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/ling-3.0-flash) | inclusionAI/Ling-3.0-flash |  | `ling` | 54,195 | replaying the set | none |
| 2026-07-21 | [nanbeige4.2-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/nanbeige4.2-3b) | Nanbeige/Nanbeige4.2-3B |  | `qwen3`, tagged calls | 54,195 | replaying the set | none |
| 2026-07-14 | [inkling](https://github.com/smg-project/bellwether/tree/main/fixtures/inkling) | thinkingmachines/Inkling |  | — | — | pending | `inkling`, `inkling` |
| 2026-06-22 | [qwen-agentworld-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen-agentworld-35b-a3b) | Qwen/Qwen-AgentWorld-35B-A3B |  | `qwen3`, tagged calls | 54,193 | ready | `qwen`, `qwen3` |
| 2026-06-13 | [kimi-k3](https://github.com/smg-project/bellwether/tree/main/fixtures/kimi-k3) | moonshotai/Kimi-K3 |  | `kimi_k3` | 54,418 | ready | `kimi_k3`, `kimi_k3` |
| 2026-06-02 | [minimax-m3](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m3) | MiniMaxAI/MiniMax-M3 |  | `minimax_m3` | 54,180 | replaying the set | `minimax_m3`, `minimax_m3` |
| 2026-04-27 | [mimo-v2.5](https://github.com/smg-project/bellwether/tree/main/fixtures/mimo-v2.5) | XiaomiMiMo/MiMo-V2.5 |  | `qwen3`, tagged calls | 54,181 | replaying the set | none |
| 2026-04-23 | [laguna-xs.2](https://github.com/smg-project/bellwether/tree/main/fixtures/laguna-xs.2) | poolside/Laguna-XS.2 |  | — | — | pending | none |
| 2026-04-21 | [qwen3.6-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.6-27b) | Qwen/Qwen3.6-27B |  | `qwen3`, tagged calls | 54,193 | ready | `qwen_xml`, `qwen3` |
| 2026-04-15 | [qwen3.6-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.6-35b-a3b) | Qwen/Qwen3.6-35B-A3B |  | `qwen3`, tagged calls | 54,193 | ready | `qwen_xml`, `qwen3` |
| 2026-04-09 | [minimax-m2.7](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m2.7) | MiniMaxAI/MiniMax-M2.7 |  | — | — | pending | `minimax_m2`, `minimax` |
| 2026-04-06 | [granite-4.1-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/granite-4.1-3b) | ibm-granite/granite-4.1-3b |  | `qwen2.5` | 41,533 | replaying the set | none |
| 2026-03-02 | [gemma-4-e4b-it](https://github.com/smg-project/bellwether/tree/main/fixtures/gemma-4-e4b-it) | google/gemma-4-E4B-it |  | — | — | pending | `json`, none |
| 2026-02-28 | [qwen3.5-2b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-2b) | Qwen/Qwen3.5-2B | Qwen3.5-0.8B | `qwen3`, tagged calls | 41,360 | ready | `qwen_xml`, `qwen3` |
| 2026-02-27 | [qwen3.5-9b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-9b) | Qwen/Qwen3.5-9B | Qwen3.5-4B | `qwen3`, tagged calls | 54,179 | ready | `qwen_xml`, `qwen3` |
| 2026-02-24 | [qwen3.5-27b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-27b) | Qwen/Qwen3.5-27B |  | `qwen3`, tagged calls | 54,179 | ready | `qwen_xml`, `qwen3` |
| 2026-02-24 | [qwen3.5-35b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3.5-35b-a3b) | Qwen/Qwen3.5-35B-A3B | Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | `qwen3`, tagged calls | 54,179 | ready | `qwen_xml`, `qwen3` |
| 2026-02-13 | [webworld-32b](https://github.com/smg-project/bellwether/tree/main/fixtures/webworld-32b) | Qwen/WebWorld-32B | WebWorld-8B, WebWorld-14B | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2026-02-01 | [step-3.5-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/step-3.5-flash) | stepfun-ai/Step-3.5-Flash |  | `qwen3`, tagged calls | 54,181 | replaying the set | `step3`, none |
| 2026-01-30 | [qwen3-coder-next](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-coder-next) | Qwen/Qwen3-Coder-Next |  | `qwen3`, tagged calls | 41,374 | replaying the set | `qwen_xml`, `qwen3` |
| 2026-01-19 | [glm-4.7-flash](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.7-flash) | zai-org/GLM-4.7-Flash |  | — | — | pending | `glm47_moe`, none |
| 2026-01-06 | [ai21-jamba2-3b](https://github.com/smg-project/bellwether/tree/main/fixtures/ai21-jamba2-3b) | ai21labs/AI21-Jamba2-3B |  | `qwen2.5` | 41,533 | replaying the set | none |
| 2026-01-06 | [lfm2.5-1.2b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/lfm2.5-1.2b-instruct) | LiquidAI/LFM2.5-1.2B-Instruct |  | `lfm2.5` | 54,418 | replaying the set | none |
| 2025-12-26 | [k-exaone-236b-a23b](https://github.com/smg-project/bellwether/tree/main/fixtures/k-exaone-236b-a23b) | LGAI-EXAONE/K-EXAONE-236B-A23B |  | `qwen3` | 54,416 | replaying the set | none |
| 2025-12-04 | [nvidia-nemotron-3-nano-30b-a3b-bf16](https://github.com/smg-project/bellwether/tree/main/fixtures/nvidia-nemotron-3-nano-30b-a3b-bf16) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |  | `qwen3`, tagged calls | 12,829 | replaying the set | `qwen_xml`, `nano_v3` |
| 2025-12-01 | [trinity-mini](https://github.com/smg-project/bellwether/tree/main/fixtures/trinity-mini) | arcee-ai/Trinity-Mini |  | — | — | pending | none |
| 2025-11-19 | [olmo-3-7b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/olmo-3-7b-instruct) | allenai/Olmo-3-7B-Instruct |  | `olmo3` | 41,533 | replaying the set | none |
| 2025-10-22 | [minimax-m2](https://github.com/smg-project/bellwether/tree/main/fixtures/minimax-m2) | MiniMaxAI/MiniMax-M2 |  | — | — | pending | `minimax_m2`, `minimax` |
| 2025-10-11 | [qwen3-vl-8b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-8b-instruct) | Qwen/Qwen3-VL-8B-Instruct | Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-32B-Instruct | `qwen3` | 41,533 | ready | `qwen`, `qwen3` |
| 2025-10-11 | [qwen3-vl-8b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-8b-thinking) | Qwen/Qwen3-VL-8B-Thinking | Qwen3-VL-2B-Thinking, Qwen3-VL-4B-Thinking, Qwen3-VL-32B-Thinking | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-30 | [qwen3-4b-saferl](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-saferl) | Qwen/Qwen3-4B-SafeRL |  | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-30 | [qwen3-vl-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-30b-a3b-instruct) | Qwen/Qwen3-VL-30B-A3B-Instruct | Qwen3-VL-235B-A22B-Instruct | `qwen3` | 41,533 | ready | `qwen`, `qwen3` |
| 2025-09-29 | [glm-4.6](https://github.com/smg-project/bellwether/tree/main/fixtures/glm-4.6) | zai-org/GLM-4.6 |  | — | — | pending | `glm45_moe`, none |
| 2025-09-23 | [qwen3guard-gen-0.6b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3guard-gen-0.6b) | Qwen/Qwen3Guard-Gen-0.6B | Qwen3Guard-Gen-4B, Qwen3Guard-Gen-8B | — | — | pending | `qwen`, `qwen3` |
| 2025-09-22 | [qwen3-vl-235b-a22b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-vl-235b-a22b-thinking) | Qwen/Qwen3-VL-235B-A22B-Thinking | Qwen3-VL-30B-A3B-Thinking | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-09-20 | [qwen3-omni-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-omni-30b-a3b-instruct) | Qwen/Qwen3-Omni-30B-A3B-Instruct |  | — | — | pending | `qwen`, `qwen3` |
| 2025-09-15 | [qwen3-omni-30b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-omni-30b-a3b-thinking) | Qwen/Qwen3-Omni-30B-A3B-Thinking |  | — | — | pending | `qwen`, `qwen3` |
| 2025-09-09 | [qwen3-next-80b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-next-80b-a3b-instruct) | Qwen/Qwen3-Next-80B-A3B-Instruct |  | `qwen3` | 41,533 | replaying the set | `qwen`, `qwen3` |
| 2025-09-09 | [qwen3-next-80b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-next-80b-a3b-thinking) | Qwen/Qwen3-Next-80B-A3B-Thinking |  | `qwen3` | 54,418 | replaying the set | `qwen`, `qwen3` |
| 2025-09-08 | [ernie-4.5-21b-a3b-thinking](https://github.com/smg-project/bellwether/tree/main/fixtures/ernie-4.5-21b-a3b-thinking) | baidu/ERNIE-4.5-21B-A3B-Thinking |  | — | — | pending | none |
| 2025-08-30 | [hermes-4-14b](https://github.com/smg-project/bellwether/tree/main/fixtures/hermes-4-14b) | NousResearch/Hermes-4-14B |  | `qwen2.5` | 41,533 | replaying the set | none |
| 2025-08-21 | [deepseek-v3.1](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3.1) | deepseek-ai/DeepSeek-V3.1 |  | — | — | pending | `deepseek31`, `deepseek_v31` |
| 2025-08-20 | [seed-oss-36b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/seed-oss-36b-instruct) | ByteDance-Seed/Seed-OSS-36B-Instruct |  | `seed_oss` | 53,178 | replaying the set | none |
| 2025-08-13 | [apertus-8b-instruct-2509](https://github.com/smg-project/bellwether/tree/main/fixtures/apertus-8b-instruct-2509) | swiss-ai/Apertus-8B-Instruct-2509 |  | — | — | pending | none |
| 2025-08-05 | [qwen3-4b-instruct-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-instruct-2507) | Qwen/Qwen3-4B-Instruct-2507 |  | `qwen3` | 41,533 | replaying the set | `qwen`, `qwen3` |
| 2025-08-05 | [qwen3-4b-thinking-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-4b-thinking-2507) | Qwen/Qwen3-4B-Thinking-2507 |  | `qwen3` | 54,418 | replaying the set | `qwen`, `qwen3` |
| 2025-07-31 | [qwen3-coder-30b-a3b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-coder-30b-a3b-instruct) | Qwen/Qwen3-Coder-30B-A3B-Instruct | Qwen3-Coder-480B-A35B-Instruct | `qwen3`, tagged calls | 41,362 | replaying the set | `qwen_xml`, `qwen3` |
| 2025-07-29 | [qwen3-30b-a3b-thinking-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b-thinking-2507) | Qwen/Qwen3-30B-A3B-Thinking-2507 | Qwen3-235B-A22B-Thinking-2507 | `qwen3` | 54,418 | replaying the set | `qwen`, `qwen3` |
| 2025-07-28 | [qwen3-30b-a3b-instruct-2507](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b-instruct-2507) | Qwen/Qwen3-30B-A3B-Instruct-2507 | Qwen3-235B-A22B-Instruct-2507 | `qwen3` | 41,533 | replaying the set | `qwen`, `qwen3` |
| 2025-07-28 | [step3](https://github.com/smg-project/bellwether/tree/main/fixtures/step3) | stepfun-ai/step3 |  | — | — | pending | `step3`, `step3` |
| 2025-06-25 | [hunyuan-a13b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/hunyuan-a13b-instruct) | tencent/Hunyuan-A13B-Instruct |  | — | — | pending | none |
| 2025-04-27 | [qwen3-30b-a3b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-30b-a3b) | Qwen/Qwen3-30B-A3B | Qwen3-235B-A22B | `qwen3` | 54,418 | replaying the set | `qwen`, `qwen3` |
| 2025-04-27 | [qwen3-8b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen3-8b) | Qwen/Qwen3-8B | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-14B, Qwen3-32B | `qwen3` | 54,418 | ready | `qwen`, `qwen3` |
| 2025-03-27 | [llama-xlam-2-8b-fc-r](https://github.com/smg-project/bellwether/tree/main/fixtures/llama-xlam-2-8b-fc-r) | Salesforce/Llama-xLAM-2-8b-fc-r |  | `xlam` | 40,529 | replaying the set | `json`, none |
| 2025-03-24 | [deepseek-v3-0324](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-v3-0324) | deepseek-ai/DeepSeek-V3-0324 |  | — | — | pending | `deepseek`, none |
| 2025-03-22 | [qwen2.5-omni-7b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-omni-7b) | Qwen/Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | `qwen2.5` | 27,865 | replaying the set | `qwen`, `qwen3` |
| 2025-03-21 | [qwen2.5-vl-32b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-vl-32b-instruct) | Qwen/Qwen2.5-VL-32B-Instruct |  | `qwen2.5` | 27,865 | replaying the set | `qwen`, `qwen3` |
| 2025-03-05 | [qwq-32b](https://github.com/smg-project/bellwether/tree/main/fixtures/qwq-32b) | Qwen/QwQ-32B |  | — | — | pending | `qwen`, `qwen3` |
| 2025-02-24 | [phi-4-multimodal-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/phi-4-multimodal-instruct) | microsoft/Phi-4-multimodal-instruct |  | — | — | pending | none |
| 2025-02-19 | [phi-4-mini-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/phi-4-mini-instruct) | microsoft/Phi-4-mini-instruct |  | `plain` | 27,865 | ready | none |
| 2025-01-26 | [qwen2.5-vl-7b-instruct](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-vl-7b-instruct) | Qwen/Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-72B-Instruct | `qwen2.5` | 27,865 | replaying the set | `qwen`, `qwen3` |
| 2025-01-23 | [qwen2.5-7b-instruct-1m](https://github.com/smg-project/bellwether/tree/main/fixtures/qwen2.5-7b-instruct-1m) | Qwen/Qwen2.5-7B-Instruct-1M | Qwen2.5-14B-Instruct-1M | `qwen2.5` | 41,533 | ready | `qwen`, `qwen3` |
| 2025-01-20 | [deepseek-r1](https://github.com/smg-project/bellwether/tree/main/fixtures/deepseek-r1) | deepseek-ai/DeepSeek-R1 |  | — | — | pending | `pythonic`, `deepseek_r1` |
| 2024-05-22 | [mistral-7b-instruct-v0.3](https://github.com/smg-project/bellwether/tree/main/fixtures/mistral-7b-instruct-v0.3) | mistralai/Mistral-7B-Instruct-v0.3 |  | — | — | pending | `mistral`, none |
| 2023-12-30 | [tinyllama-1.1b-chat-v1.0](https://github.com/smg-project/bellwether/tree/main/fixtures/tinyllama-1.1b-chat-v1.0) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |  | `plain` | 27,865 | ready | `json`, none |
| 2023-12-05 | [llava-1.5-7b-hf](https://github.com/smg-project/bellwether/tree/main/fixtures/llava-1.5-7b-hf) | llava-hf/llava-1.5-7b-hf |  | — | — | pending | none |
<!-- models: end -->

Why the crate has a name: the community has spent years on many parser implementations, and none
of them is right the way we want it, SMG's own two crates included. Not right as working code;
right as a made thing. This crate is the attempt to write the one that is. Code here is a craft,
and "this can be better" is always a good enough reason for a change.

Rules this crate works under:

- It changes nothing outside `crates/symphony/`. It depends on other crates but never edits them;
  wiring it into the gateway is a separate step taken once recorded fixtures show parity.
- Formats are data. A format definition yields both the parser and the guided-decoding grammar;
  hand-written parsers are the exception and say why.
- Every format ships with fixtures recorded from the vendor's reference and the engines, replayed
  at many chunkings, and five property tests: conservation of bytes, prefix-stable arguments,
  chunking invariance, committed output stays committed, and token identity for markers.
- Every commit carries `Co-authored-by: Chang Su`, whose design this is.
