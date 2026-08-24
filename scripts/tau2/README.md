# τ²-bench nightly A/B — `scripts/tau2/`

**Track B** of the multi-turn parser-verification suite: run **τ²-bench** (`tau2-bench`) against serving arms. Normal legs diff pass^1/pass^k between pure vLLM and SMG; an `smg_only` leg reports absolute scores for a checkpoint the pinned vLLM cannot serve. The companion **Track A** (an offline, deterministic parser-conformance gate) is proposed but not yet built — see the parser-verification proposal in `docs/proposals/`.

## The experiment

Two arms expose an identical OpenAI `/v1` endpoint. The same `tau2` CLI is pointed at each arm for the **agent LLM**; the **user-simulator is always the external `gpt-5.2`** (via `OPENAI_API_KEY`), identical on both arms. **Everything is held fixed except the agent frontend**, so any pass^k delta is attributable to the tokenization + tool/reasoning parsing layer — the number that argues for an engine adopting SMG's frontend.

| | baseline | candidate |
|---|---|---|
| arm | **pure vLLM** | **SMG → vLLM (gRPC)** |
| who renders the chat template + tokenizes | vLLM | SMG |
| who parses tool calls / reasoning | vLLM (`--tool-call-parser qwen3_xml`) | SMG (`--tool-call-parser qwen_xml`) |
| user-simulator | **gpt-5.2 (fixed, identical)** | **gpt-5.2 (fixed, identical)** |
| model · engine · checkpoint · sampling | **identical** | **identical** |

**Why native FC mode puts SMG's parser on the critical path.** τ²-bench's agent instructs the model via native `tools` and reads `response.choices[].message.tool_calls` — the *server's parsed output*. This means SMG's Rust tool-call parser (or vLLM's) is on the critical path for every agent step, and any parsing error shows up directly as a failed task. The user-sim side is purely the OpenAI API and is identical on both arms.

## Files

| file | what |
|---|---|
| `launch_arms.sh` | bring up one arm (`a` = pure vLLM, `b` = SMG in front of a vLLM or SGLang gRPC worker); prints its base_url; `stop` tears down via pidfiles. |
| `run_ab.py` | point `tau2 run` at arms, read `results.json`, and compute pass^1/pass^k. A/B mode emits deltas and a regression gate; `--report-arm` emits absolute single-arm scores with the same completeness gate. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: install tau2-bench (uv required); and ninja in the vLLM env (see Gotchas)
git clone https://github.com/sierra-research/tau2-bench ~/tau/tau2-bench
cd ~/tau/tau2-bench && uv sync
~/vllm-env/bin/pip install ninja          # then ensure ~/vllm-env/bin is on PATH

# 1) bring up both arms (here: Qwen3.6-27B, TP=2, one arm per GPU pair)
export TAU2_MODEL=Qwen/Qwen3.6-27B VLLM_BIN=~/vllm-env/bin/vllm \
       VLLM_PYTHON=~/vllm-env/bin/python SMG_LAUNCH="$HOME/smg/target/ci/smg launch" \
       TAU2_TP=2 TAU2_MAX_MODEL_LEN=16384 PATH=~/vllm-env/bin:$PATH
A_URL=$(TAU2_GPU=0,1 TAU2_VLLM_TOOL_PARSER=qwen3_xml TAU2_VLLM_REASONING_PARSER=qwen3 bash launch_arms.sh a)
B_URL=$(TAU2_GPU=2,3 TAU2_SMG_TOOL_PARSER=qwen_xml   TAU2_SMG_REASONING_PARSER=qwen3  bash launch_arms.sh b)

# 2) set your OpenAI key so the user-sim (gpt-5.2) can reach the API
export OPENAI_API_KEY=sk-...

# 3) run the A/B (run_ab.py is stdlib-only — any python3; give its full path or
#    run from the smg repo root as shown)
python scripts/tau2/run_ab.py \
    --baseline  "vllm=$A_URL" \
    --candidate "smg=$B_URL" \
    --agent-model Qwen/Qwen3.6-27B \
    --user-llm gpt-5.2 \
    --tau2 ~/tau/tau2-bench/.venv/bin/tau2 \
    --data-dir ~/tau/tau2-bench/data \
    --domains retail,airline,telecom \
    --num-trials 2 \
    --max-concurrency 16 \
    --out ~/tau2_ab.md --json-out ~/tau2_ab.json

# 4) teardown
bash launch_arms.sh stop
```

Key env knobs for `launch_arms.sh`: `TAU2_GPU` (CUDA_VISIBLE_DEVICES, e.g. `0,1`), `TAU2_TP` (tensor-parallel size — match the GPU count), `TAU2_MAX_MODEL_LEN`, `TAU2_{VLLM,SMG}_{TOOL,REASONING}_PARSER`, `TAU2_ARM_B_WORKER` (`vllm` or `sglang`), the engine-specific `TAU2_{VLLM,SGLANG}_EXTRA` flags, and the `TAU2_ARM_*_PORT` overrides.

`run_ab.py` exits non-zero if the candidate's overall pass^k drops more than `--tolerance` (default 2pp) below the baseline. Use `--max-concurrency N` to run N tau2 simulations per arm in parallel (the lever that keeps large runs within a time budget).

## Agent-vs-user routing (no config file)

τ²-bench routes the agent and user LLMs **per-call** — no LiteLLM config file is needed. `run_ab.py` passes the arm's URL directly via `--agent-llm-args`:

```
--agent-llm openai/<model>
--agent-llm-args '{"api_base": "<arm>/v1", "api_key": "smg-local"}'
--user-llm gpt-5.2
```

The user-simulator (`gpt-5.2`) is reached via the standard OpenAI API using `OPENAI_API_KEY`. The agent is routed to whichever arm is under test. `run_ab.py` constructs and passes these arguments automatically — you only need to supply `--agent-model`, `--user-llm`, and `--baseline`/`--candidate`.

Results land at:

```
<DATA_DIR>/simulations/ab_<arm_name>_<domain>/results.json
```

where `<DATA_DIR>` is the path you pass to the **required** `--data-dir` flag (typically `tau2-bench/data`); tau2 reads and writes results under it (override tau2's own location with `TAU2_DATA_DIR` if it differs).

## Per-model parser flags (the nightly matrix)

Mirrors `nightly-bfcl.yml`'s 7-leg matrix. Most legs run both arms concurrently;
`glm-5.2` runs sequentially on a whole node, and Muse-Glimmer runs one SGLang-backed
SMG arm. Expensive legs are capped at `num_tasks=30`/domain to bound GPU time and
gpt-5.2 user-simulator spend. The PR trigger is disabled; use a targeted dispatch
with retail / one trial / a few tasks for pre-merge smoke coverage.

| leg | model | runner (TP) | vLLM tool/reason | SMG tool/reason |
|---|---|---|---|---|
| qwen3.6 | Qwen/Qwen3.6-27B | 4-gpu-h100 (2) | `qwen3_xml` / `qwen3` | `qwen_xml` / `qwen3` |
| gpt-oss | openai/gpt-oss-120b | 4-gpu-h100 (2) | `openai` / — | — / — (harmony auto) |
| deepseek-v4 | deepseek-ai/DeepSeek-V4-Flash-0731 | blackwell (4) | `deepseek_v4` / `deepseek_v4` | `deepseek_v4` / `deepseek_v4` |
| minimax-m2.7 | MiniMaxAI/MiniMax-M2.7 | blackwell (4) | `minimax_m2` / `minimax_m2` | `minimax_m2` / `minimax` |
| kimi-k2.6 | moonshotai/Kimi-K2.6 | blackwell (4) | `kimi_k2` / `kimi_k2` | `kimik2` / `kimi_thinking` |
| glm-5.2 | zai-org/GLM-5.2-FP8 | blackwell (8, seq) | `glm47` / `glm45` | `glm47_moe` / `glm45` |
| muse-glimmer | meta-models/Muse-Glimmer-30B | 4-gpu-h100 (2, single arm) | unsupported | SGLang + `muse_glimmer` / `muse_glimmer` |

> Dispatch `only=<leg>` runs a single leg; `model=` overrides its weights. SKU ids and
> vLLM parser names may shift; confirm against the installed vLLM build:
> `vllm serve --help | grep -A40 tool-call-parser`.
>
> Muse-Glimmer's τ² result is an absolute SMG score, not a frontend A/B. It is also
> not directly comparable to the model card's τ3-Banking number: benchmark version,
> domain, and user-simulator setup differ.

## Gotchas discovered while bringing this up (read before debugging)

- **Install `ninja` in the vLLM env (do NOT reach for `--enforce-eager`).** vLLM's torch.compile / CUDA-graph path shells out to `ninja` to build kernels; if it's missing the engine dies with `No such file or directory: 'ninja'`. `--enforce-eager` only *hides* this by skipping compilation (slower). Real fix: `pip install ninja` in the vLLM env **and put its bin on `PATH`** (vLLM execs `ninja` by name).
- **Cap the context.** Qwen3 models default to a very large `max_model_len` → OOM on init. Pass `--max-model-len 16384` (the launch helper defaults to this); use the **same** value on both arms to keep conditions identical.
- **SMG auto model→parser mapping lags new SKUs.** SMG's factory doesn't yet map `Qwen3.6*` (it falls back to the JSON `qwen` parser, wrong for the XML format), so pass `--tool-call-parser qwen_xml` explicitly via `TAU2_SMG_TOOL_PARSER=qwen_xml`. Adding a `Qwen3.6*`→`qwen_xml` mapping to `crates/tool_parser` is a good follow-up.
- **`tau2-bench` install: use `uv sync`.** Clone the repo and run `uv sync` inside it — this resolves all deps (including any optional extras) into `.venv`. The executable is `tau2-bench/.venv/bin/tau2`; pass its full path to `--tau2`.
- **Results path.** Results land at `<DATA_DIR>/simulations/<save-to>/results.json`. `run_ab.py` sets `save-to` to `ab_<arm>_<domain>` automatically and pins tau2's `TAU2_DATA_DIR` to `--data-dir` so writes land where it reads. If a tau2 run dies before writing results (API error, engine death), `run_ab.py` warns and skips that domain — it renders as `—` in the report rather than aborting the whole A/B.
- **Exclude `banking_knowledge`.** The `banking_knowledge` domain requires OPENAI_API_KEY-authenticated embeddings to set up its retrieval fixture — it will fail without dedicated infra. Omit it from `--domains`; the default (`retail,airline,telecom`) already excludes it.
- **`OPENAI_API_KEY` is required.** The user-simulator (`gpt-5.2`) goes through the OpenAI API. If the key is unset or invalid, every user turn fails and pass^k collapses to 0 on both arms equally — a bad signal to debug.

## Validation status — ran end-to-end ✅

Brought up end-to-end on an 8×B200 box (`moirai-b200`), **`Qwen/Qwen3.6-27B` at TP=1 per arm** (the 183 GB Blackwell cards fit the 27B model on a single GPU), user-sim **`gpt-5.2`** at temp 0. Stack: vLLM 0.24.0 (CUDA 13.0, Blackwell sm_100), smg 1.7.0, tau2-bench 1.0.0.

**FC confirmed on both arms** — identical `get_weather(city="Paris")` tool call, `finish_reason=tool_calls`. One frontend delta observed: the pure-vLLM arm returned empty `reasoning_content` while the SMG arm populated it (a reasoning-parse/template difference; the tool call itself is identical).

**Preliminary A/B — retail, n=10 tasks, 1 trial** (a pipeline smoke, *not* a statistical result):

| domain | pure vLLM pass^1 | SMG → vLLM gRPC pass^1 | Δ |
|---|---|---|---|
| retail (n=10, k=1) | 70.00 | 60.00 | −10.00 |

The arms **agree on 7/10 tasks** and diverge on 3 (SMG loses tasks 5 & 6, wins task 8 → net −1). The split is *mixed*, consistent with multi-turn trajectory divergence — small frontend differences (e.g. the `reasoning_content` delta above) amplified across turns via the LLM user-sim — rather than a systematic parser defect. **n=10 / k=1 is not conclusive**; a real parity verdict needs the nightly's larger, k=2 run (ideally several). This mixed-trajectory signal is exactly what single-turn BFCL AST matching cannot surface — the reason τ²-bench is worth running.

**Throughput & cost** (these set the nightly knobs): 20 dialogues (retail 10 tasks × 2 arms, 1 trial) took **19.5 min** wall at tau2's default concurrency (~85 s/dialogue) and **$0.17** of `gpt-5.2` user-sim spend (~$0.009/dialogue). At default concurrency the full 3-domain × k=2 nightly would take ~18 h, so the workflow sets **`--max-concurrency 16`** to keep it within the 360-min ceiling. Scale trials/domains for tighter confidence intervals.
