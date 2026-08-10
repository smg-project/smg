# BFCL nightly A/B — `scripts/bfcl/`

**Track B** of the parser-verification proposal (`docs/proposals/2026-06-10-bfcl-nightly-parser-verification.md`): run the **official** Berkeley Function Calling Leaderboard (`bfcl-eval`) against two serving "arms" and diff the scores. The companion **Track A** (offline, deterministic parser-conformance gate) lives in `crates/parser_conformance/`.

## The experiment

Two arms expose an identical OpenAI `/v1` endpoint. The same official `bfcl` CLI (FC mode) is pointed at each; **everything is held fixed except the frontend**, so any score delta is attributable to the tokenization + parsing layer — the number that argues for an engine adopting SMG's frontend.

| | baseline | candidate |
|---|---|---|
| arm | **pure vLLM** | **SMG → vLLM (gRPC)** |
| who renders the chat template + tokenizes | vLLM | SMG |
| who parses tool calls / reasoning | vLLM (`--tool-call-parser hermes`) | SMG (`--tool-call-parser qwen`) |
| model · engine · checkpoint · sampling | **identical** | **identical** |

**Why FC mode is mandatory.** BFCL's `…-FC` model handlers send the native `tools` param and score `response.choices[].message.tool_calls` — i.e. the *server's parsed output*. The non-FC (prompt) handlers format tools into the prompt and parse the text themselves, bypassing the server parser. Only FC mode puts SMG's / vLLM's parser on the critical path, so the driver always uses the `-FC` handler (e.g. `Qwen/Qwen3-4B-Instruct-2507-FC`).

## Files

| file | what |
|---|---|
| `launch_arm.sh` | bring up one arm (`a` = pure vLLM, `b` = vLLM-gRPC + SMG); prints its base_url; `stop` tears down via pidfiles. Fully env-parameterised. |
| `run_ab.py` | point official `bfcl generate`+`evaluate` (FC mode) at both arms, parse per-category accuracy + test-case count, emit a markdown + JSON comparison table (with both **unweighted** = mean-of-categories and **weighted-by-n** = micro overall, matching BFCL's `calculate_unweighted/weighted_accuracy`), and a regression gate. Arms must already be serving. |
| `register_bfcl_model.py` | register a model that bfcl-eval doesn't ship a handler for yet (new SKUs), by cloning an existing FC entry. Idempotent. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: a venv with bfcl-eval (+ soundfile), and ninja in the *vLLM* env
#    for torch.compile / CUDA-graph kernel builds (see Gotchas).
python -m venv ~/bfcl-env && ~/bfcl-env/bin/pip install bfcl-eval soundfile
~/vllm-env/bin/pip install ninja          # then ensure ~/vllm-env/bin is on PATH

# teach bfcl about a model it doesn't ship a handler for yet (new SKUs)
~/bfcl-env/bin/python register_bfcl_model.py --model-id Qwen/Qwen3.6-27B

# 1) bring up both arms (here: Qwen3.6-27B, TP=2, one arm per GPU pair)
export BFCL_MODEL=Qwen/Qwen3.6-27B VLLM_BIN=~/vllm-env/bin/vllm \
       VLLM_PYTHON=~/vllm-env/bin/python SMG_LAUNCH="$HOME/smg/target/ci/smg launch" \
       BFCL_TP=2 BFCL_MAX_MODEL_LEN=16384 PATH=~/vllm-env/bin:$PATH
A_URL=$(BFCL_GPU=0,1 BFCL_VLLM_TOOL_PARSER=qwen3_xml BFCL_VLLM_REASONING_PARSER=qwen3 bash launch_arm.sh a)
B_URL=$(BFCL_GPU=2,3 BFCL_SMG_TOOL_PARSER=qwen_xml  BFCL_SMG_REASONING_PARSER=qwen3 bash launch_arm.sh b)

# 2) run the official A/B
~/bfcl-env/bin/python run_ab.py \
    --baseline  "vllm=$A_URL" \
    --candidate "smg=$B_URL" \
    --bfcl-model Qwen/Qwen3.6-27B-FC \
    --categories simple_python,multiple,parallel,irrelevance \
    --bfcl ~/bfcl-env/bin/bfcl --project-root ~/bfcl_ab \
    --out ~/bfcl_ab.md --json-out ~/bfcl_ab.json

# 3) teardown
bash launch_arm.sh stop
```

Key env knobs for `launch_arm.sh`: `BFCL_GPU` (CUDA_VISIBLE_DEVICES, e.g. `0,1`), `BFCL_TP` (tensor-parallel size — match the GPU count), `BFCL_MAX_MODEL_LEN`, `BFCL_{VLLM,SMG}_{TOOL,REASONING}_PARSER`, and `BFCL_VLLM_EXTRA` for extra vLLM flags.

`run_ab.py` exits non-zero if the candidate's overall accuracy drops more than `--tolerance` (default 2pp) below the baseline.

## Per-model parser flags (the nightly matrix)

| model (matrix leg) | runner | TP/arm | pure-vLLM `--tool-call-parser` / `--reasoning-parser` | SMG `--tool-call-parser` / `--reasoning-parser` |
|---|---|---|---|---|
| Qwen3.6-27B (`qwen3.6`) | `4-gpu-h100` | 2 | `qwen3_xml` / `qwen3` | `qwen_xml` / `qwen3` |
| gpt-oss-120b (`gpt-oss`) | `4-gpu-h100` | 2 | `openai` / — | _(none — SMG auto-routes harmony)_ / — |
| DeepSeek-V4-Flash-0731 (`deepseek-v4`) | `blackwell` | 4 | `deepseek_v4` / `deepseek_v4` (+`--tokenizer-mode deepseek_v4 --trust-remote-code`) | `deepseek_v4` / `deepseek_v4` |
| MiniMax-M2.7 (`minimax-m2.7`) | `blackwell` | 4 | `minimax_m2` / `minimax_m2` (+`--trust-remote-code`) | `minimax_m2` / `minimax` |
| Kimi-K2.6 int4 (`kimi-k2.6`) | `blackwell` | 4 | `kimi_k2` / `kimi_k2` (+`--trust-remote-code`) | `kimik2` / `kimi_k25`† |

> **gpt-oss has no SMG tool-call-parser.** SMG handles gpt-oss through its harmony
> pipeline (`model_gateway/src/routers/grpc/harmony/`), auto-activated by
> `HarmonyDetector` on a `gpt-oss` model id / `GptOssForCausalLM` architecture. So
> the SMG arm passes **no** `--tool-call-parser` (`BFCL_SMG_TOOL_PARSER=""`). The
> `—` in **both** reasoning-parser columns for gpt-oss is likewise intentional:
> harmony carries its own reasoning channel, so neither arm sets a reasoning parser.
>
> **† Reasoning-parser fallbacks.** SMG's reasoning registry has no `kimi_k2` entry
> yet; the closest existing parser (`kimi_k25`) is used. Confirm on the first
> nightly; adding exact parsers to `crates/reasoning_parser` is a follow-up if
> outputs diverge.
>
> The mid-2026 SKU ids and a couple of vLLM parser names may shift; confirm against
> the installed vLLM build: `vllm serve --help | grep -A40 tool-call-parser`.

## Matrix & runners

The nightly (`.github/workflows/nightly-bfcl.yml`) runs the A/B as a GitHub Actions
matrix — one leg per model, `fail-fast: false`, each on its own runner:

- `4-gpu-h100` — Qwen3.6-27B and gpt-oss-120b, TP=2 per arm (GPUs 0,1 + 2,3).
- `blackwell` (B200) — DeepSeek-V4-Flash-0731, MiniMax-M2.7, Kimi-K2.6 int4, TP=4 per arm
  (GPUs 0-3 + 4-7).

All legs use `max_model_len` **32768**: the `multi_turn` categories emit ~18k-token
prompts that 400'd ("decoder prompt longer than the maximum model length") at 16384.
Weights already fit at each leg's TP and the extra KV for a 32k window is only a few
GB/sequence, so the larger context needs no extra GPUs — arms stay **concurrent**.

Each leg sets `arm_mode`:

- **concurrent** (all current legs) — both arms serve at once on opposite GPU halves;
  `run_ab.py` scores them **in parallel** (separate servers/GPUs, no contention) and diffs.
- **sequential** (in reserve, unused) — for a model that needs the whole node (TP=8)
  so the arms can't coexist: `run_ab.py --score-arm` scores arm A alone → tears it
  down → scores arm B alone → `--diff-baseline/--diff-candidate` compares the two
  saved score files. Flip a leg's `arm_mode` to enable it.

Per the A/B's premise, model size is irrelevant — a smaller same-family checkpoint
exercises the identical parser — so the matrix uses DeepSeek-V4-Flash-0731 and int4
Kimi-K2.6 to validate the `deepseek_v4` / `kimi_k2` parsers without the full weights.

`workflow_dispatch` can target one leg via the `only` input and override
`model`/`bfcl_model`/parsers per run. PRs touching this pipeline run **all** legs
(H100 + Blackwell) as an end-to-end sanity check, but cheaply — the PR category set
is a tiny non-live subset (`simple_python,irrelevance`) for every leg.

## Gotchas discovered while bringing this up (read before debugging)

- **`bfcl-eval` needs `soundfile`.** Its Qwen handler imports `qwen_agent` → `soundfile`; without it `bfcl --help` itself crashes. `pip install soundfile`.
- **Cap the context.** Qwen3-4B defaults to a 256K `max_model_len` → ~36 GiB KV cache → engine init OOM. Pass `--max-model-len 16384` (the launch helper defaults to this); use the **same** value on both arms.
- **Install `ninja` in the vLLM env (do NOT reach for `--enforce-eager`).** vLLM's torch.compile / CUDA-graph path shells out to `ninja` to build kernels (required for newer archs like Qwen3.6's `qwen3_5`); if it's missing the engine dies with `No such file or directory: 'ninja'`. `--enforce-eager` only *hides* this by skipping compilation (slower). Real fix: `pip install ninja` in the vLLM env **and put its bin on `PATH`** (vLLM execs `ninja` by name) — then run with CUDA graphs, no `--enforce-eager`.
- **Don't force HF offline.** With the model cached, bfcl runs fine online (~7 req/s measured); a one-off slow run is usually a transient HF hiccup, not a systematic per-request throttle. `run_ab.py` does **not** set `HF_HUB_OFFLINE`; set it yourself only for air-gapped boxes. No HF token is needed for public models.
- **New models need a bfcl handler.** bfcl-eval pins a fixed model list; a brand-new SKU (e.g. `Qwen/Qwen3.6-27B`) isn't in it, so `bfcl generate --model <id>-FC` fails with "Unknown model_name". Run `register_bfcl_model.py --model-id <id>` first — it registers the id with **`OpenAICompletionsHandler`** (generic OpenAI Chat Completions FC: native `tools`, server-parsed `tool_calls`, OpenAI-message multi-turn). Do **not** use the model-family *local* handlers (e.g. `QwenFCHandler`): they re-serialize multi-turn tool calls into a text prompt assuming a `{"name","arguments"}` shape and `KeyError` on SKUs whose tool-call JSON differs (DeepSeek-V4 → `'arguments'`, MiniMax-M2 → `'name'`). `run_ab.py` points bfcl at each arm via `OPENAI_BASE_URL`.
- **SMG auto model→parser mapping lags new SKUs.** SMG's factory doesn't yet map `Qwen3.6*` (it falls back to the JSON `qwen` parser, wrong for the XML format), so pass `--tool-call-parser qwen_xml` explicitly. Adding a `Qwen3.6*`→`qwen_xml` mapping to `crates/tool_parser` is a good follow-up.
- **Use the `-FC` handler.** `Qwen/Qwen3-4B-Instruct-2507-FC`, not the bare name (which is prompt mode and bypasses the server parser).
- **`bfcl generate --skip-server-setup`** points at `LOCAL_SERVER_ENDPOINT` / `LOCAL_SERVER_PORT`. (Custom full base_urls behind a proxy are still rigid — gorilla issue #1280.)
- **`EngineDeadError` on startup is usually the missing-`ninja` issue above** (the engine dies around CUDA-graph capture / first compile). Install `ninja` and put it on `PATH` rather than falling back to `--enforce-eager`.

## Validation status — ran end-to-end ✅

Run on a dev H100 box, **`Qwen/Qwen3.6-27B` at TP=2** (one arm per GPU pair), the **full `non_live` set** (1390 cases), FC mode, temp 0.001 — **no `--enforce-eager`, online HF** (the clean config, after `pip install ninja`):

| category | pure vLLM (`qwen3_xml`) | SMG → vLLM gRPC (`qwen_xml`) | Δ |
|---|---|---|---|
| simple_python | 95.00 | 94.75 | −0.25 |
| simple_java | 64.00 | 64.00 | 0.00 |
| simple_javascript | 72.00 | 72.00 | 0.00 |
| multiple | 91.00 | 91.50 | +0.50 |
| parallel | 89.50 | 90.00 | +0.50 |
| parallel_multiple | 91.00 | 91.50 | +0.50 |
| irrelevance | 84.58 | 84.58 | 0.00 |
| **overall (unweighted)** | **83.87** | **84.05** | **+0.18** |

So SMG's Rust frontend is **at parity** with vLLM's native parser across the full non-live set — marginally ahead overall (+0.18pp), and never worse than −0.25pp on any category. Both arms reasoning-parse `<think>` into `reasoning_content` and emit native `tool_calls` (FC confirmed end to end). The low java/js numbers are the *model's* non-Python ability (identical on both arms) — confirming the A/B isolates the frontend, not model quality. (An earlier `simple_python`-only run on Qwen3-4B gave 95.50 vs 95.25, same parity story.)

Scope note: the table above is the `non_live` slice. The **nightly** runs the broader reproducible set — **`non_live` + `live` + `multi_turn`** (17 categories; `live` is real-user data, `multi_turn` is state-based simulation — both static, no internet). It excludes the agentic/executable categories (`web_search_*`, `memory_*`, `exec_*`) that need web-search/memory/sandbox infra. **PRs** that touch the pipeline run only a quick non-live sanity subset (`simple_python,irrelevance`); per-PR correctness is the cheap CPU Track-A parser-conformance gate (`crates/parser_conformance`). Scale to multiple runs × the five target models for tight confidence intervals.
