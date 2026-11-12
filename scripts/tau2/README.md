# τ²-bench nightly A/B — `scripts/tau2/`

**Track B** of the multi-turn parser-verification suite: run **τ²-bench** (`tau2-bench`) against two serving "arms" and diff pass^1 and pass^k. The companion **Track A** (an offline, deterministic parser-conformance gate) is proposed but not yet built.

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
| `launch_arms.sh` | bring up one arm (`a` = pure vLLM, `b` = vLLM gRPC worker + SMG gateway); prints its base_url on stdout. AMD uses owned sessions and completion receipts; NVIDIA retains its existing launcher and cleanup. Fully env-parameterised. |
| `run_ab.py` | point `tau2 run` at both arms (one domain at a time), read back `results.json`, compute pass^1 and pass^k per domain, emit a markdown + JSON comparison table, and apply a regression gate. Arms must already be serving. |

## Quick start (manual, e.g. on a GPU box)

```bash
# 0) one-time: install tau2-bench (uv required); and ninja in the vLLM env (see Gotchas)
git clone https://github.com/sierra-research/tau2-bench ~/tau/tau2-bench
cd ~/tau/tau2-bench && uv sync
~/vllm-env/bin/pip install ninja          # then ensure ~/vllm-env/bin is on PATH

# 1) bring up both arms (here: Qwen3.8-27B, TP=2, one arm per GPU pair)
# Admission and approved paid-policy setup are separate prerequisites.
export TAU2_SCOPE="manual-$(date +%s)-$$"
export TAU2_RUN_DIR="${TMPDIR:-/tmp}/$TAU2_SCOPE"  # separate directory per comparison
export TAU2_MODEL=Qwen/Qwen3.8-27B VLLM_BIN=~/vllm-env/bin/vllm \
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
    --agent-model Qwen/Qwen3.8-27B \
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

Key env knobs for `launch_arms.sh`: `TAU2_GPU` (selected vendor device list, e.g. `0,1`), `TAU2_TP` (tensor-parallel size — match the GPU count), `TAU2_MAX_MODEL_LEN`, `TAU2_{VLLM,SMG}_{TOOL,REASONING}_PARSER`, `TAU2_VLLM_EXTRA` for extra vLLM flags, and `TAU2_ARM_A_PORT` / `TAU2_ARM_B_GRPC_PORT` / `TAU2_ARM_B_GW_PORT` / `TAU2_ARM_B_METRICS_PORT` to pin ports (default: OS-assigned free ports, so concurrent arms/jobs on one host don't collide).

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

## Per-model parser flags (the weekly matrix)

Mirrors `nightly-bfcl.yml`'s 5-leg matrix and runs weekly (Thursdays 07:17 UTC). The 2
H100 legs run full task sets; the 3 Blackwell legs are capped at `num_tasks=30`/domain
(tau2 is multi-turn + spends gpt-5.2 per turn) and are skipped on the schedule while
the `blackwell` runner is offline (no job since 2026-09-18); a dispatch still runs them. `deepseek-v4.1` runs **sequential** (whole 8-GPU node per arm — `run_ab.py
--score-arm` each arm, then `--diff`); the rest run both arms concurrently on opposite
GPU halves. On PRs **all** legs run on a tiny retail / 1-trial / few-task subset — a
quick "does each leg launch + parse + score" smoke (the heavy Blackwell legs are
dominated by model-load time, serialized by a host lock, so a PR run is not fast).

| leg | model | runner (TP) | vLLM tool/reason | SMG tool/reason |
|---|---|---|---|---|
| qwen3.8 | Qwen/Qwen3.8-27B | 4-gpu-h100 (2) | `qwen3_xml` / `qwen3` | `qwen_xml` / `qwen3` |
| gpt-oss | openai/gpt-oss-120b | 4-gpu-h100 (2) | `openai` / — | — / — (harmony auto) |
| deepseek-v4.1 | deepseek-ai/DeepSeek-V4.1-Flash-1017 | blackwell (8, seq) | `deepseek_v41` / `deepseek_v41` | `deepseek_v41` / `deepseek_v41` |
| minimax-m3 | MiniMaxAI/MiniMax-M3-MXFP8 | blackwell (4) | `minimax_m3` / `minimax_m3` | `minimax_m3` / `minimax_m3` |
| glm-5.3-flash | zai-org/GLM-5.3-Flash | blackwell (4) | `glm47` / `glm45` | `glm47_moe` / `glm45` |

> Dispatch `only=<leg>` runs a single leg; `model=` overrides its weights. SKU ids and
> vLLM parser names may shift; confirm against the installed vLLM build:
> `vllm serve --help | grep -A40 tool-call-parser`.

## Gotchas discovered while bringing this up (read before debugging)

- **Install `ninja` in the vLLM env (do NOT reach for `--enforce-eager`).** vLLM's torch.compile / CUDA-graph path shells out to `ninja` to build kernels; if it's missing the engine dies with `No such file or directory: 'ninja'`. `--enforce-eager` only *hides* this by skipping compilation (slower). Real fix: `pip install ninja` in the vLLM env **and put its bin on `PATH`** (vLLM execs `ninja` by name).
- **Cap the context.** Qwen3 models default to a very large `max_model_len` → OOM on init. Pass `--max-model-len 16384` (the launch helper defaults to this); use the **same** value on both arms to keep conditions identical.
- **SMG auto model→parser mapping lags new SKUs.** SMG's factory doesn't yet map `Qwen3.6*` (it falls back to the JSON `qwen` parser, wrong for the XML format), so pass `--tool-call-parser qwen_xml` explicitly via `TAU2_SMG_TOOL_PARSER=qwen_xml`. Adding a `Qwen3.6*`→`qwen_xml` mapping to `crates/tool_parser` is a good follow-up.
- **`tau2-bench` install: use `uv sync`.** Clone the repo and run `uv sync` inside it — this resolves all deps (including any optional extras) into `.venv`. The executable is `tau2-bench/.venv/bin/tau2`; pass its full path to `--tau2`.
- **Results path.** Results land at `<DATA_DIR>/simulations/<save-to>/results.json`. `run_ab.py` sets `save-to` to `ab_<arm>_<domain>` automatically and pins tau2's `TAU2_DATA_DIR` to `--data-dir` so writes land where it reads. If a tau2 run dies before writing results (API error, engine death), `run_ab.py` warns and skips that domain — it renders as `—` in the report rather than aborting the whole A/B.
- **Exclude `banking_knowledge`.** The `banking_knowledge` domain requires OPENAI_API_KEY-authenticated embeddings to set up its retrieval fixture — it will fail without dedicated infra. Omit it from `--domains`; the default (`retail,airline,telecom`) already excludes it.
- **`OPENAI_API_KEY` is required.** The user-simulator (`gpt-5.2`) goes through the OpenAI API. If the key is unset or invalid, every user turn fails and pass^k collapses to 0 on both arms equally — a bad signal to debug.

## Validation status — ran end-to-end ✅

Brought up end-to-end on one node, **`Qwen/Qwen3.6-27B` at TP=1 per arm** (on a high-memory GPU: the 27B model's ~54 GB of BF16 weights fit on one card; the Quick start and the weekly matrix run a 27B at TP=2), user-sim **`gpt-5.2`** at temp 0. Stack: vLLM 0.24.0 (CUDA 13.0), smg 1.7.0, tau2-bench 1.0.0.

**FC confirmed on both arms** — identical `get_weather(city="Paris")` tool call, `finish_reason=tool_calls`. One frontend delta observed: the pure-vLLM arm returned empty `reasoning_content` while the SMG arm populated it (a reasoning-parse/template difference; the tool call itself is identical).

**Preliminary A/B — retail, n=10 tasks, 1 trial** (a pipeline smoke, *not* a statistical result):

| domain | pure vLLM pass^1 | SMG → vLLM gRPC pass^1 | Δ |
|---|---|---|---|
| retail (n=10, k=1) | 70.00 | 60.00 | −10.00 |

The arms **agree on 7/10 tasks** and diverge on 3 (SMG loses tasks 5 & 6, wins task 8 → net −1). The split is *mixed*, consistent with multi-turn trajectory divergence — small frontend differences (e.g. the `reasoning_content` delta above) amplified across turns via the LLM user-sim — rather than a systematic parser defect. **n=10 / k=1 is not conclusive**; a real parity verdict needs the nightly's larger, k=2 run (ideally several). This mixed-trajectory signal is exactly what single-turn BFCL AST matching cannot surface — the reason τ²-bench is worth running.

**Throughput & cost** (these set the nightly knobs): 20 dialogues (retail 10 tasks × 2 arms, 1 trial) took **19.5 min** wall at tau2's default concurrency (~85 s/dialogue) and **$0.17** of `gpt-5.2` user-sim spend (~$0.009/dialogue). At default concurrency the full 3-domain × k=2 nightly would take ~18 h, so the workflow sets **`--max-concurrency 16`** to keep it within the 360-min ceiling. Scale trials/domains for tighter confidence intervals.

## Paid request budget entry

`run_ab.py` now launches each domain with the pinned τ² venv interpreter and
`-I scripts/tau2/guarded_cli.py`, then calls the real `tau2.cli:main`. The guard
requires Python3.12 and checks the installed τ² source at `5bfa7e37b36656b37dc6d022156be6563c1007f3`,
LiteLLM 1.81.11, OpenAI 2.20.0 and HTTPX 0.28.1, including the critical request
source hashes. The nightly uses that source's frozen lock and pins its extra
import dependencies. An unqualified source/version refuses before simulation.
The guard stays installed for the child process lifetime, including exceptions
and thread shutdown. Paid clients are closed and removed from the allowlist in
each synchronous call’s finally; live pools are bounded by inflight calls.
Redirects are disabled for both paid and local SDK requests; best-effort `sitecustomize` is not an admission mechanism.

The supported paid scope is the existing plain-text `llm_agent` /
`user_simulator`, base split, retail/airline/telecom. Both `gpt-5.2` user turns and
the required `gpt-4.1-2025-04-14` natural-language evaluator are charged. Retail
has 112/114 base tasks requiring that evaluator; its quality checks are retained.
Only the exact declared local agent model and loopback endpoint are uncharged.
Voice, knowledge/embeddings, optional reviews, other providers, streaming,
asynchronous SDK requests and unknown paid callsites refuse without sending.
This is a boundary for the pinned text pipeline, not a general egress sandbox or
support claim for all τ² modes. CUDA launch/cleanup and AMD qualification remain
separate; this change does not add a paid or AMD execution authorization.

Before live scoring, supply all of these explicitly (paths may also be passed as
`--budget-*` flags). No real prices, provider limits or spending approval are
provided by this repository:

- `TAU2_BUDGET_POLICY`: JSON containing `scope_id`, integer `cap_nusd`, `calls`,
  `calls_per_arm`, and `models` with exactly `openai/gpt-5.2` and
  `openai/gpt-4.1-2025-04-14`. Each model has explicit decimal-string
  `input_per_million` / `output_per_million`, positive integer `input_limit` /
  `output_limit`, and `service_tier: "default"`. `cap_nusd` is billionths of USD.
- `TAU2_BUDGET_CONTEXTS`: a separately verified provider-contract JSON with
  `schema_version: 1` and `models` keyed by those two model names without the
  `openai/` prefix. Each entry has the complete provider `context_tokens` bound,
  the official `source_url`, and the explicitly reviewed resolved
  `response_models` IDs covered by the same pricing/context contract.
- `TAU2_BUDGET_CONTEXTS_SHA256`: the independently reviewed SHA256 of the exact
  context file bytes. A URL plus an invented number is not provider evidence.
  The tool does not discover/attest provider limits: real values must be verified
  outside the policy before execution. No real bound is supplied here; fixtures
  use synthetic contracts only. An estimate of the last message's tokens or the
  policy's `input_limit` itself is not a complete input bound.
- `TAU2_BUDGET_LEDGER`: one absolute SQLite file shared by both arms, every domain
  and both sequential score processes. The explicit guarded `--initialize` entry creates it once using an exclusive
  new file before either arm; live scoring, guarded children and saved-score
  acceptance open an existing file only. Do not rerun initialization after loss of a live ledger; restore its state or
  obtain a new scope/approval. An old or missing/incompatible ledger is not reset or silently recreated at acceptance.
- `TAU2_BUDGET_SCOPE`: the exact approved comparison scope matching the policy.
  In CI it is `<run_id>:<run_attempt>:<matrix.name>`. Approval is either for
  that exact job or an explicit standing per-job allowance described below.
  This is a per-comparison-job cap, not an aggregate monthly/multi-host billing
  controller. Matrix and repeated-run exposure must be explicitly approved.

The nightly reads the policy/context JSON and context SHA from the corresponding
`TAU2_BUDGET_POLICY_JSON`, `TAU2_PROVIDER_CONTEXTS_JSON` and
`TAU2_PROVIDER_CONTEXTS_SHA256` repository configuration. Missing configuration
fails before model staging/arm launch. No variables are changed by this source
proposal. The policy files/ledger live in one scope-hash directory (preventing shared-temp
collisions) and are included with the existing result artifact;
providers' credentials and prompts are not ledger fields.

Every actual paid HTTP send reserves the entire independently verified provider
context bound plus `max_completion_tokens` at that model's rates, under one
atomic global USD/call/per-arm cap. This deliberately over-reserves accepted
input, including full system/history/tool schemas and framing, instead of using
an optimistic token estimate. The final wire must have one non-streaming
completion, explicit `service_tier=default`, the protected output cap (including
reasoning tokens), and only the supported text/function-tool fields. The
returned model and actual tier must match the reviewed contract. A missing or
unknown tier/model/usage, a transport error, an over-bound count, or a refused
request retains its reservation and halts further admission; no successful call
refunds its worst-case reservation. Requested standard pricing is never treated
as known actual spend after an unknown-tier response. The guarantee depends on
the approved provider contract; an unknown outcome is incomplete, not a billing
success claim.

Task/transport retries cannot get an unreserved send. Task retries are disabled,
known SDK retry options are forced to zero, and the final transport boundary
still covers the SDK's own retries and LiteLLM's parameter/role reformat retries.
Requests already reserved/in flight may finish when another fails; their exposure
is already included in the cap. Each row records actual model/role/arm/domain/
invocation/task/trial, status and known token usage, with no key/prompt. A domain
checks only its own pending/unknown requests when it ends; the final gate checks
global pending/unknown/halted after both arms join. Sequential saved-score diffs
reopen the same policy/scope/ledger and recheck current state, so an earlier clean
summary cannot hide a later halt or deletion. Legacy score files without budget
provenance remain readable but cannot pass the normal budgeted CLI gate.

CPU tests execute the pinned UserSimulator, NL evaluator, task wrapper and
LiteLLM/OpenAI request chain through fake HTTP transports with socket connections
forbidden and fixed offline keys. Synthetic two-model prices/bounds validate
mechanisms, not actual pricing, quality, spend, GPU cleanup or continuous CI.

For continuous schedules, an operator can instead explicitly approve a **standing
per-comparison-job allowance**. Replace the exact `scope_id` with this `approval`
object while retaining the same explicit caps, two-model prices and independently
reviewed context contract:

```json
{"mode":"standing", "budget_unit":"per_comparison_job",
 "workflow_ref":"ORG/REPO/.github/workflows/nightly-tau2.yml@refs/heads/main",
 "allowed_events":["schedule"], "matrix_names":["EXPLICIT_MODEL_NAME"],
 "not_before":"2026-11-09T00:00:00Z", "expires_at":"2026-11-10T00:00:00Z"}
```

These illustrative values are not approval. The workflow's trusted run ID,
attempt, workflow/ref, event and matrix name must match the approved template and
UTC start window before any directory/ledger initialization. It materializes a
concrete `run:attempt:matrix` policy, then uses the same exclusive initializer and
dual-arm ledger. One standing configuration can cover multiple explicitly listed
matrix jobs and natural scheduled runs: **each comparison job gets the stated
allowance**, so total exposure is the allowance multiplied by admitted jobs and
recurrences. This is not a monthly or multi-host aggregate cap. Exact-job approval
remains available and cannot be mixed with standing mode. Neither mode recreates
a missing ledger during scoring or saved-result acceptance. No actual standing
approval, repository variable change or paid execution is part of this proposal.


The earlier `costmap_shim/sitecustomize.py` startup hook remains a separate,
explicit `SMG_TAU2_BUDGET_GUARD=1` opt-in for legacy callers. Its unconfigured
synchronous authorizer still refuses requests; it is not the paid authorizer for
this guarded workflow. `run_ab.py` launches `guarded_cli.py` with Python `-I`,
which ignores the inherited `PYTHONPATH` shim. The explicit guard owns the actual
SDK/send boundary, so the legacy hook cannot add another reservation or reject
an otherwise approved guarded call. Both legacy startup/hook tests and the new
pinned-SDK/result suites run in the same CPU workflow; this compatibility does
not grant a policy, spending cap or execution permission.

The source check resolves the actual imported `tau2` package, including the
checkout used by `uv sync` editable installations, and verifies the same pinned
request-path hashes in either installation mode. Distribution versions and SDK
source fingerprints remain mandatory; an editable install does not waive them.

A timeout or HTTP error from the exact uncharged loopback agent remains a
simulation/quality failure, without halting the shared paid ledger. Unknown
clients, redirects outside the fixed request contract, and paid request failures
still halt it and make budget acceptance fail independently of quality tolerance.

### Optional AMD native arms (source capability, off by default)

`SMG_RUN_AMD_TAU2=true` adds `qwen3.8-amd` to the existing matrix. This is
independent of ordinary AMD PR enablement. It uses the existing digest-pinned
vLLM 0.31.0 ROCm image and `ci_setup_vllm_rocm_image.sh`, retaining
`Qwen/Qwen3.8-27B`, concurrent TP2 arms (0,1 and 2,3), REST vLLM baseline and
raw-token vLLM gRPC plus SMG candidate, the original parsers, retail/airline/
telecom, 30 tasks per domain and two trials. AMD model/parser/task/timeout
overrides that change this comparison are rejected. NVIDIA rows retain their
native commands and package setup. No CUDA engine-wheel override runs on AMD.

The AMD runner selector must resolve to the installed host preclaim service
with a complete **390-minute** envelope: preparation15 + job360 + cleanup10 +
margin5. That service owns the existing `/run/lock/amd-ci-gpu.lock` through the
whole Worker and cleanup, checks the daily 05:00–06:30 UTC Router window and
its profile expiry. A temporary mixed profile expires within24h; enabling the
workflow does not extend it or establish permanent Sunday routing. The job
creates no second host lock and never invokes GPU nuke/unknown-process cleanup
on AMD. Opt-in does not grant runner maintenance, GPU admission or paid budget.

On AMD, `launch_arms.sh` delegates to `native_arms.py`. Every AMD a/b/stop invocation
requires the same absolute `TAU2_RUN_DIR` and nonempty unique `TAU2_SCOPE`.
The workflow uses run ID, attempt and matrix name for both, including its
independent always-cleanup step. The directory is private0700, refuses symlinks
and foreign scope, and records only owned session supervisor identities
(PID/start time/PGID/session, UID and random invocation cookie). The supervisor
outlives an engine leader and reaps its session descendants before recording
completion. Cleanup sends no signal if identity/cookie ownership is uncertain;
missing/incomplete receipts fail the job. Linux pidfds pin each owner/member
before identity checks and signal delivery; there is no numeric-PID signal fallback. TERM and KILL cleanup are finite;
logs require no tail process. Startup failure, cancellation and timeout also
clean already-started arms. This is process ownership, not proof of reclaimed
GPU memory or escaped native processes; host preclaim remains the final
full-Worker cleanup boundary. Unique scope directories are not reused across
jobs. Sequential NVIDIA a-stop-b remains supported within one scope.

NVIDIA keeps the original launch commands, PID files, process-group teardown
and existing host-lock/recovery path. A failed stop retains a failing job status
while that existing NVIDIA fallback still runs. AMD ownership refusal never
enters the NVIDIA fallback. The pidfd and receipt guarantees above apply to the
AMD helper; restoring the NVIDIA path does not establish new ownership guarantees
or grant permission to clean processes outside its existing resource scope.

Result uploads stage the current summary, JSON comparison, expected-task manifest,
transcripts and available budget files under one workspace directory. This avoids
the container runner's partial translation of a multi-line upload path and retains
available diagnostics after a failed run; unrelated caches are not included.

AMD selects each disjoint physical list through `HIP_VISIBLE_DEVICES`, rejects
inherited nonempty HIP/ROCR/CUDA visibility to avoid double mapping, and checks
unique numeric IDs and TP count before spawn. Actual ROCm ordinal isolation,
exact model/quantization/parser/kernel support, endpoint/tool/reasoning output,
GPU memory cleanup and quality parity still require admitted runtime testing
on this exact image; CPU command capture cannot establish them. The other
scheduled/manual models have no AMD capability claim from this batch.

Existing task manifests/seed300, trial coverage, same-policy paid ledger across
both arms/domains, request guards and final hard budget gate remain mandatory.
Public provider facts have been collected separately; execution still needs
approved call/USD/output/scope bounds and paid/GPU authorization. No provider
call, GPU test or quality score is authorized by this source feature. A source
review, command fixture or passing CPU workflow is not sustained CI acceptance.
