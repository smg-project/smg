# vendor_probe

Ground-truth probe harness. Sends exhaustive request combinations to the **real**
OpenAI Responses and Anthropic Messages APIs, records complete behavior (bodies,
SSE transcripts, error shapes, headers of interest), and emits structural
fingerprints. The recordings are the source of truth for auditing SMG's
compatibility — especially the Responses API.

Dual-target by design: the identical probe set replays against an SMG gateway by
swapping the provider adapter (`smg-openai` / `smg-anthropic`), producing a
structural diff.

Lives **outside** `e2e_test/` on purpose: `.github/workflows/vendor-probe.yml`
runs the recorder, and `vendor_probe/**` feeds the `agentic` change family in
`pr-test-rust.yml` so the real-engine conformance suite
(`e2e_test/vendor_compat/`, see below) revalidates probe/baseline changes.

## Layout

```
vendor_probe/
  probes/openai_responses.py     # curated tier: ~225 probes, plain data
  probes/anthropic_messages.py   # curated tier: ~163 probes, plain data
  genmatrix.py                   # generated tier: ~6.5K OpenAI / ~5.1K Anthropic
  runner.py                      # async httpx runner + provider adapters
  compat_diff.py                 # structural differ (+ --baseline gate mode)
  baseline.py                    # distills recordings into checked-in baselines
  baselines/<date>/{openai,anthropic}/   # TRACKED: cluster ground truth
  baselines/<date>/known_divergences.jsonl  # TRACKED: grandfathered divergences
```

Two tiers. The **curated** tier encodes semantic intent and dependency chains
(tool loops, background polling, conversation CRUD). The **generated** tier is a
deterministic expansion of axis specs grounded in
`crates/protocols/src/{responses,messages}.rs` — five strategies, in
budget-priority order:

| strategy | category | what it records |
|----------|----------|-----------------|
| field mutations | `gen-mutation` | exact 4xx envelope per field path (wrong type / null / empty / unknown enum / unknown sibling) |
| pairwise | `gen-pairwise` | greedy pairwise covering array over all interacting axes (seeded RNG) |
| full cartesian | `gen-3wise` | full product over the most-interacting axis family (⊇ full 3-wise); constraint-pruned so every combo is a valid request |
| boundary sweeps | `gen-boundary` | numeric ranges + length limits at {below-min, min, min+ε, typical, max, above-max} |
| content shapes | `gen-content` | system-form × turn-count × content-block combos |

`python -m vendor_probe.genmatrix` prints per-strategy counts and runs the
constraint checker (unique ids, serializable bodies, cross-field validity).
`summary.json` reports `fingerprint_clusters` per category — how many distinct
behaviors the probes collapsed into, i.e. what the volume actually bought.

Each probe is a dict: `id`, `category`, `endpoint`, `method`, `body`, `stream`,
`depends_on`, `expect` (`ok|error`), `headers` (extra/override), `poll`, `note`.
Bodies are **data**, never code. Model names are sentinels (`@MODEL`,
`@MODEL_CLASSIC`, ...) injected at run time. Dependent probes reference a prior
recording via `{{probe_id#body.output[0].id}}` placeholders (supports indexes and
`[?type]` selectors, e.g. `output[?function_call].call_id`); `{{self#body.id}}` is
the probe's own initial response (used by background polling).

## Run locally

```bash
pip install httpx

# validate the whole matrix without network (dependencies, placeholders, models)
python -m vendor_probe.runner --provider openai    --dry-run
python -m vendor_probe.runner --provider anthropic --dry-run

# record against real vendors
export OPENAI_API_KEY=...    ANTHROPIC_API_KEY=...
python -m vendor_probe.runner --provider openai    --out results/openai
python -m vendor_probe.runner --provider anthropic --out results/anthropic

# a subset
python -m vendor_probe.runner --provider openai --filter 'fntool|stream' --out /tmp/x
```

### Env (cheap defaults)

| var | default | note |
|-----|---------|------|
| `OPENAI_API_KEY` | — | required for openai |
| `OPENAI_MODEL` | `gpt-5-nano` | reasoning-capable, matches e2e pin |
| `OPENAI_MODEL_CLASSIC` | `gpt-4.1-nano` | temperature/top_p/logprobs probes |
| `OPENAI_MODEL_CUA` / `_SHELL` | `computer-use-preview` / `codex-mini-latest` | gated 4xx recordings |
| `OPENAI_VECTOR_STORE_ID` | — | file_search probes skip if unset |
| `OPENAI_PROBE_IMAGES` | — | `=1` to opt into image_generation cost |
| `ANTHROPIC_API_KEY` | — | required for anthropic |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | exercises the legacy param surface SMG models |
| `ANTHROPIC_VERSION` | `2023-06-01` | pinned |
| `ANTH_PROBE_EXPENSIVE` | — | `=1` to opt into the ~200K context-overflow probe |
| `*_BASE_URL`, `SMG_BASE_URL`, `SMG_API_KEY` | vendor hosts | replay targets |

Flags: `--tier curated|generated|all` (curated), `--budget N` (generated-tier
truncation by strategy priority; 0 = unlimited), `--max-probes N` (post-filter
cap), `--concurrency` (4; generated runs want 24), `--timeout`,
`--stream-timeout`, `--max-retries` (3), `--model`, `--model-classic`,
`--filter <regex>`.

```bash
# generated-tier smoke run (500 probes) and full dry-run validation
python -m vendor_probe.runner --provider openai --tier generated --budget 500 --dry-run
python -m vendor_probe.runner --provider openai --tier all --dry-run
```

The PR-triggered workflow run stays on the curated tier; `workflow_dispatch`
defaults to `tier=all` at concurrency 24 (`tier`, `concurrency`, `budget`
inputs).

## Dual-target replay (SMG diff)

```bash
export SMG_BASE_URL=http://localhost:8080 SMG_API_KEY=...
python -m vendor_probe.runner --provider smg-openai    --out results/smg-openai
python -m vendor_probe.runner --provider smg-anthropic --out results/smg-anthropic
```

The `smg-*` adapters reuse the same probe matrices with SMG's base_url/auth
(`smg-openai` sends `Authorization: Bearer`, `smg-anthropic` sends `x-api-key`).

### Local replay against a mock worker

The maximum local surface is the gRPC (TokenSpeed) lane: the gateway itself
builds the Responses/Messages envelopes, validates requests, renders SSE
framing, and serves the conversations/lifecycle CRUD from the in-memory
history backend — exactly the structural surface the probes record. The
canned mock worker streams fixed token ids, so response *content* is fake but
*structure* is real.

```bash
cargo build --release -p smg -p mock-worker

# 1) one canned gRPC mock worker; --tokenizer points at any local HF snapshot
#    dir with tokenizer.json (the gateway autoloads it for the worker's model)
target/release/mock-worker --grpc-base-port 19700 --grpc-count 1 \
  --model mock-model --tokenizer ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<rev> \
  --output-tokens 8 --gen-ms 0

# 2) gateway in regular gRPC mode; aliases map the vendor model names the
#    probes send onto the mock's model id (gated models like
#    computer-use-preview stay unmapped on purpose: their unknown-model
#    envelope is part of the recording). A generic chat template keeps
#    arbitrary role sequences renderable with a base tokenizer.
target/release/smg launch --host 127.0.0.1 --port 31700 \
  --prometheus-host 127.0.0.1 --prometheus-port 31701 \
  --worker-urls grpc://127.0.0.1:19700 --backend tokenspeed \
  --model-path <same tokenizer dir> --chat-template chatml.jinja \
  --history-backend memory --tool-call-parser json --reasoning-parser deepseek_r1 \
  --model-alias gpt-5-nano=mock-model --model-alias gpt-4.1-nano=mock-model \
  --model-alias claude-haiku-4-5=mock-model --policy round_robin

# 3) replay (local target: no retries, tight timeouts)
SMG_BASE_URL=http://127.0.0.1:31700 python -m vendor_probe.runner \
  --provider smg-openai --tier all --concurrency 24 --max-retries 0 \
  --timeout 30 --stream-timeout 60 --out results/smg-openai
SMG_BASE_URL=http://127.0.0.1:31700 python -m vendor_probe.runner \
  --provider smg-anthropic --tier all --concurrency 24 --max-retries 0 \
  --timeout 30 --stream-timeout 60 --out results/smg-anthropic
```

Auth stays disabled: SMG's serving auth is Bearer-only while the Anthropic
adapter authenticates via `x-api-key`, so enabling `--api-key` would 401 the
whole Anthropic matrix. The differ classifies vendor-401 probes as
`config-limited` accordingly.

## Structural compat differ

```bash
python -m vendor_probe.compat_diff \
  --vendor results/openai    --smg results/smg-openai    --provider openai    --out compat/openai
python -m vendor_probe.compat_diff \
  --vendor results/anthropic --smg results/smg-anthropic --provider anthropic --out compat/anthropic
```

Per probe it compares status (exact + class), the nested field-path inventory
of JSON bodies (direction-marked), run-length-collapsed SSE event-type
sequences, and 4xx envelope shape (`error.type`/`code`/`param`, and whether
the mutated field is named in the message — recovered from `*.mut.<slug>.*`
probe ids). Probes aggregate into the vendor-side fingerprint clusters and
the report ranks divergent clusters:

- **S1** status-class mismatch — subtyped `server-error` (SMG 5xx) >
  `rejects-valid` (vendor 2xx, SMG 4xx) > `accepts-invalid` (vendor 4xx, SMG 2xx)
- **S2** field-inventory drift (with an aggregate rollup of the most common
  vendor-only / SMG-only paths across all 200-vs-200 probes)
- **S3** SSE core-sequence drift (content-bound events stripped;
  `response.completed`/`response.incomplete` unified — which one fires is
  content-bound against a canned mock; nameless `data: [DONE]` frames are
  surfaced as `[DONE]`)
- **S4** error-envelope drift

Content-dependent differences (`output[]` item mix, `content[]` block mix,
`incomplete_details`, tool/reasoning/annotation events, delta arity) are
classified **mock-limited**, never divergences. Value-level volatility (ids,
timestamps, model names, token counts) never registers: body comparison is
path-based, and envelope comparisons scrub volatile substrings first.
Outputs: `report.md` (ranked, human) + `diff.jsonl` (one line per probe,
machine-readable, tagged with cluster id and verdict).

## Checked-in baselines (`baselines/<date>/`)

Raw recordings are tens of MB and stay out of git (`.gitignore`); the durable
copies live as assets on the `vendor-truth-<date>` GitHub release that full
`workflow_dispatch` runs publish. What IS tracked is the distillation
`baseline.py` produces — per vendor fingerprint cluster (the same key
`compat_diff` aggregates by): the signature (status + sorted field paths or
run-collapsed SSE event types), a stable `sig_hash`, member probe ids with
category/expect, and one normalized exemplar. Exemplars are scrubbed exactly
as `compat_diff` normalizes (ids/timestamps/model names/token counts →
placeholders, volatile integers → 0); error bodies are kept verbatim (the S4
envelope comparison needs their values), large 2xx bodies keep only the
field-path skeleton. Output is fully deterministic (sorted clusters, members,
keys), ~1.6 MB (OpenAI, 138 clusters) + ~0.7 MB (Anthropic, 26 clusters);
`manifest.json` pins provenance (probe-set git sha, recording run id, models,
counts, cost).

```bash
python -m vendor_probe.baseline \
    --results results/openai --provider openai \
    --out vendor_probe/baselines/<date>/openai \
    --probe-set-sha <sha> --run-id <run> --date <date>
```

### Baseline diff + regression gate

`compat_diff --baseline` diffs an SMG replay against a baseline dir instead of
raw vendor recordings — same severity ranking, same report format, plus stable
cluster ids (`Bnnn`) and a regression gate: with `--allowlist`, any divergent
cluster whose `sig_hash` is not in `known_divergences.jsonl` — or whose
severity got worse than allowlisted — exits 1 (plus a replay-coverage floor so
a dead gateway can't fake green). The checked-in allowlist grandfathers the
150 divergent clusters present when the baseline was cut (133 OpenAI + 17
Anthropic); the gate catches *regressions beyond that state*, and
`--write-allowlist` regenerates a provider's lines after divergences are
deliberately fixed or accepted.

```bash
python -m vendor_probe.compat_diff \
    --baseline vendor_probe/baselines/2026-08-21/openai \
    --smg results/smg-openai --provider openai --out compat/openai \
    --allowlist vendor_probe/baselines/2026-08-21/known_divergences.jsonl
```

### Refreshing the baselines

1. Full recording: dispatch the Vendor Probe workflow (`tier=all`). Healthy
   runs auto-publish gzipped recordings to the `vendor-truth-<date>` release.
2. `gh release download vendor-truth-<date>`, run `baseline.py` per provider
   into `vendor_probe/baselines/<new-date>/` with the new run's provenance.
3. Replay locally (mock-worker recipe below, or a real stack) and run
   `compat_diff --baseline ... --write-allowlist` per provider to re-grandfather
   the then-current divergences into the new `known_divergences.jsonl` —
   diff it against the old allowlist and account for every entry that
   appeared or disappeared before committing.

## Real-engine conformance suite (`e2e_test/vendor_compat/`)

The ongoing gate is an e2e suite that replays cluster representatives (up to 3
member probes per cluster — first/middle/last of the sorted ids — plus
dependency closure; ~380 requests total) against the PR lane's real gateway +
engine, and asserts every baseline cluster still holds structurally. One test
per (surface, cluster); grandfathered divergences xfail; a new divergent
cluster or a known one getting *worse* fails with a ready-to-paste
`TO-TRIAGE` allowlist line. It rides the `e2e-1gpu-responses` lane in
`pr-test-rust.yml` (gated on the `common`/`agentic` change families, which
include `crates/protocols`, `model_gateway`, and `vendor_probe/**`).

Because that lane runs a real model, content-dependent clusters that a canned
mock could never adjudicate get compared for real there; structural drift they
were hiding surfaces as failures to triage — add allowlist entries with a
`TO-TRIAGE` note rather than silently absorbing them.

Run it against any running SMG (E2E-style env):

```bash
# lane-style: launch workers + gateway via the e2e harness
pytest e2e_test/vendor_compat -k openai -s -vv

# knobs
VENDOR_COMPAT_FULL=1              # replay every baseline member (nightly)
VENDOR_COMPAT_CONCURRENCY=16
VENDOR_COMPAT_TIMEOUT=120 VENDOR_COMPAT_STREAM_TIMEOUT=240
```

The CPU-side logic (representative selection, cluster judgment) is
unit-tested in `e2e_test/vendor_compat/test_conformance_unit.py`, which runs
in the harness unit-test job without a GPU.

### CI permission note

The `publish-truth` job in `vendor-probe.yml` carries `contents: write`
(wider than the workflow's default `contents: read`) to create the
`vendor-truth-<date>` release and upload assets. It is scoped to that job
only; the probe job holding the vendor API keys stays read-only.

## Artifact layout (per provider `--out` dir)

- `results.jsonl` — one line per probe: `{probe_id, category, request{method,path,body},
  response{status, headers(subset), body | sse_events[]}, timing_ms, attempt_count,
  error, expect, recorded, transport_failure, skipped}`.
- `fingerprints.jsonl` — per probe: `field_paths` (sorted dotted key inventory,
  arrays collapsed to `[]`) or `event_type_sequence` + `event_count` for streams.
- `summary.json` — counts per category, recorded/skipped/transport-failure tallies,
  expected-error recordings, token totals + rough cost estimate.

## Classification

- A behavioral **4xx** (from an `expect: "error"` probe or anywhere) is a
  **successful recording**, not a failure — the error envelope is the ground truth.
- A **transport failure** is a network/timeout error after retries, or a persistent
  `429`/`5xx`. Only these count against the CI health gate (>20% fails the job).
- **Skipped**: missing optional resource (vector store), cost-gated probe, or a
  dependency that did not produce a usable recording (cascade).

## Cost / rate limits

Retries honor `Retry-After` with exponential backoff + jitter (max 3). Dependency
chains run strictly serial (topological); independent probes run at concurrency 4.
Full runs: OpenAI ~$1–3 (hard-bounded <$10), Anthropic ~$0.60 (<$3 with retries).
Error probes are free (rejected pre-billing). Nightly, not per-PR.
