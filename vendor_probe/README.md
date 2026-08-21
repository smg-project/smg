# vendor_probe

Ground-truth probe harness. Sends exhaustive request combinations to the **real**
OpenAI Responses and Anthropic Messages APIs, records complete behavior (bodies,
SSE transcripts, error shapes, headers of interest), and emits structural
fingerprints. The recordings are the source of truth for auditing SMG's
compatibility — especially the Responses API.

Dual-target by design: the identical probe set replays against an SMG gateway by
swapping the provider adapter (`smg-openai` / `smg-anthropic`), producing a
structural diff.

Lives **outside** `e2e_test/` on purpose — no existing CI path filter picks it up.
Only `.github/workflows/vendor-probe.yml` runs it.

## Layout

```
vendor_probe/
  probes/openai_responses.py     # curated tier: ~225 probes, plain data
  probes/anthropic_messages.py   # curated tier: ~163 probes, plain data
  genmatrix.py                   # generated tier: ~6.5K OpenAI / ~5.1K Anthropic
  runner.py                      # async httpx runner + provider adapters
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

## Dual-target replay (future SMG diff)

```bash
export SMG_BASE_URL=http://localhost:8080 SMG_API_KEY=...
python -m vendor_probe.runner --provider smg-openai    --out results/smg-openai
python -m vendor_probe.runner --provider smg-anthropic --out results/smg-anthropic
```

The `smg-*` adapters reuse the same probe matrices with SMG's base_url/auth. Diff
`fingerprints.jsonl` (vendor vs SMG) by `probe_id`: `field_paths` for JSON bodies,
`event_type_sequence` for streams. The fingerprint block is what a diff tool
consumes first; `results.jsonl` holds the raw bytes for deeper inspection.

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
