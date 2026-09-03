# smg-rl

RL control plane for the Shepherd Model Gateway. Enabled with `--enable-rl`.

| Route | Purpose |
|---|---|
| `GET /v1/rl/workers` | list workers with engine, topology, health, weight version, capabilities |
| `GET /v1/rl/workers/{id}` | one worker |
| `GET\|POST /v1/rl/workers/{id}/engine/{path}` | proxy one engine-native route to one worker |
| `GET\|POST /v1/rl/engine/{path}?selector=...` | the same call fanned out to every matching worker |

Flags: `--enable-rl`, `--rl-control-timeout-secs` (600), `--rl-fanout-concurrency` (32).
Recommended RL launch profile: `--enable-rl --disable-health-check --disable-circuit-breaker --request-timeout-secs 14400`.

Python client: `from smg.rl import RL` (stdlib only). Example: `examples/rl/refit_from_disk.py`.

Design: `docs/superpowers/specs/2026-09-03-rl-m1-discovery-passthrough-design.md`.
