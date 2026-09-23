# RL rollouts on TokenSpeed behind SMG

SMG's RL control plane (`--enable-rl`, `/v1/rl/*`) drives TokenSpeed engines
through their in-engine SGLang-compatible control app while the data plane
stays on gRPC.

## Launch

Each engine, on its own GPUs:

    python3 -m smg_grpc_servicer.tokenspeed --model /ckpt/policy --host 0.0.0.0 --port 30000 \
      --rl-control-host 0.0.0.0 --rl-control-port 30400 --rl-control-api-key "$RL_KEY" \
      --enable-output-logprobs

The gateway:

    smg launch --worker-urls grpc://rollout-1:30000 grpc://rollout-2:30000 \
      --policy cache_aware --enable-rl --disable-health-check --disable-circuit-breaker \
      --request-timeout-secs 14400

Register the engines with their control key so the proxy authenticates:

    curl -X POST http://smg:30000/workers -H 'content-type: application/json' \
      -d '{"url":"grpc://rollout-1:30000","api_key":"'"$RL_KEY"'"}'

`GET /v1/rl/workers` then shows `engine: tokenspeed`, `connection_mode: grpc`,
`control_url: http://rollout-1:30400` and the engine's advertised capabilities.

## Refit

TokenSpeed's scheduler has no receive path for a disk or tensor refit:
`update_weights_from_disk` and `update_weights_from_tensor` both answer HTTP
501 (`{"success": false, "message": "... is not implemented by this build's
scheduler; supported sources: distributed"}`) and the engine keeps serving.
`examples/rl/refit_from_disk.py` is for SGLang only — do not point it at a
TokenSpeed selector.

The only refit path TokenSpeed implements is the trainer-driven NCCL
broadcast (slime's path): the trainer calls, per worker, through `/v1/rl`,

    init_weights_update_group   (once, to join the trainer's process group)
    update_weights_from_distributed   (once per refit)
    destroy_weights_update_group      (on teardown)

with `pause_generation` / `continue_generation` fanned out around the
`update_weights_from_distributed` call, same as a disk refit. The next
`/generate` through SMG reports the new `meta_info.weight_version`, stamped
by the engine on the gRPC response.

`examples/rl/refit_from_trainer.py` is that refit, end to end:

    python examples/rl/refit_from_trainer.py --smg http://smg:30000 \
      --master-address trainer-0.internal \
      --model-path /ckpt/step-42 --weight-version 42 --selector engine=tokenspeed

It runs on the trainer's host, needs `torch` built with CUDA and
`transformers`, and loads `--model-path` onto `--device` (default `cuda:0`) in
place of a live policy. `--master-address` is the trainer host's address as the
engines see it, since each engine dials it to join the group; the default
`127.0.0.1` only works when engines and trainer share a host, and the script
refuses to start when `--smg` names a non-loopback host while
`--master-address` is still loopback. Ranks come from each worker's `tp_size`: the trainer
takes rank 0 and engine *k* takes `rank_offset_k .. rank_offset_k + tp_k - 1`,
so `world_size = 1 + sum(tp)`. An engine launched without an explicit
parallelism flag reports no `tp_size` (TokenSpeed leaves `attn_tp_size` unset,
so discovery has nothing to fold into the label); the script assumes 1 and says
so on stderr, and `--tp-size` overrides it. `init_weights_update_group` is a
per-worker call, because each worker gets a different `rank_offset`; the
broadcast is a fan-out, because the body is identical. `--chunk` (default 64) parameters ride
on each `update_weights_from_distributed` call, so a large policy streams in
batches. The version is stamped only on the last chunk, so a refit is never
advertised as complete before every weight has landed.

An HTTP-level failure is cleaned up: the group is destroyed, the cache flushed,
the engines resumed, and any engine left holding a part-old, part-new model is
named on stderr. An engine that stops participating in the collective is not.
The trainer's main thread is inside NCCL by then, so torch's watchdog takes the
process down without raising into Python and nothing resumes the fleet. Recover
with a `continue_generation` fan-out:

    python -c 'from smg.rl import RL; RL("http://smg:30000").fanout(
        "continue_generation", {}, selector="engine=tokenspeed")'

The engine expects the checkpoint's own HuggingFace parameter names, in
broadcast order; its `load_weights` does the fused/stacked mapping (`q_proj`,
`k_proj`, `v_proj` into `qkv_proj`, and so on) exactly as it did for the
initial load.

## `/generate` parity with slime

slime's rollout client speaks SGLang's native `/generate` format directly over
gRPC: it sends one prompt per request (`text` as a string, or `input_ids` as a
flat token list) and never sets `model`. The gRPC router matches SGLang's
shape for that case: a single prompt with `n` unset (or `1`) and exactly one
response comes back as one JSON object, so `resp["meta_info"]` indexes
directly instead of unwrapping a one-element list; a batch or `n > 1` still
comes back as a list. When `model` is omitted, the gateway resolves it to the
single real model at least one worker is tagged with (an untagged worker
doesn't count as a model of its own); with zero or more than one such model
behind the gateway an unnamed request still 404s (`model_not_found`), since
there is no longer a single unambiguous target. Sending `text` as a list of
prompts (SGLang's batch text form) is rejected before it reaches any
router — the JSON body fails to parse (422) on every path, gRPC included,
since `text` is a plain string field. Use `input_ids` as a list of lists for
a token-id batch instead; gRPC rejects that too (400), but HTTP accepts it.

## Security

The control app accepts weight updates from anyone who can reach it. Bind it
on a routable host only with `--rl-control-api-key`, and give SMG the same key
as the worker's `api_key`.

## Older engines

Engines that predate advertisement get SMG's static capability row (`wait`
and `abort`, `distributed` only) and need the label supplied at
registration: `{"url":"grpc://…","labels":{"rl.control_url":"http://…:30400"}}`.
See `crates/rl/NOTES.md` for their route-level drift.
