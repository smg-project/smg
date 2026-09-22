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
by the engine on the gRPC response. A worked trainer-side example will land
under `examples/rl` once available; until then, drive the three calls
directly with `smg.rl.RL.call`/`fanout` as shown in `crates/rl/README.md`.

## Security

The control app accepts weight updates from anyone who can reach it. Bind it
on a routable host only with `--rl-control-api-key`, and give SMG the same key
as the worker's `api_key`.

## Older engines

Engines that predate advertisement get SMG's static capability row (`wait`
and `abort`, `distributed` only) and need the label supplied at
registration: `{"url":"grpc://…","labels":{"rl.control_url":"http://…:30400"}}`.
See `crates/rl/NOTES.md` for their route-level drift.
