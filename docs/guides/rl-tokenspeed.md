# RL rollouts on TokenSpeed behind SMG

SMG's RL control plane (`--enable-rl`, `/v1/rl/*`) drives TokenSpeed engines
through their in-engine control app (the SGLang-style routes slime speaks)
while the data plane stays on gRPC.

## Launch

Each engine, on its own GPUs:

    python3 -m smg_grpc_servicer.tokenspeed --model /ckpt/policy --host 0.0.0.0 --port 30000 \
      --rl-control-host 10.0.0.11 --rl-control-port 30400 --rl-control-api-key "$RL_KEY" \
      --enable-output-logprobs

`--rl-control-host` is the address the gateway reaches the engine on. The
engine advertises `rl.control_url` only for a concrete host: a wildcard bind
(`0.0.0.0`, `::`) is not advertised, so the worker would have no control
endpoint and every control call would answer 422 `no_control_endpoint`. The
default, loopback, is right only when SMG runs on the same machine.

The gateway, with no startup workers:

    smg launch --policy cache_aware --enable-rl --disable-health-check --disable-circuit-breaker \
      --request-timeout-secs 14400

Register each engine with its control key, which the proxy sends as the
bearer to the control app:

    curl -X POST http://smg:30000/workers -H 'content-type: application/json' \
      -d '{"url":"grpc://rollout-1:30000","api_key":"'"$RL_KEY"'"}'

A worker named in `--worker-urls` registers without a key, and registering
it again this way is a 409, so list no startup workers and register every
engine as above.

`GET /v1/rl/workers` then shows `engine: tokenspeed`, `connection_mode: grpc`,
`control_url: http://10.0.0.11:30400` and the engine's advertised capabilities.

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
Ranks come from each worker's `tp_size`, which discovery reads from the
engine's server args (TokenSpeed's own spelling, `attn_tp_size`, is folded
into it). The newest TokenSpeed builds nest their parallelism under
`mapping.*`, which discovery does not read yet, so such an engine reports
`tp_size: null` and the trainer must be told the width (or assume 1).

## Security

The control app accepts weight updates from anyone who can reach it, and a
remote gateway needs it on a routable host, so always set
`--rl-control-api-key` and give SMG the same key as the worker's `api_key`.

## Older engines

Engines that predate advertisement get SMG's static capability row (`wait`
and `abort`, `distributed` only) and need the label supplied at
registration: `{"url":"grpc://…","labels":{"rl.control_url":"http://…:30400"}}`.
See `crates/rl/NOTES.md` for their route-level drift.
