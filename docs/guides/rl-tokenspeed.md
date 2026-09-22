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

    python3 examples/rl/refit_from_disk.py --smg http://smg:30000 \
      --model-path /ckpt/step-42 --weight-version 42 --selector engine=tokenspeed

The next `/generate` through SMG reports `meta_info.weight_version: "42"`,
stamped by the engine on the gRPC response.

## Security

The control app accepts weight updates from anyone who can reach it. Bind it
on a routable host only with `--rl-control-api-key`, and give SMG the same key
as the worker's `api_key`.

## Older engines

Engines that predate advertisement get SMG's static capability row (`wait`
and `abort`, `disk` and `distributed`) and need the label supplied at
registration: `{"url":"grpc://…","labels":{"rl.control_url":"http://…:30400"}}`.
See `crates/rl/NOTES.md` for their route-level drift.
