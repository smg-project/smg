# TokenSpeed Worker Transport Benchmark: gRPC vs ZMQ

Date: 2026-08-31 (America/Los_Angeles)

## Conclusion

For a colocated TokenSpeed engine, ZMQ is a CPU-efficiency improvement, not a
TTFT improvement.

- Router + Worker + engine-frontend CPU fell by 41–43% across the tested load
  points because ZMQ removes the Python gRPC engine frontend.
- Total service CPU, including the TokenSpeed scheduler, fell by 5% at 1 RPS
  and about 12% at 8–16 RPS.
- Output throughput was unchanged within the tested envelope.
- At 8–16 RPS, ZMQ TTFT was 3–4 ms slower at p50 and about 8 ms slower at p99.
- ZMQ recovered that time during streaming: E2E p50 improved 2–4% and ITL p99
  improved 6–22%. High-load E2E p99 was 4% worse, so the tail is mixed.

Keep Router-to-Worker Rust gRPC as the stable cross-node contract and use ZMQ
for the colocated TokenSpeed engine boundary. Do not claim a latency or maximum
throughput win from this result. The next optimization target is the ZMQ
first-frame/wakeup path.

## Environment

- SMG commit: `67ad87ae2`
- GPU: one NVIDIA B200 on `gpu-dp-6jdq4-kcqtx`
- model: `Qwen/Qwen3-1.7B`
- image: `lightseekorg/tokenspeed-runner:latest`
- image digest: `sha256:d6067daeeb1fafecc531d45e282797076e1cd2e2c16eaa90712634dd76a709ca`
- TokenSpeed source: `c3ea3cd883048e4a4a444ec0481d270b19f0103d`
- TP 1, max model length 4096, max sequences 16, GPU memory utilization
  0.5, host KV store disabled
- both arms: HTTP Router -> Rust `WorkerInference` gRPC -> Worker adapter
- gRPC arm: Worker -> Python TokenSpeed gRPC frontend -> scheduler
- ZMQ arm: Worker -> msgpack ZMQ IPC -> headless TokenSpeed scheduler

## Workload

- OpenAI streaming completions, 128 output tokens per request
- synthetic fixed-size prompt: 128 words, observed as roughly 611–637 input
  tokens
- no shared-prefix reuse
- open-loop Poisson arrivals at 1, 8, and 16 requested RPS
- 30 seconds per run, three runs per transport/load pair, identical seeds
- medians below are the median of the three run-level statistics
- load client ran on the host; service CPU came from the container cgroup
- GPU utilization and power were sampled every 500 ms

The gRPC arm performed its built-in engine warmup plus an external warmup. The
ZMQ arm has no built-in request warmup, so two external warmup rounds were run
before measurements. Warmup samples are excluded.

## Results

All latency columns are milliseconds. `CPU cores` is average total service CPU
over the run. `Control CPU` is CPU seconds per 30-second run for Router,
Worker, and the engine frontend, excluding the TokenSpeed scheduler.

| Offered RPS | Transport | Success | Output tok/s | TTFT p50 | TTFT p99 | ITL p99 | E2E p50 | E2E p99 | CPU cores | Control CPU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | gRPC | 104/104 | 142.5 | 24.1 | 45.1 | 4.61 | 320.0 | 377.4 | 1.145 | 1.72 |
| 1 | ZMQ | 104/104 | 142.6 | 24.0 | 45.0 | 4.45 | 310.8 | 370.5 | 1.088 | 0.98 |
| 8 | gRPC | 713/713 | 1027.9 | 34.2 | 50.4 | 36.63 | 396.8 | 484.6 | 1.529 | 7.54 |
| 8 | ZMQ | 713/713 | 1027.4 | 37.4 | 58.4 | 34.33 | 387.6 | 481.0 | 1.347 | 4.32 |
| 16 | gRPC | 1413/1433 | 1957.5 | 27.6 | 50.3 | 39.47 | 414.0 | 557.5 | 1.706 | 12.07 |
| 16 | ZMQ | 1414/1432 | 1957.0 | 31.6 | 58.0 | 30.78 | 399.5 | 580.7 | 1.500 | 7.15 |

### ZMQ relative to gRPC

| Offered RPS | Output tok/s | TTFT p50 | TTFT p99 | ITL p99 | E2E p50 | E2E p99 | Total CPU | Control CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.1% | -0.4% | -0.2% | -3.5% | -2.9% | -1.8% | -5.0% | -43.0% |
| 8 | ~0% | +9.4% | +15.9% | -6.3% | -2.3% | -0.7% | -11.9% | -42.7% |
| 16 | ~0% | +14.5% | +15.3% | -22.0% | -3.5% | +4.2% | -12.1% | -40.8% |

At 16 RPS both transports entered the same bounded-admission region: gRPC
completed 98.60% and ZMQ 98.74% of submitted requests. This test therefore
shows equivalent behavior at the configured concurrency limit, not a measured
maximum-throughput difference.

## Interpretation

At 8 RPS the median gRPC engine frontend used 3.49 CPU seconds per run. At 16
RPS it used 5.13 seconds. ZMQ removes that process; the Rust Worker spends about
0.3 seconds more per run on the native adapter, leaving a net control-path CPU
reduction near 41–43%.

The few-millisecond TTFT regression is not GPU compute: GPU power and output
throughput were effectively the same. It is most likely in the ZMQ
request-to-first-frame scheduling/wakeup path. That is an inference from this
benchmark and should be confirmed with tracing before changing the protocol.

## Reproduction and caveats

Use `scripts/bench_two_tier_transport.py` around `scripts/sim_load.py`. The
runner records latency/throughput, cgroup CPU, per-process CPU, GPU utilization,
and power in one JSON result.

- The Python gRPC frontend logs each request at INFO; removing the frontend and
  its logging is part of the measured architecture difference.
- ZMQ startup enables TokenSpeed's grammar and output-logprob capabilities by
  default, but this workload requested neither feature.
- Three runs on one model, host, and GPU are enough for an engineering decision,
  not a universal performance claim.
- Output token counts use TokenSpeed SSE chunks. Each measured request emitted
  128 text chunks, matching the requested output length.
