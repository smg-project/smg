//! Runtime configuration for the mock worker fleet, parsed from CLI flags.

use std::{path::PathBuf, time::Duration};

use crate::engine::{Calibration, EngineParams, LoadsLike, TimingModel};

/// Configuration shared by every mocked HTTP and gRPC worker in the process.
#[derive(Debug, Clone)]
pub struct Config {
    /// Bind address for every listener.
    pub host: String,
    /// First HTTP port; `http_count` workers bind `[http_base_port, +count)`.
    pub http_base_port: u16,
    /// Number of HTTP workers to start.
    pub http_count: u16,
    /// First gRPC port; `grpc_count` workers bind `[grpc_base_port, +count)`.
    pub grpc_base_port: u16,
    /// Number of gRPC workers to start.
    pub grpc_count: u16,
    /// Frontend handshake `ipc://` address ZMQ mock engines connect to (they
    /// dial the SMG/test frontend, which binds the sockets).
    pub zmq_handshake: Option<String>,
    /// Number of ZMQ mock EngineCore ranks to start (each a DP rank).
    pub zmq_count: u16,
    /// Engine index of the first ZMQ rank; ranks use `[start, start+count)`.
    pub zmq_start_index: u32,
    /// Model id advertised by every worker (one model, many replicas).
    pub model_id: String,
    /// Tokenizer path advertised by gRPC workers (for gateway autoload).
    pub tokenizer_path: String,
    /// Simulated per-request generation latency (canned mode only).
    pub gen_delay: Duration,
    /// Number of canned output tokens per generation; also the default output
    /// length for realistic mode when a request omits `max_tokens`.
    pub output_tokens: u32,
    /// When true, each worker runs the realistic continuous-batching engine
    /// simulator ([`EngineParams`]); when false, the cheap canned path.
    pub realistic: bool,
    /// Engine-simulator parameters (only used when `realistic`).
    pub engine: EngineParams,
    /// Port of the process-wide admin API (fleet, request records, cache
    /// dumps, resets); off when `None`.
    pub admin_port: Option<u16>,
    /// Context length advertised to the gateway (`max_context_length`,
    /// `max_req_input_len`, `max_model_len`).
    pub context_length: u32,
    /// vLLM-wire ZMQ KV-event publishers (realistic engines only): the first
    /// PUB port; worker `i` publishes on `base + 2i` and answers replay on
    /// `base + 2i + 1`. Off when `None`.
    pub kv_events_zmq_base_port: Option<u16>,
    /// Whether the publishers serve replay requests (ROUTER at `port + 1`).
    pub kv_events_replay: bool,
    /// Topic frame of every published message (vLLM's default is empty).
    pub kv_events_topic: String,
    /// Batches each publisher keeps for replay.
    pub kv_events_buffer_steps: usize,
    /// Which engine's wire the publishers speak.
    pub kv_events_wire: crate::kv_zmq::Wire,
    /// Which backend's load report the workers imitate (`GetLoads`, `/v1/loads`).
    pub loads_like: LoadsLike,
    /// Settings for replay testing (gRPC workers only).
    pub replay: ReplayConfig,
}

/// Settings for replaying recorded requests through the gateway against
/// this mock. Kept in one struct so that adding a replay flag does not
/// touch every `Config` literal.
#[derive(Debug, Clone, Default)]
pub struct ReplayConfig {
    /// `--capture PATH`: append every gRPC `Generate` request to `PATH`,
    /// one JSON object per line.
    pub capture: Option<PathBuf>,
}

/// Timing flags collected while parsing; resolved into one [`TimingModel`]
/// at the end so flag order does not matter.
#[derive(Default)]
struct TimingFlags {
    /// `polynomial`, `linear` or `fit:<path>`.
    kind: Option<String>,
    prefill_poly: Option<[f64; 3]>,
    decode_poly: Option<[f64; 3]>,
    prefill_tps: Option<f64>,
    decode_base_ms: Option<f64>,
    decode_per_req_ms: Option<f64>,
}

impl TimingFlags {
    /// The model, and the calibration it came from (for its capacity and overhead).
    fn resolve(self) -> Result<(TimingModel, Option<Calibration>), String> {
        if let Some(path) = self.kind.as_deref().and_then(|k| k.strip_prefix("fit:")) {
            let calibration = Calibration::load(path)?;
            return Ok((TimingModel::fitted(&calibration), Some(calibration)));
        }
        let linear_override = self.prefill_tps.is_some()
            || self.decode_base_ms.is_some()
            || self.decode_per_req_ms.is_some();
        let kind = match self.kind.as_deref() {
            Some("polynomial") => "polynomial",
            Some("linear") => "linear",
            Some(other) => {
                return Err(format!(
                    "--timing must be polynomial|linear|fit:<path>, got {other}"
                ))
            }
            None if linear_override => "linear",
            None => "polynomial",
        };
        let model = if kind == "linear" {
            TimingModel::Linear {
                prefill_tps: self.prefill_tps.unwrap_or(8000.0),
                decode_base_ms: self.decode_base_ms.unwrap_or(6.0),
                decode_per_req_ms: self.decode_per_req_ms.unwrap_or(0.35),
            }
        } else {
            TimingModel::Polynomial {
                prefill: self.prefill_poly.unwrap_or(TimingModel::POLY_PREFILL),
                decode: self.decode_poly.unwrap_or(TimingModel::POLY_DECODE),
            }
        };
        Ok((model, None))
    }
}

fn parse_poly(raw: String, flag: &str) -> Result<[f64; 3], String> {
    let parts: Vec<f64> = raw
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("invalid value for {flag}: {raw} (want a,b,c)"))?;
    match parts.as_slice() {
        [a, b, c] => Ok([*a, *b, *c]),
        _ => Err(format!("invalid value for {flag}: {raw} (want a,b,c)")),
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            http_base_port: 9000,
            http_count: 0,
            grpc_base_port: 0,
            grpc_count: 0,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: String::new(),
            gen_delay: Duration::from_millis(0),
            output_tokens: 8,
            realistic: false,
            engine: EngineParams::default(),
            admin_port: None,
            context_length: 32768,
            kv_events_zmq_base_port: None,
            kv_events_replay: true,
            kv_events_topic: String::new(),
            kv_events_buffer_steps: 10_000,
            kv_events_wire: crate::kv_zmq::Wire::Vllm,
            loads_like: LoadsLike::Mock,
            replay: ReplayConfig::default(),
        }
    }
}

impl Config {
    /// Parse the configuration from `std::env::args`, falling back to defaults.
    pub fn from_args() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    /// Parse the configuration from command-line flags, without the program
    /// name.
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut cfg = Self::default();
        let mut timing = TimingFlags::default();
        let mut kv_blocks: Option<u64> = None;
        let (mut kv_tokens_given, mut block_size_given, mut overhead_given) = (false, false, false);

        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--host" => cfg.host = value(&mut args, &flag)?,
                "--http-base-port" => cfg.http_base_port = parse(value(&mut args, &flag)?, &flag)?,
                "--http-count" => cfg.http_count = parse(value(&mut args, &flag)?, &flag)?,
                "--grpc-base-port" => cfg.grpc_base_port = parse(value(&mut args, &flag)?, &flag)?,
                "--grpc-count" => cfg.grpc_count = parse(value(&mut args, &flag)?, &flag)?,
                "--zmq-handshake" => cfg.zmq_handshake = Some(value(&mut args, &flag)?),
                "--zmq-count" => cfg.zmq_count = parse(value(&mut args, &flag)?, &flag)?,
                "--zmq-start-index" => {
                    cfg.zmq_start_index = parse(value(&mut args, &flag)?, &flag)?
                }
                "--model" => cfg.model_id = value(&mut args, &flag)?,
                "--tokenizer" => cfg.tokenizer_path = value(&mut args, &flag)?,
                "--gen-ms" => {
                    cfg.gen_delay = Duration::from_millis(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--output-tokens" => cfg.output_tokens = parse(value(&mut args, &flag)?, &flag)?,
                "--capture" => cfg.replay.capture = Some(value(&mut args, &flag)?.into()),
                "--engine" => {
                    cfg.realistic = match value(&mut args, &flag)?.as_str() {
                        "realistic" => true,
                        "canned" => false,
                        other => {
                            return Err(format!("--engine must be canned|realistic, got {other}"))
                        }
                    }
                }
                "--timing" => timing.kind = Some(value(&mut args, &flag)?),
                "--prefill-poly" => {
                    timing.prefill_poly = Some(parse_poly(value(&mut args, &flag)?, &flag)?);
                }
                "--decode-poly" => {
                    timing.decode_poly = Some(parse_poly(value(&mut args, &flag)?, &flag)?);
                }
                "--prefill-tps" => {
                    timing.prefill_tps = Some(parse(value(&mut args, &flag)?, &flag)?)
                }
                "--decode-base-ms" => {
                    timing.decode_base_ms = Some(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--decode-per-req-ms" => {
                    timing.decode_per_req_ms = Some(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--max-batched-tokens" => {
                    cfg.engine.max_batched_tokens = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--max-running" => cfg.engine.max_running = parse(value(&mut args, &flag)?, &flag)?,
                "--kv-tokens" => {
                    cfg.engine.kv_capacity_tokens = parse(value(&mut args, &flag)?, &flag)?;
                    kv_tokens_given = true;
                }
                "--kv-blocks" => kv_blocks = Some(parse(value(&mut args, &flag)?, &flag)?),
                "--request-overhead-ms" => {
                    cfg.engine.request_overhead_ms = parse(value(&mut args, &flag)?, &flag)?;
                    overhead_given = true;
                }
                "--prefill-first" => {
                    cfg.engine.prefill_first = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--reserve-full-isl" => {
                    cfg.engine.reserve_full_isl = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--context-length" => {
                    cfg.context_length = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--kv-events-zmq-base-port" => {
                    cfg.kv_events_zmq_base_port = Some(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--kv-events-replay" => {
                    cfg.kv_events_replay = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--kv-events-topic" => cfg.kv_events_topic = value(&mut args, &flag)?,
                "--kv-events-buffer-steps" => {
                    cfg.kv_events_buffer_steps = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--kv-events-wire" => cfg.kv_events_wire = value(&mut args, &flag)?.parse()?,
                "--loads-like" => cfg.loads_like = value(&mut args, &flag)?.parse()?,
                "--block-size" => {
                    cfg.engine.block_size = parse(value(&mut args, &flag)?, &flag)?;
                    block_size_given = true;
                }
                "--admin-port" => cfg.admin_port = Some(parse(value(&mut args, &flag)?, &flag)?),
                "--prefix-cache" => {
                    cfg.engine.prefix_cache = parse(value(&mut args, &flag)?, &flag)?
                }
                "-h" | "--help" => return Err(usage()),
                other => return Err(format!("unknown flag: {other}\n\n{}", usage())),
            }
        }

        if cfg.tokenizer_path.is_empty() {
            cfg.tokenizer_path = cfg.model_id.clone();
        }
        let (model, calibration) = timing.resolve()?;
        cfg.engine.timing = model;
        if let Some(c) = calibration {
            // The calibration's block size, capacity and overhead apply unless
            // the flags say otherwise.
            if let (Some(bs), false) = (c.block_size, block_size_given) {
                cfg.engine.block_size = bs;
            }
            if let (Some(tokens), false, None) = (c.kv_capacity_tokens, kv_tokens_given, kv_blocks)
            {
                cfg.engine.kv_capacity_tokens = tokens;
            }
            // The decode fit was measured against the calibration's pool: a
            // smaller pool from the flags changes room, not step time.
            if c.kv_capacity_tokens.is_some() {
                cfg.engine.decode_reference_tokens = c.kv_capacity_tokens;
            }
            if !overhead_given {
                cfg.engine.request_overhead_ms = c.request_overhead_ms;
            }
        }
        if let Some(blocks) = kv_blocks {
            cfg.engine.kv_capacity_tokens = blocks * u64::from(cfg.engine.block_size);
        }
        // `--output-tokens` doubles as the realistic engine's default output
        // length when a request omits `max_tokens`.
        cfg.engine.max_new_default = cfg.output_tokens;
        if cfg.http_count == 0 && cfg.grpc_count == 0 && cfg.zmq_count == 0 {
            return Err(format!(
                "nothing to do: pass --http-count, --grpc-count, and/or --zmq-count\n\n{}",
                usage()
            ));
        }
        if cfg.grpc_count > 0 && cfg.grpc_base_port == 0 {
            return Err("--grpc-base-port is required when --grpc-count > 0".to_string());
        }
        if cfg.zmq_count > 0 && cfg.zmq_handshake.is_none() {
            return Err("--zmq-handshake <ipc://…> is required when --zmq-count > 0".to_string());
        }
        Ok(cfg)
    }
}

impl Config {
    /// The KV-event publisher of worker number `index` (gRPC workers first,
    /// then ZMQ ranks), when publishing is on and the engine is realistic.
    /// `dp_rank` is the rank the publisher stamps on the vLLM wire: 0 for a
    /// gRPC worker, the engine index a ZMQ rank advertises.
    pub(crate) fn kv_zmq_for(
        &self,
        index: u16,
        dp_rank: i32,
    ) -> Option<crate::kv_zmq::KvZmqConfig> {
        if !self.realistic {
            return None;
        }
        let base = self.kv_events_zmq_base_port?;
        let port = base.checked_add(index.checked_mul(2)?)?;
        Some(crate::kv_zmq::KvZmqConfig {
            host: self.host.clone(),
            port,
            replay: self.kv_events_replay,
            topic: self.kv_events_topic.clone(),
            buffer_steps: self.kv_events_buffer_steps,
            dp_rank,
            wire: self.kv_events_wire,
        })
    }
}

fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse<T: std::str::FromStr>(raw: String, flag: &str) -> Result<T, String> {
    raw.parse()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn usage() -> String {
    "mock-worker — multi-port mock HTTP/gRPC inference workers for SMG scale testing\n\n\
     Flags:\n\
       --host <addr>            bind address (default 127.0.0.1)\n\
       --http-base-port <port>  first HTTP port (default 9000)\n\
       --http-count <n>         number of HTTP workers (default 0)\n\
       --grpc-base-port <port>  first gRPC port (required if --grpc-count > 0)\n\
       --grpc-count <n>         number of gRPC workers (default 0)\n\
       --zmq-handshake <addr>   frontend ipc:// handshake addr (required if --zmq-count > 0)\n\
       --zmq-count <n>          number of ZMQ mock EngineCore ranks (default 0)\n\
       --zmq-start-index <n>    engine index of the first ZMQ rank (default 0)\n\
       --model <id>             advertised model id (default mock-model)\n\
       --tokenizer <path>       tokenizer path for gRPC autoload (default = model)\n\
       --gen-ms <ms>            canned per-request latency (default 0)\n\
       --output-tokens <n>      output tokens per request when unspecified (default 8)\n\
       --capture <path>         append each gRPC Generate request to <path> as a JSON line\n\
     \n\
     Realistic engine simulator (vLLM pass loop over a block-level KV pool; opt-in):\n\
       --engine <canned|realistic>  engine mode (default canned)\n\
       --timing <polynomial|linear|fit:<path>>  pass duration model (default polynomial);\n\
                                fit:<path> reads a hardware calibration JSON (polynomials, KV capacity,\n\
                                block size, per-request overhead; flags given explicitly win)\n\
       --prefill-poly <a,b,c>   prefill ms = a + b*T + c*T^2 over uncached tokens T in the pass\n\
                                (default 16.50142,1.518344e-2,4.209989e-7)\n\
       --decode-poly <a,b,c>    decode ms = max(1, a + b*u + c*u^2) over KV utilisation u\n\
                                (default 5.74,54.01,-25.74)\n\
       --prefill-tps <f>        linear model: prefill tokens/sec (default 8000; selects linear)\n\
       --decode-base-ms <f>     linear model: fixed decode-step ms (default 6.0)\n\
       --decode-per-req-ms <f>  linear model: decode ms per running request (default 0.35)\n\
       --max-batched-tokens <n> token budget per pass (default 8192)\n\
       --max-running <n>        max sequences per pass (default 256)\n\
       --kv-tokens <n>          KV cache capacity in tokens (default 524288)\n\
       --kv-blocks <n>          KV cache capacity in blocks (overrides --kv-tokens)\n\
       --request-overhead-ms <f>  fixed per-request latency added to every event of a stream (default 0)\n\
       --block-size <n>         cache block/page size in tokens (default 16)\n\
       --prefix-cache <bool>    enable prefix caching + KV events (default true)\n\
       --prefill-first <bool>   SGLang-style: a pass with prefill runs prefill only (default false)\n\
       --reserve-full-isl <bool>  admit a prompt only with KV room for all of it beyond its cached\n\
                                blocks, head-of-line (vLLM's scheduler_reserve_full_isl; default true)\n\
       --context-length <n>     advertised context length (default 32768)\n\
       --kv-events-zmq-base-port <port>  vLLM-wire ZMQ KV-event publishers: worker i publishes\n\
                                on base+2i (PUB) and replays on base+2i+1 (ROUTER) (default off)\n\
       --kv-events-replay <bool>  serve replay requests on the ROUTER port (default true)\n\
       --kv-events-topic <s>    topic frame (default empty, as vLLM)\n\
       --kv-events-buffer-steps <n>  batches kept for replay (default 10000)\n\
       --kv-events-wire <vllm|sglang>  which engine's publisher to imitate (default vllm)\n\
       --loads-like <mock|vllm>  load report: everything the simulator knows, or only what the\n\
                                vLLM servicer reports (running, waiting, token_usage, maxima)\n\
       --admin-port <port>      process-wide admin API: fleet, request records with the\n\
                                arrival-time oracle, cache dumps, resets (default off)"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(flags: &[&str]) -> Result<Config, String> {
        Config::parse(flags.iter().map(|flag| (*flag).to_string()))
    }

    #[test]
    fn publishers_carry_their_worker_rank_and_port() {
        let cfg = Config {
            realistic: true,
            kv_events_zmq_base_port: Some(19_000),
            ..Config::default()
        };
        // A gRPC worker publishes as rank 0 on base + 2 * index.
        let grpc = cfg
            .kv_zmq_for(1, 0)
            .expect("publisher for a realistic engine");
        assert_eq!((grpc.port, grpc.dp_rank), (19_002, 0));
        // A ZMQ rank publishes under the rank it advertises, not rank 0.
        let rank = cfg.kv_zmq_for(3, 2).expect("publisher for a ZMQ rank");
        assert_eq!((rank.port, rank.dp_rank), (19_006, 2));
        assert!(
            Config::default().kv_zmq_for(0, 0).is_none(),
            "no publisher for a canned worker"
        );
    }

    #[test]
    fn capture_flag_sets_the_capture_path() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert_eq!(cfg.replay.capture, None, "capture is off by default");

        let cfg = parse(&[&grpc[..], &["--capture", "/tmp/generate.jsonl"]].concat())
            .expect("--capture parses");
        assert_eq!(
            cfg.replay.capture,
            Some(PathBuf::from("/tmp/generate.jsonl"))
        );
    }
}
