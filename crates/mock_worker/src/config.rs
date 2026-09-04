//! Runtime configuration for the mock worker fleet, parsed from CLI flags.

use std::time::Duration;

use crate::engine::EngineParams;

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
}

impl Config {
    /// Parse the configuration from `std::env::args`, falling back to defaults.
    pub fn from_args() -> Result<Self, String> {
        let mut cfg = Self {
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
        };

        let mut args = std::env::args().skip(1);
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
                    cfg.zmq_start_index = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--model" => cfg.model_id = value(&mut args, &flag)?,
                "--tokenizer" => cfg.tokenizer_path = value(&mut args, &flag)?,
                "--gen-ms" => {
                    cfg.gen_delay = Duration::from_millis(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--output-tokens" => cfg.output_tokens = parse(value(&mut args, &flag)?, &flag)?,
                "--engine" => {
                    cfg.realistic = match value(&mut args, &flag)?.as_str() {
                        "realistic" => true,
                        "canned" => false,
                        other => {
                            return Err(format!("--engine must be canned|realistic, got {other}"))
                        }
                    }
                }
                "--prefill-tps" => cfg.engine.prefill_tps = parse(value(&mut args, &flag)?, &flag)?,
                "--decode-base-ms" => {
                    cfg.engine.decode_base_ms = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--decode-per-req-ms" => {
                    cfg.engine.decode_per_req_ms = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--prefill-chunk" => {
                    cfg.engine.prefill_chunk_tokens = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--max-running" => cfg.engine.max_running = parse(value(&mut args, &flag)?, &flag)?,
                "--kv-tokens" => {
                    cfg.engine.kv_capacity_tokens = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--block-size" => cfg.engine.block_size = parse(value(&mut args, &flag)?, &flag)?,
                "--prefix-cache" => {
                    cfg.engine.prefix_cache = parse(value(&mut args, &flag)?, &flag)?;
                }
                "-h" | "--help" => return Err(usage()),
                other => return Err(format!("unknown flag: {other}\n\n{}", usage())),
            }
        }

        if cfg.tokenizer_path.is_empty() {
            cfg.tokenizer_path.clone_from(&cfg.model_id);
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
     \n\
     Realistic engine simulator (continuous batching; opt-in):\n\
       --engine <canned|realistic>  engine mode (default canned)\n\
       --prefill-tps <f>        prefill throughput, tokens/sec (default 8000)\n\
       --decode-base-ms <f>     fixed decode-step latency, ms (default 6.0)\n\
       --decode-per-req-ms <f>  added decode-step latency per running req (default 0.35)\n\
       --prefill-chunk <n>      max prompt tokens prefilled per step (default 2048)\n\
       --max-running <n>        max concurrent running requests (default 256)\n\
       --kv-tokens <n>          KV cache capacity in tokens (default 524288)\n\
       --block-size <n>         cache block/page size in tokens (default 16)\n\
       --prefix-cache <bool>    enable prefix caching + KV events (default true)"
        .to_string()
}
