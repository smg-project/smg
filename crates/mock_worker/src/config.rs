//! Runtime configuration for the mock worker fleet, parsed from CLI flags.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::engine::{Calibration, EngineParams, KvCacheGroup, LoadsLike, OutputSeed, TimingModel};

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
    /// Data-parallel ranks per gRPC worker (realistic engines only): the
    /// worker advertises `dp_size`, runs one engine per rank, places a request
    /// on its `data_parallel_rank` when pinned and on the rank with the fewest
    /// requests of its own otherwise, and reports one `GetLoads` entry and one
    /// KV-event stream per rank. 1 (the default) is a single-rank engine.
    pub grpc_dp_size: u16,
    /// The scheduler proto the gRPC workers serve (`--grpc-wire`). The
    /// gateway tells a worker's runtime by the proto whose health check it
    /// answers, so a worker serves exactly one.
    pub grpc_wire: SchedulerWire,
    /// Frontend handshake `ipc://` address ZMQ mock engines connect to (they
    /// dial the SMG/test frontend, which binds the sockets).
    pub zmq_handshake: Option<String>,
    /// Number of ZMQ mock EngineCore ranks to start (each a DP rank).
    pub zmq_count: u16,
    /// Engine index of the first ZMQ rank; ranks use `[start, start+count)`.
    pub zmq_start_index: u32,
    /// A ZMQ rank's step period in milliseconds. When set, the rank sends an
    /// engine's steps: one message per period carrying the next output of
    /// every request that has one; a request that produced faster than the
    /// period waits its turn, one output per step. Zero, the default, sends
    /// one message per token as it comes, the shape the tests have read so
    /// far.
    pub zmq_step_ms: u64,
    /// Whether the ZMQ ranks of this process step as one lockstep
    /// data-parallel group (`--zmq-lockstep`): they advertise the group's
    /// size at the handshake, sync every step, park together when no rank
    /// has work (each reports the drained wave) and wake on the frontend's
    /// wave start, the engine side of the wave protocol the gateway's
    /// connector drives. Off, the default, every rank is an independent
    /// engine.
    pub zmq_lockstep: bool,
    /// How long after reading an add a ZMQ rank's loop sees it, in
    /// milliseconds (`--zmq-add-delay-ms`): an engine's input thread decodes
    /// and preprocesses a request before its loop does, a wave start it hands
    /// over at once. Zero, the default, hands an add over as it is read.
    pub zmq_add_delay_ms: u64,
    /// Model id advertised by every worker (one model, many replicas).
    pub model_id: String,
    /// Tokenizer path advertised by gRPC workers (for gateway autoload).
    pub tokenizer_path: String,
    /// The model's sampling defaults as a gRPC worker's model info reports
    /// them: the compact JSON of the `generation_config.json` keys an engine
    /// servicer advertises (`temperature`, `top_p`, `top_k`, `min_p`,
    /// `repetition_penalty`), read from the model directory at startup; empty
    /// when the model is not a directory or its config declares none.
    pub default_sampling_params_json: String,
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
    /// Extra `server_args` entries the gRPC `GetServerInfo` advertises, as
    /// string values (e.g. `rl.control_url`). Empty leaves `server_args` unset.
    pub server_args: BTreeMap<String, String>,
    /// Weight version stamped on every gRPC generate chunk and completion;
    /// `None` leaves the field unset, like an engine that predates it.
    pub weight_version: Option<String>,
    /// Port of the process-wide admin API (fleet, request records, cache
    /// dumps, resets); off when `None`.
    pub admin_port: Option<u16>,
    /// How long a termination signal (SIGTERM or SIGINT) lets the requests
    /// in flight finish before the process exits regardless; no worker
    /// accepts new work from the signal on.
    pub drain_timeout: Duration,
    /// Context length advertised to the gateway (`max_context_length`,
    /// `max_req_input_len`, `max_model_len`).
    pub context_length: u32,
    /// Largest gRPC message a worker decodes or encodes, in bytes. A gateway
    /// sends a prompt's token ids as one `Generate`, and a million of them are
    /// a few megabytes; tonic's own default (4 MiB) refuses that, while the
    /// engine servicers run without a limit, so none is the default.
    pub grpc_max_message_bytes: usize,
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
    /// What the load record says of the mock's wire: `true` imitates an
    /// engine that reports every scheduler step (vLLM's stats-only step
    /// batches), `false` (the default) a wire that carries token batches
    /// only, whose silence the gateway judges against its prefill bound.
    pub step_reports: bool,
    /// The HTTP or gRPC workers' role in a disaggregated fleet
    /// (`--worker-type`): a prefill worker answers every request with the
    /// prompt side's completion alone and binds its bootstrap port, a decode
    /// worker generates as a regular one, an encode worker (the vLLM gRPC
    /// wire alone) answers the encoder's completion with the media identity
    /// and a transfer handle per item; each advertises the role in its
    /// server info as the engines do (`disaggregation_mode`, or `kv_role`
    /// and `ec_role` behind their connectors on the vLLM wire). One process
    /// gives the role to one wire's workers; the ZMQ workers take none.
    /// `None` is a regular worker.
    pub role: Option<Role>,
    /// First bootstrap port of the prefill or encode workers
    /// (`--bootstrap-base-port`): worker `i` binds `base + i`, where a
    /// partner's engine would rendezvous for the handoff, and advertises it
    /// (`disaggregation_bootstrap_port`); the gateway probes it. `None` binds
    /// and advertises nothing.
    pub bootstrap_base_port: Option<u16>,
    /// The pairing protocol the process environment names
    /// (`SMG_PAIRING_PROTOCOL`), advertised in the gRPC workers' server args
    /// as the engine servicers advertise theirs (`pairing_protocol`): the
    /// gateway pairs a prefill and a decode worker on it alone when set.
    pub pairing_protocol: Option<String>,
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
    /// `--script PATH`: a JSONL of scripted answers keyed by the prompt's
    /// token ids, served on every wire ahead of the engine
    /// ([`crate::script`]).
    pub script: Option<PathBuf>,
}

/// A worker's role in a disaggregated fleet (`--worker-type`, on the HTTP
/// and the gRPC workers; the gateway's worker API spells it `worker_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Answers every request with the prompt side's completion alone and
    /// binds its bootstrap port: the side a gateway in PD mode sends the
    /// prompt to first.
    Prefill,
    /// Generates as a regular worker: the side whose output the client reads.
    Decode,
    /// Answers every request with the encoder's completion alone (no output,
    /// the media identity, one encoder-cache transfer handle per media item)
    /// and binds its bootstrap port when given one: the side a gateway in
    /// encode-prefill-decode mode sends a request's media to first. The vLLM
    /// wire alone serves it (its scheduler proto carries the encoder-cache
    /// handoff); the TokenSpeed encoder is a separate service this mock does
    /// not serve, and the SGLang wire and the HTTP workers have no encode
    /// role.
    Encode,
}

impl Role {
    /// The role as the engines' server args spell it (`disaggregation_mode`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
        }
    }

    /// Whether this role binds a bootstrap port when given one: the sides a
    /// partner's engine comes to for a handoff (a decode partner to the
    /// prefill side's, a prefill partner to the encoder's).
    pub fn hosts_rendezvous(self) -> bool {
        matches!(self, Self::Prefill | Self::Encode)
    }
}

impl std::str::FromStr for Role {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "encode" => Ok(Self::Encode),
            other => Err(format!(
                "--worker-type must be prefill|decode|encode, got {other}"
            )),
        }
    }
}

/// The scheduler proto a gRPC worker serves (`--grpc-wire`): one of the
/// three engine wires the gateway's gRPC client speaks. The canned and
/// scripted answers, the realistic engine, the rank placement and the roles
/// are the same behind each; the wire module shapes the messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerWire {
    #[default]
    TokenSpeed,
    Vllm,
    Sglang,
}

impl SchedulerWire {
    /// The wire as the flag spells it, and as the gateway names the runtime.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TokenSpeed => "tokenspeed",
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

impl std::str::FromStr for SchedulerWire {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "tokenspeed" => Ok(Self::TokenSpeed),
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            other => Err(format!(
                "--grpc-wire must be tokenspeed|vllm|sglang, got {other}"
            )),
        }
    }
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
            grpc_dp_size: 1,
            grpc_wire: SchedulerWire::TokenSpeed,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            zmq_step_ms: 0,
            zmq_lockstep: false,
            zmq_add_delay_ms: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: String::new(),
            default_sampling_params_json: String::new(),
            gen_delay: Duration::from_millis(0),
            output_tokens: 8,
            realistic: false,
            engine: EngineParams::default(),
            server_args: BTreeMap::new(),
            weight_version: None,
            admin_port: None,
            drain_timeout: Duration::from_secs(5),
            context_length: 32768,
            grpc_max_message_bytes: usize::MAX,
            kv_events_zmq_base_port: None,
            kv_events_replay: true,
            kv_events_topic: String::new(),
            kv_events_buffer_steps: 10_000,
            kv_events_wire: crate::kv_zmq::Wire::Vllm,
            loads_like: LoadsLike::Mock,
            step_reports: false,
            role: None,
            bootstrap_base_port: None,
            pairing_protocol: None,
            replay: ReplayConfig::default(),
        }
    }
}

impl Config {
    /// Parse the configuration from `std::env::args`, falling back to
    /// defaults, with what the environment names (`SMG_PAIRING_PROTOCOL`).
    pub fn from_args() -> Result<Self, String> {
        let mut cfg = Self::parse(std::env::args().skip(1))?;
        cfg.pairing_protocol = std::env::var("SMG_PAIRING_PROTOCOL")
            .ok()
            .map(|protocol| protocol.trim().to_string())
            .filter(|protocol| !protocol.is_empty());
        Ok(cfg)
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
                "--grpc-dp-size" => cfg.grpc_dp_size = parse(value(&mut args, &flag)?, &flag)?,
                "--grpc-wire" => cfg.grpc_wire = value(&mut args, &flag)?.parse()?,
                "--zmq-handshake" => cfg.zmq_handshake = Some(value(&mut args, &flag)?),
                "--zmq-count" => cfg.zmq_count = parse(value(&mut args, &flag)?, &flag)?,
                "--zmq-step-ms" => cfg.zmq_step_ms = parse(value(&mut args, &flag)?, &flag)?,
                "--zmq-start-index" => {
                    cfg.zmq_start_index = parse(value(&mut args, &flag)?, &flag)?
                }
                "--zmq-lockstep" => cfg.zmq_lockstep = true,
                "--zmq-add-delay-ms" => {
                    cfg.zmq_add_delay_ms = parse(value(&mut args, &flag)?, &flag)?
                }
                "--model" => cfg.model_id = value(&mut args, &flag)?,
                "--tokenizer" => cfg.tokenizer_path = value(&mut args, &flag)?,
                "--gen-ms" => {
                    cfg.gen_delay = Duration::from_millis(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--output-tokens" => cfg.output_tokens = parse(value(&mut args, &flag)?, &flag)?,
                "--output-prefix-ids" => {
                    cfg.engine.output_prefix = parse_ids(value(&mut args, &flag)?, &flag)?;
                }
                "--capture" => cfg.replay.capture = Some(value(&mut args, &flag)?.into()),
                "--script" => cfg.replay.script = Some(value(&mut args, &flag)?.into()),
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
                "--output-seed" => {
                    cfg.engine.output_seed = match value(&mut args, &flag)?.as_str() {
                        "request-id" => OutputSeed::RequestId,
                        "prompt" => OutputSeed::Prompt,
                        other => {
                            return Err(format!(
                                "--output-seed must be request-id|prompt, got {other}"
                            ))
                        }
                    };
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
                "--grpc-max-message-bytes" => {
                    cfg.grpc_max_message_bytes = parse(value(&mut args, &flag)?, &flag)?;
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
                "--kv-cache-groups" => {
                    cfg.engine.kv_cache_groups =
                        parse_kv_cache_groups(value(&mut args, &flag)?, &flag)?;
                }
                "--loads-like" => cfg.loads_like = value(&mut args, &flag)?.parse()?,
                "--step-reports" => cfg.step_reports = parse(value(&mut args, &flag)?, &flag)?,
                "--block-size" => {
                    cfg.engine.block_size = parse(value(&mut args, &flag)?, &flag)?;
                    block_size_given = true;
                }
                "--admin-port" => cfg.admin_port = Some(parse(value(&mut args, &flag)?, &flag)?),
                "--worker-type" => cfg.role = Some(value(&mut args, &flag)?.parse()?),
                "--bootstrap-base-port" => {
                    cfg.bootstrap_base_port = Some(parse(value(&mut args, &flag)?, &flag)?);
                }
                "--drain-timeout-secs" => {
                    let secs: f64 = parse(value(&mut args, &flag)?, &flag)?;
                    cfg.drain_timeout =
                        Duration::try_from_secs_f64(secs).map_err(|e| format!("{flag}: {e}"))?;
                }
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
        cfg.default_sampling_params_json = sampling_defaults_json(Path::new(&cfg.model_id));
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
        if cfg.grpc_dp_size == 0 {
            return Err("--grpc-dp-size must be at least 1".to_string());
        }
        if cfg.grpc_dp_size > 1 && !cfg.realistic {
            return Err("--grpc-dp-size above 1 needs --engine realistic".to_string());
        }
        if cfg.kv_events_wire == crate::kv_zmq::Wire::Sglang && cfg.engine.kv_cache_groups.len() > 1
        {
            return Err(
                "--kv-cache-groups names several groups, but the SGLang wire (--kv-events-wire \
                 sglang) publishes one cache group"
                    .to_string(),
            );
        }
        if cfg.grpc_dp_size > 1 && cfg.kv_events_zmq_base_port.is_some() {
            return Err(
                "--grpc-dp-size above 1 does not publish over --kv-events-zmq-base-port \
                 (the publisher ports are per worker, so it would carry rank 0 alone)"
                    .to_string(),
            );
        }
        if cfg.zmq_count > 0 && cfg.zmq_handshake.is_none() {
            return Err("--zmq-handshake <ipc://…> is required when --zmq-count > 0".to_string());
        }
        if cfg.zmq_lockstep && cfg.zmq_count < 2 {
            return Err(
                "--zmq-lockstep needs --zmq-count of 2 or more: a lockstep group is several ranks"
                    .to_string(),
            );
        }
        if cfg.zmq_lockstep && cfg.zmq_step_ms == 0 {
            return Err(
                "--zmq-lockstep needs --zmq-step-ms above 0: the group syncs once per step"
                    .to_string(),
            );
        }
        if cfg.replay.capture.is_some() && cfg.grpc_wire != SchedulerWire::TokenSpeed {
            return Err(format!(
                "--capture records the request's fields as the tokenspeed wire names them: it \
                 needs --grpc-wire tokenspeed, not {}",
                cfg.grpc_wire.as_str()
            ));
        }
        if cfg.role.is_some() {
            if cfg.zmq_count > 0 {
                return Err(
                    "--worker-type is the HTTP and gRPC workers' role: the ZMQ workers of this \
                     mock take none"
                        .to_string(),
                );
            }
            if cfg.http_count > 0 && cfg.grpc_count > 0 {
                return Err(
                    "--worker-type gives one wire's workers the role: pass --http-count or \
                     --grpc-count, not both (worker i of either wire would bind the same \
                     bootstrap port)"
                        .to_string(),
                );
            }
        }
        if cfg.role == Some(Role::Encode) && cfg.http_count > 0 {
            return Err(
                "--worker-type encode answers the vLLM wire's encoder-cache handoff: the HTTP \
                 workers have no encode role (pass --grpc-count with --grpc-wire vllm)"
                    .to_string(),
            );
        }
        if cfg.role == Some(Role::Encode) && cfg.grpc_wire != SchedulerWire::Vllm {
            return Err(format!(
                "--worker-type encode answers the vLLM wire's encoder-cache handoff: pass \
                 --grpc-wire vllm, not {} (the TokenSpeed encoder is a separate service this \
                 mock does not serve; the SGLang wire has no encode role)",
                cfg.grpc_wire.as_str()
            ));
        }
        if cfg.bootstrap_base_port.is_some() && !cfg.role.is_some_and(Role::hosts_rendezvous) {
            return Err(
                "--bootstrap-base-port needs --worker-type prefill or encode: only the side a \
                 partner comes to binds a bootstrap port"
                    .to_string(),
            );
        }
        Ok(cfg)
    }
}

impl Config {
    /// The bootstrap port of worker number `index` (its offset from its
    /// wire's base port, `--grpc-base-port` or `--http-base-port`):
    /// `--bootstrap-base-port` plus the offset for a
    /// prefill or encode worker given one, `None` for every other worker.
    pub fn bootstrap_port_for(&self, index: u16) -> Option<u16> {
        if !self.role.is_some_and(Role::hosts_rendezvous) {
            return None;
        }
        self.bootstrap_base_port?.checked_add(index)
    }

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

/// A comma-separated list of token ids.
fn parse_ids(raw: String, flag: &str) -> Result<Vec<u32>, String> {
    raw.split(',')
        .map(|id| {
            id.trim()
                .parse()
                .map_err(|_| format!("invalid value for {flag}: {raw}"))
        })
        .collect()
}

/// `kind[:window],...`: the KV cache groups a realistic engine publishes,
/// exactly one of them its main-attention group, at any index (`full_attention`
/// alone by default; a hybrid engine adds `sliding_window:<tokens>` or `mamba`
/// groups, and may publish its main group behind them).
fn parse_kv_cache_groups(raw: String, flag: &str) -> Result<Vec<KvCacheGroup>, String> {
    let invalid = || format!("invalid value for {flag}: {raw}");
    let groups = raw
        .split(',')
        .map(|item| {
            let item = item.trim();
            let (kind, window) = item.split_once(':').unwrap_or((item, ""));
            let windowed = KvCacheGroup::WINDOW_KINDS.contains(&kind);
            if !windowed && !KvCacheGroup::UNWINDOWED_KINDS.contains(&kind) {
                return Err(format!(
                    "{flag}: unknown cache group kind {kind:?} (one of {})",
                    KvCacheGroup::KINDS_HELP
                ));
            }
            let sliding_window: u32 = if window.is_empty() {
                0
            } else {
                window.parse().map_err(|_| invalid())?
            };
            if windowed && sliding_window == 0 {
                return Err(format!(
                    "{flag}: {kind} needs its window in tokens, as {kind}:<tokens>"
                ));
            }
            if !windowed && !window.is_empty() {
                return Err(format!("{flag}: {kind} has no window"));
            }
            Ok(KvCacheGroup {
                kind: kind.to_string(),
                sliding_window,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if groups.is_empty() || groups.len() > 8 {
        return Err(format!("{flag} takes one to eight groups"));
    }
    let main = groups
        .iter()
        .filter(|group| KvCacheGroup::MAIN_KINDS.contains(&group.kind.as_str()))
        .count();
    if main > 1 {
        return Err(format!(
            "{flag}: at most one group is the main-attention group (one of {}), at any index; got {main}",
            KvCacheGroup::MAIN_KINDS.join(", ")
        ));
    }
    // A model without a main-attention group (every attention layer a
    // sliding window) starts with its window group.
    if main == 0 && !KvCacheGroup::WINDOW_KINDS.contains(&groups[0].kind.as_str()) {
        return Err(format!(
            "{flag}: a model without a main-attention group starts with a window group (one of {}), not {}",
            KvCacheGroup::WINDOW_KINDS.join(", "),
            groups[0].kind
        ));
    }
    Ok(groups)
}

/// The flags, as `--help` prints them.
pub fn usage() -> String {
    "mock-worker — multi-port mock HTTP/gRPC inference workers for SMG scale testing\n\n\
     Flags:\n\
       --host <addr>            bind address (default 127.0.0.1)\n\
       --http-base-port <port>  first HTTP port (default 9000)\n\
       --http-count <n>         number of HTTP workers (default 0)\n\
       --grpc-base-port <port>  first gRPC port (required if --grpc-count > 0)\n\
       --grpc-count <n>         number of gRPC workers (default 0)\n\
       --grpc-dp-size <n>       data-parallel ranks per gRPC worker (default 1; realistic engine only):\n\
                                dp_size advertised, one engine per rank, a pinned request runs on its\n\
                                rank, an unpinned one on the rank with the fewest of its own requests,\n\
                                GetLoads and the KV-event stream report per rank; no\n\
                                --kv-events-zmq-base-port publisher (it would carry rank 0 alone)\n\
       --grpc-wire <tokenspeed|vllm|sglang>  the scheduler proto the gRPC workers serve (default\n\
                                tokenspeed): the same answers, engine, ranks and roles behind each;\n\
                                the gateway detects the runtime from it (--capture: tokenspeed only)\n\
       --grpc-max-message-bytes <n>  largest gRPC message a worker decodes or encodes, in bytes\n\
                                (default none, as the engine servicers; tonic's own default is 4 MiB)\n\
       --zmq-handshake <addr>   frontend ipc:// handshake addr (required if --zmq-count > 0)\n\
       --zmq-count <n>          number of ZMQ mock EngineCore ranks (default 0)\n\
       --zmq-start-index <n>    engine index of the first ZMQ rank (default 0)\n\
       --zmq-step-ms <ms>       the ranks' step period: one output message per period with the next\n\
                                token of every running request (an engine's step; a faster request\n\
                                waits its turn); 0 sends one message per token (default 0)\n\
       --zmq-lockstep           the ranks step as one lockstep data-parallel group: they sync every\n\
                                step, park together when idle and wake on the frontend's wave start\n\
                                (needs --zmq-count of 2 or more and --zmq-step-ms); off, every rank\n\
                                is an independent engine\n\
       --zmq-add-delay-ms <ms>  how long after reading an add a rank's loop sees it, as an engine's\n\
                                input thread preprocesses a request first (a wave start lands at\n\
                                once; default 0)\n\
       --model <id|dir>         advertised model id, or a model directory whose\n\
                                generation_config.json gives the sampling defaults gRPC\n\
                                workers report (default mock-model)\n\
       --tokenizer <path>       tokenizer path for gRPC autoload (default = model)\n\
       --gen-ms <ms>            canned per-request latency (default 0)\n\
       --output-tokens <n>      output tokens per request when unspecified (default 8)\n\
       --output-prefix-ids <a,b,..>  token ids every output starts with, before the synthetic ids\n\
                                (a stream shaped for a gateway that parses the engine's tokens: the\n\
                                canned gRPC/ZMQ workers and the realistic engine; default none)\n\
       --capture <path>         append each gRPC Generate request to <path> as a JSON line\n\
       --script <path>          scripted answers (JSONL keyed by the prompt's token ids), served on\n\
                                every wire ahead of the engine; the admin API enqueues more\n\
     \n\
     Realistic engine simulator (vLLM pass loop over a block-level KV pool; opt-in):\n\
       --engine <canned|realistic>  engine mode (default canned)\n\
       --output-seed <request-id|prompt>  synthetic output identity (default request-id);\n\
                                prompt ignores transport IDs for repeatable cross-router output\n\
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
       --kv-cache-groups <kind[:window],...>  the KV cache groups a realistic engine publishes, exactly
                                one of them a main-attention kind, at any index (default full_attention),
                                one to eight of full_attention, mla_attention, sink_full_attention,
                                sliding_window:<tokens>, sliding_window_mla:<tokens> and mamba: a
                                window group's stores name the window tail and a Mamba group's the
                                state block at the prompt's replay boundary, as vLLM 0.31.0 retains
                                them by default (one group on the SGLang wire)\n\
       --loads-like <mock|vllm>  load report: everything the simulator knows, or only what the\n\
                                vLLM servicer reports (running, waiting, token_usage, maxima)\n\
       --step-reports <bool>    the load record says the wire reports every scheduler step (as\n\
                                vLLM's stats-only step batches do) instead of token batches only\n\
                                (default false): decides which bound the gateway judges the\n\
                                engine's silence against\n\
       --admin-port <port>      process-wide admin API: fleet, request records with the\n\
                                arrival-time oracle, cache dumps, resets (default off)\n\
       --worker-type <prefill|decode|encode>  the HTTP or gRPC workers' role in a disaggregated\n\
                                fleet (default none; one wire per process, the ZMQ workers take\n\
                                none): a prefill worker answers every request with the\n\
                                prompt side's completion alone (no output, no script), a decode\n\
                                worker generates as a regular one, an encode worker (the vllm\n\
                                gRPC wire alone) answers the encoder's completion with the media\n\
                                identity and a transfer handle per item; each advertises the\n\
                                role in its server info (disaggregation_mode; kv_role and\n\
                                ec_role behind their connectors on the vllm wire)\n\
       --bootstrap-base-port <port>  first bootstrap port of the prefill or encode workers: worker\n\
                                i binds base+i, accepts and closes each connection (the gateway's\n\
                                probe) and advertises it (disaggregation_bootstrap_port)\n\
                                (default none)\n\
       --drain-timeout-secs <s> on SIGTERM or SIGINT, how long the requests in flight may\n\
                                finish before the process exits regardless (default 5)\n\
     \n\
     Environment:\n\
       SMG_PAIRING_PROTOCOL=<name>  advertised by every gRPC worker as pairing_protocol, as the engine\n\
                                servicers do: the gateway pairs a prefill and a decode that name the\n\
                                same one (default none advertised)\n"
        .to_string()
}

/// The `generation_config.json` keys an engine servicer reports as the
/// model's sampling defaults, in the order they are written.
const SAMPLING_DEFAULT_KEYS: [&str; 5] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
];

/// The sampling defaults of the model at `model_dir`, as an engine servicer
/// reports them: the compact JSON of `SAMPLING_DEFAULT_KEYS` present and
/// non-null in `generation_config.json`. Empty when `model_dir` is not a
/// directory (a model id, a file), has no such config, the file does not
/// parse, or it declares none of the keys.
pub fn sampling_defaults_json(model_dir: &Path) -> String {
    if !model_dir.is_dir() {
        return String::new();
    }
    let path = model_dir.join("generation_config.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        // A model without the file declares no defaults; anything else that
        // keeps the file from being read is said, so a bed with a broken
        // fixture does not read like such a model.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return String::new(),
        Err(e) => {
            tracing::warn!("{}: not read, no sampling defaults: {e}", path.display());
            return String::new();
        }
    };
    let config = match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(serde_json::Value::Object(config)) => config,
        Ok(_) => {
            tracing::warn!(
                "{}: not a JSON object, no sampling defaults",
                path.display()
            );
            return String::new();
        }
        Err(e) => {
            tracing::warn!(
                "{}: does not parse, no sampling defaults: {e}",
                path.display()
            );
            return String::new();
        }
    };
    let defaults: serde_json::Map<String, serde_json::Value> = SAMPLING_DEFAULT_KEYS
        .iter()
        .filter_map(|&key| {
            config
                .get(key)
                .filter(|value| !value.is_null())
                .map(|value| (key.to_string(), value.clone()))
        })
        .collect();
    if defaults.is_empty() {
        String::new()
    } else {
        serde_json::Value::Object(defaults).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(flags: &[&str]) -> Result<Config, String> {
        Config::parse(flags.iter().map(|flag| (*flag).to_string()))
    }

    /// `--kv-cache-groups` names each group's kind and window in order, the
    /// main-attention group one of them at any index; the SGLang wire has no
    /// cache groups, so several are refused there.
    #[test]
    fn the_kv_cache_groups_flag_names_each_group_and_stays_single_on_the_sglang_wire() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let with = |flags: &[&str]| parse(&[&grpc[..], flags].concat());
        let cfg = with(&[
            "--kv-cache-groups",
            "full_attention,sliding_window:128,mamba",
        ])
        .expect("three groups");
        assert_eq!(
            cfg.engine.kv_cache_groups,
            vec![
                KvCacheGroup::full_attention(),
                KvCacheGroup {
                    kind: "sliding_window".to_string(),
                    sliding_window: 128,
                },
                KvCacheGroup {
                    kind: "mamba".to_string(),
                    sliding_window: 0,
                },
            ]
        );
        let err = with(&[
            "--kv-events-wire",
            "sglang",
            "--kv-cache-groups",
            "full_attention,mamba",
        ])
        .expect_err("one group on the SGLang wire");
        assert!(err.contains("one cache group"), "{err}");
        assert!(
            with(&[
                "--kv-events-wire",
                "sglang",
                "--kv-cache-groups",
                "full_attention"
            ])
            .is_ok(),
            "one group is the SGLang wire's shape"
        );
        // A model without a main-attention group, and the main group at any
        // index: behind a window group, or last.
        for spec in [
            "sliding_window:64",
            "sliding_window:64,mamba",
            "sliding_window:64,full_attention",
            "mamba,full_attention",
            "mamba,mla_attention,sliding_window:32",
        ] {
            let cfg = with(&["--kv-cache-groups", spec]).expect(spec);
            assert_eq!(
                cfg.engine.kv_cache_groups.len(),
                spec.matches(',').count() + 1,
                "{spec}"
            );
        }
        for (spec, reason) in [
            ("sliding_window", "needs its window"),
            ("full_attention,sliding_window_mla:0", "needs its window"),
            ("mamba:128", "has no window"),
            ("full_attention:64", "has no window"),
            ("swa:128", "unknown cache group kind"),
            ("sliding-window:128", "unknown cache group kind"),
            ("mamba", "starts with a window group"),
            ("mamba,sliding_window:64", "starts with a window group"),
            (
                "full_attention,mla_attention",
                "at most one group is the main-attention group",
            ),
        ] {
            let err = with(&["--kv-cache-groups", spec]).expect_err(spec);
            assert!(err.contains(reason), "{spec}: {err}");
        }
    }

    /// The gRPC workers serve the tokenspeed wire unless `--grpc-wire` names
    /// another; a wire the gateway's client does not speak is refused, and
    /// the last flag wins, as a class's own argument after a lane's.
    #[test]
    fn the_grpc_wire_defaults_to_tokenspeed_and_takes_the_three_names() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        assert_eq!(
            parse(&grpc).expect("defaults").grpc_wire,
            SchedulerWire::TokenSpeed
        );
        for (name, wire) in [
            ("tokenspeed", SchedulerWire::TokenSpeed),
            ("vllm", SchedulerWire::Vllm),
            ("sglang", SchedulerWire::Sglang),
        ] {
            let cfg = parse(&[&grpc[..], &["--grpc-wire", name]].concat()).expect(name);
            assert_eq!(cfg.grpc_wire, wire);
            assert_eq!(cfg.grpc_wire.as_str(), name);
        }
        let last = parse(&[&grpc[..], &["--grpc-wire", "vllm", "--grpc-wire", "sglang"]].concat())
            .expect("two flags");
        assert_eq!(last.grpc_wire, SchedulerWire::Sglang);
        let refused =
            parse(&[&grpc[..], &["--grpc-wire", "trtllm"]].concat()).expect_err("an unknown wire");
        assert!(
            refused.contains("--grpc-wire must be tokenspeed|vllm|sglang, got trtllm"),
            "{refused}"
        );
    }

    /// The capture file's keys are the tokenspeed request's field names, so a
    /// capture on another wire is refused rather than misnamed.
    #[test]
    fn a_capture_needs_the_tokenspeed_wire() {
        let flags = [
            "--grpc-base-port",
            "19000",
            "--grpc-count",
            "1",
            "--capture",
            "/tmp/requests.jsonl",
        ];
        assert!(parse(&flags).is_ok(), "the default wire captures");
        let refused = parse(&[&flags[..], &["--grpc-wire", "vllm"]].concat())
            .expect_err("a capture on the vllm wire");
        assert!(
            refused.contains("--capture") && refused.contains("not vllm"),
            "{refused}"
        );
    }

    #[test]
    fn sampling_defaults_are_the_generation_configs_advertised_keys() {
        let dir = tempfile::tempdir().expect("a model directory");
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"bos_token_id": 1, "do_sample": true, "temperature": 0.7, "top_p": 0.8,
                "top_k": 20, "min_p": null, "repetition_penalty": 1.05, "max_new_tokens": 32}"#,
        )
        .expect("write the config");
        assert_eq!(
            sampling_defaults_json(dir.path()),
            r#"{"temperature":0.7,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}"#
        );
        // The parsed configuration reads the directory it was given as the model.
        let flags = ["--grpc-base-port", "19000", "--grpc-count", "1", "--model"];
        let cfg = parse(&[&flags[..], &[dir.path().to_str().expect("utf-8 path")]].concat())
            .expect("a model directory");
        assert_eq!(
            cfg.default_sampling_params_json,
            r#"{"temperature":0.7,"top_p":0.8,"top_k":20,"repetition_penalty":1.05}"#
        );
    }

    #[test]
    fn sampling_defaults_are_empty_without_a_directory_a_config_or_the_keys() {
        assert_eq!(sampling_defaults_json(Path::new("owner/model-id")), "");
        let dir = tempfile::tempdir().expect("a model directory");
        assert_eq!(sampling_defaults_json(dir.path()), "", "no config file");
        let config = dir.path().join("generation_config.json");
        std::fs::write(
            &config,
            r#"{"bos_token_id": 1, "max_new_tokens": 32, "top_k": null}"#,
        )
        .expect("write the config");
        assert_eq!(sampling_defaults_json(dir.path()), "", "none of the keys");
        std::fs::write(&config, "not json").expect("write the config");
        assert_eq!(
            sampling_defaults_json(dir.path()),
            "",
            "a config that does not parse"
        );
        assert_eq!(
            parse(&["--grpc-base-port", "19000", "--grpc-count", "1"])
                .expect("defaults")
                .default_sampling_params_json,
            "",
            "the default model id is no directory"
        );
    }

    #[test]
    fn output_seed_is_explicit_and_rejects_unknown_modes() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        assert_eq!(
            parse(&grpc).expect("defaults").engine.output_seed,
            OutputSeed::RequestId
        );
        for (flag, expected) in [
            ("prompt", OutputSeed::Prompt),
            ("request-id", OutputSeed::RequestId),
        ] {
            let cfg =
                parse(&[&grpc[..], &["--output-seed", flag]].concat()).expect("valid seed mode");
            assert_eq!(cfg.engine.output_seed, expected);
        }
        assert!(parse(&[&grpc[..], &["--output-seed", "random"]].concat())
            .expect_err("unknown seed mode")
            .contains("--output-seed must be request-id|prompt"));
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
    fn a_worker_of_several_ranks_has_no_zmq_publisher() {
        let grpc = [
            "--grpc-base-port",
            "19000",
            "--grpc-count",
            "1",
            "--engine",
            "realistic",
            "--grpc-dp-size",
            "2",
        ];
        assert_eq!(
            parse(&grpc)
                .expect("ranks on the realistic engine")
                .grpc_dp_size,
            2
        );
        assert!(
            parse(&[&grpc[..], &["--kv-events-zmq-base-port", "19100"]].concat())
                .expect_err("no per-rank publisher ports")
                .contains("--grpc-dp-size above 1 does not publish")
        );
    }

    #[test]
    fn drain_timeout_flag_is_seconds_and_must_be_a_duration() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert_eq!(
            cfg.drain_timeout,
            Duration::from_secs(5),
            "the default drain"
        );

        let cfg = parse(&[&grpc[..], &["--drain-timeout-secs", "0.5"]].concat())
            .expect("fractional seconds parse");
        assert_eq!(cfg.drain_timeout, Duration::from_millis(500));
        assert!(
            parse(&[&grpc[..], &["--drain-timeout-secs", "-1"]].concat()).is_err(),
            "a negative drain is refused"
        );
        assert!(parse(&[&grpc[..], &["--drain-timeout-secs", "soon"]].concat()).is_err());
    }

    #[test]
    fn output_prefix_ids_are_a_comma_separated_list() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert!(cfg.engine.output_prefix.is_empty(), "none by default");
        let mut flags = grpc.to_vec();
        flags.extend(["--output-prefix-ids", "200005, 17196,200008"]);
        let cfg = parse(&flags).expect("a list of ids parses");
        assert_eq!(cfg.engine.output_prefix, vec![200_005, 17_196, 200_008]);
        let mut flags = grpc.to_vec();
        flags.extend(["--output-prefix-ids", "200005,x"]);
        assert!(
            parse(&flags).is_err(),
            "a value that is not an id is refused"
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

    #[test]
    fn script_flag_sets_the_script_path() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert_eq!(cfg.replay.script, None, "no script by default");

        let cfg = parse(&[&grpc[..], &["--script", "/tmp/answers.jsonl"]].concat())
            .expect("--script parses");
        assert_eq!(cfg.replay.script, Some(PathBuf::from("/tmp/answers.jsonl")));
    }

    /// The encode role is the vLLM wire's: it parses there, binds a bootstrap
    /// port when given one as a prefill worker does, and is refused on the
    /// other two wires with the reason.
    #[test]
    fn the_encode_role_is_the_vllm_wires_alone() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "2"];
        let encode = parse(
            &[
                &grpc[..],
                &["--grpc-wire", "vllm", "--worker-type", "encode"],
            ]
            .concat(),
        )
        .expect("encode workers on the vllm wire");
        assert_eq!(encode.role, Some(Role::Encode));
        assert_eq!(encode.bootstrap_port_for(0), None, "no port given");
        let hosting = parse(
            &[
                &grpc[..],
                &[
                    "--grpc-wire",
                    "vllm",
                    "--worker-type",
                    "encode",
                    "--bootstrap-base-port",
                    "8998",
                ],
            ]
            .concat(),
        )
        .expect("an encode worker hosts the rendezvous when told to");
        assert_eq!(
            (hosting.bootstrap_port_for(0), hosting.bootstrap_port_for(1)),
            (Some(8998), Some(8999))
        );
        for wire in ["tokenspeed", "sglang"] {
            let refused =
                parse(&[&grpc[..], &["--grpc-wire", wire, "--worker-type", "encode"]].concat())
                    .expect_err("the encode role is refused off the vllm wire");
            assert!(
                refused.contains("pass --grpc-wire vllm") && refused.contains(wire),
                "{refused}"
            );
        }
        assert!(parse(&[&grpc[..], &["--worker-type", "encode"]].concat())
            .expect_err("the default wire is tokenspeed")
            .contains("pass --grpc-wire vllm"));
    }

    /// The prefill and decode roles are the HTTP and the gRPC workers',
    /// with the same bootstrap numbering on both wires, one wire per
    /// process; the ZMQ workers take none and the encode role is a gRPC
    /// wire's.
    #[test]
    fn the_prefill_and_decode_roles_are_the_http_and_grpc_workers() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "2"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert_eq!((cfg.role, cfg.bootstrap_base_port), (None, None));
        assert_eq!(cfg.bootstrap_port_for(0), None, "a regular worker");

        let prefill = parse(
            &[
                &grpc[..],
                &["--worker-type", "prefill", "--bootstrap-base-port", "8998"],
            ]
            .concat(),
        )
        .expect("prefill workers with bootstrap ports");
        assert_eq!(prefill.role, Some(Role::Prefill));
        assert_eq!(
            (prefill.bootstrap_port_for(0), prefill.bootstrap_port_for(1)),
            (Some(8998), Some(8999)),
            "worker i binds base + i"
        );
        let unbound = parse(&[&grpc[..], &["--worker-type", "prefill"]].concat())
            .expect("a prefill worker without a bootstrap port");
        assert_eq!(unbound.bootstrap_port_for(0), None);

        let decode =
            parse(&[&grpc[..], &["--worker-type", "decode"]].concat()).expect("decode workers");
        assert_eq!(decode.role, Some(Role::Decode));
        assert_eq!(decode.bootstrap_port_for(0), None, "a decode binds no port");

        assert!(parse(&[&grpc[..], &["--worker-type", "embed"]].concat())
            .expect_err("no such role")
            .contains("--worker-type must be prefill|decode|encode"));
        let http = ["--http-base-port", "18000", "--http-count", "2"];
        let http_prefill = parse(
            &[
                &http[..],
                &["--worker-type", "prefill", "--bootstrap-base-port", "8998"],
            ]
            .concat(),
        )
        .expect("HTTP prefill workers with bootstrap ports");
        assert_eq!(http_prefill.role, Some(Role::Prefill));
        assert_eq!(
            (
                http_prefill.bootstrap_port_for(0),
                http_prefill.bootstrap_port_for(1)
            ),
            (Some(8998), Some(8999)),
            "HTTP worker i binds base + i as a gRPC worker does"
        );
        assert_eq!(
            parse(&[&http[..], &["--worker-type", "decode"]].concat())
                .expect("HTTP decode workers")
                .role,
            Some(Role::Decode)
        );
        assert!(parse(&[&http[..], &["--worker-type", "encode"]].concat())
            .expect_err("the HTTP workers have no encode role")
            .contains("HTTP workers have no encode role"));
        assert!(
            parse(&[&grpc[..], &["--http-count", "1", "--worker-type", "decode"]].concat())
                .expect_err("one wire's workers take the role")
                .contains("not both")
        );
        assert!(parse(&[
            "--zmq-handshake",
            "ipc:///tmp/mock-handshake",
            "--zmq-count",
            "1",
            "--worker-type",
            "prefill",
        ])
        .expect_err("the ZMQ workers take no role")
        .contains("ZMQ workers"));
        assert!(parse(
            &[
                &grpc[..],
                &["--worker-type", "decode", "--bootstrap-base-port", "8998"],
            ]
            .concat()
        )
        .expect_err("a decode binds no bootstrap port")
        .contains("needs --worker-type prefill or encode"));
        assert!(
            parse(&[&grpc[..], &["--bootstrap-base-port", "8998"]].concat()).is_err(),
            "a regular worker binds no bootstrap port"
        );
    }

    #[test]
    fn grpc_message_limit_is_off_by_default_and_settable() {
        let grpc = ["--grpc-base-port", "19000", "--grpc-count", "1"];
        let cfg = parse(&grpc).expect("gRPC flags parse");
        assert_eq!(
            cfg.grpc_max_message_bytes,
            usize::MAX,
            "no limit by default"
        );

        let cfg = parse(&[&grpc[..], &["--grpc-max-message-bytes", "4194304"]].concat())
            .expect("--grpc-max-message-bytes parses");
        assert_eq!(cfg.grpc_max_message_bytes, 4_194_304);
    }

    /// A ZMQ rank sends one message per token unless `--zmq-step-ms` gives
    /// it a step period.
    #[test]
    fn zmq_step_ms_default_zero_and_takes_milliseconds() {
        let zmq = [
            "--zmq-handshake",
            "ipc:///tmp/mock-handshake",
            "--zmq-count",
            "1",
        ];
        assert_eq!(parse(&zmq).expect("defaults").zmq_step_ms, 0);
        let stepping = parse(&[&zmq[..], &["--zmq-step-ms", "5"]].concat()).expect("a period");
        assert_eq!(stepping.zmq_step_ms, 5);
        let refused = parse(&[&zmq[..], &["--zmq-step-ms", "soon"]].concat())
            .expect_err("not a number of milliseconds");
        assert!(refused.contains("--zmq-step-ms"), "{refused}");
    }

    /// A lockstep group is several ranks syncing at one step period; the add
    /// hand-over delay is milliseconds and the ranks are independent without
    /// the flag.
    #[test]
    fn zmq_lockstep_needs_two_ranks_and_a_step_period() {
        let two = [
            "--zmq-handshake",
            "ipc:///tmp/mock-handshake",
            "--zmq-count",
            "2",
        ];
        assert!(!parse(&two).expect("independent ranks").zmq_lockstep);
        let group = parse(
            &[
                &two[..],
                &[
                    "--zmq-lockstep",
                    "--zmq-step-ms",
                    "5",
                    "--zmq-add-delay-ms",
                    "20",
                ],
            ]
            .concat(),
        )
        .expect("a group");
        assert!(group.zmq_lockstep);
        assert_eq!(group.zmq_add_delay_ms, 20);
        let one_rank = parse(
            &[
                "--zmq-handshake",
                "ipc:///tmp/mock-handshake",
                "--zmq-count",
                "1",
                "--zmq-lockstep",
                "--zmq-step-ms",
                "5",
            ][..],
        )
        .expect_err("one rank is no group");
        assert!(one_rank.contains("--zmq-count"), "{one_rank}");
        let no_period =
            parse(&[&two[..], &["--zmq-lockstep"]].concat()).expect_err("no step period");
        assert!(no_period.contains("--zmq-step-ms"), "{no_period}");
    }
}
