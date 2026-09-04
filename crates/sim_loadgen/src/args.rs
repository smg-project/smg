//! Runtime configuration for the load generator, parsed from CLI flags.

/// How a session picks its SMG for turn 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ingress {
    /// Consistent choice: splitmix64 of the routing key modulo the URL count.
    Hash,
    /// A fresh uniform choice per request.
    Random,
}

impl Ingress {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hash => "hash",
            Self::Random => "random",
        }
    }
}

/// How a session picks its SMG for turn 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turn2Ingress {
    /// Reuse the SMG turn 1 landed on.
    Same,
    /// Consistent hash of the routing key (matches turn 1 under `--ingress hash`).
    Hash,
    /// A fresh uniform choice.
    Random,
}

impl Turn2Ingress {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::Hash => "hash",
            Self::Random => "random",
        }
    }
}

/// Wire format for the prompt in the `/generate` body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payload {
    /// Pre-tokenized `input_ids` — the gateway routes on its token tree
    /// (or the hash index / sticky override when those are configured).
    Ids,
    /// Untokenized `text`: the token context space-joined as decimal words.
    /// The gateway routes on its approximate string tree; the mock worker
    /// re-derives one stable id per word, so prefix reuse is preserved.
    Text,
}

impl Payload {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ids => "ids",
            Self::Text => "text",
        }
    }
}

/// Configuration for one load-generation run.
#[derive(Debug, Clone)]
pub struct Args {
    /// SMG base URLs the generator spreads sessions across (required).
    pub smg_urls: Vec<String>,
    /// Length of the session-arrival window; in-flight sessions still finish.
    pub duration_secs: u64,
    /// Poisson session arrival rate.
    pub session_rps: f64,
    /// Per-turn continue probability: after each turn the session sends
    /// another with this probability (name kept from the two-turn contract,
    /// where it was exactly the turn-2 probability).
    pub t2_ratio: f64,
    /// Hard cap on turns per session; context growth also ends a session
    /// when the next turn would exceed `--prompt-max` (the model's
    /// context-window limit stands in for both).
    pub max_turns: u32,
    /// Per-request client timeout; a wedged stream records status 0 instead
    /// of hanging the end-of-run drain.
    pub request_timeout_secs: u64,
    /// Give every turn a fresh routing key (models clients that do not
    /// carry a stable session key): each turn re-pins under the sticky
    /// override and, under hash ingress, may land on a different SMG.
    pub key_per_turn: bool,
    /// Mean of the exponential think time between turn 1 and turn 2.
    pub think_secs: f64,
    /// Request SSE streaming responses (TTFT is only measurable when true).
    pub stream: bool,
    /// Speak HTTP/2 prior knowledge to the SMGs (multiplexed streams).
    pub http2: bool,
    /// Independent connections per SMG origin (requests round-robin across
    /// them). In `--http2` mode each client multiplexes ONE connection per
    /// origin, so this bounds concurrent streams per SMG; without it a
    /// small gateway count throttles the generator, not the gateway.
    pub conns_per_origin: usize,
    /// Turn-1 SMG choice.
    pub ingress: Ingress,
    /// Turn-2 SMG choice.
    pub turn2_ingress: Turn2Ingress,
    /// Fraction of sessions using one of 32 shared routing keys instead of a
    /// unique per-session key.
    pub routing_key_reuse: f64,
    /// Shared warm prefix length; 0 = per-session unique prefix (cold).
    pub system_prefix_tokens: u32,
    /// Number of distinct shared system prefixes ("agents"): each session
    /// picks one, so the population reuses `system_prefix_pool` large
    /// prefixes. 1 = a single global shared prefix (byte-identical to the
    /// pre-pool behavior).
    pub system_prefix_pool: u32,
    /// Images per session.
    pub image_count: u32,
    /// Base64 characters per image payload.
    pub image_bytes: usize,
    /// Token id marking image positions in `input_ids`.
    pub image_placeholder_id: u32,
    /// Placeholder ids emitted per image.
    pub image_placeholder_run: u32,
    /// Fresh ids appended after the echoed turn-1 output in turn 2.
    pub t2_suffix_tokens: u32,
    /// Prompt-length CDF anchors as (tokens, cumulative) pairs.
    pub prompt_cdf: Vec<(u32, f64)>,
    /// Prompt length at cumulative 1.0 (the CDF tail anchor).
    pub prompt_max: u32,
    /// Output-length CDF anchors as (tokens, cumulative) pairs.
    pub output_cdf: Vec<(u32, f64)>,
    /// Output length at cumulative 1.0 (the CDF tail anchor).
    pub output_max: u32,
    /// Send `x-smg-routing-tokens` (first <=512 input ids) with each request.
    pub tokens_hint: bool,
    /// Prompt wire format: `ids` (default) or `text`.
    pub payload: Payload,
    /// `model` field for the request body; empty omits it. IGW-mode
    /// gateways (gRPC worker legs) reject /generate without a model.
    pub model: String,
    /// Global cap on in-flight requests (a permit per request, not session).
    pub max_inflight: usize,
    /// Requests finishing this early are excluded from summary stats.
    pub warmup_secs: u64,
    /// Base seed; every random stream in the run derives from it.
    pub seed: u64,
    /// Output directory for requests.jsonl and summary.json.
    pub out: String,
}

impl Args {
    /// Flag defaults; the single source `from_args` mutates and tests build
    /// synthetic configs from.
    pub(crate) fn defaults() -> Self {
        Self {
            smg_urls: Vec::new(),
            duration_secs: 60,
            session_rps: 5.0,
            t2_ratio: 0.5,
            think_secs: 30.0,
            stream: true,
            http2: false,
            conns_per_origin: 4,
            max_turns: 2,
            request_timeout_secs: 300,
            key_per_turn: false,
            ingress: Ingress::Hash,
            turn2_ingress: Turn2Ingress::Same,
            routing_key_reuse: 0.0,
            system_prefix_tokens: 2048,
            system_prefix_pool: 1,
            image_count: 1,
            image_bytes: 620_000,
            image_placeholder_id: 151_655,
            image_placeholder_run: 256,
            t2_suffix_tokens: 64,
            prompt_cdf: vec![(5000, 0.216), (10_000, 0.530), (20_000, 0.994)],
            prompt_max: 32_000,
            output_cdf: vec![(1000, 0.351), (2000, 0.513), (5000, 0.998)],
            output_max: 8192,
            tokens_hint: false,
            payload: Payload::Ids,
            model: String::new(),
            max_inflight: 200_000,
            warmup_secs: 0,
            seed: 42,
            out: "sim_out".to_string(),
        }
    }

    /// Parse the configuration from `std::env::args`, falling back to defaults.
    pub fn from_args() -> Result<Self, String> {
        let mut cfg = Self::defaults();

        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--smg-urls" => {
                    cfg.smg_urls = value(&mut args, &flag)?
                        .split(',')
                        .map(|url| url.trim().trim_end_matches('/').to_string())
                        .filter(|url| !url.is_empty())
                        .collect();
                }
                "--duration-secs" => cfg.duration_secs = parse(value(&mut args, &flag)?, &flag)?,
                "--session-rps" => cfg.session_rps = parse(value(&mut args, &flag)?, &flag)?,
                "--t2-ratio" => cfg.t2_ratio = parse(value(&mut args, &flag)?, &flag)?,
                "--think-secs" => cfg.think_secs = parse(value(&mut args, &flag)?, &flag)?,
                "--stream" => cfg.stream = parse(value(&mut args, &flag)?, &flag)?,
                "--http2" => cfg.http2 = parse(value(&mut args, &flag)?, &flag)?,
                "--conns-per-origin" => {
                    cfg.conns_per_origin = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--max-turns" => cfg.max_turns = parse(value(&mut args, &flag)?, &flag)?,
                "--request-timeout-secs" => {
                    cfg.request_timeout_secs = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--key-per-turn" => cfg.key_per_turn = parse(value(&mut args, &flag)?, &flag)?,
                "--ingress" => {
                    cfg.ingress = match value(&mut args, &flag)?.as_str() {
                        "hash" => Ingress::Hash,
                        "random" => Ingress::Random,
                        other => return Err(format!("--ingress must be hash|random, got {other}")),
                    }
                }
                "--turn2-ingress" => {
                    cfg.turn2_ingress = match value(&mut args, &flag)?.as_str() {
                        "same" => Turn2Ingress::Same,
                        "hash" => Turn2Ingress::Hash,
                        "random" => Turn2Ingress::Random,
                        other => {
                            return Err(format!(
                                "--turn2-ingress must be same|hash|random, got {other}"
                            ))
                        }
                    }
                }
                "--routing-key-reuse" => {
                    cfg.routing_key_reuse = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--system-prefix-tokens" => {
                    cfg.system_prefix_tokens = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--system-prefix-pool" => {
                    cfg.system_prefix_pool = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--image-count" => cfg.image_count = parse(value(&mut args, &flag)?, &flag)?,
                "--image-bytes" => cfg.image_bytes = parse(value(&mut args, &flag)?, &flag)?,
                "--image-placeholder-id" => {
                    cfg.image_placeholder_id = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--image-placeholder-run" => {
                    cfg.image_placeholder_run = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--t2-suffix-tokens" => {
                    cfg.t2_suffix_tokens = parse(value(&mut args, &flag)?, &flag)?;
                }
                "--prompt-cdf" => cfg.prompt_cdf = parse_cdf(&value(&mut args, &flag)?, &flag)?,
                "--prompt-max" => cfg.prompt_max = parse(value(&mut args, &flag)?, &flag)?,
                "--output-cdf" => cfg.output_cdf = parse_cdf(&value(&mut args, &flag)?, &flag)?,
                "--output-max" => cfg.output_max = parse(value(&mut args, &flag)?, &flag)?,
                "--tokens-hint" => cfg.tokens_hint = parse(value(&mut args, &flag)?, &flag)?,
                "--payload" => {
                    cfg.payload = match value(&mut args, &flag)?.as_str() {
                        "ids" => Payload::Ids,
                        "text" => Payload::Text,
                        other => return Err(format!("--payload must be ids|text, got {other}")),
                    }
                }
                "--model" => cfg.model = value(&mut args, &flag)?,
                "--max-inflight" => cfg.max_inflight = parse(value(&mut args, &flag)?, &flag)?,
                "--warmup-secs" => cfg.warmup_secs = parse(value(&mut args, &flag)?, &flag)?,
                "--seed" => cfg.seed = parse(value(&mut args, &flag)?, &flag)?,
                "--out" => cfg.out = value(&mut args, &flag)?,
                "-h" | "--help" => return Err(usage()),
                other => return Err(format!("unknown flag: {other}\n\n{}", usage())),
            }
        }

        if cfg.smg_urls.is_empty() {
            return Err(format!("--smg-urls is required\n\n{}", usage()));
        }
        if cfg.session_rps <= 0.0 || !cfg.session_rps.is_finite() {
            return Err("--session-rps must be a positive finite number".to_string());
        }
        for (flag, ratio) in [
            ("--t2-ratio", cfg.t2_ratio),
            ("--routing-key-reuse", cfg.routing_key_reuse),
        ] {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(format!("{flag} must be within [0, 1], got {ratio}"));
            }
        }
        if cfg.think_secs < 0.0 || !cfg.think_secs.is_finite() {
            return Err("--think-secs must be a non-negative finite number".to_string());
        }
        if cfg.max_inflight == 0 {
            return Err("--max-inflight must be at least 1".to_string());
        }
        for (max_flag, cdf_flag, cdf, max) in [
            (
                "--prompt-max",
                "--prompt-cdf",
                &cfg.prompt_cdf,
                cfg.prompt_max,
            ),
            (
                "--output-max",
                "--output-cdf",
                &cfg.output_cdf,
                cfg.output_max,
            ),
        ] {
            if let Some(&(tokens, _)) = cdf.last() {
                if max < tokens {
                    return Err(format!(
                        "{max_flag} ({max}) must be at least the last {cdf_flag} anchor ({tokens})"
                    ));
                }
            }
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

fn parse_cdf(raw: &str, flag: &str) -> Result<Vec<(u32, f64)>, String> {
    let mut anchors = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        let Some((tokens, cum)) = part.split_once(':') else {
            return Err(format!(
                "invalid {flag} anchor (want tokens:cumulative): {part}"
            ));
        };
        let tokens: u32 = tokens
            .parse()
            .map_err(|_| format!("invalid {flag} token count: {part}"))?;
        let cum: f64 = cum
            .parse()
            .map_err(|_| format!("invalid {flag} cumulative value: {part}"))?;
        anchors.push((tokens, cum));
    }
    if anchors.is_empty() {
        return Err(format!(
            "{flag} needs at least one tokens:cumulative anchor"
        ));
    }
    if anchors.iter().any(|&(_, cum)| cum <= 0.0 || cum > 1.0) {
        return Err(format!("{flag} cumulative values must be within (0, 1]"));
    }
    if anchors
        .windows(2)
        .any(|pair| pair[1].0 <= pair[0].0 || pair[1].1 <= pair[0].1)
    {
        return Err(format!(
            "{flag} anchors must be strictly increasing in both tokens and cumulative"
        ));
    }
    Ok(anchors)
}

fn usage() -> String {
    "sim-loadgen — open-loop /generate load generator for SMG scale simulation\n\n\
     Flags:\n\
       --smg-urls <u1,u2,...>       SMG base URLs (required)\n\
       --duration-secs <n>          session-arrival window length (default 60)\n\
       --session-rps <f>            Poisson session arrival rate (default 5.0)\n\
       --t2-ratio <f>               probability a session sends turn 2 (default 0.5)\n\
       --think-secs <f>             mean exponential think time before turn 2 (default 30)\n\
       --stream <bool>              request SSE streaming responses (default true)\n\
       --http2 <bool>               HTTP/2 prior knowledge to the SMGs (default false)\n\
       --conns-per-origin <n>       connections per SMG, round-robined (default 4)\n\
       --max-turns <n>              turn cap per session; each turn continues with\n\
                                    probability --t2-ratio (default 2)\n\
       --request-timeout-secs <n>   per-request client timeout (default 300)\n\
       --key-per-turn <bool>        fresh routing key every turn (default false)\n\
       --ingress <hash|random>      turn-1 SMG choice (default hash)\n\
       --turn2-ingress <same|hash|random>  turn-2 SMG choice (default same)\n\
       --routing-key-reuse <f>      fraction of sessions sharing one of 32 keys (default 0.0)\n\
       --system-prefix-tokens <n>   shared warm prefix length; 0 = cold (default 2048)\n\
       --image-count <n>            images per session (default 1)\n\
       --image-bytes <n>            base64 chars per image (default 620000)\n\
       --image-placeholder-id <n>   token id marking image positions (default 151655)\n\
       --image-placeholder-run <n>  placeholder ids per image (default 256)\n\
       --t2-suffix-tokens <n>       fresh ids appended in turn 2 (default 64)\n\
       --prompt-cdf <t:c,...>       prompt-length CDF anchors\n\
                                    (default 5000:0.216,10000:0.530,20000:0.994)\n\
       --prompt-max <n>             prompt length at cumulative 1.0 (default 32000)\n\
       --output-cdf <t:c,...>       output-length CDF anchors\n\
                                    (default 1000:0.351,2000:0.513,5000:0.998)\n\
       --output-max <n>             output length at cumulative 1.0 (default 8192)\n\
       --tokens-hint <bool>         send x-smg-routing-tokens, first <=512 ids (default false)\n\
       --max-inflight <n>           global in-flight request cap (default 200000)\n\
       --warmup-secs <n>            exclude requests finishing this early from stats (default 0)\n\
       --seed <n>                   base seed for all randomness (default 42)\n\
       --out <dir>                  output directory (default sim_out)"
        .to_string()
}
