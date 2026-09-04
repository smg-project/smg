//! Session lifecycle: build paired turn-1/turn-2 `/generate` requests, send
//! them through the shared client, and parse the SGLang-native responses
//! (single JSON or SSE frames).

use std::{
    fmt::Write as _,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use serde_json::Value;
use tokio::sync::{mpsc::UnboundedSender, Semaphore};

use crate::{
    args::{Args, Ingress, Payload, Turn2Ingress},
    dist::{self, PiecewiseCdf, Rng},
    report::RequestRecord,
};

/// Number of routing keys shared by `--routing-key-reuse` sessions.
const SHARED_ROUTING_KEYS: usize = 32;

/// Most input ids a routing-tokens hint carries (the gateway caps it there).
const TOKENS_HINT_CAP: usize = 512;

/// Shared state every session task needs.
pub struct Ctx {
    pub args: Arc<Args>,
    /// One client per configured connection-per-origin; requests
    /// round-robin so h2 streams spread over several connections.
    pub clients: Vec<reqwest::Client>,
    pub next_client: AtomicU64,
    pub limiter: Arc<Semaphore>,
    pub records: UnboundedSender<RequestRecord>,
    pub prompt_cdf: PiecewiseCdf,
    pub output_cdf: PiecewiseCdf,
    pub sent: AtomicU64,
    pub done: AtomicU64,
    pub errors: AtomicU64,
}

/// Turn-1 SMG index for a routing key under hash ingress — the stand-in for
/// ingress consistent hashing, so it must be a pure function of the key.
pub(crate) fn hash_smg(key: &str, smg_count: usize) -> usize {
    (dist::hash_str(key) % smg_count.max(1) as u64) as usize
}

/// Whether the session sends another turn after `turn`.
pub(crate) fn session_continues(cont_draw: bool, turn: u32, max_turns: u32) -> bool {
    cont_draw && turn < max_turns
}

/// Whether the next turn's context (current ⊕ output ⊕ suffix) still fits
/// the model window; a session that would exceed it ends, standing in for
/// the context-length limit.
pub(crate) fn next_context_fits(
    context_len: usize,
    output_len: usize,
    suffix_len: usize,
    prompt_max: u32,
) -> bool {
    context_len + output_len + suffix_len <= prompt_max as usize
}

/// Milliseconds since the Unix epoch.
pub fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Run one session: turn 1, then after each turn continue with probability
/// `--t2-ratio` (up to `--max-turns`), each new turn extending the context
/// with the previous turn's returned output plus a fresh suffix.
pub async fn run(ctx: Arc<Ctx>, sid: u64) {
    let args = &ctx.args;
    let session_seed = dist::sub_seed(dist::sub_seed(args.seed, dist::SALT_SESSION), sid);
    let mut rng = Rng::new(session_seed);

    let reuse_draw = rng.next_f64();
    let mut key = if reuse_draw < args.routing_key_reuse {
        format!("shared-{}", rng.next_index(SHARED_ROUTING_KEYS))
    } else {
        format!("sess-{sid}")
    };

    let prompt_len = ctx.prompt_cdf.sample(rng.next_f64()) as usize;
    let max_new_1 = ctx.output_cdf.sample(rng.next_f64());

    // Warm runs share one prefix stream; cold runs give each session its own,
    // so no cross-session prefix ever matches.
    let prefix_seed = if args.system_prefix_tokens > 0 {
        // A population of shared system prompts ("agents"): each session
        // picks one of `system_prefix_pool`, so a small set of large
        // prefixes is reused across many sessions — the agentic/chat
        // sharing shape. pool <= 1 keeps the single-global-prefix
        // behavior byte-identical.
        let base = dist::sub_seed(args.seed, dist::SALT_PREFIX);
        if args.system_prefix_pool > 1 {
            let agent = session_seed % args.system_prefix_pool as u64;
            dist::sub_seed(base, agent)
        } else {
            base
        }
    } else {
        dist::sub_seed(session_seed, dist::SALT_PREFIX)
    };
    let mut input_ids = dist::token_ids(prefix_seed, args.system_prefix_tokens as usize);
    let placeholder_total = args.image_placeholder_run as usize * args.image_count as usize;
    input_ids.resize(
        input_ids.len() + placeholder_total,
        args.image_placeholder_id,
    );
    let pad = prompt_len.saturating_sub(input_ids.len());
    input_ids.extend(dist::token_ids(
        dist::sub_seed(session_seed, dist::SALT_PAD),
        pad,
    ));

    let n = args.smg_urls.len();
    let t1_smg = match args.ingress {
        Ingress::Hash => hash_smg(&key, n),
        Ingress::Random => rng.next_index(n),
    };

    // Turn loop: each turn extends the context with the previous turn's
    // returned output plus a fresh suffix; the session ends on the continue
    // draw, the turn cap, a failed turn, or when the next context would
    // exceed the model window (`--prompt-max`).
    let mut context = input_ids;
    let mut turn: u32 = 1;
    let mut smg = t1_smg;
    loop {
        let max_new = if turn == 1 {
            max_new_1
        } else {
            ctx.output_cdf.sample(rng.next_f64())
        };
        let output_ids = send_turn(
            &ctx,
            &TurnRequest {
                sid,
                session_seed,
                turn: turn.min(u32::from(u8::MAX)) as u8,
                key: &key,
                smg,
                input_ids: &context,
                max_new,
            },
        )
        .await;

        let cont = rng.next_f64() < args.t2_ratio;
        let think = rng.next_exp(args.think_secs);
        if !session_continues(cont, turn, args.max_turns) {
            return;
        }
        // A failed turn has no output to extend; the session ends there.
        let Some(output_ids) = output_ids else {
            return;
        };
        let suffix = args.t2_suffix_tokens as usize;
        if !next_context_fits(context.len(), output_ids.len(), suffix, args.prompt_max) {
            return;
        }
        tokio::time::sleep(Duration::from_secs_f64(think)).await;

        context.extend(output_ids);
        context.extend(dist::token_ids(
            dist::sub_seed(
                dist::sub_seed(session_seed, dist::SALT_SUFFIX),
                u64::from(turn),
            ),
            suffix,
        ));
        turn += 1;
        // Clients without a stable session key present a fresh key each
        // turn: the sticky override re-pins, and hash ingress re-hashes.
        if args.key_per_turn {
            key = format!("sess-{sid}-t{turn}");
        }
        smg = match args.turn2_ingress {
            Turn2Ingress::Same => t1_smg,
            Turn2Ingress::Hash => hash_smg(&key, n),
            Turn2Ingress::Random => rng.next_index(n),
        };
    }
}

struct TurnRequest<'a> {
    sid: u64,
    session_seed: u64,
    turn: u8,
    key: &'a str,
    smg: usize,
    input_ids: &'a [u32],
    max_new: u32,
}

/// Send one `/generate` request and record its outcome. Returns the returned
/// `output_ids` on success (empty when the response carried none), `None` on
/// any error.
async fn send_turn(ctx: &Ctx, req: &TurnRequest<'_>) -> Option<Vec<u32>> {
    let args = &ctx.args;
    let permit = match ctx.limiter.clone().acquire_owned().await {
        Ok(permit) => permit,
        // The semaphore is never closed; a close means shutdown.
        Err(_) => return None,
    };

    // Image payloads are regenerated from the session seed inside the permit,
    // so waiting sessions do not hold hundreds of KB, and turn 2 reproduces
    // byte-identical bytes without storing them across the think time.
    let images: Vec<String> = (0..args.image_count)
        .map(|i| {
            dist::base64_blob(
                dist::sub_seed(req.session_seed, dist::SALT_IMAGE + u64::from(i)),
                args.image_bytes,
            )
        })
        .collect();
    let body = build_body(
        req.input_ids,
        &images,
        req.max_new,
        args.stream,
        args.payload,
        &args.model,
    );
    drop(images);

    let hint = args.tokens_hint.then(|| {
        let head = &req.input_ids[..req.input_ids.len().min(TOKENS_HINT_CAP)];
        let mut joined = String::with_capacity(head.len() * 7);
        for (i, id) in head.iter().enumerate() {
            if i > 0 {
                joined.push(',');
            }
            let _ = write!(joined, "{id}");
        }
        joined
    });

    ctx.sent.fetch_add(1, Ordering::Relaxed);
    let url = format!("{}/generate", args.smg_urls[req.smg]);
    let start_ms = epoch_ms();
    let started = Instant::now();
    let pick = ctx.next_client.fetch_add(1, Ordering::Relaxed) as usize % ctx.clients.len();
    let mut request = ctx.clients[pick]
        .post(&url)
        .header("content-type", "application/json")
        .header("x-smg-routing-key", req.key);
    if let Some(hint) = &hint {
        request = request.header("x-smg-routing-tokens", hint.as_str());
    }

    let mut status: u16 = 0;
    let mut ttft_ms: Option<f64> = None;
    let mut response: Option<Value> = None;
    let mut index_source: Option<String> = None;
    let mut index_predicted_tokens: Option<u64> = None;
    if let Ok(resp) = request.body(body).send().await {
        status = resp.status().as_u16();
        // Remote-index echo headers (absent unless the gateway runs with
        // --kv-indexer-url); captured before the body consumes `resp`.
        index_source = resp
            .headers()
            .get("x-smg-index-source")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        index_predicted_tokens = resp
            .headers()
            .get("x-smg-index-predicted-tokens")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        if resp.status().is_success() {
            if args.stream {
                match consume_sse(resp, started).await {
                    Ok((ttft, last)) => {
                        ttft_ms = ttft;
                        response = last;
                    }
                    // A mid-stream transport failure is an incomplete
                    // request, not a success at the original status.
                    Err(_) => status = 0,
                }
            } else {
                match resp.bytes().await {
                    // The gRPC router returns an array of responses for
                    // non-streaming /generate; the HTTP mock returns one
                    // object. Normalize to the first (only) element.
                    Ok(bytes) => {
                        response = serde_json::from_slice::<Value>(&bytes)
                            .ok()
                            .map(|v| match v {
                                Value::Array(mut items) if !items.is_empty() => {
                                    items.swap_remove(0)
                                }
                                other => other,
                            })
                    }
                    Err(_) => status = 0,
                }
            }
        } else {
            // Drain the error body so the connection can be reused.
            let _ = resp.bytes().await;
        }
    }
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;
    drop(permit);

    let mut worker_port = None;
    let mut cached_tokens = None;
    let mut completion_tokens = None;
    let mut output_ids: Option<Vec<u32>> = None;
    if let Some(value) = &response {
        let meta = &value["meta_info"];
        // The HTTP mock injects its port directly; the gRPC router carries
        // no worker identity, so gRPC legs register each worker with a
        // `weight_version` label of its port, relayed here verbatim.
        worker_port = meta["worker_port"].as_u64().or_else(|| {
            meta["weight_version"]
                .as_str()
                .and_then(|v| v.parse::<u64>().ok())
        });
        cached_tokens = meta["cached_tokens"].as_u64();
        completion_tokens = meta["completion_tokens"].as_u64();
        output_ids = value["output_ids"].as_array().map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_u64().map(|id| id as u32))
                .collect()
        });
    }

    let is_error = !(200..300).contains(&status);
    if is_error {
        ctx.errors.fetch_add(1, Ordering::Relaxed);
    }
    ctx.done.fetch_add(1, Ordering::Relaxed);

    let record = RequestRecord {
        turn: req.turn,
        session: req.sid,
        key: req.key.to_string(),
        smg: req.smg,
        worker_port,
        prompt_tokens: req.input_ids.len(),
        cached_tokens,
        completion_tokens,
        max_new: req.max_new,
        ttft_ms,
        e2e_ms,
        status,
        start_ms,
        index_source,
        index_predicted_tokens,
    };
    // A send error means the collector is gone (shutdown); nothing to do.
    let _ = ctx.records.send(record);

    if is_error {
        None
    } else {
        Some(output_ids.unwrap_or_default())
    }
}

/// Serialize the request body by hand: the multi-hundred-KB image strings are
/// appended directly instead of being copied through an intermediate
/// `serde_json::Value` (the base64 alphabet needs no JSON escaping).
///
/// `Payload::Text` sends the same token context as a `text` field of
/// space-joined decimal words instead of `input_ids`. Appending tokens
/// appends text, so a turn's context is a string prefix of the next turn's
/// and prefix matching survives the format change. The gateway then routes
/// on its approximate string tree; the mock re-derives one stable id per
/// word. Image placeholders are only expanded on the ids path, so text runs
/// should use `--image-count 0`.
fn build_body(
    input_ids: &[u32],
    images: &[String],
    max_new: u32,
    stream: bool,
    payload: Payload,
    model: &str,
) -> String {
    let image_len: usize = images.iter().map(|image| image.len() + 3).sum();
    let mut body = String::with_capacity(image_len + input_ids.len() * 7 + 128);
    body.push('{');
    if !model.is_empty() {
        let _ = write!(body, "\"model\":\"{model}\",");
    }
    match payload {
        Payload::Ids => {
            body.push_str("\"input_ids\":[");
            for (i, id) in input_ids.iter().enumerate() {
                if i > 0 {
                    body.push(',');
                }
                let _ = write!(body, "{id}");
            }
            body.push(']');
        }
        Payload::Text => {
            body.push_str("\"text\":\"");
            for (i, id) in input_ids.iter().enumerate() {
                if i > 0 {
                    body.push(' ');
                }
                let _ = write!(body, "{id}");
            }
            body.push('"');
        }
    }
    if !images.is_empty() {
        body.push_str(",\"image_data\":[");
        for (i, image) in images.iter().enumerate() {
            if i > 0 {
                body.push(',');
            }
            body.push('"');
            body.push_str(image);
            body.push('"');
        }
        body.push(']');
    }
    let _ = write!(
        body,
        ",\"sampling_params\":{{\"max_new_tokens\":{max_new}}},\"stream\":{stream}}}"
    );
    body
}

/// Drain an SSE response: TTFT is the elapsed time to the first data frame,
/// and the last data frame before `[DONE]` carries the same full JSON shape
/// as a non-streaming response. Frames may span chunks, so bytes are buffered
/// and consumed line by line.
async fn consume_sse(
    resp: reqwest::Response,
    started: Instant,
) -> Result<(Option<f64>, Option<Value>), reqwest::Error> {
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut ttft_ms = None;
    let mut last = None;
    'read: while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(newline) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=newline).collect();
            let line = line.strip_suffix(b"\n").unwrap_or(&line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line
                .strip_prefix(b"data: ")
                .or_else(|| line.strip_prefix(b"data:"))
            else {
                continue;
            };
            if data == b"[DONE]".as_slice() {
                break 'read;
            }
            if ttft_ms.is_none() {
                ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            }
            if let Ok(value) = serde_json::from_slice::<Value>(data) {
                last = Some(value);
            }
        }
    }
    // Drain any trailing bytes after [DONE] so the connection can be reused.
    while let Some(chunk) = stream.next().await {
        if chunk.is_err() {
            break;
        }
    }
    Ok((ttft_ms, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_ingress_is_deterministic_and_spreads() {
        // Same key always maps to the same SMG (ingress stickiness)...
        for key in ["sess-1", "sess-2", "shared-7"] {
            assert_eq!(hash_smg(key, 8), hash_smg(key, 8));
        }
        // ...and distinct keys reach every SMG (no degenerate hashing).
        let mut seen = [false; 8];
        for sid in 0..256 {
            seen[hash_smg(&format!("sess-{sid}"), 8)] = true;
        }
        assert!(seen.iter().all(|&s| s), "keys must spread over all SMGs");
        // A single SMG never divides by zero.
        assert_eq!(hash_smg("sess-1", 1), 0);
    }

    #[test]
    fn session_continuation_respects_draw_and_turn_cap() {
        assert!(session_continues(true, 1, 2), "draw passed, under cap");
        assert!(
            !session_continues(false, 1, 8),
            "failed draw ends the session"
        );
        assert!(!session_continues(true, 2, 2), "turn cap ends the session");
        assert!(session_continues(true, 7, 8));
        assert!(!session_continues(true, 8, 8));
    }

    #[test]
    fn context_window_caps_session_growth() {
        assert!(next_context_fits(10_000, 2000, 256, 24_576));
        assert!(!next_context_fits(23_000, 2000, 256, 24_576));
        // Exactly at the window still fits.
        assert!(next_context_fits(24_000, 500, 76, 24_576));
    }

    #[test]
    fn model_field_prepends_only_when_set() {
        let with: serde_json::Value =
            serde_json::from_str(&build_body(&[1], &[], 2, false, Payload::Ids, "mock-model"))
                .unwrap();
        assert_eq!(with["model"], "mock-model");
        let without: serde_json::Value =
            serde_json::from_str(&build_body(&[1], &[], 2, false, Payload::Ids, "")).unwrap();
        assert!(
            without.get("model").is_none(),
            "empty model must be omitted"
        );
    }

    #[test]
    fn text_payload_is_valid_json_and_prefix_preserving() {
        let turn1 = build_body(&[12, 3], &[], 8, false, Payload::Text, "");
        let turn2 = build_body(&[12, 3, 45], &[], 8, false, Payload::Text, "");
        let v1: serde_json::Value = serde_json::from_str(&turn1).unwrap();
        let v2: serde_json::Value = serde_json::from_str(&turn2).unwrap();
        assert_eq!(v1["text"], "12 3");
        assert_eq!(v2["text"], "12 3 45");
        assert!(v1.get("input_ids").is_none(), "text mode must omit ids");
        // Extending the token context extends the text — the string tree
        // sees turn 1 as a prefix of turn 2.
        assert!(v2["text"]
            .as_str()
            .unwrap()
            .starts_with(v1["text"].as_str().unwrap()));
        // The ids path is untouched by the payload switch.
        let ids: serde_json::Value =
            serde_json::from_str(&build_body(&[12, 3], &[], 8, false, Payload::Ids, "")).unwrap();
        assert_eq!(ids["input_ids"], serde_json::json!([12, 3]));
    }
}
