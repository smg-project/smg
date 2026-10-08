//! String-stop resolution for engines that receive token ids only.
//!
//! EngineCore (and TokenSpeed over ZMQ) never sees a stop *string*: the
//! frontend that owns the tokenizer must match strings on the decoded output.
//! These helpers are that frontend's request-side half: drop the strings from
//! the engine request, forward the ones that are a single token as
//! `stop_token_ids` (so the engine still stops early on them), and hand the
//! strings back as the caller's obligation to match and trim.

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use smg_grpc_client::{tokenspeed_proto, vllm_proto as vllm};
use tracing::{debug, warn};

/// Convert single-token stop strings into `stop_token_ids` entries so the engine
/// can halt generation early for the common case (e.g. `["."]`, `["\n"]`).
///
/// The proto `stop_token_ids` field is a flat list of single token ids, so a
/// multi-token stop string cannot be represented there — pushing its sub-tokens
/// would stop far too eagerly (on any one of them). Multi-token, empty, and
/// unknown stops are therefore left to the caller's `StopSequenceDecoder`,
/// which detokenizes worker output and trims the stop text. Existing
/// `stop_token_ids` are preserved and deduped.
pub fn encode_single_token_stops(
    stops: Vec<String>,
    stop_token_ids: &mut Vec<u32>,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    // Without a tokenizer we cannot encode (not expected on paths that resolve
    // one to tokenize the prompt). Safe: the strings are already dropped by the
    // caller, so the router-side decoder remains the source of truth.
    let Some(tokenizer) = tokenizer else {
        if !stops.is_empty() {
            warn!(
                "No tokenizer available to encode string stop sequences; \
                 relying on router-side stop decoder only"
            );
        }
        return;
    };

    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        // add_special_tokens=false: we want the literal token(s) for the stop
        // string, not a BOS/EOS-wrapped encoding.
        match tokenizer.encode(&stop, false) {
            Ok(encoding) => match encoding.token_ids() {
                [id] => {
                    if !stop_token_ids.contains(id) {
                        stop_token_ids.push(*id);
                    }
                }
                ids => debug!(
                    stop = %stop,
                    token_count = ids.len(),
                    "string stop is not single-token; handled by router-side stop decoder"
                ),
            },
            Err(e) => warn!(
                stop = %stop,
                error = %e,
                "Failed to encode string stop sequence; relying on router-side stop decoder"
            ),
        }
    }
}

/// [`take_vllm_string_stops`] for a TokenSpeed request: the same split, on
/// the TokenSpeed proto's sampling params.
pub fn take_tokenspeed_string_stops(
    params: &mut tokenspeed_proto::SamplingParams,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) -> Vec<String> {
    let stops = std::mem::take(&mut params.stop);
    encode_single_token_stops(stops.clone(), &mut params.stop_token_ids, tokenizer);
    stops
}

/// Take the string stops off a vLLM request's sampling params: single-token
/// stops join `stop_token_ids`, and every taken string is returned for the
/// caller to match on the decoded output. Empty when the request has none.
pub fn take_vllm_string_stops(
    params: &mut vllm::SamplingParams,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) -> Vec<String> {
    let stops = std::mem::take(&mut params.stop);
    encode_single_token_stops(stops.clone(), &mut params.stop_token_ids, tokenizer);
    stops
}
