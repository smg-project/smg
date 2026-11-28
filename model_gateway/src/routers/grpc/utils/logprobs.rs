//! Proto-to-OpenAI logprob conversion functions.

use std::{collections::HashMap, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::common::{ChatLogProbs, ChatLogProbsContent, LogProbs, TopLogProb};

use crate::routers::grpc::proto_wrapper::{ProtoInputLogProbs, ProtoOutputLogProbs};

/// Convert an output's logprobs, read in place on the engine message, to the
/// OpenAI ChatLogProbs format.
///
/// Generic over the token decoding strategy. The `decode_token` closure maps a
/// single token ID to its text representation. `requested` is how many
/// candidates per position the answer carries: the engine may report more
/// (the chosen token, its own list's length), and only the first `requested`
/// of them are decoded; `None` keeps every candidate the engine sent.
pub(crate) fn convert_proto_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    requested: Option<usize>,
    decode_token: impl Fn(u32) -> String,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(proto_logprobs.len());

    for position in proto_logprobs.positions() {
        let token_text = decode_token(position.token_id);
        let bytes = Some(token_text.as_bytes().to_vec());

        // The position's candidates, in the engine's order, cut to the requested
        // count before any is decoded: the list is sized from the cut.
        let top_logprobs = position
            .top_logprobs
            .iter()
            .take(requested.unwrap_or(usize::MAX))
            .map(|(token_id, logprob)| {
                let text = if token_id == position.token_id {
                    token_text.clone()
                } else {
                    decode_token(token_id)
                };
                let bytes = Some(text.as_bytes().to_vec());
                TopLogProb {
                    token: text,
                    logprob,
                    bytes,
                }
            })
            .collect();

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob: position.logprob,
            bytes,
            top_logprobs,
        });
    }

    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
        refusal: None,
    }
}

/// The candidates per position a chat answer carries for the request's
/// `top_logprobs`: exactly that many, none when the request named none. The
/// engine may report more, e.g. the chosen token when only its own logprob
/// was requested; every chat path cuts to this count before it decodes, so
/// the rule lives here once.
pub(crate) fn requested_alternatives(top_logprobs: Option<u32>) -> usize {
    top_logprobs.unwrap_or(0) as usize
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format using a Tokenizer.
///
/// `top_logprobs` is the number of alternatives the request asked for per
/// token (`None` when it asked for none); [`requested_alternatives`] turns it
/// into the candidates the answer carries, and only those are decoded.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    tokenizer: &Arc<dyn Tokenizer>,
    top_logprobs: Option<u32>,
) -> ChatLogProbs {
    let requested = Some(requested_alternatives(top_logprobs));
    convert_proto_logprobs(proto_logprobs, requested, token_decoder(tokenizer))
}

/// One token id's text as the logprobs carry it: the tokenizer's decode of
/// the id alone, or a placeholder naming the id when the decode fails. The
/// one decode every logprobs conversion and writer shares.
pub(crate) fn token_decoder(tokenizer: &Arc<dyn Tokenizer>) -> impl Fn(u32) -> String + '_ {
    move |token_id| {
        tokenizer
            .decode(&[token_id], false)
            .unwrap_or_else(|_| format!("<token_{token_id}>"))
    }
}

/// Whether `logprobs` carries at least one token's entry.
pub(crate) fn has_entries(logprobs: Option<&ChatLogProbs>) -> bool {
    matches!(
        logprobs,
        Some(ChatLogProbs::Detailed {
            content: Some(items),
            ..
        }) if !items.is_empty()
    )
}

/// Convert OutputLogProbs to Generate format `Vec<Vec<Option<f64>>>`
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (u32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
) -> Vec<Vec<Option<f64>>> {
    // `positions()` filters, so its size hint is not exact: the outer list is
    // sized from the position count, as `convert_proto_logprobs` sizes its own.
    let mut rows = Vec::with_capacity(proto_logprobs.len());
    rows.extend(proto_logprobs.positions().map(|position| {
        vec![
            Some(f64::from(position.logprob)),
            Some(f64::from(position.token_id)),
        ]
    }));
    rows
}

/// Convert InputLogProbs to Generate format `Vec<Vec<Option<f64>>>`
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &ProtoInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&token_logprob, &token_id)| {
            // token_logprob is Option<f32> in unified type
            let logprob_value = token_logprob.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

/// Convert an output's logprobs, read in place on the engine message, to the
/// legacy `/v1/completions` shape: per generated token its own decode, its
/// logprob, the position's candidates and its character offset in the
/// choice's text. The offsets count the characters of the tokens before each
/// one from `text_offset`, which is moved past the positions converted so
/// that a stream's chunks continue one count.
///
/// `top_logprobs` is the request's `logprobs`: the public API returns that
/// many most likely tokens and always the sampled token's logprob, up to
/// `logprobs + 1` entries per position. So a position's map holds the
/// sampled token's entry first (whether or not the engine ranked it: a wire
/// sends no row for `logprobs: 0`), then the engine's candidates in its
/// order up to that many, one per token id, keyed by their text as the
/// engine's own server keys them (a later candidate of the same text takes
/// the entry).
pub(crate) fn convert_proto_to_completion_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    tokenizer: &Arc<dyn Tokenizer>,
    top_logprobs: u32,
    text_offset: &mut u32,
) -> LogProbs {
    completion_logprobs(
        proto_logprobs,
        top_logprobs,
        text_offset,
        token_decoder(tokenizer),
    )
}

/// [`convert_proto_to_completion_logprobs`] over any token decode.
pub(crate) fn completion_logprobs(
    proto_logprobs: ProtoOutputLogProbs<'_>,
    top_logprobs: u32,
    text_offset: &mut u32,
    decode_token: impl Fn(u32) -> String,
) -> LogProbs {
    let count = proto_logprobs.len();
    let mut logprobs = LogProbs {
        tokens: Vec::with_capacity(count),
        token_logprobs: Vec::with_capacity(count),
        top_logprobs: Vec::with_capacity(count),
        text_offset: Vec::with_capacity(count),
    };
    let entries = top_logprobs as usize + 1;
    for position in proto_logprobs.positions() {
        let token = decode_token(position.token_id);
        // The sampled token's entry, then the candidates, by text; a repeated
        // id (a wire that leads its row with the sampled token and ranks it
        // too) takes no second entry.
        let mut candidates = HashMap::with_capacity(entries.min(position.top_logprobs.len() + 1));
        let mut kept: Vec<u32> = Vec::with_capacity(entries);
        kept.push(position.token_id);
        candidates.insert(token.clone(), position.logprob);
        for (token_id, logprob) in position.top_logprobs.iter() {
            if kept.len() >= entries {
                break;
            }
            if kept.contains(&token_id) {
                continue;
            }
            kept.push(token_id);
            candidates.insert(decode_token(token_id), logprob);
        }
        logprobs.text_offset.push(*text_offset);
        *text_offset =
            text_offset.saturating_add(u32::try_from(token.chars().count()).unwrap_or(u32::MAX));
        logprobs.tokens.push(token);
        logprobs.token_logprobs.push(Some(position.logprob));
        logprobs.top_logprobs.push(Some(candidates));
    }
    logprobs
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/utils/logprobs/tests.rs"]
mod tests;
