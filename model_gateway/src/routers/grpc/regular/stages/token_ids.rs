//! The vocabulary bound of the token ids a request names: an id that can reach no row of the
//! engine's embedding table is refused at the gateway, on `/generate`'s `input_ids`, on
//! `/v1/completions`' prompt of ids and on a `logit_bias` map's keys alike, before any worker
//! is spoken to.

use std::collections::HashMap;

use llm_tokenizer::traits::Tokenizer;

use crate::routers::common::snapshot::WorkerRegistry;

/// The worker label that carries the engine's embedding-table size: `vocab_size` of the
/// engine's model info, kept on every worker that reports it.
const VOCAB_SIZE_LABEL: &str = "vocab_size";

/// The embedding-table size the model's workers report, the smallest among those that report
/// one (an id valid on a larger table is not on a smaller); `None` when no worker reports it.
/// Read only after an id the tokenizer does not know, so a valid prompt never pays for it.
pub(crate) fn reported_embedding_rows(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<usize> {
    worker_registry
        .get_by_model(model_id)
        .iter()
        .filter_map(|worker| {
            worker
                .metadata()
                .spec
                .labels
                .get(VOCAB_SIZE_LABEL)
                .and_then(|rows| rows.parse::<usize>().ok())
        })
        .min()
}

/// Whether the tokenizer knows `id`: by range below its base vocabulary (one comparison, the
/// common id), by lookup above it (an added token; the special tokens live there). The range
/// is the tokenizer's own report and a fast path, not a guarantee: a tiktoken encoding leaves
/// holes below it (ids with neither a rank nor a special token), which only a decode finds.
fn known_to(tokenizer: &dyn Tokenizer, base_vocab: usize, id: u32) -> bool {
    usize::try_from(id).is_ok_and(|id| id < base_vocab) || tokenizer.id_to_token(id).is_some()
}

/// An id a request named that is not a token of the model: its position among the ids
/// given, its value, and the bound it missed, worded for a refusal.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct UnknownTokenId {
    pub(crate) index: usize,
    pub(crate) id: u32,
    pub(crate) bound: String,
}

/// The first id that can reach no row of the engine's embedding table, with the bound it
/// missed; `None` when every id is a token of the model.
///
/// An id below the tokenizer's base vocabulary is known by its range alone; one above it is an
/// added token (the special tokens live there) and is resolved by lookup, so the common id costs
/// one comparison and only the rare one a map read. The engines index their embeddings by these
/// ids and the headless lanes run none of the checks an engine's own HTTP front end would, so an
/// id past the table must not leave the gateway: on the engine it is an out-of-range index, not a
/// refused request. The table is padded past the tokenizer's last id on most models and an
/// engine samples over the padded logits, so an id the tokenizer does not know is judged by the
/// table size the model's workers report (`embedding_rows`, read once, only then): below it the
/// id passes, as it would on the engine. When no worker reports a table the tokenizer's ids are
/// the bound, the one case where the gateway is stricter than the engine; a tokenizer that
/// reports no vocabulary either skips the bound rather than refusing every id.
pub(crate) fn first_unknown_token_id(
    tokenizer: &dyn Tokenizer,
    ids: &[u32],
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Option<UnknownTokenId> {
    let base_vocab = tokenizer.vocab_size();
    let known = |id: u32| known_to(tokenizer, base_vocab, id);
    let first_unknown = if base_vocab == 0 {
        // Without a vocabulary every id would be a lookup: the table alone judges them.
        0
    } else {
        ids.iter().position(|&id| !known(id))?
    };
    match embedding_rows() {
        Some(rows) => {
            let inside = |id: u32| usize::try_from(id).is_ok_and(|id| id < rows);
            let offset = ids[first_unknown..]
                .iter()
                .position(|&id| !inside(id) && !known(id))?;
            Some(UnknownTokenId {
                index: first_unknown + offset,
                id: ids[first_unknown + offset],
                bound: format!("its embedding table holds {rows} rows"),
            })
        }
        None if base_vocab == 0 => None,
        None => Some(UnknownTokenId {
            index: first_unknown,
            id: ids[first_unknown],
            bound: format!("its vocabulary holds {base_vocab} tokens plus its added tokens"),
        }),
    }
}

/// Refuses the first id of a prompt that can reach no row of the engine's embedding table
/// ([`first_unknown_token_id`]), naming its position, its value and the bound it missed.
/// `field` names the request field in the refusal: `input_ids`, `prompt`, or `prompt[b]` for
/// one list of a batch.
pub(crate) fn reject_unknown_token_id(
    tokenizer: &dyn Tokenizer,
    ids: &[u32],
    field: &str,
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Result<(), String> {
    match first_unknown_token_id(tokenizer, ids, embedding_rows) {
        None => Ok(()),
        Some(UnknownTokenId { index, id, bound }) => Err(format!(
            "{field}[{index}] = {id} is not a token of this model: {bound}"
        )),
    }
}

/// Refuses a `logit_bias` key that is no token of the model, judged as a pretokenized
/// prompt's ids are ([`first_unknown_token_id`]). Request validation proved the keys
/// non-negative integers; the engines' own front ends check them against the model, and the
/// gateway's engine connections bypass those front ends, so an unchecked id would reach the
/// sampler as an out-of-range index.
pub(crate) fn validate_logit_bias_token_ids(
    logit_bias: Option<&HashMap<String, f32>>,
    tokenizer: &dyn Tokenizer,
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Result<(), String> {
    let Some(bias) = logit_bias.filter(|bias| !bias.is_empty()) else {
        return Ok(());
    };
    let ids: Vec<u32> = bias.keys().filter_map(|key| key.parse().ok()).collect();
    match first_unknown_token_id(tokenizer, &ids, embedding_rows) {
        None => Ok(()),
        Some(UnknownTokenId { id, bound, .. }) => Err(format!(
            "Invalid key in 'logit_bias': '{id}' is not a token of this model: {bound}"
        )),
    }
}

/// The ids of a prompt the bound has passed that the tokenizer can decode, copied for the
/// prompt's text: all of them on a prompt the tokenizer knows throughout (the common one, one
/// comparison per id as in the bound), without the ids known only to the engine's embedding
/// table otherwise. Such an id sits in the padding past the tokenizer's last id, and no decoder
/// has text for it: the byte-level and HF decoders drop it on their own, the tiktoken decoder
/// refuses the whole decode. Filtered here, the text is the same on every decoder and the
/// prompt rides with its ids as given. A hole inside the range (see [`known_to`]) passes here
/// as it passes the bound; its decode is the decoder's refusal.
pub(crate) fn decodable_ids(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Vec<u32> {
    let base_vocab = tokenizer.vocab_size();
    let known = |id: u32| known_to(tokenizer, base_vocab, id);
    match ids.iter().position(|&id| !known(id)) {
        None => ids.to_vec(),
        Some(first) => {
            let mut kept = Vec::with_capacity(ids.len() - 1);
            kept.extend_from_slice(&ids[..first]);
            kept.extend(ids[first + 1..].iter().copied().filter(|&id| known(id)));
            kept
        }
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/stages/token_ids/tests.rs"]
mod tests;
