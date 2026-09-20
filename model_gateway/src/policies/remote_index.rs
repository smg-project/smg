//! Remote radix index access (`--kv-indexer-url`): a per-process client
//! handle plus the routing-time query result carriers. Lives in the
//! policy layer (not a router) because the cache-aware policy owns the
//! index query/publish, so every router gets it through one uniform
//! call. With the flag unset the handle is `None` and every caller's
//! fast path is a `None` check.

use std::{sync::Arc, time::Duration};

use radix_index::client::{QueryOutcome, RemoteIndex};

/// Hard deadline for the routing-time overlap query; a miss falls back
/// to expected-wait for that one decision.
pub(crate) const QUERY_DEADLINE: Duration = Duration::from_millis(2);

/// String-mode block size, in raw bytes. Fixed for now: string-mode
/// (`SymbolKind::Bytes`) routing hashes raw request text into blocks of
/// this many bytes, a separate keyspace from the token tree. Kept a
/// constant rather than a config knob until the byte-affinity approach
/// is signed off (see the string-mode notes on the PR).
pub(crate) const BYTE_BLOCK: usize = 256;

/// Rough bytes-per-token ratio used to express a byte block in token
/// units where the policy's decay math needs one (its backlog input is
/// a token count). An estimate, not a tokenizer: the decay only reorders
/// equal-overlap holders, and string mode is coarser than token mode by
/// construction.
pub(crate) const APPROX_BYTES_PER_TOKEN: usize = 4;

/// The connected remote-index client plus the keyspace block size it was
/// configured for. One per process, owned by `AppContext` and shared
/// with the `PolicyRegistry`.
pub struct RemoteIndexHandle {
    client: Arc<RemoteIndex>,
    block_size: usize,
}

impl RemoteIndexHandle {
    /// `block_size` is the KEYSPACE block size — the engine-side page
    /// size the index was fed at (worker events / bridge `--block-size`),
    /// not the routing block.
    pub fn connect(url: &str, block_size: usize) -> Arc<Self> {
        Arc::new(Self {
            client: RemoteIndex::connect(url.to_string()),
            block_size: block_size.max(1),
        })
    }

    pub(crate) fn client(&self) -> &RemoteIndex {
        &self.client
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }
}

impl std::fmt::Debug for RemoteIndexHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteIndexHandle")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

/// What the routing-time query resolved, kept on the request context for
/// the placement publish and the response echo headers.
#[derive(Debug, Clone)]
pub(crate) struct IndexPrediction {
    /// remote_hit | remote_unscorable | remote_empty | remote_timeout
    /// | remote_disconnected
    pub source: &'static str,
    /// Per-worker (url, reusable prefix blocks) as folded by
    /// [`fold_answers`], descending, zero scores dropped. Empty on
    /// anything but a scorable hit.
    pub scores: Vec<(String, u32)>,
    pub block_size: usize,
    /// The request prefix's content hashes (block-aligned from 0),
    /// republished as the placement chain after successful dispatch.
    pub content_hashes: Vec<u64>,
    pub model: String,
    /// `true` when this prediction is string-mode (`SymbolKind::Bytes`):
    /// the hashes are over raw request bytes and `block_size` is the byte
    /// block, so the placement must publish under the `Bytes` keyspace.
    /// `false` is the token keyspace.
    pub bytes: bool,
    /// The request's cache partition, carried so the placement publish
    /// keys the same chain the query asked under. A query that
    /// partitions and a publish that does not would file the placement
    /// where no later query can find it.
    pub namespace: Option<crate::policies::CacheNamespace>,
}

impl IndexPrediction {
    /// Predicted cached tokens on `worker_url` — the echo header the gRPC
    /// router surfaces so the harness can separate index error from
    /// policy spill.
    pub(crate) fn predicted_tokens_for(&self, worker_url: &str) -> usize {
        self.scores
            .iter()
            .find(|(url, _)| url == worker_url)
            .map_or(0, |(_, blocks)| *blocks as usize * self.block_size)
    }

    pub(crate) fn source(&self) -> &'static str {
        self.source
    }
}

pub(crate) fn outcome_label(outcome: &QueryOutcome) -> &'static str {
    match outcome {
        QueryOutcome::Scores(_) => "remote_hit",
        QueryOutcome::Empty => "remote_empty",
        QueryOutcome::Timeout => "remote_timeout",
        QueryOutcome::Disconnected => "remote_disconnected",
    }
}

/// An answer came back but nothing in it could be scored, so routing
/// falls to load alone. Counted apart from `remote_hit`: otherwise a
/// fleet whose every answer folds to nothing reads as a fleet answering
/// perfectly, and the hit rate stops measuring whether the index is
/// steering anything.
pub(crate) const UNSCORABLE: &str = "remote_unscorable";

/// The request was too short to fill one keyspace block, so no query
/// went out. Still counted, and still a miss for the caller: a prompt
/// under one block is a large share of chat traffic, and letting it
/// fall through to plain selection would fill a local prefix tree with
/// exactly the per-gateway view the shared index exists to remove.
pub(crate) const TOO_SHORT_TO_HASH: &str = "remote_too_short";

/// Fold the index's per-holder answers into per-worker scores (worker
/// url, reusable PROMPT blocks), descending, zero scores dropped.
///
/// Only `matched_blocks` counts — the index's contiguous depth from
/// position 0. A run that begins past 0 is a middle hit the engine
/// cannot resume from, so crediting it would steer the request to a
/// worker that then recomputes the whole prefix anyway. `keyed_len`
/// bounds the score below the request's own last unit, which is always
/// computed and so is never reusable however deep the answer.
///
/// `marker_units` is how much of the key is the cache-namespace marker
/// rather than prompt. It comes off every score, so a worker that
/// shares only the namespace scores nothing and the echoed prediction
/// stays in prompt units whether or not the request was partitioned.
pub(crate) fn fold_answers(
    answers: Vec<radix_index::client::HolderAnswer>,
    keyed_len: usize,
    unit: usize,
    marker_units: u32,
) -> Vec<(String, u32)> {
    if unit == 0 {
        return Vec::new();
    }
    let limit = (keyed_len.saturating_sub(1) / unit) as u32;
    let mut scores: Vec<(String, u32)> = Vec::with_capacity(answers.len());
    for answer in answers {
        let depth = answer
            .matched_blocks
            .min(limit)
            .saturating_sub(marker_units);
        if depth == 0 {
            continue;
        }
        // Two answers naming one worker would otherwise let the blend
        // see it twice and weigh it twice; keep the deeper claim.
        match scores.iter_mut().find(|(url, _)| *url == answer.holder) {
            Some(entry) => entry.1 = entry.1.max(depth),
            None => scores.push((answer.holder, depth)),
        }
    }
    // Sorted here rather than trusting the answer order, so the blend's
    // input ordering is a property of this function and a change to the
    // index's reply order cannot quietly reorder equal-overlap workers.
    scores.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores
}

#[cfg(test)]
mod tests {
    use radix_index::client::HolderAnswer;

    use super::*;

    fn answer(holder: &str, matched_blocks: u32) -> HolderAnswer {
        HolderAnswer {
            holder: holder.to_string(),
            matched_blocks,
            total_blocks: 0,
            event_fed: true,
            intervals: if matched_blocks == 0 {
                Vec::new()
            } else {
                vec![(0, matched_blocks)]
            },
            lane_meta: Vec::new(),
        }
    }

    #[test]
    fn deeper_answers_sort_first_and_empty_ones_drop_out() {
        let folded = fold_answers(
            vec![
                answer("http://a", 4),
                answer("http://b", 0),
                answer("http://c", 7),
            ],
            1024,
            16,
            0,
        );
        assert_eq!(
            folded,
            vec![("http://c".to_string(), 7), ("http://a".to_string(), 4)]
        );
    }

    #[test]
    fn the_requests_own_last_unit_is_never_reusable() {
        // 64 tokens at a 16-token block is four blocks, but the last one
        // is what this request is about to compute, so three is the most
        // any worker can be credited with however deep it answered.
        let folded = fold_answers(vec![answer("http://a", 9)], 64, 16, 0);
        assert_eq!(folded, vec![("http://a".to_string(), 3)]);
    }

    #[test]
    fn one_worker_answering_twice_is_weighed_once() {
        let folded = fold_answers(
            vec![answer("http://a", 2), answer("http://a", 6)],
            1024,
            16,
            0,
        );
        assert_eq!(folded, vec![("http://a".to_string(), 6)]);
    }

    #[test]
    fn a_prompt_shorter_than_one_unit_credits_nobody() {
        assert!(fold_answers(vec![answer("http://a", 5)], 8, 16, 0).is_empty());
    }

    #[test]
    fn the_namespace_marker_is_not_a_reusable_prompt_block() {
        // One marker block plus six prompt blocks matched. Sharing only
        // the partition is not sharing a prompt, so the worker that
        // matched the marker alone drops out entirely.
        let folded = fold_answers(
            vec![answer("http://a", 7), answer("http://b", 1)],
            1024,
            16,
            1,
        );
        assert_eq!(folded, vec![("http://a".to_string(), 6)]);
    }
}
