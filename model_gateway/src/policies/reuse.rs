//! Reuse rules over the shared index's lane answers.
//!
//! The index stores each independently evicted position set of a worker
//! as its own holder (`worker#lane`) and answers with the covered runs
//! of the query per holder plus the publisher's opaque lane metadata.
//! It knows nothing about attention kinds. This module is where the
//! engine semantics live: given a worker's lanes, the largest prefix
//! position every lane accepts at that same position, which is what
//! vLLM's hybrid coordinator computes natively
//! (`HybridKVCacheCoordinator.find_longest_cache_hit`, an iterative
//! fixed point over per-type managers):
//!
//! - full / MLA attention: cached contiguously from position 0 through
//!   the candidate;
//! - sliding window: the `W - 1` tokens before the candidate cached
//!   contiguously, rounded up to whole engine blocks the way the native
//!   manager counts them;
//! - Mamba / GDN (recurrent state): a saved state at exactly the
//!   candidate, i.e. a stored block ending there;
//! - candidates are aligned to the least common multiple of the lanes'
//!   engine blocks, as the native lookup aligns whole-block hits (the
//!   engine's optional finer partial-block hits are not representable
//!   in the event feed, whose partial blocks the converter drops).
//!
//! All arithmetic is in keyspace units (the index block). A holder
//! without lane metadata is one contiguous cache and scores by its
//! depth from 0, exactly as before lanes existed.

/// The publisher's description of a lane, decoded from the opaque
/// `lane_meta` bytes the bridge writes as `kind=..;window=..;block=..`
/// (tokens). Anything else is `None`: the lane cannot be scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneMeta {
    pub kind: LaneKind,
    /// Sliding-window width in tokens (window lanes only).
    pub window_tokens: u32,
    /// The engine block of this lane in tokens.
    pub block_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneKind {
    /// Cached contiguously from 0 (`full_attention`, `mla_attention`).
    Full,
    /// The recent history must be cached (`sliding_window`).
    Window,
    /// A saved state at exactly the candidate (`mamba`).
    Checkpoint,
}

impl LaneMeta {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(bytes).ok()?;
        let (mut kind, mut window, mut block) = (None, 0u32, 0u32);
        for field in text.split(';') {
            let (key, value) = field.split_once('=')?;
            match key {
                "kind" => {
                    kind = Some(match value {
                        "full_attention" | "mla_attention" => LaneKind::Full,
                        "sliding_window" => LaneKind::Window,
                        "mamba" => LaneKind::Checkpoint,
                        _ => return None,
                    });
                }
                "window" => window = value.parse().ok()?,
                "block" => block = value.parse().ok()?,
                _ => {}
            }
        }
        let kind = kind?;
        if block == 0 || (kind == LaneKind::Window && window == 0) {
            return None;
        }
        Some(Self {
            kind,
            window_tokens: window,
            block_tokens: block,
        })
    }
}

/// One lane of one worker as the index answered it.
#[derive(Debug, Clone)]
pub struct Lane<'a> {
    /// `None` for a holder published without lane metadata: a single
    /// contiguous cache.
    pub meta: Option<LaneMeta>,
    /// Covered `[start, end)` runs of the query in keyspace units,
    /// ascending, non-overlapping.
    pub intervals: &'a [(u32, u32)],
}

/// Split a holder name into its worker and lane parts
/// (`radix_index::engine::LANE_SEPARATOR`).
pub fn worker_of(holder: &str) -> (&str, Option<&str>) {
    match holder.split_once(radix_index::engine::LANE_SEPARATOR) {
        Some((worker, lane)) => (worker, Some(lane)),
        None => (holder, None),
    }
}

/// The largest query position (in keyspace units) every lane of a
/// worker accepts, or `None` when a lane cannot be scored (unknown
/// kind, engine block not tiling the unit). `request_tokens` bounds the
/// answer below the request's last token, which is always computed.
pub fn reusable_units(
    request_tokens: usize,
    unit_tokens: usize,
    lanes: &[Lane<'_>],
) -> Option<u32> {
    if lanes.is_empty() || unit_tokens == 0 {
        return None;
    }
    let limit = (request_tokens.saturating_sub(1) / unit_tokens) as u32;
    // A worker published as one contiguous cache: the depth from 0.
    if lanes.len() == 1 && lanes[0].meta.is_none() {
        return Some(covered_from_zero(lanes[0].intervals).min(limit));
    }
    let mut alignment: u32 = 1;
    let mut lookbacks: Vec<u32> = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let meta = lane.meta.as_ref()?;
        let block_units = (meta.block_tokens as usize / unit_tokens) as u32;
        if block_units == 0 || !(meta.block_tokens as usize).is_multiple_of(unit_tokens) {
            return None;
        }
        alignment = lcm(alignment, block_units);
        // The native window manager needs cdiv(W - 1, block) contiguous
        // engine blocks ending at the candidate (or reaching 0), and at
        // least the block ending there: a one-token window still resumes
        // only from a cached block.
        let lookback_blocks = meta
            .window_tokens
            .saturating_sub(1)
            .div_ceil(meta.block_tokens)
            .max(1);
        lookbacks.push(lookback_blocks.saturating_mul(block_units));
    }
    // Walk the aligned candidates down from the request's bound; the
    // first one every lane accepts is the answer (0 always is).
    let accepts = |p: u32| {
        lanes.iter().zip(&lookbacks).all(|(lane, &lookback)| {
            let Some(meta) = lane.meta.as_ref() else {
                return false;
            };
            let block_units = (meta.block_tokens as usize / unit_tokens) as u32;
            match meta.kind {
                LaneKind::Full => p <= covered_from_zero(lane.intervals),
                LaneKind::Window => covers(lane.intervals, p.saturating_sub(lookback), p),
                // A checkpoint lane's runs are whole engine blocks, so a
                // state exists at every block end inside a run, not only
                // at the run's end (adjacent blocks merge into one run).
                LaneKind::Checkpoint => lane.intervals.iter().any(|&(start, end)| {
                    p > start && p <= end && (p - start).is_multiple_of(block_units)
                }),
            }
        })
    };
    let mut p = limit - limit % alignment;
    while p > 0 {
        if accepts(p) {
            return Some(p);
        }
        p -= alignment;
    }
    Some(0)
}

/// Fold the index's per-holder answers into per-worker scores (worker
/// url, reusable keyspace units), descending, zero scores dropped. A
/// worker answered through lanes is scored by the reuse rules over its
/// lanes; a worker answered as one holder keeps its contiguous depth.
pub fn aggregate(
    answers: Vec<radix_index::client::HolderAnswer>,
    request_len: usize,
    unit: usize,
) -> Vec<(String, u32)> {
    // Per worker: its lanes (metadata, runs) and whether one of them
    // could not be read.
    type WorkerLanes = (String, Vec<(Option<LaneMeta>, Vec<(u32, u32)>)>, bool);
    let mut by_worker: Vec<WorkerLanes> = Vec::new();
    for answer in answers {
        let (worker, lane) = worker_of(&answer.holder);
        let at = match by_worker.iter().position(|(w, _, _)| w == worker) {
            Some(at) => at,
            None => {
                by_worker.push((worker.to_string(), Vec::new(), false));
                by_worker.len() - 1
            }
        };
        let entry = &mut by_worker[at];
        match lane.map(|_| LaneMeta::parse(&answer.lane_meta)) {
            // A lane whose description the consumer cannot read makes
            // the worker unscorable: better no claim than a wrong one.
            Some(None) => entry.2 = true,
            Some(Some(meta)) => entry.1.push((Some(meta), answer.intervals)),
            None => entry.1.push((None, answer.intervals)),
        }
    }
    let mut scores: Vec<(String, u32)> = by_worker
        .into_iter()
        .filter_map(|(worker, lanes, unreadable)| {
            if unreadable {
                return None;
            }
            let has_lanes = lanes.iter().any(|(m, _)| m.is_some());
            let views: Vec<Lane<'_>> = lanes
                .iter()
                // With lanes present the worker's own placement holder
                // (if any) adds no constraint and is not consulted.
                .filter(|(m, _)| !has_lanes || m.is_some())
                .map(|(m, iv)| Lane {
                    meta: m.clone(),
                    intervals: iv.as_slice(),
                })
                .collect();
            let units = reusable_units(request_len, unit, &views)?;
            (units > 0).then_some((worker, units))
        })
        .collect();
    scores.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores
}

/// Contiguous coverage from position 0 (the first run's end when it
/// starts at 0; adjacent runs are merged by the index already).
fn covered_from_zero(intervals: &[(u32, u32)]) -> u32 {
    let mut covered = 0;
    for &(start, end) in intervals {
        if start > covered {
            break;
        }
        covered = covered.max(end);
    }
    covered
}

/// Is `[from, to)` entirely covered?
fn covers(intervals: &[(u32, u32)], from: u32, to: u32) -> bool {
    if from >= to {
        return true;
    }
    let mut need = from;
    for &(start, end) in intervals {
        if end <= need {
            continue;
        }
        if start > need {
            return false;
        }
        need = end;
        if need >= to {
            return true;
        }
    }
    false
}

fn lcm(a: u32, b: u32) -> u32 {
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(kind: &str, window: u32, block: u32) -> LaneMeta {
        LaneMeta::parse(format!("kind={kind};window={window};block={block}").as_bytes()).unwrap()
    }

    #[test]
    fn metadata_parses_the_bridge_encoding_and_rejects_the_rest() {
        assert_eq!(
            meta("sliding_window", 1024, 16),
            LaneMeta {
                kind: LaneKind::Window,
                window_tokens: 1024,
                block_tokens: 16
            }
        );
        assert_eq!(meta("mla_attention", 0, 64).kind, LaneKind::Full);
        assert_eq!(meta("mamba", 0, 512).kind, LaneKind::Checkpoint);
        assert!(
            LaneMeta::parse(b"kind=sliding_window;window=0;block=16").is_none(),
            "a window lane needs a window"
        );
        assert!(LaneMeta::parse(b"kind=full_attention;window=0;block=0").is_none());
        assert!(
            LaneMeta::parse(b"kind=chunked_local;window=0;block=16").is_none(),
            "unknown kinds are unscorable"
        );
        assert!(LaneMeta::parse(b"garbage").is_none());
        assert!(LaneMeta::parse(b"").is_none());
        assert_eq!(worker_of("http://w1:8000#2"), ("http://w1:8000", Some("2")));
        assert_eq!(worker_of("http://w1:8000"), ("http://w1:8000", None));
    }

    #[test]
    fn a_holder_without_metadata_scores_by_its_depth_from_zero() {
        let lanes = [Lane {
            meta: None,
            intervals: &[(0, 5), (7, 9)],
        }];
        assert_eq!(reusable_units(16 * 9, 16, &lanes), Some(5));
        // The request's last token is always computed.
        assert_eq!(reusable_units(16 * 5, 16, &lanes), Some(4));
        let holed = [Lane {
            meta: None,
            intervals: &[(2, 5)],
        }];
        assert_eq!(reusable_units(16 * 9, 16, &holed), Some(0));
    }

    #[test]
    fn the_joint_answer_is_not_the_min_of_lane_maxima() {
        // Full attention through 8 units; a checkpoint lane at 6 and 10.
        // The lane maxima are 8 and 10; the common position is 6.
        let full = meta("full_attention", 0, 4);
        let ckpt = meta("mamba", 0, 4);
        let lanes = [
            Lane {
                meta: Some(full),
                intervals: &[(0, 8)],
            },
            Lane {
                meta: Some(ckpt),
                intervals: &[(5, 6), (9, 10)],
            },
        ];
        assert_eq!(reusable_units(4 * 12, 4, &lanes), Some(6));
    }

    fn answer(
        holder: &str,
        meta: &str,
        intervals: &[(u32, u32)],
    ) -> radix_index::client::HolderAnswer {
        radix_index::client::HolderAnswer {
            holder: holder.to_string(),
            matched_blocks: intervals.first().filter(|r| r.0 == 0).map_or(0, |r| r.1),
            total_blocks: 0,
            event_fed: !meta.is_empty(),
            intervals: intervals.to_vec(),
            lane_meta: meta.as_bytes().to_vec(),
        }
    }

    #[test]
    fn aggregate_scores_lane_workers_by_the_rules_and_plain_workers_by_depth() {
        let answers = vec![
            // w1: full through 8, window (W=9, block 4 = 2 blocks) covers
            // [2,8), checkpoint at 6 -> 6 units.
            answer("w1#0", "kind=full_attention;window=0;block=4", &[(0, 8)]),
            answer("w1#1", "kind=sliding_window;window=9;block=4", &[(2, 8)]),
            answer("w1#2", "kind=mamba;window=0;block=4", &[(5, 6)]),
            // w1's own placement holder is ignored once lanes exist.
            answer("w1", "", &[(0, 8)]),
            // w2: plain contiguous holder, depth 7.
            answer("w2", "", &[(0, 7)]),
            // w3: holed plain holder scores 0 and is dropped.
            answer("w3", "", &[(1, 7)]),
            // w4: a lane the consumer cannot read makes it unscorable.
            answer("w4#0", "kind=full_attention;window=0;block=4", &[(0, 8)]),
            answer("w4#1", "kind=chunked_local;window=0;block=4", &[(0, 8)]),
        ];
        assert_eq!(
            aggregate(answers, 4 * 12, 4),
            vec![("w2".to_string(), 7), ("w1".to_string(), 6)]
        );
    }

    /// Every case of the native-lookup fixture (vLLM's own
    /// `find_longest_cache_hit` expectations, recorded in #2497) that the
    /// event feed can represent, i.e. whole-block hits. Cached block
    /// ends become the lane's covered units.
    #[test]
    fn native_lookup_fixture_agrees_with_the_rules() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/kv-cache-groups.json")).unwrap();
        let mut checked = 0;
        let mut skipped = 0;
        for case in data["cases"].as_array().unwrap() {
            if case["partial_hash_hits"].as_bool().unwrap() {
                skipped += 1;
                continue;
            }
            let unit = case["hash_unit"].as_u64().unwrap() as usize;
            let request_tokens = case["request_tokens"].as_u64().unwrap() as usize;
            let expected = case["expected_hit_tokens"].as_u64().unwrap();
            type OwnedLane = (Option<LaneMeta>, Vec<(u32, u32)>);
            let mut owned: Vec<OwnedLane> = Vec::new();
            for g in case["groups"].as_array().unwrap() {
                let block = g["block_size"].as_u64().unwrap() as u32;
                let kind = match g["kind"].as_str().unwrap() {
                    "full" => "full_attention",
                    "swa" => "sliding_window",
                    "mamba" => "mamba",
                    other => panic!("unknown fixture kind {other}"),
                };
                let window = g["window"].as_u64().unwrap_or(0) as u32;
                let block_units = block / unit as u32;
                // A cached end on a block boundary is a whole engine
                // block; other ends are the engine's fine-grained
                // partial-block hashes, which whole-block lookup (and the
                // event feed) never sees.
                let mut intervals: Vec<(u32, u32)> = g["cached"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|end| end.as_u64().unwrap() as u32)
                    .filter(|end| end % block == 0)
                    .map(|end| {
                        let end_units = end / unit as u32;
                        (end_units - block_units, end_units)
                    })
                    .collect();
                intervals.sort_unstable();
                // Merge adjacent runs the way the index reports them.
                let mut merged: Vec<(u32, u32)> = Vec::new();
                for (s, e) in intervals {
                    match merged.last_mut() {
                        Some(last) if last.1 >= s => last.1 = last.1.max(e),
                        _ => merged.push((s, e)),
                    }
                }
                owned.push((Some(meta(kind, window, block)), merged));
            }
            let lanes: Vec<Lane<'_>> = owned
                .iter()
                .map(|(m, iv)| Lane {
                    meta: m.clone(),
                    intervals: iv.as_slice(),
                })
                .collect();
            let got =
                reusable_units(request_tokens, unit, &lanes).map(|u| u64::from(u) * unit as u64);
            assert_eq!(got, Some(expected), "case {}: {case}", case["name"]);
            checked += 1;
        }
        assert!(checked >= 120, "checked {checked}, skipped {skipped}");
    }
}
