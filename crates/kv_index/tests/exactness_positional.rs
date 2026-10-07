//! Exactness harness: the production `PositionalIndexer` against the `ReferenceIndexer`.
//!
//! A seeded corpus of stores, extensions, divergent siblings, tail and whole-chain removals,
//! clears and worker removals is replayed into both indexers through the same per-worker
//! `WorkerBlockMap` handling the gateway's `KvEventMonitor` uses. After every round of events,
//! lookups built from live chains and from mutated chains must score identically in both; at the
//! end, the production index must hold exactly the reference's blocks.
//!
//! Two corpora run. One evicts tails and whole chains only, which is what a radix-tree cache such
//! as SGLang's produces (it evicts leaves). The other also evicts single blocks from the middle of
//! chains the worker keeps, which vLLM produces: a request that re-hits a shared prefix takes those
//! blocks out of the free queue and, when it finishes, re-queues them behind the unshared tail of
//! an earlier request, so that request's middle blocks fall out of the LRU before its later blocks
//! do. The engine's prefix match stops at the hole, so a worker's score must end there too. Scale
//! with `KV_INDEX_EXACTNESS_EVENTS` (default 20000) and `KV_INDEX_EXACTNESS_SEED`.
#![expect(clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use kv_index::{ContentHash, PositionalIndexer, ReferenceIndexer, SequenceHash, WorkerBlockMap};
use rustc_hash::FxHashMap;

mod common;
use common::{blocks_of, content, Rng};

struct Held {
    worker: u32,
    contents: Vec<ContentHash>,
}

/// A chain as it was when one of its middle blocks was evicted; its blocks after `hole` stay in
/// the index but are unreachable through the chain, so lookups along it must stop at `hole`.
struct Holed {
    worker: u32,
    contents: Vec<ContentHash>,
    hole: usize,
}

struct Harness {
    production: PositionalIndexer,
    reference: ReferenceIndexer,
    maps: FxHashMap<u32, WorkerBlockMap>,
    workers: Vec<u32>,
    held: Vec<Held>,
    /// Whether the corpus evicts middle blocks (see the module doc).
    holes: bool,
    holed: Vec<Holed>,
    prompts: Vec<Vec<ContentHash>>,
    next_stream: u64,
    next_worker: u32,
    rng: Rng,
    events: usize,
    stored_blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueryKind {
    Exact,
    Prefix,
    MiddleReplaced,
    SuffixReplaced,
    Extended,
    Unknown,
    /// A full chain through a block the worker evicted from its middle.
    AcrossHole,
    /// A prefix of such a chain that ends after the hole.
    PastHole,
}

#[derive(Default)]
struct Mismatches {
    by_kind: BTreeMap<QueryKind, usize>,
    issued: BTreeMap<QueryKind, usize>,
    examples: Vec<String>,
    lookups: usize,
}

impl Harness {
    fn new(seed: u64, jump_size: usize, workers: usize, holes: bool) -> Self {
        let mut rng = Rng::new(seed);
        let production = PositionalIndexer::new(jump_size);
        let mut harness = Self {
            production,
            reference: ReferenceIndexer::new(),
            maps: FxHashMap::default(),
            workers: Vec::new(),
            held: Vec::new(),
            holes,
            holed: Vec::new(),
            prompts: Vec::new(),
            next_stream: 1,
            next_worker: 0,
            rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15),
            events: 0,
            stored_blocks: 0,
        };
        for _ in 0..workers {
            harness.add_worker();
        }
        let prompt_count = rng.range(3, 8);
        for _ in 0..prompt_count {
            let len = rng.range(8, 48);
            let stream = harness.fresh_stream();
            harness
                .prompts
                .push((0..len).map(|p| content(stream, p)).collect());
        }
        harness
    }

    fn fresh_stream(&mut self) -> u64 {
        self.next_stream += 1;
        self.next_stream
    }

    fn add_worker(&mut self) -> u32 {
        let url = format!("http://worker-{}:8000", self.next_worker);
        self.next_worker += 1;
        let id = self.production.intern_worker(&url).expect("worker id");
        self.maps.insert(id, WorkerBlockMap::default());
        self.workers.push(id);
        id
    }

    fn random_worker(&mut self) -> u32 {
        self.workers[self.rng.below(self.workers.len())]
    }

    /// A new user turn: a prompt prefix plus fresh blocks.
    fn new_chain(&mut self) -> Vec<ContentHash> {
        let prompt = &self.prompts[self.rng.below(self.prompts.len())];
        let mut contents = prompt.clone();
        let turn = self.rng.range(4, 32);
        let stream = self.fresh_stream();
        contents.extend((0..turn).map(|p| content(stream, p)));
        contents
    }

    /// Store `contents` on `worker` as the engine would: only the suffix the worker does not hold,
    /// after the last block it does hold. A fully held chain is re-stored for its last block, which
    /// exercises the duplicate-store path.
    fn store(&mut self, worker: u32, contents: Vec<ContentHash>) {
        let blocks = blocks_of(&contents);
        let held = self.maps.get(&worker).expect("worker map");
        let mut known = 0;
        while known < blocks.len() && held.contains_key(&blocks[known].seq_hash) {
            known += 1;
        }
        let start = if known == blocks.len() {
            blocks.len() - 1
        } else {
            known
        };
        let parent = if start == 0 {
            None
        } else {
            Some(blocks[start - 1].seq_hash)
        };
        let map = self.maps.get_mut(&worker).expect("worker map");
        let produced = self
            .production
            .apply_stored(worker, &blocks[start..], parent, map);
        let referenced = self
            .reference
            .apply_stored(worker, &blocks[start..], parent);
        assert_eq!(
            produced.is_ok(),
            referenced.is_ok(),
            "store outcome differs: production {produced:?}, reference {referenced:?}"
        );
        if produced.is_ok() {
            self.stored_blocks += blocks.len() - start;
            self.held.push(Held { worker, contents });
        }
        self.events += 1;
    }

    fn pick_held(&mut self) -> Option<usize> {
        if self.held.is_empty() {
            return None;
        }
        let index = self.rng.below(self.held.len());
        if self.maps.contains_key(&self.held[index].worker) {
            Some(index)
        } else {
            self.held.swap_remove(index);
            None
        }
    }

    /// Evict a tail: the chain's blocks from a random position on, together with the same
    /// positions of every other held chain of that worker that runs through them. That keeps every
    /// chain the worker holds free of holes, which is what engines with prefix caching produce
    /// (a block is only reusable through its predecessors) and what the production jump search
    /// relies on when it skips from one landing to the next.
    fn remove_tail(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let keep = self.rng.below(contents.len());
        let mut hashes: Vec<SequenceHash> = Vec::new();
        for held in self.held.iter_mut().filter(|h| h.worker == worker) {
            let shared = held
                .contents
                .iter()
                .zip(&contents)
                .take_while(|(a, b)| a == b)
                .count();
            if shared > keep {
                let blocks = blocks_of(&held.contents);
                hashes.extend(blocks[keep..].iter().map(|b| b.seq_hash));
                held.contents.truncate(keep);
            }
        }
        hashes.sort_unstable_by_key(|h| h.0);
        hashes.dedup();
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_removed(worker, &hashes, map);
        self.reference.apply_removed(worker, &hashes);
        self.held.retain(|h| !h.contents.is_empty());
        self.events += 1;
    }

    /// Evict a whole conversation: the chain's blocks beyond the longest prefix it shares with
    /// another held chain of the same worker (the shared prefix stays, as it would in an engine
    /// where the sibling still references it), so no chain of the worker is left with a hole.
    fn remove_chain(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let shared = self
            .held
            .iter()
            .enumerate()
            .filter(|(i, h)| *i != index && h.worker == worker)
            .map(|(_, h)| {
                h.contents
                    .iter()
                    .zip(&contents)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .max()
            .unwrap_or(0);
        let blocks = blocks_of(&contents);
        let hashes: Vec<SequenceHash> = blocks[shared..].iter().map(|b| b.seq_hash).collect();
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_removed(worker, &hashes, map);
        self.reference.apply_removed(worker, &hashes);
        self.held.swap_remove(index);
        self.events += 1;
    }

    /// Evict one block from the middle of a chain while the blocks after it stay (the vLLM case in
    /// the module doc). Every held chain of the worker that runs through the block loses it; the
    /// full chains are kept aside so lookups can run across the hole.
    fn remove_middle(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        if self.held[index].contents.len() < 3 {
            return;
        }
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let hole = self.rng.range(1, contents.len() - 2);
        let evicted = blocks_of(&contents)[hole].seq_hash;
        for held in self.held.iter_mut().filter(|h| h.worker == worker) {
            let shared = held
                .contents
                .iter()
                .zip(&contents)
                .take_while(|(a, b)| a == b)
                .count();
            if shared > hole {
                self.holed.push(Holed {
                    worker,
                    contents: held.contents.clone(),
                    hole,
                });
                held.contents.truncate(hole);
            }
        }
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_removed(worker, &[evicted], map);
        self.reference.apply_removed(worker, &[evicted]);
        self.events += 1;
    }

    fn pick_holed(&mut self) -> Option<usize> {
        if self.holed.is_empty() {
            return None;
        }
        let index = self.rng.below(self.holed.len());
        if self.maps.contains_key(&self.holed[index].worker) {
            Some(index)
        } else {
            self.holed.swap_remove(index);
            None
        }
    }

    fn clear_worker(&mut self) {
        let worker = self.random_worker();
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_cleared(worker, map);
        self.reference.apply_cleared(worker);
        self.held.retain(|h| h.worker != worker);
        self.events += 1;
    }

    fn remove_worker(&mut self) {
        if self.workers.len() < 2 {
            return;
        }
        let position = self.rng.below(self.workers.len());
        let worker = self.workers.swap_remove(position);
        let map = self.maps.remove(&worker).expect("worker map");
        self.production.remove_worker(worker, map);
        self.reference.remove_worker(worker);
        self.held.retain(|h| h.worker != worker);
        self.add_worker();
        self.events += 1;
    }

    fn step(&mut self) {
        let roll = self.rng.below(1000);
        match roll {
            0..=399 => {
                let chain = self.new_chain();
                let worker = self.random_worker();
                self.store(worker, chain);
            }
            400..=649 => {
                // Extend a held chain by another turn on the same worker.
                let Some(index) = self.pick_held() else {
                    return;
                };
                let worker = self.held[index].worker;
                let mut contents = self.held[index].contents.clone();
                let turn = self.rng.range(4, 32);
                let stream = self.fresh_stream();
                contents.extend((0..turn).map(|p| content(stream, p)));
                self.store(worker, contents);
            }
            650..=799 => {
                // A sibling that diverges at a random position, including 1 and the last block,
                // stored on a random worker (often another one, which shares the prefix blocks).
                let Some(index) = self.pick_held() else {
                    return;
                };
                let base = &self.held[index].contents;
                if base.len() < 2 {
                    return;
                }
                let divergence = match self.rng.below(10) {
                    0 => 1,
                    1 => base.len() - 1,
                    2 => base.len(),
                    _ => self.rng.range(1, base.len()),
                };
                let mut contents: Vec<ContentHash> = base[..divergence].to_vec();
                let turn = self.rng.range(1, 24);
                let stream = self.fresh_stream();
                contents.extend((0..turn).map(|p| content(stream, p)));
                let worker = self.random_worker();
                self.store(worker, contents);
            }
            800..=899 => {
                if self.holes && self.rng.chance(1, 2) {
                    self.remove_middle();
                } else {
                    self.remove_tail();
                }
            }
            900..=949 => self.remove_chain(),
            950..=984 => {
                if self.workers.len() < 64 && self.rng.chance(1, 4) {
                    self.add_worker();
                } else {
                    let Some(index) = self.pick_held() else {
                        return;
                    };
                    let contents = self.held[index].contents.clone();
                    let worker = self.random_worker();
                    self.store(worker, contents);
                }
            }
            985..=994 => self.clear_worker(),
            _ => self.remove_worker(),
        }
    }

    fn production_scores(&self, query: &[ContentHash]) -> BTreeMap<u32, u32> {
        self.production
            .find_matches(query, false)
            .scores
            .into_iter()
            .collect()
    }

    fn check_lookup(&mut self, kind: QueryKind, query: Vec<ContentHash>, out: &mut Mismatches) {
        if query.is_empty() {
            return;
        }
        out.lookups += 1;
        *out.issued.entry(kind).or_default() += 1;
        let produced = self.production_scores(&query);
        let expected = self.reference.find_matches(&query);
        if produced != expected {
            *out.by_kind.entry(kind).or_default() += 1;
            if out.examples.len() < 12 {
                let diff: Vec<String> = expected
                    .keys()
                    .chain(produced.keys())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .filter(|w| produced.get(w) != expected.get(w))
                    .map(|w| {
                        format!(
                            "worker {w}: production {:?}, reference {:?}",
                            produced.get(w),
                            expected.get(w)
                        )
                    })
                    .collect();
                out.examples.push(format!(
                    "{kind:?} query of {} blocks after {} events: {}",
                    query.len(),
                    self.events,
                    diff.join("; ")
                ));
            }
        }
        // early_exit reports exactly the workers that match position 0, each with score 1.
        let early: BTreeMap<u32, u32> = self
            .production
            .find_matches(&query, true)
            .scores
            .into_iter()
            .collect();
        let expected_early: BTreeMap<u32, u32> = expected.keys().map(|&w| (w, 1)).collect();
        if early != expected_early {
            *out.by_kind.entry(kind).or_default() += 1;
            if out.examples.len() < 12 {
                out.examples.push(format!(
                    "{kind:?} early-exit query of {} blocks: production {early:?}, reference {expected_early:?}",
                    query.len()
                ));
            }
        }
    }

    fn lookups(&mut self, out: &mut Mismatches) {
        for _ in 0..8 {
            let Some(index) = self.pick_held() else {
                return;
            };
            let base = self.held[index].contents.clone();
            if base.is_empty() {
                continue;
            }
            self.check_lookup(QueryKind::Exact, base.clone(), out);
            let prefix_len = self.rng.range(1, base.len());
            self.check_lookup(QueryKind::Prefix, base[..prefix_len].to_vec(), out);
            if base.len() >= 2 {
                let mut middle = base.clone();
                let at = self.rng.range(1, base.len() - 1);
                let stream = self.fresh_stream();
                middle[at] = content(stream, 0);
                self.check_lookup(QueryKind::MiddleReplaced, middle, out);
                let mut suffix = base[..self.rng.range(1, base.len() - 1)].to_vec();
                let stream = self.fresh_stream();
                let extra = self.rng.range(1, 16);
                suffix.extend((0..extra).map(|p| content(stream, p)));
                self.check_lookup(QueryKind::SuffixReplaced, suffix, out);
            }
            let mut extended = base.clone();
            let stream = self.fresh_stream();
            let extra = self.rng.range(1, 40);
            extended.extend((0..extra).map(|p| content(stream, p)));
            self.check_lookup(QueryKind::Extended, extended, out);
        }
        if self.holes {
            for _ in 0..4 {
                let Some(index) = self.pick_holed() else {
                    break;
                };
                let (full, hole) = {
                    let holed = &self.holed[index];
                    (holed.contents.clone(), holed.hole)
                };
                let past = self.rng.range(hole + 1, full.len());
                self.check_lookup(QueryKind::PastHole, full[..past].to_vec(), out);
                self.check_lookup(QueryKind::AcrossHole, full, out);
            }
        }
        let stream = self.fresh_stream();
        let len = self.rng.range(1, 32);
        let unknown: Vec<ContentHash> = (0..len).map(|p| content(stream, p)).collect();
        self.check_lookup(QueryKind::Unknown, unknown, out);
    }

    fn production_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        self.production.debug_blocks().into_iter().collect()
    }
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn run_corpus(
    seed: u64,
    jump_size: usize,
    workers: usize,
    events: usize,
    holes: bool,
) -> (Harness, Mismatches) {
    let mut harness = Harness::new(seed, jump_size, workers, holes);
    let mut mismatches = Mismatches::default();
    while harness.events < events {
        harness.step();
        if harness.events.is_multiple_of(256) {
            harness.lookups(&mut mismatches);
        }
    }
    harness.lookups(&mut mismatches);
    (harness, mismatches)
}

fn assert_exact(harness: &Harness, mismatches: &Mismatches, label: &str) {
    let produced = harness.production_blocks();
    let expected = harness.reference.blocks();
    let missing: Vec<_> = expected.difference(&produced).take(5).collect();
    let phantom: Vec<_> = produced.difference(&expected).take(5).collect();
    assert!(
        missing.is_empty() && phantom.is_empty(),
        "{label}: index content differs from the reference after {} events / {} stored blocks: \
         {} reference blocks, {} production blocks; missing e.g. {missing:?}; phantom e.g. {phantom:?}",
        harness.events,
        harness.stored_blocks,
        expected.len(),
        produced.len()
    );
    assert!(
        mismatches.by_kind.is_empty(),
        "{label}: {} of {} lookups scored differently from the reference, by query kind {:?} \
         (issued {:?}); examples:\n{}",
        mismatches.by_kind.values().sum::<usize>(),
        mismatches.lookups,
        mismatches.by_kind,
        mismatches.issued,
        mismatches.examples.join("\n")
    );
}

#[test]
fn replayed_corpus_matches_the_reference() {
    let events = env_or("KV_INDEX_EXACTNESS_EVENTS", 20_000) as usize;
    let seed = env_or("KV_INDEX_EXACTNESS_SEED", 20261005);
    for (jump, workers) in [(8usize, 16usize), (64, 2), (3, 64)] {
        let (harness, mismatches) = run_corpus(seed ^ jump as u64, jump, workers, events, false);
        assert_exact(
            &harness,
            &mismatches,
            &format!("jump {jump}, {workers} workers, seed {seed}"),
        );
        assert!(
            harness.stored_blocks > events,
            "corpus too small to mean anything: {} blocks for {} events",
            harness.stored_blocks,
            events
        );
    }
}

/// The vLLM corpus: middle blocks get evicted while later blocks stay, and lookups run across and
/// past the holes. Before every position was verified, the jump search landed past a hole on an
/// entry that still named the worker and scored the whole chain.
#[test]
fn replayed_corpus_with_holes_matches_the_reference() {
    let events = env_or("KV_INDEX_EXACTNESS_EVENTS", 20_000) as usize;
    let seed = env_or("KV_INDEX_EXACTNESS_SEED", 20261005);
    for (jump, workers) in [(8usize, 16usize), (64, 2), (3, 64)] {
        let (harness, mismatches) = run_corpus(seed ^ jump as u64, jump, workers, events, true);
        assert_exact(
            &harness,
            &mismatches,
            &format!("holes, jump {jump}, {workers} workers, seed {seed}"),
        );
        let across = mismatches
            .issued
            .get(&QueryKind::AcrossHole)
            .copied()
            .unwrap_or(0);
        assert!(
            across >= 64,
            "hole corpus too small to mean anything: {across} lookups across holes"
        );
    }
}

/// A worker holds [A ..= J] (ten blocks) and evicts E while F ..= J stay, as vLLM's free queue
/// can order it. A request for the full chain hits A ..= D in the engine and recomputes the rest,
/// so the score is 4; the entries at F ..= J still name the worker and must not count.
#[test]
fn evicted_middle_block_ends_the_match() {
    let index = PositionalIndexer::new(8);
    let worker = index
        .intern_worker("http://worker-0:8000")
        .expect("worker id");
    let mut map = WorkerBlockMap::default();
    let contents: Vec<ContentHash> = (0..10).map(|p| content(7, p)).collect();
    let blocks = blocks_of(&contents);
    index
        .apply_stored(worker, &blocks, None, &mut map)
        .expect("store");
    index.apply_removed(worker, &[blocks[4].seq_hash], &mut map);
    let scores = index.find_matches(&contents, false).scores;
    assert_eq!(scores.get(&worker).copied(), Some(4), "scores {scores:?}");
    let past = index.find_matches(&contents[..7], false).scores;
    assert_eq!(past.get(&worker).copied(), Some(4), "scores {past:?}");
    let before = index.find_matches(&contents[..4], false).scores;
    assert_eq!(before.get(&worker).copied(), Some(4), "scores {before:?}");
}

/// A worker holds [A, B, C]; a request [A, X, C] shares only A with it. The jump search lands on
/// position 2, where the single stored entry for C must not be taken as a match without its prefix
/// hash: the request's chain differs from position 1 on.
#[test]
fn single_entry_shortcut_must_not_hide_a_divergence_at_the_tail() {
    let production = PositionalIndexer::new(8);
    let mut reference = ReferenceIndexer::new();
    let worker = production.intern_worker("http://w:8000").unwrap();
    let mut map = WorkerBlockMap::default();
    let held: Vec<ContentHash> = (0..3).map(|p| content(1, p)).collect();
    let blocks = blocks_of(&held);
    production
        .apply_stored(worker, &blocks, None, &mut map)
        .unwrap();
    reference.apply_stored(worker, &blocks, None).unwrap();
    let query = vec![held[0], content(2, 0), held[2]];
    let produced: BTreeMap<u32, u32> = production
        .find_matches(&query, false)
        .scores
        .into_iter()
        .collect();
    assert_eq!(reference.find_matches(&query).get(&worker), Some(&1));
    assert_eq!(
        produced.get(&worker),
        Some(&1),
        "scored {produced:?}, expected 1 (A only)"
    );
}

/// A worker holds a 20-block chain; the request replaces block 10. The jump landings at 16 and 19
/// find single entries whose content matches but whose prefix hash belongs to the stored chain, not
/// to the request's chain, which diverged at 10.
#[test]
fn single_entry_landing_must_not_hide_an_earlier_divergence() {
    let production = PositionalIndexer::new(8);
    let mut reference = ReferenceIndexer::new();
    let worker = production.intern_worker("http://w:8000").unwrap();
    let mut map = WorkerBlockMap::default();
    let held: Vec<ContentHash> = (0..20).map(|p| content(3, p)).collect();
    let blocks = blocks_of(&held);
    production
        .apply_stored(worker, &blocks, None, &mut map)
        .unwrap();
    reference.apply_stored(worker, &blocks, None).unwrap();
    let mut query = held.clone();
    query[10] = content(4, 0);
    let produced: BTreeMap<u32, u32> = production
        .find_matches(&query, false)
        .scores
        .into_iter()
        .collect();
    assert_eq!(reference.find_matches(&query).get(&worker), Some(&10));
    assert_eq!(
        produced.get(&worker),
        Some(&10),
        "scored {produced:?}, expected 10"
    );
}

/// Equal worker counts at a jump landing do not mean the same workers: w1 holds the whole chain,
/// w2 only its first 6 blocks, and w3 everything but block 0. At the landing (position 8) the
/// matching set is {w1, w3}, the same size as the active set {w1, w2}; w2 must still be scored 6.
#[test]
fn count_equality_at_a_landing_is_not_set_equality() {
    let production = PositionalIndexer::new(8);
    let mut reference = ReferenceIndexer::new();
    let w1 = production.intern_worker("http://w1:8000").unwrap();
    let w2 = production.intern_worker("http://w2:8000").unwrap();
    let w3 = production.intern_worker("http://w3:8000").unwrap();
    let held: Vec<ContentHash> = (0..20).map(|p| content(5, p)).collect();
    let blocks = blocks_of(&held);
    let (mut m1, mut m2, mut m3) = (
        WorkerBlockMap::default(),
        WorkerBlockMap::default(),
        WorkerBlockMap::default(),
    );
    production.apply_stored(w1, &blocks, None, &mut m1).unwrap();
    reference.apply_stored(w1, &blocks, None).unwrap();
    production
        .apply_stored(w2, &blocks[..6], None, &mut m2)
        .unwrap();
    reference.apply_stored(w2, &blocks[..6], None).unwrap();
    production.apply_stored(w3, &blocks, None, &mut m3).unwrap();
    reference.apply_stored(w3, &blocks, None).unwrap();
    production.apply_removed(w3, &[blocks[0].seq_hash], &mut m3);
    reference.apply_removed(w3, &[blocks[0].seq_hash]);
    let expected = reference.find_matches(&held);
    assert_eq!(expected.get(&w1), Some(&20));
    assert_eq!(expected.get(&w2), Some(&6));
    assert_eq!(expected.get(&w3), None);
    let produced: BTreeMap<u32, u32> = production
        .find_matches(&held, false)
        .scores
        .into_iter()
        .collect();
    assert_eq!(produced, expected);
}
