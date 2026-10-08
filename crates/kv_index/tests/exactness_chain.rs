//! Exactness harness: the run-compressed `ChainIndex` against the `ReferenceIndexer`.
//!
//! The same seeded corpus as `exactness_positional.rs` (stores, extensions, divergent siblings,
//! tail and whole-chain removals, clears, worker removals) plus evictions that leave holes: a
//! block in the middle of a chain, or its first block, goes away while the worker keeps the blocks
//! after it.
//! An engine with prefix caching produces exactly that when it evicts by block, and it keeps
//! reporting the later blocks as stored until it evicts them too. The reference says such a chain
//! matches up to the hole and no further; the chain index must say the same, and must heal when the
//! engine re-stores the missing block after its parent.
//!
//! After every round of events, lookups built from live chains and from mutated chains must score
//! identically in both indexers; at the end, the chain index must hold exactly the reference's
//! blocks. Scale with `KV_INDEX_EXACTNESS_EVENTS` (default 20000) and `KV_INDEX_EXACTNESS_SEED`.
#![expect(clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use kv_index::{
    ChainBlockMap, ChainIndex, ContentHash, ReferenceIndexer, SequenceHash, ShardedChainIndex,
    StoredBlock,
};
use rustc_hash::FxHashMap;

mod common;
use common::{blocks_of, content, Rng};

struct Held {
    worker: u32,
    contents: Vec<ContentHash>,
}

struct Harness {
    production: ShardedChainIndex,
    reference: ReferenceIndexer,
    maps: FxHashMap<u32, ChainBlockMap>,
    workers: Vec<u32>,
    held: Vec<Held>,
    prompts: Vec<Vec<ContentHash>>,
    next_stream: u64,
    next_worker: u32,
    rng: Rng,
    events: usize,
    stored_blocks: usize,
    holes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueryKind {
    Exact,
    Prefix,
    MiddleReplaced,
    SuffixReplaced,
    Extended,
    Unknown,
}

#[derive(Default)]
struct Mismatches {
    by_kind: BTreeMap<QueryKind, usize>,
    examples: Vec<String>,
    lookups: usize,
}

impl Harness {
    fn new(seed: u64, workers: usize) -> Self {
        let mut rng = Rng::new(seed);
        let mut harness = Self {
            // `KV_INDEX_EXACTNESS_SHARDS` shards (default one, the chain index itself); workers are
            // interned round robin, so two shards hold every other worker and share most content.
            production: ShardedChainIndex::new(
                env_or("KV_INDEX_EXACTNESS_SHARDS", 1) as usize,
                1024,
            ),
            reference: ReferenceIndexer::new(),
            maps: FxHashMap::default(),
            workers: Vec::new(),
            held: Vec::new(),
            prompts: Vec::new(),
            next_stream: 1,
            next_worker: 0,
            rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15),
            events: 0,
            stored_blocks: 0,
            holes: 0,
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
        self.maps.insert(id, ChainBlockMap::default());
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
    /// exercises the duplicate-store path. A chain with a hole is re-stored from the hole on, which
    /// is how an engine heals one.
    fn store(&mut self, worker: u32, contents: Vec<ContentHash>) {
        let blocks = blocks_of(&contents);
        let held = self.maps.get(&worker).expect("worker map");
        let mut known = 0;
        while known < blocks.len()
            && self
                .production
                .is_held(worker, held, blocks[known].seq_hash)
        {
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

    fn remove(&mut self, worker: u32, hashes: &[SequenceHash]) {
        let map = self.maps.get_mut(&worker).expect("worker map");
        self.production.apply_removed(worker, hashes, map);
        self.reference.apply_removed(worker, hashes);
        self.events += 1;
    }

    /// Evict a tail: the chain's blocks from a random position on, together with the same
    /// positions of every other held chain of that worker that runs through them.
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
        self.remove(worker, &hashes);
        self.held.retain(|h| !h.contents.is_empty());
    }

    /// Evict one to three blocks strictly inside a chain, or its first block, and keep the rest:
    /// a hole. The held chain is left as it is, so later lookups cross the hole and a later
    /// extension re-stores from the hole on.
    fn remove_middle(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        if contents.len() < 3 {
            return;
        }
        let (from, to) = if self.rng.chance(1, 8) {
            (0, 1)
        } else {
            let from = self.rng.range(1, contents.len() - 2);
            let to = (from + self.rng.range(1, 3)).min(contents.len() - 1);
            (from, to)
        };
        let blocks = blocks_of(&contents);
        let hashes: Vec<SequenceHash> = blocks[from..to].iter().map(|b| b.seq_hash).collect();
        self.remove(worker, &hashes);
        self.holes += 1;
    }

    /// Evict a whole conversation: the chain's blocks beyond the longest prefix it shares with
    /// another held chain of the same worker.
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
        self.remove(worker, &hashes);
        self.held.swap_remove(index);
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

    /// Store `blocks` after `parent` exactly as given, on both indexers; the outcomes must agree.
    /// Returns whether the store was accepted.
    fn store_blocks(
        &mut self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> bool {
        let map = self.maps.get_mut(&worker).expect("worker map");
        let produced = self.production.apply_stored(worker, blocks, parent, map);
        let referenced = self.reference.apply_stored(worker, blocks, parent);
        assert_eq!(
            produced.is_ok(),
            referenced.is_ok(),
            "store outcome differs after {} events: production {produced:?}, reference {referenced:?}",
            self.events
        );
        self.events += 1;
        if produced.is_ok() {
            self.stored_blocks += blocks.len();
        }
        produced.is_ok()
    }

    /// The engine stores a tail after a parent it names regardless of whether the index still
    /// holds it (removed, or never stored on this worker): both indexers reject it the same way,
    /// and the gateway's fallback then files the tail without a parent, at position 0 under the
    /// tail's own hashes.
    fn store_tail_under_named_parent(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = if self.rng.chance(1, 4) {
            self.random_worker()
        } else {
            self.held[index].worker
        };
        let contents = self.held[index].contents.clone();
        if contents.len() < 2 {
            return;
        }
        let blocks = blocks_of(&contents);
        let cut = self.rng.range(1, blocks.len() - 1);
        let parent = Some(blocks[cut - 1].seq_hash);
        if !self.store_blocks(worker, &blocks[cut..], parent) && self.rng.chance(3, 4) {
            self.store_blocks(worker, &blocks[cut..], None);
        }
    }

    /// The engine announces a held chain whole, from the root: blocks it filed elsewhere (a
    /// parentless tail) move back to their positions, blocks already in place stay.
    fn restore_whole_chain(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let blocks = blocks_of(&contents);
        self.store_blocks(worker, &blocks, None);
    }

    /// A second or third physical copy of a block the engine already reported: the same store
    /// again, one block or a run of blocks, under its real parent.
    fn store_again(&mut self) {
        let Some(index) = self.pick_held() else {
            return;
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let blocks = blocks_of(&contents);
        let from = self.rng.below(blocks.len());
        let to = (from + self.rng.range(1, 4)).min(blocks.len());
        let parent = (from > 0).then(|| blocks[from - 1].seq_hash);
        self.store_blocks(worker, &blocks[from..to], parent);
    }

    /// A twin: the same content under the same parent, named by another engine hash (an engine
    /// whose hash carries more than the chain, or a feed read against the wrong tokens). The
    /// index files it onto the position it holds and counts the conflict; the reference keeps
    /// one block per engine hash. `salt` keeps a twin's names distinct per round.
    fn store_twins(&mut self, salt: u64) -> Vec<SequenceHash> {
        let Some(index) = self.pick_held() else {
            return Vec::new();
        };
        let worker = self.held[index].worker;
        let contents = self.held[index].contents.clone();
        let blocks = blocks_of(&contents);
        let from = self.rng.below(blocks.len());
        let to = (from + self.rng.range(1, 6)).min(blocks.len());
        let twins: Vec<StoredBlock> = blocks[from..to]
            .iter()
            .map(|b| StoredBlock {
                seq_hash: SequenceHash(b.seq_hash.0 ^ salt.rotate_left(17) ^ 0x7777_0000_0000_0001),
                content_hash: b.content_hash,
            })
            .collect();
        let parent = (from > 0).then(|| blocks[from - 1].seq_hash);
        if self.store_blocks(worker, &twins, parent) {
            twins.iter().map(|b| b.seq_hash).collect()
        } else {
            Vec::new()
        }
    }

    /// One step of the wild corpus: the replayed corpus's events plus stores under parents the
    /// index may not hold, whole-chain re-announcements, repeated stores and, with `twins`,
    /// twin names; `twin_names` collects twin names for later removal by name.
    fn step_wild(&mut self, twins: bool, twin_names: &mut Vec<(u32, SequenceHash)>) {
        let roll = self.rng.below(1000);
        match roll {
            0..=599 => self.step(),
            600..=729 => self.store_tail_under_named_parent(),
            730..=809 => self.restore_whole_chain(),
            810..=889 => self.store_again(),
            890..=949 => {
                if twins {
                    let salt = self.rng.next();
                    let worker = self.held.last().map_or(0, |h| h.worker);
                    let names = self.store_twins(salt);
                    twin_names.extend(names.into_iter().map(|n| (worker, n)));
                } else {
                    self.remove_middle();
                }
            }
            950..=979 => {
                // Remove a twin by its own name, or the original name under a twin.
                if !twin_names.is_empty() && self.rng.chance(1, 2) {
                    let at = self.rng.below(twin_names.len());
                    let (worker, name) = twin_names.swap_remove(at);
                    if self.maps.contains_key(&worker) {
                        self.remove(worker, &[name]);
                    }
                } else {
                    self.remove_tail();
                }
            }
            _ => self.clear_worker(),
        }
    }

    fn step(&mut self) {
        let roll = self.rng.below(1000);
        match roll {
            0..=369 => {
                let chain = self.new_chain();
                let worker = self.random_worker();
                self.store(worker, chain);
            }
            370..=599 => {
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
            600..=729 => {
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
            730..=819 => self.remove_tail(),
            820..=899 => self.remove_middle(),
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
        let stream = self.fresh_stream();
        let len = self.rng.range(1, 32);
        let unknown: Vec<ContentHash> = (0..len).map(|p| content(stream, p)).collect();
        self.check_lookup(QueryKind::Unknown, unknown, out);
    }

    /// The per-worker counters must agree with the maps the lanes keep.
    fn check_counters(&self, label: &str) {
        for (&worker, map) in &self.maps {
            assert_eq!(
                self.production.worker_block_count(worker),
                map.len(),
                "{label}: block count of worker {worker} after {} events",
                self.events
            );
        }
        let total: usize = self.maps.values().map(|map| map.len()).sum();
        assert_eq!(
            self.production.current_size(),
            total,
            "{label}: total blocks"
        );
        // Distinct blocks are counted per shard: content held on two shards is stored twice.
        assert_eq!(
            self.production.entry_count(),
            self.production
                .debug_blocks()
                .iter()
                .map(|(worker, position, content, prefix)| {
                    (
                        ShardedChainIndex::shard_of(*worker),
                        *position,
                        *content,
                        *prefix,
                    )
                })
                .collect::<BTreeSet<_>>()
                .len(),
            "{label}: distinct blocks"
        );
    }
}

fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn run_corpus(seed: u64, workers: usize, events: usize) -> (Harness, Mismatches) {
    let mut harness = Harness::new(seed, workers);
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
    let produced = harness.production.debug_blocks();
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
        "{label}: {} of {} lookups scored differently from the reference, by query kind {:?}; examples:\n{}",
        mismatches.by_kind.values().sum::<usize>(),
        mismatches.lookups,
        mismatches.by_kind,
        mismatches.examples.join("\n")
    );
    harness.check_counters(label);
}

#[test]
fn replayed_corpus_with_holes_matches_the_reference() {
    let events = env_or("KV_INDEX_EXACTNESS_EVENTS", 20_000) as usize;
    let seed = env_or("KV_INDEX_EXACTNESS_SEED", 20261005);
    for (salt, workers) in [(8u64, 16usize), (64, 2), (3, 64)] {
        let (harness, mismatches) = run_corpus(seed ^ salt, workers, events);
        assert_exact(
            &harness,
            &mismatches,
            &format!("{workers} workers, seed {seed}"),
        );
        assert!(
            harness.stored_blocks > events,
            "corpus too small to mean anything: {} blocks for {} events",
            harness.stored_blocks,
            events
        );
        assert!(
            harness.holes * 20 > events,
            "corpus made too few holes: {} for {} events",
            harness.holes,
            events
        );
    }
}

/// A worker holds [A, B, C]; a request [A, X, C] shares only A with it.
#[test]
fn a_divergence_at_the_tail_is_not_hidden() {
    let production = ChainIndex::new();
    let mut reference = ReferenceIndexer::new();
    let worker = production.intern_worker("http://w:8000").unwrap();
    let mut map = ChainBlockMap::default();
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
    assert_eq!(produced.get(&worker), Some(&1));
}

/// w1 holds the whole chain, w2 only its first 6 blocks, and w3 everything but block 0.
#[test]
fn partial_holders_score_their_own_prefix() {
    let production = ChainIndex::new();
    let mut reference = ReferenceIndexer::new();
    let w1 = production.intern_worker("http://w1:8000").unwrap();
    let w2 = production.intern_worker("http://w2:8000").unwrap();
    let w3 = production.intern_worker("http://w3:8000").unwrap();
    let held: Vec<ContentHash> = (0..20).map(|p| content(5, p)).collect();
    let blocks = blocks_of(&held);
    let (mut m1, mut m2, mut m3) = (
        ChainBlockMap::default(),
        ChainBlockMap::default(),
        ChainBlockMap::default(),
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
    assert_eq!(production.debug_blocks(), reference.blocks());
}

/// Two holes in one chain, healed one at a time, with a sibling branching off between them: the
/// score follows the first remaining hole until both are healed.
#[test]
fn holes_heal_independently() {
    let production = ChainIndex::new();
    let mut reference = ReferenceIndexer::new();
    let worker = production.intern_worker("http://w:8000").unwrap();
    let mut map = ChainBlockMap::default();
    let held: Vec<ContentHash> = (0..24).map(|p| content(7, p)).collect();
    let blocks = blocks_of(&held);
    production
        .apply_stored(worker, &blocks, None, &mut map)
        .unwrap();
    reference.apply_stored(worker, &blocks, None).unwrap();
    let holes = [blocks[5].seq_hash, blocks[6].seq_hash, blocks[17].seq_hash];
    production.apply_removed(worker, &holes, &mut map);
    reference.apply_removed(worker, &holes);
    let mut sibling: Vec<ContentHash> = held[..10].to_vec();
    sibling.extend((0..4).map(|p| content(8, p)));
    let sibling_blocks = blocks_of(&sibling);
    production
        .apply_stored(
            worker,
            &sibling_blocks[10..],
            Some(blocks[9].seq_hash),
            &mut map,
        )
        .unwrap();
    reference
        .apply_stored(worker, &sibling_blocks[10..], Some(blocks[9].seq_hash))
        .unwrap();
    let agree = |production: &ChainIndex, reference: &ReferenceIndexer, query: &[ContentHash]| {
        let produced: BTreeMap<u32, u32> = production
            .find_matches(query, false)
            .scores
            .into_iter()
            .collect();
        assert_eq!(produced, reference.find_matches(query), "query {query:?}");
    };
    agree(&production, &reference, &held);
    agree(&production, &reference, &sibling);
    assert_eq!(reference.find_matches(&held).get(&worker), Some(&5));
    production
        .apply_stored(worker, &blocks[5..7], Some(blocks[4].seq_hash), &mut map)
        .unwrap();
    reference
        .apply_stored(worker, &blocks[5..7], Some(blocks[4].seq_hash))
        .unwrap();
    agree(&production, &reference, &held);
    agree(&production, &reference, &sibling);
    assert_eq!(reference.find_matches(&held).get(&worker), Some(&17));
    assert_eq!(reference.find_matches(&sibling).get(&worker), Some(&14));
    production
        .apply_stored(worker, &blocks[17..18], Some(blocks[16].seq_hash), &mut map)
        .unwrap();
    reference
        .apply_stored(worker, &blocks[17..18], Some(blocks[16].seq_hash))
        .unwrap();
    agree(&production, &reference, &held);
    assert_eq!(reference.find_matches(&held).get(&worker), Some(&24));
    assert_eq!(production.debug_blocks(), reference.blocks());
}

/// The production index against the reference after every few events of a wild corpus: stores
/// under parents the worker no longer holds (and the gateway's parentless fallback after the
/// rejection), whole chains announced again from the root, repeated stores of held blocks,
/// tails, holes, clears and worker churn, from several seeds. The full state is compared every
/// 16 events, the lookups every 64, the counters at the end. Scale with
/// `KV_INDEX_WILD_EVENTS` (default 3000 per seed) and `KV_INDEX_WILD_SEEDS` (default 6).
#[test]
fn wild_event_sequences_match_the_reference_at_every_step() {
    let events = env_or("KV_INDEX_WILD_EVENTS", 3000) as usize;
    let seeds = env_or("KV_INDEX_WILD_SEEDS", 6);
    let base = env_or("KV_INDEX_EXACTNESS_SEED", 20261006);
    for seed in 0..seeds {
        let mut harness = Harness::new(base.wrapping_add(seed.wrapping_mul(0x9e37)), 6);
        let mut mismatches = Mismatches::default();
        let mut twin_names = Vec::new();
        let label = format!("wild seed {seed}");
        while harness.events < events {
            let before = harness.events;
            harness.step_wild(false, &mut twin_names);
            if harness.events == before {
                continue;
            }
            if harness.events.is_multiple_of(16) {
                let produced = harness.production.debug_blocks();
                let expected = harness.reference.blocks();
                assert!(
                    produced == expected,
                    "{label}: state differs after {} events: {} reference blocks, {} production blocks; missing e.g. {:?}; phantom e.g. {:?}",
                    harness.events,
                    expected.len(),
                    produced.len(),
                    expected.difference(&produced).take(4).collect::<Vec<_>>(),
                    produced.difference(&expected).take(4).collect::<Vec<_>>()
                );
            }
            if harness.events.is_multiple_of(64) {
                harness.lookups(&mut mismatches);
            }
        }
        harness.lookups(&mut mismatches);
        assert_exact(&harness, &mismatches, &label);
        assert_eq!(
            harness.production.stats().engine_conflicts,
            0,
            "{label}: no twins in this corpus"
        );
    }
}

/// The same corpus with twins (one position named by two engine hashes): the index files a
/// twin onto the position it holds and counts the conflict, so it holds at most what the
/// reference holds and scores no worker higher than the reference does; removals by either
/// name, stores under either name and the cut-and-join they lead to must never panic.
#[test]
fn twins_never_panic_and_never_overstate_the_reference() {
    let events = env_or("KV_INDEX_WILD_EVENTS", 3000) as usize;
    let seeds = env_or("KV_INDEX_WILD_SEEDS", 6);
    let base = env_or("KV_INDEX_EXACTNESS_SEED", 20261006);
    for seed in 0..seeds {
        let mut harness = Harness::new(base.wrapping_add(seed.wrapping_mul(0x51f1)), 6);
        let mut twin_names = Vec::new();
        let label = format!("twins seed {seed}");
        while harness.events < events {
            let before = harness.events;
            harness.step_wild(true, &mut twin_names);
            if harness.events == before {
                continue;
            }
            if harness.events.is_multiple_of(16) {
                let produced = harness.production.debug_blocks();
                let expected = harness.reference.blocks();
                let phantom: Vec<_> = produced.difference(&expected).take(4).collect();
                assert!(
                    phantom.is_empty(),
                    "{label}: production holds blocks the reference does not after {} events: {phantom:?}",
                    harness.events
                );
                for (&worker, map) in &harness.maps {
                    assert!(
                        harness.production.worker_block_count(worker) <= map.len(),
                        "{label}: worker {worker} credited beyond its lane map after {} events",
                        harness.events
                    );
                }
            }
            if harness.events.is_multiple_of(64) {
                for _ in 0..4 {
                    let Some(index) = harness.pick_held() else {
                        break;
                    };
                    let query = harness.held[index].contents.clone();
                    let produced = harness.production_scores(&query);
                    let expected = harness.reference.find_matches(&query);
                    for (worker, score) in &produced {
                        assert!(
                            expected.get(worker).is_some_and(|e| e >= score),
                            "{label}: worker {worker} scores {score} in production against {:?} in the reference after {} events",
                            expected.get(worker),
                            harness.events
                        );
                    }
                }
            }
        }
        assert!(
            harness.production.stats().engine_conflicts > 0,
            "{label}: the corpus produced no twin"
        );
    }
}
