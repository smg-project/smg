//! Exactness of the gateway's index backends (`crates/kv_index/docs/kv-router-leap.md`,
//! guardrails 1 to 3): the positional indexer and the chain index, fed through
//! the monitor's own apply path, must answer every lookup exactly as the
//! reference indexer does, after every batch, and hold exactly the same blocks.
//!
//! Two corpora. Engine streams generated in-process by the mock engine
//! (`mock_streams`: a vLLM-shaped run with a restart, an SGLang-shaped run,
//! two data-parallel ranks merged by receive order, and a run with a host
//! tier) leave the engine on its publisher wire and go through the relay's
//! decoder and normalizer, then through `KvEventMonitor::apply_event`, so
//! what reaches the index is what reaches it in production; at every
//! checkpoint the backends must also predict for every prompt the hit the
//! engine itself holds. A seeded synthetic corpus then does what an engine's
//! own stream does too little of: holes (evictions in the middle or at the
//! head of a chain the engine keeps using), heals (the missing block stored
//! again after its parent), divergent siblings sharing a prefix, duplicate
//! physical copies, host-tier copies, clears and worker removals, over
//! several workers at once.
//!
//! The request set replayed against the indexes is built from the chains the
//! events describe: every chain in full, its first half, a copy with the middle
//! block replaced, one with the last block replaced, one extended by a block
//! nobody stored, and a request nobody stored at all.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use kv_index::{
    compute_content_hash, compute_request_content_hashes, request_prefix_hashes, ContentHash,
    SequenceHash,
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheCleared, KvCacheEvent,
    KvCacheTier, KvEventBatch,
};

use super::{
    mock_streams::{self, Shape, Stream},
    KvIndex,
};
use crate::worker::kv_event_monitor::{KvEventMonitor, WorkerIndexState};

// ---------------------------------------------------------------------------
// The three backends side by side
// ---------------------------------------------------------------------------

struct Backend {
    index: KvIndex,
    workers: BTreeMap<String, (u32, WorkerIndexState)>,
}

impl Backend {
    fn new(index: KvIndex) -> Self {
        Self {
            index,
            workers: BTreeMap::new(),
        }
    }

    fn worker(&mut self, name: &str) -> &mut (u32, WorkerIndexState) {
        if !self.workers.contains_key(name) {
            let id = self.index.intern_worker(name).expect("worker id");
            self.workers
                .insert(name.to_string(), (id, WorkerIndexState::default()));
        }
        self.workers.get_mut(name).expect("just inserted")
    }

    fn names(&self) -> BTreeMap<u32, String> {
        self.workers
            .iter()
            .map(|(name, (id, _))| (*id, name.clone()))
            .collect()
    }

    /// Scores by worker name, so backends with different interned ids compare.
    fn scores(&self, query: &[ContentHash], early_exit: bool) -> BTreeMap<String, u32> {
        let names = self.names();
        self.index
            .find_matches(query, early_exit)
            .scores
            .into_iter()
            .map(|(id, score)| (names[&id].clone(), score))
            .collect()
    }

    fn blocks(&self) -> BTreeSet<(String, usize, ContentHash, SequenceHash)> {
        let names = self.names();
        self.index
            .debug_blocks()
            .into_iter()
            .map(|(id, position, content, prefix)| (names[&id].clone(), position, content, prefix))
            .collect()
    }

    fn counts(&self) -> BTreeMap<String, usize> {
        self.workers
            .iter()
            .map(|(name, (id, _))| (name.clone(), self.index.worker_block_count(*id)))
            .collect()
    }
}

/// The production backends next to the reference, fed identically.
struct Trio {
    backends: Vec<Backend>,
    lookups: usize,
}

impl Trio {
    fn new() -> Self {
        Self {
            backends: vec![
                Backend::new(KvIndex::reference()),
                Backend::new(KvIndex::positional(64)),
                Backend::new(KvIndex::chain()),
            ],
            lookups: 0,
        }
    }

    fn add_worker(&mut self, name: &str) {
        for backend in &mut self.backends {
            backend.worker(name);
        }
    }

    /// One batch through the monitor's apply path, for every backend.
    fn apply(&mut self, worker: &str, batch: &KvEventBatch) {
        for backend in &mut self.backends {
            let Backend { index, workers } = backend;
            if !workers.contains_key(worker) {
                let id = index.intern_worker(worker).expect("worker id");
                workers.insert(worker.to_string(), (id, WorkerIndexState::default()));
            }
            let (id, state) = workers.get_mut(worker).expect("present");
            for event in &batch.events {
                KvEventMonitor::apply_event(event, *id, index, state);
            }
        }
    }

    fn remove_worker(&mut self, worker: &str) {
        for backend in &mut self.backends {
            let Some((id, state)) = backend.workers.remove(worker) else {
                continue;
            };
            backend.index.remove_worker(id, state.blocks);
        }
    }

    /// Every backend answers every query as the reference does, holds the
    /// reference's blocks and counts them the same way.
    fn check(&mut self, queries: &[Vec<ContentHash>], label: &str) {
        let (reference, production) = self.backends.split_first().expect("three backends");
        for backend in production {
            let name = backend.index.name();
            for query in queries {
                for early_exit in [false, true] {
                    let want = reference.scores(query, early_exit);
                    let got = backend.scores(query, early_exit);
                    assert_eq!(
                        got, want,
                        "{label}: {name} scores (early_exit {early_exit}) for {query:?}"
                    );
                    self.lookups += 1;
                }
            }
            assert_eq!(
                backend.counts(),
                reference.counts(),
                "{label}: {name} block counts"
            );
            assert_eq!(
                backend.blocks(),
                reference.blocks(),
                "{label}: {name} index content"
            );
        }
    }

    /// Every backend predicts for every prompt the hit the engine holds: the
    /// prompt's leading blocks the worker has, by the engine's own count at
    /// the checkpoint. Returns the prompts checked.
    fn agree(
        &mut self,
        worker: &str,
        prompts: &BTreeSet<Vec<u32>>,
        held: &HashSet<u64>,
        label: &str,
    ) -> usize {
        for prompt in prompts {
            let want = Stream::prefix_match(held, prompt) as u32;
            let hashes = compute_request_content_hashes(prompt, mock_streams::BLOCK);
            for backend in &self.backends {
                let got = backend
                    .scores(&hashes, false)
                    .get(worker)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(
                    got,
                    want,
                    "{label}: {} predicts {got} cached blocks of a prompt the engine holds {want} of",
                    backend.index.name()
                );
            }
            self.lookups += self.backends.len();
        }
        prompts.len()
    }
}

// ---------------------------------------------------------------------------
// Queries from chains
// ---------------------------------------------------------------------------

/// A request nobody stored: distinct per call, never a real content hash.
fn novel(counter: &mut u64) -> ContentHash {
    *counter += 1;
    ContentHash(0xdead_beef_0000_0000 | *counter)
}

/// The request set for a chain: itself, its first half, the middle block
/// replaced, the last block replaced, extended by a block nobody stored.
fn variants(chain: &[ContentHash], counter: &mut u64) -> Vec<Vec<ContentHash>> {
    let mut out = vec![chain.to_vec()];
    if chain.len() > 1 {
        out.push(chain[..chain.len() / 2].to_vec());
        let mut middle = chain.to_vec();
        middle[chain.len() / 2] = novel(counter);
        out.push(middle);
        let mut suffix = chain.to_vec();
        *suffix.last_mut().expect("non-empty") = novel(counter);
        out.push(suffix);
    }
    let mut extended = chain.to_vec();
    extended.push(novel(counter));
    out.push(extended);
    out
}

/// The query set over every chain seen so far, plus one nobody stored.
fn query_set(chains: &BTreeSet<Vec<ContentHash>>) -> Vec<Vec<ContentHash>> {
    let mut counter = 0u64;
    let mut queries: Vec<Vec<ContentHash>> = chains
        .iter()
        .flat_map(|chain| variants(chain, &mut counter))
        .collect();
    queries.push(vec![novel(&mut counter), novel(&mut counter)]);
    queries
}

// ---------------------------------------------------------------------------
// Engine streams
// ---------------------------------------------------------------------------

/// Chains the stores of a stream describe, as a request would hash them:
/// each store's blocks appended to the chain of its parent block. Only plain
/// (unsalted, non-LoRA) stores, which is all the engine's stores carry.
#[derive(Default)]
struct Chains {
    /// Engine hash of a block -> the token chain from the root through it.
    tokens: BTreeMap<i64, Vec<u32>>,
    block_size: usize,
    seen: BTreeSet<Vec<ContentHash>>,
}

impl Chains {
    fn note(&mut self, batch: &KvEventBatch) {
        for event in &batch.events {
            let Some(kv_cache_event::Data::Stored(stored)) = &event.data else {
                continue;
            };
            assert!(
                stored.lora_name.is_none() && stored.cache_salt.is_none(),
                "the engine's stores carry no namespace"
            );
            let mut chain = stored
                .parent_block_hash
                .and_then(|parent| self.tokens.get(&parent).cloned())
                .unwrap_or_default();
            for block in &stored.blocks {
                if self.block_size == 0 {
                    self.block_size = block.block_size as usize;
                }
                chain.extend_from_slice(&block.token_ids);
                self.tokens.insert(block.block_hash, chain.clone());
            }
            if self.block_size > 0 {
                self.seen
                    .insert(compute_request_content_hashes(&chain, self.block_size));
            }
        }
    }
}

/// What a stream carried, by its normalized batches: each shape's claims are
/// checked against these counts.
#[derive(Debug, Default)]
struct Carried {
    stored: usize,
    removed: usize,
    cleared: usize,
    /// The ranks that stored blocks.
    ranks: BTreeSet<Option<i32>>,
    /// Stores of a block its rank already held: second physical copies.
    second_copies: usize,
    /// Device removals that left a copy on the same rank: the block stays.
    pinned_by_copy: usize,
    /// Stores of a block another rank held at the time.
    shared_across_ranks: usize,
    host_stored: usize,
    host_removed: usize,
    /// Device removals of a block the host still held: the block stays.
    pinned_by_host: usize,
    /// Host removals of a block no device copy was left of: the block goes.
    last_copy: usize,
}

/// The copies a stream has announced so far: per rank and block on the
/// device, and the blocks on the host.
#[derive(Default)]
struct Copies {
    device: HashMap<Option<i32>, HashMap<i64, u32>>,
    host: HashSet<i64>,
}

impl Copies {
    fn on_device(&self, hash: i64) -> bool {
        self.device.values().any(|held| held.contains_key(&hash))
    }
}

impl Carried {
    fn count(batches: &[KvEventBatch]) -> Self {
        let mut carried = Self::default();
        let mut copies = Copies::default();
        for batch in batches {
            for event in &batch.events {
                match &event.data {
                    Some(kv_cache_event::Data::Stored(stored)) => {
                        carried.note_stored(&mut copies, batch.dp_rank, stored);
                    }
                    Some(kv_cache_event::Data::Removed(removed)) => {
                        carried.note_removed(&mut copies, batch.dp_rank, removed);
                    }
                    Some(kv_cache_event::Data::Cleared(_)) => {
                        carried.cleared += 1;
                        copies = Copies::default();
                    }
                    None => {}
                }
            }
        }
        carried
    }

    fn note_stored(&mut self, copies: &mut Copies, rank: Option<i32>, stored: &KvBlocksStored) {
        let hashes = stored.blocks.iter().map(|block| block.block_hash);
        if is_host(stored.tier) {
            self.host_stored += 1;
            copies.host.extend(hashes);
            return;
        }
        self.stored += 1;
        self.ranks.insert(rank);
        for hash in hashes {
            if copies
                .device
                .iter()
                .any(|(other, held)| *other != rank && held.contains_key(&hash))
            {
                self.shared_across_ranks += 1;
            }
            let count = copies
                .device
                .entry(rank)
                .or_default()
                .entry(hash)
                .or_insert(0);
            if *count > 0 {
                self.second_copies += 1;
            }
            *count += 1;
        }
    }

    fn note_removed(&mut self, copies: &mut Copies, rank: Option<i32>, removed: &KvBlocksRemoved) {
        if is_host(removed.tier) {
            self.host_removed += 1;
            for hash in &removed.block_hashes {
                copies.host.remove(hash);
                if !copies.on_device(*hash) {
                    self.last_copy += 1;
                }
            }
            return;
        }
        self.removed += 1;
        let held = copies.device.entry(rank).or_default();
        for hash in &removed.block_hashes {
            if let Some(count) = held.get_mut(hash) {
                *count -= 1;
                if *count > 0 {
                    self.pinned_by_copy += 1;
                } else {
                    held.remove(hash);
                }
            }
            if copies.host.contains(hash) {
                self.pinned_by_host += 1;
            }
        }
    }
}

fn is_host(tier: Option<i32>) -> bool {
    tier == Some(KvCacheTier::Host as i32)
}

/// What a replayed stream established.
struct Replayed {
    carried: Carried,
    /// Ranks that published payloads.
    publishers: usize,
    checkpoints: usize,
    lookups: usize,
    agreements: usize,
}

impl Replayed {
    /// Enough of the stream was checked: the engine was caught up at enough
    /// checkpoints, and every one compared the whole query set and every
    /// prompt.
    fn assert_coverage(&self) {
        assert!(
            self.checkpoints >= 16,
            "{} checkpoints with the engine caught up",
            self.checkpoints
        );
        assert!(self.lookups > 100_000, "{} lookups compared", self.lookups);
        assert!(
            self.agreements > 5_000,
            "{} hits predicted",
            self.agreements
        );
    }
}

/// One engine stream through the monitor into every backend. At every
/// checkpoint the backends answer the query set as the reference does, hold
/// its blocks, and predict for every prompt the hit the engine holds.
fn replay_stream(shape: Shape, seed: u64, requests: usize) -> Replayed {
    let stream = Stream::generate(shape, seed, requests);
    let batches = stream.normalized();
    let label = shape.name();
    assert_eq!(
        batches.len(),
        stream.payloads.len(),
        "{label}: every payload decodes"
    );
    let worker = format!("grpc://{label}");
    let mut trio = Trio::new();
    trio.add_worker(&worker);
    let prompts: BTreeSet<Vec<u32>> = stream.prompts.iter().cloned().collect();
    let mut chains = Chains::default();
    let mut checkpoints = stream.checkpoints.iter().peekable();
    let mut agreements = 0;
    for (index, batch) in batches.iter().enumerate() {
        chains.note(batch);
        trio.apply(&worker, batch);
        while let Some(checkpoint) = checkpoints.next_if(|checkpoint| checkpoint.after == index + 1)
        {
            let at = format!("{label} batch {index}");
            trio.check(&query_set(&chains.seen), &at);
            agreements += trio.agree(&worker, &prompts, &checkpoint.held, &at);
        }
    }
    assert!(
        checkpoints.next().is_none(),
        "{label}: a checkpoint past the stream"
    );
    let publishers: BTreeSet<usize> = stream.payloads.iter().map(|payload| payload.rank).collect();
    Replayed {
        carried: Carried::count(&batches),
        publishers: publishers.len(),
        checkpoints: stream.checkpoints.len(),
        lookups: trio.lookups,
        agreements,
    }
}

#[test]
fn vllm_stream_scores_as_the_reference_and_predicts_the_engine_hit() {
    let replayed = replay_stream(Shape::Vllm, 1, 320);
    let carried = &replayed.carried;
    assert!(
        carried.stored >= 500 && carried.removed >= 300,
        "{carried:?}"
    );
    assert_eq!(carried.cleared, 1, "the restart's clear: {carried:?}");
    assert!(
        carried.second_copies >= 100 && carried.pinned_by_copy >= 30,
        "second physical copies and the removals they survived: {carried:?}"
    );
    assert_eq!(replayed.publishers, 1);
    replayed.assert_coverage();
}

#[test]
fn sglang_stream_scores_as_the_reference_and_predicts_the_engine_hit() {
    let replayed = replay_stream(Shape::Sglang, 2, 320);
    let carried = &replayed.carried;
    assert!(
        carried.stored >= 500 && carried.removed >= 300,
        "{carried:?}"
    );
    assert_eq!(carried.cleared, 0, "{carried:?}");
    assert_eq!(replayed.publishers, 1);
    replayed.assert_coverage();
}

#[test]
fn two_rank_stream_scores_as_the_reference_and_predicts_the_engine_hit() {
    let replayed = replay_stream(Shape::TwoRank, 3, 400);
    let carried = &replayed.carried;
    assert_eq!(
        carried.ranks,
        BTreeSet::from([Some(0), Some(1)]),
        "{carried:?}"
    );
    assert!(
        carried.shared_across_ranks >= 200,
        "blocks held by both ranks: {carried:?}"
    );
    assert!(
        carried.stored >= 500 && carried.removed >= 300,
        "{carried:?}"
    );
    assert_eq!(replayed.publishers, 2, "both ranks published");
    replayed.assert_coverage();
}

#[test]
fn host_tier_stream_scores_as_the_reference_and_predicts_the_engine_hit() {
    let replayed = replay_stream(Shape::HostTier, 4, 320);
    let carried = &replayed.carried;
    assert!(
        carried.host_stored >= 1000 && carried.host_removed >= 1000,
        "{carried:?}"
    );
    assert!(
        carried.pinned_by_host >= 1000,
        "device removals the host copy survived: {carried:?}"
    );
    assert!(
        carried.last_copy >= 1000,
        "host removals that took the last copy: {carried:?}"
    );
    assert_eq!(replayed.publishers, 1);
    replayed.assert_coverage();
}

// ---------------------------------------------------------------------------
// Synthetic corpus with holes
// ---------------------------------------------------------------------------

/// xorshift64*, as in `kv_index`'s exactness tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        lo + self.below(hi_inclusive - lo + 1)
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }
}

const BLOCK: usize = 4;

/// Token ids of block `position` of content stream `stream`: distinct per
/// (stream, position), so a chain's content hashes are distinct too.
fn tokens(stream: u64, position: usize) -> Vec<u32> {
    vec![
        (stream & 0xffff_ffff) as u32,
        (stream >> 32) as u32,
        position as u32,
        7,
    ]
}

/// One block of a chain as the engine names it: its tokens and its engine
/// hash, the chain hash of the contents so far (shared by every worker that
/// stores the same prefix, as the engines' hashes are).
#[derive(Clone)]
struct Block {
    tokens: Vec<u32>,
    hash: i64,
}

/// A chain a worker stored, with what the generator believes is still held.
struct Chain {
    blocks: Vec<Block>,
    alive: Vec<bool>,
}

impl Chain {
    fn contents(&self) -> Vec<ContentHash> {
        self.blocks
            .iter()
            .map(|block| compute_content_hash(&block.tokens))
            .collect()
    }
}

fn chain_of(token_blocks: &[Vec<u32>]) -> Vec<Block> {
    let contents: Vec<ContentHash> = token_blocks
        .iter()
        .map(|tokens| compute_content_hash(tokens))
        .collect();
    token_blocks
        .iter()
        .zip(request_prefix_hashes(&contents))
        .map(|(tokens, prefix)| Block {
            tokens: tokens.clone(),
            hash: prefix.0 as i64,
        })
        .collect()
}

fn kv_block(block: &Block) -> KvBlock {
    KvBlock {
        block_hash: block.hash,
        token_ids: block.tokens.clone(),
        block_size: BLOCK as i32,
        ..Default::default()
    }
}

fn stored(parent: Option<i64>, blocks: &[Block], tier: Option<KvCacheTier>) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
            blocks: blocks.iter().map(kv_block).collect(),
            parent_block_hash: parent,
            tier: tier.map(|tier| tier as i32),
            ..Default::default()
        })),
    }
}

fn removed(hashes: Vec<i64>, tier: Option<KvCacheTier>) -> KvCacheEvent {
    KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
            block_hashes: hashes,
            tier: tier.map(|tier| tier as i32),
            ..Default::default()
        })),
    }
}

fn cleared() -> KvCacheEvent {
    KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
    }
}

fn batch(events: Vec<KvCacheEvent>) -> KvEventBatch {
    KvEventBatch {
        events,
        ..Default::default()
    }
}

struct Corpus {
    rng: Rng,
    workers: Vec<String>,
    held: BTreeMap<String, Vec<Chain>>,
    prompts: Vec<Vec<Vec<u32>>>,
    chains: BTreeSet<Vec<ContentHash>>,
    next_stream: u64,
    holes: usize,
    heals: usize,
}

impl Corpus {
    fn new(seed: u64, workers: usize) -> Self {
        let mut rng = Rng::new(seed);
        let mut next_stream = 1u64;
        let prompts = (0..rng.range(3, 6))
            .map(|_| {
                next_stream += 1;
                (0..rng.range(4, 24))
                    .map(|position| tokens(next_stream, position))
                    .collect()
            })
            .collect();
        Self {
            rng,
            workers: (0..workers).map(|i| format!("grpc://w{i}:9000")).collect(),
            held: BTreeMap::new(),
            prompts,
            chains: BTreeSet::new(),
            next_stream,
            holes: 0,
            heals: 0,
        }
    }

    fn fresh_blocks(&mut self, count: usize) -> Vec<Vec<u32>> {
        self.next_stream += 1;
        let stream = self.next_stream;
        (0..count)
            .map(|position| tokens(stream, position))
            .collect()
    }

    fn worker(&mut self) -> String {
        self.workers[self.rng.below(self.workers.len())].clone()
    }

    fn remember(&mut self, chain: &Chain) {
        self.chains.insert(chain.contents());
    }

    /// Store a chain: a shared prompt prefix with a novel suffix (the common
    /// case) or something entirely new, in one event or split at a parent.
    fn store_new(&mut self, trio: &mut Trio) {
        let worker = self.worker();
        let mut token_blocks = if self.rng.chance(3, 4) {
            let prompt = &self.prompts[self.rng.below(self.prompts.len())];
            let keep = self.rng.range(1, prompt.len());
            prompt[..keep].to_vec()
        } else {
            Vec::new()
        };
        let suffix = self.rng.range(1, 12);
        token_blocks.extend(self.fresh_blocks(suffix));
        let blocks = chain_of(&token_blocks);
        let events = if blocks.len() > 2 && self.rng.chance(1, 2) {
            let cut = self.rng.range(1, blocks.len() - 1);
            vec![
                stored(None, &blocks[..cut], None),
                stored(Some(blocks[cut - 1].hash), &blocks[cut..], None),
            ]
        } else {
            vec![stored(None, &blocks, None)]
        };
        trio.apply(&worker, &batch(events));
        let chain = Chain {
            alive: vec![true; blocks.len()],
            blocks,
        };
        self.remember(&chain);
        self.held.entry(worker).or_default().push(chain);
    }

    fn pick_chain(&mut self) -> Option<(String, usize)> {
        let worker = self.worker();
        let count = self.held.get(&worker).map_or(0, Vec::len);
        (count > 0).then(|| (worker, self.rng.below(count)))
    }

    /// Decode extends the chain after its last block, while that block is
    /// held (an engine extends what it holds; a store naming an evicted parent
    /// is the gateway's fallback regime, which has its own test below).
    fn extend(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        if self.held[&worker][at].alive.last() != Some(&true) {
            return;
        }
        let more = self.rng.range(1, 3);
        let fresh = self.fresh_blocks(more);
        let chain = &self.held[&worker][at];
        let mut token_blocks: Vec<Vec<u32>> = chain
            .blocks
            .iter()
            .map(|block| block.tokens.clone())
            .collect();
        let parent = chain.blocks.last().map(|block| block.hash);
        token_blocks.extend(fresh);
        let blocks = chain_of(&token_blocks);
        let new = &blocks[blocks.len() - more..];
        trio.apply(&worker, &batch(vec![stored(parent, new, None)]));
        let held = &mut self.held.get_mut(&worker).expect("held")[at];
        held.blocks.extend_from_slice(new);
        held.alive.resize(held.blocks.len(), true);
        let contents = held.contents();
        self.chains.insert(contents);
    }

    /// A sibling diverges from a chain's prefix, possibly on another worker.
    fn sibling(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        let source = &self.held[&worker][at];
        if source.blocks.len() < 2 {
            return;
        }
        let keep = self.rng.range(1, source.blocks.len() - 1);
        let mut token_blocks: Vec<Vec<u32>> = source.blocks[..keep]
            .iter()
            .map(|block| block.tokens.clone())
            .collect();
        let more = self.rng.range(1, 6);
        token_blocks.extend(self.fresh_blocks(more));
        let blocks = chain_of(&token_blocks);
        let target = if self.rng.chance(1, 2) {
            worker.clone()
        } else {
            self.worker()
        };
        // The target stores the shared prefix first unless it holds it already
        // (the parent must be its own block), then the divergent tail.
        let holds_prefix = self.held.get(&target).is_some_and(|chains| {
            chains.iter().any(|chain| {
                chain.blocks.len() >= keep
                    && chain.blocks[..keep]
                        .iter()
                        .zip(&blocks[..keep])
                        .all(|(a, b)| a.hash == b.hash)
                    && chain.alive[..keep].iter().all(|&alive| alive)
            })
        });
        let mut events = Vec::new();
        if !holds_prefix {
            events.push(stored(None, &blocks[..keep], None));
        }
        events.push(stored(Some(blocks[keep - 1].hash), &blocks[keep..], None));
        trio.apply(&target, &batch(events));
        let chain = Chain {
            alive: vec![true; blocks.len()],
            blocks,
        };
        self.remember(&chain);
        self.held.entry(target).or_default().push(chain);
    }

    /// Evict from the tail, or punch a hole in the middle or at the head
    /// while the blocks after it stay (what block-level LRU eviction does).
    fn evict(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        let chain = &mut self.held.get_mut(&worker).expect("held")[at];
        let alive: Vec<usize> = (0..chain.blocks.len())
            .filter(|&i| chain.alive[i])
            .collect();
        if alive.is_empty() {
            return;
        }
        let kind = self.rng.below(4);
        let victims: Vec<usize> = match kind {
            0 | 1 => {
                let count = self.rng.range(1, alive.len());
                alive[alive.len() - count..].to_vec()
            }
            2 if alive.len() > 2 => vec![alive[self.rng.range(1, alive.len() - 2)]],
            _ => vec![alive[0]],
        };
        if kind >= 2 {
            self.holes += 1;
        }
        let hashes: Vec<i64> = victims.iter().map(|&i| chain.blocks[i].hash).collect();
        for &i in &victims {
            chain.alive[i] = false;
        }
        trio.apply(&worker, &batch(vec![removed(hashes, None)]));
    }

    /// Store a dead block again after its parent (or at the head), as an
    /// engine does when the request comes back.
    fn heal(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        let chain = &mut self.held.get_mut(&worker).expect("held")[at];
        let Some(dead) = (0..chain.blocks.len()).find(|&i| !chain.alive[i]) else {
            return;
        };
        if dead > 0 && !chain.alive[dead - 1] {
            // The parent is gone too: healing starts from the head.
            return;
        }
        let parent = (dead > 0).then(|| chain.blocks[dead - 1].hash);
        let block = chain.blocks[dead].clone();
        chain.alive[dead] = true;
        self.heals += 1;
        trio.apply(&worker, &batch(vec![stored(parent, &[block], None)]));
    }

    /// A second physical copy of a held prefix (vLLM re-prefills the last
    /// block of an exact resend) followed later by its own removal.
    fn duplicate(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        let chain = &self.held[&worker][at];
        let keep = chain.alive.iter().take_while(|&&alive| alive).count();
        if keep == 0 {
            return;
        }
        let blocks = chain.blocks[..keep].to_vec();
        let last = blocks[keep - 1].hash;
        trio.apply(&worker, &batch(vec![stored(None, &blocks, None)]));
        // One copy of the last block goes away: the block must stay indexed.
        trio.apply(&worker, &batch(vec![removed(vec![last], None)]));
    }

    /// HiCache write-through: a host copy of a held prefix, the device copy
    /// evicted (the block stays through the host copy), then the host copy.
    fn host_copy(&mut self, trio: &mut Trio) {
        let Some((worker, at)) = self.pick_chain() else {
            return;
        };
        let chain = &mut self.held.get_mut(&worker).expect("held")[at];
        let keep = chain.alive.iter().take_while(|&&alive| alive).count();
        if keep == 0 {
            return;
        }
        let blocks = chain.blocks[..keep].to_vec();
        let hashes: Vec<i64> = blocks.iter().map(|block| block.hash).collect();
        trio.apply(
            &worker,
            &batch(vec![stored(None, &blocks, Some(KvCacheTier::Host))]),
        );
        trio.apply(
            &worker,
            &batch(vec![removed(hashes.clone(), Some(KvCacheTier::Device))]),
        );
        if self.rng.chance(1, 2) {
            trio.apply(
                &worker,
                &batch(vec![removed(hashes, Some(KvCacheTier::Host))]),
            );
            for alive in chain.alive.iter_mut().take(keep) {
                *alive = false;
            }
        }
    }

    fn clear(&mut self, trio: &mut Trio) {
        let worker = self.worker();
        trio.apply(&worker, &batch(vec![cleared()]));
        self.held.remove(&worker);
    }

    fn replace_worker(&mut self, trio: &mut Trio) {
        let worker = self.worker();
        trio.remove_worker(&worker);
        self.held.remove(&worker);
        trio.add_worker(&worker);
    }

    fn step(&mut self, trio: &mut Trio) {
        match self.rng.below(100) {
            0..=24 => self.store_new(trio),
            25..=39 => self.extend(trio),
            40..=54 => self.sibling(trio),
            55..=74 => self.evict(trio),
            75..=84 => self.heal(trio),
            85..=89 => self.duplicate(trio),
            90..=94 => self.host_copy(trio),
            95..=97 => self.clear(trio),
            _ => self.replace_worker(trio),
        }
    }
}

fn run_corpus(seed: u64, workers: usize, steps: usize, checkpoint: usize) -> (Corpus, usize) {
    let mut trio = Trio::new();
    let mut corpus = Corpus::new(seed, workers);
    for worker in corpus.workers.clone() {
        trio.add_worker(&worker);
    }
    for step in 0..steps {
        corpus.step(&mut trio);
        if step % checkpoint == checkpoint - 1 {
            trio.check(
                &query_set(&corpus.chains),
                &format!("seed {seed} step {step}"),
            );
        }
    }
    trio.check(&query_set(&corpus.chains), &format!("seed {seed} end"));
    (corpus, trio.lookups)
}

#[test]
fn synthetic_corpus_with_holes_scores_identically_on_every_backend() {
    for seed in [1, 2, 3] {
        let (corpus, lookups) = run_corpus(seed, 5, 400, 10);
        assert!(corpus.holes > 20, "seed {seed}: {} holes", corpus.holes);
        assert!(corpus.heals > 5, "seed {seed}: {} heals", corpus.heals);
        assert!(lookups > 10_000, "seed {seed}: {lookups} lookups compared");
    }
}

/// The gateway's fallback regime: a store names a parent the index no longer
/// holds, so the monitor stores it again without a parent and the blocks land
/// at position 0 under engine hashes that belong further down the chain. When
/// the engine later stores the same hashes at their true positions, the
/// latest store wins in every index: a hash is held at one place per worker,
/// the mislaid pair scores nothing, and nothing is left for the next worker
/// interned into a freed id to inherit.
#[test]
fn a_hash_stored_again_at_its_true_position_after_a_fallback() {
    let mut trio = Trio::new();
    let worker = "grpc://w0:9000";
    trio.add_worker(worker);
    let token_blocks = Corpus::new(7, 1).fresh_blocks(6);
    let blocks = chain_of(&token_blocks);
    let contents: Vec<ContentHash> = token_blocks
        .iter()
        .map(|tokens| compute_content_hash(tokens))
        .collect();
    // b0..b3 held, b2 and b3 evicted, then the engine extends after b3: the
    // parent is unknown to the index, the fallback puts b4 and b5 at 0 and 1.
    trio.apply(worker, &batch(vec![stored(None, &blocks[..4], None)]));
    trio.apply(
        worker,
        &batch(vec![removed(vec![blocks[2].hash, blocks[3].hash], None)]),
    );
    trio.apply(
        worker,
        &batch(vec![stored(Some(blocks[3].hash), &blocks[4..], None)]),
    );
    let mislaid: Vec<ContentHash> = contents[4..].to_vec();
    trio.check(&[mislaid.clone(), contents.clone()], "after the fallback");
    assert_eq!(
        trio.backends[0].scores(&mislaid, false).get(worker),
        Some(&2),
        "the fallback placed the pair at the head"
    );
    // The engine recomputes the chain and announces it whole: b4 and b5 move
    // to positions 4 and 5.
    trio.apply(worker, &batch(vec![stored(None, &blocks, None)]));
    trio.check(
        &[mislaid.clone(), contents.clone()],
        "after the true-position store",
    );
    for backend in &trio.backends {
        let name = backend.index.name();
        assert_eq!(
            backend.scores(&contents, false).get(worker),
            Some(&6),
            "{name}: the whole chain is held"
        );
        assert!(
            backend.scores(&mislaid, false).is_empty(),
            "{name}: the mislaid pair scores nothing"
        );
        assert_eq!(
            backend.counts()[worker],
            6,
            "{name}: six blocks, one place each"
        );
    }
    // The worker leaves and another takes its id: nothing comes with it.
    trio.remove_worker(worker);
    trio.add_worker("grpc://w9:9000");
    for backend in &trio.backends {
        let name = backend.index.name();
        assert!(
            backend.scores(&contents, false).is_empty(),
            "{name}: a fresh worker holds nothing"
        );
        assert_eq!(backend.counts()["grpc://w9:9000"], 0, "{name}");
    }
}

/// A second physical copy of a block arrives inside a longer store under the
/// same parent (its first blocks second copies, the rest first copies), as
/// vLLM publishes one; a removal takes one copy. The block stays, and a
/// lookup through it still hits, until its last copy goes.
#[test]
fn a_second_copy_inside_a_longer_store_pins_the_block_until_its_last_removal() {
    let mut trio = Trio::new();
    let worker = "grpc://w0:9000";
    trio.add_worker(worker);
    let token_blocks = Corpus::new(9, 1).fresh_blocks(7);
    let blocks = chain_of(&token_blocks);
    let contents: Vec<ContentHash> = token_blocks
        .iter()
        .map(|tokens| compute_content_hash(tokens))
        .collect();
    // The preamble b0..b3; b4 b5 under b3; then b4 b5 b6 under b3: second
    // copies of b4 and b5 and the first copy of b6.
    trio.apply(worker, &batch(vec![stored(None, &blocks[..4], None)]));
    trio.apply(
        worker,
        &batch(vec![stored(Some(blocks[3].hash), &blocks[4..6], None)]),
    );
    trio.apply(
        worker,
        &batch(vec![stored(Some(blocks[3].hash), &blocks[4..7], None)]),
    );
    let queries = [contents[..6].to_vec(), contents.clone()];
    trio.check(&queries, "two copies");
    // One copy each of b5 and b4 goes, tail first: the chain still hits
    // through b6.
    trio.apply(
        worker,
        &batch(vec![removed(vec![blocks[5].hash, blocks[4].hash], None)]),
    );
    trio.check(&queries, "one copy left");
    for backend in &trio.backends {
        let name = backend.index.name();
        assert_eq!(
            backend.scores(&contents, false).get(worker),
            Some(&7),
            "{name}: the whole chain hits with one copy of b4 and b5 left"
        );
        assert_eq!(backend.counts()[worker], 7, "{name}");
    }
    // The last copies go: the chain is cut at b4; b6 stays, unreachable.
    trio.apply(
        worker,
        &batch(vec![removed(vec![blocks[5].hash, blocks[4].hash], None)]),
    );
    trio.check(&queries, "no copy left");
    for backend in &trio.backends {
        let name = backend.index.name();
        assert_eq!(
            backend.scores(&contents, false).get(worker),
            Some(&4),
            "{name}: the chain is cut at b4"
        );
        assert_eq!(backend.counts()[worker], 5, "{name}: the preamble and b6");
    }
}

// ---------------------------------------------------------------------------
// Guardrails 2 and 3 on the chain index
// ---------------------------------------------------------------------------

/// Lookups leave the chain index as they found it: the same runs, arena words
/// and blocks before and after a burst of queries (the store-free property of
/// the read path is established in `kv_index`; this checks the gateway's use
/// of it adds nothing).
#[test]
fn chain_index_lookups_change_nothing() {
    let mut trio = Trio::new();
    let mut corpus = Corpus::new(11, 4);
    for worker in corpus.workers.clone() {
        trio.add_worker(&worker);
    }
    for _ in 0..200 {
        corpus.step(&mut trio);
    }
    let chain = trio
        .backends
        .iter()
        .find(|backend| backend.index.name() == "chain")
        .expect("chain backend");
    let before = chain.index.chain_stats().expect("chain stats");
    let blocks = chain.blocks();
    let queries = query_set(&corpus.chains);
    for _ in 0..20 {
        for query in &queries {
            let _ = chain.index.find_matches(query, false);
            let _ = chain.index.find_matches(query, true);
        }
    }
    let after = chain.index.chain_stats().expect("chain stats");
    assert_eq!(
        format!("{after:?}"),
        format!("{before:?}"),
        "lookups changed the index's counters"
    );
    assert_eq!(
        chain.blocks(),
        blocks,
        "lookups changed the index's content"
    );
}

/// A clear and a worker removal give the chain index's memory back: nothing
/// stays live, every arena word is back in a free list, and the same corpus
/// stored again after the release is served from what was freed, so neither
/// the run slab nor the arena grows across fill-release cycles (guardrail 3:
/// bounded, recycled).
#[test]
fn chain_index_releases_state_on_clear_and_worker_removal() {
    let mut trio = Trio {
        backends: vec![Backend::new(KvIndex::chain())],
        lookups: 0,
    };
    let mut first_fill: Option<(usize, usize)> = None;
    for cycle in 0..3 {
        let mut corpus = Corpus::new(100, 4);
        for worker in corpus.workers.clone() {
            trio.add_worker(&worker);
        }
        for _ in 0..300 {
            corpus.step(&mut trio);
        }
        let chain = &mut trio.backends[0];
        assert!(
            chain.index.current_size() > 0,
            "cycle {cycle}: nothing indexed"
        );
        let filled = chain.index.chain_stats().expect("stats");
        match first_fill {
            None => first_fill = Some((filled.runs_allocated, filled.arena_bytes)),
            // The root's child table may be rebuilt once more; nothing else may grow.
            Some((runs, arena)) => assert!(
                filled.runs_allocated <= runs + 8 && filled.arena_bytes <= arena + 16 * 1024,
                "cycle {cycle}: the refill grew the index: {filled:?} after {runs} runs, {arena} B"
            ),
        }
        // Half the workers clear and leave, the rest just leave.
        let names: Vec<String> = chain.workers.keys().cloned().collect();
        for (n, name) in names.iter().enumerate() {
            let (id, mut state) = chain.workers.remove(name).expect("state");
            if n % 2 == 0 {
                chain.index.apply_cleared(id, &mut state.blocks);
                assert!(
                    state.blocks.is_empty(),
                    "cycle {cycle}: {name} cleared state"
                );
                assert_eq!(
                    chain.index.worker_block_count(id),
                    0,
                    "cycle {cycle}: {name}"
                );
            }
            chain.index.remove_worker(id, state.blocks);
            assert_eq!(
                chain.index.worker_block_count(id),
                0,
                "cycle {cycle}: {name}"
            );
        }
        let stats = chain.index.chain_stats().expect("stats");
        assert_eq!(stats.runs_live, 0, "cycle {cycle}: live runs after release");
        assert_eq!(
            stats.blocks_live, 0,
            "cycle {cycle}: live blocks after release"
        );
        assert_eq!(chain.index.current_size(), 0, "cycle {cycle}");
        assert_eq!(chain.index.entry_count(), 0, "cycle {cycle}");
        assert!(chain.index.is_empty(), "cycle {cycle}");
        assert!(chain.index.debug_blocks().is_empty(), "cycle {cycle}");
        // Everything but the root's own child table is back in a free list.
        assert!(
            stats.arena_free_bytes + 16 * 1024 >= stats.arena_bytes,
            "cycle {cycle}: arena words not back in free lists: {stats:?}"
        );
    }
}
