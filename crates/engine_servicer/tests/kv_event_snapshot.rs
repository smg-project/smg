//! The relay's state snapshot against generated publisher streams in the
//! engines' own hashes: the live set the snapshot emits equals a reference
//! replay of the normalized stream, copies and tiers included; a live parent
//! always precedes its children; the emitted chains re-verify under the
//! engine's own hash, which needs every parent's digest before its child and
//! so fails on any ordering slip; and the chunks are stamped the way the
//! gateway's cursor needs them.
//!
//! The streams come from a seeded generator, not from recordings: two DP
//! ranks storing chains block by block after live parents, second physical
//! copies, removals of live blocks, host-tier copies and their evictions,
//! and one clear of a rank midway, every hash computed the way the engine
//! computes it so the relay's hash check verifies the whole stream.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr
)]

use std::collections::{BTreeMap, HashSet};

use engine_servicer::{
    engine_hash::{self, Digest32, EngineHash},
    kv_state::{LiveState, SnapshotChunks, CHUNK_BLOCKS},
    kv_wire::{
        BlockHash, Counts, EventTail, ExtraKey, Normalizer, WireBatch, WireEvent, WireRemoved,
        WireStored, WireTokens,
    },
};
use smg_grpc_client::common_proto::{
    kv_block_extra_key, kv_cache_event, KvBlocksStored, KvEventBatch,
};

/// `(rank, tier, hash)` with its physical copies.
type LiveSet = BTreeMap<(Option<i32>, i32, i64), u32>;

const BLOCK_SIZE: usize = 16;
const RANKS: usize = 2;

// ---------------------------------------------------------------------------
// The generated publisher
// ---------------------------------------------------------------------------

/// xorshift64*: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
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

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// A block the publisher has stored: what a child chains on, what a second
/// copy or a host backup repeats, and how many physical copies it has.
#[derive(Clone)]
struct Published {
    hash: i64,
    digest: Digest32,
    tokens: Vec<u32>,
    parent: Option<i64>,
    copies: u32,
}

#[derive(Default)]
struct RankModel {
    /// Device blocks with their record intact at the relay (stored, never
    /// removed since): parents, repeats and host backups come from here.
    live: Vec<Published>,
    /// Blocks a copy of which was removed: no longer parents at the relay,
    /// still holding copies the engine frees one at a time.
    fading: Vec<Published>,
    /// Blocks backed up to the host tier and not yet evicted there.
    host: Vec<Published>,
    /// Children holding copies, per block: only a block without any is
    /// freed, as the engines free the tail of a chain first, so a live child
    /// always has its parent in the state.
    children: BTreeMap<i64, u32>,
}

impl RankModel {
    fn held(&mut self, hash: i64) {
        *self.children.entry(hash).or_default() += 1;
    }

    /// The last copy of a block with `parent` went.
    fn released(&mut self, parent: Option<i64>) {
        if let Some(parent) = parent {
            if let Some(count) = self.children.get_mut(&parent) {
                *count -= 1;
                if *count == 0 {
                    self.children.remove(&parent);
                }
            }
        }
    }

    fn is_leaf(&self, hash: i64) -> bool {
        !self.children.contains_key(&hash)
    }
}

struct Generator {
    rng: Rng,
    engine: EngineHash,
    next_token: u32,
    ranks: Vec<RankModel>,
    /// The batch that clears rank 0.
    clear_at: usize,
}

/// One block's digest and published integer in `engine`'s algorithm.
fn block(engine: EngineHash, prior: Option<&Digest32>, tokens: &[u32]) -> (Digest32, i64) {
    match engine {
        EngineHash::Sglang => {
            let digest = engine_hash::sglang_page(prior, tokens);
            (digest, engine_hash::sglang_event_int(&digest))
        }
        EngineHash::VllmSha256Cbor => {
            let digest = engine_hash::vllm_block(prior, tokens, None);
            (digest, engine_hash::vllm_event_int(&digest))
        }
    }
}

impl Generator {
    fn new(engine: EngineHash, seed: u64, batches: usize) -> Self {
        Self {
            rng: Rng(seed.max(1)),
            engine,
            next_token: 0,
            ranks: (0..RANKS).map(|_| RankModel::default()).collect(),
            clear_at: batches / 3,
        }
    }

    /// Tokens no earlier block had, so every new block is a new hash and the
    /// only repeated hashes are the deliberate second copies.
    fn fresh_tokens(&mut self) -> Vec<u32> {
        (0..BLOCK_SIZE)
            .map(|_| {
                self.next_token += 1;
                self.next_token
            })
            .collect()
    }

    fn medium(&self, host: bool) -> &'static str {
        match (self.engine, host) {
            (EngineHash::Sglang, true) => "CPU_PINNED",
            (EngineHash::VllmSha256Cbor, true) => "CPU",
            (_, false) => "GPU",
        }
    }

    fn tail(&self, host: bool) -> EventTail {
        let vllm = self.engine == EngineHash::VllmSha256Cbor;
        EventTail {
            medium: Some(self.medium(host).to_string()),
            group_idx: vllm.then_some(0),
            kv_cache_spec_kind: vllm.then(|| "full_attention".to_string()),
            ..EventTail::default()
        }
    }

    fn stored(
        &self,
        hashes: Vec<i64>,
        parent: Option<i64>,
        tokens: Vec<u32>,
        host: bool,
    ) -> WireEvent {
        WireEvent::BlockStored(WireStored {
            block_hashes: hashes.into_iter().map(BlockHash).collect(),
            parent_block_hash: parent.map(BlockHash),
            token_ids: WireTokens::Ids(tokens),
            block_size: BLOCK_SIZE as i64,
            lora_id: None,
            lora_name: None,
            cache_salt: None,
            extra_keys: None,
            tail: self.tail(host),
        })
    }

    fn removed(&self, hashes: Vec<i64>, host: bool) -> WireEvent {
        WireEvent::BlockRemoved(WireRemoved {
            block_hashes: hashes.into_iter().map(BlockHash).collect(),
            tail: self.tail(host),
        })
    }

    /// A chain of one to three new blocks after a live parent, or a new root.
    fn chain(&mut self, rank: usize) -> WireEvent {
        let parent = {
            let live = &self.ranks[rank].live;
            (!live.is_empty() && self.rng.chance(75))
                .then(|| live[self.rng.below(live.len())].clone())
        };
        let mut prior = parent.as_ref().map(|parent| parent.digest);
        let mut prior_hash = parent.as_ref().map(|parent| parent.hash);
        let mut hashes = Vec::new();
        let mut tokens = Vec::new();
        for _ in 0..=self.rng.below(3) {
            let block_tokens = self.fresh_tokens();
            let (digest, hash) = block(self.engine, prior.as_ref(), &block_tokens);
            let model = &mut self.ranks[rank];
            model.live.push(Published {
                hash,
                digest,
                tokens: block_tokens.clone(),
                parent: prior_hash,
                copies: 1,
            });
            if let Some(prior_hash) = prior_hash {
                model.held(prior_hash);
            }
            hashes.push(hash);
            tokens.extend(block_tokens);
            prior = Some(digest);
            prior_hash = Some(hash);
        }
        self.stored(hashes, parent.map(|parent| parent.hash), tokens, false)
    }

    /// A second physical copy of a live block whose parent is still live
    /// (so the repeat verifies), capped like the gateway counts copies.
    fn second_copy(&mut self, rank: usize) -> Option<WireEvent> {
        let model = &self.ranks[rank];
        let live_hashes: HashSet<i64> = model.live.iter().map(|block| block.hash).collect();
        let candidates: Vec<usize> = (0..model.live.len())
            .filter(|&index| {
                let block = &model.live[index];
                block.copies < 8
                    && block
                        .parent
                        .is_none_or(|parent| live_hashes.contains(&parent))
            })
            .collect();
        let index = *candidates.get(self.rng.below(candidates.len().max(1)))?;
        let block = &mut self.ranks[rank].live[index];
        block.copies += 1;
        let (hash, parent, tokens) = (block.hash, block.parent, block.tokens.clone());
        Some(self.stored(vec![hash], parent, tokens, false))
    }

    /// One copy of a leaf freed: a never-removed block moves to `fading`, a
    /// fading block loses one more copy, the last copy releases its parent.
    fn free_leaf(&mut self, rank: usize) -> Option<i64> {
        let model = &self.ranks[rank];
        let mut candidates: Vec<(bool, usize)> = (0..model.live.len())
            .filter(|&index| model.is_leaf(model.live[index].hash))
            .map(|index| (true, index))
            .collect();
        candidates.extend(
            (0..model.fading.len())
                .filter(|&index| model.is_leaf(model.fading[index].hash))
                .map(|index| (false, index)),
        );
        let (from_live, index) = *candidates.get(self.rng.below(candidates.len().max(1)))?;
        let model = &mut self.ranks[rank];
        if from_live {
            let mut block = model.live.swap_remove(index);
            block.copies -= 1;
            let (hash, parent) = (block.hash, block.parent);
            if block.copies == 0 {
                model.released(parent);
            } else {
                model.fading.push(block);
            }
            return Some(hash);
        }
        model.fading[index].copies -= 1;
        let hash = model.fading[index].hash;
        if model.fading[index].copies == 0 {
            let block = model.fading.swap_remove(index);
            model.released(block.parent);
        }
        Some(hash)
    }

    /// One event of `rank`: a chain, a second copy, a removal of one or two
    /// leaf copies, a host backup of a live root, or a host eviction.
    fn event(&mut self, rank: usize) -> WireEvent {
        let roll = self.rng.below(100);
        if roll < 55 || self.ranks[rank].live.is_empty() {
            return self.chain(rank);
        }
        if roll < 70 {
            if let Some(event) = self.second_copy(rank) {
                return event;
            }
            return self.chain(rank);
        }
        if roll < 85 {
            let mut hashes = Vec::new();
            for _ in 0..=self.rng.below(2) {
                if let Some(hash) = self.free_leaf(rank) {
                    if !hashes.contains(&hash) {
                        hashes.push(hash);
                    }
                }
            }
            if !hashes.is_empty() {
                return self.removed(hashes, false);
            }
            return self.chain(rank);
        }
        if roll < 93 {
            let roots: Vec<Published> = self.ranks[rank]
                .live
                .iter()
                .filter(|block| block.parent.is_none())
                .cloned()
                .collect();
            if !roots.is_empty() {
                let root = roots[self.rng.below(roots.len())].clone();
                let event = self.stored(vec![root.hash], None, root.tokens.clone(), true);
                self.ranks[rank].host.push(root);
                return event;
            }
        }
        let host = &mut self.ranks[rank].host;
        if host.is_empty() {
            return self.chain(rank);
        }
        let index = self.rng.below(host.len());
        let evicted = host.swap_remove(index).hash;
        self.removed(vec![evicted], true)
    }

    fn batch(&mut self, index: usize) -> WireBatch {
        let rank = self.rng.below(RANKS);
        let mut events = Vec::new();
        if index == self.clear_at {
            events.push(WireEvent::AllBlocksCleared { ownership: None });
            self.ranks[0] = RankModel::default();
        }
        let rank = if index == self.clear_at { 0 } else { rank };
        for _ in 0..=self.rng.below(4) {
            events.push(self.event(rank));
        }
        WireBatch {
            ts: 1_700_000_000.0 + index as f64,
            events,
            dp_rank: Some(rank as i32),
        }
    }
}

/// `batches` publisher batches of `engine`'s stream, normalized with the
/// engine-hash check on, as the relay forwards them.
fn generated(engine: EngineHash, seed: u64, batches: usize) -> (Vec<KvEventBatch>, Counts) {
    let mut generator = Generator::new(engine, seed, batches);
    let mut normalizer = Normalizer::with_hash_check(engine);
    let mut event_id = 0;
    let stream = (0..batches)
        .map(|index| {
            let batch = generator.batch(index);
            normalizer.normalize_batch(batch, index as u64, &mut event_id)
        })
        .collect();
    (stream, normalizer.counts().clone())
}

// ---------------------------------------------------------------------------
// The reference and the snapshot's reading
// ---------------------------------------------------------------------------

/// One batch into the live set a naive replay of the normalized stream
/// leaves: stores add a copy (capped like the gateway counts them), removals
/// take one, a clear empties the rank.
fn replay(live: &mut LiveSet, batch: &KvEventBatch) {
    for event in &batch.events {
        match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => {
                let tier = stored.tier.expect("the relay sets the tier");
                for block in &stored.blocks {
                    let copies = live
                        .entry((batch.dp_rank, tier, block.block_hash))
                        .or_insert(0);
                    if *copies < 8 {
                        *copies += 1;
                    }
                }
            }
            Some(kv_cache_event::Data::Removed(removed)) => {
                let tier = removed.tier.expect("the relay sets the tier");
                for &hash in &removed.block_hashes {
                    let key = (batch.dp_rank, tier, hash);
                    if let Some(copies) = live.get_mut(&key) {
                        *copies -= 1;
                        if *copies == 0 {
                            live.remove(&key);
                        }
                    }
                }
            }
            Some(kv_cache_event::Data::Cleared(_)) => {
                live.retain(|(rank, _, _), _| *rank != batch.dp_rank);
            }
            None => {}
        }
    }
}

fn reference(batches: &[KvEventBatch]) -> LiveSet {
    let mut live = LiveSet::new();
    for batch in batches {
        replay(&mut live, batch);
    }
    live
}

/// Every stored event of the chunks with the rank it came under.
fn stores(chunks: &[KvEventBatch]) -> Vec<(Option<i32>, &KvBlocksStored)> {
    chunks
        .iter()
        .flat_map(|chunk| {
            chunk
                .events
                .iter()
                .filter_map(move |event| match &event.data {
                    Some(kv_cache_event::Data::Stored(stored)) => Some((chunk.dp_rank, stored)),
                    _ => None,
                })
        })
        .collect()
}

fn emitted(chunks: &[KvEventBatch]) -> LiveSet {
    let mut live = LiveSet::new();
    for (rank, stored) in stores(chunks) {
        for block in &stored.blocks {
            *live
                .entry((rank, stored.tier.unwrap(), block.block_hash))
                .or_insert(0) += 1;
        }
    }
    live
}

/// A snapshot store as the publisher would have sent it, so the relay's
/// hash check can rehash the emitted chain.
fn wire_stored(stored: &KvBlocksStored) -> WireEvent {
    let first = &stored.blocks[0];
    WireEvent::BlockStored(WireStored {
        block_hashes: stored
            .blocks
            .iter()
            .map(|block| BlockHash(block.block_hash))
            .collect(),
        parent_block_hash: stored.parent_block_hash.map(BlockHash),
        token_ids: WireTokens::Ids(
            stored
                .blocks
                .iter()
                .flat_map(|block| block.token_ids.iter().copied())
                .collect(),
        ),
        block_size: i64::from(first.block_size),
        lora_id: first.lora_id,
        lora_name: stored.lora_name.clone(),
        cache_salt: stored.cache_salt.clone(),
        extra_keys: Some(
            stored
                .blocks
                .iter()
                .map(|block| {
                    (!block.extra_keys.is_empty()).then(|| {
                        block
                            .extra_keys
                            .iter()
                            .map(|key| match key.key.clone().expect("a key") {
                                kv_block_extra_key::Key::Text(text) => ExtraKey::Text(text),
                                kv_block_extra_key::Key::Number(number) => ExtraKey::Number(number),
                                kv_block_extra_key::Key::Blob(blob) => ExtraKey::Blob(blob),
                                kv_block_extra_key::Key::Multimodal(mm) => ExtraKey::Multimodal {
                                    identifier: mm.identifier,
                                    offset: mm.offset,
                                },
                            })
                            .collect()
                    })
                })
                .collect(),
        ),
        tail: EventTail {
            medium: stored.medium.clone(),
            group_idx: stored.group_idx,
            kv_cache_spec_kind: stored.kv_cache_spec_kind.clone(),
            kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
            locality: None,
            ownership: stored.ownership.clone(),
            session_id: stored.session_id.clone(),
        },
    })
}

/// Rehash the emitted stores in order with `engine`'s algorithm.
fn rehash(chunks: &[KvEventBatch], engine: EngineHash) -> Counts {
    let mut checker = Normalizer::with_hash_check(engine);
    let mut event_id = 0;
    for (rank, stored) in stores(chunks) {
        event_id += 1;
        checker
            .normalize(wire_stored(stored), rank, event_id)
            .expect("a snapshot store is forwardable");
    }
    checker.counts().clone()
}

struct Case {
    name: &'static str,
    engine: EngineHash,
    seed: u64,
}

const CASES: &[Case] = &[
    Case {
        name: "vllm",
        engine: EngineHash::VllmSha256Cbor,
        seed: 0x5eed_0001,
    },
    Case {
        name: "sglang",
        engine: EngineHash::Sglang,
        seed: 0x5eed_0002,
    },
];

/// Enough batches for the live set to span several chunks.
const BATCHES: usize = 4_000;

#[test]
fn snapshots_of_the_generated_streams_equal_their_live_sets() {
    for case in CASES {
        let name = case.name;
        let (batches, stream_counts) = generated(case.engine, case.seed, BATCHES);
        // The generator speaks the engine's hash: everything verified.
        assert!(stream_counts.hash_checked > 0, "{name}: blocks checked");
        assert_eq!(
            (stream_counts.hash_mismatch, stream_counts.hash_unverifiable),
            (0, 0),
            "{name}: the generated stream verifies"
        );
        assert!(stream_counts.duplicate_stores > 0, "{name}: second copies");
        assert!(stream_counts.forwarded_removed > 0, "{name}: removals");
        assert_eq!(stream_counts.forwarded_cleared, 1, "{name}: the clear");

        let mut state = LiveState::new();
        for batch in &batches {
            state.apply(batch);
        }
        let through = batches.iter().map(|b| b.sequence_number).max().unwrap();
        let chunks: Vec<KvEventBatch> =
            SnapshotChunks::new(state.snapshot(), through, 1.0, 0).collect();

        let want = reference(&batches);
        let got = emitted(&chunks);
        assert_eq!(got, want, "{name}: the emitted live set");
        assert_eq!(
            state.blocks(),
            want.values().map(|&copies| u64::from(copies)).sum::<u64>(),
            "{name}: live copies"
        );
        assert_eq!(state.entries(), want.len(), "{name}: live entries");
        assert!(
            want.keys().any(|(_, tier, _)| *tier != 1),
            "{name}: the stream has host-tier entries"
        );
        assert!(chunks.len() > 1, "{name}: the live set spans chunks");

        // Framing: the clear first, every chunk marked, stamps up to `through`.
        assert!(matches!(
            chunks[0].events[0].data,
            Some(kv_cache_event::Data::Cleared(_))
        ));
        let count = chunks.len() as u32;
        for (index, chunk) in chunks.iter().enumerate() {
            let marker = chunk.snapshot.as_ref().expect("marked");
            assert_eq!(
                (marker.index, marker.count),
                (index as u32, count),
                "{name}"
            );
            assert_eq!(marker.blocks, state.blocks(), "{name}");
            assert_eq!(
                chunk.sequence_number,
                through + 1 - u64::from(count) + index as u64,
                "{name}: stamp of chunk {index}"
            );
            let blocks: usize = stores(std::slice::from_ref(chunk))
                .iter()
                .map(|(_, stored)| stored.blocks.len())
                .sum();
            assert!(
                blocks <= CHUNK_BLOCKS,
                "{name}: chunk {index} holds {blocks}"
            );
        }

        // Order: a live parent precedes its children, per rank.
        let live_hashes: HashSet<(Option<i32>, i64)> =
            want.keys().map(|(rank, _, hash)| (*rank, *hash)).collect();
        let mut seen: HashSet<(Option<i32>, i64)> = HashSet::new();
        for (rank, stored) in stores(&chunks) {
            if let Some(parent) = stored.parent_block_hash {
                assert!(
                    seen.contains(&(rank, parent)) || !live_hashes.contains(&(rank, parent)),
                    "{name}: parent {parent} of {} emitted after its child",
                    stored.blocks[0].block_hash
                );
            }
            for block in &stored.blocks {
                seen.insert((rank, block.block_hash));
            }
        }

        // The engine's own hash over the emitted chains: every block checked
        // with a known parent, every digest reproduced.
        let counts = rehash(&chunks, case.engine);
        let blocks: u64 = got.values().map(|&copies| u64::from(copies)).sum();
        assert_eq!(counts.hash_checked, blocks, "{name}: every block rehashed");
        assert_eq!(counts.hash_unverifiable, 0, "{name}: a parent was missing");
        assert_eq!(counts.hash_mismatch, 0, "{name}: the chain rehashes");
        eprintln!(
            "{name}: {} batches -> {} live entries, {} copies, {} chunks; stream hash check \
             {}/{} mismatched, snapshot {}/{}",
            batches.len(),
            state.entries(),
            state.blocks(),
            chunks.len(),
            stream_counts.hash_mismatch,
            stream_counts.hash_checked,
            counts.hash_mismatch,
            counts.hash_checked,
        );
    }
}

/// Cutting the stream anywhere and snapshotting there equals the reference
/// at that point: the state follows removals and clears, not just stores.
#[test]
fn snapshots_at_every_cut_of_a_stream_follow_the_reference() {
    let case = &CASES[0];
    let (batches, _) = generated(case.engine, case.seed, 600);
    let mut state = LiveState::new();
    let mut want = LiveSet::new();
    for batch in &batches {
        state.apply(batch);
        replay(&mut want, batch);
        let chunks: Vec<KvEventBatch> =
            SnapshotChunks::new(state.snapshot(), batch.sequence_number, 1.0, 0).collect();
        assert_eq!(
            emitted(&chunks),
            want,
            "after batch {}",
            batch.sequence_number
        );
    }
}
