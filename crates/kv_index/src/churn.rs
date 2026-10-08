//! Churn generator for the chain index: workers with block-LRU caches serving prefix requests
//! over a shared chain pool, the way the mock engines do in a soak, so the index sees the
//! stores, evictions and heals that fragment runs over hours. Shared by the churn bench
//! (timing, series, JSON) and the gate test (a few minutes of events, a bound on runs live).
//!
//! Every request is one lookup (timed by the caller through the report) routed to the worker
//! with the longest prefix, which then "computes" the blocks it lacks and stores them under the
//! last block it holds, touches the whole prefix in its LRU, and evicts past its capacity in
//! the configured order: least recently used first, equal-age blocks tail-first (the mock's
//! order, which frees suffixes) or by hash (which frees middle stretches).

#![expect(clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    chain_prefix_hash, compute_content_hash, request_prefix_hashes, ChainBlockMap, ContentHash,
    ReferenceIndexer, SequenceHash, ShardedChainIndex, StoredBlock,
};

/// Order among equal-age blocks when a worker frees past its capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeOrder {
    /// Highest position of the chain first: an eviction takes a suffix.
    TailFirst,
    /// By block hash: an eviction takes stretches anywhere, holes included.
    Hash,
}

#[derive(Clone, Debug)]
pub struct ChurnConfig {
    pub workers: usize,
    pub chains: usize,
    /// Blocks per chain, drawn uniformly from `min_chain_len..=max_chain_len`.
    pub min_chain_len: usize,
    pub max_chain_len: usize,
    /// Chains after the first share a prefix with an earlier chain with this probability; the
    /// shared length is uniform over the parent's length, which is where runs branch.
    pub share: f64,
    /// Blocks a worker keeps before it evicts.
    pub cache_blocks: usize,
    pub free_order: FreeOrder,
    /// Blocks a request generates after its prompt prefix, as the mock's decode does: private
    /// content stored block by block under the prefix, released to the LRU with the rest.
    pub decode_blocks: usize,
    /// Probability that a request decodes at all (an output shorter than a block stores none).
    pub decode_probability: f64,
    /// Every this many requests one worker restarts: its cache is emptied (`apply_cleared`) and
    /// the next `refill_requests` requests are routed to it whatever the holders, as a balancing
    /// router does with a cold engine. Zero: never.
    pub restart_every: u64,
    pub refill_requests: u64,
    pub seed: u64,
}

/// A chain of the pool with the offset where it diverged from its parent (its own length when
/// it has none): the content's branch points, which bound how many runs the index needs.
pub struct Chain {
    pub contents: Vec<ContentHash>,
    pub blocks: Vec<StoredBlock>,
    pub parent: Option<usize>,
    pub divergence: usize,
}

pub struct Pool {
    pub chains: Vec<Chain>,
}

impl Pool {
    pub fn generate(cfg: &ChurnConfig, rng: &mut Rng) -> Self {
        let mut chains: Vec<Chain> = Vec::with_capacity(cfg.chains);
        for stream in 0..cfg.chains {
            let len = cfg.min_chain_len + rng.below(cfg.max_chain_len - cfg.min_chain_len + 1);
            let (parent, shared) = if stream > 0 && rng.unit() < cfg.share {
                let parent = rng.below(stream);
                let shared = 1 + rng.below(
                    chains[parent]
                        .contents
                        .len()
                        .min(len.saturating_sub(1))
                        .max(1),
                );
                (Some(parent), shared.min(len))
            } else {
                (None, 0)
            };
            let mut contents = Vec::with_capacity(len);
            if let Some(parent) = parent {
                contents.extend_from_slice(&chains[parent].contents[..shared]);
            }
            for position in contents.len()..len {
                contents.push(content(stream as u64, position));
            }
            let blocks: Vec<StoredBlock> = contents
                .iter()
                .zip(request_prefix_hashes(&contents))
                .map(|(&content_hash, seq_hash)| StoredBlock {
                    seq_hash,
                    content_hash,
                })
                .collect();
            chains.push(Chain {
                contents,
                blocks,
                parent,
                divergence: if parent.is_some() { shared } else { len },
            });
        }
        Self { chains }
    }

    /// Distinct branch points of the content: pairs (parent chain, offset) where a chain leaves
    /// its parent, counted once each. With every chain stored whole, the index needs at most one
    /// run per chain plus one per branch point; the gate test holds runs live to a multiple of
    /// that.
    pub fn divergence_points(&self) -> usize {
        let mut points = BTreeSet::new();
        for chain in &self.chains {
            if let Some(parent) = chain.parent {
                points.insert((parent, chain.divergence));
            }
        }
        points.len()
    }
}

fn content(stream: u64, position: usize) -> ContentHash {
    compute_content_hash(&[stream as u32, (stream >> 32) as u32, position as u32])
}

/// xorshift64*, enough for a workload generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A block in a worker's cache: where it sits and when it was last touched.
#[derive(Clone, Copy)]
struct Held {
    chain: usize,
    position: usize,
    age: u64,
    /// A decode block (private tail) rather than pool content.
    decode: bool,
}

/// Eviction order key: age first, then the configured tie-break.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EvictKey {
    age: u64,
    tie: u64,
    chain: usize,
    position: usize,
}

struct Worker {
    id: u32,
    map: ChainBlockMap,
    held: HashMap<SequenceHash, Held>,
    order: BTreeMap<EvictKey, SequenceHash>,
}

impl Worker {
    fn tie(free_order: FreeOrder, _chain: usize, position: usize, hash: SequenceHash) -> u64 {
        match free_order {
            // Tail first: a higher position sorts earlier among equal ages.
            FreeOrder::TailFirst => u64::MAX - position as u64,
            FreeOrder::Hash => hash.0,
        }
    }

    fn touch(
        &mut self,
        free_order: FreeOrder,
        chain: usize,
        position: usize,
        hash: SequenceHash,
        age: u64,
        decode: bool,
    ) {
        if let Some(old) = self.held.get(&hash).copied() {
            let key = EvictKey {
                age: old.age,
                tie: Self::tie(free_order, old.chain, old.position, hash),
                chain: old.chain,
                position: old.position,
            };
            self.order.remove(&key);
        }
        self.held.insert(
            hash,
            Held {
                chain,
                position,
                age,
                decode,
            },
        );
        let key = EvictKey {
            age,
            tie: Self::tie(free_order, chain, position, hash),
            chain,
            position,
        };
        self.order.insert(key, hash);
    }

    /// Blocks past the capacity, least recently used first, equal ages in the configured order.
    fn victims(&mut self, capacity: usize) -> Vec<SequenceHash> {
        let excess = self.held.len().saturating_sub(capacity);
        let mut out = Vec::with_capacity(excess);
        while out.len() < excess {
            let Some((&key, &hash)) = self.order.iter().next() else {
                break;
            };
            self.order.remove(&key);
            self.held.remove(&hash);
            out.push(hash);
        }
        out
    }

    /// Longest held prefix of `chain`: the engine's own prefix match stops at the first block
    /// it lacks, whatever it holds beyond.
    fn held_prefix(&self, chain: &Chain) -> usize {
        chain
            .blocks
            .iter()
            .take_while(|block| self.held.contains_key(&block.seq_hash))
            .count()
    }
}

/// What one request did.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepReport {
    pub lookup_ns: u64,
    pub runs_walked: usize,
    pub holders: usize,
    pub best_score: u32,
    pub stored_blocks: usize,
    pub removed_blocks: usize,
    /// Stored blocks that lay before a block the worker still held: a hole healed.
    pub healed_blocks: usize,
    /// Decode blocks stored (private tails).
    pub decoded_blocks: usize,
}

pub struct Churn {
    cfg: ChurnConfig,
    pub pool: Pool,
    workers: Vec<Worker>,
    rng: Rng,
    clock: u64,
    scores: Vec<(u32, u32)>,
    /// A restarted worker still being refilled: its index and the requests left to send it.
    refill: Option<(usize, u64)>,
    pub restarts: u64,
}

impl Churn {
    /// Interns `cfg.workers` workers round robin over the shards.
    pub fn new(cfg: ChurnConfig, index: &ShardedChainIndex) -> Self {
        let mut rng = Rng::new(cfg.seed);
        let pool = Pool::generate(&cfg, &mut rng);
        let workers = (0..cfg.workers)
            .map(|w| Worker {
                id: index
                    .intern_worker_in(w % index.shards(), &format!("churn-{w}"))
                    .expect("worker slots"),
                map: ChainBlockMap::default(),
                held: HashMap::new(),
                order: BTreeMap::new(),
            })
            .collect();
        Self {
            cfg,
            pool,
            workers,
            rng,
            clock: 0,
            scores: Vec::new(),
            refill: None,
            restarts: 0,
        }
    }

    /// Decode blocks still held by some worker: an upper bound on the live private tails, each of
    /// which is a run of its own.
    pub fn live_decode_blocks(&self) -> usize {
        self.workers
            .iter()
            .map(|w| w.held.values().filter(|h| h.decode).count())
            .sum()
    }

    pub fn worker_ids(&self) -> Vec<u32> {
        self.workers.iter().map(|w| w.id).collect()
    }

    /// One request: a lookup over a random prefix of a random chain, routed to the best holder,
    /// which stores what it lacks and evicts past capacity. `reference` receives the same events.
    pub fn step(
        &mut self,
        index: &ShardedChainIndex,
        mut reference: Option<&mut ReferenceIndexer>,
    ) -> StepReport {
        self.clock += 1;
        if self.cfg.restart_every > 0 && self.clock.is_multiple_of(self.cfg.restart_every) {
            // A worker restarts: everything it held is gone, and it is refilled cold.
            let which = self.rng.below(self.workers.len());
            let worker = &mut self.workers[which];
            index.apply_cleared(worker.id, &mut worker.map);
            if let Some(reference) = reference.as_mut() {
                reference.apply_cleared(worker.id);
            }
            worker.held.clear();
            worker.order.clear();
            self.refill = Some((which, self.cfg.refill_requests));
            self.restarts += 1;
        }
        let chain_pick = self.rng.below(self.pool.chains.len());
        let chain = &self.pool.chains[chain_pick];
        let prefix = 1 + self.rng.below(chain.contents.len());
        let query = &chain.contents[..prefix];
        self.scores.clear();
        let started = std::time::Instant::now();
        let walked = index.score_into(
            query,
            |c| c.0,
            false,
            |worker, score| {
                self.scores.push((worker, score));
            },
        );
        let lookup_ns = started.elapsed().as_nanos() as u64;
        let best = self.scores.iter().max_by_key(|(_, s)| *s).copied();
        let routed = match best {
            Some((worker, _)) => self
                .workers
                .iter()
                .position(|w| w.id == worker)
                .unwrap_or_else(|| self.rng.below(self.workers.len())),
            None => self.rng.below(self.workers.len()),
        };
        // A cold worker being refilled takes the request instead of the best holder.
        let worker_index = match self.refill {
            Some((which, left)) if left > 0 => {
                self.refill = Some((which, left - 1));
                which
            }
            _ => routed,
        };
        let mut report = StepReport {
            lookup_ns,
            runs_walked: walked,
            holders: self.scores.len(),
            best_score: best.map_or(0, |(_, s)| s),
            ..StepReport::default()
        };
        let free_order = self.cfg.free_order;
        let capacity = self.cfg.cache_blocks;
        let worker = &mut self.workers[worker_index];
        let known = worker.held_prefix(chain);
        if known < prefix {
            let blocks = &chain.blocks[known..prefix];
            let parent = (known > 0).then(|| chain.blocks[known - 1].seq_hash);
            // A block stored before one the worker still holds further along is a heal.
            report.healed_blocks = blocks
                .iter()
                .filter(|b| worker.held.contains_key(&b.seq_hash))
                .count();
            index
                .apply_stored(worker.id, blocks, parent, &mut worker.map)
                .expect("store after a held parent");
            if let Some(reference) = reference.as_mut() {
                reference
                    .apply_stored(worker.id, blocks, parent)
                    .expect("reference store");
            }
            report.stored_blocks = blocks.len();
        }
        for (position, block) in chain.blocks[..prefix].iter().enumerate() {
            worker.touch(
                free_order,
                chain_pick,
                position,
                block.seq_hash,
                self.clock,
                false,
            );
        }
        // Decode: private blocks appended under the prefix, one store per block as the mock
        // emits them, touched with the request and released with it.
        if self.cfg.decode_blocks > 0 && self.rng.unit() < self.cfg.decode_probability {
            let mut previous = chain.blocks[prefix - 1].seq_hash;
            for k in 0..self.cfg.decode_blocks {
                let content = compute_content_hash(&[
                    self.clock as u32,
                    (self.clock >> 32) as u32,
                    u32::MAX - k as u32,
                ]);
                let block = StoredBlock {
                    seq_hash: chain_prefix_hash(previous, content),
                    content_hash: content,
                };
                index
                    .apply_stored(
                        worker.id,
                        std::slice::from_ref(&block),
                        Some(previous),
                        &mut worker.map,
                    )
                    .expect("decode store under the block before");
                if let Some(reference) = reference.as_mut() {
                    reference
                        .apply_stored(worker.id, std::slice::from_ref(&block), Some(previous))
                        .expect("reference decode store");
                }
                worker.touch(
                    free_order,
                    chain_pick,
                    prefix + k,
                    block.seq_hash,
                    self.clock,
                    true,
                );
                previous = block.seq_hash;
                report.decoded_blocks += 1;
            }
        }
        let victims = worker.victims(capacity);
        if !victims.is_empty() {
            index.apply_removed(worker.id, &victims, &mut worker.map);
            if let Some(reference) = reference.as_mut() {
                reference.apply_removed(worker.id, &victims);
            }
            report.removed_blocks = victims.len();
        }
        report
    }

    /// Score a sample of chains against the reference: the exactness check at a checkpoint.
    pub fn check_exact(
        &mut self,
        index: &ShardedChainIndex,
        reference: &ReferenceIndexer,
        samples: usize,
    ) -> Result<(), String> {
        for _ in 0..samples {
            let chain = &self.pool.chains[self.rng.below(self.pool.chains.len())];
            let prefix = 1 + self.rng.below(chain.contents.len());
            let query = &chain.contents[..prefix];
            let mut ours: Vec<(u32, u32)> = index
                .find_matches(query, false)
                .scores
                .into_iter()
                .collect();
            let mut theirs: Vec<(u32, u32)> = reference.find_matches(query).into_iter().collect();
            ours.sort_unstable();
            theirs.sort_unstable();
            if ours != theirs {
                return Err(format!(
                    "prefix {prefix} of a chain: index {ours:?}, reference {theirs:?}"
                ));
            }
        }
        Ok(())
    }
}
