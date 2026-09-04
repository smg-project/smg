//! Generic prefix-membership index.
//!
//! The referee lives in `tests/differential.rs` (every
//! implementation must equal the reference model there, always). Two
//! cores share the contract:
//! [`RadixTree`] (the chain-native primary, `chain.rs`) and
//! [`FlatTree`] (below in this file) — a flat positional entry map
//! keyed by `(position, content)` with lineage-disambiguated
//! membership and an internal per-holder registry;
//! order-insensitive set semantics make convergence hold by
//! construction. Both are single-writer: no locks, no atomics, no
//! shards (§8).

#![forbid(unsafe_code)]

mod chain;
/// The chain-native core is the crate's primary type: chains stored
/// as contiguous data shared by every holder, membership as maximal
/// runs over interned holder sets, one entry probe per query. The
/// flat core remains as [`FlatTree`] — a second, independently
/// verified implementation the dual-core harness keeps asserting
/// against the model.
pub use chain::RadixTree;
use rustc_hash::FxHashMap;

/// Position-independent content identity (the matching currency).
pub type ContentHash = u64;
/// The publisher's removal key (backend block hash / placement chain
/// hash). Opaque to matching.
pub type BlockKey = u64;

/// Generational holder id (§5): operations through a retired id fail
/// loudly, never alias a recycled slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HolderId {
    index: u32,
    generation: u64,
}

impl HolderId {
    /// Raw (index, generation) — for callers keeping side tables
    /// keyed by holder. Stale ids stay detectable through the
    /// generation.
    pub fn parts(self) -> (u32, u64) {
        (self.index, self.generation)
    }

    pub(crate) fn assemble(index: u32, generation: u64) -> Self {
        Self { index, generation }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Hard bound on chain length; a store extending past it is
    /// rejected whole (`StoreError::ChainTooLong`).
    pub max_chain_len: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_chain_len: 65_536,
        }
    }
}

/// Advisory (§4): outside the convergence contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreOutcome {
    pub applied: u32,
    pub duplicates: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// Unknown or stale-generation holder id.
    UnknownHolder,
    /// Parent key not registered; the ONLY error whose sanctioned
    /// recovery is re-anchoring at position 0 (§4).
    ParentNotFound,
    /// Batch would extend past `max_chain_len`; terminal, do not
    /// re-anchor.
    ChainTooLong,
}

/// One query answer row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overlap {
    pub holder: HolderId,
    pub depth: u32,
    pub total_blocks: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub holders: u64,
    /// Sum of per-holder block counts (capacity arithmetic).
    pub holder_blocks: u64,
    /// Unique (position, content, lineage) memberships-bearing
    /// entries — the oracle-parity metric (§4).
    pub distinct_entries: u64,
    /// Rough resident estimate; drift-tolerant, monotone with state.
    pub bytes_estimate: u64,
}

/// A block's placement inside its holder: position, content, and the
/// lineage fingerprint of the chain prefix it was stored under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockInfo {
    pos: u32,
    content: ContentHash,
    lineage: u64,
}

#[derive(Debug, Default)]
struct HolderState {
    name: String,
    /// BlockKey -> placement. Internal — never caller-owned (§4).
    /// Deliberately the ONLY per-block holder structure: position
    /// order (truncate_tail/enumerate — cold capacity/snapshot
    /// paths) is derived from it on demand, costing those calls an
    /// O(n log n) scan instead of costing every block 16 resident
    /// bytes of standing order log.
    registry: FxHashMap<BlockKey, BlockInfo>,
}

impl HolderState {
    /// (position, key) pairs in ascending order, derived on demand.
    fn ordered(&self) -> Vec<(u32, BlockKey)> {
        let mut v: Vec<(u32, BlockKey)> = self.registry.iter().map(|(&k, i)| (i.pos, k)).collect();
        v.sort_unstable();
        v
    }
}

#[derive(Debug)]
struct HolderSlot {
    generation: u64,
    /// `None` = retired slot awaiting reuse.
    state: Option<HolderState>,
}

/// Lineage fingerprint chain: an INTERNAL rolling 64-bit mix — not
/// the wire hash scheme (the crate is hash-agnostic, §2).
#[inline]
fn lineage_root(content: ContentHash) -> u64 {
    splitmix(0xA0761D6478BD642F ^ content)
}

#[inline]
fn lineage_step(prev: u64, content: ContentHash) -> u64 {
    splitmix(prev.rotate_left(23) ^ content.wrapping_mul(0x9E3779B97F4A7C15))
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Interned, immutable, sorted holder set. Consecutive positions of a
/// shared chain hold IDENTICAL sets; hash-consing them means (a) the
/// query walk proves "membership unchanged since the previous
/// position" by POINTER EQUALITY — a sound O(1) run-skip, unlike the
/// oracle's count heuristic — and (b) the per-position duplication of
/// holder arrays collapses to one allocation per distinct set (the
/// run-compression memory win, without a linked tree).
type SetRef = std::sync::Arc<[u32]>;

#[derive(Debug, Default)]
struct SetInterner {
    /// set-content hash -> interned sets with that hash.
    table: FxHashMap<u64, Vec<SetRef>>,
}

impl SetInterner {
    fn hash_of(set: &[u32]) -> u64 {
        let mut h = 0x51_7C_C1_B7_27_22_0A_95u64 ^ (set.len() as u64);
        for &x in set {
            h = splitmix(h ^ (x as u64).wrapping_mul(0x9E3779B97F4A7C15));
        }
        h
    }

    /// Intern a sorted holder array (consumes the scratch build).
    fn intern(&mut self, set: &[u32]) -> SetRef {
        // Empty sets are transient placeholders — a caller shrinking a
        // span to zero holders immediately converts it to `PosSet::Empty`
        // and drops this ref without `release`. Tabling it would strand
        // it there forever (strong_count 1), so keep empties out of the
        // table; no live span ever holds one.
        if set.is_empty() {
            return set.into();
        }
        let h = Self::hash_of(set);
        let bucket = self.table.entry(h).or_default();
        for existing in bucket.iter() {
            if existing.as_ref() == set {
                return existing.clone();
            }
        }
        let arc: SetRef = set.into();
        bucket.push(arc.clone());
        arc
    }

    /// Release a reference that a membership is dropping. When only
    /// the table still holds the set, it is removed (single-writer,
    /// so the count is exact here).
    fn release(&mut self, set: SetRef) {
        let h = Self::hash_of(&set);
        // The caller's clone + the table's = 2 when orphaned.
        if std::sync::Arc::strong_count(&set) == 2 {
            if let Some(bucket) = self.table.get_mut(&h) {
                bucket.retain(|s| !std::sync::Arc::ptr_eq(s, &set));
                if bucket.is_empty() {
                    self.table.remove(&h);
                }
            }
        }
    }
}

/// Membership at one (position, content) entry, lineage-disambiguated.
/// The common case — one holder, one lineage — is inline and
/// allocation-free (§9). Shared entries hold one dense sorted holder
/// array per lineage: 4 B/holder, slice-comparable, streamed by the
/// query merge.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Membership {
    One {
        lineage: u64,
        holder: u32,
    },
    /// Buckets sorted by lineage; holders are interned sorted sets.
    Many(Vec<(u64, SetRef)>),
}

/// What [`Membership::insert`] did — drives both counter maintenance
/// and the out-of-contract same-triple dedup in `store`.
#[derive(PartialEq, Eq)]
enum Inserted {
    /// The (lineage, holder) pair was already present: the holder
    /// already holds a block at this exact (position, content,
    /// lineage) under a DIFFERENT key. Out-of-contract; the caller
    /// treats it as a duplicate and does not register the new key.
    Existing,
    AddedToExistingLineage,
    AddedNewLineage,
}

impl Membership {
    fn insert(&mut self, lineage: u64, holder: u32, interner: &mut SetInterner) -> Inserted {
        match self {
            Membership::One {
                lineage: l,
                holder: h,
            } => {
                if *l == lineage && *h == holder {
                    Inserted::Existing
                } else if *l == lineage {
                    let set = if *h < holder {
                        [*h, holder]
                    } else {
                        [holder, *h]
                    };
                    *self = Membership::Many(vec![(lineage, interner.intern(&set))]);
                    Inserted::AddedToExistingLineage
                } else {
                    let mut buckets = vec![(*l, interner.intern(&[*h]))];
                    let pos = usize::from(*l < lineage);
                    buckets.insert(pos, (lineage, interner.intern(&[holder])));
                    *self = Membership::Many(buckets);
                    Inserted::AddedNewLineage
                }
            }
            Membership::Many(buckets) => {
                match buckets.binary_search_by_key(&lineage, |&(l, _)| l) {
                    Ok(bi) => {
                        let set = &buckets[bi].1;
                        match set.binary_search(&holder) {
                            Ok(_) => Inserted::Existing,
                            Err(at) => {
                                let mut grown = Vec::with_capacity(set.len() + 1);
                                grown.extend_from_slice(&set[..at]);
                                grown.push(holder);
                                grown.extend_from_slice(&set[at..]);
                                let old =
                                    std::mem::replace(&mut buckets[bi].1, interner.intern(&grown));
                                interner.release(old);
                                Inserted::AddedToExistingLineage
                            }
                        }
                    }
                    Err(bi) => {
                        buckets.insert(bi, (lineage, interner.intern(&[holder])));
                        Inserted::AddedNewLineage
                    }
                }
            }
        }
    }

    /// Remove; true when the whole membership is now empty.
    fn remove(&mut self, lineage: u64, holder: u32, interner: &mut SetInterner) -> bool {
        match self {
            Membership::One {
                lineage: l,
                holder: h,
            } => *l == lineage && *h == holder,
            Membership::Many(buckets) => {
                if let Ok(bi) = buckets.binary_search_by_key(&lineage, |&(l, _)| l) {
                    let set = &buckets[bi].1;
                    if let Ok(at) = set.binary_search(&holder) {
                        if set.len() == 1 {
                            let (_, old) = buckets.remove(bi);
                            interner.release(old);
                        } else {
                            let mut shrunk = Vec::with_capacity(set.len() - 1);
                            shrunk.extend_from_slice(&set[..at]);
                            shrunk.extend_from_slice(&set[at + 1..]);
                            let old =
                                std::mem::replace(&mut buckets[bi].1, interner.intern(&shrunk));
                            interner.release(old);
                        }
                    }
                }
                buckets.is_empty()
            }
        }
    }

    /// Is this exact (lineage, holder) pair present?
    fn contains(&self, lineage: u64, holder: u32) -> bool {
        match self {
            Membership::One {
                lineage: l,
                holder: h,
            } => *l == lineage && *h == holder,
            Membership::Many(buckets) => buckets
                .binary_search_by_key(&lineage, |&(l, _)| l)
                .is_ok_and(|bi| buckets[bi].1.binary_search(&holder).is_ok()),
        }
    }

    /// Does any OTHER holder share (lineage) here?
    fn lineage_shared_beyond(&self, lineage: u64, holder: u32) -> bool {
        match self {
            Membership::One {
                lineage: l,
                holder: h,
            } => *l == lineage && *h != holder,
            Membership::Many(buckets) => {
                match buckets.binary_search_by_key(&lineage, |&(l, _)| l) {
                    Ok(bi) => buckets[bi].1.iter().any(|&h| h != holder),
                    Err(_) => false,
                }
            }
        }
    }
}

/// Caller-owned query scratch: keeps `overlap` allocation-free once
/// warm while the tree itself stays `&self` on the read path (§8).
#[derive(Debug, Default)]
pub struct OverlapScratch {
    active: Vec<u32>,
    next: Vec<u32>,
    lineages: Vec<u64>,
}

pub struct FlatTree {
    cfg: Config,
    entries: FxHashMap<(u32, ContentHash), Membership>,
    slots: Vec<HolderSlot>,
    by_name: FxHashMap<String, u32>,
    free: Vec<u32>,
    holder_blocks_total: u64,
    distinct_lineage_entries: u64,
    interner: SetInterner,
}

impl FlatTree {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            entries: FxHashMap::default(),
            slots: Vec::new(),
            by_name: FxHashMap::default(),
            free: Vec::new(),
            holder_blocks_total: 0,
            distinct_lineage_entries: 0,
            interner: SetInterner::default(),
        }
    }

    // ---- holder lifecycle (§5) ----

    /// Create (or return the live holder of this name). Recycles
    /// retired slots; the returned id's generation detects staleness.
    pub fn create_holder(&mut self, name: &str) -> HolderId {
        if let Some(&index) = self.by_name.get(name) {
            return HolderId {
                index,
                generation: self.slots[index as usize].generation,
            };
        }
        let state = HolderState {
            name: name.to_string(),
            ..HolderState::default()
        };
        let index = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.state = Some(state);
            index
        } else {
            assert!(
                self.slots.len() < u32::MAX as usize,
                "holder slot space exhausted (u32 indices)"
            );
            self.slots.push(HolderSlot {
                generation: 0,
                state: Some(state),
            });
            (self.slots.len() - 1) as u32
        };
        self.by_name.insert(name.to_string(), index);
        HolderId {
            index,
            generation: self.slots[index as usize].generation,
        }
    }

    pub fn holder_name(&self, id: HolderId) -> Option<&str> {
        self.live(id).map(|s| s.name.as_str())
    }

    /// Release every byte attributable to the holder; the id becomes
    /// stale (generation bumped) and the slot reusable.
    pub fn retire_holder(&mut self, id: HolderId) {
        if self.live(id).is_none() {
            return;
        }
        self.clear(id);
        let slot = &mut self.slots[id.index as usize];
        let state = slot.state.take().expect("checked live");
        self.by_name.remove(&state.name);
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
    }

    /// O(1).
    pub fn holder_blocks(&self, id: HolderId) -> u64 {
        self.live(id).map_or(0, |s| s.registry.len() as u64)
    }

    /// Read-only: the position at which `holder` holds `key` (see
    /// `RadixTree::position_of`).
    pub fn position_of(&self, id: HolderId, key: BlockKey) -> Option<u32> {
        self.live(id)?.registry.get(&key).map(|info| info.pos)
    }

    // ---- writes (§4) ----

    /// All-or-nothing on error; see `StoreError` for the recovery
    /// contract per variant.
    pub fn store(
        &mut self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> Result<StoreOutcome, StoreError> {
        if self.live(id).is_none() {
            return Err(StoreError::UnknownHolder);
        }
        if blocks.is_empty() {
            return Ok(StoreOutcome {
                applied: 0,
                duplicates: 0,
            });
        }
        let state = self.slots[id.index as usize]
            .state
            .as_ref()
            .expect("checked live");
        // Resolve the anchor BEFORE any mutation (all-or-nothing).
        let (start_pos, mut lineage_prev) = match parent {
            Some(parent_key) => match state.registry.get(&parent_key) {
                None => return Err(StoreError::ParentNotFound),
                Some(info) => (info.pos + 1, Some(info.lineage)),
            },
            None => {
                // Re-publish dedup: a parent-None batch whose first
                // key already anchors a chain at position 0 extends
                // that chain (the model's rule; in-contract the fresh
                // lineage recomputation matches the stored one).
                (0, None)
            }
        };
        if start_pos as u64 + blocks.len() as u64 > self.cfg.max_chain_len as u64 {
            return Err(StoreError::ChainTooLong);
        }

        let mut applied = 0u32;
        let mut duplicates = 0u32;
        for (i, &(key, content)) in blocks.iter().enumerate() {
            let pos = start_pos + i as u32;
            let lineage = match lineage_prev {
                None => lineage_root(content),
                Some(prev) => lineage_step(prev, content),
            };
            lineage_prev = Some(lineage);
            let info = BlockInfo {
                pos,
                content,
                lineage,
            };
            let state = self.slots[id.index as usize]
                .state
                .as_mut()
                .expect("checked live");
            match state.registry.get(&key) {
                Some(existing) if *existing == info => {
                    duplicates += 1;
                    continue;
                }
                _ => {}
            }
            // §4 alias pre-check, BEFORE any mutation: if the holder
            // already holds this exact (position, content, lineage)
            // under another key, the store is a duplicate — and a
            // would-be MOVE is refused NON-destructively (the old
            // placement stays; review finding: the earlier order
            // unindexed first and destroyed a block while reporting
            // a no-op).
            let dest_taken = self
                .entries
                .get(&(info.pos, info.content))
                .is_some_and(|m| m.contains(info.lineage, id.index));
            if dest_taken {
                duplicates += 1;
                continue;
            }
            if let Some(&existing) = self.slots[id.index as usize]
                .state
                .as_ref()
                .expect("checked live")
                .registry
                .get(&key)
            {
                // §4: re-registration MOVES the block.
                Self::unindex(
                    &mut self.entries,
                    &mut self.interner,
                    &mut self.distinct_lineage_entries,
                    id.index,
                    existing,
                );
                self.slots[id.index as usize]
                    .state
                    .as_mut()
                    .expect("checked live")
                    .registry
                    .remove(&key);
                self.holder_blocks_total -= 1;
            }
            let inserted = self.index_block(id.index, info);
            debug_assert!(inserted, "alias pre-check guarantees insertion");
            let state = self.slots[id.index as usize]
                .state
                .as_mut()
                .expect("checked live");
            state.registry.insert(key, info);
            self.holder_blocks_total += 1;
            applied += 1;
        }
        Ok(StoreOutcome {
            applied,
            duplicates,
        })
    }

    /// Read-only no-op predicate: true iff `store(id, parent, blocks)`
    /// would apply nothing (see `RadixTree::covered` — same contract,
    /// gated against the model by the same fuzz assertions).
    pub fn covered(
        &self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> bool {
        self.dup_prefix(id, parent, blocks).1
    }

    /// See `RadixTree::dup_prefix` — same contract, gated against the
    /// model by the same fuzz assertions.
    pub fn dup_prefix(
        &self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> (u32, bool) {
        if self.live(id).is_none() {
            return (0, false);
        }
        if blocks.is_empty() {
            return (0, true);
        }
        let state = self.slots[id.index as usize]
            .state
            .as_ref()
            .expect("checked live");
        let (start_pos, mut lineage_prev) = match parent {
            Some(parent_key) => match state.registry.get(&parent_key) {
                None => return (0, false),
                Some(info) => (info.pos + 1, Some(info.lineage)),
            },
            None => (0, None),
        };
        if start_pos as u64 + blocks.len() as u64 > self.cfg.max_chain_len as u64 {
            return (0, false);
        }
        let mut run = 0u32;
        let mut run_live = true;
        for (i, &(key, content)) in blocks.iter().enumerate() {
            let pos = start_pos + i as u32;
            let lineage = match lineage_prev {
                None => lineage_root(content),
                Some(prev) => lineage_step(prev, content),
            };
            lineage_prev = Some(lineage);
            let info = BlockInfo {
                pos,
                content,
                lineage,
            };
            let plain = state.registry.get(&key) == Some(&info);
            let held = plain
                || self
                    .entries
                    .get(&(info.pos, info.content))
                    .is_some_and(|m| m.contains(info.lineage, id.index));
            if !held {
                return (run, false);
            }
            if run_live {
                if plain {
                    run += 1;
                } else {
                    run_live = false;
                }
            }
        }
        (run, true)
    }

    /// Idempotent; returns blocks actually removed (advisory).
    pub fn remove(&mut self, id: HolderId, keys: &[BlockKey]) -> u32 {
        if self.live(id).is_none() {
            return 0;
        }
        let mut removed = 0u32;
        for &key in keys {
            let state = self.slots[id.index as usize]
                .state
                .as_mut()
                .expect("checked live");
            let Some(info) = state.registry.remove(&key) else {
                continue;
            };
            self.holder_blocks_total -= 1;
            Self::unindex(
                &mut self.entries,
                &mut self.interner,
                &mut self.distinct_lineage_entries,
                id.index,
                info,
            );
            removed += 1;
        }
        removed
    }

    /// Forest-wide prefix-closed eviction (§4): drop blocks in
    /// strictly decreasing position order, ties by key, until `keep`
    /// remain. Returns dropped count.
    pub fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64 {
        if self.live(id).is_none() {
            return 0;
        }
        let state = self.slots[id.index as usize]
            .state
            .as_mut()
            .expect("checked live");
        if state.registry.len() as u64 <= keep {
            return 0;
        }
        let mut ordered = state.ordered();
        let mut dropped = 0u64;
        while ordered.len() as u64 > keep {
            let (pos, key) = ordered.pop().expect("non-empty");
            let state = self.slots[id.index as usize]
                .state
                .as_mut()
                .expect("checked live");
            let info = state.registry.remove(&key).expect("derived from registry");
            debug_assert_eq!(info.pos, pos);
            self.holder_blocks_total -= 1;
            Self::unindex(
                &mut self.entries,
                &mut self.interner,
                &mut self.distinct_lineage_entries,
                id.index,
                info,
            );
            dropped += 1;
        }
        dropped
    }

    /// Drop all blocks; the holder (id, name, epoch posture at the
    /// caller) survives — this is the epoch-bump primitive (§2).
    pub fn clear(&mut self, id: HolderId) {
        if self.live(id).is_none() {
            return;
        }
        let state = self.slots[id.index as usize]
            .state
            .as_mut()
            .expect("checked live");
        let registry = std::mem::take(&mut state.registry);
        self.holder_blocks_total -= registry.len() as u64;
        for (_, info) in registry {
            Self::unindex(
                &mut self.entries,
                &mut self.interner,
                &mut self.distinct_lineage_entries,
                id.index,
                info,
            );
        }
    }

    // ---- reads ----

    /// §6 exact overlap: lineage-true consecutive depth per holder.
    /// `out` is cleared and filled (holders at depth 0 absent).
    ///
    /// Two-phase walk: probing each position's entry has no data
    /// dependency on its neighbors (lineages are a pure rolling
    /// function of the QUERY), so phase 1 issues all map probes
    /// back-to-back — the CPU overlaps their cache misses — and only
    /// phase 2's cheap merge is sequential. This is what makes exact
    /// matching latency-competitive without the oracle's unsound
    /// skip heuristics.
    pub fn overlap(
        &self,
        chain: &[ContentHash],
        scratch: &mut OverlapScratch,
        out: &mut Vec<Overlap>,
    ) {
        out.clear();
        if chain.is_empty() {
            return;
        }
        // Phase 0: lineages up front.
        let lineages = &mut scratch.lineages;
        lineages.clear();
        let mut lineage = 0u64;
        for (p, &content) in chain.iter().enumerate() {
            lineage = if p == 0 {
                lineage_root(content)
            } else {
                lineage_step(lineage, content)
            };
            lineages.push(lineage);
        }
        // Phase 1: independent probes resolving each position's
        // dense holder run, endpoints touched so the run data rides
        // the same overlapped misses. Stops at the first absent
        // entry (no deeper position can matter).
        enum RunProbe<'a> {
            One(u32),
            Slice(&'a [u32]),
            Miss,
        }
        let mut probes: Vec<RunProbe> = Vec::with_capacity(chain.len().min(1024));
        for (p, &content) in chain.iter().enumerate() {
            match self.entries.get(&(p as u32, content)) {
                Some(Membership::One { lineage, holder }) => {
                    probes.push(if *lineage == lineages[p] {
                        RunProbe::One(*holder)
                    } else {
                        RunProbe::Miss
                    })
                }
                Some(Membership::Many(buckets)) => {
                    let run = if buckets.len() == 1 {
                        (buckets[0].0 == lineages[p]).then(|| &*buckets[0].1)
                    } else {
                        buckets
                            .binary_search_by_key(&lineages[p], |&(l, _)| l)
                            .ok()
                            .map(|bi| &*buckets[bi].1)
                    };
                    match run {
                        Some(run) => {
                            // Endpoint touches: pull the run's first
                            // and last cache lines now, overlapped.
                            std::hint::black_box(run[0]);
                            std::hint::black_box(run[run.len() - 1]);
                            probes.push(RunProbe::Slice(run));
                        }
                        None => probes.push(RunProbe::Miss),
                    }
                }
                None => break,
            }
            if matches!(probes.last(), Some(RunProbe::Miss)) {
                break;
            }
        }
        // Phase 2: sequential merge over dense holder runs.
        let mut active = std::mem::take(&mut scratch.active);
        let mut next = std::mem::take(&mut scratch.next);
        active.clear();
        next.clear();
        let mut survivors_depth = 0u32;
        let mut one_hold = [0u32; 1];
        // The interned-set identity `active` currently equals, when it
        // exactly equals one: pointer equality against a position's
        // run then proves "membership unchanged" in O(1), soundly
        // (interning guarantees identical sets share one allocation).
        let mut active_src: Option<*const u32> = None;
        for (p, probe) in probes.iter().enumerate() {
            let run: &[u32] = match probe {
                RunProbe::One(h) => {
                    one_hold[0] = *h;
                    &one_hold[..]
                }
                RunProbe::Slice(r) => r,
                RunProbe::Miss => &[],
            };
            if p == 0 {
                if run.is_empty() {
                    break;
                }
                active.extend_from_slice(run);
                if matches!(probe, RunProbe::Slice(_)) {
                    active_src = Some(run.as_ptr());
                }
            } else {
                if active_src.is_some_and(|src| std::ptr::eq(src, run.as_ptr())) {
                    // O(1) sound run-skip: same interned set object.
                } else if run == active.as_slice() {
                    // Content-equal after a hand-built subset: resync
                    // to the interned identity.
                    if matches!(probe, RunProbe::Slice(_)) {
                        active_src = Some(run.as_ptr());
                    }
                } else {
                    // Merge-intersect two sorted runs; dropped
                    // holders get depth p in the same pass.
                    next.clear();
                    let mut ai = 0usize;
                    for &h in run {
                        while ai < active.len() && active[ai] < h {
                            self.push_answer(active[ai], p as u32, out);
                            ai += 1;
                        }
                        if ai < active.len() && active[ai] == h {
                            next.push(h);
                            ai += 1;
                        }
                    }
                    while ai < active.len() {
                        self.push_answer(active[ai], p as u32, out);
                        ai += 1;
                    }
                    std::mem::swap(&mut active, &mut next);
                    // next ⊆ run; equal lengths ⟹ equal sets.
                    active_src = if active.len() == run.len() && matches!(probe, RunProbe::Slice(_))
                    {
                        Some(run.as_ptr())
                    } else {
                        None
                    };
                }
            }
            if active.is_empty() {
                break;
            }
            survivors_depth = p as u32 + 1;
        }
        // Holders still active survived every visited position.
        for &h in active.iter() {
            self.push_answer(h, survivors_depth, out);
        }
        drop(probes);
        scratch.active = active;
        scratch.next = next;
    }

    /// Position-ordered enumeration (snapshot/Pull; §4). Order is
    /// derived on demand — snapshot is a cold path.
    pub fn enumerate(
        &self,
        id: HolderId,
    ) -> impl Iterator<Item = (u32, BlockKey, ContentHash)> + '_ {
        let state = self.live(id);
        let ordered = state.map(|s| s.ordered()).unwrap_or_default();
        ordered.into_iter().map(move |(pos, key)| {
            let info = state
                .expect("ordered non-empty implies live")
                .registry
                .get(&key)
                .expect("derived from registry");
            (pos, key, info.content)
        })
    }

    pub fn stats(&self) -> Stats {
        let holders = self.slots.iter().filter(|s| s.state.is_some()).count() as u64;
        // Rough model: entry map + memberships + registries.
        let bytes = self.entries.len() as u64 * 48 + self.holder_blocks_total * 72 + holders * 128;
        Stats {
            holders,
            holder_blocks: self.holder_blocks_total,
            distinct_entries: self.distinct_lineage_entries,
            bytes_estimate: bytes,
        }
    }

    // ---- verification ----

    /// Full-state consistency audit: recomputes every counter and
    /// cross-checks every structure from first principles. O(state);
    /// meant for the fuzz harness and debug assertions, not the hot
    /// path. Returns the first violation found.
    pub fn audit(&self) -> Result<(), String> {
        use std::collections::HashSet;
        // Slot / name / free-list coherence.
        let mut live = 0u64;
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(state) = &slot.state {
                live += 1;
                match self.by_name.get(&state.name) {
                    Some(&i) if i as usize == idx => {}
                    other => {
                        return Err(format!(
                            "by_name[{}] = {:?}, expected {}",
                            state.name, other, idx
                        ))
                    }
                }
            }
        }
        if self.by_name.len() as u64 != live {
            return Err(format!(
                "by_name has {} entries, {} live slots",
                self.by_name.len(),
                live
            ));
        }
        let mut seen_free = HashSet::new();
        for &f in &self.free {
            if f as usize >= self.slots.len() {
                return Err(format!("free-list index {f} out of range"));
            }
            if self.slots[f as usize].state.is_some() {
                return Err(format!("free-list index {f} is live"));
            }
            if !seen_free.insert(f) {
                return Err(format!("free-list duplicate {f}"));
            }
        }
        // Counters + forward containment (registry -> entries).
        let mut blocks = 0u64;
        for (idx, slot) in self.slots.iter().enumerate() {
            let Some(state) = &slot.state else { continue };
            blocks += state.registry.len() as u64;
            for (&key, info) in &state.registry {
                let Some(m) = self.entries.get(&(info.pos, info.content)) else {
                    return Err(format!(
                        "registry block {key:#x} of holder {idx} at ({},{:#x}) has no entry",
                        info.pos, info.content
                    ));
                };
                let mut found = false;
                match m {
                    Membership::One { lineage, holder } => {
                        found = *lineage == info.lineage && *holder as usize == idx;
                    }
                    Membership::Many(buckets) => {
                        if let Ok(bi) = buckets.binary_search_by_key(&info.lineage, |&(l, _)| l) {
                            found = buckets[bi].1.binary_search(&(idx as u32)).is_ok();
                        }
                    }
                }
                if !found {
                    return Err(format!(
                        "membership missing for holder {idx} block {key:#x} at ({},{:#x}) lineage {:#x}",
                        info.pos, info.content, info.lineage
                    ));
                }
            }
        }
        if blocks != self.holder_blocks_total {
            return Err(format!(
                "holder_blocks_total {} != recount {}",
                self.holder_blocks_total, blocks
            ));
        }
        // Reverse containment (entries -> registries) + shape + distinct.
        let mut holder_triples: Vec<HashSet<(u32, u64, u64)>> =
            vec![HashSet::new(); self.slots.len()];
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(state) = &slot.state {
                for info in state.registry.values() {
                    holder_triples[idx].insert((info.pos, info.content, info.lineage));
                }
            }
        }
        let mut distinct = 0u64;
        for (&(pos, content), m) in &self.entries {
            match m {
                Membership::One { lineage, holder } => {
                    distinct += 1;
                    if self
                        .slots
                        .get(*holder as usize)
                        .and_then(|s| s.state.as_ref())
                        .is_none()
                    {
                        return Err(format!(
                            "entry ({pos},{content:#x}) holds retired holder {holder}"
                        ));
                    }
                    if !holder_triples[*holder as usize].contains(&(pos, content, *lineage)) {
                        return Err(format!(
                            "entry ({pos},{content:#x}) lineage {lineage:#x} not in holder {holder}'s registry"
                        ));
                    }
                }
                Membership::Many(buckets) => {
                    if buckets.is_empty() {
                        return Err(format!("empty Many at ({pos},{content:#x})"));
                    }
                    let mut prev_l = None;
                    for (l, holders) in buckets {
                        distinct += 1;
                        if holders.is_empty() {
                            return Err(format!("empty bucket at ({pos},{content:#x})"));
                        }
                        if prev_l.is_some_and(|p: u64| p >= *l) {
                            return Err(format!("unsorted buckets at ({pos},{content:#x})"));
                        }
                        prev_l = Some(*l);
                        let mut prev_h = None;
                        for &h in holders.iter() {
                            if prev_h.is_some_and(|p: u32| p >= h) {
                                return Err(format!("unsorted holders at ({pos},{content:#x})"));
                            }
                            prev_h = Some(h);
                            if self
                                .slots
                                .get(h as usize)
                                .and_then(|s| s.state.as_ref())
                                .is_none()
                            {
                                return Err(format!(
                                    "entry ({pos},{content:#x}) holds retired holder {h}"
                                ));
                            }
                            if !holder_triples[h as usize].contains(&(pos, content, *l)) {
                                return Err(format!(
                                    "entry ({pos},{content:#x}) lineage {l:#x} not in holder {h}'s registry"
                                ));
                            }
                        }
                    }
                }
            }
        }
        if distinct != self.distinct_lineage_entries {
            return Err(format!(
                "distinct_lineage_entries {} != recount {}",
                self.distinct_lineage_entries, distinct
            ));
        }
        // Interner coherence: every bucket's set is findable in the
        // table under its content hash, and the table holds no orphan
        // sets (kept alive by nothing but the table = a leak).
        for m in self.entries.values() {
            if let Membership::Many(buckets) = m {
                for (_, set) in buckets {
                    let h = SetInterner::hash_of(set);
                    let found = self
                        .interner
                        .table
                        .get(&h)
                        .is_some_and(|v| v.iter().any(|s| std::sync::Arc::ptr_eq(s, set)));
                    if !found {
                        return Err(format!("interned set {:?} missing from table", &set[..]));
                    }
                }
            }
        }
        for (h, sets) in &self.interner.table {
            for set in sets {
                if std::sync::Arc::strong_count(set) == 1 {
                    return Err(format!(
                        "orphan interned set under hash {h:#x}: {:?}",
                        &set[..]
                    ));
                }
            }
        }
        Ok(())
    }

    // ---- internals ----

    fn live(&self, id: HolderId) -> Option<&HolderState> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.state.as_ref()
    }

    fn push_answer(&self, holder: u32, depth: u32, out: &mut Vec<Overlap>) {
        if depth == 0 {
            return;
        }
        let slot = &self.slots[holder as usize];
        let Some(state) = slot.state.as_ref() else {
            return;
        };
        out.push(Overlap {
            holder: HolderId {
                index: holder,
                generation: slot.generation,
            },
            depth,
            total_blocks: state.registry.len() as u64,
        });
    }

    /// Add the (lineage, holder) membership for a block. Returns
    /// false for the out-of-contract case where the holder ALREADY
    /// holds this exact (position, content, lineage) under another
    /// key — the caller then treats the store as a duplicate and the
    /// new key is never registered (deterministic; audited).
    fn index_block(&mut self, holder: u32, info: BlockInfo) -> bool {
        let interner = &mut self.interner;
        match self.entries.entry((info.pos, info.content)) {
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(Membership::One {
                    lineage: info.lineage,
                    holder,
                });
                self.distinct_lineage_entries += 1;
                true
            }
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                match occupied.get_mut().insert(info.lineage, holder, interner) {
                    Inserted::Existing => false,
                    Inserted::AddedToExistingLineage => true,
                    Inserted::AddedNewLineage => {
                        self.distinct_lineage_entries += 1;
                        true
                    }
                }
            }
        }
    }

    fn unindex(
        entries: &mut FxHashMap<(u32, ContentHash), Membership>,
        interner: &mut SetInterner,
        distinct: &mut u64,
        holder: u32,
        info: BlockInfo,
    ) {
        let Some(membership) = entries.get_mut(&(info.pos, info.content)) else {
            return;
        };
        let shared = membership.lineage_shared_beyond(info.lineage, holder);
        if membership.remove(info.lineage, holder, interner) {
            entries.remove(&(info.pos, info.content));
        }
        if !shared {
            *distinct -= 1;
        }
    }
}
