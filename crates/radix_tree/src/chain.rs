//! The chain-native core: chains as contiguous data, canonical
//! maximal-run membership spans, one entry probe per query. Same
//! public API and matching/convergence contract as the flat core;
//! the differential referee proves equality.

use rustc_hash::FxHashMap;

use crate::{
    lineage_root, lineage_step, BlockKey, Config, ContentHash, HolderId, Overlap, OverlapScratch,
    SetInterner, SetRef, Stats, StoreError, StoreOutcome,
};

/// One trie chain: contiguous contents/keys stored ONCE, shared by
/// every holder covering any span of it.
#[derive(Debug, Default)]
struct ChainData {
    /// Trie edge: (parent chain, absolute position of the parent
    /// block). `None` = a root chain (base_pos == 0).
    parent: Option<(u32, u32)>,
    /// Absolute position of contents[0].
    base_pos: u32,
    contents: Vec<ContentHash>,
    /// Lineage fingerprint at contents[0] (audit + canonical iden).
    start_lineage: u64,
    /// Lineage at the last position (extension cache).
    end_lineage: u64,
    /// Canonical maximal-run membership: sorted, non-overlapping,
    /// non-adjacent-equal, non-empty sets, within [base_pos,
    /// base_pos + len).
    spans: Vec<Span>,
    /// Child forks: (fork position ON THIS CHAIN, first child
    /// content, child chain), sorted.
    children: Vec<(u32, ContentHash, u32)>,
}

#[derive(Debug, Clone)]
struct Span {
    start: u32,
    len: u32,
    holders: SetRef,
}

impl ChainData {
    fn end_pos(&self) -> u32 {
        self.base_pos + self.contents.len() as u32
    }
    fn content_at(&self, pos: u32) -> ContentHash {
        self.contents[(pos - self.base_pos) as usize]
    }
    fn child_at(&self, fork_pos: u32, content: ContentHash) -> Option<u32> {
        self.children
            .binary_search_by_key(&(fork_pos, content), |&(p, c, _)| (p, c))
            .ok()
            .map(|i| self.children[i].2)
    }
    /// Is `holder` in the span covering `pos` (if any)?
    fn covered(&self, pos: u32, holder: u32) -> bool {
        self.span_index(pos)
            .is_some_and(|i| self.spans[i].holders.binary_search(&holder).is_ok())
    }
    fn span_index(&self, pos: u32) -> Option<usize> {
        let i = self.spans.partition_point(|s| s.start + s.len <= pos);
        (i < self.spans.len() && self.spans[i].start <= pos).then_some(i)
    }
}

#[derive(Debug, Default)]
struct HolderState3 {
    name: String,
    /// key -> (chain, absolute pos). Per-holder (out-of-contract
    /// inputs can register one key differently across holders).
    keys: FxHashMap<BlockKey, (u32, u32)>,
    /// Chains this holder covers (maintenance index).
    chains: rustc_hash::FxHashSet<u32>,
}

#[derive(Debug)]
struct Slot3 {
    generation: u64,
    state: Option<HolderState3>,
}

pub struct RadixTree {
    cfg: Config,
    chains: Vec<ChainData>,
    free_chains: Vec<u32>,
    /// Root chains by their start lineage. The value is a collision
    /// list (essentially always length 1): entries are CONTENT-
    /// verified on every resolution, which makes R3 immune to
    /// fingerprint collisions — the flat core cannot verify (it
    /// stores no contents), R3 always can.
    roots: FxHashMap<u64, Vec<u32>>,
    slots: Vec<Slot3>,
    by_name: FxHashMap<String, u32>,
    free: Vec<u32>,
    interner: SetInterner,
    holder_blocks_total: u64,
    /// Distinct covered (position, content, lineage) = chain
    /// positions with a non-empty holder set.
    distinct_entries: u64,
}

impl RadixTree {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            chains: Vec::new(),
            free_chains: Vec::new(),
            roots: FxHashMap::default(),
            slots: Vec::new(),
            by_name: FxHashMap::default(),
            free: Vec::new(),
            interner: SetInterner::default(),
            holder_blocks_total: 0,
            distinct_entries: 0,
        }
    }

    // ---- holder lifecycle (mirrors the flat core exactly) ----

    pub fn create_holder(&mut self, name: &str) -> HolderId {
        if let Some(&index) = self.by_name.get(name) {
            return HolderId::assemble(index, self.slots[index as usize].generation);
        }
        let state = HolderState3 {
            name: name.to_string(),
            ..HolderState3::default()
        };
        let index = if let Some(index) = self.free.pop() {
            self.slots[index as usize].state = Some(state);
            index
        } else {
            assert!(
                self.slots.len() < u32::MAX as usize,
                "holder slot space exhausted (u32 indices)"
            );
            self.slots.push(Slot3 {
                generation: 0,
                state: Some(state),
            });
            (self.slots.len() - 1) as u32
        };
        self.by_name.insert(name.to_string(), index);
        HolderId::assemble(index, self.slots[index as usize].generation)
    }

    pub fn holder_name(&self, id: HolderId) -> Option<&str> {
        self.live(id).map(|s| s.name.as_str())
    }

    pub fn retire_holder(&mut self, id: HolderId) {
        if self.live(id).is_none() {
            return;
        }
        self.clear(id);
        let slot = &mut self.slots[id.parts().0 as usize];
        let state = slot.state.take().expect("checked live");
        self.by_name.remove(&state.name);
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.parts().0);
    }

    pub fn holder_blocks(&self, id: HolderId) -> u64 {
        self.live(id).map_or(0, |s| s.keys.len() as u64)
    }

    fn live(&self, id: HolderId) -> Option<&HolderState3> {
        let (index, generation) = id.parts();
        let slot = self.slots.get(index as usize)?;
        if slot.generation != generation {
            return None;
        }
        slot.state.as_ref()
    }

    // ---- writes ----

    pub fn store(
        &mut self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> Result<StoreOutcome, StoreError> {
        if self.live(id).is_none() {
            return Err(StoreError::UnknownHolder);
        }
        let holder = id.parts().0;
        if blocks.is_empty() {
            return Ok(StoreOutcome {
                applied: 0,
                duplicates: 0,
            });
        }
        // Resolve the anchor BEFORE any mutation.
        let anchor: Option<(u32, u32)> = match parent {
            Some(parent_key) => {
                let state = self.state_of(holder);
                match state.keys.get(&parent_key) {
                    None => return Err(StoreError::ParentNotFound),
                    Some(&at) => Some(at),
                }
            }
            None => None,
        };
        let start_pos = anchor.map_or(0, |(_, p)| p + 1);
        if start_pos as u64 + blocks.len() as u64 > self.cfg.max_chain_len as u64 {
            return Err(StoreError::ChainTooLong);
        }

        // Walk-insert: follow the trie from the anchor, consuming
        // blocks; matching content = membership add (or §4 alias
        // duplicate), mismatch or chain end = fork/extend.
        let mut applied = 0u32;
        let mut duplicates = 0u32;
        // Current walk point: chain + next absolute position to fill,
        // or None before the first block when anchoring a root.
        let mut cursor: Option<(u32, u32)> = anchor.map(|(c, p)| (c, p + 1));
        let mut lineage = anchor.map(|(c, p)| self.lineage_at(c, p));
        for &(key, content) in blocks {
            let next_lineage = match lineage {
                None => lineage_root(content),
                Some(prev) => lineage_step(prev, content),
            };
            let landed = self.place_block(holder, id, key, content, next_lineage, &mut cursor)?;
            match landed {
                Placed::Applied => applied += 1,
                Placed::Duplicate => duplicates += 1,
            }
            lineage = Some(next_lineage);
        }
        Ok(StoreOutcome {
            applied,
            duplicates,
        })
    }

    /// Place one block at the cursor. Advances the cursor to the
    /// placed position.
    fn place_block(
        &mut self,
        holder: u32,
        id: HolderId,
        key: BlockKey,
        content: ContentHash,
        lineage: u64,
        cursor: &mut Option<(u32, u32)>,
    ) -> Result<Placed, StoreError> {
        // Resolve the target (chain, pos) for this block canonically.
        let target: (u32, u32) = match *cursor {
            None => {
                // Root anchor: canonical root chain by root lineage,
                // CONTENT-verified (a colliding lineage with a
                // different first content gets its own chain).
                let existing = self.roots.get(&lineage).and_then(|list| {
                    list.iter()
                        .copied()
                        .find(|&c| self.chains[c as usize].contents[0] == content)
                });
                match existing {
                    Some(c) => (c, 0),
                    None => {
                        let c = self.alloc_chain(ChainData {
                            parent: None,
                            base_pos: 0,
                            contents: vec![content],
                            start_lineage: lineage,
                            end_lineage: lineage,
                            spans: Vec::new(),
                            children: Vec::new(),
                        });
                        self.roots.entry(lineage).or_default().push(c);
                        (c, 0)
                    }
                }
            }
            Some((chain, pos)) => {
                let cd = &self.chains[chain as usize];
                if pos < cd.end_pos() {
                    if cd.content_at(pos) == content {
                        (chain, pos)
                    } else {
                        // Fork at pos-1 (or a new root when pos==0 is
                        // impossible here: cursor mid-chain implies
                        // pos > base 0 path came through parent).
                        match cd.child_at(pos - 1, content) {
                            Some(child) => {
                                let base = self.chains[child as usize].base_pos;
                                (child, base)
                            }
                            None => {
                                let c = self.new_child(chain, pos - 1, content, lineage);
                                (c, pos)
                            }
                        }
                    }
                } else {
                    // Past the tip: extend in place, or descend into
                    // an existing child continuation.
                    match cd.child_at(pos - 1, content) {
                        Some(child) => {
                            let base = self.chains[child as usize].base_pos;
                            (child, base)
                        }
                        None => {
                            let has_children_at_tip =
                                cd.children.iter().any(|&(p, _, _)| p + 1 == pos);
                            if has_children_at_tip {
                                // The tip already forks; a new
                                // continuation is another child (in-
                                // place extension would change the
                                // fork point's shape non-canonically).
                                let c = self.new_child(chain, pos - 1, content, lineage);
                                (c, pos)
                            } else {
                                let cd = &mut self.chains[chain as usize];
                                cd.contents.push(content);
                                cd.end_lineage = lineage;
                                (chain, pos)
                            }
                        }
                    }
                }
            }
        };
        let (chain, pos) = target;
        *cursor = Some((chain, pos + 1));

        // §4 semantics at the resolved position, PER HOLDER (chaos
        // finding: chain-global key canonicity diverged from the
        // model — aliasing is a per-holder concept, so the chain
        // stores no keys at all; per-holder key maps carry them).
        // Covered ⇒ this holder already holds the exact triple: a
        // plain duplicate, an alias (their other key), or a refused
        // move — all observably identical, all non-destructive.
        if self.chains[chain as usize].covered(pos, holder) {
            return Ok(Placed::Duplicate);
        }
        // Not covered: a move relocates the key first, then join.
        // Key BEFORE membership (chain GC scans key maps; the
        // reverse order makes GC see our own dying reference and
        // leak the chain — the audit's orphan rule caught this).
        // The CURSOR chain is pinned against GC for the gap between
        // this removal and the add just below: an in-batch move can
        // otherwise free the chain the batch is standing on, and the
        // next block would write into a freed slot (found by the
        // chaos fuzz).
        if let Some(old) = self.state_of(holder).keys.get(&key).copied() {
            self.state_of_mut(holder).keys.remove(&key);
            self.remove_membership_pinned(holder, old, Some(chain));
        }
        self.add_membership(holder, chain, pos);
        self.state_of_mut(holder).keys.insert(key, (chain, pos));
        let _ = id;
        Ok(Placed::Applied)
    }

    /// Read-only no-op predicate: true iff `store(id, parent, blocks)`
    /// would apply NOTHING — every block already sits at its resolved
    /// (position, content, lineage) for this holder (plain duplicate,
    /// §4 alias, or refused move: all observably identical no-ops).
    /// Exactly `store(..) == Ok(StoreOutcome { applied: 0, .. })`,
    /// and false wherever store would error — the caller falls through
    /// to the mutating path, which surfaces the error itself. This is
    /// the shared-lock fast path for duplicate-dominated multi-writer
    /// placement streams: the walk mirrors `place_block`'s resolution,
    /// and any step that would create, extend, or fork a chain is by
    /// definition not covered.
    pub fn covered(
        &self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> bool {
        self.dup_prefix(id, parent, blocks).1
    }

    /// One read-only walk answering two questions for the shared-lock
    /// apply paths: `(plain_dup_run, fully_covered)`.
    ///
    /// - `plain_dup_run`: length of the LEADING run of blocks that are
    ///   plain same-key duplicates — the holder's key map holds each
    ///   block's own key at exactly the resolved placement. A caller
    ///   may re-anchor a store at `blocks[run - 1].0` and apply only
    ///   the suffix: the skipped prefix is bit-for-bit already there,
    ///   key included, so the split store lands the identical state
    ///   (aliases deliberately do NOT extend the run — their key may
    ///   not be an anchor).
    /// - `fully_covered`: the [`Self::covered`] predicate — every
    ///   block (aliases included) is a no-op, i.e. `store` would
    ///   return `applied == 0`.
    pub fn dup_prefix(
        &self,
        id: HolderId,
        parent: Option<BlockKey>,
        blocks: &[(BlockKey, ContentHash)],
    ) -> (u32, bool) {
        if self.live(id).is_none() {
            return (0, false);
        }
        let holder = id.parts().0;
        if blocks.is_empty() {
            return (0, true);
        }
        let anchor: Option<(u32, u32)> = match parent {
            Some(parent_key) => match self.state_of(holder).keys.get(&parent_key) {
                None => return (0, false),
                Some(&at) => Some(at),
            },
            None => None,
        };
        let start_pos = anchor.map_or(0, |(_, p)| p + 1);
        if start_pos as u64 + blocks.len() as u64 > self.cfg.max_chain_len as u64 {
            return (0, false);
        }
        let mut cursor: Option<(u32, u32)> = anchor.map(|(c, p)| (c, p + 1));
        let mut lineage = anchor.map(|(c, p)| self.lineage_at(c, p));
        let mut run = 0u32;
        let mut run_live = true;
        for &(key, content) in blocks {
            let next_lineage = match lineage {
                None => lineage_root(content),
                Some(prev) => lineage_step(prev, content),
            };
            let target: (u32, u32) = match cursor {
                None => {
                    match self.roots.get(&next_lineage).and_then(|list| {
                        list.iter()
                            .copied()
                            .find(|&c| self.chains[c as usize].contents[0] == content)
                    }) {
                        Some(c) => (c, 0),
                        None => return (run, false),
                    }
                }
                Some((chain, pos)) => {
                    let cd = &self.chains[chain as usize];
                    if pos < cd.end_pos() && cd.content_at(pos) == content {
                        (chain, pos)
                    } else {
                        match cd.child_at(pos - 1, content) {
                            Some(child) => {
                                let base = self.chains[child as usize].base_pos;
                                (child, base)
                            }
                            None => return (run, false),
                        }
                    }
                }
            };
            let (chain, pos) = target;
            if !self.chains[chain as usize].covered(pos, holder) {
                return (run, false);
            }
            if run_live {
                if self.state_of(holder).keys.get(&key) == Some(&(chain, pos)) {
                    run += 1;
                } else {
                    run_live = false;
                }
            }
            cursor = Some((chain, pos + 1));
            lineage = Some(next_lineage);
        }
        (run, true)
    }

    /// Read-only: the absolute position at which `holder` holds `key`,
    /// or None. The digest fast path uses this to confirm a placement
    /// chain is fully present without receiving its contents. Two facts
    /// make that sound:
    /// - Contiguity: an inferred (placement-fed) holder's held
    ///   positions are a contiguous prefix per chain (stores add
    ///   contiguously, truncate cuts the tail, and mid-chain removes
    ///   only arrive via `Removed`, which pins the holder event-fed —
    ///   and event-fed holders reject every digest). A tip key at
    ///   position p therefore proves p+1 blocks are held.
    /// - Identity: the tip key is a chained seq hash, so it encodes the
    ///   ENTIRE prefix lineage — a key found at the expected position
    ///   cannot belong to a different chain that coincidentally aligns
    ///   (modulo 64-bit hash collision, the same assumption every seq
    ///   lookup makes). The parent/tip positions need no common-chain
    ///   check for this reason.
    pub fn position_of(&self, id: HolderId, key: BlockKey) -> Option<u32> {
        self.live(id)?;
        let holder = id.parts().0;
        self.state_of(holder).keys.get(&key).map(|&(_, pos)| pos)
    }

    pub fn remove(&mut self, id: HolderId, keys: &[BlockKey]) -> u32 {
        if self.live(id).is_none() {
            return 0;
        }
        let holder = id.parts().0;
        let mut removed = 0u32;
        for &key in keys {
            let Some(&(chain, pos)) = self.state_of(holder).keys.get(&key) else {
                continue;
            };
            self.state_of_mut(holder).keys.remove(&key);
            self.remove_membership(holder, (chain, pos));
            removed += 1;
        }
        removed
    }

    pub fn truncate_tail(&mut self, id: HolderId, keep: u64) -> u64 {
        if self.live(id).is_none() {
            return 0;
        }
        let holder = id.parts().0;
        let state = self.state_of(holder);
        if state.keys.len() as u64 <= keep {
            return 0;
        }
        let mut ordered: Vec<(u32, BlockKey)> =
            state.keys.iter().map(|(&k, &(_, p))| (p, k)).collect();
        ordered.sort_unstable();
        let mut dropped = 0u64;
        while self.state_of(holder).keys.len() as u64 > keep {
            let (_, key) = ordered.pop().expect("non-empty");
            let (chain, pos) = self.state_of(holder).keys[&key];
            self.state_of_mut(holder).keys.remove(&key);
            self.remove_membership(holder, (chain, pos));
            dropped += 1;
        }
        dropped
    }

    pub fn clear(&mut self, id: HolderId) {
        if self.live(id).is_none() {
            return;
        }
        let holder = id.parts().0;
        // Wipe the key map FIRST: chain GC scans key maps for
        // references, so clearing spans before keys made the GC see
        // the holder's own keys and bail — every retire leaked its
        // chains (caught by the churn-boundedness gate on the swap).
        let state = self.state_of_mut(holder);
        let key_count = state.keys.len() as u64;
        state.keys = FxHashMap::default();
        let chains: Vec<u32> = std::mem::take(&mut state.chains).into_iter().collect();
        self.holder_blocks_total -= key_count;
        for chain in chains {
            self.drop_holder_from_chain(holder, chain);
        }
    }

    // ---- reads ----

    pub fn overlap(
        &self,
        chain_query: &[ContentHash],
        scratch: &mut OverlapScratch,
        out: &mut Vec<Overlap>,
    ) {
        out.clear();
        if chain_query.is_empty() {
            return;
        }
        let root_lineage = lineage_root(chain_query[0]);
        let Some(root) = self.roots.get(&root_lineage).and_then(|list| {
            list.iter()
                .copied()
                .find(|&c| self.chains[c as usize].contents[0] == chain_query[0])
        }) else {
            return;
        };
        // Walk the trie along the query, collecting the matched path
        // as (chain, from, to) segments.
        let mut segments: Vec<(u32, u32, u32)> = Vec::new();
        let mut chain = root;
        let mut pos = 0u32;
        let limit = chain_query.len().min(u32::MAX as usize) as u32;
        'walk: while pos < limit {
            let cd = &self.chains[chain as usize];
            let seg_from = pos;
            while pos < limit && pos < cd.end_pos() {
                if cd.content_at(pos) != chain_query[pos as usize] {
                    if pos > seg_from {
                        segments.push((chain, seg_from, pos));
                    }
                    // Divergence: a sibling fork may continue.
                    if pos == seg_from {
                        // Diverged immediately at segment start:
                        // fork search happens at pos-1 on the SAME
                        // parent chain — handled below via child_at
                        // when entering; nothing matched here.
                    }
                    match (pos > 0)
                        .then(|| self.fork_of(chain, pos - 1, chain_query[pos as usize]))
                        .flatten()
                    {
                        Some(child) => {
                            chain = child;
                            continue 'walk;
                        }
                        None => break 'walk,
                    }
                }
                pos += 1;
            }
            if pos > seg_from {
                segments.push((chain, seg_from, pos));
            }
            if pos >= limit {
                break;
            }
            // Chain exhausted: follow the child fork at the tip.
            match self.fork_of(chain, pos - 1, chain_query[pos as usize]) {
                Some(child) => chain = child,
                None => break,
            }
        }
        if segments.is_empty() {
            return;
        }
        // Merge holder runs along the matched path (same active-set
        // logic as the flat walk, but over spans — few per segment).
        let active = &mut scratch.active;
        let next = &mut scratch.next;
        active.clear();
        next.clear();
        let mut depth = 0u32;
        let mut first = true;
        // Data pointer of the interned set the `active` buffer was copied
        // from, for the O(1) identity run-skip. Comparing against
        // `active.as_ptr()` (a distinct scratch allocation) can never
        // match, so the skip only works against the SOURCE set's pointer —
        // mirrors FlatTree's `active_src`.
        let mut active_src: Option<*const u32> = None;
        'runs: for &(c, from, to) in &segments {
            let cd = &self.chains[c as usize];
            let mut p = from;
            while p < to {
                let (run_holders, run_end) = match cd.span_index(p) {
                    Some(i) if cd.spans[i].start <= p => {
                        let s = &cd.spans[i];
                        (Some(&s.holders), (s.start + s.len).min(to))
                    }
                    Some(i) => (None, cd.spans[i].start.min(to)),
                    None => (None, to),
                };
                match run_holders {
                    None => {
                        // Uncovered positions: everyone drops here.
                        for &h in active.iter() {
                            push_answer3(&self.slots, h, depth, out);
                        }
                        active.clear();
                        break 'runs;
                    }
                    Some(set) => {
                        if first {
                            active.extend_from_slice(set);
                            active_src = Some(set.as_ptr());
                            first = false;
                        } else if active_src.is_some_and(|src| std::ptr::eq(src, set.as_ptr())) {
                            // O(1) identity skip: the same interned set, so
                            // every active holder continues unchanged.
                        } else if set.as_ref() == active.as_slice() {
                            // Content-equal after a hand-built subset:
                            // resync to the interned identity.
                            active_src = Some(set.as_ptr());
                        } else {
                            next.clear();
                            let mut ai = 0usize;
                            for &h in set.iter() {
                                while ai < active.len() && active[ai] < h {
                                    push_answer3(&self.slots, active[ai], depth, out);
                                    ai += 1;
                                }
                                if ai < active.len() && active[ai] == h {
                                    next.push(h);
                                    ai += 1;
                                }
                            }
                            while ai < active.len() {
                                push_answer3(&self.slots, active[ai], depth, out);
                                ai += 1;
                            }
                            std::mem::swap(active, next);
                            // next ⊆ set; equal lengths ⟹ equal sets.
                            active_src = (active.len() == set.len()).then(|| set.as_ptr());
                        }
                        if active.is_empty() {
                            break 'runs;
                        }
                        depth = run_end;
                    }
                }
                p = run_end;
            }
        }
        for &h in active.iter() {
            push_answer3(&self.slots, h, depth, out);
        }
    }

    pub fn enumerate(
        &self,
        id: HolderId,
    ) -> impl Iterator<Item = (u32, BlockKey, ContentHash)> + '_ {
        let rows = self.live(id).map(|state| {
            let mut v: Vec<(u32, BlockKey, ContentHash)> = state
                .keys
                .iter()
                .map(|(&k, &(c, p))| (p, k, self.chains[c as usize].content_at(p)))
                .collect();
            v.sort_unstable();
            v
        });
        rows.unwrap_or_default().into_iter()
    }

    pub fn stats(&self) -> Stats {
        let holders = self.slots.iter().filter(|s| s.state.is_some()).count() as u64;
        let chain_bytes: u64 = self
            .chains
            .iter()
            .map(|c| {
                16 * c.contents.len() as u64
                    + 24 * c.spans.len() as u64
                    + 16 * c.children.len() as u64
            })
            .sum();
        // Interned holder sets: the `Arc<[u32]>` data plus per-bucket
        // bookkeeping. Omitting this hid an interner leak from the
        // retire-churn memory gate (audit finding).
        let interner_bytes: u64 = self
            .interner
            .table
            .values()
            .map(|bucket| bucket.iter().map(|s| 16 + 4 * s.len() as u64).sum::<u64>() + 24)
            .sum();
        Stats {
            holders,
            holder_blocks: self.holder_blocks_total,
            distinct_entries: self.distinct_entries,
            bytes_estimate: chain_bytes
                + interner_bytes
                + self.holder_blocks_total * 24
                + holders * 128,
        }
    }

    // ---- internals ----

    fn state_of(&self, holder: u32) -> &HolderState3 {
        self.slots[holder as usize].state.as_ref().expect("live")
    }
    fn state_of_mut(&mut self, holder: u32) -> &mut HolderState3 {
        self.slots[holder as usize].state.as_mut().expect("live")
    }

    fn lineage_at(&self, chain: u32, pos: u32) -> u64 {
        // Roll from the chain's start (parents cached via
        // start_lineage; positions within a chain need a walk —
        // used on the STORE path only for the anchor, so bounded by
        // one chain's length).
        let cd = &self.chains[chain as usize];
        let mut l = cd.start_lineage;
        for p in (cd.base_pos + 1)..=pos {
            l = lineage_step(l, cd.content_at(p));
        }
        l
    }

    fn fork_of(&self, chain: u32, fork_pos: u32, content: ContentHash) -> Option<u32> {
        self.chains[chain as usize].child_at(fork_pos, content)
    }

    fn alloc_chain(&mut self, data: ChainData) -> u32 {
        if let Some(c) = self.free_chains.pop() {
            self.chains[c as usize] = data;
            c
        } else {
            self.chains.push(data);
            (self.chains.len() - 1) as u32
        }
    }

    fn new_child(&mut self, parent: u32, fork_pos: u32, content: ContentHash, lineage: u64) -> u32 {
        let c = self.alloc_chain(ChainData {
            parent: Some((parent, fork_pos)),
            base_pos: fork_pos + 1,
            contents: vec![content],
            start_lineage: lineage,
            end_lineage: lineage,
            spans: Vec::new(),
            children: Vec::new(),
        });
        let pd = &mut self.chains[parent as usize];
        let at = pd
            .children
            .binary_search_by_key(&(fork_pos, content), |&(p, cc, _)| (p, cc))
            .unwrap_err();
        pd.children.insert(at, (fork_pos, content, c));
        c
    }

    /// Add `holder` to the membership at (chain, pos), renormalizing
    /// runs canonically.
    fn add_membership(&mut self, holder: u32, chain: u32, pos: u32) {
        let mut m = self.take_position_membership(chain, pos);
        let was_empty = matches!(m, PosSet::Empty);
        match &mut m {
            PosSet::Empty => m = PosSet::Set(self.interner.intern(&[holder])),
            PosSet::Set(set) => {
                if let Err(at) = set.binary_search(&holder) {
                    let mut grown = Vec::with_capacity(set.len() + 1);
                    grown.extend_from_slice(&set[..at]);
                    grown.push(holder);
                    grown.extend_from_slice(&set[at..]);
                    let new = self.interner.intern(&grown);
                    let old = std::mem::replace(set, new);
                    self.interner.release(old);
                }
            }
        }
        self.put_position_membership(chain, pos, m);
        if was_empty {
            self.distinct_entries += 1;
        }
        self.holder_blocks_total += 1;
        self.state_of_mut(holder).chains.insert(chain);
    }

    fn remove_membership(&mut self, holder: u32, at: (u32, u32)) {
        self.remove_membership_pinned(holder, at, None);
    }

    fn remove_membership_pinned(&mut self, holder: u32, at: (u32, u32), pinned: Option<u32>) {
        let (chain, pos) = at;
        let mut m = self.take_position_membership(chain, pos);
        let mut now_empty = false;
        match &mut m {
            PosSet::Empty => {}
            PosSet::Set(set) => {
                if let Ok(idx) = set.binary_search(&holder) {
                    if set.len() == 1 {
                        let old = std::mem::replace(set, self.interner.intern(&[]));
                        self.interner.release(old);
                        now_empty = true;
                        m = PosSet::Empty;
                    } else {
                        let mut shrunk = Vec::with_capacity(set.len() - 1);
                        shrunk.extend_from_slice(&set[..idx]);
                        shrunk.extend_from_slice(&set[idx + 1..]);
                        let new = self.interner.intern(&shrunk);
                        let old = std::mem::replace(set, new);
                        self.interner.release(old);
                    }
                }
            }
        }
        self.put_position_membership(chain, pos, m);
        if now_empty {
            self.distinct_entries -= 1;
            self.maybe_gc_chain_pinned(chain, pinned);
        }
        self.holder_blocks_total -= 1;
        // holder->chains index pruned lazily in drop_holder_from_chain
        // and audit-tolerated as a superset.
    }

    /// Remove the holder from every span of one chain (clear path).
    fn drop_holder_from_chain(&mut self, holder: u32, chain: u32) {
        let cd = &self.chains[chain as usize];
        let positions: Vec<u32> = cd
            .spans
            .iter()
            .filter(|s| s.holders.binary_search(&holder).is_ok())
            .flat_map(|s| s.start..s.start + s.len)
            .collect();
        for pos in positions {
            let mut m = self.take_position_membership(chain, pos);
            if let PosSet::Set(set) = &mut m {
                if let Ok(idx) = set.binary_search(&holder) {
                    if set.len() == 1 {
                        let old = std::mem::replace(set, self.interner.intern(&[]));
                        self.interner.release(old);
                        m = PosSet::Empty;
                        self.distinct_entries -= 1;
                    } else {
                        let mut shrunk = Vec::with_capacity(set.len() - 1);
                        shrunk.extend_from_slice(&set[..idx]);
                        shrunk.extend_from_slice(&set[idx + 1..]);
                        let new = self.interner.intern(&shrunk);
                        let old = std::mem::replace(set, new);
                        self.interner.release(old);
                    }
                }
            }
            self.put_position_membership(chain, pos, m);
        }
        self.maybe_gc_chain(chain);
    }

    /// Extract the membership at one position, splitting its span so
    /// the position stands alone; `put` re-normalizes neighbors.
    fn take_position_membership(&mut self, chain: u32, pos: u32) -> PosSet {
        let cd = &mut self.chains[chain as usize];
        let Some(i) = cd.span_index(pos) else {
            return PosSet::Empty;
        };
        let s = cd.spans[i].clone();
        // Split into [start, pos), [pos, pos+1), (pos+1, end)
        let mut replacement = Vec::with_capacity(3);
        if pos > s.start {
            replacement.push(Span {
                start: s.start,
                len: pos - s.start,
                holders: s.holders.clone(),
            });
        }
        let taken = s.holders.clone();
        if s.start + s.len > pos + 1 {
            replacement.push(Span {
                start: pos + 1,
                len: s.start + s.len - pos - 1,
                holders: s.holders.clone(),
            });
        }
        // Replace the original span with its fragments; the original
        // ref is released (fragments and `taken` hold clones).
        let original = cd.spans[i].clone();
        cd.spans.splice(i..=i, replacement);
        self.interner.release(original.holders);
        PosSet::Set(taken)
    }

    /// Re-insert the membership at one position and renormalize with
    /// neighbors (canonical maximal runs).
    fn put_position_membership(&mut self, chain: u32, pos: u32, m: PosSet) {
        let cd = &mut self.chains[chain as usize];
        if let PosSet::Set(set) = m {
            if !set.is_empty() {
                let at = cd.spans.partition_point(|s| s.start < pos);
                cd.spans.insert(
                    at,
                    Span {
                        start: pos,
                        len: 1,
                        holders: set,
                    },
                );
            } else {
                self.interner.release(set);
            }
        }
        // Renormalize around pos: merge adjacent spans with identical
        // (pointer-equal) sets.
        let cd = &mut self.chains[chain as usize];
        let mut i = cd.spans.partition_point(|s| s.start + s.len < pos);
        i = i.saturating_sub(1);
        while i + 1 < cd.spans.len() {
            let (a, b) = (&cd.spans[i], &cd.spans[i + 1]);
            if a.start + a.len == b.start && std::sync::Arc::ptr_eq(&a.holders, &b.holders) {
                let merged_len = a.len + b.len;
                let dropped = cd.spans.remove(i + 1);
                cd.spans[i].len = merged_len;
                self.interner.release(dropped.holders);
            } else if b.start <= pos + 1 {
                i += 1;
            } else {
                break;
            }
        }
    }

    /// Free a chain when nothing references it.
    fn maybe_gc_chain(&mut self, chain: u32) {
        self.maybe_gc_chain_pinned(chain, None);
    }

    fn maybe_gc_chain_pinned(&mut self, chain: u32, pinned: Option<u32>) {
        if pinned == Some(chain) {
            return;
        }
        let cd = &self.chains[chain as usize];
        // Already freed (double-GC guard: reachable via a remove on
        // one path racing a parent-recursion free on another).
        if cd.contents.is_empty() {
            return;
        }
        if !cd.spans.is_empty() || !cd.children.is_empty() {
            return;
        }
        // Any key map still pointing here keeps it alive (out-of-
        // contract survivors); scan is O(holders) worst case but the
        // common in-contract path hits the fast bail above.
        // (Correct-but-slow; the audit keeps it honest.)
        for slot in &self.slots {
            if let Some(state) = &slot.state {
                if state.keys.values().any(|&(c, _)| c == chain) {
                    return;
                }
            }
        }
        let parent = cd.parent;
        let start_lineage = cd.start_lineage;
        let base = cd.base_pos;
        let first_content = cd.contents.first().copied();
        self.chains[chain as usize] = ChainData::default();
        self.free_chains.push(chain);
        match parent {
            None => {
                if let Some(list) = self.roots.get_mut(&start_lineage) {
                    list.retain(|&c| c != chain);
                    if list.is_empty() {
                        self.roots.remove(&start_lineage);
                    }
                }
            }
            Some((p, fork_pos)) => {
                let content = first_content.expect("chains are non-empty");
                let pd = &mut self.chains[p as usize];
                if let Ok(idx) = pd
                    .children
                    .binary_search_by_key(&(fork_pos, content), |&(fp, c, _)| (fp, c))
                {
                    pd.children.remove(idx);
                }
                let _ = base;
                self.maybe_gc_chain_pinned(p, pinned);
            }
        }
    }

    /// Internal structure sizes for leak hunting (soak diagnostics).
    pub fn debug_footprint(&self) -> String {
        let live_chains = self
            .chains
            .iter()
            .filter(|c| !c.contents.is_empty())
            .count();
        let span_total: usize = self.chains.iter().map(|c| c.spans.len()).sum();
        let span_cap: usize = self.chains.iter().map(|c| c.spans.capacity()).sum();
        let content_cap: usize = self.chains.iter().map(|c| c.contents.capacity()).sum();
        let child_total: usize = self.chains.iter().map(|c| c.children.len()).sum();
        let intern_sets: usize = self.interner.table.values().map(|v| v.len()).sum();
        let key_total: usize = self
            .slots
            .iter()
            .filter_map(|s| s.state.as_ref())
            .map(|s| s.keys.len())
            .sum();
        let key_cap: usize = self
            .slots
            .iter()
            .filter_map(|s| s.state.as_ref())
            .map(|s| s.keys.capacity())
            .sum();
        format!(
            "chains {} (vec {}, free {}) contents_cap {} spans {} (cap {}) children {} roots {} intern {} ({} sets) slots {} keys {} (cap {})",
            live_chains,
            self.chains.len(),
            self.free_chains.len(),
            content_cap,
            span_total,
            span_cap,
            child_total,
            self.roots.len(),
            self.interner.table.len(),
            intern_sets,
            self.slots.len(),
            key_total,
            key_cap,
        )
    }

    // ---- verification ----

    pub fn audit(&self) -> Result<(), String> {
        use std::collections::HashSet;
        // Slot/name/free coherence (same rules as the flat core).
        let mut live = 0u64;
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(state) = &slot.state {
                live += 1;
                if self.by_name.get(&state.name) != Some(&(idx as u32)) {
                    return Err(format!("by_name incoherent for {}", state.name));
                }
            }
        }
        if self.by_name.len() as u64 != live {
            return Err("by_name size mismatch".into());
        }
        let mut freed = HashSet::new();
        for &f in &self.free {
            if self.slots[f as usize].state.is_some() || !freed.insert(f) {
                return Err(format!("free-list bad entry {f}"));
            }
        }
        // Chains: shape, spans normalized, children sorted+linked.
        let mut live_chains = HashSet::new();
        // Reject a double-freed chain index (mirrors the `free` list
        // check above): a duplicate would let two logical chains later
        // alias one physical `ChainData` -> silently wrong overlaps.
        let mut freed_chains = HashSet::new();
        for &fc in &self.free_chains {
            if !freed_chains.insert(fc) {
                return Err(format!("free-chains duplicate entry {fc} (double free)"));
            }
        }
        for (ci, cd) in self.chains.iter().enumerate() {
            let ci = ci as u32;
            if freed_chains.contains(&ci) {
                if !cd.contents.is_empty() {
                    return Err(format!("freed chain {ci} not empty"));
                }
                continue;
            }
            if cd.contents.is_empty() {
                // Never-allocated default slots only exist as freed.
                return Err(format!("live chain {ci} is empty"));
            }
            live_chains.insert(ci);
            let mut prev_end = cd.base_pos;
            let mut prev_set: Option<&SetRef> = None;
            for s in &cd.spans {
                if s.len == 0 || s.holders.is_empty() {
                    return Err(format!("chain {ci} empty span"));
                }
                if s.start < prev_end && prev_set.is_some() {
                    return Err(format!("chain {ci} overlapping spans"));
                }
                if s.start < cd.base_pos || s.start + s.len > cd.end_pos() {
                    return Err(format!("chain {ci} span out of bounds"));
                }
                if let Some(ps) = prev_set {
                    if s.start == prev_end && std::sync::Arc::ptr_eq(ps, &s.holders) {
                        return Err(format!("chain {ci} non-maximal adjacent spans"));
                    }
                }
                let mut ph = None;
                for &h in s.holders.iter() {
                    if ph.is_some_and(|p: u32| p >= h) {
                        return Err(format!("chain {ci} unsorted span holders"));
                    }
                    ph = Some(h);
                    if self
                        .slots
                        .get(h as usize)
                        .and_then(|s| s.state.as_ref())
                        .is_none()
                    {
                        return Err(format!("chain {ci} span holds retired holder {h}"));
                    }
                }
                prev_end = s.start + s.len;
                prev_set = Some(&s.holders);
            }
            let mut prev_child = None;
            for &(fp, c, child) in &cd.children {
                if prev_child.is_some_and(|p| p >= (fp, c)) {
                    return Err(format!("chain {ci} unsorted children"));
                }
                prev_child = Some((fp, c));
                if fp < cd.base_pos || fp >= cd.end_pos() {
                    return Err(format!("chain {ci} child fork out of bounds"));
                }
                let ch = &self.chains[child as usize];
                if ch.parent != Some((ci, fp)) || ch.base_pos != fp + 1 {
                    return Err(format!("chain {ci} child {child} link broken"));
                }
            }
            match cd.parent {
                None => {
                    let listed = self
                        .roots
                        .get(&cd.start_lineage)
                        .is_some_and(|l| l.contains(&ci));
                    if cd.base_pos != 0 || !listed {
                        return Err(format!("root chain {ci} unindexed"));
                    }
                }
                Some((p, fp)) => {
                    let pd = &self.chains[p as usize];
                    if pd.child_at(fp, cd.contents[0]) != Some(ci) {
                        return Err(format!("chain {ci} not in parent's children"));
                    }
                }
            }
            // start/end lineage coherence.
            let mut l = cd.start_lineage;
            for i in 1..cd.contents.len() {
                l = lineage_step(l, cd.contents[i]);
            }
            if l != cd.end_lineage {
                return Err(format!("chain {ci} end_lineage stale"));
            }
        }
        for (&l, list) in &self.roots {
            if list.is_empty() {
                return Err(format!("roots entry {l:#x} empty list"));
            }
            let mut contents_seen = HashSet::new();
            for &c in list {
                let cd = &self.chains[c as usize];
                if cd.parent.is_some() || cd.start_lineage != l || freed_chains.contains(&c) {
                    return Err(format!("roots entry {l:#x} bad chain {c}"));
                }
                if !contents_seen.insert(cd.contents[0]) {
                    return Err(format!("roots entry {l:#x} duplicate first content"));
                }
            }
        }
        // Key maps <-> coverage; counters.
        let mut blocks = 0u64;
        for (idx, slot) in self.slots.iter().enumerate() {
            let Some(state) = &slot.state else { continue };
            blocks += state.keys.len() as u64;
            for (&k, &(c, p)) in &state.keys {
                if !live_chains.contains(&c) {
                    return Err(format!("holder {idx} key {k:#x} points at dead chain {c}"));
                }
                let cd = &self.chains[c as usize];
                if p < cd.base_pos || p >= cd.end_pos() {
                    return Err(format!("holder {idx} key {k:#x} position out of bounds"));
                }
                if !cd.covered(p, idx as u32) {
                    return Err(format!("holder {idx} key {k:#x} not covered by spans"));
                }
            }
        }
        if blocks != self.holder_blocks_total {
            return Err(format!(
                "holder_blocks_total {} != recount {blocks}",
                self.holder_blocks_total
            ));
        }
        // Reverse: every covered (position, holder) is backed by at
        // least one key in that holder's map (keys are per-holder;
        // build the reverse view once).
        let mut reverse: HashSet<(u32, u32, u32)> = HashSet::new();
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(state) = &slot.state {
                for &(c, p) in state.keys.values() {
                    reverse.insert((idx as u32, c, p));
                }
            }
        }
        for &ci in &live_chains {
            let cd = &self.chains[ci as usize];
            for s in &cd.spans {
                for pos in s.start..s.start + s.len {
                    for &h in s.holders.iter() {
                        if !reverse.contains(&(h, ci, pos)) {
                            return Err(format!(
                                "span coverage (chain {ci}, pos {pos}, holder {h}) has no key map entry"
                            ));
                        }
                    }
                }
            }
        }
        // GC coherence: a live chain must be referenced by spans,
        // children, or at least one key map (otherwise the GC missed
        // it — the leak class the churn gate caught).
        let mut key_ref_chains: HashSet<u32> = HashSet::new();
        for slot in &self.slots {
            if let Some(state) = &slot.state {
                for &(c, _) in state.keys.values() {
                    key_ref_chains.insert(c);
                }
            }
        }
        for &ci in &live_chains {
            let cd = &self.chains[ci as usize];
            if cd.spans.is_empty() && cd.children.is_empty() && !key_ref_chains.contains(&ci) {
                return Err(format!("live chain {ci} is orphaned (GC leak)"));
            }
        }
        let mut distinct = 0u64;
        for &ci in &live_chains {
            for s in &self.chains[ci as usize].spans {
                distinct += s.len as u64;
            }
        }
        if distinct != self.distinct_entries {
            return Err(format!(
                "distinct_entries {} != recount {distinct}",
                self.distinct_entries
            ));
        }
        // Interner: no orphaned sets. A set the table still holds but no
        // live span references (strong_count == 1 — only the table) is a
        // leak; the release path is meant to drop it once the last
        // membership reference goes (mirrors FlatTree::audit's detector).
        for bucket in self.interner.table.values() {
            for set in bucket {
                if std::sync::Arc::strong_count(set) == 1 {
                    return Err(format!(
                        "interner orphan: set {:?} held only by the table (leak)",
                        set.as_ref()
                    ));
                }
            }
        }
        Ok(())
    }
}

enum PosSet {
    Empty,
    Set(SetRef),
}

enum Placed {
    Applied,
    Duplicate,
}

fn push_answer3(slots: &[Slot3], holder: u32, depth: u32, out: &mut Vec<Overlap>) {
    if depth == 0 {
        return;
    }
    let slot = &slots[holder as usize];
    let Some(state) = slot.state.as_ref() else {
        return;
    };
    out.push(Overlap {
        holder: HolderId::assemble(holder, slot.generation),
        depth,
        total_blocks: state.keys.len() as u64,
    });
    let _ = &state.name;
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use crate::Config;

    /// The strengthened audit must reject a double-freed chain index — a
    /// duplicate in `free_chains` would let two logical chains later alias
    /// one physical `ChainData`, silently corrupting overlap answers.
    #[test]
    fn audit_rejects_a_double_freed_chain() {
        let mut t = RadixTree::new(Config::default());
        let h = t.create_holder("w1");
        t.store(h, None, &[(1, 10), (2, 20)]).expect("store");
        t.clear(h); // frees the chain(s)
        t.audit().expect("a cleared tree is valid");
        assert!(!t.free_chains.is_empty(), "clear frees the chain");

        let dup = t.free_chains[0];
        t.free_chains.push(dup);
        let err = t.audit().expect_err("a double free must be caught");
        assert!(err.contains("double free"), "unexpected audit error: {err}");
    }

    /// The strengthened audit must reject an interner set held only by the
    /// table (strong_count 1) — the leak class the release path can strand.
    #[test]
    fn audit_rejects_an_interner_orphan() {
        let mut t = RadixTree::new(Config::default());
        let h = t.create_holder("w1");
        t.store(h, None, &[(1, 10)]).expect("store");
        t.audit().expect("a fresh tree is valid");

        // An Arc no live span references — only the table holds it.
        let orphan: std::sync::Arc<[u32]> = std::sync::Arc::from([99u32].as_slice());
        t.interner.table.entry(0xDEAD).or_default().push(orphan);
        let err = t.audit().expect_err("an interner orphan must be caught");
        assert!(err.contains("orphan"), "unexpected audit error: {err}");
    }
}
