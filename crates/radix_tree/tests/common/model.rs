//! The trivially-correct model of the matching/convergence contract.
//!
//! Representationally complete: every block carries its OWN literal
//! lineage (the content prefix it was stored under, including
//! itself), exactly mirroring the subject's registry of
//! (position, content, lineage-fingerprint) — with the fingerprint
//! replaced by the literal vector, so no hashing and no collisions.
//! This makes the model defined on ARBITRARY inputs (aliases, moves,
//! twin keys), which lets the chaos fuzz use it as referee too.
//!
//! §4 alias semantics, mirrored exactly:
//! - same key, same (pos, content, lineage): duplicate;
//! - different key, same (pos, content, lineage) already held: the
//!   new key is a duplicate and is NEVER registered — and when that
//!   key was previously registered elsewhere (a refused MOVE), its
//!   old placement stays intact;
//! - same key, different placement: MOVE (old placement removed).

use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use super::Op;

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockRec {
    pos: u32,
    content: u64,
    /// Literal lineage: contents at positions 0..=pos as stored.
    lineage: Rc<Vec<u64>>,
}

#[derive(Debug, Clone, Default)]
struct Holder {
    registry: HashMap<u64, BlockRec>,
}

/// Model-level store outcome, mirroring the §4 error surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreResult {
    Applied { applied: u32, duplicates: u32 },
    ParentNotFound,
    ChainTooLong,
}

#[derive(Debug, Clone, Default)]
pub struct Model {
    holders: Vec<Holder>,
    max_chain_len: u32,
}

impl Model {
    pub fn new(holder_count: usize) -> Self {
        Self::with_max(holder_count, 65_536)
    }

    /// Mirror a subject configured with a specific max_chain_len.
    pub fn with_max(holder_count: usize, max_chain_len: u32) -> Self {
        Self {
            holders: vec![Holder::default(); holder_count],
            max_chain_len,
        }
    }

    pub fn apply(&mut self, op: &Op) -> Option<StoreResult> {
        match op {
            Op::Store {
                holder,
                parent,
                blocks,
            } => Some(self.store(*holder, *parent, blocks)),
            Op::Remove { holder, keys } => {
                self.remove(*holder, keys);
                None
            }
            Op::Clear { holder } => {
                self.clear(*holder);
                None
            }
        }
    }

    pub fn store(
        &mut self,
        holder: usize,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
    ) -> StoreResult {
        if blocks.is_empty() {
            return StoreResult::Applied {
                applied: 0,
                duplicates: 0,
            };
        }
        let h = &mut self.holders[holder];
        let (start_pos, mut prefix) = match parent {
            None => (0u32, Vec::new()),
            Some(parent_key) => match h.registry.get(&parent_key) {
                None => return StoreResult::ParentNotFound,
                Some(rec) => (rec.pos + 1, rec.lineage.as_ref().clone()),
            },
        };
        if start_pos as u64 + blocks.len() as u64 > self.max_chain_len as u64 {
            return StoreResult::ChainTooLong;
        }
        let mut applied = 0u32;
        let mut duplicates = 0u32;
        for (i, &(key, content)) in blocks.iter().enumerate() {
            let pos = start_pos + i as u32;
            prefix.push(content);
            let candidate = BlockRec {
                pos,
                content,
                lineage: Rc::new(prefix.clone()),
            };
            if h.registry.get(&key) == Some(&candidate) {
                duplicates += 1;
                continue;
            }
            // Alias: any OTHER key already at this exact triple?
            let aliased = h
                .registry
                .iter()
                .any(|(&k, rec)| k != key && *rec == candidate);
            if aliased {
                // Duplicate in every observable sense; a would-be MOVE
                // is refused non-destructively (old placement stays).
                duplicates += 1;
                continue;
            }
            // MOVE or fresh insert: (re)register.
            h.registry.insert(key, candidate);
            applied += 1;
        }
        StoreResult::Applied {
            applied,
            duplicates,
        }
    }

    /// Read-only mirror of [`Self::store`]: true iff the store would
    /// return `Applied { applied: 0, .. }` (every block an exact-triple
    /// duplicate or alias), false where it would error. Referees the
    /// subject's shared-lock duplicate fast path.
    pub fn covered(&self, holder: usize, parent: Option<u64>, blocks: &[(u64, u64)]) -> bool {
        self.dup_prefix(holder, parent, blocks).1
    }

    /// Read-only mirror of the subjects' `dup_prefix`: (leading run of
    /// plain same-key duplicates, store-would-apply-nothing).
    pub fn dup_prefix(
        &self,
        holder: usize,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
    ) -> (u32, bool) {
        if blocks.is_empty() {
            return (0, true);
        }
        let h = &self.holders[holder];
        let (start_pos, mut prefix) = match parent {
            None => (0u32, Vec::new()),
            Some(parent_key) => match h.registry.get(&parent_key) {
                None => return (0, false),
                Some(rec) => (rec.pos + 1, rec.lineage.as_ref().clone()),
            },
        };
        if start_pos as u64 + blocks.len() as u64 > self.max_chain_len as u64 {
            return (0, false);
        }
        let mut run = 0u32;
        let mut run_live = true;
        for (i, &(key, content)) in blocks.iter().enumerate() {
            let pos = start_pos + i as u32;
            prefix.push(content);
            let candidate = BlockRec {
                pos,
                content,
                lineage: Rc::new(prefix.clone()),
            };
            let plain = h.registry.get(&key) == Some(&candidate);
            let held = plain
                || h.registry
                    .iter()
                    .any(|(&k, rec)| k != key && *rec == candidate);
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

    /// Read-only mirror of the subjects' `position_of`.
    pub fn position_of(&self, holder: usize, key: u64) -> Option<u32> {
        self.holders[holder].registry.get(&key).map(|r| r.pos)
    }

    pub fn remove(&mut self, holder: usize, keys: &[u64]) -> u32 {
        let h = &mut self.holders[holder];
        let mut removed = 0;
        for key in keys {
            if h.registry.remove(key).is_some() {
                removed += 1;
            }
        }
        removed
    }

    pub fn clear(&mut self, holder: usize) {
        self.holders[holder] = Holder::default();
    }

    /// Forest-wide §4 truncate: strictly decreasing positions, ties
    /// by key, until `keep` remain.
    pub fn truncate_tail(&mut self, holder: usize, keep: u64) -> u64 {
        let h = &mut self.holders[holder];
        if h.registry.len() as u64 <= keep {
            return 0;
        }
        let mut ordered: Vec<(u32, u64)> = h.registry.iter().map(|(&k, r)| (r.pos, k)).collect();
        ordered.sort_unstable();
        let mut dropped = 0u64;
        while h.registry.len() as u64 > keep {
            let (_, key) = ordered.pop().expect("non-empty");
            h.registry.remove(&key);
            dropped += 1;
        }
        dropped
    }

    /// §6 depth: largest d such that for every p < d the holder has a
    /// block at position p whose content is query[p] and whose
    /// LITERAL lineage equals query[0..=p].
    pub fn depth(&self, holder: usize, query: &[u64]) -> u32 {
        let mut d = 0u32;
        'outer: for (p, &q) in query.iter().enumerate() {
            for rec in self.holders[holder].registry.values() {
                if rec.pos == p as u32
                    && rec.content == q
                    && rec.lineage.len() == p + 1
                    && rec.lineage[..] == query[..=p]
                {
                    d = p as u32 + 1;
                    continue 'outer;
                }
            }
            break;
        }
        d
    }

    /// Full holder->depth map (§10.1: map equality, depth-0 absent).
    pub fn overlap(&self, query: &[u64]) -> BTreeMap<usize, u32> {
        let mut out = BTreeMap::new();
        for holder in 0..self.holders.len() {
            let d = self.depth(holder, query);
            if d > 0 {
                out.insert(holder, d);
            }
        }
        out
    }

    pub fn holder_blocks(&self, holder: usize) -> u64 {
        self.holders[holder].registry.len() as u64
    }

    /// Position-ordered (pos, key, content) — the model's `enumerate`.
    pub fn enumerate(&self, holder: usize) -> Vec<(u32, u64, u64)> {
        let mut v: Vec<(u32, u64, u64)> = self.holders[holder]
            .registry
            .iter()
            .map(|(&k, r)| (r.pos, k, r.content))
            .collect();
        v.sort_unstable();
        v
    }

    /// Distinct (position, content, lineage) across all holders — the
    /// model's stats().distinct_entries.
    pub fn distinct_entries(&self) -> u64 {
        let mut set = std::collections::HashSet::new();
        for h in &self.holders {
            for rec in h.registry.values() {
                set.insert((rec.pos, rec.content, rec.lineage.clone()));
            }
        }
        set.len() as u64
    }

    /// Census discriminator: content match at pos under ANY lineage.
    pub fn holds_content_at(&self, holder: usize, pos: u32, q: u64) -> bool {
        self.holders[holder]
            .registry
            .values()
            .any(|r| r.pos == pos && r.content == q)
    }

    /// Census discriminator: lineage-true block at pos for this query
    /// (model depth stopped earlier only because of a gap).
    pub fn holds_lineage_true_at(&self, holder: usize, pos: u32, query: &[u64]) -> bool {
        self.holders[holder].registry.values().any(|r| {
            r.pos == pos
                && r.content == query[pos as usize]
                && r.lineage.len() == pos as usize + 1
                && r.lineage[..] == query[..=pos as usize]
        })
    }

    /// Keep the compiler honest about BTreeMap being intentional.
    #[allow(dead_code)]
    fn _uses(_: &BTreeMap<usize, u32>) {}
}
