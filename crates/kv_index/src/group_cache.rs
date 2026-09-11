//! Cache evidence reconstructed from received group events, not an engine snapshot.

use radix_tree::{Config, OverlapScratch, PrefixContext, RadixTree};
use rustc_hash::FxHashMap;

pub type CacheKey = Vec<u8>;

#[derive(Clone, Debug)]
pub enum GroupEvent {
    Store {
        group_id: u32,
        kind: String,
        window: usize,
        keys: Vec<CacheKey>,
        parent: Option<CacheKey>,
        tokens: Vec<u32>,
        block_size: usize,
        tokens_matchable: bool,
    },
    Remove {
        group_id: u32,
        keys: Vec<CacheKey>,
    },
    Clear,
    Invalid,
}

#[derive(Default)]
struct Group {
    metadata: Option<(String, usize)>,
    reported: FxHashMap<CacheKey, usize>,
}

/// One worker's observation session. Live descendants keep their ancestry;
/// wholly evicted histories may be forgotten. Discontinuity discards the view.
pub struct GroupCache {
    index: RadixTree,
    groups: FxHashMap<u32, Group>,
    positions: FxHashMap<CacheKey, PrefixContext>,
    by_prefix: FxHashMap<PrefixContext, Vec<CacheKey>>,
    pending: FxHashMap<CacheKey, FxHashMap<CacheKey, Vec<u32>>>,
    parents_by_child: FxHashMap<CacheKey, Vec<CacheKey>>,
    last_sequence: Option<u64>,
}

impl Default for GroupCache {
    fn default() -> Self {
        Self {
            index: RadixTree::new(Config {
                // Token positions use u32; this is not a cache capacity policy.
                max_chain_len: u32::MAX,
            }),
            groups: FxHashMap::default(),
            positions: FxHashMap::default(),
            by_prefix: FxHashMap::default(),
            pending: FxHashMap::default(),
            parents_by_child: FxHashMap::default(),
            last_sequence: None,
        }
    }
}

/// Token identities prepared once across workers, before cache read guards.
pub struct GroupRequest {
    contents: Vec<u64>,
}

impl GroupRequest {
    pub fn new(tokens: &[u32]) -> Self {
        Self {
            contents: tokens.iter().map(|&token| u64::from(token)).collect(),
        }
    }
}

impl GroupCache {
    pub fn invalidate(&mut self) {
        *self = Self::default();
    }

    pub fn apply_batch(
        &mut self,
        sequence: u64,
        events: &[GroupEvent],
    ) -> Result<(), &'static str> {
        if let Some(previous) = self.last_sequence {
            if sequence <= previous {
                return Ok(());
            }
            if previous.checked_add(1) != Some(sequence) {
                self.invalidate();
            }
        }
        let mut changed = Vec::new();
        for event in events {
            if let Err(reason) = self.apply(event, &mut changed) {
                self.invalidate();
                return Err(reason);
            }
        }
        self.reclaim(changed);
        self.last_sequence = Some(sequence);
        Ok(())
    }

    fn apply(
        &mut self,
        event: &GroupEvent,
        changed: &mut Vec<CacheKey>,
    ) -> Result<(), &'static str> {
        match event {
            GroupEvent::Clear => {
                // Clearing blocks does not retract the groups already reported.
                for group in self.groups.values_mut() {
                    group.reported.clear();
                }
                changed.extend(self.positions.keys().cloned());
                self.pending.clear();
                self.parents_by_child.clear();
            }
            GroupEvent::Invalid => return Err("invalid group event"),
            GroupEvent::Remove { group_id, keys } => {
                let group = self.groups.entry(*group_id).or_default();
                for key in keys {
                    // A repeated STORE is not proof of another physical copy.
                    group.reported.remove(key);
                    changed.push(key.clone());
                }
            }
            GroupEvent::Store {
                group_id,
                kind,
                window,
                keys,
                parent,
                tokens,
                block_size,
                tokens_matchable,
            } => {
                if *block_size == 0 || keys.iter().any(Vec::is_empty) {
                    return Err("invalid group block identity or span");
                }
                let group = self.groups.entry(*group_id).or_default();
                if let Some((old_kind, old_window)) = &group.metadata {
                    if old_kind != kind || old_window != window {
                        return Err("group metadata changed within one stream");
                    }
                } else {
                    group.metadata = Some((kind.clone(), *window));
                }
                for key in keys {
                    group
                        .reported
                        .entry(key.clone())
                        .and_modify(|span| *span = (*span).max(*block_size))
                        .or_insert(*block_size);
                    changed.push(key.clone());
                }
                // Sparse reports retain the entire token span but omit some
                // hashes. Their positions can only come from other reports.
                if *tokens_matchable && keys.len().checked_mul(*block_size) == Some(tokens.len()) {
                    let mut parent = parent.clone();
                    for (key, chunk) in keys.iter().zip(tokens.chunks(*block_size)) {
                        self.learn(key.clone(), parent, chunk.to_vec(), changed)?;
                        parent = Some(key.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn learn(
        &mut self,
        key: CacheKey,
        parent: Option<CacheKey>,
        tokens: Vec<u32>,
        changed: &mut Vec<CacheKey>,
    ) -> Result<(), &'static str> {
        let mut ready = vec![(key, parent, tokens)];
        while let Some((key, parent, tokens)) = ready.pop() {
            let base = match parent {
                None => None,
                Some(parent) => match self.positions.get(&parent).copied() {
                    Some(position) => Some(position),
                    None => {
                        if self.positions.contains_key(&key) {
                            continue;
                        }
                        // Different groups can describe the same endpoint via
                        // different parent/span pairs. Resolve either path.
                        let children = self.pending.entry(parent.clone()).or_default();
                        if children.get(&key).is_some_and(|old| *old != tokens) {
                            return Err("conflicting tokens for one parent and child");
                        }
                        if children.insert(key.clone(), tokens).is_none() {
                            self.parents_by_child.entry(key).or_default().push(parent);
                        }
                        continue;
                    }
                },
            };
            let contents: Vec<_> = tokens.iter().map(|&token| u64::from(token)).collect();
            let position = self
                .index
                .learn_context(base, &contents)
                .map_err(|_| "invalid group prefix context")?;
            changed.push(key.clone());
            if let Some(old) = self.positions.get(&key) {
                if *old != position {
                    return Err("conflicting native hash identity");
                }
                continue;
            }
            self.positions.insert(key.clone(), position);
            self.by_prefix
                .entry(position)
                .or_default()
                .push(key.clone());
            // Once an identity is known, alternative unresolved descriptions
            // are unnecessary. Dropping one can release its pending ancestors.
            self.detach_pending_parents(&key, changed);
            if let Some(children) = self.pending.remove(&key) {
                for (child, tokens) in children {
                    if let Some(parents) = self.parents_by_child.get_mut(&child) {
                        parents.retain(|parent| parent != &key);
                        if parents.is_empty() {
                            self.parents_by_child.remove(&child);
                        }
                    }
                    ready.push((child, Some(key.clone()), tokens));
                }
            }
        }
        Ok(())
    }

    fn needed(&self, key: &CacheKey) -> bool {
        self.groups
            .values()
            .any(|group| group.reported.contains_key(key))
            || self.pending.contains_key(key)
    }

    fn detach_pending_parents(&mut self, key: &CacheKey, changed: &mut Vec<CacheKey>) {
        if let Some(parents) = self.parents_by_child.remove(key) {
            for parent in parents {
                if let Some(children) = self.pending.get_mut(&parent) {
                    children.remove(key);
                    if children.is_empty() {
                        self.pending.remove(&parent);
                    }
                }
                changed.push(parent);
            }
        }
    }

    fn reclaim(&mut self, mut changed: Vec<CacheKey>) {
        // Re-pin returning evidence before releasing anything: a sparse STORE
        // can revive an old endpoint while its last child is removed in this batch.
        for key in &changed {
            if self.needed(key) {
                if let Some(&context) = self.positions.get(key) {
                    self.index.retain_context(context);
                }
            }
        }
        while let Some(key) = changed.pop() {
            if self.needed(&key) {
                continue;
            }
            self.detach_pending_parents(&key, &mut changed);
            if let Some(&context) = self.positions.get(&key) {
                if !self.by_prefix[&context]
                    .iter()
                    .any(|alias| self.needed(alias))
                {
                    self.index.release_context(context);
                }
            }
        }
        // The core retains ancestors of live chains and collects whole chains.
        // Forget native identities only when their backing chain is gone.
        for context in self.index.drain_retired_contexts() {
            if let Some(keys) = self.by_prefix.remove(&context) {
                for key in keys {
                    self.positions.remove(&key);
                }
            }
        }
    }

    /// A common boundary supported by resolved reports for the observed groups.
    /// Unresolved histories may hide additional matches. Native scheduling may
    /// impose other constraints; this is neither a cache reservation nor its
    /// exact current reusable-token count.
    pub fn reusable_tokens(&self, request: &GroupRequest) -> Option<usize> {
        if self.groups.is_empty() {
            return None;
        }
        let mut limit = request.contents.len().saturating_sub(1);
        let mut endpoints = Vec::new();
        self.index.matching_contexts(
            &request.contents,
            &mut OverlapScratch::default(),
            &mut endpoints,
        );
        let matching: Vec<_> = endpoints
            .iter()
            .flat_map(|endpoint| {
                self.by_prefix[endpoint]
                    .iter()
                    .map(move |key| (key, endpoint.depth() as usize))
            })
            .collect();
        let mut views = Vec::new();
        let mut candidates = vec![0];
        for group in self.groups.values() {
            let (kind, window) = group.metadata.as_ref()?;
            if !matches!(
                kind.as_str(),
                "full_attention" | "mla_attention" | "sliding_window" | "mamba"
            ) || (kind == "sliding_window" && *window == 0)
            {
                return None;
            }
            let mut intervals: Vec<_> = matching
                .iter()
                .filter_map(|(key, end)| {
                    let span = group.reported.get(*key)?;
                    end.checked_sub(*span).map(|start| (start, *end))
                })
                .collect();
            if kind == "sliding_window" {
                intervals.sort_unstable();
            }
            candidates.extend(
                intervals
                    .iter()
                    .map(|&(_, end)| end)
                    .filter(|&end| end <= limit),
            );
            if matches!(kind.as_str(), "full_attention" | "mla_attention") {
                // Full attention needs continuous coverage from zero. Compute
                // its bound once, rather than again for every candidate.
                let mut covered = 0;
                for &(start, end) in &intervals {
                    // Endpoints arrive in depth order. A later span may
                    // bridge a gap; earlier skipped spans cannot extend it.
                    if start <= covered {
                        covered = end;
                    }
                }
                limit = limit.min(covered);
            } else {
                views.push((kind.as_str(), *window, intervals));
            }
        }
        candidates.retain(|&end| end <= limit);
        candidates.sort_unstable();
        candidates.dedup();
        // Each range/candidate cursor advances only once through its group.
        for (kind, window, intervals) in views {
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            if kind == "mamba" {
                ranges.extend(intervals.iter().map(|&(_, end)| (end, end)));
            } else {
                for (start, end) in intervals {
                    if let Some(last) = ranges.last_mut().filter(|last| start <= last.1) {
                        last.1 = last.1.max(end);
                    } else {
                        ranges.push((start, end));
                    }
                }
                let lookback = window.saturating_sub(1).max(1);
                ranges.retain_mut(|(start, end)| {
                    if *start == 0 {
                        return true;
                    }
                    if *end - *start < lookback {
                        return false;
                    }
                    *start += lookback;
                    true
                });
            }
            let mut cursor = 0;
            candidates.retain(|&position| {
                while cursor < ranges.len() && ranges[cursor].1 < position {
                    cursor += 1;
                }
                position == 0
                    || ranges
                        .get(cursor)
                        .is_some_and(|&(start, _)| start <= position)
            });
        }
        candidates.last().copied()
    }
}

#[cfg(test)]
#[path = "group_cache_tests.rs"]
mod tests;
