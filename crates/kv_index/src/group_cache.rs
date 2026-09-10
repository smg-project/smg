//! Cache evidence reconstructed from received group events, not an engine snapshot.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Position {
    end: usize,
    prefix: u128,
}

#[derive(Default)]
struct Group {
    metadata: Option<(String, usize)>,
    reported: FxHashMap<CacheKey, usize>,
}

/// One worker's observation session. Context survives removals: a later child
/// can still refer to an evicted parent. Discontinuity discards the view.
#[derive(Default)]
pub struct GroupCache {
    groups: FxHashMap<u32, Group>,
    positions: FxHashMap<CacheKey, Position>,
    by_prefix: FxHashMap<Position, Vec<CacheKey>>,
    pending: FxHashMap<CacheKey, FxHashMap<CacheKey, Vec<u32>>>,
    last_sequence: Option<u64>,
}

// Bound retained historical context as well as live membership. Overflow only
// discards routing evidence; it does not change engine cache or reject requests.
const MAX_ENTRIES: usize = 100_000;

fn extend_prefix(prefix: u128, token: u32) -> u128 {
    let mut bytes = [0u8; 20];
    bytes[..16].copy_from_slice(&prefix.to_le_bytes());
    bytes[16..].copy_from_slice(&token.to_le_bytes());
    xxhash_rust::xxh3::xxh3_128(&bytes)
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
        for event in events {
            if let Err(reason) = self.apply(event) {
                self.invalidate();
                return Err(reason);
            }
        }
        self.last_sequence = Some(sequence);
        Ok(())
    }

    fn apply(&mut self, event: &GroupEvent) -> Result<(), &'static str> {
        match event {
            GroupEvent::Clear => {
                // Clearing blocks does not retract the groups already reported.
                for group in self.groups.values_mut() {
                    group.reported.clear();
                }
            }
            GroupEvent::Invalid => return Err("invalid group event"),
            GroupEvent::Remove { group_id, keys } => {
                let group = self.groups.entry(*group_id).or_default();
                for key in keys {
                    // A repeated STORE is not proof of another physical copy.
                    group.reported.remove(key);
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
                }
                // Sparse reports retain the entire token span but omit some
                // hashes. Their positions can only come from other reports.
                if *tokens_matchable && keys.len().checked_mul(*block_size) == Some(tokens.len()) {
                    let mut parent = parent.clone();
                    for (key, chunk) in keys.iter().zip(tokens.chunks(*block_size)) {
                        self.learn(key.clone(), parent, chunk.to_vec())?;
                        parent = Some(key.clone());
                    }
                }
            }
        }
        let pending = self.pending.values().map(FxHashMap::len).sum::<usize>();
        let reported = self
            .groups
            .values()
            .map(|g| g.reported.len())
            .sum::<usize>();
        if self.positions.len() + pending > MAX_ENTRIES || reported > MAX_ENTRIES {
            return Err("group event index capacity reached");
        }
        Ok(())
    }

    fn learn(
        &mut self,
        key: CacheKey,
        parent: Option<CacheKey>,
        tokens: Vec<u32>,
    ) -> Result<(), &'static str> {
        let mut ready = vec![(key, parent, tokens)];
        while let Some((key, parent, tokens)) = ready.pop() {
            let base = match parent {
                None => Position { end: 0, prefix: 0 },
                Some(parent) => match self.positions.get(&parent).copied() {
                    Some(position) => position,
                    None => {
                        if self.positions.contains_key(&key) {
                            continue;
                        }
                        // Different groups can describe the same endpoint via
                        // different parent/span pairs. Resolve either path.
                        let children = self.pending.entry(parent).or_default();
                        if children.get(&key).is_some_and(|old| *old != tokens) {
                            return Err("conflicting tokens for one parent and child");
                        }
                        children.insert(key, tokens);
                        continue;
                    }
                },
            };
            let position = Position {
                end: base
                    .end
                    .checked_add(tokens.len())
                    .ok_or("prefix length overflow")?,
                prefix: tokens
                    .iter()
                    .fold(base.prefix, |hash, &t| extend_prefix(hash, t)),
            };
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
            if let Some(children) = self.pending.remove(&key) {
                ready.extend(
                    children
                        .into_iter()
                        .map(|(child, tokens)| (child, Some(key.clone()), tokens)),
                );
            }
        }
        Ok(())
    }

    /// A common boundary supported by resolved reports for the observed groups.
    /// Unresolved histories may hide additional matches. Native scheduling may
    /// impose other constraints; this is neither a cache reservation nor its
    /// exact current reusable-token count.
    pub fn reusable_tokens(&self, tokens: &[u32]) -> Option<usize> {
        if self.groups.is_empty() {
            return None;
        }
        let limit = tokens.len().saturating_sub(1);
        let mut prefix = 0;
        let mut matching = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            prefix = extend_prefix(prefix, token);
            if let Some(keys) = self.by_prefix.get(&Position { end: i + 1, prefix }) {
                matching.extend(keys.iter().map(|key| (key, i + 1)));
            }
        }
        let mut views = Vec::new();
        let mut candidates = vec![0];
        for group in self.groups.values() {
            let (kind, window) = group.metadata.as_ref()?;
            if !matches!(kind.as_str(), "full_attention" | "sliding_window" | "mamba")
                || (kind == "sliding_window" && *window == 0)
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
            intervals.sort_unstable();
            candidates.extend(
                intervals
                    .iter()
                    .map(|&(_, end)| end)
                    .filter(|&end| end <= limit),
            );
            views.push((kind.as_str(), *window, intervals));
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates.into_iter().rev().find(|&position| {
            position == 0
                || views.iter().all(|(kind, window, intervals)| {
                    if *kind == "mamba" {
                        return intervals.iter().any(|&(_, end)| end == position);
                    }
                    let mut covered = if *kind == "sliding_window" {
                        position.saturating_sub(window.saturating_sub(1).max(1))
                    } else {
                        0
                    };
                    for &(start, end) in intervals {
                        if start > covered {
                            break;
                        }
                        covered = covered.max(end);
                        if covered >= position {
                            return true;
                        }
                    }
                    false
                })
        })
    }
}

#[cfg(test)]
#[path = "group_cache_tests.rs"]
mod tests;
