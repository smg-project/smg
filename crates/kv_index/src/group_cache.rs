//! Cache evidence reconstructed from received group events, not an engine snapshot.

use radix_tree::{Config, OverlapScratch, PrefixContext, RadixTree};
use rustc_hash::{FxHashMap, FxHashSet};

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
        /// Root-to-run-start ancestry, or delta after skipped_token_parent_hash.
        /// None needs native parent resolution.
        skipped_tokens: Option<Vec<u32>>,
        /// Authoritative remaining copies, parallel to keys. Empty is legacy evidence.
        /// Zero supplies context only and does not claim residency.
        copy_counts: Vec<u32>,
        /// Registered ancestor before skipped_tokens. Requires skipped_tokens.
        skipped_token_parent_hash: Option<CacheKey>,
    },
    Remove {
        group_id: u32,
        keys: Vec<CacheKey>,
        /// Authoritative remaining copies; empty removes one observed copy.
        copy_counts: Vec<u32>,
    },
    Clear,
    Invalid,
}

#[derive(Default)]
struct Group {
    metadata: Option<(String, usize)>,
    reported: FxHashMap<CacheKey, Membership>,
    unmatchable: bool,
    anchor_requirements: FxHashMap<CacheKey, CacheKey>,
}

#[derive(Clone, Copy)]
struct Membership {
    span: usize,
    copies: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct AnchoredParent {
    group_id: u32,
    native_parent: Option<CacheKey>,
    skipped_len: usize,
}

struct PendingContext {
    tokens: Vec<u32>,
    anchored: Vec<AnchoredParent>,
}

/// One worker's observation session. Live descendants keep their ancestry;
/// wholly evicted histories may be forgotten. Discontinuity discards the view.
pub struct GroupCache {
    index: RadixTree,
    groups: FxHashMap<u32, Group>,
    positions: FxHashMap<CacheKey, PrefixContext>,
    by_prefix: FxHashMap<PrefixContext, Vec<CacheKey>>,
    pending: FxHashMap<CacheKey, FxHashMap<CacheKey, PendingContext>>,
    parents_by_child: FxHashMap<CacheKey, Vec<CacheKey>>,
    // A marked snapshot can describe zero-copy ancestry in earlier chunks
    // than its live descendants. Keep contexts until the final chunk arrives.
    snapshot_changes: Option<FxHashSet<CacheKey>>,
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
            snapshot_changes: None,
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

    /// Discard cache evidence without forgetting which groups this worker needs.
    pub fn clear_evidence(&mut self) {
        let mut groups = std::mem::take(&mut self.groups);
        for group in groups.values_mut() {
            group.reported.clear();
            group.anchor_requirements.clear();
        }
        *self = Self {
            groups,
            ..Self::default()
        };
    }

    /// Begin an admitted multi-chunk snapshot. Context descriptions are applied
    /// immediately; reclamation waits for the resident membership chunks.
    pub fn begin_snapshot(&mut self) {
        self.snapshot_changes.get_or_insert_with(FxHashSet::default);
    }

    /// Finish an entire admitted snapshot and release its unneeded history.
    /// Interrupted or malformed snapshots must call clear_evidence/invalidate.
    pub fn finish_snapshot(&mut self) {
        if let Some(changed) = self.snapshot_changes.take() {
            self.reclaim(changed.into_iter().collect());
        }
    }

    /// Apply a batch whose ordering and continuity the caller has already admitted.
    pub fn apply_events(&mut self, events: &[GroupEvent]) -> Result<(), &'static str> {
        let positions_before = self.positions.len();
        let mut changed = Vec::new();
        for event in events {
            if let Err(reason) = self.apply(event, &mut changed) {
                self.clear_evidence();
                return Err(reason);
            }
        }
        if self.positions.len() > positions_before {
            // Identities only grow until reclamation. Clear resolved requirements
            // once per batch, rather than rescanning them for every new identity.
            for group in self.groups.values_mut() {
                group
                    .anchor_requirements
                    .retain(|_, anchor| !self.positions.contains_key(anchor));
            }
        }
        if let Some(snapshot_changes) = &mut self.snapshot_changes {
            snapshot_changes.extend(changed);
        } else {
            self.reclaim(changed);
        }
        Ok(())
    }

    pub fn has_groups(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Replayable logical state. Historical descriptions have zero copies;
    /// they establish native identities without reviving evicted membership.
    /// Sequence/rank ownership remains with the caller.
    #[expect(
        clippy::expect_used,
        reason = "registered contexts retain ancestry and nonempty aliases; anchored descriptions are created only after group metadata is set"
    )]
    pub fn snapshot_events(&self) -> Vec<GroupEvent> {
        let mut events = Vec::new();
        let mut groups: Vec<_> = self.groups.iter().collect();
        groups.sort_unstable_by_key(|&(id, _)| *id);
        for (&group_id, group) in &groups {
            if let Some((kind, window)) = &group.metadata {
                events.push(GroupEvent::Store {
                    group_id,
                    kind: kind.clone(),
                    window: *window,
                    keys: Vec::new(),
                    parent: None,
                    tokens: Vec::new(),
                    block_size: 0,
                    tokens_matchable: !group.unmatchable,
                    skipped_tokens: Some(Vec::new()),
                    skipped_token_parent_hash: None,
                    copy_counts: Vec::new(),
                });
            } else {
                // An observed removal still establishes a required unknown group.
                events.push(GroupEvent::Remove {
                    group_id,
                    keys: Vec::new(),
                    copy_counts: Vec::new(),
                });
            }
        }
        let Some((context_group, (kind, window))) = groups
            .iter()
            .find_map(|(id, group)| group.metadata.as_ref().map(|metadata| (**id, metadata)))
        else {
            return events;
        };
        let mut contexts: Vec<_> = self.by_prefix.iter().collect();
        contexts.sort_unstable_by_key(|&(context, _)| context.depth());
        for (&context, keys) in contexts {
            let mut parent = self.index.context_parent(context);
            while parent.is_some_and(|parent| !self.by_prefix.contains_key(&parent)) {
                parent = parent.and_then(|parent| self.index.context_parent(parent));
            }
            let contents = self
                .index
                .context_contents_since(context, parent)
                .expect("registered context has retained ancestry");
            let tokens: Vec<_> = contents.into_iter().map(|token| token as u32).collect();
            let parent_key = parent.map(|parent| {
                self.by_prefix[&parent]
                    .iter()
                    .min()
                    .expect("context alias")
                    .clone()
            });
            let mut keys: Vec<_> = keys.iter().collect();
            keys.sort_unstable();
            for key in keys {
                events.push(GroupEvent::Store {
                    group_id: context_group,
                    kind: kind.clone(),
                    window: *window,
                    keys: vec![key.clone()],
                    parent: parent_key.clone(),
                    block_size: tokens.len(),
                    tokens: tokens.clone(),
                    tokens_matchable: true,
                    skipped_tokens: parent.is_none().then(Vec::new),
                    skipped_token_parent_hash: None,
                    copy_counts: vec![0],
                });
            }
        }
        // Unresolved dense descriptions must survive a snapshot as pending
        // parent links. Do not invent an ordinal position for sparse reports.
        for (parent, children) in &self.pending {
            for (key, description) in children {
                if description.anchored.is_empty() {
                    events.push(GroupEvent::Store {
                        group_id: context_group,
                        kind: kind.clone(),
                        window: *window,
                        keys: vec![key.clone()],
                        parent: Some(parent.clone()),
                        block_size: description.tokens.len(),
                        tokens: description.tokens.clone(),
                        tokens_matchable: true,
                        skipped_tokens: None,
                        skipped_token_parent_hash: None,
                        copy_counts: vec![0],
                    });
                } else {
                    for proof in &description.anchored {
                        let (kind, window) = self.groups[&proof.group_id]
                            .metadata
                            .as_ref()
                            .expect("anchored group metadata");
                        events.push(GroupEvent::Store {
                            group_id: proof.group_id,
                            kind: kind.clone(),
                            window: *window,
                            keys: vec![key.clone()],
                            parent: proof.native_parent.clone(),
                            block_size: description.tokens.len() - proof.skipped_len,
                            tokens: description.tokens[proof.skipped_len..].to_vec(),
                            tokens_matchable: true,
                            skipped_tokens: Some(description.tokens[..proof.skipped_len].to_vec()),
                            skipped_token_parent_hash: Some(parent.clone()),
                            copy_counts: vec![0],
                        });
                    }
                }
            }
        }
        for (&group_id, group) in groups {
            let Some((kind, window)) = &group.metadata else {
                continue;
            };
            let mut memberships: Vec<_> = group.reported.iter().collect();
            memberships.sort_unstable_by_key(|&(key, _)| key);
            for (key, membership) in memberships {
                let missing_anchor = group
                    .anchor_requirements
                    .get(key)
                    .filter(|anchor| !self.positions.contains_key(*anchor))
                    .cloned();
                events.push(GroupEvent::Store {
                    group_id,
                    kind: kind.clone(),
                    window: *window,
                    keys: vec![key.clone()],
                    parent: None,
                    tokens: Vec::new(),
                    block_size: membership.span,
                    tokens_matchable: !group.unmatchable,
                    skipped_tokens: missing_anchor.as_ref().map(|_| Vec::new()),
                    skipped_token_parent_hash: missing_anchor,
                    copy_counts: vec![membership.copies],
                });
            }
        }
        events
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
                    group.anchor_requirements.clear();
                }
                changed.extend(self.positions.keys().cloned());
                self.pending.clear();
                self.parents_by_child.clear();
            }
            GroupEvent::Invalid => return Err("invalid group event"),
            GroupEvent::Remove {
                group_id,
                keys,
                copy_counts,
            } => {
                let group = self.groups.entry(*group_id).or_default();
                if !copy_counts.is_empty() && copy_counts.len() != keys.len() {
                    return Err("copy counts do not match group keys");
                }
                for (i, key) in keys.iter().enumerate() {
                    if let Some(membership) = group.reported.get_mut(key) {
                        membership.copies = copy_counts
                            .get(i)
                            .copied()
                            .unwrap_or_else(|| membership.copies.saturating_sub(1));
                        if membership.copies == 0 {
                            group.reported.remove(key);
                            group.anchor_requirements.remove(key);
                        }
                    }
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
                skipped_tokens,
                copy_counts,
                skipped_token_parent_hash,
            } => {
                let group = self.groups.entry(*group_id).or_default();
                if let Some((old_kind, old_window)) = &group.metadata {
                    if old_kind != kind || old_window != window {
                        return Err("group metadata changed within one stream");
                    }
                } else {
                    group.metadata = Some((kind.clone(), *window));
                }
                group.unmatchable |= !tokens_matchable;
                if (!keys.is_empty() && *block_size == 0) || keys.iter().any(Vec::is_empty) {
                    return Err("invalid group block identity or span");
                }
                if !copy_counts.is_empty() && copy_counts.len() != keys.len() {
                    return Err("copy counts do not match group keys");
                }
                if skipped_token_parent_hash
                    .as_ref()
                    .is_some_and(Vec::is_empty)
                    || (skipped_token_parent_hash.is_some() && skipped_tokens.is_none())
                {
                    return Err("skipped token anchor requires ancestry and a native identity");
                }
                for (i, key) in keys.iter().enumerate() {
                    if let Some(anchor) = skipped_token_parent_hash {
                        if !self.positions.contains_key(anchor) {
                            group
                                .anchor_requirements
                                .insert(key.clone(), anchor.clone());
                        }
                    }
                    let copies = copy_counts.get(i).copied().unwrap_or(1);
                    if copies > 0 {
                        group
                            .reported
                            .entry(key.clone())
                            .and_modify(|membership| {
                                membership.span = membership.span.max(*block_size);
                                membership.copies = if copy_counts.is_empty() {
                                    membership.copies.max(1)
                                } else {
                                    copies
                                };
                            })
                            .or_insert(Membership {
                                span: *block_size,
                                copies,
                            });
                    }
                    changed.push(key.clone());
                }
                // Sparse reports retain the entire token span but omit some
                // hashes. Their positions can only come from other reports.
                if *tokens_matchable
                    && !keys.is_empty()
                    && keys.len().checked_mul(*block_size) == Some(tokens.len())
                {
                    let mut parent = parent.clone();
                    for (i, (key, chunk)) in keys.iter().zip(tokens.chunks(*block_size)).enumerate()
                    {
                        let ancestry = (i == 0).then_some(skipped_tokens.as_deref()).flatten();
                        let anchor = (i == 0)
                            .then_some(skipped_token_parent_hash.as_ref())
                            .flatten()
                            .map(|anchor| (*group_id, anchor));
                        self.learn(
                            key.clone(),
                            parent,
                            chunk.to_vec(),
                            ancestry,
                            anchor,
                            changed,
                        )?;
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
        skipped_tokens: Option<&[u32]>,
        anchor: Option<(u32, &CacheKey)>,
        changed: &mut Vec<CacheKey>,
    ) -> Result<(), &'static str> {
        let (parent, tokens, ancestry, anchored) = if let Some((group_id, anchor)) = anchor {
            let delta = skipped_tokens.ok_or("anchor without skipped token delta")?;
            let proof = AnchoredParent {
                group_id,
                native_parent: parent,
                skipped_len: delta.len(),
            };
            let mut contents = delta.to_vec();
            contents.extend(tokens);
            (Some(anchor.clone()), contents, None, vec![proof])
        } else {
            (
                parent,
                tokens,
                skipped_tokens.map(<[u32]>::to_vec),
                Vec::new(),
            )
        };
        let mut ready = vec![(key, parent, tokens, ancestry, anchored)];
        while let Some((key, parent, mut tokens, ancestry, anchored)) = ready.pop() {
            let base = if let Some(mut ancestry) = ancestry {
                if let Some(known) = parent
                    .as_ref()
                    .and_then(|parent| self.positions.get(parent))
                {
                    if self.index.context_contents(*known).as_deref()
                        != Some(
                            ancestry
                                .iter()
                                .map(|&token| u64::from(token))
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                    {
                        return Err("skipped ancestry conflicts with known parent");
                    }
                }
                ancestry.extend(tokens);
                tokens = ancestry;
                None
            } else {
                match parent {
                    None => None,
                    Some(parent) => match self.positions.get(&parent).copied() {
                        Some(position) => Some(position),
                        None => {
                            if let Some(&known) =
                                self.positions.get(&key).filter(|_| anchored.is_empty())
                            {
                                let reported: Vec<_> =
                                    tokens.iter().map(|&token| u64::from(token)).collect();
                                if self.index.context_suffix(known, reported.len()).as_deref()
                                    != Some(reported.as_slice())
                                {
                                    return Err("conflicting tokens for known native hash");
                                }
                                continue;
                            }
                            // Different groups can describe the same endpoint via
                            // different parent/span pairs. Resolve either path.
                            let children = self.pending.entry(parent.clone()).or_default();
                            if children.get(&key).is_some_and(|old| old.tokens != tokens) {
                                return Err("conflicting tokens for one parent and child");
                            }
                            let description = children.entry(key.clone()).or_insert_with(|| {
                                self.parents_by_child.entry(key).or_default().push(parent);
                                PendingContext {
                                    tokens,
                                    anchored: Vec::new(),
                                }
                            });
                            for proof in anchored {
                                if !description.anchored.contains(&proof) {
                                    description.anchored.push(proof);
                                }
                            }
                            continue;
                        }
                    },
                }
            };
            for proof in &anchored {
                if let Some(&native_parent) = proof
                    .native_parent
                    .as_ref()
                    .and_then(|parent| self.positions.get(parent))
                {
                    let skipped: Vec<_> = tokens[..proof.skipped_len]
                        .iter()
                        .map(|&token| u64::from(token))
                        .collect();
                    if self
                        .index
                        .context_contents_since(native_parent, base)
                        .as_deref()
                        != Some(skipped.as_slice())
                    {
                        return Err("skipped delta conflicts with known native parent");
                    }
                }
            }
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
            self.detach_pending_parents(&key, changed, true);
            if let Some(children) = self.pending.remove(&key) {
                for (child, description) in children {
                    if let Some(parents) = self.parents_by_child.get_mut(&child) {
                        parents.retain(|parent| parent != &key);
                        if parents.is_empty() {
                            self.parents_by_child.remove(&child);
                        }
                    }
                    ready.push((
                        child,
                        Some(key.clone()),
                        description.tokens,
                        None,
                        description.anchored,
                    ));
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

    fn detach_pending_parents(
        &mut self,
        key: &CacheKey,
        changed: &mut Vec<CacheKey>,
        keep_anchored: bool,
    ) {
        if let Some(parents) = self.parents_by_child.remove(key) {
            let mut retained = Vec::new();
            for parent in parents {
                if keep_anchored
                    && self
                        .pending
                        .get(&parent)
                        .and_then(|children| children.get(key))
                        .is_some_and(|description| !description.anchored.is_empty())
                {
                    retained.push(parent);
                    continue;
                }
                if let Some(children) = self.pending.get_mut(&parent) {
                    children.remove(key);
                    if children.is_empty() {
                        self.pending.remove(&parent);
                    }
                }
                changed.push(parent);
            }
            if !retained.is_empty() {
                self.parents_by_child.insert(key.clone(), retained);
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
            self.detach_pending_parents(&key, &mut changed, false);
            for group in self.groups.values_mut() {
                if !group.reported.contains_key(&key) {
                    group.anchor_requirements.remove(&key);
                }
            }
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
                "full_attention"
                    | "sink_full_attention"
                    | "mla_attention"
                    | "sliding_window"
                    | "mamba"
            ) || (kind == "sliding_window" && *window == 0)
                || group.unmatchable
                || group.anchor_requirements.iter().any(|(key, anchor)| {
                    group.reported.contains_key(key) && !self.positions.contains_key(anchor)
                })
            {
                return None;
            }
            let mut intervals: Vec<_> = matching
                .iter()
                .filter_map(|(key, end)| {
                    let membership = group.reported.get(*key)?;
                    end.checked_sub(membership.span).map(|start| (start, *end))
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
            if matches!(
                kind.as_str(),
                "full_attention" | "sink_full_attention" | "mla_attention"
            ) {
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
