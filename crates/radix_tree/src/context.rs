//! Historical prefix identity, independent of holder membership.

use super::*;

/// A complete learned prefix in its originating tree. Contexts and the paths
/// they reference survive membership removal until explicitly released or the
/// tree is dropped. Released history may remain while its chain is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixContext {
    pub(super) chain: u32,
    pub(super) position: u32,
    pub(super) lineage: u64,
}

impl PrefixContext {
    pub fn depth(self) -> u32 {
        self.position + 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextError {
    EmptyContext,
    InvalidParent,
    ChainTooLong,
}

impl RadixTree {
    /// Append from a previously learned parent, without claiming residency.
    /// Only the final endpoint is registered; intermediate content is shared.
    /// Learning or retaining the same endpoint is idempotent, not reference
    /// counted. Callers sharing an endpoint must release it only after all of
    /// their uses end.
    pub fn learn_context(
        &mut self,
        parent: Option<PrefixContext>,
        contents: &[ContentHash],
    ) -> Result<PrefixContext, ContextError> {
        if contents.is_empty() {
            return Err(ContextError::EmptyContext);
        }
        if parent.is_some_and(|context| {
            self.chains
                .get(context.chain as usize)
                .and_then(|chain| chain.contexts.get(&context.position))
                .map(|(lineage, _)| lineage)
                != Some(&context.lineage)
        }) {
            return Err(ContextError::InvalidParent);
        }
        let start = parent.map_or(0, PrefixContext::depth);
        if contents.len() as u64 + u64::from(start) > u64::from(self.cfg.max_chain_len) {
            return Err(ContextError::ChainTooLong);
        }
        let mut cursor = parent.map(|context| (context.chain, context.depth()));
        let mut lineage = parent.map(|context| context.lineage);
        for &content in contents {
            let next = lineage.map_or_else(
                || lineage_root(content),
                |previous| lineage_step(previous, content),
            );
            self.place_content(content, next, &mut cursor);
            lineage = Some(next);
        }
        let (chain, end) = cursor.expect("nonempty contents");
        let endpoint = PrefixContext {
            chain,
            position: end - 1,
            lineage: lineage.expect("nonempty contents"),
        };
        let chain = &mut self.chains[chain as usize];
        if !chain
            .contexts
            .insert(endpoint.position, (endpoint.lineage, true))
            .is_some_and(|(_, pinned)| pinned)
        {
            chain.context_pins += 1;
        }
        Ok(endpoint)
    }

    /// Retain an existing endpoint again, for example when its evidence returns.
    pub fn retain_context(&mut self, context: PrefixContext) -> bool {
        self.set_context_retained(context, true)
    }

    /// Release an endpoint. Whole chains are reclaimed once no retained
    /// endpoint, holder membership or child needs them. Drain retired contexts
    /// after mutations to remove corresponding caller-owned identity records.
    pub fn release_context(&mut self, context: PrefixContext) -> bool {
        if !self.set_context_retained(context, false) {
            return false;
        }
        self.maybe_gc_chain(context.chain);
        true
    }

    fn set_context_retained(&mut self, context: PrefixContext, retained: bool) -> bool {
        let Some(chain) = self.chains.get_mut(context.chain as usize) else {
            return false;
        };
        let Some((lineage, pinned)) = chain.contexts.get_mut(&context.position) else {
            return false;
        };
        if *lineage != context.lineage {
            return false;
        }
        if *pinned != retained {
            if retained {
                chain.context_pins += 1;
            } else {
                chain.context_pins -= 1;
            }
            *pinned = retained;
        }
        true
    }

    /// Endpoints whose backing chains have been collected. Discard their
    /// caller-owned identity mappings before learning new endpoints: recreating
    /// an identical path can reuse the same canonical handle.
    pub fn drain_retired_contexts(&mut self) -> impl Iterator<Item = PrefixContext> + '_ {
        self.retired_contexts.drain(..)
    }

    /// Unique content units retained, including historical paths without holders.
    pub fn retained_contents(&self) -> usize {
        self.retained_contents
    }

    /// The nearest registered historical endpoint on this endpoint's path.
    pub fn context_parent(&self, context: PrefixContext) -> Option<PrefixContext> {
        if self
            .chains
            .get(context.chain as usize)?
            .contexts
            .get(&context.position)?
            .0
            != context.lineage
        {
            return None;
        }
        let mut chain_id = context.chain;
        let mut before = context.position;
        loop {
            let chain = &self.chains[chain_id as usize];
            if let Some((&position, &(lineage, _))) = chain.contexts.range(..before).next_back() {
                return Some(PrefixContext {
                    chain: chain_id,
                    position,
                    lineage,
                });
            }
            let (parent, position) = chain.parent?;
            chain_id = parent;
            before = position + 1;
        }
    }

    /// Complete content ancestry, including released history still held by a child.
    pub fn context_contents(&self, context: PrefixContext) -> Option<Vec<ContentHash>> {
        self.context_contents_since(context, None)
    }

    /// The last `length` content units, without copying the complete ancestry.
    pub fn context_suffix(
        &self,
        context: PrefixContext,
        length: usize,
    ) -> Option<Vec<ContentHash>> {
        if length > context.depth() as usize
            || self
                .chains
                .get(context.chain as usize)?
                .contexts
                .get(&context.position)?
                .0
                != context.lineage
        {
            return None;
        }
        let mut chain_id = context.chain;
        let mut end = context.depth();
        let mut remaining = length;
        let mut segments = Vec::new();
        while remaining > 0 {
            let chain = &self.chains[chain_id as usize];
            let stop = (end - chain.base_pos) as usize;
            let take = remaining.min(stop);
            segments.push(&chain.contents[stop - take..stop]);
            remaining -= take;
            if remaining > 0 {
                let (parent, position) = chain.parent?;
                chain_id = parent;
                end = position + 1;
            }
        }
        Some(segments.into_iter().rev().flatten().copied().collect())
    }

    /// Content after an ancestor. Reject stale or unrelated endpoint handles.
    pub fn context_contents_since(
        &self,
        context: PrefixContext,
        ancestor: Option<PrefixContext>,
    ) -> Option<Vec<ContentHash>> {
        let valid = |context: PrefixContext| {
            self.chains
                .get(context.chain as usize)
                .and_then(|chain| chain.contexts.get(&context.position))
                .is_some_and(|&(lineage, _)| lineage == context.lineage)
        };
        if !valid(context)
            || ancestor
                .is_some_and(|ancestor| !valid(ancestor) || ancestor.depth() > context.depth())
        {
            return None;
        }
        let mut chain_id = context.chain;
        let mut end = context.position + 1;
        let mut segments = Vec::new();
        loop {
            let chain = &self.chains[chain_id as usize];
            if let Some(ancestor) = ancestor.filter(|ancestor| ancestor.chain == chain_id) {
                if ancestor.position >= end {
                    return None;
                }
                segments.push(
                    &chain.contents[(ancestor.depth() - chain.base_pos) as usize
                        ..(end - chain.base_pos) as usize],
                );
                break;
            }
            segments.push(&chain.contents[..(end - chain.base_pos) as usize]);
            match chain.parent {
                Some((parent, position)) => {
                    chain_id = parent;
                    end = position + 1;
                }
                None if ancestor.is_none() => break,
                None => return None,
            }
        }
        Some(segments.into_iter().rev().flatten().copied().collect())
    }

    /// Return registered endpoints whose entire prefix matches, in depth order.
    /// Partial matches within a report do not establish that report's identity.
    pub fn matching_contexts(
        &self,
        query: &[ContentHash],
        scratch: &mut OverlapScratch,
        out: &mut Vec<PrefixContext>,
    ) {
        out.clear();
        self.match_path(query, &mut scratch.segments);
        for &(chain, from, to) in &scratch.segments {
            for (&position, &(lineage, _)) in self.chains[chain as usize].contexts.range(from..to) {
                out.push(PrefixContext {
                    chain,
                    position,
                    lineage,
                });
            }
        }
    }
}
