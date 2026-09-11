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

    /// Endpoints whose backing chains have been collected. These handles can no
    /// longer anchor children; the caller must discard their identity mappings.
    pub fn drain_retired_contexts(&mut self) -> impl Iterator<Item = PrefixContext> + '_ {
        self.retired_contexts.drain(..)
    }

    /// Unique content units retained, including historical paths without holders.
    pub fn retained_contents(&self) -> usize {
        self.retained_contents
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
