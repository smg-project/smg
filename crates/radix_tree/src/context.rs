//! Historical prefix identity, independent of holder membership.

use super::*;

/// A complete learned prefix in its originating tree. Contexts and the paths
/// they reference survive membership removal and remain valid until tree drop.
/// Callers retaining contexts must bound the tree's history separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrefixContext {
    chain: u32,
    position: u32,
    lineage: u64,
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
        self.chains[chain as usize]
            .contexts
            .insert(endpoint.position, endpoint.lineage);
        Ok(endpoint)
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
            for (&position, &lineage) in self.chains[chain as usize].contexts.range(from..to) {
                out.push(PrefixContext {
                    chain,
                    position,
                    lineage,
                });
            }
        }
    }
}
