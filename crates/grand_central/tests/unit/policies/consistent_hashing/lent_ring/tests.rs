use super::{Head, LentRing};

/// A ring for the crate's cases, of the addresses it holds in the order
/// stated: a walk starts as many workers along that order as its position
/// says, and goes round once.
#[derive(Debug)]
pub(crate) struct InOrder(pub(crate) Vec<String>);

impl LentRing for InOrder {
    fn position(&self, key: &str) -> u64 {
        crate::engine::fingerprint(key.as_bytes())
    }

    // Another recipe than a key's text has: the fingerprint of the head's
    // bytes turned half round, so a case that asks the one place for the
    // other reads another number.
    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        let bytes = match head {
            Head::Tokens(ids) => ids.iter().flat_map(|id| id.to_ne_bytes()).collect(),
            Head::Text(text) => text.as_bytes().to_vec(),
        };
        Some(crate::engine::fingerprint(&bytes).rotate_left(32))
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let (before, behind) = self.0.split_at(position as usize % self.0.len().max(1));
        let _ = behind.iter().chain(before).any(|address| take(address));
    }
}
