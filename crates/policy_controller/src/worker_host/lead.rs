//! The term a replica writes under.

/// The lead a host's adapters write under.
///
/// The store has one writer at a time, the replica that leads. An adapter
/// drops a fact received while following or after the term it was read
/// under has ended. A controller can take the term from its election.
/// A gateway with its own store can use [`SoleLead`].
pub trait Lead: Send + Sync + 'static {
    /// The term this replica leads under, or `None` while it follows.
    ///
    /// Called under a worker object's mutation lock, and under the store's
    /// locks too, at a verdict: it must not block, and must take no lock
    /// that is held while the registry or the store is written.
    fn term(&self) -> Option<u64>;
}

/// The lead of a host with no election: it leads from its start, under one
/// term, and never changes.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoleLead;

impl SoleLead {
    /// The one term such a host leads under.
    pub const TERM: u64 = 0;
}

impl Lead for SoleLead {
    fn term(&self) -> Option<u64> {
        Some(Self::TERM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_with_no_election_leads_under_its_one_term() {
        assert_eq!(SoleLead::TERM, 0);
        assert_eq!(SoleLead.term(), Some(0));
    }
}
