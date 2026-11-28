//! A request's budget, as the engine over worker lists shows it to a stage.
//!
//! A host in process states no budget by `0`: the decision of such a
//! request is unbounded ([`Request::facts`](crate::engine::Request::facts)).
//! A stage reads the number as the time left
//! ([`RequestFacts::budget_micros`](crate::engine::view::RequestFacts::budget_micros)),
//! as a controller's frame needs it read, whose `0` says none is left: a
//! plugin's stage passes such a request by and counts a failure of its own.
//! So this engine shows a stage the widest value where the host stated
//! none. It does so where it takes the request its maker makes
//! ([`Loaded::made`](super::Loaded::made)), which every entry
//! that makes one calls: a single's and a pair's, with a lease and without.
//!
//! The two calls a host makes with no engine above its lists turn a `0`
//! the same way, in the host's own request and beside the instant they
//! write there ([`PairLists::decide`](super::pair::PairLists::decide),
//! [`EncodeLists::decide`](super::encode::EncodeLists::decide)).

/// `stated`, the budget a host in process states for its request, as a
/// stage is shown it: the widest value where the host stated none, and
/// `stated` otherwise. Only a stage and the floor of a shadow read it: no
/// lease and no record keeps a request's budget.
#[inline]
pub(super) const fn shown(stated: u32) -> u32 {
    match stated {
        0 => u32::MAX,
        stated => stated,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/budget/tests.rs"]
mod tests;
