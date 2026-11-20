//! The signal sink: where a source writes the values a report states by
//! name, one value for one rank's row.
//!
//! A value stands beside its rank's load sample, under the same lock, and
//! reaches a decision in a publication, as the row does: no stage is told a
//! value as it arrives. The store keeps cells for the names the document in
//! force declares ([`AppliedSpec::signal_columns`]) and for no other.
//!
//! A source writes a report's row first ([`LoadSink`]), then its values. A
//! publication cut between the two shows the new row beside the values of
//! the report before, for that one publication, and a stage relies on no
//! more: a value may ride one record in ten. A row the load sink passed
//! over takes its values with it: its source writes none of them.
//!
//! A report that leaves a name out removes nothing. The value on file
//! stands until the signal's maximum age ends it, or until its rank's load
//! sample goes: a worker that leaves and joins again starts with no value.
//!
//! [`AppliedSpec::signal_columns`]: crate::AppliedSpec::signal_columns
//! [`LoadSink`]: super::LoadSink

use std::{
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Instant,
};

use crate::{signals::SignalColumn, view::WorkerId};

#[cfg(test)]
mod tests;

/// Where a source writes a signal's value: the store's door to the cells
/// of the document in force.
pub trait SignalSink: Send + Sync {
    /// The declared signals of the document in force, in column order:
    /// what a report's conversion walks, so it offers declared names alone.
    fn signal_columns(&self) -> Arc<[SignalColumn]>;

    /// One value by its name, for one rank's row, sampled at `sampled_at`
    /// on the host's clock: the value's own age where its report states
    /// one, else the age of the record that carried it, else its receipt.
    ///
    /// The value is taken where the document in force declares the name,
    /// `unit` is the declared one, the value is finite, the worker is a
    /// member and no newer sample of the name is on file for the rank. Of
    /// two sources of one signal the newer sample stands, whichever arrives
    /// last. Every other write is dropped and counted by what became of it.
    /// The name, the unit and the value are read before any lock is taken:
    /// a value under a name no document declares costs one lookup.
    fn signal_value(
        &self,
        id: WorkerId,
        rank: u16,
        name: &str,
        unit: &str,
        value: f64,
        sampled_at: Instant,
    ) -> Taken;
}

/// What became of a write to the signal sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Taken {
    /// The value is on file and is shown with the next publication.
    Written,
    /// The document in force declares no signal of the name.
    Undeclared,
    /// The stated unit is not the one the document declares.
    WrongUnit,
    /// The value is not a finite number.
    NotFinite,
    /// The worker is no member.
    NotAMember,
    /// A newer sample of the name is on file for the rank.
    Older,
}

impl Taken {
    /// Every outcome, in the order of the store's counts
    /// ([`StoreCounts::signal_writes`](super::StoreCounts::signal_writes)).
    pub const EACH: [Self; 6] = [
        Self::Written,
        Self::Undeclared,
        Self::WrongUnit,
        Self::NotFinite,
        Self::NotAMember,
        Self::Older,
    ];

    /// The outcome's label in a metric series.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Written => "written",
            Self::Undeclared => "undeclared",
            Self::WrongUnit => "wrong_unit",
            Self::NotFinite => "not_finite",
            Self::NotAMember => "not_a_member",
            Self::Older => "older",
        }
    }

    /// The outcome's place in [`Self::EACH`].
    const fn place(self) -> usize {
        match self {
            Self::Written => 0,
            Self::Undeclared => 1,
            Self::WrongUnit => 2,
            Self::NotFinite => 3,
            Self::NotAMember => 4,
            Self::Older => 5,
        }
    }
}

/// The writes to a sink by what became of each: one count an outcome, read
/// and written with no lock.
#[derive(Debug, Default)]
pub(crate) struct Writes([AtomicU64; Taken::EACH.len()]);

impl Writes {
    /// One more write that came to `taken`.
    pub(crate) fn count(&self, taken: Taken) {
        self.0[taken.place()].fetch_add(1, Relaxed);
    }

    /// The counts as they stand, in the order of [`Taken::EACH`].
    pub(crate) fn read(&self) -> [u64; Taken::EACH.len()] {
        std::array::from_fn(|at| self.0[at].load(Relaxed))
    }
}
