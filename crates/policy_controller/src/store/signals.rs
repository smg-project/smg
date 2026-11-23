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
//! A source reads the declared columns once a report
//! ([`SignalSink::signal_columns`]) and writes each value under the column
//! it read. A value means its column's name, unit and reduction across
//! ranks, so the store takes it only where the document in force
//! continues that column. A document applied between the read and the
//! write moves no value to a column that is not its own.
//!
//! A report that leaves a name out removes nothing. The value on file
//! stands until the signal's maximum age ends it, until its rank's load
//! sample goes, or until a new document declares the name under another
//! unit or reduction. A worker that leaves and joins again starts with no
//! value.
//!
//! [`AppliedSpec::signal_columns`]: crate::document::AppliedSpec::signal_columns
//! [`LoadSink`]: super::LoadSink

use std::{
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::Instant,
};

use crate::{engine::view::WorkerId, load::signals::SignalColumn};

#[cfg(test)]
mod tests;

/// Where a source writes a signal's value: the store's door to the cells
/// of the document in force.
pub trait SignalSink: Send + Sync {
    /// The declared signals of the document in force, in column order:
    /// what a report's conversion walks, so it offers declared names alone,
    /// and what a host's own feeder reads ahead of its writes.
    fn signal_columns(&self) -> Arc<[SignalColumn]>;

    /// One value under `column`, a column its source read from
    /// [`Self::signal_columns`], for one rank's row, sampled at
    /// `sampled_at` on the host's clock: the value's own age where its
    /// report states one, else the age of the record that carried it, else
    /// its receipt.
    ///
    /// `unit` is the unit the value is in: the one its report states, or
    /// the one a host's feeder holds in its code. The value is taken where
    /// `unit` is the column's, the document in force continues the column
    /// (the same name, unit and reduction across ranks), the value is
    /// finite and of magnitude at most 1e15, the worker is a member and no
    /// newer sample of the name is on file for the rank. Of two sources of
    /// one signal the newer sample stands, whichever arrives last. Every
    /// other write is dropped and counted by what became of it.
    /// The column, the unit and the value are read before any lock is
    /// taken: a value under a column no document continues costs one
    /// lookup.
    fn signal_value(
        &self,
        id: WorkerId,
        rank: u16,
        column: &SignalColumn,
        unit: &str,
        value: f64,
        sampled_at: Instant,
    ) -> SignalWriteOutcome;
}

/// What became of a write to the signal sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalWriteOutcome {
    /// The value is on file and is shown with the next publication.
    Written,
    /// The document in force does not continue the column the value was
    /// written under: it declares no signal of the name, or declares it
    /// under another unit or another reduction across ranks.
    Undeclared,
    /// The unit the value is in is not the column's.
    WrongUnit,
    /// The value is not a finite number, or its magnitude is above 1e15:
    /// the bound a cost rule's own parameters are held to.
    OutOfRange,
    /// The worker is no member.
    NotAMember,
    /// A newer sample of the name is on file for the rank.
    Older,
}

impl SignalWriteOutcome {
    /// Every outcome, in the order of the store's counts
    /// ([`StoreStats::signal_writes`](super::StoreStats::signal_writes)).
    pub const EACH: [Self; 6] = [
        Self::Written,
        Self::Undeclared,
        Self::WrongUnit,
        Self::OutOfRange,
        Self::NotAMember,
        Self::Older,
    ];

    /// The outcome's label in a metric series.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Written => "written",
            Self::Undeclared => "undeclared",
            Self::WrongUnit => "wrong_unit",
            Self::OutOfRange => "out_of_range",
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
            Self::OutOfRange => 3,
            Self::NotAMember => 4,
            Self::Older => 5,
        }
    }
}

/// The writes to a sink by what became of each: one count an outcome, read
/// and written with no lock.
#[derive(Debug, Default)]
pub(crate) struct Writes([AtomicU64; SignalWriteOutcome::EACH.len()]);

impl Writes {
    /// One more write that came to `taken`.
    pub(crate) fn count(&self, taken: SignalWriteOutcome) {
        self.0[taken.place()].fetch_add(1, Relaxed);
    }

    /// The counts as they stand, in the order of [`SignalWriteOutcome::EACH`].
    pub(crate) fn read(&self) -> [u64; SignalWriteOutcome::EACH.len()] {
        std::array::from_fn(|at| self.0[at].load(Relaxed))
    }
}
