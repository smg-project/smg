//! Signal cells: named values of a worker's report, read beside the rows.
//!
//! A signal here is a named number about one row at one instant: a value
//! a report states under a name the document declares
//! ([`SignalSpec`](crate::document::spec::SignalSpec)). The
//! lifecycle's event [`Signal`](crate::tracking::Signal) is the other thing of that
//! name: one report, health change or veto about one worker, as a stage
//! hears it.
//!
//! A stage resolves a signal once, when it is built
//! ([`StageBuild::signal_column`](crate::stages::StageBuild::signal_column)), and
//! from then on holds a [`SignalColumn`]. A decision reads by the row's
//! position and the column's number ([`SignalColumn::read`]): an index
//! and a compare of an age. No name and no hash is read in a decision.
//!
//! The cells stand beside the rows, not in them: a decision reads them
//! through the request's facts
//! ([`RequestFacts::signal_cells`](crate::engine::RequestFacts::signal_cells)),
//! under a profile whose stages read a signal and under no other. The
//! engine over worker lists makes them itself, from the load reports its
//! decision holds.
//!
//! A stage that keeps a number of each load report is told a report's
//! named values with the report, where its host states them
//! ([`ReportSignals`]). It reads one by the column it holds
//! ([`SignalColumn::told`]): the same index, and no age.

use crate::{
    document::spec::{AcrossRanks, MAX_SIGNALS},
    engine::view::{Candidate, RequestFacts},
};

/// The largest magnitude of a signal's value: the bound a cost rule's own
/// parameters are held to. A value times a coefficient then stays far
/// inside the finite range, and so does a sum of such terms. A cost that
/// left the range would level the score of every other row.
#[cfg(feature = "in-process")]
const VALUE_MAGNITUDE_LIMIT: f64 = {
    use crate::document::spec::COST_PARAMETER_LIMIT;
    COST_PARAMETER_LIMIT
};

/// Whether `value` is one a decision may read: a finite number of
/// magnitude at most 1e15. The crate's one write of a cell holds a
/// report's value to it ([`SignalCells::write`]).
#[cfg(feature = "in-process")]
pub(crate) fn in_range(value: f64) -> bool {
    value.is_finite() && value.abs() <= VALUE_MAGNITUDE_LIMIT
}

/// One value and when it was sampled, in microseconds on the host's clock.
///
/// A value is finite and of magnitude at most 1e15. The engine over worker
/// lists takes no other where it reads a load report. A host that hands
/// cells it built passes no such test, and holds its values to the same
/// bound itself.
///
/// The record gains fields: code outside this crate makes a cell by
/// [`SignalCell::of`] or starts from [`SignalCell::NONE`], and a pattern of
/// it ends with `..`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SignalCell {
    pub value: f64,
    pub sampled_at_micros: u64,
}

impl SignalCell {
    /// No value. A value is finite by rule, so none collides with it. The
    /// rule is the source's: the engine over worker lists writes no other
    /// value, and a host that hands cells it built hands finite values or
    /// this cell. A read tests for this cell alone, since a test of every
    /// value would stand on each read's path, and a debug build asserts
    /// the rule there.
    pub const NONE: Self = Self {
        value: f64::NAN,
        sampled_at_micros: 0,
    };

    /// A cell of `value`, sampled at `sampled_at_micros`. It takes both
    /// fields: a cell is a value and when it was sampled, and neither has a
    /// neutral value beside the cell of no value ([`Self::NONE`]). It
    /// checks nothing: a host holds its values to the record's bound. Code
    /// outside this crate starts here: the record gains fields, so no
    /// literal of it builds there.
    #[inline]
    #[must_use]
    pub const fn of(value: f64, sampled_at_micros: u64) -> Self {
        Self {
            value,
            sampled_at_micros,
        }
    }

    /// Whether the cell holds no value.
    pub fn is_none(&self) -> bool {
        self.value.is_nan()
    }
}

/// The signal cells of one decision's rows: for each row, one cell a
/// signal the document declares, in the rows' order. It owns its cells: a
/// host builds one, keeps it and refills it in place
/// ([`SignalCells::cells_mut`]). The default is the empty one, of no
/// signal.
#[derive(Clone, Debug, Default)]
pub struct SignalCells {
    cells: Vec<SignalCell>,
    /// The cells a row: the document's count of signals, as [`counted`]
    /// takes it.
    signals: u32,
}

/// A row's count of cells as the reads take it. A count past the
/// document's cap ([`MAX_SIGNALS`]) is no document's. A copy of a row's
/// cells would make room for it, and past a `u32` a place computed under
/// it could wrap onto another row's cell. It is taken as no signal, so
/// every read answers no value.
fn counted(signals: usize) -> u32 {
    match u32::try_from(signals) {
        Ok(count) if signals <= MAX_SIGNALS => count,
        _ => 0,
    }
}

impl SignalCells {
    /// For a host that builds the cells itself: `cells` holds `signals`
    /// cells a row, in the rows' order, and a row's cells stand in the
    /// order of the document's names. A count past the document's cap
    /// ([`MAX_SIGNALS`]) is taken as no signal.
    pub fn new(cells: Vec<SignalCell>, signals: usize) -> Self {
        Self {
            cells,
            signals: counted(signals),
        }
    }

    /// The cells, to refill in place: no owner allocates in a decision.
    /// The count of cells a row stays as it was built: a host whose new
    /// document declares another count of signals builds a new
    /// `SignalCells`, and may hand it the old one's `Vec` to keep the room.
    pub fn cells_mut(&mut self) -> &mut Vec<SignalCell> {
        &mut self.cells
    }

    /// The cells emptied, to fill in place for rows of `signals` cells
    /// each: the crate's own refill, which sets the count as well.
    pub(crate) fn refill(&mut self, signals: usize) -> &mut Vec<SignalCell> {
        self.cells.clear();
        self.signals = counted(signals);
        &mut self.cells
    }

    /// The cells of `rows` rows of `signals` cells each, every one with no
    /// value: what a reader of the rows' reports writes the stated values
    /// into ([`Self::write`]). The room the cells held is kept.
    #[cfg(feature = "in-process")]
    pub(crate) fn blank(&mut self, rows: usize, signals: usize) {
        let signals = counted(signals) as usize;
        self.refill(signals)
            .resize(rows.saturating_mul(signals), SignalCell::NONE);
    }

    /// One value a report states for the row at `position` under `column`,
    /// in `unit`, sampled at `sampled_at_micros`, and whether it was
    /// written. It is written where a decision may read it: the unit is
    /// the column's, byte for byte, and the value is in range
    /// ([`in_range`]). A place outside the cells takes nothing.
    #[cfg(feature = "in-process")]
    pub(crate) fn write(
        &mut self,
        position: u32,
        column: &SignalColumn,
        unit: &str,
        value: f64,
        sampled_at_micros: u64,
    ) -> bool {
        if column.unit() != unit || !in_range(value) {
            return false;
        }
        let place = u64::from(position) * u64::from(self.signals) + u64::from(column.column);
        let held = usize::try_from(place)
            .ok()
            .filter(|_| u32::from(column.column) < self.signals)
            .and_then(|place| self.cells.get_mut(place));
        let Some(held) = held else {
            return false;
        };
        *held = SignalCell {
            value,
            sampled_at_micros,
        };
        true
    }

    /// The cell of the row at `position` under `column`: `None` for a
    /// place outside the cells. A column past a row's cells is outside
    /// them too: read on, it would answer the next row's value.
    #[inline]
    fn cell(&self, position: u32, column: u16) -> Option<&SignalCell> {
        if u32::from(column) >= self.signals {
            return None;
        }
        // No wrap: a `u32` times a `u32`, and a `u16` more, is inside a
        // `u64`.
        let place = u64::from(position) * u64::from(self.signals) + u64::from(column);
        self.cells.get(usize::try_from(place).ok()?)
    }

    /// The cells a row: the document's count of signals.
    pub(crate) fn signals(&self) -> usize {
        self.signals as usize
    }

    /// The cells of the row at `position`, in the columns' order: `None`
    /// for a row outside the cells.
    pub(crate) fn row(&self, position: u32) -> Option<&[SignalCell]> {
        let start = usize::try_from(u64::from(position) * u64::from(self.signals)).ok()?;
        self.cells.get(start..start.checked_add(self.signals())?)
    }

    /// Whether the cells are those of `rows` rows: what a debug build
    /// asserts where a decision takes its inputs.
    pub(crate) fn fits(&self, rows: usize) -> bool {
        rows.checked_mul(self.signals()) == Some(self.cells.len())
    }
}

/// The named values one load report states, as a stage is told them with
/// the report ([`Signal::Load`](crate::tracking::Signal)): one cell a
/// signal the document declares, in the order of the document's names,
/// each value with its own sample instant. A name the report does not
/// state holds the empty cell ([`SignalCell::NONE`]).
///
/// A stage reads a value by the column it took at its build
/// ([`SignalColumn::told`]). The cells are lent for the call that hands
/// them: a stage copies out what it keeps, and a cell is `Copy`.
///
/// Two of them are equal where they hold as many cells and each cell
/// holds the same sample instant and the same value by its bits: the
/// empty cell equals itself, which no compare of two numbers would say.
#[derive(Clone, Copy, Debug)]
pub struct ReportSignals<'a>(&'a [SignalCell]);

impl<'a> ReportSignals<'a> {
    /// No named value: every column reads none. A report states it under
    /// a document in which no stage that keeps reports keeps a signal's
    /// column. So does a report that is not the whole of its worker
    /// object. A host that states no named value tells every report with
    /// it.
    pub const NONE: Self = Self(&[]);

    /// The values in `cells`: one cell a signal the document declares, in
    /// the order of the document's names
    /// ([`AppliedSpec::signal_columns`](crate::document::AppliedSpec::signal_columns)),
    /// as one row of [`SignalCells`] stands. For a host that builds the
    /// load signal itself, and for a case that tells a stage a report
    /// with values ([`Signal::with_signals`](crate::tracking::Signal::with_signals)).
    /// Such a host holds its values to the cells' bound itself
    /// ([`SignalCell`]).
    #[must_use]
    pub const fn new(cells: &'a [SignalCell]) -> Self {
        Self(cells)
    }
}

impl PartialEq for ReportSignals<'_> {
    fn eq(&self, other: &Self) -> bool {
        let same = |(ours, theirs): (&SignalCell, &SignalCell)| {
            ours.value.to_bits() == theirs.value.to_bits()
                && ours.sampled_at_micros == theirs.sampled_at_micros
        };
        self.0.len() == other.0.len() && self.0.iter().zip(other.0).all(same)
    }
}

impl Eq for ReportSignals<'_> {}

/// One declared signal, resolved: its column's number, its unit, its
/// maximum age and how a report of several ranks makes one row's value.
#[derive(Clone, Debug)]
pub struct SignalColumn {
    name: Box<str>,
    unit: Box<str>,
    column: u16,
    /// The maximum age, [`u64::MAX`] where the document writes none: a
    /// read then compares one age either way.
    max_age_micros: u64,
    across_ranks: AcrossRanks,
}

impl SignalColumn {
    pub(crate) fn new(name: &str, unit: &str, column: u16, max_age_ms: Option<u32>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            column,
            max_age_micros: max_age_ms.map_or(u64::MAX, |ms| u64::from(ms) * 1000),
            across_ranks: AcrossRanks::default(),
        }
    }

    /// The column under the reduction its signal declares.
    pub(crate) fn across(mut self, across_ranks: AcrossRanks) -> Self {
        self.across_ranks = across_ranks;
        self
    }

    /// The signal's name, as the document declares it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The unit the document declares for the signal's values.
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// How one row's value is made where its report holds several ranks,
    /// as the document declares it.
    pub fn across_ranks(&self) -> AcrossRanks {
        self.across_ranks
    }

    /// The signal's maximum age in microseconds, as the document declares
    /// it (`max_age_ms`): `None` where it declares none, and a value does
    /// not end by its age. A decision's read holds a value to it
    /// ([`Self::read`]). A stage that keeps a value it was told
    /// ([`Self::told`]) holds the value to it at its own decisions.
    #[must_use]
    pub fn max_age_micros(&self) -> Option<u64> {
        (self.max_age_micros != u64::MAX).then_some(self.max_age_micros)
    }

    /// The row's value at this decision: `None` where the facts hand no
    /// cells, where the row has no value, and where the value is older
    /// than the signal's maximum age. A value at exactly that age is read.
    /// The age is the facts' instant less the cell's sample instant, so a
    /// value sampled after that instant is read.
    #[inline]
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> Option<f64> {
        let cell = facts.signal_cells?.cell(row.position, self.column)?;
        fresh(cell, self.oldest(facts)).map(|cell| cell.value)
    }

    /// The value a told report states under this column, with its own
    /// sample instant ([`Signal::Load`](crate::tracking::Signal)): `None`
    /// where the report states none under the signal's name, and where it
    /// states no named value at all ([`ReportSignals::NONE`]). No age is
    /// tested, since a telling has no request's clock: a stage that keeps
    /// the value ages it at its decisions, by the cell's sample instant
    /// and the signal's maximum age ([`Self::max_age_micros`]).
    #[inline]
    #[must_use]
    pub fn told(&self, signals: ReportSignals<'_>) -> Option<SignalCell> {
        // The first instant of all: no value was sampled before it.
        self.held(signals.0, 0).copied()
    }

    /// The oldest sample instant a decision still reads of the signal: one
    /// value for every row of a decision, where an age is a subtraction a
    /// row. A pass takes it once.
    #[inline]
    pub(crate) fn oldest(&self, facts: &RequestFacts<'_>) -> u64 {
        facts.now_micros.saturating_sub(self.max_age_micros)
    }

    /// The value among one row's cells ([`SignalCells::row`]), for a pass
    /// that took [`Self::oldest`] once: `None` where the row holds no
    /// value, or one sampled before that instant.
    #[inline]
    pub(crate) fn among(&self, cells: &[SignalCell], oldest: u64) -> Option<f64> {
        self.held(cells, oldest).map(|cell| cell.value)
    }

    /// The column's cell among one row's cells, where it holds a value
    /// sampled at or after `oldest`: the one index of a row's cells, for
    /// a pass's read and for a told report's. Cells that end before the
    /// column, as the empty ones do, answer `None`.
    #[inline]
    fn held<'c>(&self, cells: &'c [SignalCell], oldest: u64) -> Option<&'c SignalCell> {
        fresh(cells.get(usize::from(self.column))?, oldest)
    }
}

/// The cell, where it holds a value sampled at or after `oldest`: the one
/// test for the empty cell, behind every read.
#[inline]
fn fresh(cell: &SignalCell, oldest: u64) -> Option<&SignalCell> {
    debug_assert!(
        cell.is_none() || cell.value.is_finite(),
        "a signal's value is finite: no source states another"
    );
    (!cell.is_none() && cell.sampled_at_micros >= oldest).then_some(cell)
}

/// The column of the signal `name` among a document's resolved columns,
/// which stand in name order. The one search behind every stage that
/// names a signal: a registered stage's context and the typed stages ask
/// here.
pub(crate) fn column_named<'a>(
    signals: &'a [SignalColumn],
    name: &str,
) -> Option<&'a SignalColumn> {
    let at = signals
        .binary_search_by(|column| column.name().cmp(name))
        .ok()?;
    signals.get(at)
}

/// A signal a sum prices: its column, resolved where the stage is built,
/// and its coefficient.
pub(crate) type SignalTerm = (SignalColumn, f64);

/// One pass's reads of the signals a stage sums. Once a pass: the facts'
/// cells, and each term's oldest instant still read. Once a row: the
/// row's cells ([`Self::row`]). A term is then an index, a test and a
/// multiply and add.
pub(crate) struct SignalSums<'a> {
    cells: Option<&'a SignalCells>,
    terms: &'a [SignalTerm],
    /// Each term's oldest instant still read, in the terms' order.
    oldest: [u64; MAX_SIGNALS],
}

impl<'a> SignalSums<'a> {
    #[inline(always)]
    pub(crate) fn new(facts: &RequestFacts<'a>, terms: &'a [SignalTerm]) -> Self {
        debug_assert!(
            terms.len() <= MAX_SIGNALS,
            "a sum names a declared signal once"
        );
        let mut oldest = [0; MAX_SIGNALS];
        for (oldest, (signal, _)) in oldest.iter_mut().zip(terms) {
            *oldest = signal.oldest(facts);
        }
        Self {
            cells: facts.signal_cells,
            terms,
            oldest,
        }
    }

    /// The row's cells: `None` where the facts hand no cells, and for a
    /// row outside them.
    #[inline(always)]
    pub(crate) fn row(&self, row: &Candidate) -> Option<&'a [SignalCell]> {
        self.cells?.row(row.position)
    }

    /// `sum` with each term's value times its coefficient added, in the
    /// terms' order, and whether every term had a fresh value. A term
    /// with none adds nothing.
    #[inline(always)]
    pub(crate) fn add(&self, mut sum: f64, cells: Option<&[SignalCell]>) -> (f64, bool) {
        let Some(cells) = cells else {
            return (sum, self.terms.is_empty());
        };
        let mut whole = true;
        for ((signal, coefficient), &oldest) in self.terms.iter().zip(&self.oldest) {
            match signal.among(cells, oldest) {
                Some(value) => sum += value * coefficient,
                None => whole = false,
            }
        }
        (sum, whole)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/load/signals/tests.rs"]
mod tests;
