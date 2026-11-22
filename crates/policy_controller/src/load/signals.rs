//! Signal cells: named values of a worker's report, read beside the rows.
//!
//! A signal here is a named number about one row at one instant: a value
//! a report states under a name the document declares
//! ([`SignalSpec`](crate::document::spec::SignalSpec)), published with the rows. The
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
//! store-backed engine fills them from its store. The engine over worker
//! lists makes them itself, from the load reports its decision holds.

use crate::{
    document::spec::{AcrossRanks, COST_PARAMETER_LIMIT, MAX_SIGNALS},
    engine::view::{Candidate, RequestFacts},
};

/// The largest magnitude of a signal's value: the bound a cost rule's own
/// parameters are held to. A value times a coefficient then stays far
/// inside the finite range, and so does a sum of such terms. A cost that
/// left the range would level the score of every other row.
const VALUE_MAGNITUDE_LIMIT: f64 = COST_PARAMETER_LIMIT;

/// Whether `value` is one a decision may read: a finite number of
/// magnitude at most 1e15. Every source of a value is held to it, where
/// it writes a cell.
pub(crate) fn in_range(value: f64) -> bool {
    value.is_finite() && value.abs() <= VALUE_MAGNITUDE_LIMIT
}

/// One value and when it was sampled, in microseconds on the host's clock.
///
/// A value is finite and of magnitude at most 1e15. The store's sink takes
/// no other, and neither does the engine over worker lists where it reads
/// a load report. A host that hands cells it built passes neither, and
/// holds its values to the same bound itself.
#[derive(Clone, Copy, Debug)]
pub struct SignalCell {
    pub value: f64,
    pub sampled_at_micros: u64,
}

impl SignalCell {
    /// No value. A value is finite by rule, so none collides with it. The
    /// rule is the source's: a store's sink takes no other value, and a
    /// host that hands cells it built hands finite values or this cell. A
    /// read tests for this cell alone, since a test of every value would
    /// stand on each read's path, and a debug build asserts the rule there.
    pub const NONE: Self = Self {
        value: f64::NAN,
        sampled_at_micros: 0,
    };

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
    /// For a host that keeps no store: `cells` holds `signals` cells a
    /// row, in the rows' order, and a row's cells stand in the order of
    /// the document's names. A count past the document's cap
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

    /// The cells, row after row.
    pub(crate) fn cells(&self) -> &[SignalCell] {
        &self.cells
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

    /// The row's value at this decision: `None` where the facts hand no
    /// cells, where the row has no value, and where the value is older
    /// than the signal's maximum age. A value at exactly that age is read.
    /// The age is the facts' instant less the cell's sample instant, so a
    /// value sampled after that instant is read.
    #[inline]
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> Option<f64> {
        let cell = facts.signal_cells?.cell(row.position, self.column)?;
        fresh(cell, self.oldest(facts))
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
        fresh(cells.get(usize::from(self.column))?, oldest)
    }
}

/// The cell's value, where it holds one sampled at or after `oldest`.
#[inline]
fn fresh(cell: &SignalCell, oldest: u64) -> Option<f64> {
    debug_assert!(
        cell.is_none() || cell.value.is_finite(),
        "a signal's value is finite: no source states another"
    );
    (!cell.is_none() && cell.sampled_at_micros >= oldest).then_some(cell.value)
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
mod tests {
    use super::*;

    const HELD: SignalCell = SignalCell {
        value: 7.0,
        sampled_at_micros: 0,
    };

    /// Under a count of `2^32 + 1` a place of 64 bits wraps: left as
    /// handed, the second row's first cell would be read at the place of
    /// the first row's second.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn a_count_of_signals_past_a_u32_holds_no_signal() {
        let cells = SignalCells::new(vec![HELD; 4], (1 << 32) + 1);
        assert_eq!(cells.signals(), 0);
        for position in 0..4 {
            assert!(cells.cell(position, 0).is_none(), "row {position}");
            assert_eq!(cells.row(position).map(<[_]>::len), Some(0));
        }
    }

    #[test]
    fn a_count_of_signals_past_the_cap_holds_no_signal() {
        for past in [MAX_SIGNALS + 1, u32::MAX as usize] {
            let mut cells = SignalCells::new(vec![HELD; 4], past);
            assert_eq!(cells.signals(), 0, "a count of {past}");
            assert!(cells.cell(0, 0).is_none(), "a count of {past}");
            assert_eq!(cells.row(0).map(<[_]>::len), Some(0));
            cells.refill(past);
            assert_eq!(cells.signals(), 0, "a refill under {past}");
        }
    }

    #[test]
    fn the_caps_count_stands_and_reads_inside_the_cells_alone() {
        let cells = SignalCells::new(vec![HELD; 4], MAX_SIGNALS);
        assert_eq!(cells.signals(), MAX_SIGNALS);
        assert!(cells.cell(0, 3).is_some());
        assert!(cells.cell(0, 4).is_none(), "past the cells' end");
        assert!(cells.cell(1, 0).is_none(), "a second row is outside them");
        assert!(
            cells.cell(u32::MAX, u16::MAX).is_none(),
            "a column past the count"
        );
        // The last column under the cap, at the last position.
        assert!(cells.cell(u32::MAX, 63).is_none(), "the last place of all");
        assert!(cells.row(0).is_none(), "a row no cells hold whole");
        assert!(
            !cells.fits(usize::MAX),
            "a product past the index is no fit"
        );
    }

    #[test]
    fn a_refill_empties_the_cells_keeps_their_room_and_sets_the_count() {
        let mut cells = SignalCells::new(vec![HELD; 6], 3);
        let room = cells.cells_mut().capacity();
        cells.refill(2).extend([HELD; 4]);
        assert_eq!(cells.signals(), 2);
        assert!(cells.fits(2));
        assert_eq!(cells.cells_mut().capacity(), room);
        assert!(cells.cell(1, 1).is_some());
        assert!(cells.cell(1, 2).is_none(), "the old count's third column");
    }

    /// A stated value is written at its row's place under its column, and
    /// nowhere a decision may not read it: under another unit than the
    /// column's, out of range, or at a place outside the cells.
    #[cfg(feature = "in-process")]
    #[test]
    fn a_stated_value_is_written_under_its_columns_unit_in_range_inside_the_cells() {
        let held = SignalColumn::new("held", "tokens", 1, None);
        let valued =
            |cells: &SignalCells| cells.cells().iter().filter(|cell| !cell.is_none()).count();
        let mut cells = SignalCells::default();
        cells.blank(3, 2);
        assert!(cells.fits(3) && valued(&cells) == 0, "every cell is blank");
        assert!(cells.write(2, &held, "tokens", 7.0, 9));
        let written = cells.cell(2, 1).expect("the cell");
        assert_eq!((written.value, written.sampled_at_micros), (7.0, 9));
        assert_eq!(valued(&cells), 1, "one cell was written");
        for (position, unit, value) in [
            (0, "bytes", 1.0),
            (0, "Tokens", 1.0),
            (0, "tokens", f64::NAN),
            (0, "tokens", f64::INFINITY),
            (0, "tokens", 1e15 + 1.0),
            (0, "tokens", -1e15 - 1.0),
            (3, "tokens", 1.0),
        ] {
            let taken = cells.write(position, &held, unit, value, 9);
            assert!(!taken, "{value} {unit} at row {position}");
        }
        // A column past the cells' count of signals names no cell.
        let past = SignalColumn::new("past", "tokens", 2, None);
        assert!(!cells.write(0, &past, "tokens", 1.0, 9));
        assert_eq!(valued(&cells), 1, "none of them was written");
        // The bound itself is in range, at either sign.
        assert!(cells.write(0, &held, "tokens", 1e15, 9));
        assert!(cells.write(1, &held, "tokens", -1e15, 9));
        // Blank again, over fewer rows: no value stays, and the room does.
        let room = cells.cells_mut().capacity();
        cells.blank(2, 2);
        assert!(cells.fits(2) && valued(&cells) == 0);
        assert_eq!(cells.cells_mut().capacity(), room);
    }

    #[test]
    fn the_default_cells_hold_no_signal() {
        let cells = SignalCells::default();
        assert!(cells.fits(0) && cells.fits(64));
        assert!(cells.cell(0, 0).is_none());
    }
}
