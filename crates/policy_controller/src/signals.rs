//! Signal cells: named values of a worker's report, read beside the rows.
//!
//! A signal here is a named number about one row at one instant: a value
//! a report states under a name the document declares
//! ([`SignalSpec`](crate::spec::SignalSpec)), published with the rows. The
//! lifecycle's event [`Signal`](crate::Signal) is the other thing of that
//! name: one report, health change or veto about one worker, as a stage
//! hears it.
//!
//! A stage resolves a signal once, when it is built
//! ([`StageBuild::signal_column`](crate::StageBuild::signal_column)), and
//! from then on holds a [`SignalColumn`]. A decision reads by the row's
//! position and the column's number ([`SignalColumn::read`]): an index
//! and a compare of an age. No name and no hash is read in a decision.
//!
//! The cells stand beside the rows, not in them: a host hands them
//! through the request's facts
//! ([`RequestFacts::signal_cells`](crate::RequestFacts::signal_cells)) for
//! a profile whose stages read a signal, and for no other.

use crate::view::{Candidate, RequestFacts};

/// One value and when it was sampled, in microseconds on the host's clock.
#[derive(Clone, Copy, Debug)]
pub struct SignalCell {
    pub value: f64,
    pub sampled_at_micros: u64,
}

impl SignalCell {
    /// No value. A value is finite by rule, so none collides with it.
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

/// A row's count of cells as the reads take it. A count a `u32` does not
/// hold is no document's, and a place computed under it could wrap onto
/// another row's cell: it is taken as no signal, so every read answers no
/// value.
fn counted(signals: usize) -> u32 {
    u32::try_from(signals).unwrap_or(0)
}

impl SignalCells {
    /// For a host that keeps no store: `cells` holds `signals` cells a
    /// row, in the rows' order, and a row's cells stand in the order of
    /// the document's names.
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

/// One declared signal, resolved: its column's number, its unit and its
/// maximum age.
#[derive(Clone, Debug)]
pub struct SignalColumn {
    name: Box<str>,
    unit: Box<str>,
    column: u16,
    /// The maximum age, [`u64::MAX`] where the document writes none: a
    /// read then compares one age either way.
    max_age_micros: u64,
}

impl SignalColumn {
    pub(crate) fn new(name: &str, unit: &str, column: u16, max_age_ms: Option<u32>) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            column,
            max_age_micros: max_age_ms.map_or(u64::MAX, |ms| u64::from(ms) * 1000),
        }
    }

    /// The signal's name, as the document declares it.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The unit the document declares for the signal's values.
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// The row's value at this decision: `None` where the facts hand no
    /// cells, where the row has no value, and where the value is older
    /// than the signal's maximum age. A value at exactly that age is read.
    /// The age is the facts' instant less the cell's sample instant, so a
    /// value sampled after that instant is read.
    #[inline]
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> Option<f64> {
        let cell = facts.signal_cells?.cell(row.position, self.column)?;
        // The oldest sample instant still read: one value for every row
        // of a decision, where an age is a subtraction a row.
        let oldest = facts.now_micros.saturating_sub(self.max_age_micros);
        (!cell.is_none() && cell.sampled_at_micros >= oldest).then_some(cell.value)
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
    fn the_last_count_a_u32_holds_stands_and_reads_inside_the_cells_alone() {
        let cells = SignalCells::new(vec![HELD; 4], u32::MAX as usize);
        assert!(cells.cell(0, 3).is_some());
        assert!(cells.cell(0, 4).is_none(), "past the cells' end");
        assert!(cells.cell(1, 0).is_none(), "a second row is outside them");
        assert!(
            cells.cell(u32::MAX, u16::MAX).is_none(),
            "the last place of all"
        );
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

    #[test]
    fn the_default_cells_hold_no_signal() {
        let cells = SignalCells::default();
        assert!(cells.fits(0) && cells.fits(64));
        assert!(cells.cell(0, 0).is_none());
    }
}
