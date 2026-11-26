use super::*;

impl SignalCells {
    /// The cells, row after row.
    #[cfg(feature = "in-process")]
    pub(crate) fn cells(&self) -> &[SignalCell] {
        &self.cells
    }
}

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
    let valued = |cells: &SignalCells| cells.cells().iter().filter(|cell| !cell.is_none()).count();
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

/// A told report's value is read by its column, with its own sample
/// instant and at any age. The third signal ends past a millisecond at
/// a decision, whose read of the same cells passes the value over. A
/// cell with no value reads none. So does a column past the report's
/// cells, as every column is under a report that states no value.
#[test]
fn a_told_value_is_read_by_its_column_with_its_instant_at_any_age() {
    let cell = |value, sampled_at_micros| SignalCell {
        value,
        sampled_at_micros,
    };
    let cells = [cell(7.0, 9), SignalCell::NONE, cell(-0.5, 0)];
    let column = |at: u16| SignalColumn::new("held", "tokens", at, (at == 2).then_some(1));
    let told = |at: u16, signals| {
        let cell = column(at).told(signals)?;
        Some((cell.value, cell.sampled_at_micros))
    };
    let stated = ReportSignals::new(&cells);
    assert_eq!(told(0, stated), Some((7.0, 9)));
    assert_eq!(told(1, stated), None, "the empty cell");
    assert_eq!(told(2, stated), Some((-0.5, 0)), "at any age");
    assert_eq!(column(2).among(&cells, 1), None, "a decision ages it");
    assert_eq!(column(2).among(&cells, 0), Some(-0.5));
    let ages = [column(0), column(2)].map(|column| column.max_age_micros());
    assert_eq!(ages, [None, Some(1_000)], "as the document declares");
    let longer = SignalColumn::new("held", "tokens", 0, Some(2_500));
    assert_eq!(longer.max_age_micros(), Some(2_500_000), "each its own");
    assert_eq!(told(3, stated), None, "past the report's cells");
    assert_eq!(told(u16::MAX, stated), None);
    for at in 0..3 {
        assert_eq!(told(at, ReportSignals::NONE), None, "column {at}");
    }
}

/// Two reports' values are equal cell by cell: each value by its bits,
/// so the empty cell equals itself and the two zeros differ, and each
/// sample instant. A report of fewer cells is another report.
#[test]
fn two_reports_values_are_equal_by_each_cells_bits_and_instant() {
    let cell = |value, sampled_at_micros| SignalCell {
        value,
        sampled_at_micros,
    };
    let held = [cell(7.0, 9), SignalCell::NONE];
    let same = [cell(7.0, 9), SignalCell::NONE];
    assert_eq!(ReportSignals::new(&held), ReportSignals::new(&same));
    assert_eq!(ReportSignals::new(&[]), ReportSignals::NONE);
    for other in [
        [cell(7.0, 8), SignalCell::NONE],
        [cell(7.5, 9), SignalCell::NONE],
        [cell(7.0, 9), cell(0.0, 0)],
        [SignalCell::NONE, SignalCell::NONE],
    ] {
        let other = ReportSignals::new(&other);
        assert_ne!(ReportSignals::new(&held), other);
    }
    assert_ne!(ReportSignals::new(&held), ReportSignals::new(&held[..1]));
    assert_ne!(ReportSignals::new(&held[..1]), ReportSignals::NONE);
    let zeros = [cell(0.0, 0), cell(-0.0, 0)];
    let (plus, minus) = (&zeros[..1], &zeros[1..]);
    assert_ne!(ReportSignals::new(plus), ReportSignals::new(minus));
}

#[test]
fn the_default_cells_hold_no_signal() {
    let cells = SignalCells::default();
    assert!(cells.fits(0) && cells.fits(64));
    assert!(cells.cell(0, 0).is_none());
}
