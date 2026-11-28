//! The guarantees page's table of limits, read against the crate's source:
//! the page shows the rows this file holds, and each row's constants stand
//! in the source as written here. Cases of the pages' test: two more hold
//! the walk of the table, and its rule of a row, to short texts.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{cells_between, fence, guarantees};

/// One limit of the guarantees page: a row's first cell, its value's cell,
/// and where the crate holds the value, as a source file under `src` and the
/// constant's line there. A row names the constant of each limit that has
/// one: a header's position is held to a byte's range, and names none.
type Limit = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);

/// The guarantees page's limits, in the order of its table.
const LIMITS: [Limit; 15] = [
    (
        "An incoming `Decide` frame, unknown fields included",
        "64 KiB",
        &[(
            "service/controller/wire.rs",
            "pub(crate) const MAX_FRAME: usize = 64 * 1024;",
        )],
    ),
    (
        "A request ID",
        "1 to 256 bytes",
        &[(
            "service/controller/wire.rs",
            "pub(crate) const MAX_REQUEST_ID: usize = 256;",
        )],
    ),
    (
        "The headers of a request",
        "256, at positions 0 to 255",
        &[("document/spec.rs", "pub const MAX_HEADERS: usize = 256;")],
    ),
    (
        "The credits of a decide stream",
        "128",
        &[(
            "service/controller/decide.rs",
            "pub(crate) const WINDOW: u32 = 128;",
        )],
    ),
    (
        "The decide streams of a gateway",
        "256",
        &[(
            "service/controller/decide.rs",
            "pub(crate) const MAX_STREAMS: u32 = 256;",
        )],
    ),
    (
        "The wait for `Hello`",
        "5 seconds",
        &[(
            "service/controller/session.rs",
            "pub(crate) const HELLO_DEADLINE: Duration = Duration::from_secs(5);",
        )],
    ),
    (
        "A gateway's grace after a disconnect",
        "10 seconds",
        &[(
            "service/controller/session.rs",
            "pub(crate) const GRACE_PERIOD: Duration = Duration::from_secs(10);",
        )],
    ),
    (
        "The gateways in the table",
        "4,096",
        &[(
            "service/controller/session.rs",
            "pub(crate) const PRESENT_MAX: usize = 4_096;",
        )],
    ),
    (
        "The vetoes of a gateway",
        "4,096, and 8 kinds a worker",
        &[
            (
                "service/controller/gateways.rs",
                "pub(crate) const VETOES_MAX: usize = 4_096;",
            ),
            (
                "service/controller/gateways.rs",
                "pub(crate) const KINDS_MAX: usize = 8;",
            ),
        ],
    ),
    (
        "The watches of a principal at a replica",
        "4",
        &[(
            "service/controller/watch_workers.rs",
            "pub(crate) const WATCHES_MAX: usize = 4;",
        )],
    ),
    (
        "A watch page",
        "128 rows and 2 MiB of row data",
        &[
            (
                "service/controller/watch_workers.rs",
                "const PAGE: usize = 128;",
            ),
            (
                "document/operator/snapshot.rs",
                "pub(crate) const PAGE_BYTES: usize = 2 * 1024 * 1024;",
            ),
        ],
    ),
    (
        "The model IDs of a watch or a snapshot",
        "128",
        &[(
            "document/operator/snapshot.rs",
            "pub(crate) const MODELS: usize = 128;",
        )],
    ),
    (
        "The snapshots that wait",
        "32",
        &[(
            "service/controller/snapshot.rs",
            "pub(crate) const SNAPSHOTS_WAITING: usize = 32;",
        )],
    ),
    (
        "The watched events kept for replication",
        "4,096",
        &[(
            "service/controller/replication.rs",
            "pub(crate) const WATCHED_EVENTS: usize = 4_096;",
        )],
    ),
    (
        "The snapshot restores a leader holds at once",
        "4",
        &[(
            "service/controller/replication.rs",
            "const RESTORES: usize = 4;",
        )],
    ),
];

/// The crate's source folder.
fn source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// A row of a table as exactly `count` cells. `None` where the row is not
/// `count` cells between bars of its own, the first bar at the line's
/// start and the last at its end. A bar behind a backslash is a cell's
/// own, at the row's end too. The text between the two bars is split as
/// every row of the pages is (`cells_between`).
fn framed_cells(row: &str, count: usize) -> Option<Vec<String>> {
    let inner = row.strip_prefix('|')?.strip_suffix('|')?;
    if inner.ends_with('\\') {
        return None;
    }
    let cells = cells_between(inner);
    (cells.len() == count).then_some(cells)
}

/// The rows of the table of limits on `page`, each as its two cells, and
/// into `faults` each thing that keeps the table from being whole.
///
/// Fence lines (`fence`) open and close a block by turns: a fence line
/// and a line of a block read as blank lines here. A line is read less
/// the blanks at its end. The header and its ruled line stand at their
/// lines' start, one right behind the other: where they do not, that is
/// the one fault and no line is read as a row. The table ends at a blank
/// line or with the page: each line before that is a row of two cells
/// ([`framed_cells`]), and any other line there is a fault.
fn limits_table(page: &str, faults: &mut Vec<String>) -> Vec<Vec<String>> {
    let mut fenced = false;
    let mut lines = page
        .lines()
        .map(|line| {
            if fence(line).is_some() {
                fenced = !fenced;
                return "";
            }
            if fenced {
                ""
            } else {
                line.trim_end()
            }
        })
        .skip_while(|line| *line != "| Limit | Value |");
    if lines.nth(1) != Some("|---|---|") {
        faults.push("the table of limits has no header with its ruled line".to_owned());
        return Vec::new();
    }
    let mut rows = Vec::new();
    for row in lines.take_while(|line| !line.is_empty()) {
        match framed_cells(row, 2) {
            Some(cells) => rows.push(cells),
            None => faults.push(format!(
                "a line of the table of limits is not two cells between three bars: {row:?}"
            )),
        }
    }
    rows
}

#[test]
fn the_guarantees_page_states_each_limit_as_the_crates_source_holds_it() {
    let page = fs::read_to_string(guarantees()).expect("the guarantees page is readable");
    let mut faults = Vec::new();
    // The table of limits: a row a limit, in this list's order.
    let rows = limits_table(&page, &mut faults);
    let shown: Vec<(&str, &str)> = rows
        .iter()
        .map(|row| {
            (
                row.first().map_or("", String::as_str),
                row.get(1).map_or("", String::as_str),
            )
        })
        .collect();
    let held: Vec<(&str, &str)> = LIMITS
        .iter()
        .map(|(limit, value, _)| (*limit, *value))
        .collect();
    if shown != held {
        faults.push(format!(
            "the table shows {shown:?}, and this case holds {held:?}"
        ));
    }
    // Each value at its constant: the constant's line, trimmed, stands once
    // in its file. Where the line's text changes or stands there no more,
    // this fails: the page's row and this list are then read again.
    for (limit, _, constants) in LIMITS {
        for (file, line) in constants {
            let text = fs::read_to_string(source().join(file))
                .expect("a source file of the crate is readable");
            let found = text.lines().filter(|read| read.trim() == *line).count();
            if found != 1 {
                faults.push(format!(
                    "the limit {limit:?}: `{line}` stands {found} times in src/{file}"
                ));
            }
        }
    }
    assert!(
        faults.is_empty(),
        "the guarantees page's limits are not the source's:\n{}",
        faults.join("\n")
    );
}

#[test]
fn a_row_of_two_cells_stands_between_three_bars_of_its_own() {
    for row in ["| a | b |", "|a|b|", "| a \\| c | b |"] {
        assert!(
            framed_cells(row, 2).is_some(),
            "a row of two cells: {row:?}"
        );
    }
    assert_eq!(
        framed_cells("| a \\| c | b |", 2),
        Some(vec!["a \\| c".to_owned(), "b".to_owned()]),
        "a bar behind a backslash is its cell's own"
    );
    // Each of these breaks one condition of the rule: no bar ahead, none
    // behind, a row set in, a bar too many ahead and one behind, three
    // cells, one cell, and a last bar that is its cell's own.
    for row in [
        "a | b |",
        "| a | b",
        " | a | b |",
        "|| a | b |",
        "| a | b ||",
        "| a | b | c |",
        "| a |",
        "| a | b |\\|",
    ] {
        assert!(
            framed_cells(row, 2).is_none(),
            "no row of two cells: {row:?}"
        );
    }
}

#[test]
fn the_walk_of_the_table_of_limits_reads_a_whole_table_and_nothing_else() {
    // The rows read and the faults named, by their counts, on a short page.
    let read = |page: &str| {
        let mut faults = Vec::new();
        let rows = limits_table(page, &mut faults).len();
        (rows, faults.len())
    };
    let table = "| Limit | Value |\n|---|---|\n| a | 1 |\n| b | 2 |\n";
    let set_in = format!("    {}", table.replace("\n|", "\n    |"));
    let pages = [
        (table.to_owned(), (2, 0), "a table at the page's end"),
        (
            format!("Before.\n\n{table}\nBehind.\n"),
            (2, 0),
            "a table between two blank lines",
        ),
        (
            "| Limit | Value | \n|---|---| \n| a | 1 | \n".to_owned(),
            (1, 0),
            "blanks behind each line",
        ),
        // A block that closed ahead of the table keeps no line of it.
        (
            format!("```text\n| x | y |\n```\n{table}"),
            (2, 0),
            "a closed block ahead of the table",
        ),
        // A fence line ends the table: a table behind the block is none of it.
        (
            format!("{table}```text\n```\n | Other | Value |\n |---|---|\n"),
            (2, 0),
            "a fenced block right behind the table",
        ),
        // What keeps a table from being whole.
        (
            format!("{table}extra | 9\n"),
            (2, 1),
            "a line right behind the last row",
        ),
        (format!("{table} | c | 3 |\n"), (2, 1), "a row set in"),
        (
            table.replace("| a | 1 |", "|| a | 1 |"),
            (1, 1),
            "a bar too many ahead of a row",
        ),
        (
            table.replace("|---|---|\n", "```text\n```\n|---|---|\n"),
            (0, 1),
            "a block between the header and its ruled line",
        ),
        (
            format!("```text\n{table}```\n"),
            (0, 1),
            "a table inside a block",
        ),
        (set_in, (0, 1), "a table set in by four blanks"),
        (
            "| Limit | Value |\n\nOutside.\n| x | 1 |\n".to_owned(),
            (0, 1),
            "a row further down behind a header with no ruled line",
        ),
    ];
    for (page, counts, what) in pages {
        assert_eq!(read(&page), counts, "{what}");
    }
    // The rows themselves: each as its two cells, in the page's order.
    let mut faults = Vec::new();
    assert_eq!(
        limits_table(table, &mut faults),
        [["a", "1"], ["b", "2"]],
        "the rows of a whole table"
    );
    assert!(faults.is_empty(), "a whole table has no fault: {faults:?}");
}
