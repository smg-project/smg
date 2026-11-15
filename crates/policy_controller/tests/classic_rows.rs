//! The classic policies' outcomes, replayed through this crate's decision.
//! A row is a pool of workers, what a router knows of them, one request
//! shape and what the gateway's own policy code picked for it: the sequence
//! of picks when no draw decides, or how often each worker won over a number
//! of single decisions when one does. The rows are recorded by running that
//! policy code, not written by hand, and each sheet names the commit it was
//! recorded from. A drawn row is one sample: a second recording of the same
//! commit draws other counts within the same spread. One module a classic
//! policy holds its rows' shape and their replay.
//!
//! The sheets live outside this repository, with the recorder:
//! `SMG_POLICY_CLASSIC_ROWS` names their directory for the ignored test. A
//! few rows of each form are copied into the policy's module from its
//! sheet, so every run of the suite with the `ingest` feature replays a
//! subset. A row's reports reach the view through that feature's row
//! builder, so a bare test run of the crate builds no part of this target
//! and replays no recorded row. The ignored test holds every copied row to
//! the sheet's row of its name.
//!
//! A sequence row must replay pick for pick. A drawn row must give each
//! worker the recorded share within the spread of two samples of that size,
//! and a worker never picked on one side must never be picked on the other.
//! No row is decided differently. A row whose form this crate does not
//! build yet is owed: [`OWED`] names it with the validator's words and the
//! change that builds it, and the replay fails once the row is no longer
//! refused in those words.
#![expect(clippy::expect_used, reason = "a replay that stops on a broken setup")]

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{Map, Value};

mod classic {
    pub mod manual;
    pub mod power_of_two;
}

/// What a row recorded, and what its replay gives in the same shape: the
/// worker each pick chose, in order, or how often each worker was chosen
/// over a number of single decisions.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Sequence(Vec<usize>),
    Counts { draws: u32, counts: Vec<u32> },
}

/// What a row's replay gives: its outcome in the recorded shape, or the
/// validator's words when this crate refuses to build the row's form.
enum Replayed {
    Ran(Outcome),
    Refused(String),
}

/// A recorded row whose form this crate does not build yet: validation
/// refuses it in the words `refused`, and `by` names the change that builds
/// it. The row passes while it is refused that way and fails the day it is
/// not, so an entry cannot outlive its debt.
struct Owed {
    row: &'static str,
    refused: &'static str,
    by: &'static str,
}

/// The owed rows. The classic manual policy keeps a key's pins apart under
/// each role of a pair; here a sessions rule runs under the single plan.
const OWED: &[Owed] = &[Owed {
    row: "manual-one-key-pins-apart-under-the-prefill-and-the-decode-role",
    refused: "sessions run under the single plan",
    by: "sessions on the sides of a pair, which read the pair's partner lists",
}];

fn shares(counts: &[u32], draws: u32) -> Vec<f64> {
    counts
        .iter()
        .map(|&count| f64::from(count) / f64::from(draws))
        .collect()
}

/// Whether two share lists are one distribution: a share of zero on one
/// side is zero on the other, and any other pair lies within four standard
/// deviations of the difference of two samples of `draws`.
fn same_shares(here: &[f64], there: &[f64], draws: u32) -> bool {
    here.len() == there.len()
        && here.iter().zip(there).all(|(&a, &b)| {
            if a == 0.0 || b == 0.0 {
                return a == b;
            }
            let p = (a + b) / 2.0;
            (a - b).abs() <= 4.0 * (2.0 * p * (1.0 - p) / f64::from(draws)).sqrt()
        })
}

/// Whether a replayed outcome is the recorded one: pick for pick, or share
/// for share.
fn same_outcome(recorded: &Outcome, here: &Outcome) -> bool {
    match (here, recorded) {
        (Outcome::Sequence(a), Outcome::Sequence(b)) => a == b,
        (
            Outcome::Counts { draws, counts },
            Outcome::Counts {
                counts: recorded, ..
            },
        ) => same_shares(&shares(counts, *draws), &shares(recorded, *draws), *draws),
        _ => false,
    }
}

/// `None` when a row's replay is its recorded outcome, or when the row is
/// owed and refused in its entry's words; the reason otherwise.
fn verdict(name: &str, recorded: &Outcome, replayed: &Replayed) -> Option<String> {
    let owed = OWED.iter().find(|owed| owed.row == name);
    match (replayed, owed) {
        (Replayed::Refused(why), Some(owed)) if why.contains(owed.refused) => None,
        (Replayed::Refused(why), _) => Some(format!("refused here: {why}")),
        (Replayed::Ran(_), Some(owed)) => Some(format!(
            "its form is built now ({}): take the row out of OWED",
            owed.by
        )),
        (Replayed::Ran(here), None) => (!same_outcome(recorded, here))
            .then(|| format!("recorded {recorded:?}, replayed {here:?}")),
    }
}

/// A recorded row of one classic policy, as its module reads it: what was
/// recorded, and what this crate's decision gives for the row.
trait Recorded: DeserializeOwned {
    /// The classic policy the rows record, as a sheet's head names it.
    const POLICY: &'static str;

    fn name(&self) -> &str;

    fn outcome(&self) -> &Outcome;

    /// This crate's outcome for the row in the shape of the recorded one,
    /// or its refusal of the row's form.
    fn replay(&self) -> Replayed;
}

/// The rows of a sheet, in one policy's shape.
#[derive(Deserialize)]
#[serde(bound = "R: Recorded")]
struct Rows<R> {
    rows: Vec<R>,
}

/// The rows of one policy's sheet replayed: how many, and what did not
/// replay.
fn replay_rows<R: Recorded>(text: &str, source: &str) -> (usize, Vec<String>) {
    let sheet: Rows<R> = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!(
                    "{source} is not a sheet of {} rows: {err}",
                    R::POLICY
                )],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .filter_map(|row| {
            verdict(row.name(), row.outcome(), &row.replay())
                .map(|why| format!("{source}: {}: {why}", row.name()))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// The names of the rows of one policy's sheet.
fn row_names<R: Recorded>(text: &str) -> Vec<String> {
    let sheet: Rows<R> = serde_json::from_str(text).expect("a sheet of rows");
    sheet
        .rows
        .iter()
        .map(|row| row.name().to_string())
        .collect()
}

/// A sheet's head: the commit its rows were recorded from, and the classic
/// policy they record, which names the module that replays them.
#[derive(Debug, Deserialize)]
struct Head {
    recorded_from: String,
    policy: String,
}

/// A classic policy a module here replays: the name a sheet's head gives
/// the policy, the rows copied from its sheet, and the replay of a sheet's
/// text in the module's row shape.
struct Module {
    policy: &'static str,
    inline: &'static str,
    replay: fn(&str, &str) -> (usize, Vec<String>),
}

/// The entry of the module whose rows are `R`: the policy's name and the
/// replay come from the one type, so no entry pairs two modules.
const fn module<R: Recorded>(inline: &'static str) -> Module {
    Module {
        policy: R::POLICY,
        inline,
        replay: replay_rows::<R>,
    }
}

/// Every module that replays, the one list. A sheet finds its module here
/// ([`replay_sheet`]), and each entry is held to one sheet and to its copied
/// rows ([`replay_sheets`]): a module added here has both.
const MODULES: [Module; 2] = [
    module::<classic::manual::Row>(classic::manual::INLINE),
    module::<classic::power_of_two::Row>(classic::power_of_two::INLINE),
];

/// The number of rows replayed and what did not replay; a text that is no
/// sheet is one defect, and so is a sheet of a policy no module replays.
fn replay_sheet(text: &str, source: &str) -> (usize, Vec<String>) {
    let head: Head = match serde_json::from_str(text) {
        Ok(head) => head,
        Err(err) => return (0, vec![format!("{source} is not a sheet of rows: {err}")]),
    };
    let module = MODULES.iter().find(|module| module.policy == head.policy);
    let (rows, mut defects) = match module {
        Some(module) => (module.replay)(text, source),
        None => (
            0,
            vec![format!(
                "{source}: a sheet of {}, which no module replays",
                head.policy
            )],
        ),
    };
    if head.recorded_from.is_empty() {
        defects.push(format!("{source}: no commit the rows were recorded from"));
    }
    (rows, defects)
}

/// A sheet's rows as written, for the comparison of a copied row with the
/// sheet's row of its name: every field but the outcome is an input.
#[derive(Deserialize)]
struct Written {
    rows: Vec<Map<String, Value>>,
}

/// A written row in its parts: its name and its outcome, and every field
/// but the outcome as its inputs.
struct Parts {
    name: String,
    outcome: Outcome,
    inputs: Map<String, Value>,
}

fn parts(mut row: Map<String, Value>) -> Option<Parts> {
    let outcome = serde_json::from_value(row.remove("outcome")?).ok()?;
    let name = row.get("name")?.as_str()?.to_string();
    Some(Parts {
        name,
        outcome,
        inputs: row,
    })
}

/// The rows of a sheet as written, each in its parts.
fn written_rows(text: &str, source: &str) -> Result<Vec<Parts>, String> {
    let written: Written = serde_json::from_str(text)
        .map_err(|err| format!("{source} is not a sheet of rows: {err}"))?;
    written
        .rows
        .into_iter()
        .map(|row| parts(row).ok_or_else(|| format!("{source}: a row with no name or no outcome")))
        .collect()
}

/// What separates the rows copied into a module from the sheet they were
/// copied from: a copied row the sheet does not hold, one whose inputs are
/// not the sheet's, or one whose outcome is not the sheet's by the replay's
/// own rule: a sequence pick for pick, a drawn row over the same number of
/// draws and share for share.
fn copied_row_defects(inline: &str, sheet: &str, source: &str) -> Vec<String> {
    let (copied, held) = match (
        written_rows(inline, "the inline rows"),
        written_rows(sheet, source),
    ) {
        (Ok(copied), Ok(held)) => (copied, held),
        (Err(why), _) | (_, Err(why)) => return vec![why],
    };
    copied
        .iter()
        .filter_map(|row| {
            let Parts {
                name,
                outcome,
                inputs,
            } = row;
            let Some(sheet_row) = held.iter().find(|held| held.name == *name) else {
                return Some(format!("{source}: the inline row {name} is on no sheet"));
            };
            let recorded = &sheet_row.outcome;
            if *inputs != sheet_row.inputs {
                return Some(format!(
                    "{source}: the inline row {name} has other inputs than the sheet's row"
                ));
            }
            let same_draws = match (outcome, recorded) {
                (Outcome::Counts { draws: here, .. }, Outcome::Counts { draws: there, .. }) => {
                    here == there
                }
                _ => true,
            };
            (!same_draws || !same_outcome(recorded, outcome)).then(|| {
                format!("{source}: the inline row {name} holds {outcome:?}, the sheet {recorded:?}")
            })
        })
        .collect()
}

/// The sheets, by their place, whose head names `policy`.
fn sheets_of(policy: &str, sheets: &[(String, String)]) -> Vec<usize> {
    sheets
        .iter()
        .enumerate()
        .filter(|(_, (_, text))| {
            serde_json::from_str::<Head>(text).is_ok_and(|head| head.policy == policy)
        })
        .map(|(at, _)| at)
        .collect()
}

/// Every sheet replayed and every module's copied rows held to its sheet:
/// a sheet's rows and what did not hold of it, in the order given, then
/// what no one sheet answers for: a policy with a module here and no sheet
/// among them, whose module replayed no recorded row (a defect, never a
/// pass), or with more than one.
fn replay_sheets(sheets: &[(String, String)]) -> (Vec<(usize, Vec<String>)>, Vec<String>) {
    let mut of_sheet: Vec<(usize, Vec<String>)> = sheets
        .iter()
        .map(|(source, text)| replay_sheet(text, source))
        .collect();
    let mut unheld = Vec::new();
    for module in &MODULES {
        let policy = module.policy;
        match sheets_of(policy, sheets).as_slice() {
            [] => unheld.push(format!(
                "no sheet of {policy}: its module replayed no recorded row"
            )),
            &[at] => {
                if let (Some((source, text)), Some((_, defects))) =
                    (sheets.get(at), of_sheet.get_mut(at))
                {
                    defects.extend(copied_row_defects(module.inline, text, source));
                }
            }
            more => unheld.push(format!(
                "{} sheets of {policy}: one sheet a policy",
                more.len()
            )),
        }
    }
    (of_sheet, unheld)
}

/// The whole replay's report over the sheets given, each a name and its
/// text. Its lines: one a sheet, in the order given (its name, its rows and
/// how many defects it answers for), then one a module of [`MODULES`]: the
/// policy with its one sheet's rows and defects, or that it has no sheet,
/// or how many. Its defects: what no one sheet answers for, then each
/// sheet's own. The ignored test prints the lines and fails on a defect.
/// The replay job reads the lines too and fails unless every policy line
/// counts rows and no defect, so a run that checks nothing cannot pass.
fn replay_report(sheets: &[(String, String)]) -> (Vec<String>, Vec<String>) {
    let (of_sheet, mut defects) = replay_sheets(sheets);
    let mut lines = Vec::with_capacity(sheets.len() + MODULES.len());
    for ((name, _), (rows, found)) in sheets.iter().zip(&of_sheet) {
        lines.push(format!("{name}: {rows} rows, {} defects", found.len()));
    }
    for module in &MODULES {
        let policy = module.policy;
        let held = sheets_of(policy, sheets);
        lines.push(match held.as_slice() {
            [] => format!("policy {policy}: no sheet"),
            &[at] => of_sheet.get(at).map_or_else(String::new, |(rows, found)| {
                format!("policy {policy}: {rows} rows, {} defects", found.len())
            }),
            more => format!("policy {policy}: {} sheets", more.len()),
        });
    }
    defects.extend(of_sheet.into_iter().flat_map(|(_, found)| found));
    (lines, defects)
}

/// The sheets of a directory, each as its path and its text, by path: every
/// file with the `json` extension, whatever its name, so a sheet nobody
/// expects is read and answered for. No other file is read: the recorder
/// and its notes live beside the sheets.
fn sheets_under(dir: &Path) -> Vec<(String, String)> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .expect("the sheets' directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            let text = fs::read_to_string(path).expect("a sheet is readable");
            (path.to_string_lossy().into_owned(), text)
        })
        .collect()
}

#[test]
fn the_classic_power_of_two_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::power_of_two::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_manual_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::manual::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// Every owed row is in an inline subset, so every run of the suite checks
/// that it is still refused, not the ignored test alone.
#[test]
fn every_owed_row_is_an_inline_row() {
    let inline = row_names::<classic::manual::Row>(classic::manual::INLINE);
    for owed in OWED {
        assert!(inline.iter().any(|name| name == owed.row), "{}", owed.row);
    }
}

/// A copied row is held to the sheet's row of its name: a row the sheet
/// lacks, another input, another number of draws, another sequence and
/// another distribution are one defect each, and the sheet the rows were
/// copied from is none.
#[test]
fn a_copied_row_is_held_to_the_sheets_row_of_its_name() {
    let inline = classic::manual::INLINE;
    assert_eq!(
        copied_row_defects(inline, inline, "sheet"),
        Vec::<String>::new()
    );
    let defect_of = |change: &dyn Fn(&mut Vec<Value>)| {
        let mut sheet: Value = serde_json::from_str(inline).expect("the inline sheet");
        change(
            sheet
                .get_mut("rows")
                .and_then(Value::as_array_mut)
                .expect("its rows"),
        );
        let mut defects = copied_row_defects(inline, &sheet.to_string(), "sheet");
        assert_eq!(defects.len(), 1, "{defects:#?}");
        defects.remove(0)
    };
    fn outcome<'a>(rows: &'a mut [Value], kind: &str) -> &'a mut Value {
        rows.iter_mut()
            .find_map(|row| row.get_mut("outcome")?.get_mut(kind))
            .expect("a row of that outcome")
    }
    assert!(defect_of(&|rows| drop(rows.remove(0))).contains("is on no sheet"));
    assert!(defect_of(&|rows| {
        let row = rows.first_mut().and_then(Value::as_object_mut);
        row.expect("a row")
            .insert("a field the copy lacks".to_string(), Value::from(1));
    })
    .contains("other inputs"));
    assert!(defect_of(&|rows| {
        let picks = outcome(rows, "sequence").as_array_mut();
        picks.expect("its picks").push(Value::from(0));
    })
    .contains("the sheet Sequence"));
    assert!(defect_of(&|rows| {
        let drawn = outcome(rows, "counts");
        let draws = drawn.get("draws").and_then(Value::as_u64).expect("draws");
        *drawn.get_mut("draws").expect("draws") = Value::from(draws + 1);
    })
    .contains("the sheet Counts"));
    // Every draw on the worker the copy picked least: a share of zero on one
    // side and not on the other.
    assert!(defect_of(&|rows| {
        let drawn = outcome(rows, "counts");
        let draws = drawn.get("draws").and_then(Value::as_u64).expect("draws");
        let counts = drawn.get_mut("counts").and_then(Value::as_array_mut);
        let counts = counts.expect("its counts");
        let least = (0..counts.len())
            .min_by_key(|&i| counts.get(i).and_then(Value::as_u64))
            .expect("a worker");
        for (i, count) in counts.iter_mut().enumerate() {
            *count = Value::from(if i == least { draws } else { 0 });
        }
    })
    .contains("the sheet Counts"));
}

/// The copied rows of every module as the sheets of a directory, each under
/// its policy's name.
fn copied_sheets() -> Vec<(String, String)> {
    MODULES
        .iter()
        .map(|module| (format!("{}.json", module.policy), module.inline.to_string()))
        .collect()
}

/// A policy with a module here and no sheet is a defect, and so is a second
/// sheet of one policy; one sheet a policy holds its module's copied rows.
#[test]
fn a_policy_with_no_sheet_is_a_defect_and_so_is_a_second_sheet() {
    let sheets = copied_sheets();
    let (of_sheet, unheld) = replay_sheets(&sheets);
    assert!(
        of_sheet
            .iter()
            .all(|(rows, defects)| *rows >= 8 && defects.is_empty()),
        "{of_sheet:#?}"
    );
    assert_eq!(unheld, Vec::<String>::new());
    let (_, unheld) = replay_sheets(sheets.get(..1).expect("the first sheet"));
    assert!(
        matches!(unheld.as_slice(), [one] if one.starts_with("no sheet of power_of_two")),
        "{unheld:#?}"
    );
    let mut twice = sheets.clone();
    twice.extend(sheets.first().cloned());
    let (_, unheld) = replay_sheets(&twice);
    assert!(
        matches!(unheld.as_slice(), [one] if one.starts_with("2 sheets of manual")),
        "{unheld:#?}"
    );
    // A sheet that lacks a copied row answers for it.
    let mut lacking = sheets.clone();
    for (_, text) in lacking.iter_mut().take(1) {
        let mut sheet: Value = serde_json::from_str(text).expect("the inline sheet");
        let rows = sheet.get_mut("rows").and_then(Value::as_array_mut);
        drop(rows.expect("its rows").remove(0));
        *text = sheet.to_string();
    }
    let (of_sheet, unheld) = replay_sheets(&lacking);
    assert_eq!(unheld, Vec::<String>::new());
    assert!(
        matches!(of_sheet.as_slice(), [(_, defects), (_, none)] if none.is_empty()
            && matches!(defects.as_slice(), [one] if one.contains("is on no sheet"))),
        "{of_sheet:#?}"
    );
}

/// A sheet of a policy no module replays is a defect, read alone and among
/// the sheets of a directory, where its line counts no row and the defect.
#[test]
fn a_sheet_of_a_policy_no_module_replays_is_a_defect() {
    let stray = r#"{"recorded_from": "a commit", "policy": "no_such_policy", "rows": []}"#;
    let (rows, defects) = replay_sheet(stray, "stray.json");
    assert_eq!(rows, 0);
    assert!(
        matches!(defects.as_slice(), [one]
            if one == "stray.json: a sheet of no_such_policy, which no module replays"),
        "{defects:#?}"
    );
    let mut sheets = copied_sheets();
    sheets.push(("stray.json".to_string(), stray.to_string()));
    let (lines, defects) = replay_report(&sheets);
    assert!(
        lines
            .iter()
            .any(|line| line == "stray.json: 0 rows, 1 defects"),
        "{lines:#?}"
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.contains("which no module replays")),
        "{defects:#?}"
    );
}

/// A sheet whose head names no commit it was recorded from is a defect by
/// that name, though every row of it replays.
#[test]
fn a_sheet_that_names_no_commit_is_a_defect() {
    let mut sheet: Value = serde_json::from_str(classic::manual::INLINE).expect("the inline sheet");
    let (rows, defects) = replay_sheet(&sheet.to_string(), "manual.json");
    assert!(rows >= 8, "{rows} inline rows");
    assert_eq!(defects, Vec::<String>::new(), "the copied sheet names one");
    *sheet.get_mut("recorded_from").expect("its commit") = Value::from("");
    let (unnamed, defects) = replay_sheet(&sheet.to_string(), "manual.json");
    assert_eq!(unnamed, rows, "every row replays all the same");
    assert_eq!(
        defects,
        ["manual.json: no commit the rows were recorded from"]
    );
}

/// The walk of the sheets' directory, as the ignored test reads it: every
/// `json` file is a sheet to answer for, whatever its name, and no other
/// file is read. Beside the copied sheets, each under its policy's name, a
/// sheet of a policy no module replays, a second sheet of one policy under
/// another name and a text that is no sheet are each refused by name.
#[test]
fn every_json_file_of_the_directory_is_a_sheet_to_answer_for() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let write = |name: &str, text: &str| {
        fs::write(dir.path().join(name), text).expect("a file is written");
    };
    for (name, text) in copied_sheets() {
        write(&name, &text);
    }
    // The recorder's own files stand beside the sheets and are not read.
    write("README.md", "how the rows are recorded");
    write("classic_rows.rs", "// the recorder");
    let (lines, defects) = replay_report(&sheets_under(dir.path()));
    assert_eq!(lines.len(), 2 * MODULES.len(), "{lines:#?}");
    assert_eq!(defects, Vec::<String>::new());

    write(
        "stray.json",
        r#"{"recorded_from": "a commit", "policy": "no_such_policy", "rows": []}"#,
    );
    write("a_second_name.json", classic::manual::INLINE);
    write("broken.json", "no sheet");
    let sheets = sheets_under(dir.path());
    let names: Vec<&str> = sheets
        .iter()
        .filter_map(|(path, _)| Path::new(path).file_name()?.to_str())
        .collect();
    assert_eq!(
        names,
        [
            "a_second_name.json",
            "broken.json",
            "manual.json",
            "power_of_two.json",
            "stray.json",
        ]
    );
    let (_, defects) = replay_report(&sheets);
    assert_eq!(defects.len(), 3, "{defects:#?}");
    for refused in [
        "2 sheets of manual",
        "broken.json is not a sheet of rows",
        "stray.json: a sheet of no_such_policy, which no module replays",
    ] {
        assert!(
            defects.iter().any(|defect| defect.contains(refused)),
            "{refused}: {defects:#?}"
        );
    }
}

/// The report the ignored test prints and judges by, over sheets built
/// here: a line a sheet, then a line a module of the table. The copied
/// sheets give clean lines and no defect. Each planted defect is refused by
/// name and shows in a line: a policy left with no sheet and a second sheet
/// of one policy in the policy's line, a text that is no sheet in its own
/// line, a recorded pick changed in its sheet's line and its policy's.
#[test]
fn the_report_holds_a_line_a_sheet_a_line_a_module_and_every_defect() {
    let sheets = copied_sheets();
    let (manual, _) = replay_sheet(classic::manual::INLINE, "the inline rows");
    let (power_of_two, _) = replay_sheet(classic::power_of_two::INLINE, "the inline rows");
    let (lines, defects) = replay_report(&sheets);
    assert_eq!(
        lines,
        [
            format!("manual.json: {manual} rows, 0 defects"),
            format!("power_of_two.json: {power_of_two} rows, 0 defects"),
            format!("policy manual: {manual} rows, 0 defects"),
            format!("policy power_of_two: {power_of_two} rows, 0 defects"),
        ]
    );
    assert_eq!(defects, Vec::<String>::new());

    let (lines, defects) = replay_report(sheets.get(..1).expect("the first sheet"));
    assert_eq!(
        lines,
        [
            format!("manual.json: {manual} rows, 0 defects"),
            format!("policy manual: {manual} rows, 0 defects"),
            "policy power_of_two: no sheet".to_string(),
        ]
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with("no sheet of power_of_two")),
        "{defects:#?}"
    );

    let mut twice = sheets.clone();
    twice.extend(sheets.first().cloned());
    let (lines, defects) = replay_report(&twice);
    assert_eq!(
        lines.get(3).map(String::as_str),
        Some("policy manual: 2 sheets")
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with("2 sheets of manual")),
        "{defects:#?}"
    );

    let mut broken = sheets.clone();
    broken.push(("broken.json".to_string(), "no sheet".to_string()));
    let (lines, defects) = replay_report(&broken);
    assert_eq!(
        lines.get(2).map(String::as_str),
        Some("broken.json: 0 rows, 1 defects")
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with("broken.json is not a sheet of rows")),
        "{defects:#?}"
    );

    // The first pick of a sequence row that is not owed, moved by one.
    let mut changed = sheets.clone();
    for (_, text) in changed.iter_mut().take(1) {
        let mut sheet: Value = serde_json::from_str(text).expect("the inline sheet");
        let rows = sheet.get_mut("rows").and_then(Value::as_array_mut);
        let pick = rows
            .expect("its rows")
            .iter_mut()
            .filter(|row| {
                let name = row.get("name").and_then(Value::as_str);
                OWED.iter().all(|owed| name != Some(owed.row))
            })
            .find_map(|row| row.get_mut("outcome")?.get_mut("sequence")?.get_mut(0))
            .expect("a sequence row");
        *pick = Value::from(pick.as_u64().expect("a pick") + 1);
        *text = sheet.to_string();
    }
    let (lines, defects) = replay_report(&changed);
    assert!(
        defects
            .iter()
            .any(|defect| defect.contains("recorded Sequence"))
            && defects
                .iter()
                .all(|defect| defect.starts_with("manual.json")),
        "{defects:#?}"
    );
    let found = defects.len();
    assert_eq!(
        lines,
        [
            format!("manual.json: {manual} rows, {found} defects"),
            format!("power_of_two.json: {power_of_two} rows, 0 defects"),
            format!("policy manual: {manual} rows, {found} defects"),
            format!("policy power_of_two: {power_of_two} rows, 0 defects"),
        ]
    );
}

/// Ignored by default because the sheets live in the test-data repository,
/// not here: `cargo test --features ingest --test classic_rows -- --ignored`
/// with `SMG_POLICY_CLASSIC_ROWS` naming their directory runs it. It fails
/// rather than passes when the variable is unset, when a policy with a
/// module here has no sheet in the directory, and when a row copied into a
/// module is not its sheet's row.
#[test]
#[ignore = "needs SMG_POLICY_CLASSIC_ROWS, the recorded sheets' directory of the test-data repository"]
#[expect(
    clippy::print_stdout,
    reason = "the per-sheet counts are the replay's report"
)]
fn every_recorded_classic_row_replays() {
    let dir = std::env::var_os("SMG_POLICY_CLASSIC_ROWS")
        .expect("SMG_POLICY_CLASSIC_ROWS names the recorded sheets' directory");
    let sheets = sheets_under(Path::new(&dir));
    assert!(
        !sheets.is_empty(),
        "no sheets under {}",
        dir.to_string_lossy()
    );
    let (lines, defects) = replay_report(&sheets);
    for line in &lines {
        println!("{line}");
    }
    assert!(defects.is_empty(), "{defects:#?}");
}
