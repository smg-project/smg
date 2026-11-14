//! The classic policies' outcomes, replayed through this crate's decision.
//! A row is a pool of workers, what a router knows of them, one request
//! shape and what the gateway's own policy code picked for it: the sequence
//! of picks when no draw decides, or how often each worker won over a number
//! of single decisions when one does. The rows are recorded by running that
//! policy code, not written by hand, and each sheet names the commit it was
//! recorded from. One module a classic policy holds its rows' shape and
//! their replay.
//!
//! The sheets live outside this repository, with the recorder:
//! `SMG_POLICY_CLASSIC_ROWS` names their directory for the ignored test. A
//! few rows of each form are copied into the policy's module from its
//! sheet, so every run of the suite replays a subset.
//!
//! A sequence row must replay pick for pick. A drawn row must give each
//! worker the recorded share within the spread of two samples of that size,
//! and a worker never picked on one side must never be picked on the other.
//! No row is excused.
#![expect(clippy::expect_used, reason = "a replay that stops on a broken setup")]

use std::{fs, path::PathBuf};

use serde::Deserialize;

mod classic {
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

/// `None` when a row's replay is its recorded outcome; the two otherwise.
fn verdict(recorded: &Outcome, here: &Outcome) -> Option<String> {
    let same = match (here, recorded) {
        (Outcome::Sequence(a), Outcome::Sequence(b)) => a == b,
        (
            Outcome::Counts { draws, counts },
            Outcome::Counts {
                counts: recorded, ..
            },
        ) => same_shares(&shares(counts, *draws), &shares(recorded, *draws), *draws),
        _ => false,
    };
    (!same).then(|| format!("recorded {recorded:?}, replayed {here:?}"))
}

/// A sheet's head: the commit its rows were recorded from, and the classic
/// policy they record, which names the module that replays them.
#[derive(Debug, Deserialize)]
struct Head {
    recorded_from: String,
    policy: String,
}

/// The number of rows replayed and what did not replay; a text that is no
/// sheet is one defect, and so is a sheet of a policy no module replays.
fn replay_sheet(text: &str, source: &str) -> (usize, Vec<String>) {
    let head: Head = match serde_json::from_str(text) {
        Ok(head) => head,
        Err(err) => return (0, vec![format!("{source} is not a sheet of rows: {err}")]),
    };
    let (rows, mut defects) = match head.policy.as_str() {
        classic::power_of_two::POLICY => classic::power_of_two::replay_rows(text, source),
        other => (
            0,
            vec![format!(
                "{source}: a sheet of {other}, which no module replays"
            )],
        ),
    };
    if head.recorded_from.is_empty() {
        defects.push(format!("{source}: no commit the rows were recorded from"));
    }
    (rows, defects)
}

#[test]
fn the_classic_power_of_two_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::power_of_two::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// Ignored by default because the sheets live in the test-data repository,
/// not here: `cargo test --test classic_rows -- --ignored` with
/// `SMG_POLICY_CLASSIC_ROWS` naming their directory runs it, and it fails
/// rather than passes when the variable is unset.
#[test]
#[ignore = "needs SMG_POLICY_CLASSIC_ROWS, the recorded sheets' directory of the test-data repository"]
#[expect(
    clippy::print_stdout,
    reason = "the per-sheet counts are the replay's report"
)]
fn every_recorded_classic_row_replays() {
    let dir = std::env::var_os("SMG_POLICY_CLASSIC_ROWS")
        .expect("SMG_POLICY_CLASSIC_ROWS names the recorded sheets' directory");
    let mut sheets: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("the sheets' directory is readable")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    sheets.sort();
    assert!(
        !sheets.is_empty(),
        "no sheets under {}",
        dir.to_string_lossy()
    );
    let mut defects = Vec::new();
    for path in &sheets {
        let name = path.to_string_lossy();
        let text = fs::read_to_string(path).expect("a sheet is readable");
        let (rows, found) = replay_sheet(&text, &name);
        println!("{name}: {rows} rows, {} defects", found.len());
        defects.extend(found);
    }
    assert!(defects.is_empty(), "{defects:#?}");
}
