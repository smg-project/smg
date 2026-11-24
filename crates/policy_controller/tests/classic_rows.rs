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
//! The sheets live beside this target, in `tests/classic/sheets`, with the
//! notes on their rows, and each recorder is a test module of the gateway
//! crate beside the code it runs. Every run of the suite with the
//! `in-process` and `worker-host` features replays every row of each sheet
//! that has a
//! module here. A few rows of each form are copied into the policy's module
//! from its sheet as well, and the replay holds every copied row to the
//! sheet's row of its name. A row's reports reach the view through the
//! crate's row builder, and the three hard-demand modules decide
//! over worker objects, which `in-process` brings: a bare test run of the
//! crate builds no part of this target and replays no recorded row.
//!
//! A sequence row must replay pick for pick. A drawn row must give each
//! worker the recorded share within the spread of two samples of that size,
//! and a worker never picked on one side must never be picked on the other.
//! A pair-index row picks nothing: it holds what the gateway's pair index
//! made of a prefill pool and a decode pool, and must replay whole.
//! An encode-pair row holds the gateway's answers to requests that carry
//! encode items. The store's part of each answer replays: the encoders a
//! prefill is handed items by.
//! A sticky rank row holds the rank each request of one grouped worker was
//! pinned to by its key. Two hashes place no key alike, so it must replay
//! with the requests of one key on one rank, and a counted row count for
//! count or share for share.
//! An overload row picks nothing: it holds whether the gateway's load
//! monitor read one worker as overloaded after each write of its loads, and
//! must replay verdict for verdict through that monitor, built as a host of
//! the engine over worker lists builds it, which `worker-host` brings.
//! A minimum-tokens row holds the rank the load monitor gave each request
//! on one worker. It must replay rank for rank through the rank cache of
//! that monitor, which is the worker crate's own rank choice.
//! A retry-wire row, a media row and a context-window row hold the
//! gateway's answer to each request of one pool, a retry that stays on its
//! wire, a request that carries media and a request of some size among
//! them: the workers the request may be placed on, or its refusal. Each
//! must replay answer for answer, a refusal by its status and its code.
//! A pair-select row holds the gateway's answers to the requests of a
//! prefill pool and a decode pool: the pair, or no pair and why. It must
//! replay answer for answer, and a drawn row pair for pair and share for
//! share. The pair-select module holds five recorded rows the pair's
//! decision answers apart to the answer each entry gives.
//! A pair-bound row is a pair-select row behind the prefill admission,
//! where a prefill that holds the admission's bound has an open partner or
//! has none. Its sheet is its own and its replay the pair-select module's,
//! answer for answer.
//! A pair-runtime row is a pair-select row on the wire that carries one
//! runtime's request, where the workers of one known runtime stand beside
//! workers of another or of none. Its sheet is its own too, and its replay
//! the pair-select module's, answer for answer.
//! No row is decided differently but where its module names it. The prefix
//! hash module holds a recorded row that a key of whole blocks cannot
//! decide as recorded to the outcome its entry gives. The sticky rank
//! module holds a row on which the gateway sets no rank to a rank of the
//! worker for every request.
//! The overload module replays a row of several ranks a second time with
//! one worker object for each rank, where the load monitor judges each
//! object by its own rank's figures, and holds the rows that answer apart
//! there to the verdicts its entries give.
//! The minimum-tokens module replays no row through the engine: the engine
//! serves no rank option, and where each rank is a worker object it picks
//! among them by its profile, which these rows do not record.
//! A row whose form this crate refuses to build is a defect. A sheet no
//! module here replays is owed: [`OWED_SHEETS`] names its policy, the rows
//! it holds and the change that brings the module, and the suite fails
//! once a module replays that policy.
#![expect(clippy::expect_used, reason = "a replay that stops on a broken setup")]

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{Map, Value};

mod classic {
    pub mod all_vetoed;
    pub mod bucket;
    pub mod cache_aware;
    pub mod context_window;
    pub mod demand_bed;
    pub mod encode_pair;
    pub mod load_bed;
    pub mod manual;
    pub mod media_refs;
    pub mod minimum_tokens;
    pub mod overload;
    pub mod pair_bound;
    pub mod pair_index;
    pub mod pair_runtime;
    pub mod pair_select;
    pub mod power_of_two;
    pub mod prefix_hash;
    pub mod retry_wire;
    pub mod sticky_override;
    pub mod sticky_rank;
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

/// A sheet of the directory no module here replays: `policy` is its head's,
/// `rows` how many rows it holds, and `by` names the change that brings the
/// module. The sheet replays no row and answers for no defect while no
/// module replays its policy, and the suite fails the day one does, so an
/// entry cannot outlive its debt; the whole replay holds the directory to
/// one sheet of each entry's policy and that sheet to the entry's count of
/// rows ([`owed_sheet_defects`]), so no owed row comes or goes unseen.
struct OwedSheet {
    policy: &'static str,
    rows: usize,
    by: &'static str,
}

/// The owed sheets: one, recorded before the change that replays
/// it landed here.
const OWED_SHEETS: &[OwedSheet] = &[OwedSheet {
    policy: "constraint-decider",
    rows: 5,
    by: "a request's hard demands: the constraint's decider on a retry",
}];

/// The owed sheet of a policy, when one is owed.
fn owed_sheet(policy: &str) -> Option<&'static OwedSheet> {
    OWED_SHEETS.iter().find(|owed| owed.policy == policy)
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

/// `None` when a row's replay is its recorded outcome; the reason
/// otherwise, a form this crate refuses to build among them.
fn verdict(recorded: &Outcome, replayed: &Replayed) -> Option<String> {
    match replayed {
        Replayed::Refused(why) => Some(format!("refused here: {why}")),
        Replayed::Ran(here) => (!same_outcome(recorded, here))
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
            verdict(row.outcome(), &row.replay())
                .map(|why| format!("{source}: {}: {why}", row.name()))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// A sheet's head: the commit its rows were recorded from, and the classic
/// policy they record, which names the module that replays them.
#[derive(Debug, Deserialize)]
struct Head {
    recorded_from: String,
    policy: String,
}

/// A classic policy a module here replays: the name a sheet's head gives
/// the policy, the rows copied from its sheet, the replay of a sheet's text
/// in the module's row shape, and what separates the copied rows from the
/// sheet they were copied from, by the rule of the module's rows.
struct Module {
    policy: &'static str,
    inline: &'static str,
    replay: fn(&str, &str) -> (usize, Vec<String>),
    copied: fn(&str, &str, &str) -> Vec<String>,
}

/// The entry of the module whose rows are `R`, rows that pick: the policy's
/// name and the replay come from the one type, so no entry pairs two
/// modules, and a copied row is held by the picks' own rule.
const fn module<R: Recorded>(inline: &'static str) -> Module {
    Module {
        policy: R::POLICY,
        inline,
        replay: replay_rows::<R>,
        copied: copied_row_defects,
    }
}

/// Every module that replays, the one list. A sheet finds its module here
/// ([`replay_sheet`]), and each entry is held to one sheet and to its copied
/// rows ([`replay_sheets`]): a module added here has both. The pair
/// index's rows pick nothing, so its module brings its own replay and its
/// own rule for a copied row. The prefix hash module brings its own replay
/// too, which holds the rows it names apart to its entries. A rank row
/// names a rank or none, so the sticky rank module brings its own replay
/// as well, and holds a copied row whole ([`whole_row_defects`]), as the
/// pair-select module does.
/// An overload row picks nothing either: its module brings its own replay,
/// over a host's load monitor, and holds a copied row whole too.
/// A minimum-tokens row names a rank or none: its module brings its own
/// replay, over the rank cache of that monitor, and holds a copied row whole.
/// An encode-pair row is replayed in its store part by its own module;
/// its copied rows name no runtime either, and the pair index's rule
/// holds them.
/// A retry-wire row, a media row and a context-window row name the
/// workers a request may take or its refusal, so each module brings its
/// own replay too, over one bed; their copied rows name each runtime by a
/// token, or name none, and the pair index's rule holds them as well.
/// A pair-bound row and a pair-runtime row have the pair-select row's
/// shape: each module names its own sheet and replays it by the pair-select
/// module's replay, and holds a copied row whole.
const MODULES: [Module; 18] = [
    module::<classic::all_vetoed::Row>(classic::all_vetoed::INLINE),
    module::<classic::bucket::Row>(classic::bucket::INLINE),
    module::<classic::cache_aware::Row>(classic::cache_aware::INLINE),
    Module {
        policy: classic::context_window::POLICY,
        inline: classic::context_window::INLINE,
        replay: classic::context_window::replay_rows,
        copied: classic::pair_index::copied_row_defects,
    },
    Module {
        policy: classic::encode_pair::POLICY,
        inline: classic::encode_pair::INLINE,
        replay: classic::encode_pair::replay_rows,
        copied: classic::pair_index::copied_row_defects,
    },
    module::<classic::manual::Row>(classic::manual::INLINE),
    Module {
        policy: classic::media_refs::POLICY,
        inline: classic::media_refs::INLINE,
        replay: classic::media_refs::replay_rows,
        copied: classic::pair_index::copied_row_defects,
    },
    Module {
        policy: classic::minimum_tokens::POLICY,
        inline: classic::minimum_tokens::INLINE,
        replay: classic::minimum_tokens::replay_rows,
        copied: whole_row_defects,
    },
    Module {
        policy: classic::overload::POLICY,
        inline: classic::overload::INLINE,
        replay: classic::overload::replay_rows,
        copied: whole_row_defects,
    },
    Module {
        policy: classic::pair_bound::POLICY,
        inline: classic::pair_bound::INLINE,
        replay: classic::pair_bound::replay_rows,
        copied: whole_row_defects,
    },
    Module {
        policy: classic::pair_index::POLICY,
        inline: classic::pair_index::INLINE,
        replay: classic::pair_index::replay_rows,
        copied: classic::pair_index::copied_row_defects,
    },
    Module {
        policy: classic::pair_runtime::POLICY,
        inline: classic::pair_runtime::INLINE,
        replay: classic::pair_runtime::replay_rows,
        copied: whole_row_defects,
    },
    Module {
        policy: classic::pair_select::POLICY,
        inline: classic::pair_select::INLINE,
        replay: classic::pair_select::replay_rows,
        copied: whole_row_defects,
    },
    module::<classic::power_of_two::Row>(classic::power_of_two::INLINE),
    Module {
        policy: classic::prefix_hash::POLICY,
        inline: classic::prefix_hash::INLINE,
        replay: classic::prefix_hash::replay_rows,
        copied: copied_row_defects,
    },
    Module {
        policy: classic::retry_wire::POLICY,
        inline: classic::retry_wire::INLINE,
        replay: classic::retry_wire::replay_rows,
        copied: classic::pair_index::copied_row_defects,
    },
    module::<classic::sticky_override::Row>(classic::sticky_override::INLINE),
    Module {
        policy: classic::sticky_rank::POLICY,
        inline: classic::sticky_rank::INLINE,
        replay: classic::sticky_rank::replay_rows,
        copied: whole_row_defects,
    },
];

/// The number of rows replayed and what did not replay; a text that is no
/// sheet is one defect, and so is a sheet of a policy no module replays
/// unless [`OWED_SHEETS`] names it.
fn replay_sheet(text: &str, source: &str) -> (usize, Vec<String>) {
    let head: Head = match serde_json::from_str(text) {
        Ok(head) => head,
        Err(err) => return (0, vec![format!("{source} is not a sheet of rows: {err}")]),
    };
    let module = MODULES.iter().find(|module| module.policy == head.policy);
    let (rows, mut defects) = match module {
        Some(module) => (module.replay)(text, source),
        None if owed_sheet(&head.policy).is_some() => (0, Vec::new()),
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

/// What separates the rows copied into a module from the sheet they were
/// copied from, where nothing in a row is drawn: a copied row the sheet
/// does not hold, or one that is not the sheet's row of its name, whole.
fn whole_row_defects(inline: &str, sheet: &str, source: &str) -> Vec<String> {
    let rows = |text: &str, source: &str| {
        let rows = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|mut sheet| sheet.get_mut("rows").map(Value::take));
        match rows {
            Some(Value::Array(rows)) => Ok(rows),
            _ => Err(format!("{source} is not a sheet of rows")),
        }
    };
    let (copied, held) = match (rows(inline, "the inline rows"), rows(sheet, source)) {
        (Ok(copied), Ok(held)) => (copied, held),
        (Err(why), _) | (_, Err(why)) => return vec![why],
    };
    copied
        .iter()
        .filter_map(|row| {
            let name = row.get("name").and_then(Value::as_str).unwrap_or("?");
            let Some(sheet_row) = held.iter().find(|held| held.get("name") == row.get("name"))
            else {
                return Some(format!("{source}: the inline row {name} is on no sheet"));
            };
            (sheet_row != row)
                .then(|| format!("{source}: the inline row {name} is not the sheet's row"))
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

/// What the directory owes the owed sheets: an entry of [`OWED_SHEETS`]
/// with no sheet of its policy among them, or with more than one, or whose
/// one sheet holds another number of rows than the entry counts. An entry
/// whose sheet left the directory would otherwise stand with nothing to
/// answer for, and a row could join or leave a sheet nothing replays.
fn owed_sheet_defects(sheets: &[(String, String)]) -> Vec<String> {
    OWED_SHEETS
        .iter()
        .filter_map(|owed| match sheets_of(owed.policy, sheets).as_slice() {
            &[at] => {
                let (_, text) = sheets.get(at)?;
                let held =
                    serde_json::from_str::<Written>(text).map_or(0, |sheet| sheet.rows.len());
                (held != owed.rows).then(|| {
                    format!(
                        "the sheet of {} holds {held} rows, its entry counts {}",
                        owed.policy, owed.rows
                    )
                })
            }
            [] => Some(format!(
                "no sheet of {}: an owed sheet stays in the directory",
                owed.policy
            )),
            more => Some(format!(
                "{} sheets of {}: one sheet a policy",
                more.len(),
                owed.policy
            )),
        })
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
                    defects.extend((module.copied)(module.inline, text, source));
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
/// sheet's own. The whole replay prints the lines and fails on a defect
/// and on a policy line that counts no row, so a run that replays nothing
/// cannot pass.
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
    for (name, text) in sheets {
        let owed = serde_json::from_str::<Head>(text)
            .ok()
            .and_then(|head| owed_sheet(&head.policy));
        if let Some(owed) = owed {
            lines.push(format!(
                "owed {}: {name} waits for {} ({} rows)",
                owed.policy, owed.by, owed.rows
            ));
        }
    }
    defects.extend(of_sheet.into_iter().flat_map(|(_, found)| found));
    (lines, defects)
}

/// The sheets of a directory, each as its path and its text, by path: every
/// file with the `json` extension, whatever its name, so a sheet nobody
/// expects is read and answered for. No other file is read: the notes on
/// the rows live beside the sheets.
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
fn the_classic_bucket_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::bucket::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_power_of_two_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::power_of_two::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_all_vetoed_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::all_vetoed::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// The ring rows of the all-vetoed sheet list their workers in their key's
/// ring order. Each is replayed a second time on a ring the request's
/// caller lends, built in that order, and holds its recorded answer.
#[test]
fn the_classic_ring_rows_replay_on_a_ring_their_caller_lends() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/classic/sheets");
    let sheet = fs::read_to_string(dir.join("all_vetoed.json")).expect("the sheet is readable");
    for (text, least) in [(classic::all_vetoed::INLINE, 8), (sheet.as_str(), 12)] {
        let (rows, defects) = classic::all_vetoed::replay_on_a_lent_ring(text);
        assert!(rows >= least, "{rows} ring rows, under {least}");
        assert!(defects.is_empty(), "{defects:#?}");
    }
}

#[test]
fn the_classic_encode_pair_rows_replay_the_stores_part_list_for_list() {
    let (rows, defects) = replay_sheet(classic::encode_pair::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_manual_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::manual::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_prefix_hash_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::prefix_hash::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// The prefix hash rows that say where their requests stand on a ring are
/// replayed a second time on a ring their caller lends, built from each
/// row, and hold their recorded answers: the rows that list their workers
/// in a request's ring order, the level rows and the row with nothing to
/// hash. The pairs rows and the spread row are not among them: on a lent
/// ring which tokens a key holds, and how keys spread, is its caller's.
#[test]
fn the_classic_prefix_hash_rows_replay_on_a_ring_their_caller_lends() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/classic/sheets");
    let sheet = fs::read_to_string(dir.join("prefix_hash.json")).expect("the sheet is readable");
    for (text, rows) in [(classic::prefix_hash::INLINE, 19), (sheet.as_str(), 31)] {
        let (replayed, defects) = classic::prefix_hash::replay_on_a_lent_ring(text);
        assert_eq!(replayed, rows, "the rows replayed on a lent ring");
        assert!(defects.is_empty(), "{defects:#?}");
    }
}

/// The same rows through the one call a gateway asks, over worker objects:
/// the caller lends each row's ring and states no place. The engine cuts a
/// request into the heads the gateway's own policy cuts it into and asks
/// the lender where each stands, and every request goes to its recorded
/// worker. That the lender's own recipe for a head is the gateway's is
/// proven where a gateway lends its ring, not here.
#[cfg(feature = "in-process")]
#[test]
fn the_classic_prefix_hash_rows_replay_through_the_one_call_with_no_place_stated() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/classic/sheets");
    let sheet = fs::read_to_string(dir.join("prefix_hash.json")).expect("the sheet is readable");
    for (text, rows) in [(classic::prefix_hash::INLINE, 19), (sheet.as_str(), 31)] {
        let (replayed, defects) = classic::prefix_hash::through_the_one_call::replay(text);
        assert_eq!(replayed, rows, "the rows replayed through the one call");
        assert!(defects.is_empty(), "{defects:#?}");
    }
}

#[test]
fn the_classic_sticky_override_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::sticky_override::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_cache_aware_rows_replay_pick_for_pick_and_share_for_share() {
    let (rows, defects) = replay_sheet(classic::cache_aware::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_sticky_rank_rows_replay_rank_for_rank_and_count_for_count() {
    let (rows, defects) = replay_sheet(classic::sticky_rank::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_overload_rows_replay_verdict_for_verdict() {
    let (rows, defects) = replay_sheet(classic::overload::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_minimum_tokens_rows_replay_rank_for_rank() {
    let (rows, defects) = replay_sheet(classic::minimum_tokens::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_media_rows_replay_answer_for_answer() {
    let (rows, defects) = replay_sheet(classic::media_refs::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_context_window_rows_replay_answer_for_answer() {
    let (rows, defects) = replay_sheet(classic::context_window::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

#[test]
fn the_classic_retry_wire_rows_replay_answer_for_answer() {
    let (rows, defects) = replay_sheet(classic::retry_wire::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// A sheet whose module has not landed is owed, not a defect: it replays no
/// row, the report says what it waits for, and it still names the commit
/// its rows were recorded from. No entry names a policy a module replays.
#[test]
fn a_sheet_owed_its_module_replays_no_row_and_is_no_defect() {
    for owed in OWED_SHEETS {
        assert!(
            MODULES.iter().all(|module| module.policy != owed.policy),
            "{} has a module now: take it out of OWED_SHEETS",
            owed.policy
        );
        let text = format!(
            r#"{{"recorded_from": "a commit", "policy": "{}", "rows": []}}"#,
            owed.policy
        );
        assert_eq!(replay_sheet(&text, "owed.json"), (0, Vec::new()));
    }
    // The table's first entry, whichever sheet it is: a module that lands
    // takes its entry out and moves no line of this case.
    let Some(owed) = OWED_SHEETS.first() else {
        return;
    };
    let mut sheets = copied_sheets();
    sheets.push((
        "owed.json".to_string(),
        format!(
            r#"{{"recorded_from": "a commit", "policy": "{}", "rows": []}}"#,
            owed.policy
        ),
    ));
    let (lines, defects) = replay_report(&sheets);
    assert_eq!(defects, Vec::<String>::new());
    assert!(
        lines
            .iter()
            .any(|line| line == "owed.json: 0 rows, 0 defects"),
        "{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with(&format!("owed {}: owed.json waits for ", owed.policy))),
        "{lines:#?}"
    );
    let unnamed = format!(
        r#"{{"recorded_from": "", "policy": "{}", "rows": []}}"#,
        owed.policy
    );
    let (_, defects) = replay_sheet(&unnamed, "owed.json");
    assert!(
        matches!(defects.as_slice(), [one] if one.contains("no commit")),
        "{defects:#?}"
    );
}

#[test]
fn the_classic_pair_index_rows_replay_verdict_for_verdict_and_list_for_list() {
    let (rows, defects) = replay_sheet(classic::pair_index::INLINE, "the inline rows");
    assert!(rows >= 8, "{rows} inline rows");
    assert!(defects.is_empty(), "{defects:#?}");
}

/// Every entry of the owed sheets has its one sheet in the directory, of
/// the rows the entry counts: an entry whose sheet is gone is a defect, and
/// so are a second sheet of its policy and a sheet of another count.
#[test]
fn an_owed_policy_with_no_sheet_is_a_defect_and_so_is_a_second_sheet() {
    let sheet = |owed: &OwedSheet, rows: usize, name: &str| {
        let rows = vec!["{}"; rows].join(", ");
        let policy = owed.policy;
        let text =
            format!(r#"{{"recorded_from": "a commit", "policy": "{policy}", "rows": [{rows}]}}"#);
        (name.to_string(), text)
    };
    let mut sheets: Vec<(String, String)> = OWED_SHEETS
        .iter()
        .map(|owed| sheet(owed, owed.rows, owed.policy))
        .collect();
    assert_eq!(owed_sheet_defects(&sheets), Vec::<String>::new());
    let Some(first) = OWED_SHEETS.first() else {
        return;
    };
    // A row more on an owed sheet is counted.
    let mut grown = sheets.clone();
    if let Some(one) = grown.first_mut() {
        *one = sheet(first, first.rows + 1, first.policy);
    }
    let defects = owed_sheet_defects(&grown);
    let counted = format!(
        "the sheet of {} holds {} rows, its entry counts {}",
        first.policy,
        first.rows + 1,
        first.rows
    );
    assert_eq!(defects, [counted]);
    sheets.push(sheet(first, first.rows, "again.json"));
    let defects = owed_sheet_defects(&sheets);
    let twice = format!("2 sheets of {}", first.policy);
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with(&twice)),
        "{defects:#?}"
    );
    sheets.retain(|(name, _)| name != first.policy && name != "again.json");
    let defects = owed_sheet_defects(&sheets);
    let none = format!("no sheet of {}", first.policy);
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with(&none)),
        "{defects:#?}"
    );
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

/// A copied pair row names no runtime, transport or layout value: it is its
/// sheet's row under one renaming of those names, one to one and in the
/// sheet's order. The copy against itself and against a sheet whose names
/// are others in the same order is no defect; a row the sheet lacks, a
/// verdict's word, a partner list, two names made one and two names in the
/// other order are one defect each.
#[test]
fn a_copied_pair_row_is_its_sheets_row_under_one_renaming() {
    let inline = classic::pair_index::INLINE;
    let held = classic::pair_index::copied_row_defects;
    assert_eq!(held(inline, inline, "sheet"), Vec::<String>::new());
    // Other names in the same order: every `tx-a` and `tx-b` of the sheet.
    let renamed = inline
        .replace("tx-a", "m-first")
        .replace("tx-b", "n-second");
    assert_ne!(renamed, inline);
    assert_eq!(held(inline, &renamed, "sheet"), Vec::<String>::new());
    let defects_of = |sheet: &str| held(inline, sheet, "sheet");
    let edited = |change: &dyn Fn(&mut Vec<Value>)| {
        let mut sheet: Value = serde_json::from_str(inline).expect("the inline sheet");
        change(
            sheet
                .get_mut("rows")
                .and_then(Value::as_array_mut)
                .expect("its rows"),
        );
        sheet.to_string()
    };
    let one = |defects: Vec<String>| {
        assert_eq!(defects.len(), 1, "{defects:#?}");
        defects.into_iter().next().unwrap_or_default()
    };
    assert!(one(defects_of(&edited(&|rows| drop(rows.remove(0))))).contains("is on no sheet"));
    // One verdict's word, then one partner more, in the first row's strict
    // outcome on the gRPC wire.
    assert!(one(defects_of(&edited(&|rows| {
        let verdict = rows
            .first_mut()
            .and_then(|row| row.pointer_mut("/outcome/grpc/strict/verdicts/0/0"));
        *verdict.expect("the first verdict") = Value::from("version");
    })))
    .contains("under one renaming"));
    assert!(one(defects_of(&edited(&|rows| {
        let partners = rows
            .first_mut()
            .and_then(|row| row.pointer_mut("/outcome/grpc/strict/partners/0"))
            .and_then(Value::as_array_mut);
        partners
            .expect("the first partner list")
            .push(Value::from(9));
    })))
    .contains("under one renaming"));
    // Two names made one in the first row; then the two in the other order
    // on the whole sheet, a defect of every row that holds both.
    let merged = edited(&|rows| {
        let first = rows.first_mut().expect("a row");
        let text = first.to_string().replace("\"tx-a\"", "\"tx-b\"");
        *first = serde_json::from_str(&text).expect("a row again");
    });
    assert!(one(defects_of(&merged)).contains("under one renaming"));
    let swapped = inline
        .replace("tx-a", "tx-c")
        .replace("tx-b", "tx-a")
        .replace("tx-c", "tx-b");
    assert!(defects_of(&swapped)
        .iter()
        .all(|defect| defect.contains("under one renaming")));
    assert!(!defects_of(&swapped).is_empty());
}

/// A row copied whole is its sheet's row whole: the copy against itself is
/// no defect, and a row the sheet lacks, another input and another outcome
/// are one defect each.
#[test]
fn a_row_copied_whole_is_its_sheets_row_whole() {
    let inline = classic::sticky_rank::INLINE;
    assert_eq!(
        whole_row_defects(inline, inline, "sheet"),
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
        let mut defects = whole_row_defects(inline, &sheet.to_string(), "sheet");
        assert_eq!(defects.len(), 1, "{defects:#?}");
        defects.remove(0)
    };
    assert!(defect_of(&|rows| drop(rows.remove(0))).contains("is on no sheet"));
    let first = |rows: &mut Vec<Value>, field: &str| {
        let row = rows.first_mut().and_then(Value::as_object_mut);
        row.expect("the first row")
            .insert(field.to_string(), Value::from("another"));
    };
    assert!(defect_of(&|rows| first(rows, "name_of_an_input")).contains("is not the sheet's row"));
    assert!(defect_of(&|rows| first(rows, "outcome")).contains("is not the sheet's row"));
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
    // The table's first and last policies, whichever modules it holds.
    let first = MODULES.first().expect("a module").policy;
    let last = MODULES.last().expect("a module").policy;
    let (_, kept) = sheets.split_last().expect("a sheet a module");
    let (_, unheld) = replay_sheets(kept);
    assert!(
        matches!(unheld.as_slice(), [one] if one.starts_with(&format!("no sheet of {last}"))),
        "{unheld:#?}"
    );
    let mut twice = sheets.clone();
    twice.extend(sheets.first().cloned());
    let (_, unheld) = replay_sheets(&twice);
    assert!(
        matches!(unheld.as_slice(), [one] if one.starts_with(&format!("2 sheets of {first}"))),
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
        matches!(of_sheet.as_slice(), [(_, defects), others @ ..]
            if others.iter().all(|(_, none)| none.is_empty())
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

/// An encode-pair row reads red on a list that is not the store's and on
/// an answer that is not the decision's. An offered list that leaves the
/// class's ready encoders out is a defect of the store's part, and so is an
/// encoder offered to a request that carries no item. An offered encoder
/// that is not ready, and a permanent gap's refusal beside a class that
/// serves the request, are a defect of each part. A copied row is its
/// sheet's row under one renaming of its runtimes: other names in the same
/// order are no defect, two names made one are.
#[test]
fn an_encode_pair_row_is_held_to_the_stores_lists_and_the_decision() {
    let inline = classic::encode_pair::INLINE;
    let defects_of = |name: &str, change: &dyn Fn(&mut Value)| {
        let mut sheet: Value = serde_json::from_str(inline).expect("the inline sheet");
        let rows = sheet.get_mut("rows").and_then(Value::as_array_mut);
        let row = rows
            .expect("its rows")
            .iter_mut()
            .find(|row| row.get("name").and_then(Value::as_str) == Some(name));
        change(row.expect("the row of that name"));
        let (_, defects) = replay_sheet(&sheet.to_string(), "sheet");
        defects
    };
    let placed = "two-runtimes-the-items-go-to-the-encoders-of-the-pushing-pair";
    let unoffered = defects_of(placed, &|row| {
        let offered = row.pointer_mut("/outcome/sequence/0/placed/offered");
        *offered.expect("the offered encoders") = Value::Array(Vec::new());
    });
    assert!(
        matches!(unoffered.as_slice(), [one] if one.contains("was offered the encoders []")),
        "{unoffered:#?}"
    );
    let unready = defects_of(placed, &|row| {
        let open = row.pointer_mut("/encoders/1/open");
        *open.expect("the second encoder") = Value::from(false);
    });
    assert!(
        matches!(unready.as_slice(), [store, decision]
            if store.contains("of which [0] are ready")
                && decision.contains("replayed [0, 0, 0]")),
        "{unready:#?}"
    );
    let refused = "push-a-producer-and-a-consumer-on-two-connectors-are-refused";
    let served = defects_of(refused, &|row| {
        let connector = row.pointer("/encoders/0/connector").cloned();
        let of_prefill = row.pointer_mut("/prefills/0/connector");
        *of_prefill.expect("the prefill's connector") = connector.expect("the encoder's");
    });
    assert!(
        matches!(served.as_slice(), [store, decision]
            if store.contains("yet the class of prefill 0 holds encoder 0")
                && decision.contains("refused as ec_incompatible, replayed Pick(")),
        "{served:#?}"
    );
    let bare = "an-encoder-whose-runtime-states-no-handoff-serves-no-item";
    let read = defects_of(bare, &|row| {
        let offered = row.pointer_mut("/outcome/sequence/2/placed/offered");
        *offered.expect("the offered encoders") = Value::from(vec![0]);
    });
    assert!(
        matches!(read.as_slice(), [one] if one.contains("carries no item")),
        "{read:#?}"
    );

    let held = classic::pair_index::copied_row_defects;
    assert_eq!(held(inline, inline, "sheet"), Vec::<String>::new());
    let renamed = inline.replace("\"sigma\"", "\"m-second\"");
    assert_ne!(renamed, inline);
    assert_eq!(held(inline, &renamed, "sheet"), Vec::<String>::new());
    let merged = inline.replace("\"sigma\"", "\"zeta\"");
    let defects = held(inline, &merged, "sheet");
    assert!(!defects.is_empty());
    assert!(defects
        .iter()
        .all(|defect| defect.contains("under one renaming")));
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

/// The walk of the sheets' directory, as the whole replay reads it: every
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
    // The notes on the rows and a file of another kind stand beside the
    // sheets and are not read.
    write("README.md", "how the rows are recorded");
    write("rows.txt", "no sheet: not a json file");
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
            "all_vetoed.json",
            "broken.json",
            "bucket.json",
            "cache_aware.json",
            "context-window.json",
            "encode_pair.json",
            "manual.json",
            "media-refs.json",
            "minimum_tokens.json",
            "overload.json",
            "pair-bound.json",
            "pair-index.json",
            "pair-runtime.json",
            "pair-select.json",
            "power_of_two.json",
            "prefix_hash.json",
            "retry-wire.json",
            "sticky_override.json",
            "sticky_rank.json",
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

/// The report the whole replay prints and judges by, over sheets built
/// here: a line a sheet, then a line a module of the table. The copied
/// sheets give clean lines and no defect. Each planted defect is refused by
/// name and shows in a line: a policy left with no sheet and a second sheet
/// of one policy in the policy's line, a text that is no sheet in its own
/// line, a recorded pick changed in its sheet's line and its policy's.
#[test]
fn the_report_holds_a_line_a_sheet_a_line_a_module_and_every_defect() {
    let sheets = copied_sheets();
    // Each module's policy and the rows of its copied sheet, in the table's
    // order: a module more moves no line of this case.
    let counted: Vec<(&str, usize)> = MODULES
        .iter()
        .map(|module| {
            let (rows, _) = replay_sheet(module.inline, "the inline rows");
            (module.policy, rows)
        })
        .collect();
    let (first, _) = *counted.first().expect("a module");
    let (last, kept_counted) = counted.split_last().expect("a module");
    // The sheet this case changes last is the manual policy's, wherever the
    // table holds it.
    let manual_at = counted
        .iter()
        .position(|(policy, _)| *policy == "manual")
        .expect("the manual module");
    // The report's lines over the copied sheets, that sheet answering
    // for `found` defects and every other for none.
    let lines_with = |found: usize| -> Vec<String> {
        let of = |at: usize| if at == manual_at { found } else { 0 };
        let sheet_lines = counted
            .iter()
            .enumerate()
            .map(|(at, (policy, rows))| format!("{policy}.json: {rows} rows, {} defects", of(at)));
        let policy_lines = counted.iter().enumerate().map(|(at, (policy, rows))| {
            format!("policy {policy}: {rows} rows, {} defects", of(at))
        });
        sheet_lines.chain(policy_lines).collect()
    };
    let (lines, defects) = replay_report(&sheets);
    assert_eq!(lines, lines_with(0));
    assert_eq!(defects, Vec::<String>::new());

    // Without its sheet the table's last policy has none.
    let (_, kept) = sheets.split_last().expect("a sheet a module");
    let (lines, defects) = replay_report(kept);
    let mut expected: Vec<String> = kept_counted
        .iter()
        .map(|(policy, rows)| format!("{policy}.json: {rows} rows, 0 defects"))
        .collect();
    expected.extend(
        kept_counted
            .iter()
            .map(|(policy, rows)| format!("policy {policy}: {rows} rows, 0 defects")),
    );
    expected.push(format!("policy {}: no sheet", last.0));
    assert_eq!(lines, expected);
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with(&format!("no sheet of {}", last.0))),
        "{defects:#?}"
    );

    // A second sheet of the first policy: its line is the first policy line,
    // behind a line a sheet.
    let mut twice = sheets.clone();
    twice.extend(sheets.first().cloned());
    let (lines, defects) = replay_report(&twice);
    assert_eq!(
        lines.get(twice.len()).map(String::as_str),
        Some(format!("policy {first}: 2 sheets").as_str())
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with(&format!("2 sheets of {first}"))),
        "{defects:#?}"
    );

    // A text that is no sheet has the last sheet line.
    let mut broken = sheets.clone();
    broken.push(("broken.json".to_string(), "no sheet".to_string()));
    let (lines, defects) = replay_report(&broken);
    assert_eq!(
        lines.get(sheets.len()).map(String::as_str),
        Some("broken.json: 0 rows, 1 defects")
    );
    assert!(
        matches!(defects.as_slice(), [one] if one.starts_with("broken.json is not a sheet of rows")),
        "{defects:#?}"
    );

    // The first pick of a sequence row, moved by one, in the manual sheet.
    let mut changed = sheets.clone();
    let manual_sheet = |(name, _): &&mut (String, String)| name == "manual.json";
    for (_, text) in changed.iter_mut().filter(manual_sheet) {
        let mut sheet: Value = serde_json::from_str(text).expect("the inline sheet");
        let rows = sheet.get_mut("rows").and_then(Value::as_array_mut);
        let pick = rows
            .expect("its rows")
            .iter_mut()
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
    assert_eq!(lines, lines_with(defects.len()));
}

/// The whole replay, over the sheets beside this target: every row of each
/// sheet under `tests/classic/sheets` that has a module here, and no row
/// of an owed sheet. It fails when a policy with a module here has no
/// sheet in the directory, when an owed policy has no sheet
/// there or a second one, when a row copied into a module is not its
/// sheet's row, and when a policy's line counts no row, so a run that
/// replays nothing cannot pass.
#[test]
#[expect(
    clippy::print_stdout,
    reason = "the per-sheet counts are the replay's report"
)]
fn every_recorded_classic_row_replays() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/classic/sheets");
    let sheets = sheets_under(&dir);
    assert!(!sheets.is_empty(), "no sheets under {}", dir.display());
    let (lines, defects) = replay_report(&sheets);
    for line in &lines {
        println!("{line}");
    }
    assert!(defects.is_empty(), "{defects:#?}");
    let owed = owed_sheet_defects(&sheets);
    assert!(owed.is_empty(), "{owed:#?}");
    let policies: Vec<&String> = lines
        .iter()
        .filter(|line| line.starts_with("policy "))
        .collect();
    assert_eq!(policies.len(), MODULES.len(), "{lines:#?}");
    assert!(
        policies.iter().all(|line| !line.contains(": 0 rows, ")),
        "a policy replayed no row: {lines:#?}"
    );
}
