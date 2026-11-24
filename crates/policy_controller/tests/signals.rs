//! A document declares signals, and a stage a host registers reads one
//! beside the rows: by the row's position and the signal's column.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

use std::sync::{Arc, Mutex};

use smg_policy_controller::{
    document::{AppliedSpec, Needs, PolicySpec, MAX_SIGNALS},
    engine::{
        Candidate, Constraints, Decision, Engine, Filter, PairPools, Phase, Picker, PoolFacts,
        RequestFacts, Scorer, Scratch, Targets, TieBand, WorkerId,
    },
    load::{SignalCell, SignalCells, SignalColumn},
    stages::{Lifecycle, StageBuild, StageRegistry},
};

/// The decisions' instant on the host's clock.
const NOW: u64 = 10_000_000;

/// What one row read: its worker's id, then the two signals' values.
type Read = (u64, Option<f64>, Option<f64>);

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Names {
    first: String,
    second: String,
}

/// Reads two signals of every row it is asked and records what it read.
#[derive(Debug)]
struct ReadsTwo {
    first: SignalColumn,
    second: SignalColumn,
    seen: Arc<Mutex<Vec<Read>>>,
}

impl Lifecycle for ReadsTwo {}

impl Scorer for ReadsTwo {
    fn name(&self) -> &'static str {
        "reads-two"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let mut seen = self.seen.lock().expect("no reader panicked");
        for (&row, score) in kept.iter().zip(out) {
            let row = &candidates[row as usize];
            seen.push((
                row.id.0,
                self.first.read(facts, row),
                self.second.read(facts, row),
            ));
            *score = 0.0;
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs::NONE.with_signal_cells()
    }
}

/// The stock stages and `reads-two`, which records into `seen`.
fn catalog(seen: &Arc<Mutex<Vec<Read>>>) -> StageRegistry {
    let seen = Arc::clone(seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("reads-two", move |build: &mut StageBuild<'_>| {
            let names: Names = build.params()?;
            Ok(Box::new(ReadsTwo {
                first: build.signal_column(&names.first)?,
                second: build.signal_column(&names.second)?,
                seen: Arc::clone(&seen),
            }))
        })
        .expect("a free name");
    catalog
}

/// Two signals, and one profile whose scorer reads `pending` first and
/// `free` second. In name order `free` is column 0 and `pending` column 1.
const DOCUMENT: &str = "apiVersion: smg.policy/v1
signals:
  pending: { unit: tokens, max_age_ms: 4000 }
  free: { unit: blocks }
profiles:
  default:
    score:
      - reads-two: { first: pending, second: free }
    pick: best
";

fn document(text: &str) -> PolicySpec {
    serde_yaml::from_str(text).expect("a well-formed document")
}

/// `DOCUMENT`'s engine and the record its scorer writes.
fn engine() -> (Engine, Arc<Mutex<Vec<Read>>>) {
    let seen = Arc::default();
    let engine = document(DOCUMENT)
        .validate_in(&catalog(&seen), &[])
        .expect("the document validates");
    (engine, seen)
}

fn rows(ids: &[u64]) -> Vec<Candidate> {
    let row = |(position, &id): (u32, &u64)| Candidate {
        position,
        id: WorkerId(id),
        pair_class: 1,
        ..Candidate::default()
    };
    (0u32..).zip(ids).map(row).collect()
}

fn cell(value: f64, age_micros: u64) -> SignalCell {
    SignalCell {
        value,
        sampled_at_micros: NOW - age_micros,
    }
}

/// One decision over `rows` with `cells` on the request's facts; what the
/// scorer read.
fn reads(rows: &[Candidate], cells: Option<&SignalCells>) -> Vec<Read> {
    let (engine, seen) = engine();
    let facts = RequestFacts {
        now_micros: NOW,
        signal_cells: cells,
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &pool,
        rows,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.pick().is_some(), "the decision stands");
    let seen = seen.lock().expect("no reader panicked");
    seen.clone()
}

#[test]
fn a_document_with_no_signals_writes_no_section() {
    let plain = document("apiVersion: smg.policy/v1\nprofiles: { p: { pick: random } }\n");
    let written = serde_json::to_value(&plain).expect("JSON");
    assert!(written.get("signals").is_none());
    let applied = AppliedSpec::validate(&plain).expect("validates");
    assert!(applied.signal_columns().is_empty());
    assert!(AppliedSpec::unbound().signal_columns().is_empty());
}

#[test]
fn a_document_with_signals_round_trips_in_name_order_and_moves_the_digest() {
    let declared = document(DOCUMENT);
    let written = serde_json::to_string(&declared).expect("JSON");
    let (free, pending) = (written.find("\"free\""), written.find("\"pending\""));
    assert!(free < pending, "the names are written in their order");
    let read: PolicySpec = serde_json::from_str(&written).expect("its own JSON");
    assert_eq!(read, declared);
    let mut bare = declared.clone();
    bare.signals.clear();
    assert_ne!(
        declared.digest().expect("a digest").to_string(),
        bare.digest().expect("a digest").to_string()
    );
    // The resolved list: a name's place is its column, with its unit.
    let applied = AppliedSpec::validate_in(&declared, &catalog(&Arc::default()), &[])
        .expect("the document validates");
    let listed: Vec<_> = applied
        .signal_columns()
        .iter()
        .map(|column| (column.name(), column.unit()))
        .collect();
    assert_eq!(listed, [("free", "blocks"), ("pending", "tokens")]);
}

#[test]
fn the_signals_section_refuses_each_bad_entry_by_its_path() {
    let refusal = |signals: &str| {
        let text = format!(
            "apiVersion: smg.policy/v1\nsignals:\n{signals}profiles: {{ p: {{ pick: random }} }}\n"
        );
        let refused = document(&text).validate().expect_err("refused");
        refused.to_string()
    };
    let not_a_name =
        " is not a lower-case name of letters, digits and underscores, at most 63 bytes";
    for name in ["KV-Used", "9lives", "_led", &"a".repeat(64)] {
        assert_eq!(
            refusal(&format!("  {name}: {{ unit: tokens }}\n")),
            format!("signals = {name}{not_a_name}")
        );
    }
    let many: String = (0..=MAX_SIGNALS)
        .map(|n| format!("  s{n}: {{ unit: tokens }}\n"))
        .collect();
    assert_eq!(refusal(&many), "signals lists 65 entries; at most 64");
    // Visible ASCII alone, so a unit's bytes are its characters and no
    // space hides at an end: the third is two characters and three bytes,
    // the fourth holds a tab, the last a space.
    for unit in [
        String::new(),
        "u".repeat(32),
        "\u{b5}s".into(),
        "a\\tb".into(),
        "tokens / s".into(),
    ] {
        assert_eq!(
            refusal(&format!("  held: {{ unit: \"{unit}\" }}\n")),
            format!(
                "signals.held.unit = \"{unit}\" is not a unit of 1 to 31 ASCII characters with no space"
            )
        );
    }
    assert_eq!(
        refusal("  held: { unit: tokens, max_age_ms: 0 }\n"),
        "signals.held.max_age_ms = 0 is not above zero"
    );
    // An hour is the store's clock: a limit of an hour or more is refused.
    assert_eq!(
        refusal("  held: { unit: tokens, max_age_ms: 3600000 }\n"),
        "signals.held.max_age_ms = 3600000 is not a whole number from 1 to 3599999"
    );
    // The longest name, the longest unit, the longest age and the most
    // signals are taken.
    // The unit holds both ends of its alphabet: an exclamation mark and a
    // tilde.
    let longest = format!(
        "  {}: {{ unit: \"!{}~\", max_age_ms: 3599999 }}\n",
        "a".repeat(63),
        "u".repeat(29)
    );
    let most: String = (1..MAX_SIGNALS)
        .map(|n| format!("  s{n}: {{ unit: tokens }}\n"))
        .collect();
    let text = format!(
        "apiVersion: smg.policy/v1\nsignals:\n{longest}{most}profiles: {{ p: {{ pick: random }} }}\n"
    );
    let applied = AppliedSpec::validate(&document(&text)).expect("validates");
    assert_eq!(applied.signal_columns().len(), MAX_SIGNALS);
}

#[test]
fn a_signal_written_twice_is_refused_at_the_parse() {
    let twice = r#"{"apiVersion":"smg.policy/v1",
        "signals":{"held":{"unit":"tokens"},"held":{"unit":"blocks"}},
        "profiles":{"p":{"pick":"random"}}}"#;
    let refused = serde_json::from_str::<PolicySpec>(twice).expect_err("refused");
    assert!(refused.to_string().contains("held"), "{refused}");
}

#[test]
fn an_undeclared_signal_is_refused_at_the_stages_entry() {
    let undeclared = DOCUMENT.replace("second: free", "second: fre");
    let refused = document(&undeclared)
        .validate_in(&catalog(&Arc::default()), &[])
        .expect_err("refused");
    assert_eq!(
        refused.to_string(),
        "profiles.default.score[0].reads-two: fre is not a signal the document declares"
    );
}

#[test]
fn a_stage_reads_each_rows_value_by_its_position_and_none_past_its_age() {
    let rows = rows(&[31, 32, 33, 34]);
    // Sampled a microsecond after the decision's instant: a source's
    // clock may run ahead of the host's.
    let ahead = |value| SignalCell {
        value,
        sampled_at_micros: NOW + 1,
    };
    // A row's cells in the order of the names: `free`, then `pending`,
    // whose maximum age is 4 s.
    let cells = [
        cell(10.0, 9_000_000),
        cell(1.0, 4_000_000),
        SignalCell::NONE,
        cell(2.0, 4_000_001),
        cell(30.0, 0),
        cell(3.0, 1),
        ahead(40.0),
        ahead(4.0),
    ];
    assert_eq!(
        reads(&rows, Some(&SignalCells::new(cells.to_vec(), 2))),
        [
            // `free` has no maximum age; `pending` at exactly its own is read.
            (31, Some(1.0), Some(10.0)),
            // No value, and a value one microsecond too old.
            (32, None, None),
            (33, Some(3.0), Some(30.0)),
            // A value sampled after the decision's instant is read.
            (34, Some(4.0), Some(40.0)),
        ]
    );
}

#[test]
fn a_cell_of_no_number_reads_as_no_value_whatever_its_instant() {
    let rows = rows(&[31]);
    // Sampled this instant, and no number: the cell of no value under
    // another stamp. The largest finite value is a value.
    let cells = [cell(f64::NAN, 0), cell(f64::MAX, 0)];
    assert_eq!(
        reads(&rows, Some(&SignalCells::new(cells.to_vec(), 2))),
        [(31, Some(f64::MAX), None)]
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a signal's value is finite: no source states another")]
fn a_debug_build_refuses_a_value_that_is_not_finite() {
    let rows = rows(&[31]);
    // A host writes what it computed: an overflow, a division by zero.
    let cells = [cell(f64::INFINITY, 0), cell(1.0, 0)];
    let _ = reads(&rows, Some(&SignalCells::new(cells.to_vec(), 2)));
}

#[test]
fn facts_with_no_cells_read_no_value() {
    let rows = rows(&[31, 32]);
    assert_eq!(reads(&rows, None), [(31, None, None), (32, None, None)]);
}

#[test]
fn each_side_of_a_pair_reads_its_own_pools_cells() {
    let seen = Arc::default();
    let paired = r#"{"apiVersion":"smg.policy/v1",
        "signals":{"free":{"unit":"blocks"},"pending":{"unit":"tokens"}},
        "profiles":{"side":{"score":[{"reads-two":{"first":"pending","second":"free"}}],"pick":"best"}},
        "bindings":[{"match":{"phase":["prefill","decode"]},"plan":"pd-pair",
                     "profiles":{"prefill":"side","decode":"side"}}]}"#;
    let engine = serde_json::from_str::<PolicySpec>(paired)
        .expect("a well-formed document")
        .validate_in(&catalog(&seen), &[])
        .expect("the document validates");
    let (prefills, decodes) = (rows(&[31]), rows(&[41]));
    // Position 0 holds another value in each pool, and a third on the
    // request, which names neither pool's rows.
    let of = |free, pending| [cell(free, 0), cell(pending, 0)];
    let (prefill, decode, request) = (of(1.0, 2.0), of(3.0, 4.0), of(5.0, 6.0));
    let prefill = SignalCells::new(prefill.to_vec(), 2);
    let decode = SignalCells::new(decode.to_vec(), 2);
    let request = SignalCells::new(request.to_vec(), 2);
    let facts = RequestFacts {
        phase: Phase::Prefill,
        now_micros: NOW,
        signal_cells: Some(&request),
        ..RequestFacts::default()
    };
    let cx = Constraints::default();
    let decision = engine.decide_pair(
        &facts,
        &PoolFacts::default(),
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            prefill_signal_cells: Some(&prefill),
            decode_signal_cells: Some(&decode),
            ..PairPools::default()
        },
        &cx,
        &cx,
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "a pair was available"
    );
    let mut seen = seen.lock().expect("no reader panicked").clone();
    seen.dedup();
    assert_eq!(
        seen,
        [(31, Some(2.0), Some(1.0)), (41, Some(4.0), Some(3.0))]
    );
}

#[test]
fn cells_that_end_before_the_rows_read_no_value_past_their_end() {
    let (engine, seen) = engine();
    let rows = rows(&[31, 32]);
    // The cells of one row, for two.
    let cells = SignalCells::new(vec![cell(10.0, 0), cell(1.0, 0)], 2);
    let facts = RequestFacts {
        now_micros: NOW,
        signal_cells: Some(&cells),
        ..RequestFacts::default()
    };
    // The stage alone: a decision over such cells is refused by a debug
    // build's assertion, and a release build tests nothing first.
    let scorer = &engine.profiles()[0].scorers[0].scorer;
    let _ = scorer.score(&facts, &rows, &[0, 1], &mut [0.0; 2]);
    assert_eq!(
        *seen.lock().expect("no reader panicked"),
        [(31, Some(1.0), Some(10.0)), (32, None, None)]
    );
}

#[test]
fn a_column_past_a_rows_cells_reads_no_value() {
    let (engine, seen) = engine();
    let rows = rows(&[31, 32]);
    // One cell a row under a document of two signals: the second column
    // of a row would be the next row's first.
    let cells = SignalCells::new(vec![cell(10.0, 0), cell(20.0, 0)], 1);
    let facts = RequestFacts {
        now_micros: NOW,
        signal_cells: Some(&cells),
        ..RequestFacts::default()
    };
    let scorer = &engine.profiles()[0].scorers[0].scorer;
    let _ = scorer.score(&facts, &rows, &[0, 1], &mut [0.0; 2]);
    assert_eq!(
        *seen.lock().expect("no reader panicked"),
        [(31, None, Some(10.0)), (32, None, Some(20.0))]
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the signal cells are those of the rows the decision was given")]
fn a_debug_build_refuses_a_decision_over_cells_of_another_shape() {
    let rows = rows(&[31, 32]);
    let cells = SignalCells::new(vec![cell(10.0, 0), cell(1.0, 0)], 2);
    let _ = reads(&rows, Some(&cells));
}

#[test]
fn a_profile_declares_the_signal_cells_only_where_a_stage_resolved_a_signal() {
    let (reading, _) = engine();
    assert!(reading.needs().signal_cells);
    assert!(reading.needs_for("m", Phase::Single).signal_cells);
    // The signals are declared and no stage reads one.
    let unread = DOCUMENT.replace(
        "- reads-two: { first: pending, second: free }",
        "- least-requests",
    );
    let unread = document(&unread).validate().expect("validates");
    assert!(!unread.needs().signal_cells);
    assert!(!unread.needs_for("m", Phase::Single).signal_cells);
}

/// A scorer whose needs say `declares` and whose build resolves `free`
/// where `resolves` is set.
#[derive(Debug)]
struct Mismatched {
    declares: bool,
}

impl Lifecycle for Mismatched {}

impl Scorer for Mismatched {
    fn name(&self) -> &'static str {
        "mismatched"
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], _: &mut [f32]) -> TieBand {
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        let mut needs = Needs::NONE;
        needs.signal_cells = self.declares;
        needs
    }
}

#[test]
fn a_stage_whose_needs_and_build_disagree_on_the_signal_cells_is_refused() {
    let refusal = |declares: bool, resolves: bool| {
        let mut catalog = StageRegistry::builtin();
        catalog
            .scorer_with("mismatched", move |build: &mut StageBuild<'_>| {
                if resolves {
                    build.signal_column("free")?;
                }
                Ok(Box::new(Mismatched { declares }))
            })
            .expect("a free name");
        let text = DOCUMENT.replace(
            "- reads-two: { first: pending, second: free }",
            "- mismatched",
        );
        document(&text)
            .validate_in(&catalog, &[])
            .map(|_| ())
            .map_err(|refused| refused.to_string())
    };
    assert_eq!(refusal(true, true), Ok(()));
    assert_eq!(refusal(false, false), Ok(()));
    assert_eq!(
        refusal(false, true),
        Err(
            "profiles.default.score[0].mismatched: the stage keeps a signal's column: its needs \
             must set signal_cells"
                .to_owned()
        )
    );
    assert_eq!(
        refusal(true, false),
        Err(
            "profiles.default.score[0].mismatched: the stage's needs set signal_cells and its \
             build resolved no signal: resolve one or leave the need unset"
                .to_owned()
        )
    );
}

/// A filter and a picker whose needs say `declares`. Neither is built
/// under a context, so neither resolves a column.
#[derive(Debug)]
struct Declares {
    declares: bool,
}

impl Lifecycle for Declares {}

impl Filter for Declares {
    fn name(&self) -> &'static str {
        "declares"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn needs(&self) -> Needs {
        let mut needs = Needs::NONE;
        needs.signal_cells = self.declares;
        needs
    }
}

impl Picker for Declares {
    fn name(&self) -> &'static str {
        "declares"
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        _: &[f32],
        _: TieBand,
        _: &mut Targets,
    ) {
    }

    fn needs(&self) -> Needs {
        let mut needs = Needs::NONE;
        needs.signal_cells = self.declares;
        needs
    }
}

/// A stage the catalog builds with no context resolves no signal, so one
/// that declares the cells would have a host hand cells nothing reads. It
/// is refused as a stage a factory built is, and the refusal ends with
/// what its author can do: a scorer bare and under a weight, a filter and
/// a picker.
#[test]
fn a_stage_built_with_no_context_that_declares_the_signal_cells_is_refused() {
    // Whether the validated engine declares the cells, or the refusal.
    let validated = |catalog: &StageRegistry, text: &str| {
        document(text)
            .validate_in(catalog, &[])
            .map(|engine| engine.needs().signal_cells)
            .map_err(|refused| refused.to_string())
    };
    let held = |declares: bool, path: &str, remedy: &str| {
        if declares {
            Err(format!(
                "{path}: the stage's needs set signal_cells and its build resolved no signal: \
                 {remedy}"
            ))
        } else {
            Ok(false)
        }
    };
    for declares in [true, false] {
        let mut scorers = StageRegistry::builtin();
        scorers
            .scorer("mismatched", move || Box::new(Mismatched { declares }))
            .expect("a free name");
        for entry in ["- mismatched", "- mismatched: { weight: 2 }"] {
            let text = DOCUMENT.replace("- reads-two: { first: pending, second: free }", entry);
            assert_eq!(
                validated(&scorers, &text),
                held(
                    declares,
                    "profiles.default.score[0].mismatched",
                    "register it by scorer_with to resolve one, or leave the need unset"
                ),
                "{entry}"
            );
        }

        let mut filters = StageRegistry::builtin();
        filters
            .filter("declares", move || Box::new(Declares { declares }))
            .expect("a free name");
        let text = DOCUMENT.replace(
            "    score:\n      - reads-two: { first: pending, second: free }\n    pick: best",
            "    filter:\n      - declares\n    pick: random",
        );
        assert_eq!(
            validated(&filters, &text),
            held(
                declares,
                "profiles.default.filter[0].declares",
                "register it by filter_with to resolve one, or leave the need unset"
            )
        );

        let mut pickers = StageRegistry::builtin();
        pickers
            .picker("declares", move || Box::new(Declares { declares }))
            .expect("a free name");
        let text = DOCUMENT
            .replace(
                "- reads-two: { first: pending, second: free }",
                "- least-requests",
            )
            .replace("pick: best", "pick: declares");
        assert_eq!(
            validated(&pickers, &text),
            held(
                declares,
                "profiles.default.pick.declares",
                "register it by picker_with to resolve one, or leave the need unset"
            )
        );
    }
}

/// Keeps the rows whose pending tokens stand at or under its limit, and a
/// row of no value.
#[derive(Debug)]
struct AtMost {
    pending: SignalColumn,
    limit: f64,
    declares: bool,
}

impl Filter for AtMost {
    fn name(&self) -> &'static str {
        "at-most"
    }

    fn keep(&self, facts: &RequestFacts<'_>, row: &Candidate, _: &Constraints<'_>) -> bool {
        let pending = self.pending.read(facts, row);
        pending.is_none_or(|tokens| tokens <= self.limit)
    }

    fn needs(&self) -> Needs {
        let mut needs = Needs::NONE;
        needs.signal_cells = self.declares;
        needs
    }
}

#[test]
fn a_registered_filter_resolves_a_signal_and_is_held_to_its_need() {
    let validated = |declares: bool| {
        let mut catalog = StageRegistry::builtin();
        catalog
            .filter_with("at-most", move |build: &mut StageBuild<'_>| {
                Ok(Box::new(AtMost {
                    pending: build.signal_column("pending")?,
                    limit: 5.0,
                    declares,
                }))
            })
            .expect("a free name");
        let text = "apiVersion: smg.policy/v1
signals:
  pending: { unit: tokens }
profiles:
  default:
    filter:
      - at-most
    pick: random
";
        document(text).validate_in(&catalog, &[])
    };
    // A filter that keeps a column and says nothing would be handed no
    // cells: it is refused where its context is made, as a scorer is.
    assert_eq!(
        validated(false)
            .map(|_| ())
            .map_err(|refused| refused.to_string()),
        Err(
            "profiles.default.filter[0].at-most: the stage keeps a signal's column: its needs \
             must set signal_cells"
                .to_owned()
        )
    );
    let engine = validated(true).expect("validates");
    assert!(engine.profiles()[0].needs().signal_cells);
    // Rows 31 and 32 hold 9 and 3 pending tokens: the second alone stays.
    let rows = rows(&[31, 32]);
    let cells = SignalCells::new(vec![cell(9.0, 0), cell(3.0, 0)], 1);
    let facts = RequestFacts {
        now_micros: NOW,
        signal_cells: Some(&cells),
        ..RequestFacts::default()
    };
    let pool = PoolFacts {
        pool_size: rows.len() as u32,
        ..PoolFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &pool,
        &rows,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("a pick");
    assert_eq!(
        pick.targets.first().map(|target| target.id),
        Some(WorkerId(32))
    );
}

#[test]
fn a_signal_declares_how_several_ranks_make_one_value_and_the_sum_is_left_unwritten() {
    use smg_policy_controller::document::AcrossRanks;

    let declaring = |across: &str| {
        document(&format!(
            "apiVersion: smg.policy/v1\nsignals:\n  depth: {{ unit: items{across} }}\nprofiles: {{ p: {{ pick: random }} }}\n"
        ))
    };
    let across = |document: &PolicySpec| {
        let applied = AppliedSpec::validate(document).expect("validates");
        applied.signal_columns()[0].across_ranks()
    };
    // Left out, a report of several ranks is summed.
    let plain = declaring("");
    assert_eq!(across(&plain), AcrossRanks::Sum);
    for (word, declared) in [
        ("sum", AcrossRanks::Sum),
        ("max", AcrossRanks::Max),
        ("min", AcrossRanks::Min),
        ("mean", AcrossRanks::Mean),
    ] {
        let written = declaring(&format!(", across_ranks: {word}"));
        assert_eq!(across(&written), declared, "{word}");
        let read: PolicySpec =
            serde_json::from_str(&serde_json::to_string(&written).expect("JSON"))
                .expect("its own JSON");
        assert_eq!(read, written, "{word}");
        // The sum written out is the document that leaves it out: one
        // canonical form and one digest. Any other word moves both.
        let same = written.digest().expect("a digest").to_string()
            == plain.digest().expect("a digest").to_string();
        assert_eq!(same, declared == AcrossRanks::Sum, "{word}");
    }
    let unknown = serde_yaml::from_str::<PolicySpec>(
        "apiVersion: smg.policy/v1\nsignals:\n  depth: { unit: items, across_ranks: median }\nprofiles: { p: { pick: random } }\n",
    );
    assert!(unknown.is_err(), "a word that is no reduction");
}

#[test]
fn a_host_refills_its_cells_in_place_and_the_next_decision_reads_them() {
    let rows = rows(&[31, 32]);
    // A row's cells in the order of the names: `free`, then `pending`.
    let first = [cell(10.0, 0), cell(1.0, 0), cell(20.0, 0), cell(2.0, 0)];
    let mut cells = SignalCells::new(first.to_vec(), 2);
    assert_eq!(
        reads(&rows, Some(&cells)),
        [(31, Some(1.0), Some(10.0)), (32, Some(2.0), Some(20.0))]
    );
    let room = cells.cells_mut().capacity();
    let refilled = cells.cells_mut();
    refilled.clear();
    refilled.extend([cell(30.0, 0), cell(3.0, 0), cell(40.0, 0), cell(4.0, 0)]);
    assert_eq!(
        cells.cells_mut().capacity(),
        room,
        "the refill keeps the room"
    );
    assert_eq!(
        reads(&rows, Some(&cells)),
        [(31, Some(3.0), Some(30.0)), (32, Some(4.0), Some(40.0))]
    );
}

#[test]
fn the_default_cells_read_no_value() {
    let rows = rows(&[31, 32]);
    assert_eq!(
        reads(&rows, Some(&SignalCells::default())),
        [(31, None, None), (32, None, None)]
    );
}

/// A picker built under a context is held to its word on the signal cells
/// as a scorer is: the one maker of a context holds every kind of stage.
#[test]
fn a_picker_built_under_a_context_is_held_to_its_word_on_the_cells() {
    let refusal = |declares: bool, resolves: bool| {
        let mut catalog = StageRegistry::builtin();
        catalog
            .picker_with("declares", move |build: &mut StageBuild<'_>| {
                if resolves {
                    build.signal_column("free")?;
                }
                Ok(Box::new(Declares { declares }))
            })
            .expect("a free name");
        let text = DOCUMENT
            .replace(
                "- reads-two: { first: pending, second: free }",
                "- least-requests",
            )
            .replace("pick: best", "pick: declares");
        document(&text)
            .validate_in(&catalog, &[])
            .map(|_| ())
            .map_err(|refused| refused.to_string())
    };
    assert_eq!(refusal(true, true), Ok(()));
    assert_eq!(refusal(false, false), Ok(()));
    assert_eq!(
        refusal(false, true),
        Err(
            "profiles.default.pick.declares: the stage keeps a signal's column: its needs must \
             set signal_cells"
                .to_owned()
        )
    );
    assert_eq!(
        refusal(true, false),
        Err(
            "profiles.default.pick.declares: the stage's needs set signal_cells and its build \
             resolved no signal: resolve one or leave the need unset"
                .to_owned()
        )
    );
}
