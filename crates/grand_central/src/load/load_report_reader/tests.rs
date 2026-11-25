use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::{
    document::{PolicySpec, API_VERSION},
    engine::{
        Candidate, Constraints, Engine, Filter, LoadRow, PoolFacts, RequestFacts, Scorer, Scratch,
        TieBand, WorkerId,
    },
    load::LoadReportReader,
    stages::{Lifecycle, StageRegistry},
};

/// The view's maximum report age in the cases' documents.
const MAX_AGE_MS: u64 = 250;
const MAX_AGE_MICROS: u64 = MAX_AGE_MS * 1_000;
/// The instant of the cases' decisions on the host's clock.
const NOW: u64 = 7_400_000;

/// What the registered scorer was handed and what it read.
#[derive(Debug, Default)]
struct Probe {
    /// The reading each built scorer took from its build context.
    handed: Mutex<Vec<LoadReportReader>>,
    /// Each row's report as the scorer read it at its last decision.
    seen: Mutex<Vec<(WorkerId, Option<LoadRow>)>>,
}

/// A registered scorer that reads each row's report through the view's
/// reading and scores every row alike.
#[derive(Debug)]
struct Reported {
    reports: LoadReportReader,
    probe: Arc<Probe>,
}

impl Lifecycle for Reported {}

impl Scorer for Reported {
    fn name(&self) -> &'static str {
        "reported"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let reports = self.reports.pass(facts);
        let mut seen = self.probe.seen.lock();
        seen.clear();
        for &row in kept {
            let row = &candidates[row as usize];
            seen.push((row.id, reports.of(row).copied()));
        }
        out.fill(0.0);
        TieBand::NORMALIZED
    }
}

/// A registered filter that keeps the rows whose report the view reads.
#[derive(Debug)]
struct HasReport(LoadReportReader);

impl Filter for HasReport {
    fn name(&self) -> &'static str {
        "has-report"
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.read(facts, candidate).is_some()
    }
}

/// An engine of one profile with `stages` under a view with `max_age_ms`,
/// over a catalog that holds the registered scorer and the registered
/// filter.
fn engine(max_age_ms: Option<u64>, mut stages: Value) -> (Engine, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let kept = Arc::clone(&probe);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("reported", move |build| {
            let reports = build.load_report_reader();
            kept.handed.lock().push(reports);
            Ok(Box::new(Reported {
                reports,
                probe: Arc::clone(&kept),
            }))
        })
        .expect("a free name")
        .filter_with("has-report", |build| {
            Ok(Box::new(HasReport(build.load_report_reader())))
        })
        .expect("a free name");
    stages["pick"] = json!("best");
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "view": { "max_report_age_ms": max_age_ms },
        "profiles": { "work": stages }
    }))
    .expect("a document");
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the profile validates");
    (engine, probe)
}

/// A row whose report was sampled `age_micros` before [`NOW`]. Each row's
/// report states its own count of running requests.
fn aged(position: u32, age_micros: u64) -> Candidate {
    Candidate {
        position,
        id: WorkerId(301 + u64::from(position)),
        load: Some(LoadRow {
            running: 9 + position,
            ..LoadRow::default()
        }),
        load_sampled_at_micros: NOW - age_micros,
        ..Candidate::default()
    }
}

/// Three rows about the view's maximum age: a microsecond younger, at
/// it, and a microsecond older.
fn about_the_maximum() -> [Candidate; 3] {
    [
        aged(0, MAX_AGE_MICROS - 1),
        aged(1, MAX_AGE_MICROS),
        aged(2, MAX_AGE_MICROS + 1),
    ]
}

fn clock() -> RequestFacts<'static> {
    RequestFacts {
        now_micros: NOW,
        ..RequestFacts::default()
    }
}

/// One decision at [`NOW`] over `rows`: the scores of the rows the
/// filters kept, by worker.
fn decide(engine: &Engine, rows: &[Candidate; 3]) -> Vec<(WorkerId, f32)> {
    let pool = PoolFacts {
        pool_size: 3,
        ..PoolFacts::default()
    };
    let decision = engine.decide(
        &clock(),
        &pool,
        rows,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("the rows route");
    pick.trace.top[..usize::from(pick.trace.top_len)].to_vec()
}

/// What the registered scorer read of each of `rows` at the last decision.
fn read(probe: &Probe, rows: &[Candidate; 3]) -> [Option<LoadRow>; 3] {
    let seen = probe.seen.lock();
    rows.each_ref().map(|row| {
        let read = seen.iter().find(|(id, _)| *id == row.id);
        read.expect("the scorer read each row").1
    })
}

#[test]
fn a_registered_scorer_reads_no_report_older_than_the_views_maximum() {
    let (engine, probe) = engine(Some(MAX_AGE_MS), json!({ "score": ["reported"] }));
    let unreported = Candidate {
        load: None,
        ..aged(2, 0)
    };
    let rows = [
        aged(0, 2 * MAX_AGE_MICROS),
        aged(1, MAX_AGE_MICROS / 2),
        unreported,
    ];
    decide(&engine, &rows);
    assert_eq!(read(&probe, &rows), [None, rows[1].load, None]);

    // The reading follows the report's age, whichever row holds it.
    let rows = [
        aged(0, MAX_AGE_MICROS / 2),
        aged(1, 2 * MAX_AGE_MICROS),
        unreported,
    ];
    decide(&engine, &rows);
    assert_eq!(read(&probe, &rows), [rows[0].load, None, None]);

    // One row asked alone reads as the pass reads it.
    let reports = probe.handed.lock()[0];
    assert_eq!(reports.read(&clock(), &rows[0]), rows[0].load.as_ref());
    assert_eq!(reports.read(&clock(), &rows[1]), None);
}

#[test]
fn a_report_at_the_maximum_age_is_read_and_one_a_microsecond_older_is_not() {
    let (engine, probe) = engine(Some(MAX_AGE_MS), json!({ "score": ["reported"] }));
    let rows = about_the_maximum();
    decide(&engine, &rows);
    assert_eq!(read(&probe, &rows), [rows[0].load, rows[1].load, None]);
}

#[test]
fn a_document_with_no_maximum_age_reads_every_report() {
    let (engine, probe) = engine(None, json!({ "score": ["reported"] }));
    // The oldest report a host can hold was sampled at its clock's origin.
    let rows = [aged(0, NOW), aged(1, MAX_AGE_MICROS + 1), aged(2, 0)];
    decide(&engine, &rows);
    assert_eq!(read(&probe, &rows), rows.map(|row| row.load));
}

#[test]
fn a_registered_scorer_and_a_built_in_agree_on_the_rows_that_have_a_report() {
    // The built-in prices a row by its report's running requests and a row
    // with no report at nothing, so the rows it reads no report for score
    // above the others. The registered scorer scores every row alike.
    let built_in = json!({
        "least-cost": { "terms": { "load.running": 1 }, "unreported": "missing-as-zero" }
    });
    let (engine, probe) = engine(Some(MAX_AGE_MS), json!({ "score": [built_in, "reported"] }));
    let rows = about_the_maximum();
    let scores = decide(&engine, &rows);
    let best = scores
        .iter()
        .map(|&(_, score)| score)
        .fold(f32::MIN, f32::max);
    let built_in_read = rows.each_ref().map(|row| {
        let scored = scores.iter().find(|(id, _)| *id == row.id);
        scored.expect("the trace holds each row's score").1 < best
    });
    let registered_read = read(&probe, &rows).map(|report| report.is_some());
    assert_eq!(registered_read, built_in_read);
    assert_eq!(registered_read, [true, true, false]);
}

#[test]
fn a_registered_filter_reads_a_report_as_a_registered_scorer_does() {
    let (engine, _) = engine(
        Some(MAX_AGE_MS),
        json!({ "filter": ["has-report"], "score": ["reported"] }),
    );
    let rows = about_the_maximum();
    let mut kept: Vec<WorkerId> = decide(&engine, &rows).iter().map(|&(id, _)| id).collect();
    kept.sort_unstable();
    assert_eq!(kept, [rows[0].id, rows[1].id]);
}
