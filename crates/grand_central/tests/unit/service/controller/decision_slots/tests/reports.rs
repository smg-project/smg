//! The cases of the load reports a decision is given: the snapshot the load
//! monitor published last, and the document that is given none.

use super::*;

/// A snapshot as the load monitor publishes it for the bed's three
/// workers: the one at `idle` reports no queue, and the others are deep
/// in theirs.
fn reports(idle: usize) -> Arc<LoadSnapshot> {
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(at, name)| {
        let load = if at == idle {
            one_rank_load(0, 0.1, 10.0)
        } else {
            one_rank_load(900_000, 0.9, 10.0)
        };
        (format!("http://{name}:8000"), load)
    });
    LoadSnapshot::from_loads_for_test(loads.collect())
}

/// A deciding thread gives each frame's request the newest snapshot
/// the load monitor published. Under the least-load preset every pick
/// names the one worker whose report shows no queue. The monitor then
/// publishes another worker so, and the picks follow at the next frame.
/// It publishes an empty snapshot, as at its lead's end, and the next
/// frame reads no report of the one before: its pick is the one worker
/// that holds no request.
#[tokio::test]
async fn a_decision_reads_the_reports_the_load_monitor_last_published() {
    let mut bed = bed(Some(preset("least_load", 1)), true, (2, 8), []);
    // Three frames a publication: a worker of the bed runs four requests
    // at the most.
    for (idle, ids) in [(0, 1..=3u8), (1, 4..=6)] {
        bed.loads.send_replace(reports(idle));
        for id in ids {
            let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
            assert_eq!(picked(&answer), bed.workers[idle], "{answer:?}");
        }
    }
    bed.loads.send_replace(Arc::new(LoadSnapshot::default()));
    let answer = bed.ask(frame(7, 107)).await;
    assert_eq!(picked(&answer), bed.workers[2], "{answer:?}");
}

/// A snapshot as the load monitor publishes it for the bed's three
/// workers, each report counting its resident tokens: the one at `idle`
/// holds none, and the others hold many.
fn resident_reports(idle: usize) -> Arc<LoadSnapshot> {
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(at, name)| {
        let mut load = one_rank_load(0, 0.1, 10.0);
        load.loads[0].num_resident_tokens = Some(if at == idle { 0 } else { 900_000 });
        (format!("http://{name}:8000"), load)
    });
    LoadSnapshot::from_loads_for_test(loads.collect())
}

/// A document whose scorer is `resident-work` is given the snapshot, as
/// any document that reads reports: the scorer keeps no number of a
/// report. The monitor publishes one worker that holds no resident
/// tokens and the others many, and three frames name that worker,
/// whatever it holds in flight by then. The monitor then publishes
/// another worker so, and the picks follow. Given no snapshot, the
/// scorer would cost each worker by its requests in flight, and three
/// frames would name three workers.
#[tokio::test]
async fn a_decision_under_the_resident_work_scorer_reads_the_monitors_reports() {
    let document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": {
                "score": [{ "resident-work": { "request_tokens": 512 } }],
                "pick": "best"
              } },
              "bindings": [{ "profile": "p" }]
            }"#,
    )
    .expect("a document of this build");
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    // Three frames a publication: a worker of the bed runs four requests
    // at the most.
    for (idle, ids) in [(0, 1..=3u8), (1, 4..=6)] {
        bed.loads.send_replace(resident_reports(idle));
        for id in ids {
            let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
            assert_eq!(picked(&answer), bed.workers[idle], "{answer:?}");
        }
    }
}

/// A document with a stage that keeps load reports is given no
/// snapshot: its engine's own task tells such a stage each report, and
/// a controller runs no such task yet. The stage is the cases' own
/// ([`KeepsReports`]). The monitor publishes one worker with no queue
/// and the others deep in theirs, and three frames still name three
/// workers, each by the requests it holds. The scorer that reads
/// reports outweighs the one that keeps them eight to one: given the
/// snapshot, it would name the worker with no queue three times.
#[tokio::test]
async fn a_document_with_a_stage_that_keeps_reports_is_given_no_snapshot() {
    let document: PolicySpec = serde_json::from_str(
        r#"{
              "apiVersion": "smg.policy/v1",
              "generation": 1,
              "profiles": { "p": {
                "score": [
                  { "least-load": { "weight": 8 } },
                  "keeps-reports"
                ],
                "pick": "best"
              } },
              "bindings": [{ "profile": "p" }]
            }"#,
    )
    .expect("a document of this build");
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer("keeps-reports", || Box::new(KeepsReports))
        .expect("a free name");
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    bed.loads.send_replace(reports(0));
    let mut named = Vec::new();
    for id in 1..=3u8 {
        let answer = bed.ask(frame(id, 100 + u64::from(id))).await;
        named.push(picked(&answer));
    }
    assert!(
        named[0] != named[1] && named[1] != named[2] && named[0] != named[2],
        "the picks followed a report: {named:?}"
    );
}
