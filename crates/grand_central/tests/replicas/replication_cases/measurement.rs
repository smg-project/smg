//! A standby's snapshot and the operators' snapshot at fleets of a
//! production row's shape: measurements, run by name.

use std::{io::Write as _, time::Instant};

use prost::Message as _;
use smg_grand_central::{service::proto, store::WorkerFacts};

use crate::{client, facts, lays, place, replicate, shared, stated, turn, with_leader};

/// A worker's facts of a production row's shape: a 53-byte address, one
/// 36-byte model, four labels, a zone.
fn wide(worker: u32) -> WorkerFacts {
    let address = format!("http://engine-{worker:04}.serving-pool.example.internal:8000");
    let labels = (0..4)
        .map(|n| (format!("label-key-{n:02}"), format!("value-{worker:04}")))
        .collect();
    WorkerFacts {
        models: vec!["organization/model-name-70b-instruct".to_owned()],
        labels,
        zone: Some("region-1-zone-a".to_owned()),
        engine_kind: "engine-kind".to_owned(),
        ..facts(&address, 1, 64)
    }
}

/// A measurement, run by name with `--ignored --nocapture`: a standby's
/// snapshot at the design's top fleet and at 4,096 workers of a production
/// row's shape (a 53-byte address, one 36-byte model, four labels, a
/// zone). One line: the parts, their bytes together and the widest one,
/// and how long a standby that asks waits for the whole (the leader's
/// cut, its parts, the client's read), with the CPU of both ends over the
/// eleven snapshots timed.
/// Then the leader's memory for a held cut, which is what a watch's kept
/// cut holds: the resident growth of 32 streams left inside their
/// snapshots at 32 places. Each fleet is measured in a process of its
/// own: a resident reading starts where no cut of another fleet was let
/// go.
async fn snapshots_numbers(workers: u32) {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let mut ids = Vec::new();
        for worker in 0..workers {
            ids.push(lays(&leader, wide(worker)));
        }
        let mut out = std::io::stdout();
        let mut micros = Vec::new();
        let mut parts = Vec::new();
        let cpu = cpu_ms();
        for _ in 0..11 {
            let began = Instant::now();
            let mut standby = replicate(leader.replication, place(1, 0))
                .await
                .expect("a stream");
            standby.restored().await.expect("a snapshot");
            micros.push(began.elapsed().as_micros());
            parts = standby.parts;
        }
        let cpu = cpu_ms() - cpu;
        micros.sort_unstable();
        writeln!(
            out,
            "SNAPSHOT workers={workers} parts={} bytes={} widest_part={} median_us={} cpu_ms_of_11={cpu}",
            parts.len(),
            parts.iter().sum::<usize>(),
            parts.iter().max().copied().unwrap_or(0),
            micros[5]
        )
        .expect("stdout");
        // Streams left inside their snapshots, each at a place of its own:
        // a worker turned ready between two moves the head, so no two share
        // a cut. Each read the word and its first part, and no further.
        let mut held = Vec::new();
        let before = resident_kb();
        for id in ids.iter().take(32) {
            turn(&leader, *id, 0);
            let mut standby = replicate(leader.replication, place(1, 0))
                .await
                .expect("a stream");
            assert!(standby.batch().await.resync_required);
            assert!(standby.frame().await.snapshot.is_some());
            held.push(standby);
        }
        let grew = resident_kb().saturating_sub(before);
        writeln!(
            out,
            "HELD workers={workers} cuts=32 resident_kb_a_cut={}",
            grew / 32
        )
        .expect("stdout");
        assert_eq!(held.len(), 32, "each stream still holds its cut");
    })
    .await;
}

/// A measurement, run by name with `--ignored --nocapture`: the operators'
/// snapshot of the worker rows at a fleet of a production row's shape (a
/// 53-byte address, one 36-byte model, four labels, a zone), asked whole
/// and under a limit of one row. Each line: the rows told and whether a
/// bound cut them, the answer's bytes, how long the call takes over the
/// wire (the holder's cut, the rows' build, the client's read), and how long
/// the holder's cut takes alone: more than a membership write waits behind a
/// snapshot that copies every row, as a call with no limit and no models
/// does, and a call let in beside another. A lone call under a limit
/// copies its limit's worth alone, and a third line asks one row of the
/// widest list of models a call may state: each row's model is searched
/// in that list in the holder's turn. A last line, with a watch
/// open: one row asked eleven times, where each call shares the cut the
/// feed keeps and copies nothing.
///
/// The cut's own figure holds its sort, which runs behind the holder's
/// turn: a write waits for the copy of the rows, and not for the sort.
async fn operators_snapshot_numbers(workers: u32) {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        for worker in 0..workers {
            lays(&leader, wide(worker));
        }
        let mut cuts = Vec::new();
        for _ in 0..11 {
            let began = Instant::now();
            let cut = leader.workers.checkpoint(&leader.store);
            cuts.push(began.elapsed().as_micros());
            assert_eq!(cut.rows.len(), workers as usize);
        }
        cuts.sort_unstable();
        let mut out = std::io::stdout();
        let mut operator = client(leader.control).await;
        // The widest list a call may state: 127 models no row serves, then
        // the one every row serves.
        let widest: Vec<String> = (0..127)
            .map(|n| format!("organization/model-name-{n:03}-instruct"))
            .chain(["organization/model-name-70b-instruct".to_owned()])
            .collect();
        for (limit, models) in [(0u32, Vec::new()), (1, Vec::new()), (1, widest)] {
            let asked = models.len();
            let mut micros = Vec::new();
            let mut told = (0, false, 0);
            for _ in 0..11 {
                let request = stated(proto::SnapshotRequest {
                    slices: vec![proto::snapshot_request::Slice::Workers.into()],
                    limit,
                    models: models.clone(),
                    ..proto::SnapshotRequest::default()
                });
                let began = Instant::now();
                let answer = operator
                    .get_snapshot(request)
                    .await
                    .expect("the rows")
                    .into_inner();
                micros.push(began.elapsed().as_micros());
                told = (
                    answer.workers.len(),
                    answer.truncated,
                    answer.encoded_len(),
                );
            }
            micros.sort_unstable();
            writeln!(
                out,
                "OPERATORS_SNAPSHOT workers={workers} limit={limit} models={asked} rows={} truncated={} bytes={} median_us={} holder_cut_median_us={}",
                told.0, told.1, told.2, micros[5], cuts[5]
            )
            .expect("stdout");
        }
        // A watch is open: the feed keeps the cut its start took, and each
        // snapshot shares it.
        let mut gateway = client(leader.gateway).await;
        let mut watching = gateway
            .watch_workers(stated(proto::WatchWorkersRequest::default()))
            .await
            .expect("opened")
            .into_inner();
        watching
            .message()
            .await
            .expect("the stream is open")
            .expect("a page");
        let mut micros = Vec::new();
        for _ in 0..11 {
            let request = stated(proto::SnapshotRequest {
                slices: vec![proto::snapshot_request::Slice::Workers.into()],
                limit: 1,
                ..proto::SnapshotRequest::default()
            });
            let began = Instant::now();
            let answer = operator
                .get_snapshot(request)
                .await
                .expect("the rows")
                .into_inner();
            micros.push(began.elapsed().as_micros());
            assert_eq!(answer.workers.len(), 1);
        }
        micros.sort_unstable();
        writeln!(
            out,
            "OPERATORS_SNAPSHOT workers={workers} limit=1 watched=1 median_us={} most_us={}",
            micros[5], micros[10]
        )
        .expect("stdout");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_operators_snapshots_numbers_at_1000_workers() {
    operators_snapshot_numbers(1_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_operators_snapshots_numbers_at_10000_workers() {
    operators_snapshot_numbers(10_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_snapshots_numbers_at_512_workers() {
    snapshots_numbers(512).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_snapshots_numbers_at_4096_workers() {
    snapshots_numbers(4_096).await;
}

/// The process's CPU time so far, user and system, in milliseconds: the
/// kernel's two counts, in ticks of 10 ms.
fn cpu_ms() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("the process's stat");
    let fields = stat.rsplit_once(')').expect("the command's name").1;
    let ticks = |nth: usize| {
        fields
            .split_whitespace()
            .nth(nth)
            .and_then(|field| field.parse::<u64>().ok())
            .expect("a count of ticks")
    };
    (ticks(11) + ticks(12)) * 10
}

/// The process's resident set, in kilobytes.
fn resident_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .expect("the process's status")
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        .expect("the resident set")
}
