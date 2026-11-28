use super::*;

/// The first observed candidate as the evidence lists it.
fn first_candidate(capture: &Capture) -> Value {
    json!(capture.candidates[0])
}

/// A row is written with the evidence from the values its selection
/// kept, and it is the row the selection used to write as text where it
/// ran: the same keys, order and values for a candidate that advertises
/// one epoch, one that advertises two and one that advertises none, with
/// the prediction, score and gate recorded for it. What a worker shows
/// when the evidence is written is not in it: a load credit, an overload
/// flag and a health failure after the selection leave the row as it was
/// observed.
#[tokio::test]
async fn a_row_written_with_the_evidence_is_the_row_as_it_was_observed() {
    use openai_protocol::worker::WorkerStatus;

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let worker = |url: &str, epochs: Option<&str>| -> Arc<dyn Worker> {
        // Ready, so that the health failure below moves what a worker
        // shows after its row was observed.
        let mut builder = BasicWorkerBuilder::new(url).status(WorkerStatus::Ready);
        if let Some(epochs) = epochs {
            builder = builder.labels(std::collections::HashMap::from([(
                "cache_trace_epochs".to_string(),
                epochs.to_string(),
            )]));
        }
        Arc::new(builder.build())
    };
    let workers = vec![
        worker("grpc://one:50051", Some(r#"["first"]"#)),
        worker("grpc://two:50051", Some(r#"["first","second"]"#)),
        worker("grpc://none:50051", None),
    ];
    workers[2].increment_load();
    scope(true, async {
        // The rows as the selection wrote them before: one object per
        // candidate, read from the workers where the pick ran.
        let observed: Vec<Value> = workers
            .iter()
            .map(|worker| {
                let epochs = if worker.cache_trace_epochs_valid() {
                    worker.advertised_cache_trace_epochs()
                } else {
                    Vec::new()
                };
                let epoch = if epochs.len() == 1 {
                    epochs.first()
                } else {
                    None
                };
                json!({
                    "worker": worker.redacted_url(), "load": worker.load(), "healthy": worker.is_healthy(),
                    "overloaded": worker.is_overloaded(), "registry_revision": worker.revision(),
                    "backend_cache_epoch": epoch,
                    "backend_cache_epochs": epochs,
                    "backend_cache_epoch_source": "registration_metadata",
                })
            })
            .collect();
        assert_eq!(observed[0]["backend_cache_epoch"], "first");
        assert_ne!(observed[0]["registry_revision"], 41);
        assert_eq!(observed[1]["healthy"], true);
        assert_eq!(observed[2]["load"], 1);
        begin_selection(&workers);
        prediction(json!({"source": "approximate_tree", "overlap_blocks": 4}));
        score(json!({"worker": "grpc://two:50051", "score": 0.9}));
        gate(json!({"spill": false}));
        selection("cache_aware", "policy", &workers, Some(1));
        // After the selection, before the evidence is written: a load
        // credit, a revision raised, the overload flag, and a health
        // failure, which also withdraws the epochs.
        workers[0].increment_load();
        workers[0].raise_revision_to(41);
        workers[1].set_overloaded(true);
        workers[1].set_status(WorkerStatus::NotReady);
        assert_eq!(workers[0].revision(), 41);
        assert!(workers[1].is_overloaded());
        assert!(!workers[1].is_healthy());
        assert!(!workers[1].cache_trace_epochs_valid());
        let rows = captured_selections();
        let row = &rows[0];
        let expected = json!({
            "policy": "cache_aware", "origin": "policy",
            "worker": "grpc://two:50051",
            "prediction": {"source": "approximate_tree", "overlap_blocks": 4},
            "candidates": observed,
            "candidates_complete": true,
            "observation_started_ns": row["observation_started_ns"],
            "observation_finished_ns": row["observation_finished_ns"],
            "snapshot_atomic": false,
            "scores": [{"worker": "grpc://two:50051", "score": 0.9}],
            "load_source": "router_inflight", "eligibility_scope": "policy_candidates",
            "load_observation_phase": "before_selection",
            "gates": [{"spill": false}],
        });
        assert_eq!(*row, expected);
        assert_eq!(row.to_string(), expected.to_string());
    })
    .await;
}

/// A row with no worker (a shed, a refused commit) stays in the capture
/// in pick order: the failure line writes it and leaves it, and the
/// dispatch that follows a re-decision writes it before the row of the
/// pick that held.
#[tokio::test]
async fn a_row_without_a_worker_is_written_in_pick_order_by_the_failure_and_the_dispatch() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let workers: Vec<Arc<dyn Worker>> = ["grpc://one:50051", "grpc://two:50051"]
        .into_iter()
        .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
        .collect();
    scope(true, async {
        begin_selection(&workers);
        selection("round_robin", "policy", &workers, None);
        let failed = CAPTURE.with(|capture| failure_evidence(&capture.borrow(), Some("root"), 503));
        assert_eq!(failed["status"], 503);
        assert_eq!(failed["selections"].as_array().map(Vec::len), Some(1));
        assert!(failed["selections"][0]["worker"].is_null());
        assert_eq!(
            failed["selections"][0]["candidates"][1]["worker"],
            "grpc://two:50051"
        );
        begin_selection(&workers[1..]);
        selection("round_robin", "policy", &workers[1..], Some(0));
        let rows = captured_selections();
        assert_eq!(rows.len(), 2, "the failure line left the first row");
        assert!(rows[0]["worker"].is_null());
        assert_eq!(rows[0]["candidates"].as_array().map(Vec::len), Some(2));
        assert_eq!(rows[1]["worker"], "grpc://two:50051");
        assert_eq!(rows[1]["candidates"].as_array().map(Vec::len), Some(1));
        let header = dispatch(Some("root"), 1, vec!["engine".into()], true, "regular")
            .expect("the header for the request that asked");
        let header: Value = serde_json::from_str(&header).unwrap();
        assert!(header["selections"][0]["worker"].is_null());
        assert_eq!(header["selections"][1]["worker"], "grpc://two:50051");
        assert!(captured_selections().is_empty(), "the dispatch took them");
    })
    .await;
}

/// A selection observes 32 candidates at most and says whether that was
/// all of them; a pick beyond the 32 observed is still named.
#[tokio::test]
async fn a_row_observes_32_candidates_and_says_whether_that_was_all() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let workers: Vec<Arc<dyn Worker>> = (0..33)
        .map(|i| {
            Arc::new(BasicWorkerBuilder::new(format!("grpc://w{i}:50051")).build())
                as Arc<dyn Worker>
        })
        .collect();
    scope(true, async {
        begin_selection(&workers[..32]);
        selection("round_robin", "policy", &workers[..32], Some(31));
        begin_selection(&workers);
        selection("round_robin", "policy", &workers, Some(32));
        let rows = captured_selections();
        assert_eq!(rows[0]["candidates"].as_array().map(Vec::len), Some(32));
        assert_eq!(rows[0]["candidates_complete"], true);
        assert_eq!(rows[0]["worker"], "grpc://w31:50051");
        assert_eq!(rows[1]["candidates"].as_array().map(Vec::len), Some(32));
        assert_eq!(rows[1]["candidates_complete"], false);
        assert_eq!(rows[1]["candidates"][31]["worker"], "grpc://w31:50051");
        assert_eq!(rows[1]["worker"], "grpc://w32:50051");
    })
    .await;
}

/// A caller whose one call picks from two lists reads both ahead of the
/// call and writes a row for each behind it. Each row lists its own
/// list as it stood at the reading, whatever the call credited since,
/// under the caller's policy and origin, with the worker handed or
/// none, and with no prediction, score or gate. A list read and never
/// written leaves no row. Nothing is read where no capture is in scope.
/// A row past the bound ([`MAX_SELECTIONS`]) is dropped and marks the
/// record truncated, and the classic writer's own bound is the same.
#[tokio::test]
async fn a_row_of_a_list_read_ahead_is_the_list_as_it_was_read() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let worker = |url: &str| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>;
    let prefills = vec![worker("grpc://p1:50051"), worker("grpc://p2:50051")];
    let decodes = vec![worker("grpc://d1:50051")];
    assert!(observe(&prefills).is_none(), "no capture is in scope");
    // A list as a row lists it, read from its workers as they stand.
    let listed = |list: &[Arc<dyn Worker>]| -> Vec<Value> {
        let candidate = |worker: &Arc<dyn Worker>| {
            json!({
                "worker": worker.redacted_url(), "load": worker.load(), "healthy": worker.is_healthy(),
                "overloaded": worker.is_overloaded(), "registry_revision": worker.revision(),
                "backend_cache_epoch": null,
                "backend_cache_epochs": [],
                "backend_cache_epoch_source": "registration_metadata",
            })
        };
        list.iter().map(candidate).collect()
    };
    scope(true, async {
        let read = |list: &[Arc<dyn Worker>]| observe(list).expect("a capture is in scope");
        let (first, second, unwritten) = (read(&prefills), read(&decodes), read(&decodes));
        let as_read = (listed(&prefills), listed(&decodes));
        // Behind the reading: the call's own credit.
        prefills[1].increment_load();
        decodes[0].increment_load();
        assert_ne!(listed(&prefills), as_read.0, "the credit moved the list");
        selection_of("bucket", "engine", first, Some(&prefills[1]));
        selection_of("round_robin", "engine", second, None);
        drop(unwritten);
        let rows = captured_selections();
        let expected = |policy: &str, worker: Value, candidates: &[Value], row: &Value| {
            json!({
                "policy": policy, "origin": "engine", "worker": worker,
                "prediction": null,
                "candidates": candidates,
                "candidates_complete": true,
                "observation_started_ns": row["observation_started_ns"],
                "observation_finished_ns": row["observation_finished_ns"],
                "snapshot_atomic": false,
                "scores": [],
                "load_source": "router_inflight", "eligibility_scope": "policy_candidates",
                "load_observation_phase": "before_selection",
                "gates": [],
            })
        };
        assert_eq!(
            rows,
            [
                expected("bucket", json!("grpc://p2:50051"), &as_read.0, &rows[0]),
                expected("round_robin", Value::Null, &as_read.1, &rows[1]),
            ]
        );
        // One short of the bound: the classic writer, which states the
        // same bound in place, writes the last row the bound admits.
        for _ in 2..MAX_SELECTIONS - 1 {
            selection_of("round_robin", "engine", read(&decodes), None);
        }
        begin_selection(&decodes);
        selection("round_robin", "policy", &decodes, None);
        let truncated = || CAPTURE.with(|capture| capture.borrow().truncated);
        assert_eq!(captured_selections().len(), MAX_SELECTIONS);
        assert!(!truncated(), "the bound's rows are the record's whole");
        // Neither writer adds a row past it.
        selection_of("round_robin", "engine", read(&decodes), None);
        assert_eq!(captured_selections().len(), MAX_SELECTIONS);
        assert!(truncated(), "a row past the bound is not kept");
        begin_selection(&decodes);
        selection("round_robin", "policy", &decodes, None);
        assert_eq!(captured_selections().len(), MAX_SELECTIONS);
    })
    .await;
}

#[test]
fn candidate_observation_survives_dispatch_load_credit() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://worker:8000").build());
    let workers = vec![worker];
    let mut capture = Capture::default();
    capture.observe_candidates(&workers);
    workers[0].increment_load();
    assert_eq!(first_candidate(&capture)["load"], json!(0));
    assert_eq!(workers[0].load(), 1);
    assert!(capture.candidates_complete);
    assert!(capture.observation_started_ns <= capture.observation_finished_ns);
}

#[test]
fn candidate_epochs_are_replica_scoped_and_invalidated_after_failure() {
    use openai_protocol::worker::WorkerStatus;

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8000")
            .labels(std::collections::HashMap::from([(
                "cache_trace_epochs".to_string(),
                r#"["one","two"]"#.to_string(),
            )]))
            .build(),
    );
    let mut capture = Capture::default();
    capture.observe_candidates(std::slice::from_ref(&worker));
    assert_eq!(
        first_candidate(&capture)["backend_cache_epochs"],
        json!(["one", "two"])
    );
    assert!(first_candidate(&capture)["backend_cache_epoch"].is_null());
    worker.set_status(WorkerStatus::NotReady);
    worker.set_status(WorkerStatus::Ready);
    capture.observe_candidates(&[worker]);
    assert_eq!(first_candidate(&capture)["backend_cache_epochs"], json!([]));
    assert!(first_candidate(&capture)["backend_cache_epoch"].is_null());
}

#[test]
fn gateway_header_keeps_join_fields_without_candidate_state() {
    let mut evidence = json!({
        "schema": 1, "root_id": "root", "dispatch_id": "dispatch", "attempt": 1,
        "engine_ids": ["engine"], "engine_ids_complete": true, "truncated": false,
        "selections": [{"policy": "cache_aware", "origin": "policy",
            "prediction": {"source": "approximate_tree"},
            "worker": "selected-worker", "candidates": [{"worker": "candidate-worker", "load": 99}],
            "gates": [{"spill": true}]}],
    });
    let encoded = gateway_header(&evidence).unwrap();
    assert!(!encoded.contains("candidate-worker"));
    assert!(!encoded.contains("gates"));
    let decoded: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded["engine_ids"], json!(["engine"]));
    assert_eq!(decoded["attempt"], 1);
    assert_eq!(
        decoded["selections"][0]["worker"], "selected-worker",
        "the chosen worker joins the record"
    );
    evidence["root_id"] = Value::String("x".repeat(2049));
    assert!(gateway_header(&evidence).is_none());
}

fn fleet_evidence(candidates: usize) -> Value {
    let candidates: Vec<Value> = (0..candidates)
        .map(|index| {
            json!({
                "worker": format!("http://10.0.0.{index}:8000"), "load": index, "healthy": true,
                "overloaded": false, "registry_revision": 7, "backend_cache_epoch": null,
            })
        })
        .collect();
    json!({
        "schema": 1, "root_id": "root", "dispatch_id": "dispatch", "attempt": 1,
        "engine_ids": ["engine"], "engine_ids_complete": true, "truncated": false,
        "selections": [{"policy": "cache_aware", "origin": "policy", "worker": "http://10.0.0.3:8000",
            "prediction": {"source": "approximate_tree", "overlap_blocks": 4},
            "candidates": candidates, "candidates_complete": true,
            "scores": [{"worker": "http://10.0.0.3:8000", "score": 0.9}],
            "gates": [{"spill": false}]}],
        "unattributed_gates": [{"spill": true}],
    })
}

#[test]
fn sample_knob_logs_one_dispatch_in_every_n() {
    assert_eq!(parse_sample(None), 1);
    assert_eq!(parse_sample(Some("0")), 1);
    assert_eq!(parse_sample(Some("many")), 1);
    assert_eq!(parse_sample(Some(" 100 ")), 100);
    let every_request = Sampler::new(1);
    assert!((0..10).all(|_| every_request.admit()));
    let one_in_hundred = Sampler::new(100);
    let logged = (0..1000)
        .filter(|_| log_line(fleet_evidence(32), &one_in_hundred, None).is_some())
        .count();
    assert_eq!(logged, 10);
    assert!(
        Sampler::new(100).admit(),
        "the first dispatch is always logged"
    );
}

#[test]
fn byte_cap_keeps_the_decision_record_and_drops_the_candidate_lists() {
    assert_eq!(parse_max_bytes(None), None);
    assert_eq!(parse_max_bytes(Some("bytes")), None);
    assert_eq!(parse_max_bytes(Some("1024")), Some(1024));
    let full = encode_capped(fleet_evidence(32), None);
    assert!(
        full.len() > 3000,
        "32 candidates make the line {} bytes",
        full.len()
    );
    assert_eq!(encode_capped(fleet_evidence(32), Some(full.len())), full);
    let capped = encode_capped(fleet_evidence(32), Some(1024));
    assert!(capped.len() < 1024, "capped line is {} bytes", capped.len());
    let decoded: Value = serde_json::from_str(&capped).unwrap();
    assert_eq!(decoded["capped"], json!(true));
    assert_eq!(decoded["root_id"], "root");
    assert_eq!(decoded["engine_ids"], json!(["engine"]));
    let selection = &decoded["selections"][0];
    assert_eq!(selection["policy"], "cache_aware");
    assert_eq!(selection["worker"], "http://10.0.0.3:8000");
    assert_eq!(selection["prediction"]["overlap_blocks"], 4);
    assert!(selection.get("candidates").is_none());
    assert!(selection.get("scores").is_none());
    assert_eq!(
        selection["elided"],
        json!({"candidates": 32, "scores": 1, "gates": 1})
    );
    assert!(decoded.get("unattributed_gates").is_none());
    let failure = encode_capped(
        json!({"root_id": "root", "status": 503, "selections": [], "gates": [{"spill": true}], "truncated": false}),
        Some(0),
    );
    let decoded: Value = serde_json::from_str(&failure).unwrap();
    assert_eq!(decoded["status"], 503);
    assert!(decoded.get("gates").is_none());
}

#[test]
fn the_header_switch_alone_yields_the_header_and_no_log_line() {
    let every_dispatch = Sampler::new(1);
    let (header, line) = emit(fleet_evidence(2), false, true, &every_dispatch, None);
    let header: Value = serde_json::from_str(&header.unwrap()).unwrap();
    assert_eq!(header["selections"][0]["prediction"]["overlap_blocks"], 4);
    assert!(line.is_none(), "the log switch is off");
    let (header, line) = emit(fleet_evidence(2), true, false, &every_dispatch, None);
    assert!(header.is_none(), "the header switch is off");
    assert!(line.is_some());
    let (header, line) = emit(fleet_evidence(2), true, true, &every_dispatch, None);
    assert!(header.is_some() && line.is_some());
    let one_in_hundred = Sampler::new(100);
    let outcomes: Vec<_> = (0..200)
        .map(|_| emit(fleet_evidence(2), true, true, &one_in_hundred, None))
        .collect();
    assert!(
        outcomes.iter().all(|(header, _)| header.is_some()),
        "sampling the log line leaves the header on every dispatch"
    );
    assert_eq!(
        outcomes.iter().filter(|(_, line)| line.is_some()).count(),
        2
    );
}

#[test]
fn the_request_header_asks_for_this_requests_record() {
    let mut headers = HeaderMap::new();
    assert!(!requested(None));
    assert!(!requested(Some(&headers)));
    headers.insert(REQUEST_HEADER, http::HeaderValue::from_static("0"));
    assert!(!requested(Some(&headers)));
    headers.insert(REQUEST_HEADER, http::HeaderValue::from_static(" 1 "));
    assert!(requested(Some(&headers)));
}

#[tokio::test]
async fn a_requesting_scope_captures_its_request_alone() {
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://worker:50051").build());
    let workers = vec![worker];
    scope(true, async {
        assert!(enabled(), "the request asked: its evidence is captured");
        begin_selection(&workers);
        selection("random", "policy", &workers, Some(0));
        let header = dispatch(Some("root"), 0, vec!["engine".into()], true, "regular")
            .expect("the header for the request that asked");
        let header: Value = serde_json::from_str(&header).unwrap();
        assert_eq!(header["root_id"], "root");
        assert_eq!(header["selections"][0]["worker"], "grpc://worker:50051");
        assert_eq!(header["selections"][0]["policy"], "random");
    })
    .await;
    assert_eq!(
        enabled(),
        configured(),
        "outside that request only the process switches count"
    );
    scope(false, async { assert_eq!(enabled(), configured()) }).await;
}
