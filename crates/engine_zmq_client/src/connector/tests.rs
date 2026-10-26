use std::time::Duration;

use futures::StreamExt;

use super::*;
use crate::{
    codec::{decode_msgpack, encode_msgpack},
    mock_engine::{
        connect_to_frontend, default_ready_response, EngineInbound, IpcNamespace, MockEngine,
    },
    protocol::{
        handshake::EngineCoreReadyResponse,
        vllm::{
            output::{
                DpControlMessage, DpControlOutput, EngineCoreFinishReason, EngineCoreOutput,
                EngineCoreOutputs, RequestBatchOutputs,
            },
            request::{EngineCoreRequest, UtilityCall},
            stats::SchedulerStats,
        },
    },
    transport::{connect_handshake, ENGINE_CORE_DEAD_SENTINEL},
};

const TIMEOUT: Duration = Duration::from_secs(10);

/// The returned [`IpcNamespace`] owns the socket tempdir; hold it for the
/// test duration so the ipc files outlive the client and are cleaned up.
async fn connect() -> (EngineCoreClient, MockEngine, IpcNamespace) {
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    (
        EngineCoreClient::new(transport.unwrap()),
        engine.unwrap(),
        ns,
    )
}

/// An engine whose declared layout of a positional struct puts a field
/// inside the prefix this client fills is refused at connect, by name; a
/// wire whose requests are named-field maps has nothing to check.
#[tokio::test]
async fn an_engine_whose_declared_layout_moved_a_field_is_refused_by_name() {
    use crate::{
        check_engine_layouts,
        protocol::{sglang::SglangProtocol, vllm::VllmProtocol},
    };
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let mut moved = default_ready_response();
    moved.positional_layouts.insert(
        "SamplingParams".to_string(),
        ["max_new_tokens", "stop", "watermark"]
            .map(str::to_string)
            .to_vec(),
    );
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), moved),
    );
    let transport = transport.unwrap();
    let _engine = engine.unwrap();
    assert!(check_engine_layouts::<VllmProtocol>(&transport).is_ok());
    let error = check_engine_layouts::<SglangProtocol>(&transport)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(
            "wire layout mismatch: engine EngineId(0000): `SamplingParams` field 2 is \
             `stop_token_ids` on this side and `watermark` on the engine's"
        ),
        "{error}"
    );
}

/// Connect `engine_count` mock ranks behind one transport. `lockstep`
/// controls the data-parallel size they report at handshake, which is how
/// the client tells a group that pauses as a unit (vLLM MoE DP) from ranks
/// that step independently.
async fn connect_ranks(
    engine_count: usize,
    lockstep: bool,
) -> (EngineCoreClient, Vec<MockEngine>, IpcNamespace) {
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let ready = |rank: u32| EngineCoreReadyResponse {
        data_parallel_size: if lockstep { engine_count as u64 } else { 1 },
        data_parallel_rank: rank,
        ..default_ready_response()
    };
    let engines = (0..engine_count as u32).map(|rank| {
        connect_to_frontend(&handshake, EngineId::from_engine_index(rank), ready(rank))
    });
    let (transport, engines) = tokio::join!(
        connect_handshake(&handshake, engine_count, &input, &output, TIMEOUT),
        futures::future::join_all(engines),
    );
    let engines = engines.into_iter().map(Result::unwrap).collect();
    (EngineCoreClient::new(transport.unwrap()), engines, ns)
}

/// An add-request pinned to one rank.
fn request_for(request_id: &str, rank: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        data_parallel_rank: Some(rank),
        ..EngineCoreRequest::default()
    }
}

fn wave_control(engine_index: u32, control: DpControlMessage) -> Vec<Bytes> {
    let outputs = EngineCoreOutputs::DpControl(DpControlOutput {
        engine_index,
        timestamp: 0.0,
        control,
    });
    vec![Bytes::from(encode_msgpack(&outputs).unwrap())]
}

fn batch(engine_index: u32, output: EngineCoreOutput) -> Vec<Bytes> {
    let finished = output.finish_reason.map(|_| {
        let mut set = std::collections::BTreeSet::new();
        set.insert(output.request_id.clone());
        set
    });
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index,
        outputs: vec![output],
        finished_requests: finished,
        ..RequestBatchOutputs::default()
    });
    vec![Bytes::from(encode_msgpack(&outputs).unwrap())]
}

#[tokio::test]
async fn submit_streams_tokens_until_finish() {
    let (client, mut engine, _ns) = connect().await;

    let request = EngineCoreRequest {
        request_id: "req-1".to_string(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        ..EngineCoreRequest::default()
    };
    let mut stream = client.submit(request).await.unwrap();

    // Engine receives the Add and streams two chunks then a terminal output.
    let frames = engine.recv_request().await.unwrap();
    assert_eq!(frames[0].as_ref(), b"\x00");
    let received: EngineCoreRequest = decode_msgpack(frames[1].as_ref()).unwrap();
    assert_eq!(received.request_id, "req-1");

    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![10],
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![11],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.new_token_ids, vec![10]);
    assert!(!first.finished());
    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.new_token_ids, vec![11]);
    assert!(second.finished());
    // Terminal output ends the stream.
    assert!(stream.next().await.is_none());
}

/// The silence the servicer reports for a frozen engine: nothing while
/// idle, counted from the dispatch, reset by any output batch (a
/// stats-only step included), gone with the last in-flight request.
#[tokio::test]
async fn output_silence_counts_from_the_dispatch_and_ends_with_the_requests() {
    let (client, mut engine, _ns) = connect().await;
    assert_eq!(client.output_silence(), None, "nothing in flight");
    let mut stream = client.submit(request_for("req-1", 0)).await.unwrap();
    engine.recv_request().await.unwrap();
    let at_dispatch = client.output_silence().expect("a request is in flight");
    assert!(at_dispatch < Duration::from_millis(500), "{at_dispatch:?}");
    assert!(
        !client.reports_steps(),
        "no step without a token seen yet: a token-batch wire until shown otherwise"
    );
    assert_eq!(
        client.prefill_pending_tokens(),
        3,
        "the request's three prompt tokens are the engine's prefill in hand"
    );
    tokio::time::sleep(Duration::from_millis(80)).await;
    let grown = client.output_silence().expect("still in flight");
    assert!(
        grown >= Duration::from_millis(70),
        "grows while the engine is silent: {grown:?}"
    );

    // A stats-only step is output: the engine is alive, if quiet.
    let stats_only = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        scheduler_stats: Some(SchedulerStats {
            num_running_reqs: 1,
            ..Default::default()
        }),
        ..Default::default()
    });
    engine
        .send_output(vec![Bytes::from(encode_msgpack(&stats_only).unwrap())])
        .await
        .unwrap();
    let deadline = Instant::now() + TIMEOUT;
    while client.engine_load(0).is_none() {
        assert!(Instant::now() < deadline, "stats never arrived");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let after_stats = client.output_silence().expect("still in flight");
    assert!(
        after_stats < grown,
        "reset by the step: {after_stats:?} after {grown:?}"
    );
    assert!(
        client.reports_steps(),
        "a stats-only step shows the wire reports every step"
    );
    assert_eq!(
        client.prefill_pending_tokens(),
        3,
        "a step without an output for the request leaves its prompt pending"
    );

    // The terminal output ends the request: nothing in flight, no silence.
    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![11],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    let last = stream.next().await.unwrap().unwrap();
    assert!(last.finished());
    assert!(stream.next().await.is_none());
    assert_eq!(client.output_silence(), None, "nothing in flight again");
    assert_eq!(
        client.prefill_pending_tokens(),
        0,
        "its first output settled the prompt"
    );
}

/// A stream that outlives its client is truncated, not complete: it must
/// report the truncation rather than end like a finished generation.
#[tokio::test]
async fn dropping_the_client_errors_live_streams() {
    let (client, mut engine, _ns) = connect().await;
    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "orphan".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        })
        .await
        .unwrap();
    engine.recv_request().await.unwrap();

    drop(client);

    let error = stream
        .next()
        .await
        .expect("a terminal error item")
        .expect_err("truncation is not a clean end of stream");
    assert!(
        matches!(&error, Error::RequestStreamClosed { request_id } if request_id == "orphan"),
        "unexpected error: {error}"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn scheduler_stats_surface_as_load_signal() {
    let (client, mut engine, _ns) = connect().await;
    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "r".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    engine.recv_request().await.unwrap();

    // Stats ride a mid-stream chunk: the request is still in flight, so
    // the snapshot is current and must be stored verbatim. (A terminal
    // batch's snapshot predates its own finish; once the rank empties,
    // the queue counts are zeroed — covered separately.)
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "r".into(),
            new_token_ids: vec![7],
            ..Default::default()
        }],
        scheduler_stats: Some(SchedulerStats {
            num_running_reqs: 3,
            num_waiting_reqs: 5,
            ..Default::default()
        }),
        ..Default::default()
    });
    engine
        .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
        .await
        .unwrap();

    let chunk = stream.next().await.expect("mid-stream chunk").unwrap();
    assert!(!chunk.finished());
    let load = client.engine_load(0).expect("load recorded");
    assert_eq!(load.num_running, 3);
    assert_eq!(load.num_waiting, 5);
}

#[tokio::test]
async fn unpinned_requests_prefer_the_least_loaded_engine() {
    let (client, mut engines, _ns) = connect_ranks(2, false).await;

    // Make engine 0 report load via a pinned warm-up request that stays
    // in flight (a finished-and-emptied rank has its queue counts zeroed,
    // because a terminal batch's snapshot predates its own finish);
    // engine 1 never reports and therefore scores zero.
    let mut stream = client.submit(request_for("warm", 0)).await.unwrap();
    engines[0].recv_request().await.unwrap();
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "warm".into(),
            new_token_ids: vec![7],
            ..Default::default()
        }],
        scheduler_stats: Some(SchedulerStats {
            num_running_reqs: 3,
            num_waiting_reqs: 5,
            ..Default::default()
        }),
        ..Default::default()
    });
    engines[0]
        .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
        .await
        .unwrap();
    let chunk = stream.next().await.expect("warm chunk").unwrap();
    assert!(!chunk.finished());
    assert!(client.engine_load(0).is_some(), "warm-up load recorded");

    // An unpinned request must now land on the idle engine 1, not
    // engines.first().
    let _stream = client
        .submit(EngineCoreRequest {
            request_id: "cold".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        })
        .await
        .unwrap();
    let inbound = tokio::time::timeout(TIMEOUT, engines[1].recv())
        .await
        .expect("unpinned request should route to the least-loaded engine")
        .unwrap();
    match inbound {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "cold"),
        other => panic!("expected Add on engine 1, got {other:?}"),
    }
}

#[tokio::test]
async fn a_cold_burst_spreads_across_engines() {
    // No engine has reported load yet: the in-flight floor plus the
    // rotating scan start must spread a burst instead of dogpiling the
    // first engine (which is what a pure snapshot-min would do).
    let (client, mut engines, _ns) = connect_ranks(2, false).await;
    let unpinned = |id: &str| EngineCoreRequest {
        request_id: id.into(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        ..EngineCoreRequest::default()
    };
    let _first = client.submit(unpinned("burst-0")).await.unwrap();
    let _second = client.submit(unpinned("burst-1")).await.unwrap();

    for engine in &mut engines {
        let inbound = tokio::time::timeout(TIMEOUT, engine.recv())
            .await
            .expect("each engine should receive exactly one of the burst")
            .unwrap();
        match inbound {
            EngineInbound::Add(request) => {
                assert!(request.request_id.starts_with("burst-"));
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn failed_submit_releases_its_reservation() {
    // A rejected submit (duplicate request id) must release the in-flight
    // reservation selection took, or the failed attempt permanently
    // penalizes that engine's score.
    let (client, mut engines, _ns) = connect_ranks(2, false).await;
    let unpinned = |id: &str| EngineCoreRequest {
        request_id: id.into(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        ..EngineCoreRequest::default()
    };
    let _held = client.submit(unpinned("dup")).await.unwrap();
    engines[0].recv().await.unwrap();
    // Second submit with the same id fails at registration; its
    // reservation must roll back, leaving the counts (1, 0).
    assert!(client.submit(unpinned("dup")).await.is_err());

    // With counts (1, 0) the next request must land on engine 1. If the
    // failed attempt leaked its reservation the counts would read (1, 1)
    // and rotation could send this to engine 0.
    let _next = client.submit(unpinned("after")).await.unwrap();
    match tokio::time::timeout(TIMEOUT, engines[1].recv())
        .await
        .expect("request should route to the unloaded engine")
        .unwrap()
    {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "after"),
        other => panic!("expected Add on engine 1, got {other:?}"),
    }
}

#[tokio::test]
async fn retiring_one_request_twice_frees_only_its_own_slot() {
    // A request has several racing ways to finish: the explicit abort API,
    // the dropped stream's auto-abort, and the engine's own finished
    // report. Counting each one would retire slots the request never held
    // and leave the rank looking emptier than it is; slots are keyed by
    // request id so only the first retirement lands.
    let (client, mut engines, _ns) = connect_ranks(2, false).await;
    let pinned = |id: &str| EngineCoreRequest {
        request_id: id.into(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        data_parallel_rank: Some(0),
        ..EngineCoreRequest::default()
    };
    let doomed = client.submit(pinned("doomed")).await.unwrap();
    let _held = client.submit(pinned("held")).await.unwrap();
    engines[0].recv().await.unwrap();
    engines[0].recv().await.unwrap();

    // Abort retires "doomed"; dropping its unfinished stream sends the
    // auto-abort for the same id behind it.
    client
        .abort(&engines_id(&client, 0), "doomed")
        .await
        .unwrap();
    drop(doomed);
    for _ in 0..2 {
        tokio::time::timeout(TIMEOUT, engines[0].recv())
            .await
            .expect("both aborts should reach the engine")
            .unwrap();
    }

    // Engine 0 still holds "held". An unpinned request must therefore go
    // to the idle engine 1 — it would go to 0 if the second retirement had
    // freed "held"'s slot too.
    let _next = client
        .submit(EngineCoreRequest {
            request_id: "next".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        })
        .await
        .unwrap();
    match tokio::time::timeout(TIMEOUT, engines[1].recv())
        .await
        .expect("request should route to the idle engine")
        .unwrap()
    {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "next"),
        other => panic!("expected Add on engine 1, got {other:?}"),
    }
}

#[tokio::test]
async fn an_emptied_rank_sheds_its_stale_queue_counts() {
    // Engines report load only on output batches, and a terminal batch's
    // snapshot is sampled before its own finish commits — so the last
    // thing an idle rank ever says is "still busy". The client routed
    // every request: once a rank's in-flight set empties, its stored
    // queue counts must be zeroed or the rank is shunned forever.
    let (client, mut engines, _ns) = connect_ranks(2, false).await;

    // Rank 0 runs one request to completion; its terminal batch carries
    // the stale pre-commit snapshot (3 running, 5 waiting).
    let mut stream = client.submit(request_for("last", 0)).await.unwrap();
    engines[0].recv_request().await.unwrap();
    let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
        engine_index: 0,
        outputs: vec![EngineCoreOutput {
            request_id: "last".into(),
            new_token_ids: vec![7],
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        }],
        scheduler_stats: Some(SchedulerStats {
            num_running_reqs: 3,
            num_waiting_reqs: 5,
            ..Default::default()
        }),
        finished_requests: Some(std::collections::BTreeSet::from(["last".to_string()])),
        ..Default::default()
    });
    engines[0]
        .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    let load = client.engine_load(0).expect("snapshot stored");
    assert_eq!((load.num_running, load.num_waiting), (0, 0));

    // Rank 1 holds one live request. The next unpinned request must go to
    // the genuinely idle rank 0 — trusting the stale snapshot would send
    // it behind rank 1's real work.
    let _held = client.submit(request_for("held", 1)).await.unwrap();
    engines[1].recv().await.unwrap();
    let _next = client
        .submit(EngineCoreRequest {
            request_id: "next".into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        })
        .await
        .unwrap();
    match tokio::time::timeout(TIMEOUT, engines[0].recv())
        .await
        .expect("request should route to the emptied rank")
        .unwrap()
    {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "next"),
        other => panic!("expected Add on engine 0, got {other:?}"),
    }
}

#[tokio::test]
async fn selection_and_release_race_without_deadlocking() {
    // Selection reads the reported load and the in-flight sets together;
    // release retires an id and, when that empties the rank, zeroes its
    // queue counts. While those maps had a mutex each, the two paths took
    // them in opposite orders, so a submit racing the release of a rank's
    // last request wedged both threads forever (parking_lot: no timeout,
    // no poisoning). Under one lock the inversion is unexpressible; this
    // stress test trips its deadline against the two-mutex version.
    const THREADS: usize = 8;
    const ITERATIONS: usize = 2_000;

    let (client, _engines, _ns) = connect_ranks(4, false).await;
    let inner = client.inner.clone();
    let completed = Arc::new(AtomicUsize::new(0));

    // Plain threads, not tasks: the racing methods are synchronous, and a
    // regression must not be able to wedge the runtime this test is
    // waiting on.
    let workers: Vec<_> = (0..THREADS)
        .map(|thread| {
            let inner = inner.clone();
            let completed = completed.clone();
            std::thread::spawn(move || {
                for iteration in 0..ITERATIONS {
                    let request_id = format!("race-{thread}-{iteration}");
                    let engine_id = inner.select_engine(None, &request_id).unwrap();
                    inner.release_one(&engine_id, &request_id);
                }
                completed.fetch_add(1, Ordering::Release);
            })
        })
        .collect();

    let deadline = Instant::now() + TIMEOUT;
    while completed.load(Ordering::Acquire) < THREADS {
        assert!(
            Instant::now() < deadline,
            "selection racing release deadlocked"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    for worker in workers {
        worker.join().unwrap();
    }

    // Every reservation was released, so no rank carries a phantom floor.
    let routing = inner.routing.lock();
    assert!(routing.inflight.values().all(HashSet::is_empty));
}

/// The `EngineId` of one connected rank, by index.
fn engines_id(client: &EngineCoreClient, engine_index: u32) -> EngineId {
    client
        .engines()
        .iter()
        .find(|engine| engine.engine_id.engine_index() == Some(engine_index))
        .expect("connected rank")
        .engine_id
        .clone()
}

#[tokio::test]
async fn dropping_stream_sends_abort() {
    let (client, mut engine, _ns) = connect().await;
    let stream = client
        .submit(EngineCoreRequest {
            request_id: "req-abort".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    // Consume the Add frame.
    let add = engine.recv_request().await.unwrap();
    assert_eq!(add[0].as_ref(), b"\x00");

    drop(stream); // not finished -> auto-abort

    let abort = engine.recv_request().await.unwrap();
    assert_eq!(abort[0].as_ref(), b"\x01"); // Abort
    let ids: Vec<String> = decode_msgpack(abort[1].as_ref()).unwrap();
    assert_eq!(ids, vec!["req-abort".to_string()]);
}

#[tokio::test]
async fn engine_dead_fails_inflight_requests() {
    let (client, mut engine, _ns) = connect().await;
    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "r".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    engine.recv_request().await.unwrap();

    engine
        .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
        .await
        .unwrap();

    let result = stream.next().await.expect("an error item");
    assert!(matches!(result, Err(Error::Shared(_))));
}

#[tokio::test]
async fn persistent_decode_errors_fail_inflight_requests() {
    let (client, mut engine, _ns) = connect().await;
    let mut stream = client
        .submit(EngineCoreRequest {
            request_id: "r".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    engine.recv_request().await.unwrap();

    // One bad message is tolerated; a run of them is a schema mismatch the
    // client cannot recover from, so the request fails instead of hanging.
    for _ in 0..MAX_CONSECUTIVE_DECODE_ERRORS {
        engine
            .send_output(vec![Bytes::from_static(b"\xc1")])
            .await
            .unwrap();
    }

    let result = stream.next().await.expect("an error item");
    assert!(matches!(result, Err(Error::Shared(_))), "{result:?}");
}

/// The generic connector drives the TokenSpeed protocol over the same shared
/// transport: it frames a tagged `TokenizedGenerateReqInput` and streams
/// `BatchTokenIDOutSlim` batches back until a finish reason is seen.
#[tokio::test]
async fn tokenspeed_client_submits_and_streams() {
    use crate::protocol::tokenspeed::{
        output::BatchTokenIDOutSlim,
        request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
        sampling::SamplingParams,
    };

    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    // Held for the test duration so the ipc files outlive the client.
    let _ns = ns;
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = TokenSpeedClient::new(transport.unwrap());
    let mut engine = engine.unwrap();

    let request = TokenizedGenerateReqInput {
        rid: "ts-1".to_string(),
        input_ids: vec![1, 2, 3],
        sampling_params: SamplingParams {
            max_new_tokens: Some(4),
            ..SamplingParams::default()
        },
        stream: true,
        ..TokenizedGenerateReqInput::default()
    };
    let mut stream = client.submit(request).await.unwrap();

    // Engine sees the Add frame + the positional-array payload.
    let frames = engine.recv_request().await.unwrap();
    assert_eq!(
        TokenSpeedRequestType::from_frame(frames[0].as_ref()),
        Some(TokenSpeedRequestType::Add)
    );
    let received: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
    assert_eq!(received.rid, "ts-1");
    assert_eq!(received.input_ids, vec![1, 2, 3]);

    let chunk = BatchTokenIDOutSlim {
        rids: vec!["ts-1".into()],
        output_ids: vec![vec![10]],
        finished_reasons: vec![String::new()],
        prompt_tokens: vec![3],
        completion_tokens: vec![1],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        ..Default::default()
    };
    let done = BatchTokenIDOutSlim {
        rids: vec!["ts-1".into()],
        output_ids: vec![vec![11]],
        finished_reasons: vec!["stop".into()],
        prompt_tokens: vec![3],
        completion_tokens: vec![2],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        ..Default::default()
    };
    engine
        .send_output(vec![Bytes::from(encode_msgpack(&chunk).unwrap())])
        .await
        .unwrap();
    engine
        .send_output(vec![Bytes::from(encode_msgpack(&done).unwrap())])
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.output_ids, vec![10]);
    assert!(!first.finished());
    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.output_ids, vec![11]);
    assert_eq!(second.finish_reason.as_deref(), Some("stop"));
    assert!(second.finished());
    // Terminal output ends the stream.
    assert!(stream.next().await.is_none());
}

/// A launcher that knows the engine's scheduler statistics are on tells
/// the client so: the wire counts as step-reporting from the start, before
/// any batch, and the first request's silence counts from its dispatch;
/// a client told nothing waits for a stats-only step batch to say so.
#[tokio::test]
async fn a_client_told_its_engine_reports_steps_is_step_reporting_before_any_batch() {
    let (client, mut engine, _ns) = connect().await;
    assert!(
        !client.reports_steps(),
        "told nothing and shown nothing: a token-batch wire until shown otherwise"
    );
    client.note_reports_steps();
    assert!(
        client.reports_steps(),
        "the launcher's word holds before any batch"
    );
    let _stream = client.submit(request_for("req-1", 0)).await.unwrap();
    engine.recv_request().await.unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;
    let silence = client.output_silence().expect("a request is in flight");
    assert!(
        silence >= Duration::from_millis(70) && client.reports_steps(),
        "the first request's silence is the scheduler's own from its dispatch: {silence:?}"
    );
    assert_eq!(client.prefill_pending_tokens(), 3);
}

/// The silence the servicer reports is counted over the requests the
/// engine reports every step. A request it reports at intervals (one that
/// does not stream, one streamed every so many tokens) is quiet while
/// healthy: it counts no silence however long it runs, though its prompt
/// counts as pending prefill; a streaming request beside it counts from
/// its own dispatch, and takes the silence with it when it ends.
#[tokio::test]
async fn a_request_reported_at_intervals_counts_no_silence() {
    use crate::protocol::tokenspeed::{
        output::BatchTokenIDOutSlim, request::TokenizedGenerateReqInput, sampling::SamplingParams,
    };
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let _ns = ns;
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = TokenSpeedClient::new(transport.unwrap());
    let mut engine = engine.unwrap();
    let request =
        |rid: &str, input_ids: Vec<u32>, stream: bool, stream_interval| TokenizedGenerateReqInput {
            rid: rid.to_string(),
            input_ids,
            sampling_params: SamplingParams {
                max_new_tokens: Some(4),
                stream_interval,
                ..SamplingParams::default()
            },
            stream,
            ..TokenizedGenerateReqInput::default()
        };
    let done = |rid: &str, prompt_tokens: u32| BatchTokenIDOutSlim {
        rids: vec![rid.to_string()],
        output_ids: vec![vec![11]],
        finished_reasons: vec!["stop".into()],
        prompt_tokens: vec![prompt_tokens],
        completion_tokens: vec![1],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        ..Default::default()
    };

    // A request that does not stream is reported at its end: it is owed
    // nothing in between, so it counts no silence, however long it runs.
    let mut unstreamed = client
        .submit(request("ts-end", vec![1, 2, 3], false, None))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(client.output_silence(), None, "reported at its end only");
    assert_eq!(
        client.prefill_pending_tokens(),
        3,
        "its prompt is pending prefill all the same"
    );
    // So is one streamed every eight tokens.
    let mut spaced = client
        .submit(request("ts-spaced", vec![4, 5], true, Some(8)))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    assert_eq!(client.output_silence(), None, "streamed at an interval");
    assert_eq!(client.prefill_pending_tokens(), 5);
    // A streaming request is reported every step: the silence counts
    // from its dispatch, not from the quiet requests' earlier ones.
    let mut streamed = client
        .submit(request("ts-step", vec![6], true, None))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    let at_dispatch = client
        .output_silence()
        .expect("a request reported every step is in flight");
    assert!(
        at_dispatch < Duration::from_millis(150),
        "counted from its own dispatch: {at_dispatch:?}"
    );
    assert_eq!(client.prefill_pending_tokens(), 6);
    // Its end takes the silence with it, though the other two go on.
    engine
        .send_output(vec![Bytes::from(
            encode_msgpack(&done("ts-step", 1)).unwrap(),
        )])
        .await
        .unwrap();
    assert!(streamed.next().await.unwrap().unwrap().finished());
    assert!(streamed.next().await.is_none());
    assert_eq!(
        client.output_silence(),
        None,
        "the requests left are reported at intervals"
    );
    assert_eq!(
        client.prefill_pending_tokens(),
        5,
        "the ended request's prompt settled; the others' stand"
    );
    // The other two end: nothing pending, nothing in flight.
    for (rid, stream, prompt_tokens) in [
        ("ts-end", &mut unstreamed, 3),
        ("ts-spaced", &mut spaced, 2),
    ] {
        engine
            .send_output(vec![Bytes::from(
                encode_msgpack(&done(rid, prompt_tokens)).unwrap(),
            )])
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
        assert!(stream.next().await.is_none());
    }
    assert_eq!(client.prefill_pending_tokens(), 0);
    assert_eq!(client.output_silence(), None);
}

/// The same on the SGLang wire, embeddings included: a generation that
/// does not stream, or streams every so many tokens, counts no silence;
/// an embedding (one forward, one output) and a streaming generation at
/// an interval of one count from their own dispatch and take the silence
/// with them when they end.
#[tokio::test]
async fn an_sglang_request_reported_at_intervals_counts_no_silence() {
    use crate::protocol::sglang::{
        output::{BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput},
        request::{SglangRequest, TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
        sampling::SamplingParams,
    };
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let _ns = ns;
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = SglangClient::new(transport.unwrap());
    let mut engine = engine.unwrap();
    let generate = |rid: &str, input_ids: Vec<u32>, stream: bool, stream_interval| {
        SglangRequest::Generate(TokenizedGenerateReqInput {
            rid: rid.to_string(),
            input_ids,
            sampling_params: SamplingParams {
                max_new_tokens: Some(4),
                stream_interval,
                ..SamplingParams::default()
            },
            stream,
            ..TokenizedGenerateReqInput::default()
        })
    };
    let done = |rid: &str, prompt_tokens: u32| BatchTokenIDSlimOutput {
        rids: vec![rid.to_string()],
        output_ids: vec![vec![11]],
        finished_reasons: vec!["stop".into()],
        finished_messages: vec![None],
        finished_matched: vec![None],
        prompt_tokens: vec![prompt_tokens],
        completion_tokens: vec![1],
        cached_tokens: vec![0],
        output_token_logprobs_val: vec![vec![]],
        output_token_logprobs_idx: vec![vec![]],
        finished_status: vec![None],
        ..Default::default()
    };

    // A generation that does not stream is reported at its end: no
    // silence, however long it runs; its prompt is pending all the same.
    let mut unstreamed = client
        .submit(generate("sg-end", vec![1, 2, 3], false, None))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(client.output_silence(), None, "reported at its end only");
    assert_eq!(client.prefill_pending_tokens(), 3);
    // So is one streamed every eight tokens.
    let mut spaced = client
        .submit(generate("sg-spaced", vec![4, 5], true, Some(8)))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    assert_eq!(client.output_silence(), None, "streamed at an interval");
    assert_eq!(client.prefill_pending_tokens(), 5);
    // An embedding is owed its one output: it counts from its dispatch.
    let mut embed = client
        .submit(SglangRequest::Embed(TokenizedEmbeddingReqInput {
            rid: "sg-embed".to_string(),
            input_ids: vec![6, 7],
            ..TokenizedEmbeddingReqInput::default()
        }))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    let at_dispatch = client
        .output_silence()
        .expect("an embedding in flight is owed its output");
    assert!(
        at_dispatch < Duration::from_millis(150),
        "counted from its own dispatch: {at_dispatch:?}"
    );
    assert_eq!(client.prefill_pending_tokens(), 7);
    // Its result ends it, and the silence with it.
    let result = BatchEmbeddingSlimOutput {
        rids: vec!["sg-embed".to_string()],
        embeddings: vec![vec![0.5]],
        prompt_tokens: vec![2],
        cached_tokens: vec![0],
        finished_reasons: vec!["stop".into()],
        finished_messages: vec![None],
        finished_status: vec![None],
        ..Default::default()
    };
    engine
        .send_output(vec![Bytes::from(encode_msgpack(&result).unwrap())])
        .await
        .unwrap();
    assert!(embed.next().await.unwrap().unwrap().finished());
    assert!(embed.next().await.is_none());
    assert_eq!(
        client.output_silence(),
        None,
        "the generations left are reported at intervals"
    );
    assert_eq!(client.prefill_pending_tokens(), 5);
    // A streaming generation at the default interval is reported every
    // step: it counts from its own dispatch, not from the embedding's.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let mut streamed = client
        .submit(generate("sg-step", vec![8], true, None))
        .await
        .unwrap();
    engine.recv_request().await.unwrap();
    let at_dispatch = client
        .output_silence()
        .expect("a streaming generation at an interval of one is owed every step");
    assert!(
        at_dispatch < Duration::from_millis(150),
        "counted from its own dispatch: {at_dispatch:?}"
    );
    assert_eq!(client.prefill_pending_tokens(), 6);
    engine
        .send_output(vec![Bytes::from(
            encode_msgpack(&done("sg-step", 1)).unwrap(),
        )])
        .await
        .unwrap();
    assert!(streamed.next().await.unwrap().unwrap().finished());
    assert!(streamed.next().await.is_none());
    assert_eq!(client.output_silence(), None);
    assert_eq!(client.prefill_pending_tokens(), 5);
    // The quiet two end: nothing pending, nothing in flight.
    for (rid, stream, prompt_tokens) in [
        ("sg-end", &mut unstreamed, 3),
        ("sg-spaced", &mut spaced, 2),
    ] {
        engine
            .send_output(vec![Bytes::from(
                encode_msgpack(&done(rid, prompt_tokens)).unwrap(),
            )])
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
        assert!(stream.next().await.is_none());
    }
    assert_eq!(client.prefill_pending_tokens(), 0);
    assert_eq!(client.output_silence(), None);
}

/// A lockstep group is paused when the client connects, so the first submit
/// must wake every rank — including the one taking the request, so the
/// whole group shares one wave number.
#[tokio::test]
async fn first_submit_wakes_the_paused_lockstep_group() {
    let (client, mut engines, _ns) = connect_ranks(2, true).await;

    let _stream = client.submit(request_for("req-1", 1)).await.unwrap();

    match engines[1].recv().await.unwrap() {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
        other => panic!("rank 1 expected the Add, got {other:?}"),
    }
    assert_eq!(next_wake(&mut engines[0]).await, 1);
    assert_eq!(next_wake(&mut engines[1]).await, 1);
}

/// Every submit to a lockstep group wakes it with a FRESH wave number,
/// even submits that race the group's drain.
///
/// This is the load-bearing property. The ranks bump their own
/// `current_wave` as they park, and an engine drops any wave below its
/// own, so a wake that reuses the last number the client saw is silently
/// discarded — the holder's peers stay parked, never join its all-reduce,
/// and vLLM kills the engine on its own RPC timeout. The client's clock is
/// therefore monotonic and independent of when the `wave_complete`
/// notifications arrive.
#[tokio::test]
async fn every_wake_names_a_wave_the_ranks_will_accept() {
    let (client, mut engines, _ns) = connect_ranks(2, true).await;

    let _first = client.submit(request_for("req-1", 1)).await.unwrap();
    let first = next_wake(&mut engines[0]).await;
    assert_eq!(next_wake(&mut engines[1]).await, first);

    // The group drains the wave and parks, moving on to `first + 1` — but
    // the client has not heard about it yet. Its next wake must still
    // outrank the engines' self-increment.
    let _second = client.submit(request_for("req-2", 0)).await.unwrap();
    let second = next_wake(&mut engines[0]).await;
    assert!(
        second > first,
        "wake {second} does not outrank the parked ranks at {}",
        first + 1
    );
    assert_eq!(next_wake(&mut engines[1]).await, second);

    // A rank can also report a wave AHEAD of the client's clock — it kept
    // its `current_wave` across a gateway restart, say. Folding that in is
    // the other half of the same invariant: the next wake must clear the
    // highest wave any rank is known to hold.
    let ahead = second + 5;
    engines[0]
        .send_output(wave_control(0, DpControlMessage::WaveComplete(ahead)))
        .await
        .unwrap();
    tokio::time::timeout(TIMEOUT, async {
        while *client.inner.wave.as_ref().unwrap().lock() <= ahead {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the reported wave never reached the clock");

    let _third = client.submit(request_for("req-3", 1)).await.unwrap();
    let third = next_wake(&mut engines[0]).await;
    assert!(
        third > ahead,
        "wake {third} does not clear the rank already on {ahead}"
    );
}

/// Read one engine's inbound queue up to its next wake, returning the wave
/// it names.
async fn next_wake(engine: &mut MockEngine) -> u64 {
    loop {
        match engine.recv().await.unwrap() {
            EngineInbound::StartDpWave {
                wave,
                exclude_engine_index,
            } => {
                // No rank is excluded: the whole group shares one wave.
                assert_eq!(exclude_engine_index, None);
                return wave;
            }
            EngineInbound::Add(_) => continue,
            other => panic!("expected a wake or an Add, got {other:?}"),
        }
    }
}

/// Independent ranks (dense DP) never pause as a group, so they are never
/// woken — each rank only ever sees its own requests.
#[tokio::test]
async fn independent_ranks_are_never_woken() {
    let (client, mut engines, _ns) = connect_ranks(2, false).await;

    let _first = client.submit(request_for("req-1", 1)).await.unwrap();
    let _second = client.submit(request_for("req-2", 0)).await.unwrap();

    // Rank 0's first message is its own Add, not a wake for rank 1's.
    match engines[0].recv().await.unwrap() {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "req-2"),
        other => panic!("independent rank expected only its Add, got {other:?}"),
    }
}

/// A rank that took a request for an already-drained wave asks the client
/// to start the next one (vLLM `start_wave`). Without a DP coordinator
/// nobody else answers that request, so the client must broadcast the wake
/// itself — to EVERY rank, at a wave the asking rank will still accept.
#[tokio::test]
async fn a_ranks_start_wave_request_wakes_the_whole_group() {
    let (_client, mut engines, _ns) = connect_ranks(2, true).await;

    // The rank is ahead of the client's clock (it self-incremented on its
    // way out of the last wave), which is exactly the stale-wake trap.
    let requested = 7;
    engines[0]
        .send_output(wave_control(0, DpControlMessage::StartWave(requested)))
        .await
        .unwrap();

    assert!(next_wake(&mut engines[0]).await >= requested);
    assert!(next_wake(&mut engines[1]).await >= requested);
}

/// A drain notification is bookkeeping, not a wake: `wave_complete` folds
/// into the clock but must NOT restart the group, or the ranks would never
/// stay parked and an idle group would spin.
#[tokio::test]
async fn wave_complete_updates_the_clock_without_waking() {
    let (client, mut engines, _ns) = connect_ranks(2, true).await;

    engines[0]
        .send_output(wave_control(0, DpControlMessage::WaveComplete(3)))
        .await
        .unwrap();
    tokio::time::timeout(TIMEOUT, async {
        while *client.inner.wave.as_ref().unwrap().lock() <= 3 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the drained wave never reached the clock");

    // The next submit is the first thing that may wake the group, and it
    // must clear the wave the ranks parked on.
    let _stream = client.submit(request_for("req-1", 1)).await.unwrap();
    assert!(next_wake(&mut engines[0]).await > 3);
}

/// Independent ranks keep no wave clock, so a stray wave notification (a
/// mislabeled engine, say) must be ignored rather than panic or start
/// waking ranks that never park.
#[tokio::test]
async fn wave_notifications_from_independent_ranks_are_ignored() {
    let (client, mut engines, _ns) = connect_ranks(2, false).await;

    engines[0]
        .send_output(wave_control(0, DpControlMessage::StartWave(2)))
        .await
        .unwrap();

    let _stream = client.submit(request_for("req-1", 0)).await.unwrap();
    match engines[0].recv().await.unwrap() {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
        other => panic!("independent rank expected only its Add, got {other:?}"),
    }
    assert!(
        client.inner.wave.is_none(),
        "no clock for independent ranks"
    );
}

/// The utility RPC: the call is framed as vLLM's client frames it, goes to
/// the named rank, and its reply resolves the caller by call id.
#[tokio::test]
async fn call_utility_resolves_the_reply_by_call_id() {
    let (client, mut engine, _ns) = connect().await;
    let engine_id = engines_id(&client, 0);
    let args = vec![OpaqueValue::from(false), OpaqueValue::from(false)];
    let (result, call) = tokio::join!(
        client.call_utility(&engine_id, "reset_prefix_cache", args.clone(), TIMEOUT),
        async {
            let frames = engine.recv_request().await.unwrap();
            assert_eq!(frames[0].as_ref(), b"\x03");
            let call: UtilityCall = decode_msgpack(frames[1].as_ref()).unwrap();
            engine
                .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(true)))
                .await
                .unwrap();
            call
        }
    );
    assert_eq!(result.unwrap(), OpaqueValue::from(true));
    assert_eq!(call.client_index, 0);
    assert_eq!(call.method, "reset_prefix_cache");
    assert_eq!(call.args, args);
    assert!(client.inner.registry.lock().utility_calls.is_empty());

    // Calls are told apart by id: the next one gets a fresh id and its own
    // reply.
    let (result, next) = tokio::join!(
        client.call_utility(&engine_id, "is_sleeping", Vec::new(), TIMEOUT),
        async {
            let EngineInbound::Utility(next) = engine.recv().await.unwrap() else {
                panic!("expected a utility call");
            };
            engine
                .send_utility_reply(0, next.call_id, Ok(OpaqueValue::from(false)))
                .await
                .unwrap();
            next
        }
    );
    assert_ne!(next.call_id, call.call_id);
    assert_eq!(result.unwrap(), OpaqueValue::from(false));
}

/// The engine's failure message is the call's error.
#[tokio::test]
async fn call_utility_surfaces_the_engine_failure_message() {
    let (client, mut engine, _ns) = connect().await;
    let engine_id = engines_id(&client, 0);
    let failure = "Call to reset_prefix_cache method failed: boom";
    let (result, ()) = tokio::join!(
        client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
        async {
            let EngineInbound::Utility(call) = engine.recv().await.unwrap() else {
                panic!("expected a utility call");
            };
            engine
                .send_utility_reply(0, call.call_id, Err(failure.to_string()))
                .await
                .unwrap();
        }
    );
    assert!(
        matches!(result, Err(Error::UtilityCallFailed { ref message }) if message == failure),
        "{result:?}"
    );
}

/// An unanswered call times out and leaves nothing behind; the reply that
/// arrives late is dropped without disturbing later calls.
#[tokio::test]
async fn call_utility_times_out_and_drops_a_late_reply() {
    let (client, mut engine, _ns) = connect().await;
    let engine_id = engines_id(&client, 0);
    let result = client
        .call_utility(
            &engine_id,
            "reset_prefix_cache",
            Vec::new(),
            Duration::from_millis(200),
        )
        .await;
    assert!(
        matches!(result, Err(Error::UtilityTimeout { .. })),
        "{result:?}"
    );
    assert!(client.inner.registry.lock().utility_calls.is_empty());

    let EngineInbound::Utility(stale) = engine.recv().await.unwrap() else {
        panic!("expected a utility call");
    };
    engine
        .send_utility_reply(0, stale.call_id, Ok(OpaqueValue::from(true)))
        .await
        .unwrap();

    let (result, ()) = tokio::join!(
        client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
        async {
            let EngineInbound::Utility(call) = engine.recv().await.unwrap() else {
                panic!("expected a utility call");
            };
            assert_ne!(call.call_id, stale.call_id);
            engine
                .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(false)))
                .await
                .unwrap();
        }
    );
    assert_eq!(result.unwrap(), OpaqueValue::from(false));
}

/// A pending call fails with the requests when the engine dies.
#[tokio::test]
async fn engine_dead_fails_pending_utility_calls() {
    let (client, mut engine, _ns) = connect().await;
    let engine_id = engines_id(&client, 0);
    let (result, ()) = tokio::join!(
        client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
        async {
            engine.recv().await.unwrap();
            engine
                .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
                .await
                .unwrap();
        }
    );
    assert!(matches!(result, Err(Error::Shared(_))), "{result:?}");
    assert!(!client.is_alive());
}
/// The window a caller waits for the engine to accept a frame in the
/// frozen-core cases below: short, so they run in milliseconds.
const SEND_WINDOW: Duration = Duration::from_millis(300);

/// A 64 MiB typed tensor exercises encoding and exceeds socket buffers.
fn large_request(request_id: &str) -> (EngineCoreRequest, Vec<Bytes>) {
    use crate::{
        codec::tensor::WireTensor,
        protocol::vllm::multimodal::{
            MmBatchedField, MmFeatureSpec, MmField, MmFieldElem, MmKwargValue, PlaceholderRange,
        },
    };

    let mut request = request_for(request_id, 0);
    request.mm_features = Some(
        vec![MmFeatureSpec {
            data: Some(std::collections::BTreeMap::from([(
                "pixels".into(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(WireTensor::from_raw(
                        "uint8",
                        vec![64 << 20],
                        vec![0; 64 << 20],
                    ))),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            )])),
            modality: "image".into(),
            identifier: "large-image".into(),
            mm_position: PlaceholderRange {
                offset: 0,
                length: 1,
                is_embed: None,
            },
            mm_hash: None,
        }]
        .into(),
    );
    (request, Vec::new())
}

/// Queue large frames until our ownership budget stalls the writer, without
/// depending on the upstream's private channel capacity. No engine reads.
async fn saturate_input(client: &EngineCoreClient) {
    for _ in 0..3 {
        let (request, aux) = large_request("queue-saturation");
        client
            .inner
            .queue_frame(
                &EngineId::from_engine_index(0),
                VllmProtocol::add_frame(),
                encode_msgpack(&request).unwrap(),
                aux,
                None,
                None,
            )
            .unwrap();
    }
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if client
                .inner
                .writing_since
                .lock()
                .is_some_and(|since| since.elapsed() >= Duration::from_millis(50))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("transport-owned payload budget becomes full");
}

/// A finished one-token output for `request_id` on rank 0.
fn finished_output(request_id: &str) -> Vec<Bytes> {
    batch(
        0,
        EngineCoreOutput {
            request_id: request_id.into(),
            new_token_ids: vec![7],
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        },
    )
}

/// Read the engine's input until the abort of `request_id` arrives;
/// everything received, in order, as (kind, request id).
async fn drain_until_aborted(
    engine: &mut MockEngine,
    request_id: &str,
) -> Vec<(&'static str, String)> {
    let mut seen = Vec::new();
    loop {
        match engine.recv().await.unwrap() {
            EngineInbound::Add(request) if request.request_id == "queue-saturation" => {}
            EngineInbound::Add(request) => seen.push(("add", request.request_id)),
            EngineInbound::Abort(ids) => {
                let done = ids.iter().any(|id| id == request_id);
                seen.extend(ids.into_iter().map(|id| ("abort", id)));
                if done {
                    return seen;
                }
            }
            other => panic!("unexpected inbound {other:?}"),
        }
    }
}

/// An admitted large frame on a stalled rank does not block another peer.
#[tokio::test]
async fn an_admitted_large_frame_does_not_block_another_rank() {
    let (client, mut engines, _ns) = connect_ranks(2, false).await;
    let (request, aux) = large_request("stalled");
    let _stalled = client.submit_with_aux(request, aux).await.unwrap();
    let mut healthy = client.submit(request_for("healthy", 1)).await.unwrap();
    let inbound = tokio::time::timeout(TIMEOUT, engines[1].recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(inbound, EngineInbound::Add(request) if request.request_id == "healthy"));
    engines[1]
        .send_output(batch(
            1,
            EngineCoreOutput {
                request_id: "healthy".into(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    assert!(tokio::time::timeout(TIMEOUT, healthy.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .finished());
    assert!(client.is_alive());
    // Rank 0 has not read anything before rank 1 completed.
    assert!(
        matches!(engines[0].recv().await.unwrap(), EngineInbound::Add(request) if request.request_id == "stalled")
    );
}

/// Idle read-side EOF fails even a small request whose send already flushed.
#[tokio::test]
async fn idle_disconnect_fails_a_flushed_small_request() {
    let (client, mut engine, _ns) = connect().await;
    let mut stream = client
        .submit(request_for("idle-disconnect", 0))
        .await
        .unwrap();
    assert!(matches!(
        engine.recv().await.unwrap(),
        EngineInbound::Add(_)
    ));
    drop(engine);
    assert!(tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(!client.is_alive());
}

/// The budget counts clones until the last owner drops, and cancelled
/// reservations neither take credit nor lose the next release notification.
#[tokio::test]
async fn input_budget_tracks_frame_owners_and_cancelled_reservations() {
    let (client, _engine, _ns) = connect().await;
    client.set_input_queue_limit(2, 1024);
    let credit = client.inner.reserve_input(1024).await;
    let frame = credit.wrap(Bytes::from(vec![0; 1024]));
    let cloned = frame.clone();
    drop(credit);
    drop(frame);
    assert_eq!(client.inner.input_budget.used.lock().bytes, 1024);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), client.inner.reserve_input(1))
            .await
            .is_err()
    );
    assert_eq!(client.inner.input_budget.used.lock().frames, 1);
    drop(cloned);
    let credit = tokio::time::timeout(TIMEOUT, client.inner.reserve_input(2048))
        .await
        .unwrap();
    assert_eq!(client.inner.input_budget.used.lock().bytes, 2048);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), client.inner.reserve_input(1))
            .await
            .is_err()
    );
    drop(credit);
    let first = client.inner.reserve_input(1).await;
    let second = client.inner.reserve_input(1).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(20), client.inner.reserve_input(1))
            .await
            .is_err()
    );
    drop((first, second));
    assert_eq!(client.inner.input_budget.used.lock().frames, 0);
    assert_eq!(client.inner.input_budget.used.lock().bytes, 0);
}

/// A non-reading engine cannot accumulate the upstream's 100k-message
/// backlog: ordinary payloads exhaust our byte budget and callers time out.
#[tokio::test]
async fn a_stalled_peer_cannot_bypass_the_transport_byte_budget() {
    let (client, engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    let (request, aux) = large_request("codec-blocker");
    let mut streams = vec![client.submit_with_aux(request, aux).await.unwrap()];
    let mut blocked = false;
    for index in 0..32 {
        let result = client
            .submit_with_aux(
                request_for(&format!("queued-{index}"), 0),
                vec![Bytes::from(vec![0; 2 << 20])],
            )
            .await;
        match result {
            Ok(stream) => streams.push(stream),
            Err(Error::EngineInputBlocked { .. }) => {
                blocked = true;
                break;
            }
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
    assert!(
        blocked,
        "the byte budget must apply before 32 payloads are admitted"
    );
    assert!(client.is_alive());
    let bytes = client.inner.input_budget.used.lock().bytes;
    assert!(bytes > 0 && bytes <= ENGINE_INPUT_QUEUE_BYTES);
    assert!(streams.len() <= 17, "at most 32 MiB plus the codec blocker");
    drop(streams);
    assert_eq!(client.inner.input_budget.used.lock().bytes, bytes);
    drop(engine);
}

/// Cancelling an admitted request cannot release credits for bytes still
/// queued behind a stalled transport; peer close eventually releases them.
#[tokio::test]
async fn cancelled_request_keeps_transport_owned_payload_credit() {
    let (client, engine, _ns) = connect().await;
    let (request, aux) = large_request("first");
    let first = client.submit_with_aux(request, aux).await.unwrap();
    let (request, aux) = large_request("second");
    let second = client.submit_with_aux(request, aux).await.unwrap();
    let budget = client.inner.input_budget.clone();
    assert!(budget.used.lock().bytes >= 64 << 20);
    drop((first, second));
    assert!(budget.used.lock().bytes >= 64 << 20);
    drop(client);
    drop(engine);
    tokio::time::timeout(TIMEOUT, async {
        while budget.used.lock().frames != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("peer close releases upstream payload owners");
}

/// A transport disconnect fails requests already accepted by the local queue.
#[tokio::test]
async fn disconnected_peer_fails_already_accepted_requests() {
    let (client, engine, _ns) = connect().await;
    let (request, aux) = large_request("disconnected");
    let mut stream = client.submit_with_aux(request, aux).await.unwrap();
    drop(engine);
    assert!(tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(!client.is_alive());
}

/// Local admission completes before receipt; output silence still detects stalls.
#[tokio::test]
async fn accepted_large_frame_uses_output_silence_not_input_timeout() {
    let (client, _engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW / 3);
    client.set_input_death_timeout(SEND_WINDOW / 2);
    client.set_silence_death_timeout(SEND_WINDOW * 2);
    let (request, aux) = large_request("queued");
    let mut stream = client.submit_with_aux(request, aux).await.unwrap();
    tokio::time::sleep(SEND_WINDOW).await;
    assert!(
        client.is_alive(),
        "queue admission finished despite the stalled receiver"
    );
    assert!(tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(
        !client.is_alive(),
        "unanswered queued requests still trigger output silence"
    );
}

/// A small request to an engine that is not reading its input (a frozen
/// core) fits the socket buffer and simply waits: the engine serves it
/// when it reads again, and the client never closed.
#[tokio::test]
async fn a_small_request_to_a_frozen_engine_waits_and_is_served_on_resume() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    // The engine reads nothing until told to below.
    let mut stream = client.submit(request_for("req-small", 0)).await.unwrap();
    tokio::time::sleep(SEND_WINDOW).await;
    assert!(
        client.is_alive(),
        "the frame fit the socket buffer: nothing failed"
    );

    // The engine resumes: the frame is there and the request is served.
    let inbound = engine.recv().await.unwrap();
    assert!(
        matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-small"),
        "{inbound:?}"
    );
    engine
        .send_output(finished_output("req-small"))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().unwrap().finished());
    assert!(stream.next().await.is_none());
    assert!(client.is_alive());
}

/// A frame waiting behind a full zeromq peer queue
/// fails its own request once the send window passes, and nothing else:
/// the client stays up, the request already in flight keeps its stream,
/// a request behind the blocked one fails within the window too instead
/// of hanging, and every queued frame still reaches the engine when it
/// reads again, each abandoned add-request followed by its abort.
#[tokio::test]
async fn a_full_peer_queue_times_out_requests_without_closing_the_client() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();

    saturate_input(&client).await;
    let (request, aux) = (request_for("req-large", 0), Vec::new());
    let started = std::time::Instant::now();
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    let waited = started.elapsed();
    assert!(
        matches!(error, Error::EngineInputBlocked { timeout } if timeout == SEND_WINDOW),
        "{error:?}"
    );
    assert!(
        waited >= SEND_WINDOW && waited < TIMEOUT,
        "waited {waited:?}"
    );
    assert!(client.is_alive(), "one request failed, not the client");

    let error = client
        .submit(request_for("req-behind", 0))
        .await
        .err()
        .expect("the input is still blocked");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    assert!(client.is_alive());

    let seen = drain_until_aborted(&mut engine, "req-behind").await;
    let position = |kind: &str, id: &str| {
        seen.iter()
            .position(|(k, i)| *k == kind && i == id)
            .unwrap_or_else(|| panic!("no {kind} of {id} in {seen:?}"))
    };
    assert_eq!(position("add", "req-small"), 0, "{seen:?}");
    assert!(
        position("add", "req-large") < position("abort", "req-large"),
        "{seen:?}"
    );
    assert!(
        position("add", "req-behind") < position("abort", "req-behind"),
        "{seen:?}"
    );

    engine
        .send_output(finished_output("req-small"))
        .await
        .unwrap();
    assert!(in_flight.next().await.unwrap().unwrap().finished());
    assert!(client.is_alive());
}

/// Once the engine reads again, the client (never closed) serves the next
/// request as before.
#[tokio::test]
async fn a_request_after_the_engine_resumes_is_served() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    saturate_input(&client).await;
    let (request, aux) = large_request("req-large");
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );

    // The engine resumes and drains what was queued: the add and its abort.
    let seen = drain_until_aborted(&mut engine, "req-large").await;
    assert_eq!(
        seen.iter().filter(|(kind, _)| *kind == "add").count(),
        1,
        "{seen:?}"
    );

    let mut stream = client.submit(request_for("req-next", 0)).await.unwrap();
    let inbound = engine.recv().await.unwrap();
    assert!(
        matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-next"),
        "{inbound:?}"
    );
    engine
        .send_output(finished_output("req-next"))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().unwrap().finished());
    assert!(client.is_alive());
}

/// While a frame the engine is not reading blocks the input, the aborts
/// queued behind it (a timed-out submit's, a dropped stream's) never park
/// the dispatcher: the output the engine still produces for the requests
/// it holds is routed at once.
#[tokio::test]
async fn output_is_routed_while_aborts_wait_on_a_blocked_input() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();
    saturate_input(&client).await;
    let (request, aux) = large_request("req-large");
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    // The abort of the abandoned request is with the dispatcher by now,
    // queued behind the frame the engine is not reading.
    tokio::time::sleep(SEND_WINDOW / 10).await;
    engine
        .send_output(finished_output("req-small"))
        .await
        .unwrap();
    let output = tokio::time::timeout(SEND_WINDOW / 2, in_flight.next())
        .await
        .expect("output routed while the abort waits on the blocked input")
        .unwrap()
        .unwrap();
    assert!(output.finished());
    assert!(client.is_alive());
}

/// An engine that reads nothing for the input death window is dead, not
/// pausing: the requests it holds fail, the client closes, and a request
/// after that is refused instead of queued for nobody.
#[tokio::test]
async fn an_input_blocked_for_the_death_window_ends_the_client() {
    let (client, _engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    client.set_input_death_timeout(SEND_WINDOW * 3);
    let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();
    saturate_input(&client).await;
    let (request, aux) = large_request("req-large");
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    assert!(client.is_alive(), "one send window is a pause, not a death");
    let error = tokio::time::timeout(TIMEOUT, in_flight.next())
        .await
        .expect("the engine is given up on within the death window")
        .expect("a terminal error item")
        .expect_err("the in-flight request fails with the engine");
    assert!(
        matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead)),
        "{error:?}"
    );
    assert!(!client.is_alive());
    let error = client
        .submit(request_for("req-after", 0))
        .await
        .err()
        .expect("the client is closed");
    assert!(matches!(error, Error::ClientClosed { .. }), "{error:?}");
}

/// Accepted frames remain owned by the upstream writer after client drop.
#[tokio::test]
async fn an_accepted_large_frame_can_arrive_after_client_drop() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    let (request, aux) = large_request("req-large");
    let mut stream = client.submit_with_aux(request, aux).await.unwrap();
    assert!(client.is_alive());
    drop(client);
    assert!(stream.next().await.unwrap().is_err());
    let inbound = tokio::time::timeout(TIMEOUT, engine.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(inbound, EngineInbound::Add(request) if request.request_id == "req-large"));
}

/// A submit its caller stops polling while the frame waits for the engine
/// (the request's handler was cancelled) still aborts the request on the
/// engine, which reads the add later, and gives its registration back:
/// the same id can be submitted again.
#[tokio::test]
async fn an_abandoned_submit_aborts_its_request_and_frees_its_id() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    saturate_input(&client).await;
    let (request, aux) = large_request("req-large");
    let abandoned =
        tokio::time::timeout(SEND_WINDOW / 3, client.submit_with_aux(request, aux)).await;
    assert!(
        abandoned.is_err(),
        "the engine is not reading a frame this large"
    );
    assert!(
        !client
            .inner
            .routing
            .lock()
            .inflight
            .values()
            .any(|ids| ids.contains("req-large")),
        "the abandoned request still holds its in-flight slot"
    );
    // The engine reads again: the queued add, then its abort.
    let seen = tokio::time::timeout(TIMEOUT, drain_until_aborted(&mut engine, "req-large"))
        .await
        .expect("the abort of the abandoned add follows it");
    assert_eq!(
        seen,
        vec![
            ("add", "req-large".to_string()),
            ("abort", "req-large".to_string())
        ]
    );
    // Its registration is gone too: the id is free, and the request is
    // served.
    let mut stream = client.submit(request_for("req-large", 0)).await.unwrap();
    let inbound = engine.recv().await.unwrap();
    assert!(
        matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-large"),
        "{inbound:?}"
    );
    engine
        .send_output(finished_output("req-large"))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().unwrap().finished());
    assert!(client.is_alive());
}

/// Once the engine has left a frame unaccepted for a send window, the
/// add-requests waiting behind it are bounded: the ones within the bound
/// wait their window as before, the next is refused at once without
/// joining the queue, and when the engine reads again the queue drains in
/// order, each abandoned add followed by its abort, the refused request
/// nowhere in it.
#[tokio::test]
async fn a_blocked_input_refuses_add_requests_past_the_frame_bound() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    client.set_input_queue_limit(3, ENGINE_INPUT_QUEUE_BYTES);
    saturate_input(&client).await;
    let (request, aux) = (request_for("req-large", 0), Vec::new());
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    // req-large already waits behind saturation padding. Two more fit; a
    // third, submitted while they wait, is refused on the spot.
    let (behind_1, behind_2, refused) = tokio::join!(
        client.submit(request_for("req-behind-1", 0)),
        client.submit(request_for("req-behind-2", 0)),
        async {
            tokio::time::sleep(SEND_WINDOW / 3).await;
            let started = std::time::Instant::now();
            let result = client.submit(request_for("req-refused", 0)).await;
            (result, started.elapsed())
        },
    );
    for behind in [behind_1, behind_2] {
        let error = behind
            .err()
            .expect("the frame waits behind the blocked one");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
    }
    let (refused, waited) = refused;
    let error = refused.err().expect("the queue is at its bound");
    assert!(
        matches!(error, Error::EngineInputFull { frames: 3, .. }),
        "{error:?}"
    );
    assert!(
        waited < SEND_WINDOW / 2,
        "refused at once, not after a window: {waited:?}"
    );
    assert!(client.is_alive());
    // The engine reads again.
    let seen = drain_until_aborted(&mut engine, "req-behind-2").await;
    let adds: Vec<&str> = seen
        .iter()
        .filter(|(kind, _)| *kind == "add")
        .map(|(_, id)| id.as_str())
        .collect();
    assert_eq!(
        adds,
        ["req-large", "req-behind-1", "req-behind-2"],
        "{seen:?}"
    );
    let position = |kind: &str, id: &str| {
        seen.iter()
            .position(|(k, i)| *k == kind && i == id)
            .unwrap_or_else(|| panic!("no {kind} of {id} in {seen:?}"))
    };
    for id in ["req-large", "req-behind-1", "req-behind-2"] {
        assert!(position("add", id) < position("abort", id), "{seen:?}");
    }
    assert!(
        !seen.iter().any(|(_, id)| id == "req-refused"),
        "a refused request never reaches the engine: {seen:?}"
    );
    // The refused request gave its id back: it can be submitted now.
    let mut stream = client.submit(request_for("req-refused", 0)).await.unwrap();
    let inbound = engine.recv().await.unwrap();
    assert!(
        matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-refused"),
        "{inbound:?}"
    );
    engine
        .send_output(finished_output("req-refused"))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().unwrap().finished());
}

/// The bound in bytes: behind a blocked frame a small add-request within
/// it still waits its window (and reaches the engine on resume), while
/// one that would take the waiting payload past it is refused at once.
#[tokio::test]
async fn a_blocked_input_refuses_add_requests_past_the_byte_bound() {
    let (client, mut engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    client.set_input_queue_limit(usize::MAX, 1 << 20);
    saturate_input(&client).await;
    let (request, aux) = (request_for("req-large", 0), Vec::new());
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the engine is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    let started = std::time::Instant::now();
    let error = client
        .submit(request_for("req-small", 0))
        .await
        .err()
        .expect("the small frame waits behind the blocked one");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    assert!(
        started.elapsed() >= SEND_WINDOW,
        "a frame within the bound waits its window"
    );
    let started = std::time::Instant::now();
    let error = client
        .submit_with_aux(
            request_for("req-big", 0),
            vec![Bytes::from(vec![0u8; 2 << 20])],
        )
        .await
        .err()
        .expect("the waiting payload would pass the bound");
    assert!(matches!(error, Error::EngineInputFull { .. }), "{error:?}");
    assert!(started.elapsed() < SEND_WINDOW / 2, "refused at once");
    assert!(client.is_alive());
    let seen = drain_until_aborted(&mut engine, "req-small").await;
    let adds: Vec<&str> = seen
        .iter()
        .filter(|(kind, _)| *kind == "add")
        .map(|(_, id)| id.as_str())
        .collect();
    assert_eq!(adds, ["req-large", "req-small"], "{seen:?}");
}

/// The silence watchdog counts from the moment work became in flight when
/// no output came since: a request registered just before its tick, whose
/// first output takes longer than the tick's remainder, is served, not
/// failed as the engine's death.
#[tokio::test]
async fn a_request_registered_just_before_the_watchdog_tick_is_served() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_millis(600);
    client.set_silence_death_timeout(window);
    // Idle through most of the first window, then submit.
    tokio::time::sleep(window - Duration::from_millis(100)).await;
    let mut stream = client.submit(request_for("req-late", 0)).await.unwrap();
    engine.recv().await.unwrap();
    // Past the tick, within the window counted from the registration.
    tokio::time::sleep(Duration::from_millis(300)).await;
    engine
        .send_output(finished_output("req-late"))
        .await
        .unwrap();
    let output = stream
        .next()
        .await
        .expect("an item")
        .expect("served, not failed by the watchdog");
    assert!(output.finished());
    assert!(client.is_alive());
}

/// A wake a rank asks for while a frame is blocked ahead in the input
/// queue (a lockstep group, one rank not reading) neither parks the
/// dispatcher nor skips a rank: the output of the rank that reads is
/// routed at once, and when the engine reads again every rank gets the
/// wake, the one whose frame was blocked included.
#[tokio::test]
async fn a_wake_requested_while_a_frame_is_blocked_parks_no_output_and_skips_no_rank() {
    let (client, mut engines, _ns) = connect_ranks(2, true).await;
    client.set_send_timeout(SEND_WINDOW);
    // Rank 1 holds a request and reads its input; rank 0 reads nothing.
    let mut in_flight = client.submit(request_for("req-1", 1)).await.unwrap();
    match engines[1].recv().await.unwrap() {
        EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
        other => panic!("rank 1 expected the Add, got {other:?}"),
    }
    assert_eq!(next_wake(&mut engines[1]).await, 1);
    // A frame rank 0 does not read blocks the input queue.
    saturate_input(&client).await;
    let (request, aux) = large_request("req-large");
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("rank 0 is not reading a frame this large");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    // Rank 1 took a request for a drained wave and asks for the next one.
    let requested = 7;
    engines[1]
        .send_output(wave_control(1, DpControlMessage::StartWave(requested)))
        .await
        .unwrap();
    // The wake waits on the blocked input; the dispatcher does not.
    engines[1]
        .send_output(batch(
            1,
            EngineCoreOutput {
                request_id: "req-1".into(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    let output = tokio::time::timeout(SEND_WINDOW / 2, in_flight.next())
        .await
        .expect("output routed while the wake waits on the blocked input")
        .unwrap()
        .unwrap();
    assert!(output.finished());
    // Rank 0 reads again: its queue drains, the requested wake in it.
    let woken = tokio::time::timeout(TIMEOUT, async {
        loop {
            match engines[0].recv().await.unwrap() {
                EngineInbound::StartDpWave { wave, .. } if wave >= requested => return wave,
                EngineInbound::StartDpWave { .. }
                | EngineInbound::Add(_)
                | EngineInbound::Abort(_) => continue,
                EngineInbound::Utility(call) => panic!("unexpected utility call {call:?}"),
            }
        }
    })
    .await
    .expect("rank 0 is woken for the requested wave once it reads again");
    assert!(woken >= requested);
    // And rank 1, whose frame was behind the blocked one, has its wake too:
    // the broadcast was queued whole, not cut at the window.
    let woken = tokio::time::timeout(TIMEOUT, next_wake(&mut engines[1]))
        .await
        .expect("rank 1 is woken for the requested wave");
    assert!(woken >= requested);
    assert!(client.is_alive());
}

/// A core that reads every request and answers none (its input thread
/// alive, its step hung) is given up on once it has had requests in
/// flight for a whole window in total, even when every request is
/// abandoned before the window: the busy periods add up until an output,
/// or a whole window idle, starts the count over.
#[tokio::test]
async fn a_core_that_reads_every_request_and_answers_none_is_given_up_on() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_millis(600);
    client.set_silence_death_timeout(window);
    // One answered request arms the watchdog with that window.
    let mut warm = client.submit(request_for("req-warm", 0)).await.unwrap();
    engine.recv().await.unwrap();
    engine
        .send_output(finished_output("req-warm"))
        .await
        .unwrap();
    assert!(warm.next().await.unwrap().unwrap().finished());
    // From here the engine reads its input, adds and aborts alike, and
    // answers nothing. Requests abandoned after less than half a window
    // each, with a short gap between them: no busy period lasts a window,
    // together they do.
    let (busy, gap) = (Duration::from_millis(250), Duration::from_millis(70));
    let started = std::time::Instant::now();
    let mut given_up = None;
    for n in 0..12 {
        let request_id = format!("req-{n}");
        let mut stream = match client.submit(request_for(&request_id, 0)).await {
            Ok(stream) => stream,
            Err(error) => {
                given_up = Some(error);
                break;
            }
        };
        match engine.recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, request_id),
            other => panic!("expected the Add, got {other:?}"),
        }
        match tokio::time::timeout(busy, stream.next()).await {
            Err(_abandoned) => drop(stream),
            Ok(Some(Err(error))) => {
                given_up = Some(error);
                break;
            }
            Ok(other) => panic!("the engine answers nothing: {other:?}"),
        }
        // The abandoned request's abort, read like everything else.
        match engine.recv().await.unwrap() {
            EngineInbound::Abort(ids) => assert_eq!(ids, [request_id]),
            other => panic!("expected the Abort, got {other:?}"),
        }
        tokio::time::sleep(gap).await;
    }
    let error = given_up.expect("the engine is given up on within a few abandoned requests");
    assert!(
        matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead))
            || matches!(error, Error::ClientClosed { .. }),
        "{error:?}"
    );
    assert!(!client.is_alive());
    assert!(
        started.elapsed() >= window,
        "given up on no sooner than a window of unanswered work"
    );
}

/// A request larger than the byte bound on its own is refused while the
/// input is blocked, with nothing waiting (the bound is what may be
/// parked), and the error names the request's own size against the bound.
#[tokio::test]
async fn a_request_past_the_byte_bound_on_its_own_is_refused_with_its_size_named() {
    let (client, _engine, _ns) = connect().await;
    client.set_send_timeout(SEND_WINDOW);
    client.set_input_queue_limit(usize::MAX, 1 << 20);
    saturate_input(&client).await;
    tokio::time::sleep(SEND_WINDOW).await;
    let started = std::time::Instant::now();
    let error = client
        .submit_with_aux(
            request_for("req-big", 0),
            vec![Bytes::from(vec![0u8; 2 << 20])],
        )
        .await
        .err()
        .expect("the request alone passes the bound");
    assert!(started.elapsed() < SEND_WINDOW / 2, "refused at once");
    match &error {
        Error::EngineInputFull {
            frames: 0,
            bytes: 0,
            request_bytes,
            limit_frames: usize::MAX,
            limit_bytes,
        } => {
            assert!(*request_bytes >= 2 << 20, "{request_bytes}");
            assert_eq!(*limit_bytes, 1 << 20);
        }
        other => panic!("{other:?}"),
    }
    let text = error.to_string();
    assert!(
        text.contains("this request's") && text.contains("would pass the bound"),
        "{text}"
    );
    assert!(client.is_alive());
}

/// The watchdog's tick landing in a gap between two abandoned requests
/// does not push the give-up back a whole window: the engine is given up
/// on during the request in which its unanswered work reaches the window,
/// not at a later tick that happens to land while one is in flight.
#[tokio::test]
async fn a_tick_in_a_gap_between_requests_does_not_push_the_give_up_back() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_millis(600);
    client.set_silence_death_timeout(window);
    // One answered request arms the watchdog with that window.
    let mut warm = client.submit(request_for("req-warm", 0)).await.unwrap();
    engine.recv().await.unwrap();
    engine
        .send_output(finished_output("req-warm"))
        .await
        .unwrap();
    assert!(warm.next().await.unwrap().unwrap().finished());
    // Requests of 250 ms with 150 ms gaps: the first tick (600 ms) lands
    // in the second request and re-arms for the 150 ms left (750 ms),
    // which is the gap before the third; the work reaches the window
    // 100 ms into the third request.
    let (busy, gap) = (Duration::from_millis(250), Duration::from_millis(150));
    let started = std::time::Instant::now();
    let mut given_up = None;
    let mut submitted = 0;
    for n in 0..12 {
        let request_id = format!("req-{n}");
        let mut stream = match client.submit(request_for(&request_id, 0)).await {
            Ok(stream) => stream,
            Err(error) => {
                given_up = Some(error);
                break;
            }
        };
        submitted += 1;
        match engine.recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, request_id),
            other => panic!("expected the Add, got {other:?}"),
        }
        match tokio::time::timeout(busy, stream.next()).await {
            Err(_abandoned) => drop(stream),
            Ok(Some(Err(error))) => {
                given_up = Some(error);
                break;
            }
            Ok(other) => panic!("the engine answers nothing: {other:?}"),
        }
        match engine.recv().await.unwrap() {
            EngineInbound::Abort(ids) => assert_eq!(ids, [request_id]),
            other => panic!("expected the Abort, got {other:?}"),
        }
        tokio::time::sleep(gap).await;
    }
    let elapsed = started.elapsed();
    let error = given_up.expect("the engine is given up on within a few abandoned requests");
    assert!(
        matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead))
            || matches!(error, Error::ClientClosed { .. }),
        "{error:?}"
    );
    assert!(!client.is_alive());
    assert!(
        elapsed >= window,
        "given up on no sooner than a window of unanswered work"
    );
    assert_eq!(
        submitted, 3,
        "given up on during the request in which the work reached the window \
         (after {elapsed:?}), not at a tick a whole window after the gap"
    );
}

// ---- The other wire that carries typed tensors: the same regression
// cases with a large typed tensor on a client of the tagged-array
// protocol, whose engine resolves auxiliary frames the same way.

use crate::protocol::tokenspeed::request::TokenizedGenerateReqInput as TaggedWireRequest;

/// A client of that protocol over the same mock engine.
async fn connect_tagged() -> (TokenSpeedClient, MockEngine, IpcNamespace) {
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    (
        TokenSpeedClient::new(transport.unwrap()),
        engine.unwrap(),
        ns,
    )
}

/// A request of that wire with a 64 MiB typed feature tensor.
fn large_tagged_request(request_id: &str) -> TaggedWireRequest {
    use crate::{
        codec::tensor::WireTensor,
        protocol::tokenspeed::multimodal::{
            mm_pad_value, TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality,
        },
    };

    let pad = mm_pad_value(TokenSpeedWireModality::Image, 1);
    TaggedWireRequest {
        rid: request_id.to_string(),
        input_ids: vec![1, pad, 2],
        input_ids_unpadded: Some(vec![1, 9, 2]),
        multimodal_inputs: Some(TokenSpeedWireMmInputs {
            mm_items: vec![TokenSpeedWireMmItem {
                modality: TokenSpeedWireModality::Image,
                hash: 1,
                pad_value: pad,
                offsets: vec![(1, 1)],
                feature: Some(WireTensor::from_raw(
                    "uint8",
                    vec![64 << 20],
                    vec![0; 64 << 20],
                )),
                model_specific_data: std::collections::BTreeMap::new(),
                encode_handshake: None,
            }],
            im_token_id: Some(9),
            video_token_id: None,
            ..Default::default()
        }),
        ..TaggedWireRequest::default()
    }
}

/// What the engine read on that wire: an add's request id, or an abort's
/// ids.
async fn recv_tagged(engine: &mut MockEngine) -> (&'static str, Vec<String>) {
    let frames = engine.recv_request().await.unwrap();
    match frames[0].as_ref() {
        b"\x00" => {
            let request: TaggedWireRequest = decode_msgpack(&frames[1]).unwrap();
            ("add", vec![request.rid])
        }
        b"\x01" => ("abort", decode_msgpack(&frames[1]).unwrap()),
        other => panic!("unexpected request type frame {other:?}"),
    }
}

/// A typed tensor of the auxiliary size reaches the engine as its own
/// frame after the payload, which stays small and still decodes.
#[tokio::test]
async fn a_large_typed_tensor_reaches_the_engine_as_its_own_frame_on_the_tagged_wire() {
    let (client, mut engine, _ns) = connect_tagged().await;
    let _stream = client.submit(large_tagged_request("framed")).await.unwrap();
    let frames = tokio::time::timeout(TIMEOUT, engine.recv_request())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        frames.iter().map(Bytes::len).skip(2).collect::<Vec<_>>(),
        vec![64 << 20],
        "type, payload, one tensor frame: {} frames",
        frames.len()
    );
    assert_eq!(frames[0].as_ref(), b"\x00");
    assert!(
        frames[1].len() < 1024,
        "payload of {} bytes",
        frames[1].len()
    );
    let decoded: TaggedWireRequest = decode_msgpack(&frames[1]).unwrap();
    assert_eq!(decoded.rid, "framed");
}

/// Cancelled typed-tensor requests keep their transport credit until the
/// peer closes, as on the first wire.
#[tokio::test]
async fn cancelled_typed_tensor_requests_keep_their_credit_on_the_tagged_wire() {
    let (client, engine, _ns) = connect_tagged().await;
    let first = client.submit(large_tagged_request("first")).await.unwrap();
    let second = client.submit(large_tagged_request("second")).await.unwrap();
    let budget = client.inner.input_budget.clone();
    assert!(budget.used.lock().bytes >= 64 << 20);
    drop((first, second));
    assert!(budget.used.lock().bytes >= 64 << 20);
    drop(client);
    drop(engine);
    tokio::time::timeout(TIMEOUT, async {
        while budget.used.lock().frames != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("peer close releases upstream payload owners");
}

/// A transport disconnect fails typed-tensor requests the local queue
/// already accepted.
#[tokio::test]
async fn a_peer_disconnect_fails_accepted_typed_tensor_requests_on_the_tagged_wire() {
    let (client, engine, _ns) = connect_tagged().await;
    let mut stream = client
        .submit(large_tagged_request("disconnected"))
        .await
        .unwrap();
    drop(engine);
    assert!(tokio::time::timeout(TIMEOUT, stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    assert!(!client.is_alive());
}

/// A typed-tensor submit its caller abandons while the frames wait for
/// the engine still aborts the request there, the abort behind the add.
#[tokio::test]
async fn an_abandoned_typed_tensor_submit_aborts_its_request_on_the_tagged_wire() {
    let (client, mut engine, _ns) = connect_tagged().await;
    client.set_send_timeout(SEND_WINDOW);
    for _ in 0..3 {
        let (payload, aux) = TokenSpeedProtocol::encode_add(
            &large_tagged_request("queue-saturation"),
            &Default::default(),
        )
        .unwrap();
        client
            .inner
            .queue_frame(
                &EngineId::from_engine_index(0),
                TokenSpeedProtocol::add_frame(),
                payload,
                aux,
                None,
                None,
            )
            .unwrap();
    }
    tokio::time::timeout(TIMEOUT, async {
        loop {
            if client
                .inner
                .writing_since
                .lock()
                .is_some_and(|since| since.elapsed() >= Duration::from_millis(50))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("transport-owned payload budget becomes full");
    let abandoned = tokio::time::timeout(
        SEND_WINDOW / 3,
        client.submit(large_tagged_request("req-large")),
    )
    .await;
    assert!(
        abandoned.is_err(),
        "the engine is not reading frames this large"
    );
    let seen = tokio::time::timeout(TIMEOUT, async {
        let mut seen = Vec::new();
        loop {
            let (kind, ids) = recv_tagged(&mut engine).await;
            let done = kind == "abort" && ids.iter().any(|id| id == "req-large");
            seen.extend(
                ids.into_iter()
                    .filter(|id| id != "queue-saturation")
                    .map(|id| (kind, id)),
            );
            if done {
                return seen;
            }
        }
    })
    .await
    .expect("the abort of the abandoned add follows it");
    assert_eq!(
        seen,
        vec![
            ("add", "req-large".to_string()),
            ("abort", "req-large".to_string())
        ]
    );
}

/// A client of `P` over one mock engine, for the scenarios every wire runs.
async fn connect_wire<P: EngineProtocol>() -> (Client<P>, MockEngine, IpcNamespace) {
    let ns = IpcNamespace::new().unwrap();
    let (handshake, input, output) = (
        ns.handshake_endpoint(),
        ns.input_endpoint(),
        ns.output_endpoint(),
    );
    let (transport, engine) = tokio::join!(
        connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    (Client::new(transport.unwrap()), engine.unwrap(), ns)
}

/// An add-request of `prompt_tokens` prompt tokens, pinned to rank 0.
fn long_request(request_id: &str, prompt_tokens: usize) -> EngineCoreRequest {
    EngineCoreRequest {
        prompt_token_ids: Some(vec![1; prompt_tokens]),
        ..request_for(request_id, 0)
    }
}

/// The engine's death as a stream reports it.
fn is_engine_dead(error: &Error) -> bool {
    matches!(error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead))
}

/// On any wire: the prompt tokens pending prefill extend the silence
/// bound at the allowed rate, so an engine that answers a long prompt
/// after more than a window of silence is served, not given up on; with
/// nothing pending, a silent engine is given up on at the window as
/// before. `request(id, prompt_tokens)` builds the wire's add-request,
/// `done(id, prompt_tokens)` its finished output.
async fn a_pending_prompt_extends_the_bound_and_nothing_pending_does_not<P: EngineProtocol>(
    request: impl Fn(&str, usize) -> P::Request,
    done: impl Fn(&str, u32) -> Vec<Bytes>,
) {
    let (client, mut engine, _ns) = connect_wire::<P>().await;
    let window = Duration::from_millis(400);
    client.set_silence_death_timeout(window);
    // 20k prompt tokens at 10k tokens/s: two seconds on top of the window.
    client.set_prefill_tokens_per_sec(10_000);
    let mut stream = client.submit(request("req-long", 20_000)).await.unwrap();
    engine.recv_request().await.unwrap();
    // Three windows of silence: a prefill in progress, not a dead engine.
    tokio::time::sleep(3 * window).await;
    assert!(
        client.is_alive(),
        "the pending prompt's allowance covers the silence"
    );
    engine.send_output(done("req-long", 20_000)).await.unwrap();
    let output = stream
        .next()
        .await
        .expect("an item")
        .expect("served, not failed by the watchdog");
    assert!(output.finished());
    assert!(stream.next().await.is_none());
    // Nothing pending prefill to speak of: a silent engine is given up on
    // at the window.
    let started = std::time::Instant::now();
    let mut stream = client.submit(request("req-short", 3)).await.unwrap();
    engine.recv_request().await.unwrap();
    let error = match stream.next().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => panic!("a silent engine with nothing pending is dead, not serving"),
        None => panic!("the stream ended without the engine's death"),
    };
    assert!(is_engine_dead(&error), "{error:?}");
    let elapsed = started.elapsed();
    assert!(elapsed >= window && elapsed < 3 * window, "{elapsed:?}");
    assert!(!client.is_alive());
}

#[tokio::test]
async fn a_pending_prompt_extends_the_bound_on_the_vllm_wire() {
    a_pending_prompt_extends_the_bound_and_nothing_pending_does_not::<VllmProtocol>(
        long_request,
        |request_id, _| finished_output(request_id),
    )
    .await;
}

#[tokio::test]
async fn a_pending_prompt_extends_the_bound_on_the_tokenspeed_wire() {
    use crate::protocol::tokenspeed::{
        output::BatchTokenIDOutSlim, request::TokenizedGenerateReqInput, sampling::SamplingParams,
    };
    a_pending_prompt_extends_the_bound_and_nothing_pending_does_not::<TokenSpeedProtocol>(
        |rid, prompt_tokens| TokenizedGenerateReqInput {
            rid: rid.to_string(),
            input_ids: vec![1; prompt_tokens],
            sampling_params: SamplingParams {
                max_new_tokens: Some(4),
                ..SamplingParams::default()
            },
            stream: true,
            ..TokenizedGenerateReqInput::default()
        },
        |rid, prompt_tokens| {
            let done = BatchTokenIDOutSlim {
                rids: vec![rid.to_string()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["stop".into()],
                prompt_tokens: vec![prompt_tokens],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![]],
                output_token_logprobs_idx: vec![vec![]],
                ..Default::default()
            };
            vec![Bytes::from(encode_msgpack(&done).unwrap())]
        },
    )
    .await;
}

#[tokio::test]
async fn a_pending_prompt_extends_the_bound_on_the_sglang_wire() {
    use crate::protocol::sglang::{
        output::BatchTokenIDSlimOutput,
        request::{SglangRequest, TokenizedGenerateReqInput},
        sampling::SamplingParams,
    };
    a_pending_prompt_extends_the_bound_and_nothing_pending_does_not::<SglangProtocol>(
        |rid, prompt_tokens| {
            SglangRequest::Generate(TokenizedGenerateReqInput {
                rid: rid.to_string(),
                input_ids: vec![1; prompt_tokens],
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    ..SamplingParams::default()
                },
                stream: true,
                ..TokenizedGenerateReqInput::default()
            })
        },
        |rid, prompt_tokens| {
            let done = BatchTokenIDSlimOutput {
                rids: vec![rid.to_string()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["stop".into()],
                finished_messages: vec![None],
                finished_matched: vec![None],
                prompt_tokens: vec![prompt_tokens],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![]],
                output_token_logprobs_idx: vec![vec![]],
                finished_status: vec![None],
                ..Default::default()
            };
            vec![Bytes::from(encode_msgpack(&done).unwrap())]
        },
    )
    .await;
}

/// With no prefill allowance (zero tokens per second) the bound is the
/// window alone, whatever prompt is pending: the verdict as it was before
/// the allowance.
#[tokio::test]
async fn with_no_prefill_allowance_a_long_prompt_is_given_up_on_at_the_window() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_millis(400);
    client.set_silence_death_timeout(window);
    client.set_prefill_tokens_per_sec(0);
    let started = std::time::Instant::now();
    let mut stream = client
        .submit(long_request("req-long", 20_000))
        .await
        .unwrap();
    engine.recv().await.unwrap();
    let error = stream
        .next()
        .await
        .expect("an item")
        .expect_err("nothing allowed for the prompt: dead at the window");
    assert!(is_engine_dead(&error), "{error:?}");
    let elapsed = started.elapsed();
    assert!(elapsed >= window && elapsed < 3 * window, "{elapsed:?}");
    assert!(!client.is_alive());
}

/// A long prompt withdrawn mid-prefill (its client gave up on it) leaves
/// its allowance in hand until it passes: the engine may be mid-way
/// through it, so the next request is not given up on at its dispatch
/// for the silence the prompt explains.
#[tokio::test]
async fn a_long_prompt_withdrawn_mid_prefill_leaves_no_silence_to_answer_for() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_millis(400);
    client.set_silence_death_timeout(window);
    client.set_prefill_tokens_per_sec(10_000);
    let stream = client
        .submit(long_request("req-long", 20_000))
        .await
        .unwrap();
    engine.recv().await.unwrap();
    // Two windows of silence the prompt explains; then the client gives up.
    tokio::time::sleep(2 * window).await;
    drop(stream);
    match engine.recv().await.unwrap() {
        EngineInbound::Abort(ids) => assert_eq!(ids, ["req-long"]),
        other => panic!("expected the abort, got {other:?}"),
    }
    let mut stream = client.submit(request_for("req-next", 0)).await.unwrap();
    engine.recv().await.unwrap();
    tokio::time::sleep(window / 2).await;
    assert!(
        client.is_alive(),
        "given up on for the silence the withdrawn prompt's prefill explained"
    );
    engine
        .send_output(finished_output("req-next"))
        .await
        .unwrap();
    assert!(stream
        .next()
        .await
        .expect("an item")
        .expect("served")
        .finished());
}

/// Prompts withdrawn before their allowance passes keep explaining the
/// silence until it does (the engine may be mid-way through them), and
/// explain nothing before them: the unanswered work of a short request
/// the engine read and never answered stands, so once the prompts'
/// allowance has passed the next request is given up on for what that
/// evidence left of the window, not a whole window later.
#[tokio::test]
async fn prompts_withdrawn_before_their_allowance_passes_explain_the_silence_until_it_does() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_secs(1);
    client.set_silence_death_timeout(window);
    client.set_prefill_tokens_per_sec(10_000);
    // Evidence first: a short request the engine reads and never answers,
    // abandoned after 700 ms.
    let stream = client.submit(request_for("req-short", 0)).await.unwrap();
    engine.recv().await.unwrap();
    tokio::time::sleep(Duration::from_millis(700)).await;
    drop(stream);
    match engine.recv().await.unwrap() {
        EngineInbound::Abort(ids) => assert_eq!(ids, ["req-short"]),
        other => panic!("expected the abort, got {other:?}"),
    }
    // Two 10k-token prompts, a second of allowance each and two seconds
    // in hand together, withdrawn together after 300 ms.
    let handed = std::time::Instant::now();
    let allowance = Duration::from_secs(2);
    let first = client
        .submit(long_request("req-long-1", 10_000))
        .await
        .unwrap();
    engine.recv().await.unwrap();
    let second = client
        .submit(long_request("req-long-2", 10_000))
        .await
        .unwrap();
    engine.recv().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    drop(first);
    drop(second);
    for _ in 0..2 {
        match engine.recv().await.unwrap() {
            EngineInbound::Abort(ids) => assert!(ids[0].starts_with("req-long-"), "{ids:?}"),
            other => panic!("expected an abort, got {other:?}"),
        }
    }
    // The next request is not given up on at the window: the prompts'
    // allowance still explains the silence. Once it has passed, the
    // 700 ms of evidence stand, and the verdict comes 300 ms later, not
    // a whole window later.
    let mut stream = client.submit(request_for("req-next", 0)).await.unwrap();
    engine.recv().await.unwrap();
    let error = stream
        .next()
        .await
        .expect("an item")
        .expect_err("a silent engine once the allowance in hand has passed");
    assert!(is_engine_dead(&error), "{error:?}");
    let elapsed = handed.elapsed();
    assert!(
        elapsed >= allowance,
        "not before the withdrawn prompts' allowance passed: {elapsed:?}"
    );
    assert!(
        elapsed < allowance + window,
        "the unanswered work before the prompts stood: {elapsed:?}"
    );
    assert!(!client.is_alive());
}

/// A prompt queues its allowance when the engine accepts its frame, not
/// at registration: an add that fails before that (no engine for its
/// rank) or is refused at a blocked input's bound never reached the
/// engine, so it explains no silence, and the request the engine did
/// read is answered for at the window, not the refused prompts'
/// allowance later.
#[tokio::test]
async fn an_add_the_engine_never_received_queues_no_allowance_on_the_wire() {
    let (client, mut engine, _ns) = connect().await;
    let window = Duration::from_secs(2);
    client.set_silence_death_timeout(window);
    client.set_prefill_tokens_per_sec(10_000);
    client.set_send_timeout(SEND_WINDOW);
    client.set_input_queue_limit(0, usize::MAX);
    // The engine reads one short request and answers nothing.
    let started = std::time::Instant::now();
    let mut stream = client.submit(request_for("req-short", 0)).await.unwrap();
    engine.recv().await.unwrap();
    // A 50k-token prompt (five seconds of allowance) for a rank that does
    // not exist fails before any frame is built.
    let error = client
        .submit(EngineCoreRequest {
            prompt_token_ids: Some(vec![1; 50_000]),
            ..request_for("req-nowhere", 7)
        })
        .await
        .err()
        .expect("no engine for rank 7");
    assert!(
        matches!(error, Error::InvalidDataParallelRank { rank: 7, .. }),
        "{error:?}"
    );
    // The engine stops reading: once the transport's payload budget is
    // full, the next frame waits behind it for a send window and fails,
    // and the 50k-token prompt after it is refused at the bound on the
    // spot.
    saturate_input(&client).await;
    let (request, aux) = (request_for("req-blocked", 0), Vec::new());
    let error = client
        .submit_with_aux(request, aux)
        .await
        .err()
        .expect("the transport's payload budget is full");
    assert!(
        matches!(error, Error::EngineInputBlocked { .. }),
        "{error:?}"
    );
    tokio::time::sleep(SEND_WINDOW / 10).await;
    let refused_at = std::time::Instant::now();
    let error = client
        .submit(long_request("req-refused", 50_000))
        .await
        .err()
        .expect("the input is blocked and its queue at the bound");
    assert!(matches!(error, Error::EngineInputFull { .. }), "{error:?}");
    assert!(refused_at.elapsed() < SEND_WINDOW / 2, "refused at once");
    assert_eq!(
        client.prefill_pending_tokens(),
        3,
        "only the prompt the engine read is in hand"
    );
    // The verdict on the request the engine read comes at the window from
    // its dispatch, not ten seconds of never-handed allowance later.
    let error = match stream.next().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => panic!("the engine answers nothing"),
        None => panic!("the stream ended without the engine's death"),
    };
    assert!(is_engine_dead(&error), "{error:?}");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= window && elapsed < window + Duration::from_secs(2),
        "at the window, nothing queued for the prompts the engine never got: {elapsed:?}"
    );
    assert!(!client.is_alive());
}

/// The registry's side of it: a registered prompt is neither pending nor
/// in the bound until it is handed over; one retired before that leaves
/// both where they were; a hand-over of a prompt already handed or gone
/// queues nothing.
#[test]
fn an_add_the_engine_never_received_queues_no_allowance() {
    let mut registry = RequestRegistry::<<VllmProtocol as EngineProtocol>::Output>::default();
    let before = Instant::now();
    let _first = registry.register("req-1".to_string(), 1_000_000, true, false, false);
    let _second = registry.register("req-2".to_string(), 1_000_000, true, false, false);
    assert_eq!(registry.prefill_pending_tokens(), 0, "nothing handed yet");
    assert_eq!(
        registry.work_end.saturating_duration_since(before),
        Duration::ZERO
    );
    // req-2's add fails before the hand-over: nothing to take out.
    registry.retire("req-2");
    assert_eq!(registry.prefill_pending_tokens(), 0);
    assert_eq!(
        registry.work_end.saturating_duration_since(before),
        Duration::ZERO
    );
    registry.handed("req-1");
    assert_eq!(registry.prefill_pending_tokens(), 1_000_000);
    let in_hand = registry.work_end.saturating_duration_since(before);
    assert!(
        in_hand >= Duration::from_secs(500) && in_hand < Duration::from_secs(501),
        "a million tokens of prefill in hand: {in_hand:?}"
    );
    // Handed before, or gone: nothing more.
    registry.handed("req-1");
    registry.handed("req-2");
    assert_eq!(registry.prefill_pending_tokens(), 1_000_000);
    assert_eq!(registry.work_end.saturating_duration_since(before), in_hand);
    // Withdrawn before its first output: pending no more, the allowance
    // stays in hand.
    registry.retire("req-1");
    assert_eq!(registry.prefill_pending_tokens(), 0);
    assert_eq!(registry.work_end.saturating_duration_since(before), in_hand);
}

/// The default allowance: 500 s per million prompt tokens, none at a
/// rate of zero; prompts handed over queue their allowance behind the
/// work in hand.
#[test]
fn the_default_prefill_allowance_is_500_s_per_million_tokens() {
    let mut registry = RequestRegistry::<<VllmProtocol as EngineProtocol>::Output>::default();
    assert_eq!(
        registry.prefill_allowance(1_000_000),
        Duration::from_secs(500)
    );
    assert_eq!(registry.prefill_allowance(0), Duration::ZERO);
    let before = Instant::now();
    let _first = registry.register("req-1".to_string(), 1_000_000, true, false, false);
    let _second = registry.register("req-2".to_string(), 1_000_000, true, false, false);
    registry.handed("req-1");
    registry.handed("req-2");
    let in_hand = registry.work_end.saturating_duration_since(before);
    assert!(
        in_hand >= Duration::from_secs(1_000) && in_hand < Duration::from_secs(1_001),
        "two million tokens of prefill in hand: {in_hand:?}"
    );
    registry.prefill_tokens_per_sec = 0;
    assert_eq!(registry.prefill_allowance(1_000_000), Duration::ZERO);
}

/// The connection's layout is the one every rank declared; a rank that
/// disagrees anywhere in the group, the middle included, fails the agreement
/// and the connection falls back to the default layout; no engines read as
/// the default.
#[test]
fn a_disagreeing_rank_in_the_middle_of_the_group_fails_the_agreement() {
    use crate::protocol::sglang::{SglangLayout, SglangProtocol};

    let engine = |index: u32, tensor_aux_frames: bool| ConnectedEngine {
        engine_id: EngineId::from_engine_index(index),
        ready_response: EngineCoreReadyResponse {
            tensor_aux_frames,
            ..default_ready_response()
        },
    };
    let agreeing = [engine(0, true), engine(1, true), engine(2, true)];
    assert_eq!(
        connection_layout::<SglangProtocol>(&agreeing),
        SglangLayout {
            tensor_aux_frames: true
        }
    );
    let middle_rank_behind = [engine(0, true), engine(1, false), engine(2, true)];
    assert_eq!(
        connection_layout::<SglangProtocol>(&middle_rank_behind),
        SglangLayout::default()
    );
    assert_eq!(
        connection_layout::<SglangProtocol>(&[]),
        SglangLayout::default()
    );
}

/// A decode side waiting for its KV transfer (`do_remote_prefill` in its
/// KV-transfer params) is not in flight on the engine until its first
/// output: the registry counts no silence over it, starts no busy period on
/// it and books no prefill for its prompt; its first output admits it, and
/// its retirement then ends the busy period as any request's does.
#[test]
fn a_transfer_pending_side_is_not_in_flight_until_its_first_output() {
    let mut registry = RequestRegistry::<<VllmProtocol as EngineProtocol>::Output>::default();
    let _pending = registry.register("decode-1".to_string(), 3_000, true, true, true);
    assert_eq!(
        registry.reported_every_step(),
        0,
        "no silence counted over it"
    );
    assert!(registry.in_flight_since.is_none(), "no busy period on it");
    registry.handed("decode-1");
    assert_eq!(
        registry.prefill_pending_tokens(),
        0,
        "its prompt is the prefill side's work, not this engine's"
    );
    // A plain request beside it starts the busy period and counts.
    let _plain = registry.register("plain-1".to_string(), 10, true, false, false);
    assert_eq!(registry.reported_every_step(), 1);
    assert!(registry.in_flight_since.is_some());
    registry.handed("plain-1");
    assert_eq!(registry.prefill_pending_tokens(), 10);
    // The plain request ends: the busy period ends with it, the pending side
    // notwithstanding.
    registry.retire("plain-1");
    assert_eq!(registry.reported_every_step(), 0);
    assert!(
        registry.in_flight_since.is_none(),
        "a transfer-pending side left behind keeps no busy period"
    );
    // The side's first output admits it: from here it is in flight.
    let entry = registry.requests.get_mut("decode-1").expect("registered");
    entry.started = true;
    entry.transfer_pending = false;
    entry.counted = entry.every_step;
    registry.admitted(true, Instant::now());
    assert_eq!(
        registry.reported_every_step(),
        1,
        "counted from its first output"
    );
    assert!(
        registry.in_flight_since.is_some(),
        "the busy period starts on it"
    );
    assert_eq!(
        registry.prefill_pending_tokens(),
        0,
        "still no prefill of its own"
    );
    registry.retire("decode-1");
    assert_eq!(registry.reported_every_step(), 0);
    assert!(
        registry.in_flight_since.is_none(),
        "the last active one ended the busy period"
    );
    // A pulled side registered active (a wire that reports every step keeps
    // the clock while it waits): counted and busy from its dispatch, its
    // prompt still never booked as prefill here.
    let _active = registry.register("decode-2".to_string(), 3_000, true, false, true);
    assert_eq!(
        registry.reported_every_step(),
        1,
        "counted from its dispatch"
    );
    assert!(registry.in_flight_since.is_some(), "the busy period is on");
    registry.handed("decode-2");
    assert_eq!(
        registry.prefill_pending_tokens(),
        0,
        "a pulled prompt is never this engine's prefill"
    );
    registry.retire("decode-2");
    assert_eq!(registry.reported_every_step(), 0);
    assert!(registry.in_flight_since.is_none());
}

/// The same through the client: a vLLM decode side's silence clock starts at
/// its first output, not at its dispatch, and its prompt never counts as
/// pending prefill; a plain request is counted from its dispatch as before.
#[tokio::test]
async fn a_vllm_decode_side_waiting_for_its_transfer_counts_no_silence() {
    use crate::protocol::vllm::sampling::EngineCoreSamplingParams;
    let (client, mut engine, _ns) = connect().await;
    let decode_side = EngineCoreRequest {
        request_id: "decode-1".to_string(),
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            extra_args: Some(HashMap::from([(
                "kv_transfer_params".to_string(),
                serde_json::json!({"do_remote_prefill": true, "remote_block_ids": [7]}),
            )])),
            ..EngineCoreSamplingParams::default()
        }),
        ..EngineCoreRequest::default()
    };
    let mut side = client.submit(decode_side).await.unwrap();
    engine.recv_request().await.unwrap();
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert_eq!(
        client.output_silence(),
        None,
        "a side waiting for its KV transfer is not in flight on the engine"
    );
    assert_eq!(
        client.prefill_pending_tokens(),
        0,
        "its prompt is pulled as KV, not prefilled here"
    );
    // Its first token: the transfer is done, the side is in flight.
    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "decode-1".to_string(),
                new_token_ids: vec![11],
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    let first = side.next().await.expect("the first token").unwrap();
    assert_eq!(first.request_id, "decode-1");
    let after_first = client
        .output_silence()
        .expect("in flight from its first output");
    assert!(after_first < Duration::from_millis(500), "{after_first:?}");
    tokio::time::sleep(Duration::from_millis(80)).await;
    let grown = client.output_silence().expect("still in flight");
    assert!(
        grown >= Duration::from_millis(70),
        "counts from here on: {grown:?}"
    );
    // Its end ends the busy period.
    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "decode-1".to_string(),
                new_token_ids: vec![12],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    while side.next().await.is_some() {}
    assert_eq!(client.output_silence(), None, "nothing in flight");
}

/// On a wire that reports every step the same side counts from its dispatch:
/// the stats-only steps keep the clock while a healthy side waits, so a
/// frozen engine whose only work is pulled sides is still seen (its silence
/// grows, the busy period is on); its prompt is still no prefill of this
/// engine's.
#[tokio::test]
async fn a_vllm_decode_side_waiting_for_its_transfer_counts_silence_on_a_step_reporting_wire() {
    use crate::protocol::vllm::sampling::EngineCoreSamplingParams;
    let (client, mut engine, _ns) = connect().await;
    client.note_reports_steps();
    let decode_side = EngineCoreRequest {
        request_id: "decode-1".to_string(),
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            extra_args: Some(HashMap::from([(
                "kv_transfer_params".to_string(),
                serde_json::json!({"do_remote_prefill": true, "remote_block_ids": [7]}),
            )])),
            ..EngineCoreSamplingParams::default()
        }),
        ..EngineCoreRequest::default()
    };
    let mut side = client.submit(decode_side).await.unwrap();
    engine.recv_request().await.unwrap();
    tokio::time::sleep(Duration::from_millis(80)).await;
    let silence = client
        .output_silence()
        .expect("in flight from its dispatch on a step-reporting wire");
    assert!(
        silence >= Duration::from_millis(70),
        "counts from its dispatch: {silence:?}"
    );
    assert_eq!(
        client.prefill_pending_tokens(),
        0,
        "its prompt is pulled as KV, not prefilled here"
    );
    engine
        .send_output(batch(
            0,
            EngineCoreOutput {
                request_id: "decode-1".to_string(),
                new_token_ids: vec![11],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
    while side.next().await.is_some() {}
    assert_eq!(client.output_silence(), None, "nothing in flight");
}
