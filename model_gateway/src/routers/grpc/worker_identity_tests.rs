//! The gateway's cases over a worker and the gateway's own engine clients: a
//! ZMQ worker's connect, its probe and its admin calls through the ZMQ client,
//! the manager's flush over such workers, and a dispatch's booking on the
//! state a worker's door answers.

mod dispatch_on_the_door;

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use engine_zmq_client::protocol::handshake::EngineCoreReadyResponse;
use openai_protocol::worker::{ConnectionMode, ProfileOptions, RuntimeType, WorkerType};
use smg_worker::{
    identity::{
        basic::{BasicWorker, Worker},
        error::WorkerError,
    },
    test_support::no_health_check,
};
use tokio::time;

use crate::routers::grpc::backend_client::{
    backend_client_if_initialized, BackendClient, BasicWorkerBuilder,
};

/// What the engine declared at its handshake is read through the backend
/// peek, never through a method of its own: a direct-ZMQ worker whose
/// engine declared a bootstrap address answers it once the connect landed,
/// and nothing before (the peek never drives the connect).
#[tokio::test]
async fn the_declared_bootstrap_is_read_through_the_backend_peek() {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::handshake::EngineCoreReadyResponse,
        EngineId,
    };

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    let base = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", base.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let worker = BasicWorkerBuilder::new(ep("ts0.ipc"))
        .connection_mode(ConnectionMode::Zmq)
        .worker_type(WorkerType::Prefill)
        .health_config(no_health_check())
        .build();
    assert!(worker.engine_connection_if_initialized().is_none());

    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::TokenSpeed,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            EngineCoreReadyResponse {
                bootstrap_host: Some("10.0.0.7".to_string()),
                bootstrap_port: Some(8998),
                ..default_ready_response()
            }
        ),
    );
    let client = client.expect("adapter connect");
    let _engine = engine.expect("mock engine");
    worker
        .engine_connection
        .load_full()
        .set(Arc::new(BackendClient::Zmq(client)))
        .ok()
        .expect("cell empty");

    let backend = backend_client_if_initialized(&worker)
        .expect("the gateway's client")
        .expect("the connect landed");
    assert_eq!(backend.declared_bootstrap(), Some(("10.0.0.7", 8998)));
}

/// A ZMQ client whose engine dies must be evicted by the health probe (the
/// connection can't reconnect in place — liveness is latched), and the
/// handshake guard reset so a later probe rebinds the sockets for a
/// replacement engine. A live client stays cached across probes.
#[tokio::test]
async fn zmq_health_check_evicts_a_dead_client_and_resets_the_guard() {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        EngineId, ENGINE_CORE_DEAD_SENTINEL,
    };

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    let base = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", base.path().join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::Vllm,
            Duration::from_secs(10)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let client = client.expect("adapter connect");
    let mut engine = engine.expect("mock engine");

    let worker = BasicWorkerBuilder::new(ep("ts0.ipc"))
        .connection_mode(ConnectionMode::Zmq)
        .health_config(no_health_check())
        .build();

    // Inject a completed connect and mark the one-shot guard as set.
    let cell = worker.engine_connection.load_full();
    cell.set(Arc::new(BackendClient::Zmq(client)))
        .ok()
        .expect("cell empty");
    worker.zmq_connect_started.store(true, Ordering::SeqCst);

    // A live client stays cached and reports healthy.
    assert!(worker.zmq_health_check());
    assert!(worker.engine_connection.load_full().get().is_some());

    let replacement = BasicWorkerBuilder::new(ep("ts0.ipc"))
        .connection_mode(ConnectionMode::Zmq)
        .health_config(no_health_check())
        .build();
    assert!(replacement.inherit_shared_state_from(&worker));
    worker.abort_background_tasks();
    assert!(replacement.zmq_health_check());

    // Kill the engine and wait for the dispatcher to latch the client closed.
    engine
        .send_output(vec![bytes::Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
        .await
        .unwrap();
    let observed_dead = time::timeout(Duration::from_secs(5), async {
        loop {
            let alive = worker
                .engine_connection
                .load_full()
                .get()
                .map(|c| c.is_alive())
                .unwrap_or(false);
            if !alive {
                break;
            }
            time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        observed_dead.is_ok(),
        "client never observed ENGINE_CORE_DEAD"
    );

    // The next probe evicts the dead client and resets the guard so a later
    // probe re-runs the handshake for a replacement engine.
    assert!(!worker.zmq_health_check());
    assert!(
        worker.engine_connection.load_full().get().is_none(),
        "dead client must be evicted from the cell"
    );
    assert!(
        !worker.zmq_connect_started.load(Ordering::SeqCst),
        "handshake guard must reset to allow a reconnect"
    );
    assert!(
        replacement.engine_connection_if_initialized().is_none(),
        "eviction by an outstanding old probe must reach the replacement"
    );
}

/// A probe launched on an earlier registry handle can finish after a new
/// worker's handshake. The registered worker must still start its next one.
#[tokio::test]
async fn a_replacement_reconnects_after_a_retired_workers_probe() {
    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        EngineId, ENGINE_CORE_DEAD_SENTINEL,
    };

    use crate::routers::common::snapshot::WorkerRegistry;

    let dir = tempfile::tempdir().unwrap();
    let base = format!("ipc://{}", dir.path().join("engine").display());
    let port = portpicker::pick_unused_port().expect("an available handshake port");
    let handshake = format!("tcp://127.0.0.1:{port}");
    let build = || {
        Arc::new(
            BasicWorkerBuilder::new(&base)
                .connection_mode(ConnectionMode::Zmq)
                .runtime_type(RuntimeType::Vllm)
                .zmq_handshake_address(&handshake)
                .health_config(no_health_check())
                .build(),
        )
    };
    let registry = WorkerRegistry::new();
    let old = build();
    let id = registry.register(old.clone()).unwrap();
    let new = build();
    assert!(registry.replace(&id, new.clone()));

    assert!(!new.zmq_health_check());
    let mut first = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .unwrap();
    assert!(wait_for(|| new.zmq_health_check()).await);
    let first_cell = old.engine_connection.load_full();
    first
        .send_output(vec![bytes::Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
        .await
        .unwrap();
    assert!(wait_for(|| !first_cell.get().unwrap().is_alive()).await);
    assert!(!old.zmq_health_check());
    assert!(
        new.engine_connection.load_full().get().is_none(),
        "eviction reached the registered worker"
    );
    drop(first);
    assert!(!new.zmq_health_check());
    // A queued probe and a request can still hold the retired worker.
    // Only the registered worker may own a driver for this empty cell.
    assert!(!old.zmq_health_check());
    assert!(old.engine_connection().await.is_err());
    assert!(old.zmq_connect_abort.load().is_none());
    let _second = connect_to_frontend(
        &handshake,
        EngineId::from_engine_index(0),
        default_ready_response(),
    )
    .await
    .expect("the registered worker must rebind after its peer evicts");
    assert!(wait_for(|| new.zmq_health_check()).await);
    // A second probe read the dead cell before eviction and resumes now.
    old.engine_connection.evict(&first_cell);
    let current = new.engine_connection.load_full();
    assert!(!Arc::ptr_eq(&first_cell, &current));
    assert!(Arc::ptr_eq(&current, &old.engine_connection.load_full()));
}

/// A ZMQ worker's health check is its probe's answer, as the checker reads
/// it: no client yet fails the check, a live one passes it. The probe's own
/// cases call the probe.
#[tokio::test]
async fn a_zmq_workers_health_check_is_its_probes_answer() {
    use engine_zmq_client::mock_engine::default_ready_response;

    let dir = tempfile::tempdir().unwrap();
    let (connected, _engine) =
        zmq_worker_on_a_mock_rank(dir.path(), default_ready_response()).await;
    // The same address with its check on, and its connect marked as started
    // so the probe starts no driver of its own.
    let worker = BasicWorkerBuilder::new(connected.url())
        .connection_mode(ConnectionMode::Zmq)
        .build();
    worker.zmq_connect_started.store(true, Ordering::SeqCst);
    assert!(
        worker.check_health_async().await.is_err(),
        "no client yet: the probe says no"
    );

    let client = connected
        .engine_connection_if_initialized()
        .expect("the connect landed");
    worker
        .engine_connection
        .load_full()
        .set(client)
        .ok()
        .expect("cell empty");
    assert!(
        worker.check_health_async().await.is_ok(),
        "a live client: the probe says yes"
    );
}

/// A direct-ZMQ worker whose connect landed on one mock rank that
/// declared `ready`, the backend injected as the connect driver leaves
/// it, plus the rank itself for the test to answer what the worker sends.
async fn zmq_worker_on_a_mock_rank(
    dir: &std::path::Path,
    ready: EngineCoreReadyResponse,
) -> (BasicWorker, engine_zmq_client::mock_engine::MockEngine) {
    use engine_zmq_client::{mock_engine::connect_to_frontend, EngineId};

    use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

    std::fs::create_dir_all(dir).unwrap();
    let ep = |name: &str| format!("ipc://{}", dir.join(name).display());
    let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
    let (client, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            EosTokenIds::default(),
            RuntimeType::TokenSpeed,
            Duration::from_secs(10)
        ),
        connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
    );
    let client = client.expect("adapter connect");
    let engine = engine.expect("mock engine");
    let worker = BasicWorkerBuilder::new(ep("ts0.ipc"))
        .connection_mode(ConnectionMode::Zmq)
        .runtime_type(RuntimeType::TokenSpeed)
        .health_config(no_health_check())
        .build();
    worker
        .engine_connection
        .load_full()
        .set(Arc::new(BackendClient::Zmq(client)))
        .ok()
        .expect("cell empty");
    worker.zmq_connect_started.store(true, Ordering::SeqCst);
    (worker, engine)
}

/// The handshake of a rank whose wire carries control calls.
fn control_ready() -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        positional_layouts: std::collections::BTreeMap::from([(
            "ControlReplySlim".to_string(),
            ["call_id", "success", "message", "engine_index"]
                .iter()
                .map(ToString::to_string)
                .collect(),
        )]),
        ..engine_zmq_client::mock_engine::default_ready_response()
    }
}

/// Answer the one control call a rank receives with `success` and
/// `message`; the call's method and arguments, for the test to check.
async fn answer_control_call(
    engine: &mut engine_zmq_client::mock_engine::MockEngine,
    success: bool,
    message: Option<&str>,
) -> (String, Vec<engine_zmq_client::codec::OpaqueValue>) {
    use engine_zmq_client::{
        codec::{decode_msgpack, encode_msgpack, OpaqueValue},
        protocol::{control::ControlReplySlim, tokenspeed::request::TokenSpeedRequestType},
    };

    let frames = engine.recv_request().await.unwrap();
    assert_eq!(
        TokenSpeedRequestType::from_frame(frames[0].as_ref()),
        Some(TokenSpeedRequestType::Control)
    );
    let (call_id, method, args): (i64, String, Vec<OpaqueValue>) =
        decode_msgpack(frames[1].as_ref()).unwrap();
    let reply = ControlReplySlim {
        call_id,
        success,
        message: message.map(str::to_string),
        engine_index: 0,
    };
    engine
        .send_output(vec![bytes::Bytes::from(encode_msgpack(&reply).unwrap())])
        .await
        .unwrap();
    (method, args)
}

/// A direct-ZMQ worker whose engine declared no control reply has no
/// admin call to take: its cache flush and both profiler operations are
/// refused by name before anything is sent.
#[tokio::test]
async fn a_zmq_workers_admin_operations_are_refused_without_the_control_reply() {
    use engine_zmq_client::mock_engine::default_ready_response;

    let dir = tempfile::tempdir().unwrap();
    let (worker, _engine) = zmq_worker_on_a_mock_rank(dir.path(), default_ready_response()).await;
    let outcomes = [
        ("flush_cache", worker.flush_cache().await),
        (
            "start_profile",
            worker.start_profile(&ProfileOptions::default()).await,
        ),
        ("stop_profile", worker.stop_profile().await),
    ];
    for (operation, outcome) in outcomes {
        let error = outcome.expect_err(operation);
        assert!(
            error.to_string().contains(&format!(
                "{operation} is not supported over this engine's ZMQ wire"
            )),
            "{operation}: {error}"
        );
    }
}

/// A direct-ZMQ worker's cache flush and profiler ride its engine's
/// control wire when the handshake declared the control reply: each
/// operation is one control call named after it (the flush with its
/// timeout, the start with its options and a stamped profile id, the stop
/// with nothing), and the rank's reply is the operation's outcome.
#[tokio::test]
async fn a_zmq_workers_admin_operations_ride_its_control_wire() {
    use engine_zmq_client::codec::OpaqueValue;

    let dir = tempfile::tempdir().unwrap();
    let (worker, mut engine) = zmq_worker_on_a_mock_rank(dir.path(), control_ready()).await;

    let (flushed, (method, args)) = tokio::join!(
        worker.flush_cache(),
        answer_control_call(&mut engine, true, None)
    );
    flushed.expect("the flush succeeded");
    assert_eq!(method, "flush_cache");
    assert_eq!(args, vec![OpaqueValue::F64(0.0)]);

    let options = ProfileOptions {
        output_dir: Some("/traces".to_string()),
        num_steps: Some(8),
        profile_by_stage: true,
        ..ProfileOptions::default()
    };
    let (started, (method, args)) = tokio::join!(
        worker.start_profile(&options),
        answer_control_call(&mut engine, true, None)
    );
    started.expect("the start succeeded");
    assert_eq!(method, "start_profile");
    let fields = args[0].as_map().expect("the start's options");
    let field = |key: &str| {
        fields
            .iter()
            .find(|(name, _)| name.as_str() == Some(key))
            .map(|(_, value)| value)
    };
    assert_eq!(field("output_dir"), Some(&OpaqueValue::from("/traces")));
    assert_eq!(field("num_steps"), Some(&OpaqueValue::from(8i64)));
    assert_eq!(field("profile_by_stage"), Some(&OpaqueValue::Boolean(true)));
    assert!(field("start_step").is_none());
    let profile_id = field("profile_id")
        .and_then(OpaqueValue::as_str)
        .expect("a stamped profile id");
    assert!(profile_id.parse::<f64>().is_ok(), "{profile_id}");

    let (stopped, (method, args)) = tokio::join!(
        worker.stop_profile(),
        answer_control_call(&mut engine, false, Some("nothing is running"))
    );
    assert_eq!(method, "stop_profile");
    assert!(args.is_empty());
    let error = stopped.expect_err("the rank's failure is the operation's");
    assert!(error.to_string().contains("nothing is running"), "{error}");
}

/// A fleet-wide cache flush fans out to the direct-ZMQ workers whose
/// engine wire carries control calls and skips the others, counted apart
/// so that `total = http + grpc + zmq` holds; a fleet of nothing but
/// skipped workers answers with the skip named. (Beside the worker rather
/// than the manager: the injected backend needs the worker's own cell.)
#[tokio::test]
async fn a_fleet_flush_skips_the_zmq_workers_without_control_calls_and_counts_them() {
    use engine_zmq_client::mock_engine::default_ready_response;
    use smg_worker::health::manager::WorkerManager;

    use crate::routers::common::snapshot::WorkerRegistry;

    let dir = tempfile::tempdir().unwrap();
    let (silent, _silent_rank) =
        zmq_worker_on_a_mock_rank(&dir.path().join("silent"), default_ready_response()).await;
    let (speaking, mut engine) =
        zmq_worker_on_a_mock_rank(&dir.path().join("speaking"), control_ready()).await;
    let speaking_url = speaking.url().to_string();
    let registry = WorkerRegistry::new();
    registry.register(Arc::new(silent)).expect("registered");
    registry.register(Arc::new(speaking)).expect("registered");

    let (result, (method, _)) = tokio::join!(
        WorkerManager::flush_cache_all(&registry),
        answer_control_call(&mut engine, true, None)
    );
    assert_eq!(method, "flush_cache");
    assert_eq!(
        (
            result.total_workers,
            result.http_workers,
            result.grpc_workers,
            result.zmq_workers
        ),
        (2, 0, 1, 1)
    );
    assert_eq!(result.successful, vec![speaking_url]);
    assert!(result.failed.is_empty(), "{:?}", result.failed);

    let (alone, _alone_rank) =
        zmq_worker_on_a_mock_rank(&dir.path().join("alone"), default_ready_response()).await;
    let only_silent = WorkerRegistry::new();
    only_silent.register(Arc::new(alone)).expect("registered");
    let result = WorkerManager::flush_cache_all(&only_silent).await;
    assert_eq!((result.total_workers, result.zmq_workers), (1, 1));
    assert!(result.successful.is_empty() && result.failed.is_empty());
    assert!(
        result
            .message
            .contains("1 ZMQ workers skipped: no cache-flush RPC"),
        "{}",
        result.message
    );
}

/// Build a ZMQ worker whose sockets no engine will ever dial, plus the
/// data-plane socket path its handshake driver binds. The handshake takes
/// an OS-assigned loopback port: the port derived from a temporary ipc
/// path is a hash into a band other listeners on the host may hold, and a
/// collision there fails the bind these tests wait for.
fn unattended_zmq_worker(dir: &std::path::Path) -> (BasicWorker, std::path::PathBuf) {
    let worker = BasicWorkerBuilder::new(format!("ipc://{}", dir.join("ts0.ipc").display()))
        .connection_mode(ConnectionMode::Zmq)
        .zmq_handshake_address("tcp://127.0.0.1:0")
        .health_config(no_health_check())
        .build();
    (worker, dir.join("ts0.ipc-in.sock"))
}

/// Poll `cond` until it holds, up to five seconds.
async fn wait_for(cond: impl Fn() -> bool) -> bool {
    time::timeout(Duration::from_secs(5), async {
        while !cond() {
            time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .is_ok()
}

/// The ZMQ handshake can run as long as a model load, so acquisition on a
/// request or load-monitor path must hand it to the background driver and
/// fail fast instead of awaiting it inline.
#[tokio::test]
async fn zmq_get_backend_client_hands_the_handshake_to_the_background_driver() {
    let base = tempfile::tempdir().unwrap();
    let (worker, input_socket) = unattended_zmq_worker(base.path());

    let acquired = time::timeout(Duration::from_secs(5), worker.engine_connection())
        .await
        .expect("acquisition must not await the handshake");
    assert!(
        matches!(acquired, Err(WorkerError::ConnectionFailed { .. })),
        "acquisition must report the backend as not connected yet"
    );
    assert!(
        worker.engine_connection.load_full().get().is_none(),
        "no client can be cached before the handshake lands"
    );

    let handle = worker
        .zmq_connect_abort
        .load_full()
        .expect("acquisition must start the background handshake driver");
    assert!(
        wait_for(|| input_socket.exists()).await,
        "driver never bound the data-plane sockets"
    );
    assert!(
        !handle.is_finished(),
        "driver must still be waiting for the engine"
    );

    // A second acquisition rides the in-flight driver instead of spawning
    // another one (which would rebind the same sockets).
    assert!(worker.engine_connection().await.is_err());
    assert!(Arc::ptr_eq(
        &handle,
        &worker
            .zmq_connect_abort
            .load_full()
            .expect("driver still recorded")
    ));

    worker.abort_background_tasks();
}

/// The driver holds the ipc data-plane and TCP handshake binds until it
/// lands, so worker removal/replacement must abort it — otherwise a
/// same-URL re-registration collides with an orphan for up to the connect
/// timeout.
#[tokio::test]
async fn abort_background_tasks_cancels_the_zmq_handshake_driver() {
    let base = tempfile::tempdir().unwrap();
    let (worker, input_socket) = unattended_zmq_worker(base.path());

    assert!(
        !worker.zmq_health_check(),
        "worker is not ready until the handshake lands"
    );
    let handle = worker
        .zmq_connect_abort
        .load_full()
        .expect("probe must start the background handshake driver");
    assert!(
        wait_for(|| input_socket.exists()).await,
        "driver never bound the data-plane sockets"
    );

    worker.abort_background_tasks();

    assert!(
        wait_for(|| handle.is_finished()).await,
        "abort must cancel the in-flight handshake"
    );
    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "the aborted driver must not stay recorded"
    );
    assert!(
        !worker.zmq_connect_started.load(Ordering::SeqCst),
        "handshake guard must reset so a later probe can retry"
    );
}
