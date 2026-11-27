use std::sync::mpsc;

use super::*;
use crate::{metrics::Identity, record_error};

fn shared() -> Shared {
    let metrics = ServicerMetrics::new(
        &Identity {
            model: "m".to_string(),
            wire: "test",
            ranks: 1,
            max_model_len: 8,
            matches_string_stops: false,
            processes_media: false,
        },
        false,
    )
    .expect("metrics without a listener");
    Shared::new(None, metrics, None, None)
}

/// Before the engine is up every reader answers from the launcher's
/// figures alone: no capacity, the launcher's window or none, the
/// launcher's version or `unknown`, the launcher's ranks or one.
#[test]
fn the_handshake_readers_fall_back_to_the_launcher_before_the_engine_is_up() {
    let shared = shared();
    assert!(shared.ready().is_none());
    assert_eq!(shared.kv_capacity_tokens(), 0);
    assert_eq!(shared.running_window(64), 64);
    assert_eq!(shared.running_window(0), 0);
    assert_eq!(shared.engine_version("0.5.1", "wire-"), "0.5.1");
    assert_eq!(shared.engine_version("", "wire-"), "unknown");
    assert_eq!(shared.rank_count(4), 4);
    assert_eq!(shared.rank_count(0), 1);
    assert_eq!(shared.rank_count(-2), 1);
}

/// The stop waits only for an endpoint file of its own: a path nothing
/// holds yet. A file already there is an earlier process's, and an
/// address that names no file leaves nothing to wait for.
#[tokio::test]
async fn the_stop_waits_only_for_a_handshake_endpoint_of_its_own() {
    use std::os::unix::net::UnixListener;

    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("handshake");
    let address = format!("ipc://{}", path.display());
    assert_eq!(
        own_handshake_endpoint(&address).await,
        Some(path.display().to_string())
    );

    std::fs::write(&path, b"").expect("an earlier process's file");
    assert_eq!(own_handshake_endpoint(&address).await, None);
    assert_eq!(own_handshake_endpoint("tcp://127.0.0.1:5555").await, None);

    std::fs::remove_file(&path).expect("the file goes");
    let listener = UnixListener::bind(&path).expect("a listener");
    assert_eq!(
        own_handshake_endpoint(&address).await,
        None,
        "a socket a listener answers at is a live process's"
    );
    drop(listener);
    // The bind sequence sweeps a leftover on Linux alone.
    let swept = cfg!(target_os = "linux").then(|| path.display().to_string());
    assert_eq!(
        own_handshake_endpoint(&address).await,
        swept,
        "a leftover socket reads as free where the bind sequence sweeps it"
    );
}

/// The drain switch closes admission with health; the readiness switch
/// admits the owner's own warm-up requests alone, and its release admits
/// everyone again.
#[test]
fn the_two_switches_differ_in_admission() {
    let shared = shared();
    assert!(shared.registry.is_accepting());
    assert!(!shared.is_serving(), "no engine link yet");
    assert_eq!(shared.health_message("healthy"), "Engine is starting");
    assert_eq!(shared.snapshot().state, Some(ServicerState::Starting));

    shared.set_ready(false);
    assert!(
        shared.registry.is_accepting(),
        "readiness held: a hold is not a drain"
    );
    assert_eq!(shared.health_message("healthy"), "Warming up");
    let refused = shared
        .registry
        .register("chatcmpl-1")
        .err()
        .expect("a foreign request is refused while readiness is held");
    assert_eq!(refused.message(), "servicer is warming up");
    let (own, _cancel) = shared
        .registry
        .register("smg-warmup-1")
        .expect("the warm-up's own request is admitted");
    drop(own);

    shared.set_serving(false);
    assert!(!shared.registry.is_accepting(), "a drain closes admission");
    assert_eq!(shared.health_message("healthy"), "Draining");
    assert_eq!(shared.snapshot().state, Some(ServicerState::Draining));

    shared.set_serving(true);
    assert!(shared.registry.is_accepting());
    assert_eq!(shared.health_message("healthy"), "Engine is starting");
    assert_eq!(
        shared
            .registry
            .register("chatcmpl-2")
            .err()
            .expect("lifting the drain does not lift the readiness hold")
            .message(),
        "servicer is warming up"
    );

    shared.set_ready(true);
    let (foreign, _cancel) = shared
        .registry
        .register("chatcmpl-2")
        .expect("the release admits every request again");
    drop(foreign);
}

/// Without a link the engine is "still connecting"; a recorded failure
/// names itself.
#[test]
fn the_engine_error_reads_the_link() {
    let shared = shared();
    assert!(!shared.engine_ready());
    let status = shared.engine().err().expect("no client yet");
    assert_eq!(status.code(), tonic::Code::Unavailable);
    assert_eq!(status.message(), "engine is still connecting");

    shared.engine.fail("refused".to_string());
    let status = shared.engine().err().expect("still no client");
    assert_eq!(status.message(), "engine connection failed: refused");
    assert_eq!(shared.health_message("healthy"), "Engine connection failed");
    assert_eq!(shared.snapshot().state, Some(ServicerState::EngineDown));
}

/// A wire of the shared state alone: no engine, no load figures.
struct TestState(Shared);

impl WireState for TestState {
    fn shared(&self) -> &Shared {
        &self.0
    }
}

impl RankLoads for TestState {
    fn rank_load(
        &self,
        _dp_rank: Option<i32>,
    ) -> Option<smg_grpc_client::common_proto::EngineLoad> {
        None
    }

    fn engine_link(&self) -> &EngineLink {
        &self.0.engine
    }

    fn rank_count(&self) -> usize {
        1
    }
}

/// Once the listener is bound, the start hands the wire's connect the
/// inputs the configuration said and runs it on the server runtime; what
/// the connect records in its error slot is the server's last error.
#[test]
fn the_start_hands_the_connect_the_configurations_inputs_once_bound() {
    let state = Arc::new(TestState(shared()));
    let mut config = ServicerConfig::new("127.0.0.1:0", "ipc:///tmp/engine", "tcp://127.0.0.1:1");
    config.engine_count = 3;
    config.engine_startup_timeout = Duration::from_secs(7);
    config.engine_startup_ceiling = None;
    config.engine_silence_timeout = Duration::from_secs(11);
    config.engine_prefill_tokens_per_sec = 500;
    let (seen_tx, seen_rx) = mpsc::channel();
    let server = Server::start(
        "smg-test-servicer",
        config,
        "test.Service",
        "Test",
        &state,
        Routes::default(),
        move |inputs| async move {
            record_error(&inputs.last_error, "the test's connect failed".to_string());
            let _ = seen_tx.send((
                inputs.ipc_base_url,
                inputs.handshake_address,
                inputs.engine_count,
                inputs.startup_timeout,
                inputs.startup_ceiling,
                inputs.silence.window,
                inputs.silence.prefill_tokens_per_sec,
                std::thread::current().name().map(str::to_string),
            ));
        },
    )
    .expect("the server starts");
    let (ipc, handshake, engines, timeout, ceiling, window, rate, thread) = seen_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the connect ran");
    assert_eq!(ipc, "ipc:///tmp/engine");
    assert_eq!(handshake, "tcp://127.0.0.1:1");
    assert_eq!(engines, 3);
    assert_eq!(timeout, Duration::from_secs(7));
    assert_eq!(ceiling, None);
    assert_eq!(window, Duration::from_secs(11));
    assert_eq!(rate, 500);
    assert_eq!(
        thread.as_deref(),
        Some("smg-servicer-rt"),
        "the server runtime"
    );
    assert!(server.running());
    assert_eq!(
        server
            .last_error()
            .expect("the slot is readable")
            .as_deref(),
        Some("the test's connect failed")
    );
    server
        .stop(&state.0, Duration::from_secs(5))
        .expect("the server stops");
}
