use std::ffi::CString;

use serde_json::{json, Value};
use smg::policies::RoundRobinPolicy;
use tokio::{net::TcpListener, task::JoinHandle};

use super::{super::memory::sgl_free_string, *};

/// A listener that takes connections and never answers: a client's
/// `connect` succeeds against it (the HTTP/2 preface is written, not
/// read back), and once it is closed every request on the connection
/// fails at the send.
struct SilentListener {
    endpoint: String,
    accepting: JoinHandle<()>,
}

impl SilentListener {
    fn bind() -> Self {
        let listener = RUNTIME
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("bind loopback");
        let endpoint = format!("grpc://{}", listener.local_addr().expect("local addr"));
        let accepting = RUNTIME.spawn(async move {
            let mut open = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                open.push(socket);
            }
        });
        Self {
            endpoint,
            accepting,
        }
    }

    /// Close the listener and every connection it took.
    fn close(self) {
        self.accepting.abort();
        let _ = RUNTIME.block_on(self.accepting);
    }
}

fn connect(endpoint: &str) -> Arc<SglangSchedulerClient> {
    // The worker's HTTP client wants the process-level TLS provider the
    // create entry point installs.
    smg::server::install_crypto_provider();
    Arc::new(
        RUNTIME
            .block_on(SglangSchedulerClient::connect(endpoint))
            .expect("connect"),
    )
}

/// A policy client over one worker at `endpoint`, with the test
/// tokenizer, and that worker.
fn client_over(
    client: &Arc<SglangSchedulerClient>,
    endpoint: &str,
) -> (MultiWorkerClientHandle, Arc<GrpcWorker>) {
    let worker = Arc::new(GrpcWorker::new(Arc::clone(client), endpoint.to_string()));
    let handle = MultiWorkerClientHandle {
        workers: vec![Arc::clone(&worker) as Arc<dyn Worker>],
        grpc_workers: vec![Arc::clone(&worker)],
        policy: Arc::new(RoundRobinPolicy::new()),
        tokenizer_path: "mock".to_string(),
    };
    (handle, worker)
}

/// The chat entry point's answer to a request that fails: its code and
/// message.
fn failed_chat(handle: &mut MultiWorkerClientHandle, request: &Value) -> (SglErrorCode, String) {
    let request = CString::new(request.to_string()).unwrap();
    let mut stream: *mut SglangStreamHandle = ptr::null_mut();
    let mut error: *mut c_char = ptr::null_mut();
    // SAFETY: every pointer is a live local; the message is taken back
    // once, through the entry point that frees it.
    unsafe {
        let code = sgl_multi_client_chat_completion_stream(
            handle,
            request.as_ptr(),
            &mut stream,
            &mut error,
        );
        assert!(stream.is_null(), "no stream handle on a failure");
        let message = if error.is_null() {
            String::new()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        };
        sgl_free_string(error);
        (code, message)
    }
}

/// The load a request holds is counted when taken and given back when
/// it drops, or when it is released for the stream's bookkeeping.
#[test]
fn a_held_load_is_given_back_when_it_drops_or_is_released() {
    let listener = SilentListener::bind();
    let client = connect(&listener.endpoint);
    let worker = Arc::new(GrpcWorker::new(client, listener.endpoint.clone()));
    let held = HeldLoad::take(Arc::clone(&worker));
    let other = HeldLoad::take(Arc::clone(&worker));
    assert_eq!(worker.load(), 2);
    drop(held);
    assert_eq!(worker.load(), 1);
    assert!(Arc::ptr_eq(&other.release(), &worker));
    assert_eq!(worker.load(), 0);
    listener.close();
}

/// A request that fails after the policy selected its worker, at any of
/// the three points between the selection and the open stream (the
/// tool constraint, the generate request, the send), leaves the
/// worker's load as it found it, and counts nothing as processed.
#[test]
fn a_request_that_fails_after_selection_gives_the_workers_load_back() {
    let mut listener = Some(SilentListener::bind());
    let endpoint = listener.as_ref().unwrap().endpoint.clone();
    let client = connect(&endpoint);

    let plain = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let mut unknown_function = plain.clone();
    unknown_function["tools"] = json!([{"type": "function", "function":
        {"name": "get_time", "parameters": {"type": "object"}}}]);
    unknown_function["tool_choice"] = json!({"type": "function", "function": {"name": "missing"}});
    let mut two_constraints = plain.clone();
    two_constraints["response_format"] = json!({"type": "json_object"});
    two_constraints["ebnf"] = json!("root ::= \"a\"");
    let cases = [
        (
            "tool constraints",
            unknown_function,
            SglErrorCode::ParsingError,
            "Failed to generate tool constraints: ",
        ),
        (
            "generate request",
            two_constraints,
            SglErrorCode::ParsingError,
            "Failed to build generate request: ",
        ),
        (
            "send",
            plain,
            SglErrorCode::UnknownError,
            "Failed to send request: ",
        ),
    ];
    let mut after = Vec::new();
    for (point, request, expected_code, prefix) in cases {
        if point == "send" {
            // The engine goes away under the connected client.
            listener.take().unwrap().close();
        }
        let (mut handle, worker) = client_over(&client, &endpoint);
        for _ in 0..3 {
            let (code, message) = failed_chat(&mut handle, &request);
            assert_eq!(code, expected_code, "{point}: {message}");
            assert!(message.starts_with(prefix), "{point}: {message}");
        }
        after.push((point, worker.load(), worker.processed_requests()));
    }
    assert_eq!(
        after,
        [
            ("tool constraints", 0, 0),
            ("generate request", 0, 0),
            ("send", 0, 0),
        ],
        "(failure point, load, processed) after three failed requests"
    );
}
