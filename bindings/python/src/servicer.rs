//! Python lifecycle bindings for the Rust engine servicers.
//!
//! One binding per engine servicer, each over that engine's same-host ZMQ
//! wire; `VllmGrpcServer` is the first, and the servicers for the other ZMQ
//! engines bind the same way as their protocols land. Rust owns the
//! listener, the engine link, and the request path; Python launches the
//! headless engine, starts and stops the server, and announces draining. The
//! one request-time crossing is worker-side media processing: a request's
//! `media_refs` go to a Python object that runs the engine's own processors
//! (the Python servicer's, with vLLM's input processor behind them), and
//! what it produces comes back as bytes the engine reads directly.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use engine_servicer::{
    BoxFuture, MediaError, MediaProcessor, MediaRequest, ProcessedMedia, ServicerError,
    VllmModelInfo, VllmServicerConfig, VllmServicerServer,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use tokio::sync::oneshot;

fn to_py_err(error: ServicerError) -> PyErr {
    match error {
        ServicerError::InvalidConfig(_) => PyValueError::new_err(error.to_string()),
        ServicerError::StopTimeout => PyTimeoutError::new_err(error.to_string()),
        ServicerError::Startup(_) | ServicerError::Poisoned | ServicerError::ThreadPanicked => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

/// Install the Rust tracing subscriber for a process that only hosts a
/// servicer. `level` overrides `RUST_LOG`.
#[pyfunction]
#[pyo3(signature = (level = None))]
pub fn init_servicer_tracing(level: Option<&str>) -> PyResult<()> {
    engine_servicer::init_tracing(level).map_err(to_py_err)
}

/// The worker-side media processor behind a Python object (see
/// `smg_grpc_servicer.vllm.rust_media.RustMediaBridge`): `name`, `schemes`,
/// `source` and `max_inflight` attributes, and `probe(done)` /
/// `submit(request_id, prompt_token_ids, prompt_text, items, arrival_time,
/// want_identity, done)` methods that schedule the work on Python's loop
/// and call `done` from whichever thread finishes it. Calls into Python
/// only to hand work over, from a blocking thread: taking the GIL can wait
/// behind the processor's own Python threads, and that wait must not hold
/// one of the servicer's few runtime workers. The wait for the result is a
/// Rust channel.
struct PythonMediaProcessor {
    bridge: Arc<Py<PyAny>>,
    name: String,
    /// The schemes the backend accepts, as of its last probe (a sidecar
    /// announces its own set when it answers).
    schemes: Arc<Mutex<String>>,
    source: String,
    max_inflight: usize,
}

impl PythonMediaProcessor {
    fn new(bridge: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            name: bridge.getattr("name")?.extract()?,
            schemes: Arc::new(Mutex::new(bridge.getattr("schemes")?.extract()?)),
            source: bridge.getattr("source")?.extract()?,
            max_inflight: bridge.getattr("max_inflight")?.extract()?,
            bridge: Arc::new(bridge.clone().unbind()),
        })
    }

    /// Run `call` against the bridge on a blocking thread, with the GIL.
    async fn call_bridge(
        bridge: Arc<Py<PyAny>>,
        call: impl FnOnce(Python<'_>, &Py<PyAny>) -> PyResult<()> + Send + 'static,
    ) -> Result<(), String> {
        match tokio::task::spawn_blocking(move || Python::attach(|py| call(py, &bridge))).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(error.to_string()),
            Err(join) => Err(format!("media bridge call did not run: {join}")),
        }
    }
}

impl MediaProcessor for PythonMediaProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    fn schemes(&self) -> String {
        self.schemes
            .lock()
            .map(|schemes| schemes.clone())
            .unwrap_or_default()
    }

    fn source(&self) -> &str {
        &self.source
    }

    fn max_inflight(&self) -> usize {
        self.max_inflight
    }

    fn probe(&self) -> BoxFuture<bool> {
        let bridge = Arc::clone(&self.bridge);
        let schemes = Arc::clone(&self.schemes);
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            let scheduled = Self::call_bridge(bridge, move |py, bridge| {
                let done = Py::new(
                    py,
                    ProbeDone {
                        tx: Mutex::new(Some(tx)),
                    },
                )?;
                bridge.call_method1(py, "probe", (done,))?;
                Ok(())
            })
            .await;
            match scheduled {
                Ok(()) => match rx.await {
                    Ok((serving, announced)) => {
                        if let (Some(announced), Ok(mut schemes)) = (announced, schemes.lock()) {
                            *schemes = announced;
                        }
                        serving
                    }
                    Err(_) => false,
                },
                Err(error) => {
                    tracing::warn!(error, "media processor probe could not be scheduled");
                    false
                }
            }
        })
    }

    fn process(&self, request: MediaRequest) -> BoxFuture<Result<ProcessedMedia, MediaError>> {
        let bridge = Arc::clone(&self.bridge);
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            let scheduled = Self::call_bridge(bridge, move |py, bridge| {
                let done = Py::new(
                    py,
                    MediaDone {
                        tx: Mutex::new(Some(tx)),
                    },
                )?;
                let items: Vec<(String, String)> = request
                    .items
                    .into_iter()
                    .map(|item| (item.modality, item.url))
                    .collect();
                bridge.call_method1(
                    py,
                    "submit",
                    (
                        request.request_id,
                        request.prompt_token_ids,
                        request.prompt_text,
                        items,
                        request.arrival_time,
                        request.want_identity,
                        done,
                    ),
                )?;
                Ok(())
            })
            .await;
            if let Err(error) = scheduled {
                return Err(MediaError::Internal(format!(
                    "media processor could not take the request: {error}"
                )));
            }
            rx.await.unwrap_or_else(|_| {
                Err(MediaError::Unavailable(
                    "media processor dropped the request".to_string(),
                ))
            })
        })
    }
}

/// Python's `bytes` copied into Rust-owned bytes; for the small buffers.
fn bytes_of(value: &Bound<'_, PyBytes>) -> Bytes {
    Bytes::copy_from_slice(value.as_bytes())
}

/// Memory lent by Python without a copy: `_owner` (a read-only numpy view
/// over the request's own tensor storage) keeps it alive for as long as these
/// bytes exist, and nothing writes to it in the meantime. The buffer protocol
/// is not part of the limited API this extension builds against, so the
/// range is read off the view itself. Dropped off a Python thread, the
/// owner's reference is released on the next GIL acquisition.
struct PyBacked {
    _owner: Py<PyAny>,
    ptr: *const u8,
    len: usize,
}

// SAFETY: the memory is only read, and the Python object that owns it is
// held for the lifetime of this value; `Py<PyAny>` is itself Send + Sync.
#[expect(
    unsafe_code,
    reason = "a raw pointer into Python-owned memory the owner keeps alive"
)]
unsafe impl Send for PyBacked {}
#[expect(
    unsafe_code,
    reason = "a raw pointer into Python-owned memory the owner keeps alive"
)]
unsafe impl Sync for PyBacked {}

impl AsRef<[u8]> for PyBacked {
    #[expect(
        unsafe_code,
        reason = "the limited API has no buffer protocol; the owner guarantees the range"
    )]
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr`/`len` describe the owner's contiguous storage, which
        // outlives this value (see the struct doc).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// The `ok` payload: prompt ids, the encoded features, the lent frames (one
/// numpy view each), the cache salt, the serialized identity.
type OkPayload<'py> = (
    Vec<u32>,
    Option<Bound<'py, PyBytes>>,
    Vec<Bound<'py, PyAny>>,
    Option<String>,
    Option<Bound<'py, PyBytes>>,
);

/// A lent frame as zero-copy bytes. The range is read off the view itself,
/// never taken from the caller: an address that did not belong to `owner`
/// would be read out of bounds on a runtime thread. The view must be
/// C-contiguous and read-only (the latter is the owner's promise that no
/// Python write races the send).
fn lent_bytes(owner: Bound<'_, PyAny>) -> PyResult<Bytes> {
    let flags = owner.getattr("flags")?;
    if !flags.getattr("c_contiguous")?.extract::<bool>()? {
        return Err(PyValueError::new_err("a lent buffer must be C-contiguous"));
    }
    if flags.getattr("writeable")?.extract::<bool>()? {
        return Err(PyValueError::new_err("a lent buffer must be read-only"));
    }
    let address: usize = owner.getattr("ctypes")?.getattr("data")?.extract()?;
    let len: usize = owner.getattr("nbytes")?.extract()?;
    if address == 0 && len > 0 {
        return Err(PyValueError::new_err("a lent buffer has no address"));
    }
    Ok(Bytes::from_owner(PyBacked {
        _owner: owner.unbind(),
        ptr: address as *const u8,
        len,
    }))
}

/// Python's answer to one media request, called once: `("ok",
/// (prompt_token_ids, mm_features | None, aux_frames, cache_salt | None,
/// media_identity | None))` where each aux frame is a read-only numpy view
/// lent without a copy, or `(kind, message)` with `kind` one of `invalid`,
/// `unavailable`, `internal`.
#[pyclass(name = "_MediaDone")]
struct MediaDone {
    tx: Mutex<Option<oneshot::Sender<Result<ProcessedMedia, MediaError>>>>,
}

#[pymethods]
impl MediaDone {
    fn __call__(&self, kind: &str, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let outcome = match kind {
            "ok" => {
                let (prompt_token_ids, mm_features, aux_frames, cache_salt, media_identity): OkPayload<'_> =
                    payload.extract()?;
                Ok(ProcessedMedia {
                    prompt_token_ids,
                    mm_features: mm_features.as_ref().map(bytes_of),
                    aux_frames: aux_frames
                        .into_iter()
                        .map(lent_bytes)
                        .collect::<PyResult<_>>()?,
                    cache_salt,
                    media_identity: media_identity.as_ref().map(bytes_of),
                })
            }
            "invalid" => Err(MediaError::Invalid(payload.extract()?)),
            "unavailable" => Err(MediaError::Unavailable(payload.extract()?)),
            _ => Err(MediaError::Internal(payload.extract()?)),
        };
        let sender = self
            .tx
            .lock()
            .map_err(|_| PyRuntimeError::new_err("media completion poisoned"))?
            .take();
        // A request whose caller gave up has no receiver; nothing to report.
        if let Some(sender) = sender {
            let _ = sender.send(outcome);
        }
        Ok(())
    }
}

/// What a probe answers: whether the processor serves and, when it knows
/// them, the schemes it accepts now.
type ProbeAnswer = (bool, Option<String>);

/// Python's answer to a probe, called once.
#[pyclass(name = "_ProbeDone")]
struct ProbeDone {
    tx: Mutex<Option<oneshot::Sender<ProbeAnswer>>>,
}

#[pymethods]
impl ProbeDone {
    #[pyo3(signature = (serving, schemes = None))]
    fn __call__(&self, serving: bool, schemes: Option<String>) -> PyResult<()> {
        let sender = self
            .tx
            .lock()
            .map_err(|_| PyRuntimeError::new_err("probe completion poisoned"))?
            .take();
        if let Some(sender) = sender {
            let _ = sender.send((serving, schemes));
        }
        Ok(())
    }
}

/// Rust-owned `vllm.grpc.engine.VllmEngine` server over a same-host vLLM
/// EngineCore, with lifecycle driven from Python.
#[pyclass(name = "VllmGrpcServer")]
pub struct PyVllmGrpcServer {
    inner: VllmServicerServer,
}

#[pymethods]
impl PyVllmGrpcServer {
    #[new]
    #[pyo3(signature = (
        bind_address,
        ipc_base_url,
        handshake_address,
        model_path,
        *,
        engine_count = 1,
        tokenizer_dir = None,
        served_model_name = None,
        tokenizer_path = None,
        is_generation = true,
        max_context_length = 0,
        vocab_size = 0,
        supports_vision = false,
        model_type = String::new(),
        architectures = None,
        eos_token_ids = None,
        pad_token_id = 0,
        bos_token_id = 0,
        default_sampling_params_json = String::new(),
        data_parallel_size = 1,
        pairing_protocol = String::new(),
        kv_connector = String::new(),
        kv_role = String::new(),
        kv_engine_id = String::new(),
        kv_cache_dtype = String::new(),
        attention_backend = String::new(),
        model_dtype = String::new(),
        block_size = 0,
        structured_outputs_backend = String::new(),
        kv_events_endpoint = String::new(),
        kv_events_replay_endpoint = String::new(),
        kv_events_topic = String::new(),
        shm_namespace_id = String::new(),
        pooler_use_activation = None,
        pooler_dimensions = None,
        media_processor = None,
    ))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bind_address: String,
        ipc_base_url: String,
        handshake_address: String,
        model_path: String,
        engine_count: usize,
        tokenizer_dir: Option<String>,
        served_model_name: Option<String>,
        tokenizer_path: Option<String>,
        is_generation: bool,
        max_context_length: u32,
        vocab_size: u32,
        supports_vision: bool,
        model_type: String,
        architectures: Option<Vec<String>>,
        eos_token_ids: Option<Vec<u32>>,
        pad_token_id: i32,
        bos_token_id: i32,
        default_sampling_params_json: String,
        data_parallel_size: i32,
        pairing_protocol: String,
        kv_connector: String,
        kv_role: String,
        kv_engine_id: String,
        kv_cache_dtype: String,
        attention_backend: String,
        model_dtype: String,
        block_size: i32,
        structured_outputs_backend: String,
        kv_events_endpoint: String,
        kv_events_replay_endpoint: String,
        kv_events_topic: String,
        shm_namespace_id: String,
        pooler_use_activation: Option<bool>,
        pooler_dimensions: Option<u32>,
        media_processor: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let media_processor = media_processor
            .map(|bridge| PythonMediaProcessor::new(&bridge))
            .transpose()?
            .map(|processor| Arc::new(processor) as Arc<dyn MediaProcessor>);
        let model = VllmModelInfo {
            served_model_name: served_model_name.unwrap_or_else(|| model_path.clone()),
            tokenizer_path: tokenizer_path.unwrap_or_else(|| model_path.clone()),
            model_path,
            is_generation,
            max_context_length,
            vocab_size,
            supports_vision,
            model_type,
            architectures: architectures.unwrap_or_default(),
            eos_token_ids: eos_token_ids.unwrap_or_default(),
            pad_token_id,
            bos_token_id,
            default_sampling_params_json,
            data_parallel_size,
            pairing_protocol,
            kv_connector,
            kv_role,
            kv_engine_id,
            kv_cache_dtype,
            attention_backend,
            model_dtype,
            block_size,
            structured_outputs_backend,
            kv_events_endpoint,
            kv_events_replay_endpoint,
            kv_events_topic,
            shm_namespace_id,
            pooler_use_activation,
            pooler_dimensions,
        };
        let config = VllmServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            model,
            media_processor,
        };
        let inner = py
            .detach(|| VllmServicerServer::start(config))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// The bound `ip:port`.
    #[getter]
    fn address(&self) -> String {
        self.inner.address()
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    /// Whether the engine handshake completed; health is NOT_SERVING until it has.
    #[getter]
    fn engine_ready(&self) -> bool {
        self.inner.engine_ready()
    }

    /// The last fatal error (engine connect or server exit), if any.
    #[getter]
    fn last_error(&self) -> PyResult<Option<String>> {
        self.inner.last_error().map_err(to_py_err)
    }

    /// Announce SERVING (`True`) or drain (`False`); health flips at once.
    fn set_serving(&self, serving: bool) {
        self.inner.set_serving(serving);
    }

    /// Stop the server: open streams are cancelled, connections get most of
    /// `timeout_secs` to close, then the listener thread is joined.
    #[pyo3(signature = (timeout_secs = 5.0))]
    fn stop(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
        if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
            return Err(PyValueError::new_err(
                "timeout_secs must be finite and positive",
            ));
        }
        let timeout = Duration::from_secs_f64(timeout_secs);
        py.detach(|| self.inner.stop(timeout)).map_err(to_py_err)
    }
}
