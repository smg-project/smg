//! Python lifecycle bindings for the Rust engine servicers.
//!
//! One binding per engine servicer, each over that engine's same-host ZMQ
//! wire; `VllmGrpcServer` is the first, and the servicers for the other ZMQ
//! engines bind the same way as their protocols land. Rust owns the
//! listener, the engine link, and the request path; Python launches the
//! headless engine, starts and stops the server, and announces draining. The
//! one request-time crossing is worker-side media processing, when the
//! Python processor is chosen: a request's `media_refs` go to a Python object
//! that runs the engine's own processors (the Python servicer's, with vLLM's
//! input processor behind them), and what it produces comes back as bytes
//! the engine reads directly. The other choice, smg's own pipeline
//! ([`SmgMediaProcessor`]), never crosses: it is the Router's media pipeline
//! run in this process.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use engine_servicer::{
    BoxFuture, MediaError, MediaFeatures, MediaProcessor, MediaRequest, ProcessedMedia,
    ServicerError, SglangModelInfo, SglangServicerConfig, SglangServicerServer,
    TokenSpeedModelInfo, TokenSpeedServicerConfig, TokenSpeedServicerServer, VllmModelInfo,
    VllmServicerConfig, VllmServicerServer, DEFAULT_ENGINE_STARTUP_CEILING,
    DEFAULT_ENGINE_STARTUP_TIMEOUT,
};
use llm_multimodal::Modality;
use prost::Message;
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use smg::routers::grpc::worker_media::{
    PixelFormat, WorkerMediaError, WorkerMediaItem, WorkerMediaPipeline, WorkerMediaSettings,
    SCHEMES,
};
use smg_grpc_client::vllm_proto as vllm;
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

/// The engine startup bound from the launcher's seconds; `None` is the crate
/// default.
fn startup_timeout(secs: Option<f64>) -> PyResult<Duration> {
    match secs {
        None => Ok(DEFAULT_ENGINE_STARTUP_TIMEOUT),
        Some(secs) if secs > 0.0 => Duration::try_from_secs_f64(secs).map_err(|error| {
            PyValueError::new_err(format!("engine_startup_timeout_secs {secs}: {error}"))
        }),
        Some(secs) => Err(PyValueError::new_err(format!(
            "engine_startup_timeout_secs must be positive, got {secs}"
        ))),
    }
}

/// The bound on the whole engine start from the launcher's seconds: `None`
/// is the crate default, `0` is no ceiling.
fn startup_ceiling(secs: Option<f64>) -> PyResult<Option<Duration>> {
    let Some(secs) = secs else {
        return Ok(Some(DEFAULT_ENGINE_STARTUP_CEILING));
    };
    if secs == 0.0 {
        return Ok(None);
    }
    if secs < 0.0 || secs.is_nan() {
        return Err(PyValueError::new_err(format!(
            "engine_startup_ceiling_secs must be positive, or 0 for no ceiling, got {secs}"
        )));
    }
    Duration::try_from_secs_f64(secs)
        .map(Some)
        .map_err(|error| {
            PyValueError::new_err(format!("engine_startup_ceiling_secs {secs}: {error}"))
        })
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
                let media_identity = media_identity
                    .map(|bytes| vllm::MediaIdentity::decode(bytes.as_bytes()))
                    .transpose()
                    .map_err(|error| {
                        PyValueError::new_err(format!(
                            "media identity could not be decoded: {error}"
                        ))
                    })?;
                Ok(ProcessedMedia {
                    prompt_token_ids,
                    features: MediaFeatures::Encoded {
                        mm_features: mm_features.as_ref().map(bytes_of),
                        aux_frames: aux_frames
                            .into_iter()
                            .map(lent_bytes)
                            .collect::<PyResult<_>>()?,
                        cache_salt,
                    },
                    media_identity,
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

/// smg's own media pipeline as the worker-side processor (`--mm-processor
/// smg`): the Router's fetch, decode, preprocess and placeholder expansion,
/// run in this process, answering in the Router's batch shape. No Python on
/// the request path.
struct SmgMediaProcessor {
    pipeline: Arc<WorkerMediaPipeline>,
    source: String,
    max_inflight: usize,
}

impl MediaProcessor for SmgMediaProcessor {
    fn name(&self) -> &str {
        "smg"
    }

    fn schemes(&self) -> String {
        SCHEMES.to_string()
    }

    fn source(&self) -> &str {
        &self.source
    }

    fn max_inflight(&self) -> usize {
        self.max_inflight
    }

    fn probe(&self) -> BoxFuture<bool> {
        // In-process and stateless between requests: serving whenever up.
        Box::pin(async { true })
    }

    fn process(&self, request: MediaRequest) -> BoxFuture<Result<ProcessedMedia, MediaError>> {
        let pipeline = Arc::clone(&self.pipeline);
        Box::pin(async move {
            let items = request
                .items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let modality = match item.modality.as_str() {
                        "image" => Modality::Image,
                        "video" => Modality::Video,
                        other => {
                            return Err(MediaError::Invalid(format!(
                                "media_refs[{index}]: unsupported modality {other}"
                            )))
                        }
                    };
                    Ok(WorkerMediaItem {
                        modality,
                        url: item.url.clone(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = pipeline
                .process(request.prompt_token_ids, &items, request.want_identity)
                .await
                .map_err(|error| match error {
                    WorkerMediaError::Invalid(message) => MediaError::Invalid(message),
                    WorkerMediaError::Unavailable(message) => MediaError::Unavailable(message),
                    WorkerMediaError::Internal(message) => MediaError::Internal(message),
                })?;
            Ok(ProcessedMedia {
                prompt_token_ids: output.prompt_token_ids,
                features: MediaFeatures::Batches(output.batches),
                media_identity: output.identity,
            })
        })
    }
}

/// The settings the launcher passes as `smg_media_processor`, by key; the
/// pipeline itself is built off the GIL once the tokenizer is loaded.
struct NativeMediaOptions {
    settings: WorkerMediaSettings,
    source: String,
    max_inflight: usize,
}

fn native_media_options(
    options: &Bound<'_, PyDict>,
    served_model_name: &str,
) -> PyResult<NativeMediaOptions> {
    let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(options.get_item(key)?.filter(|value| !value.is_none()))
    };
    let string =
        |key: &str| -> PyResult<Option<String>> { item(key)?.map(|v| v.extract()).transpose() };
    let count =
        |key: &str| -> PyResult<Option<usize>> { item(key)?.map(|v| v.extract()).transpose() };
    let required = |key: &str| -> PyResult<String> {
        string(key)?
            .ok_or_else(|| PyValueError::new_err(format!("smg_media_processor needs {key:?}")))
    };
    let raw_pixels = item("raw_pixels")?
        .map(|v| v.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    let allowed_domains: Option<Vec<String>> =
        item("allowed_domains")?.map(|v| v.extract()).transpose()?;
    let processor_kwargs = match string("processor_kwargs_json")? {
        Some(json) => serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&json)
            .map_err(|error| {
                PyValueError::new_err(format!(
                    "smg_media_processor processor_kwargs_json is not a JSON object: {error}"
                ))
            })?,
        None => serde_json::Map::new(),
    };
    let fetch_timeout_ms: u64 = item("fetch_timeout_ms")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(10_000);
    // The engine's per-prompt limits by modality name; one the pipeline does
    // not fetch is of no consequence here.
    let engine_item_limits = item("engine_item_limits")?
        .map(|v| v.extract::<HashMap<String, usize>>())
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(modality, limit)| {
            let modality = match modality.as_str() {
                "image" => Modality::Image,
                "video" => Modality::Video,
                "audio" => Modality::Audio,
                _ => return None,
            };
            Some((modality, limit))
        })
        .collect();
    Ok(NativeMediaOptions {
        settings: WorkerMediaSettings {
            model_dir: required("model_dir")?,
            model_id: required("model_id")?,
            served_model_name: Some(served_model_name.to_string()),
            pixel_format: if raw_pixels {
                PixelFormat::RawU8
            } else {
                PixelFormat::Normalized
            },
            encoder_dtype: string("encoder_dtype")?.unwrap_or_else(|| "float32".to_string()),
            processor_kwargs,
            max_items: count("max_items")?,
            engine_item_limits,
            max_item_bytes: count("max_item_bytes")?,
            allowed_domains,
            fetch_timeout: Duration::from_millis(fetch_timeout_ms),
        },
        source: string("source")?.unwrap_or_else(|| "default".to_string()),
        max_inflight: count("max_inflight")?.unwrap_or(4).max(1),
    })
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
        mm_device_do_normalize = false,
        max_num_seqs = 0,
        media_processor = None,
        smg_media_processor = None,
        engine_startup_timeout_secs = None,
        engine_startup_ceiling_secs = None,
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
        mm_device_do_normalize: bool,
        max_num_seqs: i32,
        media_processor: Option<Bound<'_, PyAny>>,
        smg_media_processor: Option<Bound<'_, PyDict>>,
        engine_startup_timeout_secs: Option<f64>,
        engine_startup_ceiling_secs: Option<f64>,
    ) -> PyResult<Self> {
        if media_processor.is_some() && smg_media_processor.is_some() {
            return Err(PyValueError::new_err(
                "media_processor (the Python processors) and smg_media_processor (smg's \
                 pipeline) are two choices for one slot; pass one",
            ));
        }
        let media_processor = media_processor
            .map(|bridge| PythonMediaProcessor::new(&bridge))
            .transpose()?
            .map(|processor| Arc::new(processor) as Arc<dyn MediaProcessor>);
        let served_model_name = served_model_name.unwrap_or_else(|| model_path.clone());
        let native = smg_media_processor
            .map(|options| native_media_options(&options, &served_model_name))
            .transpose()?;
        let model = VllmModelInfo {
            served_model_name,
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
            mm_device_do_normalize,
            max_num_seqs,
        };
        let mut config = VllmServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            model,
            media_processor,
            engine_startup_timeout: startup_timeout(engine_startup_timeout_secs)?,
            engine_startup_ceiling: startup_ceiling(engine_startup_ceiling_secs)?,
        };
        let inner = py.detach(|| -> PyResult<VllmServicerServer> {
            let Some(native) = native else {
                return VllmServicerServer::start(config).map_err(to_py_err);
            };
            // The pipeline needs the tokenizer the servicer would load after
            // the engine connects; load it here and hand the same one over.
            let dir = config.tokenizer_dir.clone().ok_or_else(|| {
                PyValueError::new_err("the smg media processor needs tokenizer_dir")
            })?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| {
                    PyRuntimeError::new_err(format!("media pipeline setup runtime: {error}"))
                })?;
            let (tokenizer, pipeline) = runtime.block_on(async {
                let tokenizer = llm_tokenizer::factory::create_tokenizer_async(&dir)
                    .await
                    .map_err(|error| {
                        PyRuntimeError::new_err(format!(
                            "loading the tokenizer from {dir} for the smg media processor: {error:#}"
                        ))
                    })?;
                let pipeline = WorkerMediaPipeline::new(native.settings, Arc::clone(&tokenizer))
                    .await
                    .map_err(|error| {
                        PyValueError::new_err(format!("smg media processor: {error:#}"))
                    })?;
                Ok::<_, PyErr>((tokenizer, pipeline))
            })?;
            tracing::info!(
                spec = pipeline.spec_name(),
                pixel_format = ?pipeline.pixel_format(),
                max_inflight = native.max_inflight,
                item_limits = %pipeline.item_limits_summary(),
                "smg media processor ready"
            );
            config.media_processor = Some(Arc::new(SmgMediaProcessor {
                pipeline: Arc::new(pipeline),
                source: native.source,
                max_inflight: native.max_inflight,
            }));
            VllmServicerServer::start_with_tokenizer(config, tokenizer).map_err(to_py_err)
        })?;
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

    /// Report that the engine process is alive while the handshake runs: the
    /// launcher calls this each time it polls the process it spawned and
    /// finds it running. Each report resets the handshake's silence bound
    /// (`engine_startup_timeout_secs`); the ceiling
    /// (`engine_startup_ceiling_secs`) still holds. A no-op once the engine
    /// is connected.
    fn note_engine_alive(&self) {
        self.inner.note_engine_alive();
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

/// Rust-owned `tokenspeed.grpc.scheduler.TokenSpeedScheduler` server over
/// same-host headless TokenSpeed scheduler(s) on the msgpack ZMQ wire. The
/// Python launcher keeps the lifecycle: it starts the scheduler(s) dialing
/// `handshake_address`, polls this object, and stops it.
#[pyclass(name = "TokenSpeedGrpcServer")]
pub struct PyTokenSpeedGrpcServer {
    inner: TokenSpeedServicerServer,
}

#[pymethods]
impl PyTokenSpeedGrpcServer {
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
        model_type = String::new(),
        architectures = None,
        max_context_length = 0,
        max_req_input_len = 0,
        vocab_size = 0,
        eos_token_ids = None,
        pad_token_id = 0,
        bos_token_id = 0,
        weight_version = String::new(),
        default_sampling_params_json = String::new(),
        supports_vision = false,
        supports_multimodal = false,
        supported_modalities = None,
        model_dtype = String::new(),
        multimodal_encoder_dtype = String::new(),
        server_args_json = String::new(),
        scheduler_info_json = String::new(),
        tokenspeed_version = String::new(),
        max_running_requests = 0,
        data_parallel_size = 1,
        kv_events_endpoint = String::new(),
        kv_events_replay_endpoint = String::new(),
        kv_events_topic = String::new(),
        engine_startup_timeout_secs = None,
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
        model_type: String,
        architectures: Option<Vec<String>>,
        max_context_length: i32,
        max_req_input_len: i32,
        vocab_size: i32,
        eos_token_ids: Option<Vec<u32>>,
        pad_token_id: i32,
        bos_token_id: i32,
        weight_version: String,
        default_sampling_params_json: String,
        supports_vision: bool,
        supports_multimodal: bool,
        supported_modalities: Option<Vec<i32>>,
        model_dtype: String,
        multimodal_encoder_dtype: String,
        server_args_json: String,
        scheduler_info_json: String,
        tokenspeed_version: String,
        max_running_requests: i32,
        data_parallel_size: i32,
        kv_events_endpoint: String,
        kv_events_replay_endpoint: String,
        kv_events_topic: String,
        engine_startup_timeout_secs: Option<f64>,
    ) -> PyResult<Self> {
        let model = TokenSpeedModelInfo {
            served_model_name: served_model_name.unwrap_or_else(|| model_path.clone()),
            tokenizer_path: tokenizer_path.unwrap_or_else(|| model_path.clone()),
            model_path,
            model_type,
            architectures: architectures.unwrap_or_default(),
            max_context_length,
            max_req_input_len,
            vocab_size,
            eos_token_ids: eos_token_ids.unwrap_or_default(),
            pad_token_id,
            bos_token_id,
            weight_version,
            default_sampling_params_json,
            supports_vision,
            supports_multimodal,
            supported_modalities: supported_modalities.unwrap_or_default(),
            model_dtype,
            multimodal_encoder_dtype,
            server_args_json,
            scheduler_info_json,
            tokenspeed_version,
            max_running_requests,
            data_parallel_size,
            kv_events_endpoint,
            kv_events_replay_endpoint,
            kv_events_topic,
        };
        let config = TokenSpeedServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            model,
            engine_startup_timeout: startup_timeout(engine_startup_timeout_secs)?,
        };
        let inner = py
            .detach(|| TokenSpeedServicerServer::start(config))
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

    /// Whether the scheduler handshake completed; health is NOT_SERVING
    /// until it has.
    #[getter]
    fn engine_ready(&self) -> bool {
        self.inner.engine_ready()
    }

    /// The last fatal error (scheduler connect or server exit), if any.
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

/// Rust-owned `sglang.grpc.scheduler.SglangScheduler` server over a same-host
/// headless SGLang scheduler on the msgpack ZMQ wire (the SMG plugin inside
/// the scheduler dials in). The Python launcher keeps the lifecycle: it starts
/// the scheduler dialing `handshake_address`, polls this object, and stops it.
#[pyclass(name = "SglangGrpcServer")]
pub struct PySglangGrpcServer {
    inner: SglangServicerServer,
}

#[pymethods]
impl PySglangGrpcServer {
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
        model_type = String::new(),
        architectures = None,
        max_context_length = 0,
        max_req_input_len = 0,
        vocab_size = 0,
        eos_token_ids = None,
        pad_token_id = 0,
        bos_token_id = 0,
        weight_version = String::new(),
        preferred_sampling_params = String::new(),
        default_sampling_params_json = String::new(),
        supports_vision = false,
        id2label_json = String::new(),
        num_labels = 0,
        server_args_json = String::new(),
        scheduler_info_json = String::new(),
        sglang_version = String::new(),
        max_running_requests = 0,
        data_parallel_size = 1,
        kv_events_endpoint = String::new(),
        kv_events_replay_endpoint = String::new(),
        kv_events_topic = String::new(),
        engine_startup_timeout_secs = None,
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
        model_type: String,
        architectures: Option<Vec<String>>,
        max_context_length: i32,
        max_req_input_len: i32,
        vocab_size: i32,
        eos_token_ids: Option<Vec<u32>>,
        pad_token_id: i32,
        bos_token_id: i32,
        weight_version: String,
        preferred_sampling_params: String,
        default_sampling_params_json: String,
        supports_vision: bool,
        id2label_json: String,
        num_labels: i32,
        server_args_json: String,
        scheduler_info_json: String,
        sglang_version: String,
        max_running_requests: i32,
        data_parallel_size: i32,
        kv_events_endpoint: String,
        kv_events_replay_endpoint: String,
        kv_events_topic: String,
        engine_startup_timeout_secs: Option<f64>,
    ) -> PyResult<Self> {
        let model = SglangModelInfo {
            served_model_name: served_model_name.unwrap_or_else(|| model_path.clone()),
            tokenizer_path: tokenizer_path.unwrap_or_else(|| model_path.clone()),
            model_path,
            is_generation,
            model_type,
            architectures: architectures.unwrap_or_default(),
            max_context_length,
            max_req_input_len,
            vocab_size,
            eos_token_ids: eos_token_ids.unwrap_or_default(),
            pad_token_id,
            bos_token_id,
            weight_version,
            preferred_sampling_params,
            default_sampling_params_json,
            supports_vision,
            id2label_json,
            num_labels,
            server_args_json,
            scheduler_info_json,
            sglang_version,
            max_running_requests,
            data_parallel_size,
            kv_events_endpoint,
            kv_events_replay_endpoint,
            kv_events_topic,
        };
        let config = SglangServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            model,
            engine_startup_timeout: startup_timeout(engine_startup_timeout_secs)?,
        };
        let inner = py
            .detach(|| SglangServicerServer::start(config))
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

    /// Whether the scheduler handshake completed; health is NOT_SERVING
    /// until it has.
    #[getter]
    fn engine_ready(&self) -> bool {
        self.inner.engine_ready()
    }

    /// The last fatal error (scheduler connect or server exit), if any.
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
