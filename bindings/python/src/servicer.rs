//! Python lifecycle binding for the Rust vLLM gRPC servicer.
//!
//! Rust owns the listener, the engine link, and the request path; Python
//! launches the headless engine, starts and stops this server, and announces
//! draining. Nothing request-sensitive crosses into Python.

use std::time::Duration;

use engine_servicer::{ServicerError, VllmModelInfo, VllmServicerConfig, VllmServicerServer};
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
};

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
    ) -> PyResult<Self> {
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
        };
        let config = VllmServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            model,
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
