//! Python lifecycle binding for the node-local SMG Worker server.
//!
//! Rust owns the transport, the health state, and the engine link; Python only
//! announces coarse lifecycle transitions.

use std::{collections::HashMap, time::Duration};

use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
};
use smg::worker_node::{parse_health_state, WorkerNodeConfig, WorkerNodeError, WorkerNodeServer};

fn to_py_err(error: WorkerNodeError) -> PyErr {
    match error {
        WorkerNodeError::InvalidConfig(_) => PyValueError::new_err(error.to_string()),
        WorkerNodeError::StopTimeout => PyTimeoutError::new_err(error.to_string()),
        WorkerNodeError::Startup(_)
        | WorkerNodeError::Poisoned
        | WorkerNodeError::ThreadPanicked => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Install the Rust tracing subscriber for a process that only hosts a Worker.
#[pyfunction]
#[pyo3(signature = (level = None))]
pub fn init_tracing(level: Option<&str>) -> PyResult<()> {
    smg::worker_node::init_tracing(level).map_err(to_py_err)
}

/// Rust-owned WorkerControl server with lifecycle driven from Python.
#[pyclass(name = "WorkerControlServer")]
pub struct PyWorkerControlServer {
    inner: WorkerNodeServer,
}

#[pymethods]
impl PyWorkerControlServer {
    #[new]
    #[pyo3(signature = (
        bind_address,
        worker_id,
        engine_type,
        model_ids,
        engine_endpoint,
        instance_id = None,
        hostname = None,
        zone = String::new(),
        engine_version = String::new(),
        features = None,
        max_concurrent_requests = 0,
        inference_enabled = false,
        engine_attributes = None,
        engine_transport = String::from("grpc"),
        zmq_handshake_address = None,
        engine_count = 1,
    ))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bind_address: String,
        worker_id: String,
        engine_type: String,
        model_ids: Vec<String>,
        engine_endpoint: String,
        instance_id: Option<String>,
        hostname: Option<String>,
        zone: String,
        engine_version: String,
        features: Option<Vec<String>>,
        max_concurrent_requests: u32,
        inference_enabled: bool,
        engine_attributes: Option<HashMap<String, String>>,
        engine_transport: String,
        zmq_handshake_address: Option<String>,
        engine_count: usize,
    ) -> PyResult<Self> {
        let config = WorkerNodeConfig {
            bind_address,
            worker_id,
            instance_id,
            hostname,
            zone,
            engine_type,
            engine_version,
            engine_endpoint,
            model_ids,
            features,
            max_concurrent_requests,
            inference_enabled,
            engine_attributes: engine_attributes.unwrap_or_default(),
            engine_transport,
            zmq_handshake_address,
            engine_count,
        };
        let inner = py
            .detach(|| WorkerNodeServer::start(config))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Whether the engine transport has connected; health reports STARTING
    /// until it has, whatever lifecycle Python announced.
    #[getter]
    fn engine_ready(&self) -> bool {
        self.inner.engine_ready()
    }

    #[getter]
    fn address(&self) -> &str {
        self.inner.address()
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    #[getter]
    fn last_error(&self) -> PyResult<Option<String>> {
        self.inner.last_error().map_err(to_py_err)
    }

    #[pyo3(signature = (state, message = String::new()))]
    fn set_health(&self, state: &str, message: String) -> PyResult<()> {
        let state = parse_health_state(state).map_err(to_py_err)?;
        self.inner.set_health(state, message).map_err(to_py_err)
    }

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
