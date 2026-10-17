//! WASM Runtime
//!
//! Manages WASM component execution using wasmtime with async support.
//! Provides a thread pool for concurrent WASM execution.

use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use lru::LruCache;
use tokio::sync::oneshot;
use tracing::{debug, error};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Config, Engine, InstanceAllocationStrategy, PoolingAllocationConfig, Store, StoreLimitsBuilder,
};
use wasmtime_wasi::WasiCtx;

/// Epoch increment interval in milliseconds.
/// Epochs are used for cooperative timeout enforcement in WASM execution.
/// A smaller interval gives finer-grained timeout control but slightly more overhead.
const EPOCH_INTERVAL_MS: u64 = 100;

use crate::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmRuntimeError},
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    spec::Smg,
    types::{WasiState, WasmComponentInput, WasmComponentOutput},
};

pub struct WasmRuntime {
    config: WasmRuntimeConfig,
    thread_pool: Arc<WasmThreadPool>,
}

pub struct WasmThreadPool {
    sender: async_channel::Sender<WasmTask>,
    receiver: async_channel::Receiver<WasmTask>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

pub enum WasmTask {
    ExecuteComponent {
        /// SHA256 hash of the WASM bytes, used as the cache key for compiled
        /// components. This avoids hashing the full `Vec<u8>` on every LRU lookup.
        sha256_hash: [u8; 32],
        /// WASM component bytes wrapped in Arc to avoid cloning the full bytes
        /// on every request. Only read on cache miss (first compilation).
        wasm_bytes: Arc<Vec<u8>>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
        response: oneshot::Sender<Result<WasmComponentOutput>>,
    },
}

impl WasmRuntime {
    pub fn new(config: WasmRuntimeConfig) -> Self {
        let thread_pool = Arc::new(WasmThreadPool::new(config.clone()));

        Self {
            config,
            thread_pool,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(WasmRuntimeConfig::default())
    }

    pub fn get_config(&self) -> &WasmRuntimeConfig {
        &self.config
    }

    /// get available cpu count and max recommended cpu count
    pub fn get_cpu_info() -> (usize, usize) {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let max_recommended = cpu_count.max(1);
        (cpu_count, max_recommended)
    }

    /// Execute WASM component using WASM interface based on attach_point
    pub async fn execute_component_async(
        &self,
        sha256_hash: [u8; 32],
        wasm_bytes: Arc<Vec<u8>>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let (response_tx, response_rx) = oneshot::channel();

        let task = WasmTask::ExecuteComponent {
            sha256_hash,
            wasm_bytes,
            attach_point,
            input,
            response: response_tx,
        };

        self.thread_pool.sender.send(task).await.map_err(|e| {
            WasmRuntimeError::CallFailed(format!("Failed to send task to thread pool: {e}"))
        })?;

        response_rx.await.map_err(|e| {
            WasmRuntimeError::CallFailed(format!(
                "Failed to receive response from thread pool: {e}"
            ))
        })?
    }
}

/// Maps a wasmtime error to a WasmError, detecting epoch interruption (timeout) traps.
fn map_wasm_error(e: wasmtime::Error, timeout_ms: u64) -> WasmError {
    // Use proper trap code detection instead of brittle string matching.
    // Wasmtime uses Trap::Interrupt for epoch-based interruptions.
    if e.downcast_ref::<wasmtime::Trap>() == Some(&wasmtime::Trap::Interrupt) {
        WasmError::from(WasmRuntimeError::Timeout(timeout_ms))
    } else {
        WasmError::from(WasmRuntimeError::CallFailed(e.to_string()))
    }
}

impl WasmThreadPool {
    pub fn new(config: WasmRuntimeConfig) -> Self {
        let (sender, receiver) = async_channel::unbounded();

        let mut workers = Vec::new();
        // set thread pool size based on cpu count
        let max_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let num_workers = config.thread_pool_size.clamp(1, max_workers);

        debug!(
            target: "smg::wasm::runtime",
            "Initializing WASM runtime with {} workers",
            num_workers
        );

        for worker_id in 0..num_workers {
            let receiver = receiver.clone();
            let config = config.clone();

            let worker = std::thread::spawn(move || {
                // create independent tokio runtime for this thread
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!(
                            target: "smg::wasm::runtime",
                            worker_id = worker_id,
                            "Failed to create tokio runtime: {}",
                            e
                        );
                        return;
                    }
                };

                rt.block_on(async {
                    Self::worker_loop(worker_id, receiver, config).await;
                });
            });

            workers.push(worker);
        }

        Self {
            sender,
            receiver,
            workers,
        }
    }

    async fn worker_loop(
        worker_id: usize,
        receiver: async_channel::Receiver<WasmTask>,
        config: WasmRuntimeConfig,
    ) {
        debug!(
            target: "smg::wasm::runtime",
            worker_id = worker_id,
            thread_id = ?std::thread::current().id(),
            "Worker started"
        );

        let mut pool_config = PoolingAllocationConfig::default();
        let max_memory_bytes = (config.max_memory_pages as usize) * 65536;

        // Since this thread handles tasks sequentially, we don't need a large pool per thread.
        // A pool size of 20 allows for efficient reuse without hogging memory.
        pool_config.total_core_instances(20);
        pool_config.max_memory_size(max_memory_bytes);
        pool_config.max_component_instance_size(max_memory_bytes);
        pool_config.max_tables_per_component(5);

        let mut wasmtime_config = Config::new();
        wasmtime_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

        wasmtime_config.async_stack_size(config.max_stack_size);
        // wasmtime 45+ enables async via the `async` crate feature; Config::async_support
        // is a deprecated no-op.
        wasmtime_config.wasm_component_model(true);
        wasmtime_config.epoch_interruption(true); // Enable epoch-based timeout interruption

        let engine = match Engine::new(&wasmtime_config) {
            Ok(engine) => engine,
            Err(e) => {
                error!(
                    target: "smg::wasm::runtime",
                    worker_id = worker_id,
                    "Failed to create engine: {}",
                    e
                );
                return;
            }
        };
        let mut linker = Linker::<WasiState>::new(&engine);
        if let Err(e) = wasmtime_wasi::p2::add_to_linker_async(&mut linker) {
            error!(
                target: "smg::wasm::runtime",
                worker_id = worker_id,
                "Failed to add WASI to linker: {}",
                e
            );
            return;
        }

        let default_capacity = NonZeroUsize::new(10).unwrap_or(NonZeroUsize::MIN);
        let cache_capacity =
            NonZeroUsize::new(config.module_cache_size).unwrap_or(default_capacity);
        let mut component_cache: LruCache<[u8; 32], Component> = LruCache::new(cache_capacity);

        // Start epoch incrementer for timeout enforcement.
        // The engine's epoch counter is incremented periodically, and each Store
        // can set a deadline (number of epochs). When the deadline is reached,
        // WASM execution is interrupted with a trap.
        let engine_for_epoch = engine.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "epoch interrupt handler must run as independent background task; abort on drop ensures cleanup"
        )]
        let epoch_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(EPOCH_INTERVAL_MS));
            loop {
                interval.tick().await;
                engine_for_epoch.increment_epoch();
            }
        });

        debug!(
            target: "smg::wasm::runtime",
            worker_id = worker_id,
            epoch_interval_ms = EPOCH_INTERVAL_MS,
            "Epoch incrementer started for timeout enforcement"
        );

        loop {
            let task = match receiver.recv().await {
                Ok(task) => task,
                Err(_) => {
                    debug!(
                        target: "smg::wasm::runtime",
                        worker_id = worker_id,
                        "Worker shutting down"
                    );
                    epoch_handle.abort(); // Stop the epoch incrementer
                    break; // channel closed, exit loop
                }
            };

            match task {
                WasmTask::ExecuteComponent {
                    sha256_hash,
                    wasm_bytes,
                    attach_point,
                    input,
                    response,
                } => {
                    let result = Self::execute_component_in_worker(
                        &engine,
                        &linker,
                        &mut component_cache,
                        sha256_hash,
                        &wasm_bytes,
                        attach_point,
                        input,
                        &config,
                    )
                    .await;

                    let _ = response.send(result);
                }
            }
        }
    }

    #[expect(clippy::too_many_arguments)]
    async fn execute_component_in_worker(
        engine: &Engine,
        linker: &Linker<WasiState>,
        cache: &mut LruCache<[u8; 32], Component>,
        sha256_hash: [u8; 32],
        wasm_bytes: &[u8],
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
        config: &WasmRuntimeConfig,
    ) -> Result<WasmComponentOutput> {
        // Compile component from bytes, or retrieve from cache (keyed by SHA256
        // hash to avoid hashing the full module bytes per lookup).
        let component = if let Some(comp) = cache.get(&sha256_hash) {
            comp.clone()
        } else {
            // Compile new component
            let comp = Component::new(engine, wasm_bytes).map_err(|e| {
                WasmRuntimeError::CompileFailed(format!(
                    "failed to parse WebAssembly component: {e}. \
                     Hint: The WASM file must be in component format. \
                     If you're using wit-bindgen, use 'wasm-tools component new' to wrap the WASM module into a component."
                ))
            })?;

            cache.push(sha256_hash, comp.clone());
            comp
        };

        let mut builder = WasiCtx::builder();

        // Create memory limits from config.
        // Use the config helper to get total bytes, then safely convert to usize.
        let memory_limit_bytes =
            usize::try_from(config.get_total_memory_bytes()).map_err(|_| {
                WasmError::from(WasmRuntimeError::CallFailed(
                    "Configured WASM memory limit exceeds addressable space on this platform."
                        .to_string(),
                ))
            })?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_limit_bytes)
            .trap_on_grow_failure(true)
            .build();

        let mut store = Store::new(
            engine,
            WasiState {
                ctx: builder.build(),
                table: ResourceTable::new(),
                limits,
            },
        );

        // Apply resource limits to the store.
        // This enforces max_memory_pages by preventing memory.grow beyond the limit.
        store.limiter(|state| &mut state.limits);

        // Set epoch deadline for timeout enforcement.
        // The deadline is the number of epoch ticks before execution is interrupted.
        // With EPOCH_INTERVAL_MS=100ms and max_execution_time_ms=1000ms, deadline=10 epochs.
        let deadline_epochs = (config.max_execution_time_ms / EPOCH_INTERVAL_MS).max(1);
        store.set_epoch_deadline(deadline_epochs);

        // When the epoch deadline is reached, trap to enforce the execution timeout.
        store.epoch_deadline_callback(|_store| {
            Err(wasmtime::Error::msg("execution time limit exceeded"))
        });

        let output = match attach_point {
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest) => {
                let request = match input {
                    WasmComponentInput::MiddlewareRequest(req) => req,
                    WasmComponentInput::MiddlewareResponse(_) => {
                        return Err(WasmError::from(WasmRuntimeError::CallFailed(
                            "Expected MiddlewareRequest input for OnRequest attach point"
                                .to_string(),
                        )));
                    }
                };

                // Instantiate component (must use async instantiation when async support is enabled)
                let bindings = Smg::instantiate_async(&mut store, &component, linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-request (async call when async support is enabled)
                let action_result = bindings
                    .smg_gateway_middleware_on_request()
                    .call_on_request(&mut store, &request)
                    .await
                    .map_err(|e| map_wasm_error(e, config.max_execution_time_ms))?;

                WasmComponentOutput::MiddlewareAction(action_result)
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse) => {
                // Extract Response input
                let response = match input {
                    WasmComponentInput::MiddlewareResponse(resp) => resp,
                    WasmComponentInput::MiddlewareRequest(_) => {
                        return Err(WasmError::from(WasmRuntimeError::CallFailed(
                            "Expected MiddlewareResponse input for OnResponse attach point"
                                .to_string(),
                        )));
                    }
                };

                // Instantiate component (must use async instantiation when async support is enabled)
                let bindings = Smg::instantiate_async(&mut store, &component, linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-response (async call when async support is enabled)
                let action_result = bindings
                    .smg_gateway_middleware_on_response()
                    .call_on_response(&mut store, &response)
                    .await
                    .map_err(|e| map_wasm_error(e, config.max_execution_time_ms))?;

                WasmComponentOutput::MiddlewareAction(action_result)
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnError) => {
                return Err(WasmError::from(WasmRuntimeError::CallFailed(
                    "OnError attach point not yet implemented".to_string(),
                )));
            }
        };

        Ok(output)
    }
}

impl Drop for WasmThreadPool {
    fn drop(&mut self) {
        // close sender and receiver
        self.sender.close();
        self.receiver.close();

        // wait for all workers to complete
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, time::Instant};

    use lru::LruCache;

    use super::*;
    use crate::config::WasmRuntimeConfig;

    #[test]
    fn test_get_cpu_info() {
        let (cpu_count, max_recommended) = WasmRuntime::get_cpu_info();
        assert!(cpu_count > 0);
        assert!(max_recommended > 0);
        assert!(max_recommended >= cpu_count);
    }

    #[test]
    fn test_config_default_values() {
        let config = WasmRuntimeConfig::default();

        assert_eq!(config.max_memory_pages, 1024);
        assert_eq!(config.max_execution_time_ms, 1000);
        assert_eq!(config.max_stack_size, 1024 * 1024);
        assert!(config.thread_pool_size > 0);
        assert_eq!(config.module_cache_size, 10);
    }

    #[test]
    fn test_config_clone() {
        let config = WasmRuntimeConfig::default();
        let cloned_config = config.clone();

        assert_eq!(config.max_memory_pages, cloned_config.max_memory_pages);
        assert_eq!(
            config.max_execution_time_ms,
            cloned_config.max_execution_time_ms
        );
        assert_eq!(config.max_stack_size, cloned_config.max_stack_size);
        assert_eq!(config.thread_pool_size, cloned_config.thread_pool_size);
        assert_eq!(config.module_cache_size, cloned_config.module_cache_size);
    }
    #[test]
    fn test_wasm_instantiation_performance_threshold() {
        // A simple WASM module forcing memory allocation
        const WASM_WAT: &str = r#"
            (module
                (memory (export "memory") 1)
                (func (export "run") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add)
            )
        "#;

        let iterations = 1000;

        //  Scenario A: Baseline (No Pool, No Cache)
        let engine_standard = Engine::default();
        let start_standard = Instant::now();
        for _ in 0..iterations {
            // Simulate compilation + instantiation overhead
            let module = wasmtime::Module::new(&engine_standard, WASM_WAT).unwrap();
            let mut store = Store::new(&engine_standard, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let run_func = instance
                .get_typed_func::<(i32, i32), i32>(&mut store, "run")
                .unwrap();
            let _ = run_func.call(&mut store, (10, 20)).unwrap();
        }
        let duration_standard = start_standard.elapsed();

        // --- Scenario B: Optimized (Pool + Cache)
        let mut pool_config = PoolingAllocationConfig::default();

        pool_config.total_core_instances(100);

        let mut config = Config::new();
        config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

        let engine_pooled = Engine::new(&config).unwrap();

        // Setup LRU Cache
        let cache_capacity = NonZeroUsize::new(100).unwrap();
        let mut cache: LruCache<Vec<u8>, wasmtime::Module> = LruCache::new(cache_capacity);

        // Pre-warm cache (simulating the "cached" state)
        let key = WASM_WAT.as_bytes().to_vec();
        let module_compiled = wasmtime::Module::new(&engine_pooled, WASM_WAT).unwrap();
        cache.push(key.clone(), module_compiled);

        let start_pooled = Instant::now();
        for _ in 0..iterations {
            let module = cache.get(&key).unwrap().clone();
            let mut store = Store::new(&engine_pooled, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let run_func = instance
                .get_typed_func::<(i32, i32), i32>(&mut store, "run")
                .unwrap();
            let _ = run_func.call(&mut store, (10, 20)).unwrap();
        }
        let duration_pooled = start_pooled.elapsed();

        // Verify Speedup
        let standard_secs = duration_standard.as_secs_f64();
        let pooled_secs = duration_pooled.as_secs_f64();

        if pooled_secs > 0.0 {
            let speedup = standard_secs / pooled_secs;

            assert!(
                speedup > 5.0,
                "Optimization regression: Pooling+Caching was only {speedup:.2}x faster",
            );
        }
    }
}
