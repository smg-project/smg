//! WASM Module Manager

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use uuid::Uuid;

use crate::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmManagerError, WasmModuleError},
    module::{WasmModule, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    spec::smg::gateway::middleware_types::Action as MiddlewareAction,
    types::{WasmComponentInput, WasmComponentOutput},
};

/// A registered module beside the two counters an execution bumps. The
/// counters are atomics, so an execution touches them under the table's read
/// lock it already holds for the hash and the bytes; the table's write lock
/// belongs to registration and removal alone.
struct ModuleEntry {
    module: WasmModule,
    last_accessed_at: AtomicU64,
    access_count: AtomicU64,
}

impl ModuleEntry {
    fn new(module: WasmModule) -> Self {
        let last_accessed_at = AtomicU64::new(module.module_meta.last_accessed_at);
        let access_count = AtomicU64::new(module.module_meta.access_count);
        Self {
            module,
            last_accessed_at,
            access_count,
        }
    }

    /// The module as its readers see it: the record with the counters'
    /// current values on its metadata.
    fn snapshot(&self) -> WasmModule {
        let mut module = self.module.clone();
        module.module_meta.last_accessed_at = self.last_accessed_at.load(Ordering::Relaxed);
        module.module_meta.access_count = self.access_count.load(Ordering::Relaxed);
        module
    }

    /// One execution at `now` (nanoseconds since the epoch). The stamp only
    /// moves forward: two executions that sampled the clock in one order may
    /// store in the other, and the later stamp stays.
    fn touch(&self, now: u64) {
        self.last_accessed_at.fetch_max(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// The clock an execution stamps its module with: nanoseconds since the
/// epoch, zero when the clock reads before it.
fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

/// A poisoned lock of the module table, as the manager's error.
fn lock_failed(poisoned: impl ToString) -> WasmError {
    WasmManagerError::LockFailed(poisoned.to_string()).into()
}

pub struct WasmModuleManager {
    modules: Arc<RwLock<HashMap<Uuid, ModuleEntry>>>,
    runtime: Arc<WasmRuntime>,
    // Metrics
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
    failed_executions: AtomicU64,
    total_execution_time_ms: AtomicU64,
    max_execution_time_ms: AtomicU64,
}

impl WasmModuleManager {
    pub fn new(config: WasmRuntimeConfig) -> Self {
        let runtime = Arc::new(WasmRuntime::new(config));
        Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            runtime,
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            max_execution_time_ms: AtomicU64::new(0),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(WasmRuntimeConfig::default())
    }

    /// Register a module (for workflow steps)
    pub fn register_module_internal(&self, module: WasmModule) -> Result<()> {
        let mut modules = self.modules.write().map_err(lock_failed)?;
        modules.insert(module.module_uuid, ModuleEntry::new(module));
        Ok(())
    }

    /// Remove a module (for workflow steps)
    pub fn remove_module_internal(&self, module_uuid: Uuid) -> Result<()> {
        let mut modules = self.modules.write().map_err(lock_failed)?;
        if !modules.contains_key(&module_uuid) {
            return Err(WasmManagerError::ModuleNotFound(module_uuid).into());
        }
        modules.remove(&module_uuid);
        Ok(())
    }

    pub fn check_duplicate_sha256_hash(&self, sha256_hash: &[u8; 32]) -> Result<()> {
        let modules = self.modules.read().map_err(lock_failed)?;
        if modules
            .values()
            .any(|entry| entry.module.module_meta.sha256_hash == *sha256_hash)
        {
            return Err(WasmModuleError::DuplicateSha256((*sha256_hash).into()).into());
        }
        Ok(())
    }

    pub fn get_module(&self, module_uuid: Uuid) -> Result<Option<WasmModule>> {
        let modules = self.modules.read().map_err(lock_failed)?;
        Ok(modules.get(&module_uuid).map(ModuleEntry::snapshot))
    }

    pub fn get_modules(&self) -> Result<Vec<WasmModule>> {
        let modules = self.modules.read().map_err(lock_failed)?;
        Ok(modules.values().map(ModuleEntry::snapshot).collect())
    }

    /// get modules by attach point
    pub fn get_modules_by_attach_point(
        &self,
        attach_point: WasmModuleAttachPoint,
    ) -> Result<Vec<WasmModule>> {
        let modules = self.modules.read().map_err(lock_failed)?;
        Ok(modules
            .values()
            .filter(|entry| {
                entry
                    .module
                    .module_meta
                    .attach_points
                    .contains(&attach_point)
            })
            .map(ModuleEntry::snapshot)
            .collect())
    }

    /// Get the configured maximum body size for HTTP request/response processing
    pub fn get_max_body_size(&self) -> usize {
        self.runtime.get_config().max_body_size
    }

    /// Execute WASM module using WebAssembly component model based on attach_point
    pub async fn execute_module_interface(
        &self,
        module_uuid: Uuid,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let start_time = Instant::now();

        // The SHA256 hash, the Arc-wrapped WASM bytes and the access counters'
        // bump, all under one read lock: an execution excludes no reader.
        let (sha256_hash, wasm_bytes) = {
            let modules = self.modules.read().map_err(lock_failed)?;
            let entry = modules
                .get(&module_uuid)
                .ok_or_else(|| WasmError::from(WasmManagerError::ModuleNotFound(module_uuid)))?;
            entry.touch(now_nanos());

            (
                entry.module.module_meta.sha256_hash,
                entry.module.module_meta.wasm_bytes.clone(),
            )
        };

        let result = self
            .runtime
            .execute_component_async(sha256_hash, wasm_bytes, attach_point, input)
            .await;

        self.record_execution(start_time.elapsed().as_millis() as u64, result.is_ok());

        result
    }

    /// Count one execution that took `elapsed_ms` and ended `ok` or not:
    /// the totals, the running maximum, and the outcome's counter.
    fn record_execution(&self, elapsed_ms: u64, ok: bool) {
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(elapsed_ms, Ordering::Relaxed);
        self.max_execution_time_ms
            .fetch_max(elapsed_ms, Ordering::Relaxed);
        if ok {
            self.successful_executions.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_executions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.total_executions.load(Ordering::Relaxed),
            self.successful_executions.load(Ordering::Relaxed),
            self.failed_executions.load(Ordering::Relaxed),
            self.total_execution_time_ms.load(Ordering::Relaxed),
            self.max_execution_time_ms.load(Ordering::Relaxed),
        )
    }

    /// Execute a WASM module for a given attach point
    /// Returns the Action if successful, or None if execution failed
    pub async fn execute_module_for_attach_point(
        &self,
        module: &WasmModule,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Option<MiddlewareAction> {
        use tracing::error;

        let action_result = self
            .execute_module_interface(module.module_uuid, attach_point, input)
            .await;

        match action_result {
            Ok(output) => match output {
                WasmComponentOutput::MiddlewareAction(action) => Some(action),
            },
            Err(e) => {
                error!(
                    "Failed to execute WASM module {}: {}",
                    module.module_meta.name, e
                );
                None
            }
        }
    }
}

impl Default for WasmModuleManager {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
#[path = "../tests/unit/module_manager/tests.rs"]
mod tests;
