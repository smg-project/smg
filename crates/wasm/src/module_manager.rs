//! WASM Module Manager

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
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

pub struct WasmModuleManager {
    modules: Arc<RwLock<HashMap<Uuid, WasmModule>>>,
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
        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        modules.insert(module.module_uuid, module);
        Ok(())
    }

    /// Remove a module (for workflow steps)
    pub fn remove_module_internal(&self, module_uuid: Uuid) -> Result<()> {
        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        if !modules.contains_key(&module_uuid) {
            return Err(WasmManagerError::ModuleNotFound(module_uuid).into());
        }
        modules.remove(&module_uuid);
        Ok(())
    }

    pub fn check_duplicate_sha256_hash(&self, sha256_hash: &[u8; 32]) -> Result<()> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        if modules
            .values()
            .any(|module: &WasmModule| module.module_meta.sha256_hash == *sha256_hash)
        {
            return Err(WasmModuleError::DuplicateSha256((*sha256_hash).into()).into());
        }
        Ok(())
    }

    pub fn get_module(&self, module_uuid: Uuid) -> Result<Option<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules.get(&module_uuid).cloned())
    }

    pub fn get_modules(&self) -> Result<Vec<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules.values().cloned().collect())
    }

    /// get modules by attach point
    pub fn get_modules_by_attach_point(
        &self,
        attach_point: WasmModuleAttachPoint,
    ) -> Result<Vec<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules
            .values()
            .filter(|module| module.module_meta.attach_points.contains(&attach_point))
            .cloned()
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
        let start_time = std::time::Instant::now();

        // Get the SHA256 hash and Arc-wrapped WASM bytes under a read lock.
        let (sha256_hash, wasm_bytes) = {
            let modules = self
                .modules
                .read()
                .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
            let module = modules
                .get(&module_uuid)
                .ok_or_else(|| WasmError::from(WasmManagerError::ModuleNotFound(module_uuid)))?;

            (
                module.module_meta.sha256_hash,
                module.module_meta.wasm_bytes.clone(),
            )
        };

        {
            let mut modules = self
                .modules
                .write()
                .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
            if let Some(module) = modules.get_mut(&module_uuid) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(std::time::Duration::ZERO)
                    .as_nanos() as u64;
                module.module_meta.last_accessed_at = now;
                module.module_meta.access_count += 1;
            }
        }

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
mod tests {
    use super::*;

    /// The five counters the route reports, after one success and one
    /// failure: the totals, the running maximum, and the outcome split.
    #[test]
    fn the_execution_counters_add_up_over_a_success_and_a_failure() {
        let manager = WasmModuleManager::with_default_config();
        assert_eq!(manager.get_metrics(), (0, 0, 0, 0, 0));
        manager.record_execution(7, true);
        manager.record_execution(3, false);
        assert_eq!(manager.get_metrics(), (2, 1, 1, 10, 7));
    }
}
