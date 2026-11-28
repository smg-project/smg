//! WASM Runtime Configuration
//!
//! Defines configuration parameters for the WASM runtime,
//! including memory limits, execution timeouts, and thread pool settings.

use serde::{Deserialize, Serialize};

/// The host's cpu count; never 0, and 4 when it cannot be read.
pub(crate) fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WasmRuntimeConfig {
    /// Maximum memory size in pages (64KB per page)
    pub max_memory_pages: u32,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Maximum stack size in bytes
    pub max_stack_size: usize,
    /// Number of worker threads in the pool
    pub thread_pool_size: usize,
    /// Maximum number of modules to cache per worker
    pub module_cache_size: usize,
    /// Maximum HTTP body size in bytes for middleware request/response processing
    pub max_body_size: usize,
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        let default_thread_pool_size = available_cpus().clamp(1, 4);

        Self {
            max_memory_pages: 1024,                     // 64MB
            max_execution_time_ms: 1000,                // 1 seconds
            max_stack_size: 1024 * 1024,                // 1MB
            thread_pool_size: default_thread_pool_size, // based on cpu count and capped
            module_cache_size: 10,                      // Cache up to 10 modules per worker
            max_body_size: 10 * 1024 * 1024,            // 10MB
        }
    }
}

impl WasmRuntimeConfig {
    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        // Validate max_memory_pages
        if self.max_memory_pages == 0 {
            return Err("max_memory_pages cannot be 0".to_string());
        }
        if self.max_memory_pages > 65536 {
            return Err("max_memory_pages cannot exceed 65536 (4GB)".to_string());
        }

        // Validate max_execution_time_ms
        if self.max_execution_time_ms == 0 {
            return Err("max_execution_time_ms cannot be 0".to_string());
        }
        if self.max_execution_time_ms > 300000 {
            return Err("max_execution_time_ms cannot exceed 300000ms (5 minutes)".to_string());
        }

        // Validate max_stack_size
        if self.max_stack_size == 0 {
            return Err("max_stack_size cannot be 0".to_string());
        }
        if self.max_stack_size < 64 * 1024 {
            return Err("max_stack_size must be at least 64KB".to_string());
        }
        if self.max_stack_size > 16 * 1024 * 1024 {
            return Err("max_stack_size cannot exceed 16MB".to_string());
        }

        // Validate thread_pool_size
        if self.thread_pool_size == 0 {
            return Err("thread_pool_size cannot be 0".to_string());
        }
        if self.thread_pool_size > 128 {
            return Err("thread_pool_size cannot exceed 128".to_string());
        }

        // Validate module_cache_size
        if self.module_cache_size == 0 {
            return Err("module_cache_size cannot be 0".to_string());
        }
        if self.module_cache_size > 1000 {
            return Err("module_cache_size cannot exceed 1000".to_string());
        }

        // Validate max_body_size
        if self.max_body_size == 0 {
            return Err("max_body_size cannot be 0".to_string());
        }
        if self.max_body_size > 100 * 1024 * 1024 {
            return Err("max_body_size cannot exceed 100MB".to_string());
        }

        Ok(())
    }

    /// Create a new config with validation
    pub fn new(
        max_memory_pages: u32,
        max_execution_time_ms: u64,
        max_stack_size: usize,
        thread_pool_size: usize,
        module_cache_size: usize,
        max_body_size: usize,
    ) -> Result<Self, String> {
        let config = Self {
            max_memory_pages,
            max_execution_time_ms,
            max_stack_size,
            thread_pool_size,
            module_cache_size,
            max_body_size,
        };
        config.validate()?;
        Ok(config)
    }

    /// Get the total memory size in bytes
    pub fn get_total_memory_bytes(&self) -> u64 {
        self.max_memory_pages as u64 * 64 * 1024 // 64KB per page
    }
}

#[cfg(test)]
#[path = "../tests/unit/config/tests.rs"]
mod tests;
