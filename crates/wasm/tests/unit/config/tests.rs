use super::*;

#[test]
fn test_default_config_validation() {
    let config = WasmRuntimeConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_new_with_validation() {
    let config = WasmRuntimeConfig::new(1024, 1000, 1024 * 1024, 2, 10, 10 * 1024 * 1024);
    assert!(config.is_ok());
}

#[test]
fn test_validation_max_memory_pages_zero() {
    let config = WasmRuntimeConfig {
        max_memory_pages: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("max_memory_pages cannot be 0"));
}

#[test]
fn test_validation_max_memory_pages_too_large() {
    let config = WasmRuntimeConfig {
        max_memory_pages: 65537, // Exceeds 4GB limit
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_memory_pages cannot exceed 65536"));
}

#[test]
fn test_validation_max_execution_time_zero() {
    let config = WasmRuntimeConfig {
        max_execution_time_ms: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_execution_time_ms cannot be 0"));
}

#[test]
fn test_validation_max_execution_time_too_large() {
    let config = WasmRuntimeConfig {
        max_execution_time_ms: 300001, // Exceeds 5 minutes
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_execution_time_ms cannot exceed 300000ms"));
}

#[test]
fn test_validation_max_stack_size_too_small() {
    let config = WasmRuntimeConfig {
        max_stack_size: 32 * 1024, // Less than 64KB
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_stack_size must be at least 64KB"));
}

#[test]
fn test_validation_max_stack_size_too_large() {
    let config = WasmRuntimeConfig {
        max_stack_size: 17 * 1024 * 1024, // Exceeds 16MB
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_stack_size cannot exceed 16MB"));
}

#[test]
fn test_validation_thread_pool_size_zero() {
    let config = WasmRuntimeConfig {
        thread_pool_size: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("thread_pool_size cannot be 0"));
}

#[test]
fn test_validation_thread_pool_size_too_large() {
    let config = WasmRuntimeConfig {
        thread_pool_size: 129, // Exceeds 128
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("thread_pool_size cannot exceed 128"));
}

#[test]
fn test_validation_module_cache_size_zero() {
    let config = WasmRuntimeConfig {
        module_cache_size: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("module_cache_size cannot be 0"));
}

#[test]
fn test_validation_module_cache_size_too_large() {
    let config = WasmRuntimeConfig {
        module_cache_size: 1001, // Exceeds 1000
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("module_cache_size cannot exceed 1000"));
}

#[test]
fn test_get_total_memory_bytes() {
    let config = WasmRuntimeConfig {
        max_memory_pages: 1024,
        ..Default::default()
    };
    // 1024 pages * 64KB = 64MB
    assert_eq!(config.get_total_memory_bytes(), 64 * 1024 * 1024);
}

#[test]
fn test_validation_max_body_size_zero() {
    let config = WasmRuntimeConfig {
        max_body_size: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("max_body_size cannot be 0"));
}

#[test]
fn test_validation_max_body_size_too_large() {
    let config = WasmRuntimeConfig {
        max_body_size: 101 * 1024 * 1024, // Exceeds 100MB
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("max_body_size cannot exceed 100MB"));
}
