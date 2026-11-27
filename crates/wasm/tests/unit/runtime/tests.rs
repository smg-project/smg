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
