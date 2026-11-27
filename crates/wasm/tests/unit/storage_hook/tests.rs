use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use wasmtime::{Instance, Module, Trap, TypedFunc};

use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The hook compiles any component; the epoch tests drive its engine and
/// stores directly.
const EMPTY_COMPONENT: &[u8] = b"(component)";

/// A core module whose `spin` export loops `n` times. Every iteration
/// checks the epoch, so a call traps once its store's deadline has passed.
const SPIN_WAT: &str = r#"
        (module
            (func (export "spin") (param $n i32)
                (loop $again
                    (br_if $again
                        (local.tee $n (i32.sub (local.get $n) (i32.const 1)))))))
    "#;

fn spinner<T: 'static>(
    store: &mut Store<T>,
    module: &Module,
) -> wasmtime::Result<TypedFunc<i32, ()>> {
    Instance::new(&mut *store, module, &[])?.get_typed_func(&mut *store, "spin")
}

#[test]
fn deadline_ticks_counts_the_budget_in_whole_ticks() {
    assert_eq!(deadline_ticks(CALL_BUDGET_MS, EPOCH_INTERVAL_MS), 50);
    assert_eq!(deadline_ticks(5_099, 100), 50);
    // A budget shorter than a tick still gets one tick.
    assert_eq!(deadline_ticks(50, 100), 1);
}

/// Every store of the hook counts its deadline on the one epoch of the
/// hook's engine, so ticks are shared: a tick that lands right after a
/// call's store is created must not use up that call's budget.
#[test]
fn a_store_keeps_its_budget_when_the_epoch_ticks() -> TestResult {
    let hook = WasmStorageHook::new(EMPTY_COMPONENT)?;
    let module = Module::new(&hook.engine, SPIN_WAT)?;

    let mut store = hook.new_store();
    hook.engine.increment_epoch();
    let outcome = spinner(&mut store, &module)?.call(&mut store, 1_000);
    assert!(outcome.is_ok(), "one tick ended the call: {outcome:?}");
    Ok(())
}

#[test]
fn a_store_runs_out_once_its_budget_has_ticked_by() -> TestResult {
    let hook = WasmStorageHook::new(EMPTY_COMPONENT)?;
    let module = Module::new(&hook.engine, SPIN_WAT)?;

    let mut store = hook.new_store();
    for _ in 0..deadline_ticks(CALL_BUDGET_MS, EPOCH_INTERVAL_MS) {
        hook.engine.increment_epoch();
    }
    let trap = spinner(&mut store, &module)?
        .call(&mut store, 1_000)
        .expect_err("a call past its budget must be interrupted");
    assert_eq!(trap.downcast_ref::<Trap>(), Some(&Trap::Interrupt));
    Ok(())
}

#[test]
fn the_hook_ticks_its_engine_until_it_is_dropped() -> TestResult {
    let hook = WasmStorageHook::new(EMPTY_COMPONENT)?;
    let engine = hook.engine.clone();
    let module = Module::new(&engine, SPIN_WAT)?;

    // While the hook lives, its engine's epoch advances with no call
    // driving it: a store one tick from its deadline runs out.
    let mut store = Store::new(&engine, ());
    store.set_epoch_deadline(1);
    let spin = spinner(&mut store, &module)?;
    let started = Instant::now();
    let trap = loop {
        match spin.call(&mut store, 1_000) {
            Err(trap) => break trap,
            Ok(()) => {
                assert!(
                    started.elapsed() < Duration::from_secs(5),
                    "the hook's engine did not tick within 5 s"
                );
                thread::sleep(Duration::from_millis(10));
            }
        }
    };
    assert_eq!(trap.downcast_ref::<Trap>(), Some(&Trap::Interrupt));

    // Once the hook is dropped, the epoch stands still.
    drop(hook);
    let mut store = Store::new(&engine, ());
    store.set_epoch_deadline(1);
    thread::sleep(Duration::from_millis(500));
    let outcome = spinner(&mut store, &module)?.call(&mut store, 1_000);
    assert!(
        outcome.is_ok(),
        "the epoch ticked after the drop: {outcome:?}"
    );
    Ok(())
}

#[test]
fn operation_conversion_round_trips() {
    let ops = [
        StorageOperation::CreateConversation,
        StorageOperation::GetConversation,
        StorageOperation::UpdateConversation,
        StorageOperation::DeleteConversation,
        StorageOperation::CreateItem,
        StorageOperation::LinkItem,
        StorageOperation::LinkItems,
        StorageOperation::ListItems,
        StorageOperation::GetItem,
        StorageOperation::IsItemLinked,
        StorageOperation::DeleteItem,
        StorageOperation::StoreResponse,
        StorageOperation::GetResponse,
        StorageOperation::DeleteResponse,
        StorageOperation::GetResponseChain,
        StorageOperation::ListIdentifierResponses,
        StorageOperation::DeleteIdentifierResponses,
    ];
    // Verify all variants convert without panic
    for op in ops {
        let _ = to_wit_operation(op);
    }
}

#[test]
fn context_conversion_none_returns_empty() {
    let entries = to_wit_context(None);
    assert!(entries.is_empty());
}

#[test]
fn context_conversion_preserves_entries() {
    let mut data = HashMap::new();
    data.insert("tenant_id".to_string(), "abc".to_string());
    data.insert("user".to_string(), "bob".to_string());
    let ctx = RequestContext::with_data(data);
    let entries = to_wit_context(Some(&ctx));
    assert_eq!(entries.len(), 2);
    let map: HashMap<_, _> = entries
        .iter()
        .map(|e| (e.key.as_str(), e.value.as_str()))
        .collect();
    assert_eq!(map["tenant_id"], "abc");
    assert_eq!(map["user"], "bob");
}

#[test]
fn extra_columns_round_trip() {
    let mut extra = ExtraColumns::new();
    extra.insert("col_a".to_string(), Value::String("val_a".to_string()));
    extra.insert("col_b".to_string(), Value::Number(42.into()));

    let wit = to_wit_extra_columns(&extra);
    assert_eq!(wit.len(), 2);

    let back = from_wit_extra_columns(wit);
    assert_eq!(back.len(), 2);
    assert_eq!(back["col_a"], Value::String("val_a".to_string()));
    // Numeric values become their string representation
    assert_eq!(back["col_b"], Value::String("42".to_string()));
}

#[test]
fn invalid_wasm_bytes_returns_error() {
    let result = WasmStorageHook::new(b"not a valid wasm component");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.contains("component compilation failed"), "got: {err}");
}
