//! WASM-based storage hook bridge.
//!
//! Wraps a compiled WASM component that exports the `storage-hook` world
//! into a [`StorageHook`] implementation, allowing WASM guests to intercept
//! storage operations with custom before/after logic.

use std::{
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::Duration,
};

use async_trait::async_trait;
use serde_json::Value;
use smg_data_connector::{
    context::RequestContext,
    hooks::{BeforeHookResult, ExtraColumns, HookError, StorageHook, StorageOperation},
};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Config, Engine, Store, StoreLimitsBuilder,
};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

use crate::storage_spec::{
    smg::storage::storage_hook_types::{
        BeforeResult as WitBeforeResult, ContextEntry, ExtraColumn, Operation as WitOperation,
    },
    StorageHook as StorageHookBindings,
};

/// How often the hook's engine epoch advances: the unit a call's execution
/// budget is counted in.
const EPOCH_INTERVAL_MS: u64 = 100;

/// Execution budget of one `before()`/`after()` call, counted from the moment
/// its store is created, so instantiation is included.
const CALL_BUDGET_MS: u64 = 5_000;

/// WASI state for storage hook WASM execution.
struct StorageHookWasiState {
    ctx: WasiCtx,
    table: ResourceTable,
    limits: StoreLimits,
}

impl WasiView for StorageHookWasiState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

type StoreLimits = wasmtime::StoreLimits;

/// A [`StorageHook`] backed by a WASM component.
///
/// Compiles the WASM component once at construction. Each `before()`/`after()`
/// call instantiates the component in a fresh store, calls the export, and
/// converts the result back to Rust types.
///
/// Every call has the same execution budget of `CALL_BUDGET_MS` (5 s, to
/// within one epoch tick), measured from its own start: neither a slow call nor
/// the number of calls in flight shortens it. A call still running when its
/// budget is spent traps with an interrupt and fails with
/// [`HookError::Internal`].
pub struct WasmStorageHook {
    engine: Engine,
    component: Component,
    linker: Linker<StorageHookWasiState>,
    _epoch_ticker: EpochTicker,
}

impl WasmStorageHook {
    /// Compile a WASM component from bytes and prepare it for execution.
    ///
    /// The bytes must be a valid WASM component (not a core module).
    /// Use `wasm-tools component new` to wrap a core module if needed.
    ///
    /// Also starts the thread that advances the engine's epoch every
    /// `EPOCH_INTERVAL_MS` for as long as the hook lives; dropping the hook
    /// stops it.
    pub fn new(wasm_bytes: &[u8]) -> Result<Self, String> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        config.epoch_interruption(true);

        let engine = Engine::new(&config).map_err(|e| format!("engine creation failed: {e}"))?;
        let component = Component::new(&engine, wasm_bytes)
            .map_err(|e| format!("component compilation failed: {e}"))?;

        let mut linker = Linker::<StorageHookWasiState>::new(&engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| format!("WASI linker setup failed: {e}"))?;

        let epoch_ticker = EpochTicker::start(engine.clone())
            .map_err(|e| format!("epoch ticker spawn failed: {e}"))?;

        Ok(Self {
            engine,
            component,
            linker,
            _epoch_ticker: epoch_ticker,
        })
    }

    /// Create a fresh WASM store with default WASI context, memory limits and
    /// the execution budget of one call.
    fn new_store(&self) -> Store<StorageHookWasiState> {
        let mut builder = WasiCtx::builder();
        let limits = StoreLimitsBuilder::new()
            .memory_size(10 * 1024 * 1024) // 10 MB
            .trap_on_grow_failure(true)
            .build();

        let mut store = Store::new(
            &self.engine,
            StorageHookWasiState {
                ctx: builder.build(),
                table: ResourceTable::new(),
                limits,
            },
        );
        store.limiter(|state| &mut state.limits);
        store.set_epoch_deadline(deadline_ticks(CALL_BUDGET_MS, EPOCH_INTERVAL_MS));
        store
    }
}

/// The epoch deadline, in ticks of `interval_ms`, that gives a store
/// `budget_ms` of execution time.
///
/// A deadline counts from the engine's epoch at the moment it is set, and all
/// the hook's stores share that epoch; counting the whole budget in ticks of a
/// steady clock is what keeps each call's budget its own. Never less than one
/// tick: a deadline of zero has already passed.
fn deadline_ticks(budget_ms: u64, interval_ms: u64) -> u64 {
    (budget_ms / interval_ms).max(1)
}

/// Advances an engine's epoch every `EPOCH_INTERVAL_MS` until dropped.
///
/// A plain thread rather than a task: hook calls run guest code on the
/// caller's async workers, and a guest that never yields holds its worker
/// until the interrupt fires, so the clock behind that interrupt must not
/// need a worker itself.
struct EpochTicker {
    stop: mpsc::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
}

impl EpochTicker {
    fn start(engine: Engine) -> std::io::Result<Self> {
        let (stop, stopped) = mpsc::channel();
        let interval = Duration::from_millis(EPOCH_INTERVAL_MS);
        let thread = thread::Builder::new()
            .name("smg-hook-epoch".to_string())
            .spawn(move || {
                // Times out once per interval; anything else is the stop.
                while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(interval) {
                    engine.increment_epoch();
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for EpochTicker {
    fn drop(&mut self) {
        // The thread only ever waits in `recv_timeout`, which returns as soon
        // as the stop arrives, so the join is immediate and no tick lands
        // after the hook is gone.
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// ── Type conversions ─────────────────────────────────────────────────────

fn to_wit_operation(op: StorageOperation) -> WitOperation {
    match op {
        StorageOperation::CreateConversation => WitOperation::CreateConversation,
        StorageOperation::GetConversation => WitOperation::GetConversation,
        StorageOperation::UpdateConversation => WitOperation::UpdateConversation,
        StorageOperation::DeleteConversation => WitOperation::DeleteConversation,
        StorageOperation::CreateItem => WitOperation::CreateItem,
        StorageOperation::LinkItem => WitOperation::LinkItem,
        StorageOperation::LinkItems => WitOperation::LinkItems,
        StorageOperation::ListItems => WitOperation::ListItems,
        StorageOperation::GetItem => WitOperation::GetItem,
        StorageOperation::IsItemLinked => WitOperation::IsItemLinked,
        StorageOperation::DeleteItem => WitOperation::DeleteItem,
        StorageOperation::StoreResponse => WitOperation::StoreResponse,
        StorageOperation::GetResponse => WitOperation::GetResponse,
        StorageOperation::DeleteResponse => WitOperation::DeleteResponse,
        StorageOperation::GetResponseChain => WitOperation::GetResponseChain,
        StorageOperation::ListIdentifierResponses => WitOperation::ListIdentifierResponses,
        StorageOperation::DeleteIdentifierResponses => WitOperation::DeleteIdentifierResponses,
    }
}

fn to_wit_context(ctx: Option<&RequestContext>) -> Vec<ContextEntry> {
    match ctx {
        Some(rc) => rc
            .data()
            .iter()
            .map(|(k, v)| ContextEntry {
                key: k.clone(),
                value: v.clone(),
            })
            .collect(),
        None => Vec::new(),
    }
}

fn to_wit_extra_columns(extra: &ExtraColumns) -> Vec<ExtraColumn> {
    extra
        .iter()
        .map(|(name, value)| ExtraColumn {
            name: name.clone(),
            value: value_to_string(value),
        })
        .collect()
}

fn from_wit_extra_columns(cols: Vec<ExtraColumn>) -> ExtraColumns {
    cols.into_iter()
        .map(|ec| (ec.name, Value::String(ec.value)))
        .collect()
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ── StorageHook implementation ───────────────────────────────────────────

#[async_trait]
impl StorageHook for WasmStorageHook {
    async fn before(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
    ) -> Result<BeforeHookResult, HookError> {
        // The store's epoch deadline is this call's execution budget.
        let mut store = self.new_store();

        let bindings =
            StorageHookBindings::instantiate_async(&mut store, &self.component, &self.linker)
                .await
                .map_err(|e| HookError::Internal(format!("WASM instantiation failed: {e}")))?;

        let wit_op = to_wit_operation(operation);
        let wit_ctx = to_wit_context(context);
        let payload_str = payload.to_string();

        let result = bindings
            .smg_storage_storage_hook_before()
            .call_before(&mut store, wit_op, &wit_ctx, &payload_str)
            .await
            .map_err(|e| HookError::Internal(format!("WASM before() call failed: {e}")))?;

        match result {
            WitBeforeResult::DoContinue(extra_cols) => Ok(BeforeHookResult::Continue(
                from_wit_extra_columns(extra_cols),
            )),
            WitBeforeResult::Reject(reason) => Ok(BeforeHookResult::Reject(reason)),
        }
    }

    async fn after(
        &self,
        operation: StorageOperation,
        context: Option<&RequestContext>,
        payload: &Value,
        result: &Value,
        extra: &ExtraColumns,
    ) -> Result<ExtraColumns, HookError> {
        // The store's epoch deadline is this call's execution budget.
        let mut store = self.new_store();

        let bindings =
            StorageHookBindings::instantiate_async(&mut store, &self.component, &self.linker)
                .await
                .map_err(|e| HookError::Internal(format!("WASM instantiation failed: {e}")))?;

        let wit_op = to_wit_operation(operation);
        let wit_ctx = to_wit_context(context);
        let payload_str = payload.to_string();
        let result_str = result.to_string();
        let wit_extra = to_wit_extra_columns(extra);

        let updated = bindings
            .smg_storage_storage_hook_after()
            .call_after(
                &mut store,
                wit_op,
                &wit_ctx,
                &payload_str,
                &result_str,
                &wit_extra,
            )
            .await
            .map_err(|e| HookError::Internal(format!("WASM after() call failed: {e}")))?;

        Ok(from_wit_extra_columns(updated))
    }
}

#[cfg(test)]
mod tests {
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
}
