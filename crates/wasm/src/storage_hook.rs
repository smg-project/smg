//! WASM-based storage hook bridge.
//!
//! Wraps a compiled WASM component that exports the `storage-hook` world
//! into a [`StorageHook`] implementation, allowing WASM guests to intercept
//! storage operations with custom before/after logic.

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
pub struct WasmStorageHook {
    engine: Engine,
    component: Component,
    linker: Linker<StorageHookWasiState>,
}

impl WasmStorageHook {
    /// Compile a WASM component from bytes and prepare it for execution.
    ///
    /// The bytes must be a valid WASM component (not a core module).
    /// Use `wasm-tools component new` to wrap a core module if needed.
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

        Ok(Self {
            engine,
            component,
            linker,
        })
    }

    /// Create a fresh WASM store with default WASI context and memory limits.
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
        store.set_epoch_deadline(1);
        store
    }

    async fn instantiate(
        &self,
    ) -> Result<(Store<StorageHookWasiState>, StorageHookBindings), HookError> {
        let mut store = self.new_store();
        let bindings =
            StorageHookBindings::instantiate_async(&mut store, &self.component, &self.linker)
                .await
                .map_err(|e| HookError::Internal(format!("WASM instantiation failed: {e}")))?;
        Ok((store, bindings))
    }

    /// Spawn epoch ticker to enforce a 5-second execution budget.
    /// The task is aborted immediately after the WASM call completes,
    /// so it is safe for the gateway to shut down without waiting for it.
    #[expect(
        clippy::disallowed_methods,
        reason = "epoch ticker is aborted after WASM call"
    )]
    fn execution_budget(&self) -> tokio::task::JoinHandle<()> {
        let epoch_engine = self.engine.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            epoch_engine.increment_epoch();
        })
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
        let (mut store, bindings) = self.instantiate().await?;

        let wit_op = to_wit_operation(operation);
        let wit_ctx = to_wit_context(context);
        let payload_str = payload.to_string();

        let ticker = self.execution_budget();

        let result = bindings
            .smg_storage_storage_hook_before()
            .call_before(&mut store, wit_op, &wit_ctx, &payload_str)
            .await;

        ticker.abort();

        let result =
            result.map_err(|e| HookError::Internal(format!("WASM before() call failed: {e}")))?;

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
        let (mut store, bindings) = self.instantiate().await?;

        let wit_op = to_wit_operation(operation);
        let wit_ctx = to_wit_context(context);
        let payload_str = payload.to_string();
        let result_str = result.to_string();
        let wit_extra = to_wit_extra_columns(extra);

        let ticker = self.execution_budget();

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
            .await;

        ticker.abort();

        let updated =
            updated.map_err(|e| HookError::Internal(format!("WASM after() call failed: {e}")))?;

        Ok(from_wit_extra_columns(updated))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

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
