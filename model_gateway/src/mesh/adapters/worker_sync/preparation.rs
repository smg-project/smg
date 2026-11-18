//! Prepare a peer's tokenizer locally without stalling mesh reconciliation.

use std::{sync::Arc, time::Duration};

use smg_mesh::WorkerState;
use tracing::{info, warn};
use wfaas::WorkflowResult;

use super::{remote_worker_state, WorkerSyncAdapter, PREFIX};
use crate::{
    worker::{
        registry::{worker_model_ids, WorkerId},
        Worker,
    },
    workflow::{
        steps::local::ReleaseTokenizerStep,
        tokenizer_preparation::{prepare_worker_tokenizers, TokenizerPreparation},
    },
};

pub(super) struct PendingImport {
    pub(super) key_id: String,
    state: WorkerState,
    scope: Option<Arc<TokenizerPreparation>>,
}

fn same_generation(left: &WorkerState, right: &WorkerState) -> bool {
    left.worker_id == right.worker_id
        && left.owner == right.owner
        && left.version == right.version
        && left.url == right.url
        && left.model_id == right.model_id
        && left.spec == right.spec
}

impl WorkerSyncAdapter {
    pub(super) fn cancel_preparation(&self, worker_id: &str) {
        let pending = {
            let mut preparing = self.preparing.lock();
            let url = preparing
                .iter()
                .find(|(_, pending)| pending.key_id == worker_id)
                .map(|(url, _)| url.clone());
            url.and_then(|url| preparing.remove(&url))
        };
        if let Some(pending) = pending {
            if let Some(scope) = pending.scope {
                scope.close();
            }
            info!(worker_id, "Ended mesh tokenizer preparation");
        }
    }

    fn release_import_tokenizers(&self, worker: &Arc<dyn Worker>) {
        if let Some(app_context) = self
            .app_context
            .upgrade()
            .filter(|app| app.router_config.wait_for_tokenizer)
        {
            for model in worker_model_ids(worker) {
                ReleaseTokenizerStep::release_if_unserved(&app_context, &model);
            }
        }
    }

    pub(super) fn remove_import(&self, id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let worker = self.worker_registry.remove_remote(id)?;
        self.release_import_tokenizers(&worker);
        Some(worker)
    }

    pub(super) fn import_prepared_state(self: &Arc<Self>, worker_id: &str, state: WorkerState) {
        let Some(app_context) = self.app_context.upgrade() else {
            self.worker_registry
                .on_remote_worker_state(&remote_worker_state(state));
            return;
        };
        if !app_context.router_config.wait_for_tokenizer
            || app_context.router_config.disable_tokenizer_autoload
        {
            self.worker_registry
                .on_remote_worker_state(&remote_worker_state(state));
            return;
        }
        // Legacy publications may omit the ID in their value; the CRDT key
        // remains authoritative for identity, cancellation and reconciliation.
        let mut state = state;
        if state.worker_id.is_empty() {
            state.worker_id = worker_id.to_owned();
        }
        // A pending local REST registration owns its reserved UUID. A peer
        // must not fill it while its local preparation is still running.
        if self
            .worker_registry
            .get_id_by_url_or_reservation(&state.url)
            .is_some_and(|id| self.worker_registry.get(&id).is_none())
        {
            return;
        }
        // The first live publisher preparing this endpoint owns its pending
        // import. Another key cannot slip in after that load warms the cache.
        if self
            .preparing
            .lock()
            .get(&state.url)
            .is_some_and(|pending| pending.key_id != worker_id)
        {
            return;
        }
        let scope = TokenizerPreparation::new(&app_context);
        let scope_guard = scope.guard();
        let previous = self.worker_registry.get_by_url(&state.url);
        let blocked: WorkflowResult<_> = scope.with_pins(|models| {
            Ok(self.worker_registry.on_remote_worker_state_if(
                &remote_worker_state(state.clone()),
                |candidate| {
                    if !candidate.connection_mode().uses_grpc_pipeline() {
                        return true;
                    }
                    let required = worker_model_ids(candidate);
                    models.extend(required.iter().cloned());
                    required
                        .iter()
                        .all(|model| app_context.tokenizer_registry.contains(model))
                },
            ))
        });
        let candidate = match blocked {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                self.cancel_preparation(worker_id);
                if let Some(previous) = previous {
                    self.release_import_tokenizers(&previous);
                }
                return;
            }
            Err(error) => {
                warn!(worker_id, %error, "Mesh worker preparation ended before import");
                return;
            }
        };
        // A newer owner publication can describe a different model at the
        // same address. The previous import must stop routing while it loads.
        if let Some(previous) = previous {
            if let Some(id) = self.worker_registry.get_id_by_url(previous.url()) {
                if let Some(removed) = self
                    .worker_registry
                    .remove_remote_if_current(&id, &previous)
                {
                    self.release_import_tokenizers(&removed);
                }
            }
        }
        let (superseded, permit) = {
            let mut preparing = self.preparing.lock();
            if let Some(pending) = preparing.get(&state.url) {
                if (same_generation(&pending.state, &state) && pending.scope.is_some())
                    || pending.key_id != worker_id
                {
                    return;
                }
            }
            let superseded = preparing.remove(&state.url);
            let permit = Arc::clone(&self.preparation_slots).try_acquire_owned().ok();
            // A slotless entry still owns this endpoint while the CRDT
            // backlog waits for a later reconcile; another key cannot restore
            // its obsolete model at the same address in the meantime.
            preparing.insert(
                state.url.clone(),
                PendingImport {
                    key_id: worker_id.to_owned(),
                    state: state.clone(),
                    scope: permit.as_ref().map(|_| Arc::clone(&scope)),
                },
            );
            (superseded, permit)
        };
        if let Some(scope) = superseded.and_then(|pending| pending.scope) {
            scope.close();
        }
        let Some(permit) = permit else {
            // The CRDT namespace is the backlog. The reconcile pass retries
            // keys not admitted to this bounded pool; no task waits for a slot.
            return;
        };
        let adapter = Arc::clone(self);
        let worker_id = worker_id.to_owned();
        #[expect(
            clippy::disallowed_methods,
            reason = "one task per owned permit, bounded by a five-minute preparation deadline; its guard closes the scope on every exit"
        )]
        tokio::spawn(async move {
            let _scope_guard = scope_guard;
            let candidates = [candidate];
            let result = tokio::time::timeout(
                Duration::from_secs(300),
                prepare_worker_tokenizers(&app_context, &candidates, &scope),
            )
            .await;
            let cancelled = scope.is_cancelled();
            let succeeded = matches!(result, Ok(Ok(())));
            drop(permit);
            {
                let _intake = adapter.intake.lock();
                let retry = adapter
                    .preparing
                    .lock()
                    .get(&state.url)
                    .is_none_or(|pending| {
                        pending
                            .scope
                            .as_ref()
                            .is_some_and(|pending_scope| Arc::ptr_eq(pending_scope, &scope))
                    });
                if retry && (succeeded || cancelled) {
                    // Read and apply current store truth under the same lock
                    // as tombstones, replacement events and warm imports.
                    adapter.sync_key_from_store_inner(&format!("{PREFIX}{worker_id}"), &worker_id);
                }
                let mut preparing = adapter.preparing.lock();
                if preparing.get(&state.url).is_some_and(|pending| {
                    pending
                        .scope
                        .as_ref()
                        .is_some_and(|pending_scope| Arc::ptr_eq(pending_scope, &scope))
                }) {
                    if succeeded || cancelled {
                        preparing.remove(&state.url);
                    } else if let Some(pending) = preparing.get_mut(&state.url) {
                        pending.scope = None;
                    }
                }
            }
            if !succeeded && !cancelled {
                warn!(worker_id, error = ?result, "Mesh worker tokenizer preparation failed; a later reconcile will retry");
            }
        });
    }
}
