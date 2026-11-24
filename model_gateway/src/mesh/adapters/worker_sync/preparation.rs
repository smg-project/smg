//! Prepare a peer's tokenizer locally without stalling mesh reconciliation.

use std::{collections::HashMap, sync::Arc, time::Duration};

use smg_mesh::WorkerState;
use smg_worker::{
    identity::basic::Worker,
    registry::{worker_model_ids, WorkerId, WorkerOrigin},
};
use tracing::{info, warn};
use wfaas::WorkflowResult;

use super::{remote_worker_state, WorkerSyncAdapter, PREFIX};
use crate::{
    routers::common::snapshot::WorkerRegistry,
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

#[derive(Default)]
pub(super) struct PendingImports {
    pub(super) by_url: HashMap<String, PendingImport>,
    pub(super) urls_by_key: HashMap<String, String>,
}

impl PendingImports {
    fn take(&mut self, key: &str) -> Option<(String, PendingImport)> {
        let url = self.urls_by_key.remove(key)?;
        self.by_url.remove(&url).map(|pending| (url, pending))
    }

    fn replace(&mut self, pending: PendingImport) -> Option<PendingImport> {
        let previous = self.take(&pending.key_id).map(|(_, pending)| pending);
        let url = &pending.state.url;
        debug_assert!(!self.by_url.contains_key(url));
        self.urls_by_key.insert(pending.key_id.clone(), url.clone());
        self.by_url.insert(url.clone(), pending);
        previous
    }
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
    pub(super) fn cancel_preparation(&self, worker_id: &str) -> Option<String> {
        let (url, pending) = self.preparing.lock().take(worker_id)?;
        if let Some(scope) = pending.scope {
            scope.close();
        }
        info!(worker_id, "Ended mesh tokenizer preparation");
        Some(url)
    }

    fn release_import_tokenizers(&self, worker: &Arc<dyn Worker>) {
        if let Some(app_context) = self
            .app_context
            .upgrade()
            .filter(|app| app.router_config.wait_for_tokenizer)
        {
            for model in WorkerRegistry::policy_model_ids(worker) {
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
        // A pending local REST registration owns its reserved UUID. A peer
        // must not fill it while its local preparation is still running.
        if self.worker_registry.has_pending_reservation(&state.url) {
            return;
        }
        // The first live publisher preparing this endpoint owns its pending
        // import. Another key cannot slip in after that load warms the cache.
        if self
            .preparing
            .lock()
            .by_url
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
            if let Some(pending) = preparing.by_url.get(&state.url) {
                if (same_generation(&pending.state, &state) && pending.scope.is_some())
                    || pending.key_id != worker_id
                {
                    return;
                }
            }
            let permit = Arc::clone(&self.preparation_slots).try_acquire_owned().ok();
            // A slotless entry still owns this endpoint while the CRDT
            // backlog waits for a later reconcile; another key cannot restore
            // its obsolete model at the same address in the meantime.
            let superseded = preparing.replace(PendingImport {
                key_id: worker_id.to_owned(),
                state: state.clone(),
                scope: permit.as_ref().map(|_| Arc::clone(&scope)),
            });
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
                prepare_worker_tokenizers(&app_context, &candidates, &scope, WorkerOrigin::Mesh),
            )
            .await;
            let cancelled = scope.is_cancelled();
            let succeeded = matches!(result, Ok(Ok(())));
            drop(permit);
            adapter.finish_preparation(&worker_id, &state.url, &scope, succeeded || cancelled);
            if !succeeded && !cancelled {
                warn!(worker_id, error = ?result, "Mesh worker tokenizer preparation failed; a later reconcile will retry");
            }
        });
    }
    /// Replay only the completed generation. A deferred replay keeps the key's
    /// claim, with no task attached, until its owner returns or the key goes away.
    fn finish_preparation(
        self: &Arc<Self>,
        key: &str,
        url: &str,
        scope: &Arc<TokenizerPreparation>,
        replay: bool,
    ) {
        let refill = {
            let _intake = self.intake.lock();
            {
                let mut preparing = self.preparing.lock();
                let Some(pending) = preparing.by_url.get_mut(url) else {
                    return;
                };
                if !pending
                    .scope
                    .as_ref()
                    .is_some_and(|active| Arc::ptr_eq(active, scope))
                {
                    return;
                }
                pending.scope = None;
            }
            if !replay {
                return;
            }
            self.sync_key_from_store_inner(&format!("{PREFIX}{key}"), key)
        };
        if let Some(url) = refill {
            self.reimport_endpoint(&url);
        }
    }
}
