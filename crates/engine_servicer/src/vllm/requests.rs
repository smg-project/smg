//! In-flight request registry: the `Abort` RPC's cancellation handles, keyed
//! by request id.

use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc, Mutex},
};

use tokio::sync::oneshot;
use tonic::Status;

use super::State;

/// Registered in-flight requests: the `Abort` RPC's cancellation handles,
/// keyed by request id. The generation disambiguates an id reused after an
/// abort from the stream that registered it.
pub(super) type Registry = Mutex<HashMap<String, (u64, oneshot::Sender<()>)>>;

/// Keeps a request's registry entry for the stream's lifetime; dropping it
/// removes the entry unless an abort already did (and the id was re-registered
/// since, which the generation tells apart).
pub(super) struct Registration {
    state: Arc<State>,
    request_id: String,
    generation: u64,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(mut registry) = self.state.registry.lock() {
            if registry
                .get(&self.request_id)
                .is_some_and(|(generation, _)| *generation == self.generation)
            {
                registry.remove(&self.request_id);
            }
        }
    }
}

/// Register `request_id` before its engine submit, so an `Abort` landing in
/// the gap finds the entry. A live duplicate is refused: the id is the
/// Router's cancellation handle, and the engine would refuse it too.
pub(super) fn register(
    state: &Arc<State>,
    request_id: &str,
) -> Result<(Registration, oneshot::Receiver<()>), Status> {
    let (cancel_tx, cancel_rx) = oneshot::channel();
    let generation = state.generation.fetch_add(1, Ordering::Relaxed);
    let mut registry = state
        .registry
        .lock()
        .map_err(|_| Status::internal("request registry is poisoned"))?;
    // The id is the Router's cancellation handle, so a live duplicate
    // cannot be admitted: the engine would also refuse it.
    if registry.contains_key(request_id) {
        return Err(Status::already_exists(format!(
            "request {request_id} is already in flight"
        )));
    }
    registry.insert(request_id.to_string(), (generation, cancel_tx));
    drop(registry);
    Ok((
        Registration {
            state: Arc::clone(state),
            request_id: request_id.to_string(),
            generation,
        },
        cancel_rx,
    ))
}

/// Fire the cancellation of every listed request that is still registered.
/// Unknown ids are a no-op: cleanup is idempotent.
pub(super) fn abort(state: &State, request_ids: &[String]) -> Result<(), Status> {
    let mut registry = state
        .registry
        .lock()
        .map_err(|_| Status::internal("request registry is poisoned"))?;
    for request_id in request_ids {
        if let Some((_, cancel)) = registry.remove(request_id) {
            let _ = cancel.send(());
        }
    }
    Ok(())
}
