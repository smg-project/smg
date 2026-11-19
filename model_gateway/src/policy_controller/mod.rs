//! The policy controller in this process.
//!
//! A request's worker has two possible deciders: the classic policies, and
//! the policy controller's engine over the controller's store. One temporary
//! setting picks one of them for the whole process
//! ([`RouterConfig::policy_controller`]). The classic policies are the
//! default.
//!
//! At this step the setting starts the store and hands it the policy
//! document the routing configuration means. Nothing feeds the store yet and
//! no code of the controller decides, so one router is mounted in the place
//! of every classic router and refuses each request by name: the classic
//! policies never decide for a process the controller was picked for. With
//! the setting off nothing here is constructed, and no request reads this
//! module.

use std::sync::Arc;

use smg_policy_controller::{
    store::{MemoryStore, StaleGeneration},
    SpecError,
};

use crate::config::{PolicyDocumentError, RouterConfig};

mod undecided;

pub(crate) use undecided::Undecided;

/// Why the controller's store did not start.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    /// The routing configuration holds a setting no policy document says.
    #[error(transparent)]
    Document(#[from] PolicyDocumentError),
    /// The gateway's check of a document it hosts refuses this one: this
    /// build does not run a part of it, or its view reads a header the
    /// gateway keeps to itself.
    #[error(transparent)]
    Spec(#[from] SpecError),
    /// The store holds a generation that is not below the document's.
    #[error(transparent)]
    Stale(#[from] StaleGeneration),
}

/// The controller's store for this process, holding the policy document the
/// routing configuration means as its one generation. `None` where the
/// classic policies decide: nothing is constructed then.
///
/// A configuration the document cannot say, and a document the gateway's
/// check of a hosted document refuses ([`RouterConfig::hosted_policy`]),
/// stop the start by name. Neither is served by the classic policies in
/// silence.
pub fn start(config: &RouterConfig) -> Result<Option<Arc<MemoryStore>>, StartError> {
    if !config.policy_controller {
        return Ok(None);
    }
    let document = config.policy_document()?;
    let generation = config.hosted_policy(&document)?;
    let store = MemoryStore::new();
    store.apply(generation)?;
    Ok(Some(Arc::new(store)))
}

#[cfg(test)]
mod tests;
