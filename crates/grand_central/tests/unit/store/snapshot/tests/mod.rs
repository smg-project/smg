//! The fixtures over [`MemoryStore`] of the cases of a publication's
//! reader of one gateway's standing vetoes ([`veto_reader`]), and the one
//! count a case of the operator reads of the snapshot's writer
//! (`Writer::listed`). Every instant is handed in, none is slept.

use super::{StoreSnapshot, Writer};
use crate::{
    engine::view::{Role, WorkerId},
    store::{GatewayId, MemoryStore, WorkerFacts},
};

mod veto_reader;

impl Writer {
    /// How many members' reads under the standing generation's lists are
    /// kept ([`Self::list`], [`Self::relist`]).
    #[cfg(feature = "operator")]
    pub(in crate::store) fn listed(&self) -> usize {
        self.listed.len()
    }
}

const MODEL: &str = "m";
const ASKER: GatewayId = GatewayId(1);

fn facts(key: &str, role: Role, models: &[&str]) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation: 1,
        role,
        models: models.iter().map(|model| (*model).to_string()).collect(),
        dp_size: 1,
        ..WorkerFacts::default()
    }
}

/// The tests' way to a member a world lists: registered, and the build its
/// write owes asked at once.
fn admit(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
    let id = store.upsert(facts).id();
    store.publish_now();
    id
}

/// A worker of the model, registered and not yet ready.
fn member(store: &MemoryStore, key: &str) -> WorkerId {
    admit(store, facts(key, Role::Single, &[MODEL]))
}
