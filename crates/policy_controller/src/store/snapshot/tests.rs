//! The fixtures over [`MemoryStore`] of the cases of a publication's
//! reader of one gateway's standing vetoes ([`veto_reader`]). Every
//! instant is handed in, none is slept.

use super::StoreSnapshot;
use crate::{
    engine::view::{LoadRow, Role, WorkerId},
    store::{GatewayId, MemoryStore, WorkerFacts, WorkersStore},
};

mod veto_reader;

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

/// The tests' way to a row a decision reads: registered, and the build its
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

fn load(running: u32, age_ms: u32) -> LoadRow {
    LoadRow {
        running,
        report_age_ms: age_ms,
        ..LoadRow::default()
    }
}
