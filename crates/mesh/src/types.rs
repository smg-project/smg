//! Value types that flow across the mesh wire and into gateway
//! adapters. Kept in the mesh crate so producers and consumers
//! share a single canonical definition.

use serde::{Deserialize, Serialize};

/// Worker state entry synced across mesh nodes. `spec` is an
/// opaque JSON-serialized `WorkerSpec`; the mesh crate
/// doesn't interpret it.
///
/// `Eq`/`Hash` are intentionally omitted: `load: f64` can be
/// NaN, which would violate `Eq` reflexivity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
    /// Opaque worker specification (JSON-serialized `WorkerSpec`
    /// from the gateway; JSON because the type's serde-skip
    /// attributes don't round-trip positional formats). Empty on
    /// old nodes that don't populate this field.
    #[serde(default)]
    pub spec: Vec<u8>,
    /// Mesh server name of the node that published the entry: the key's
    /// owner. Only the owner tombstones its keys, so an importer reads this
    /// against the cluster state and leaves the keys of a node that is gone
    /// alone. Empty on nodes that predate the field (owner unknown); a
    /// positional record without it does not decode as this struct, so the
    /// gateway adapter falls back to the previous layout for those.
    #[serde(default)]
    pub owner: String,
}
