//! Generic radix membership index: block-quantized prefix chains with
//! per-holder overlap scoring. Pub in, sub out; state is a materialized
//! view of publisher streams (no consensus — see the design doc's
//! epoch/commutativity argument).

pub mod bridge;
pub mod client;
pub mod engine;
pub mod server;
pub mod wire_hash;

/// Default keyspace block size when a deployment does not configure one.
/// The gateway (`--kv-indexer-block-size`) and the event bridge
/// (`--block-size`) MUST agree — the keyspace key includes the block
/// size, so mismatched defaults silently split one fleet's state into
/// two keyspaces that never answer each other's queries. Always set the
/// flags to the backend's real block size in production; this default
/// only guarantees the two halves agree when both are left unset.
pub const DEFAULT_BLOCK_SIZE: u32 = 128;

pub use engine::{
    placement_chain, AddedControl, Applied, ApplyOutcome, Engine, EngineConfig, HolderScore,
    KeyspaceKey, SymbolKind, UpdateMsg, WireBlock, WireEvent,
};

/// Generated protobuf/tonic types.
pub mod proto {
    tonic::include_proto!("radix_index");
}

/// Position-independent content identity — the wire's matching
/// currency. Owned by the service (R2 dropped the kv_index import;
/// the numeric scheme lives in [`wire_hash`], pinned by golden
/// vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

/// Position-aware block identity (backend block hash on the event
/// feed; deterministic chain hash on the placement feed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceHash(pub u64);

impl From<&proto::Update> for UpdateMsg {
    fn from(u: &proto::Update) -> Self {
        let keyspace = u.keyspace.as_ref();
        UpdateMsg {
            keyspace: KeyspaceKey {
                model: keyspace.map(|k| k.model.clone()).unwrap_or_default(),
                symbol_kind: match keyspace.map(|k| k.symbol_kind) {
                    Some(k) if k == proto::SymbolKind::Bytes as i32 => SymbolKind::Bytes,
                    _ => SymbolKind::Tokens,
                },
                block_size: keyspace.map(|k| k.block_size).unwrap_or_default(),
            },
            holder: u.holder.clone(),
            epoch: u.epoch,
            seq: u.seq,
            events: u
                .events
                .iter()
                .filter_map(|e| e.kind.as_ref())
                .map(|kind| match kind {
                    proto::event::Kind::Stored(s) => WireEvent::Stored {
                        parent: s.parent_seq_hash.map(SequenceHash),
                        blocks: s
                            .blocks
                            .iter()
                            .map(|b| WireBlock {
                                seq_hash: SequenceHash(b.seq_hash),
                                content_hash: ContentHash(b.content_hash),
                            })
                            .collect(),
                    },
                    proto::event::Kind::Removed(r) => WireEvent::Removed {
                        seq_hashes: r.seq_hashes.iter().copied().map(SequenceHash).collect(),
                    },
                    proto::event::Kind::Cleared(_) => WireEvent::Cleared,
                    proto::event::Kind::StoredDigest(d) => WireEvent::StoredDigest {
                        parent: d.parent_seq_hash.map(SequenceHash),
                        tip: SequenceHash(d.tip_seq_hash),
                        len: d.len,
                    },
                })
                .collect(),
            added: u.added.as_ref().map(|a| AddedControl {
                capacity_blocks: a.capacity_blocks,
                event_fed: a.event_fed,
            }),
            dropped: u.dropped,
        }
    }
}

impl From<&UpdateMsg> for proto::Update {
    fn from(u: &UpdateMsg) -> Self {
        proto::Update {
            keyspace: Some(proto::Keyspace {
                model: u.keyspace.model.clone(),
                symbol_kind: match u.keyspace.symbol_kind {
                    SymbolKind::Tokens => proto::SymbolKind::Tokens as i32,
                    SymbolKind::Bytes => proto::SymbolKind::Bytes as i32,
                },
                block_size: u.keyspace.block_size,
                hash_scheme: wire_hash::HASH_SCHEME_V1,
            }),
            holder: u.holder.clone(),
            epoch: u.epoch,
            seq: u.seq,
            events: u
                .events
                .iter()
                .map(|e| proto::Event {
                    kind: Some(match e {
                        WireEvent::Stored { parent, blocks } => {
                            proto::event::Kind::Stored(proto::Stored {
                                parent_seq_hash: parent.map(|p| p.0),
                                blocks: blocks
                                    .iter()
                                    .map(|b| proto::Block {
                                        seq_hash: b.seq_hash.0,
                                        content_hash: b.content_hash.0,
                                    })
                                    .collect(),
                            })
                        }
                        WireEvent::Removed { seq_hashes } => {
                            proto::event::Kind::Removed(proto::Removed {
                                seq_hashes: seq_hashes.iter().map(|h| h.0).collect(),
                            })
                        }
                        WireEvent::Cleared => proto::event::Kind::Cleared(true),
                        WireEvent::StoredDigest { parent, tip, len } => {
                            proto::event::Kind::StoredDigest(proto::StoredDigest {
                                parent_seq_hash: parent.map(|p| p.0),
                                tip_seq_hash: tip.0,
                                len: *len,
                            })
                        }
                    }),
                })
                .collect(),
            added: u.added.as_ref().map(|a| proto::Added {
                capacity_blocks: a.capacity_blocks,
                event_fed: a.event_fed,
                metadata: Vec::new(),
            }),
            dropped: u.dropped,
        }
    }
}
