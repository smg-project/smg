//! Generic radix membership index: block-quantized prefix chains with
//! per-holder overlap scoring. Pub in, sub out; state is a materialized
//! view of publisher streams (no consensus — see the design doc's
//! epoch/commutativity argument).

pub mod bridge;
pub mod cli;
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
    placement_chain, AddedControl, Applied, ApplyOutcome, Engine, EngineConfig, HolderDigest,
    HolderScore, KeyspaceKey, SymbolKind, UpdateMsg, WireBlock, WireEvent,
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

/// Why a wire update could not be read as an engine update.
///
/// Each of these used to be normalised away. That was the wrong trade:
/// the result still lands as state in *some* keyspace, the publisher
/// reads a normal ack, and the keyspace it was quietly rewritten into
/// answers none of the queries it was meant to answer. A rejected
/// publish is a visible failure the publisher can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateDecodeError {
    /// No `keyspace` on the message. There is no safe default: the
    /// empty-model, zero-block keyspace is itself real and queryable,
    /// so filling one in files the publisher's state where nothing
    /// looks for it.
    MissingKeyspace,
    /// A `symbol_kind` this build does not know. Coercing it to tokens
    /// would put foreign-keyed hashes in the token keyspace, where they
    /// match nothing and evict entries that would have.
    UnknownSymbolKind(i32),
    /// An event whose `kind` oneof is unset. Dropping it silently turns
    /// a removal the publisher believes landed into a block the index
    /// keeps answering with.
    EmptyEvent,
}

impl std::fmt::Display for UpdateDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKeyspace => f.write_str("update has no keyspace"),
            Self::UnknownSymbolKind(kind) => {
                write!(f, "update has an unknown symbol_kind: {kind}")
            }
            Self::EmptyEvent => f.write_str("update carries an event with no kind set"),
        }
    }
}

impl std::error::Error for UpdateDecodeError {}

/// Wire to engine. Validates rather than normalises; see
/// [`UpdateDecodeError`]. The other ingress check is the hash-scheme
/// gate on the publish stream, which fails the whole stream so a
/// publisher on a scheme this build cannot serve cannot fill a keyspace
/// with hashes that match nothing.
impl TryFrom<&proto::Update> for UpdateMsg {
    type Error = UpdateDecodeError;

    fn try_from(u: &proto::Update) -> Result<Self, Self::Error> {
        let keyspace = u
            .keyspace
            .as_ref()
            .ok_or(UpdateDecodeError::MissingKeyspace)?;
        let symbol_kind = match proto::SymbolKind::try_from(keyspace.symbol_kind) {
            Ok(proto::SymbolKind::Bytes) => SymbolKind::Bytes,
            // Unspecified is the proto3 default, so a publisher that
            // simply never set the field means tokens and says so by
            // omission. That is a documented default, not a guess at a
            // value we failed to recognise.
            Ok(proto::SymbolKind::Tokens | proto::SymbolKind::Unspecified) => SymbolKind::Tokens,
            Err(_) => {
                return Err(UpdateDecodeError::UnknownSymbolKind(keyspace.symbol_kind));
            }
        };
        let mut events = Vec::with_capacity(u.events.len());
        for event in &u.events {
            let kind = event.kind.as_ref().ok_or(UpdateDecodeError::EmptyEvent)?;
            events.push(match kind {
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
            });
        }
        Ok(UpdateMsg {
            keyspace: KeyspaceKey {
                model: keyspace.model.clone(),
                symbol_kind,
                block_size: keyspace.block_size,
            },
            holder: u.holder.clone(),
            epoch: u.epoch,
            seq: u.seq,
            events,
            added: u.added.as_ref().map(|a| AddedControl {
                metadata: a.metadata.clone(),
                capacity_blocks: a.capacity_blocks,
                event_fed: a.event_fed,
            }),
            dropped: u.dropped,
        })
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
                metadata: a.metadata.clone(),
            }),
            dropped: u.dropped,
        }
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    fn wire(keyspace: Option<proto::Keyspace>, events: Vec<proto::Event>) -> proto::Update {
        proto::Update {
            keyspace,
            holder: "w1".into(),
            epoch: 1,
            seq: 1,
            events,
            added: None,
            dropped: false,
        }
    }

    fn keyspace(symbol_kind: i32) -> proto::Keyspace {
        proto::Keyspace {
            model: "m".into(),
            symbol_kind,
            block_size: 16,
            hash_scheme: wire_hash::HASH_SCHEME_V1,
        }
    }

    fn cleared() -> proto::Event {
        proto::Event {
            kind: Some(proto::event::Kind::Cleared(true)),
        }
    }

    #[test]
    fn a_keyspaceless_update_is_rejected_not_filed_under_the_empty_keyspace() {
        assert_eq!(
            UpdateMsg::try_from(&wire(None, vec![cleared()])),
            Err(UpdateDecodeError::MissingKeyspace)
        );
    }

    #[test]
    fn a_symbol_kind_this_build_does_not_know_is_rejected_not_read_as_tokens() {
        assert_eq!(
            UpdateMsg::try_from(&wire(Some(keyspace(99)), vec![cleared()])),
            Err(UpdateDecodeError::UnknownSymbolKind(99))
        );
    }

    #[test]
    fn an_event_with_no_kind_set_fails_the_update_rather_than_vanishing() {
        let empty = proto::Event { kind: None };
        assert_eq!(
            UpdateMsg::try_from(&wire(Some(keyspace(1)), vec![cleared(), empty])),
            Err(UpdateDecodeError::EmptyEvent)
        );
    }

    #[test]
    fn an_unset_symbol_kind_still_means_tokens() {
        // Proto3's default, so a publisher that never set the field is
        // saying tokens by omission rather than naming a value we could
        // not recognise. Reading it as tokens is the documented default.
        let msg = UpdateMsg::try_from(&wire(Some(keyspace(0)), vec![cleared()]))
            .expect("unset symbol kind decodes");
        assert_eq!(msg.keyspace.symbol_kind, SymbolKind::Tokens);
        assert_eq!(msg.keyspace.block_size, 16);
    }

    #[test]
    fn a_well_formed_update_round_trips_through_both_conversions() {
        let original = wire(
            Some(keyspace(proto::SymbolKind::Bytes as i32)),
            vec![cleared()],
        );
        let msg = UpdateMsg::try_from(&original).expect("well-formed update decodes");
        assert_eq!(msg.keyspace.symbol_kind, SymbolKind::Bytes);
        assert_eq!(proto::Update::from(&msg), original);
    }
}
