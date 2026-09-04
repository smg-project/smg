//! The index engine: one `radix_tree::RadixTree` per keyspace, with
//! epoch-scoped holder state, per-feed eviction semantics, and
//! overlap queries. Pure and synchronous — the gRPC surface, relay,
//! and bootstrap live in `server.rs`.
//!
//! Correctness model (see `.claude/kv-index-service/01-design.md`):
//! - Event feed: one worker-sequenced stream per (holder, epoch); apply
//!   in order, dedup on `seq <= last_seq`, batch-shaped (one seq may mix
//!   Stored and Removed).
//! - Placement feed: unsequenced (`seq == 0`), idempotent by content —
//!   publishers synthesize identical chains for identical prefixes, so
//!   cross-publisher ordering never matters.
//! - Epochs: a higher epoch clears all lower-epoch state for the holder;
//!   lower-epoch updates are dropped. Restarts and cursor loss are both
//!   epoch bumps, which is what makes relaying `Cleared` safe.
//! - Feed authority: a holder becomes event-fed on its first observed
//!   removal (or an explicit `Added { event_fed: true }`); inferred
//!   Stored updates for an event-fed holder are dropped.
//! - Freshness is holder-granular: idle inferred holders are cleared
//!   by TTL, idle dropped holders are RETIRED entirely. Capacity is
//!   runaway protection only (truncate past 2x declared, tail-first
//!   and prefix-closed) — the placement feed has no removal signal,
//!   so index-side eviction must never race the worker's own.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use radix_tree::{Config as TreeConfig, HolderId, Overlap, OverlapScratch, RadixTree, StoreError};

use crate::{ContentHash, SequenceHash};

/// One block on the wire: position-chained identity + content identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireBlock {
    pub seq_hash: SequenceHash,
    pub content_hash: ContentHash,
}

/// A cache transition within one update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireEvent {
    Stored {
        parent: Option<SequenceHash>,
        blocks: Vec<WireBlock>,
    },
    Removed {
        seq_hashes: Vec<SequenceHash>,
    },
    Cleared,
    /// Duplicate-placement digest (see proto `StoredDigest`): confirm a
    /// chain is already held without its blocks, or miss and force a
    /// full resend.
    StoredDigest {
        parent: Option<SequenceHash>,
        tip: SequenceHash,
        len: u32,
    },
}

/// Membership / capacity control payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedControl {
    pub capacity_blocks: u64,
    pub event_fed: bool,
}

/// One `Publish` message, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateMsg {
    pub keyspace: KeyspaceKey,
    pub holder: String,
    pub epoch: u64,
    /// 0 = unsequenced (placement / control), else the holder's batch seq.
    pub seq: u64,
    pub events: Vec<WireEvent>,
    pub added: Option<AddedControl>,
    pub dropped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcome {
    Applied,
    /// Dropped as a duplicate / stale seq or stale epoch.
    Deduped,
    /// Inferred Stored dropped because the holder is event-fed.
    FeedRejected,
    /// Keyspace block_size conflict — publisher misconfigured.
    KeyspaceMismatch,
    /// A `StoredDigest` the index could not confirm; the publisher must
    /// resend the chain in full (never a silent under-match).
    DigestMiss,
}

/// What one applied update resolved to — the fields a publisher ack
/// carries. Named (not a tuple): `last_seq` and `epoch` are both `u64`
/// and positional confusion between them is exactly how a wrong ack
/// would silently break the bridge's restart adoption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    pub outcome: ApplyOutcome,
    /// The holder's event-feed watermark AFTER this apply.
    pub last_seq: u64,
    /// The holder's STORED epoch after this apply — deliberately not the
    /// update's own epoch. Acks echo this so a restarted publisher (local
    /// epoch back at 1) learns the surviving index's real epoch and
    /// adopts past it, instead of feeding dead-on-arrival updates.
    pub epoch: u64,
    /// Whether state changed (the relay gate: echoes die in one hop).
    pub changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Tokens,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyspaceKey {
    pub model: String,
    pub symbol_kind: SymbolKind,
    pub block_size: u32,
}

/// Per-holder score in a query answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HolderScore {
    pub holder: String,
    pub matched_blocks: u32,
    pub total_blocks: u64,
    pub event_fed: bool,
}

pub struct EngineConfig {
    /// Idle TTL for INFERRED holder state; an inferred holder with no
    /// publish inside the window is cleared entirely (coarse but
    /// prefix-closed). Event-fed holders' BLOCK eviction is observed,
    /// never TTL'd; their liveness backstop is `event_ttl`.
    pub inferred_ttl: Duration,
    /// Liveness BACKSTOP for event-fed holders whose fleet-departure
    /// signal was lost (the primary signal is the gateway's removal
    /// workflow publishing `dropped`; a gateway crash mid-workflow
    /// loses it, and before this backstop such a holder persisted
    /// FOREVER — liveness review finding). An event-fed holder silent
    /// for this window is soft-retired (`dropped`), which stops its
    /// scoring and starts the retire clock; a next observed event
    /// batch self-heals it. Keep an order of magnitude above
    /// `inferred_ttl`; zero disables the backstop.
    pub event_ttl: Duration,
    /// Default capacity (blocks) for inferred holders that never sent
    /// `Added`.
    pub default_capacity_blocks: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            inferred_ttl: Duration::from_secs(180),
            event_ttl: Duration::from_secs(1800),
            default_capacity_blocks: u64::MAX,
        }
    }
}

struct HolderState {
    id: HolderId,
    epoch: u64,
    last_seq: u64,
    event_fed: bool,
    capacity_blocks: u64,
    dropped: bool,
    /// Nanoseconds since `Engine::start` — atomic so the shared-lock
    /// duplicate fast path can refresh freshness WITHOUT the keyspace
    /// write lock (a holder fed only by duplicate placements must not
    /// TTL out).
    last_publish_ns: std::sync::atomic::AtomicU64,
}

struct KeyspaceState {
    tree: RadixTree,
    holders: HashMap<String, HolderState>,
}

/// Poisoned-lock policy: a panic mid-mutation leaves state that must
/// never be served; crash so the replica re-bootstraps clean from a
/// sibling.
const LOCK_MSG: &str = "engine lock poisoned by a panic mid-mutation: aborting so the replica re-bootstraps clean state from a sibling";

/// The engine: all keyspaces, locked at TWO levels. The outer RwLock
/// guards only the keyspace map (creation/removal — rare); every
/// keyspace lives behind its own RwLock. Queries take the SHARED side
/// with caller-owned scratch, and so does the duplicate fast path —
/// the >=90%-duplicate multi-gateway placement stream and all
/// routing-time queries proceed concurrently, while only genuinely
/// state-changing applies take the exclusive side (the multi-writer
/// baseline measured 50-390x query-p99 degradation when every
/// duplicate held an exclusive lock for its full chain walk).
/// Convergence arguments are unaffected: all wire-visible ordering
/// (epoch, seq, feed authority) is per HOLDER, and a holder lives in
/// exactly one keyspace.
pub struct Engine {
    cfg: EngineConfig,
    /// Basis for the per-holder atomic freshness stamps.
    start: Instant,
    keyspaces: std::sync::RwLock<HashMap<KeyspaceKey, Arc<std::sync::RwLock<KeyspaceState>>>>,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self {
            cfg,
            start: Instant::now(),
            keyspaces: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn now_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    fn space(&self, key: &KeyspaceKey) -> Option<Arc<std::sync::RwLock<KeyspaceState>>> {
        self.keyspaces.read().expect(LOCK_MSG).get(key).cloned()
    }

    fn space_or_create(&self, key: &KeyspaceKey) -> Arc<std::sync::RwLock<KeyspaceState>> {
        if let Some(space) = self.space(key) {
            return space;
        }
        self.keyspaces
            .write()
            .expect(LOCK_MSG)
            .entry(key.clone())
            .or_insert_with(|| {
                Arc::new(std::sync::RwLock::new(KeyspaceState {
                    tree: RadixTree::new(TreeConfig::default()),
                    holders: HashMap::new(),
                }))
            })
            .clone()
    }

    /// Stable iteration set for the cross-keyspace paths: each entry is
    /// then locked INDIVIDUALLY, so the walk never freezes the engine.
    fn all_spaces(&self) -> Vec<(KeyspaceKey, Arc<std::sync::RwLock<KeyspaceState>>)> {
        self.keyspaces
            .read()
            .expect(LOCK_MSG)
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Apply one update. Returns what happened, the holder's applied
    /// watermark for the publisher's ack, and whether OBSERVABLE STATE
    /// CHANGED — the relay forwards only changing applies, so a
    /// symmetric-peer echo dies in one hop instead of ping-ponging
    /// forever (the metrics timeline caught ~190x apply amplification
    /// from exactly that loop).
    pub fn apply(&self, update: &UpdateMsg) -> Applied {
        // A keyspace is created on first contact; a later publisher whose
        // key differs only in block_size is a DIFFERENT keyspace by key
        // construction, so mismatch cannot silently merge. Reject only
        // the degenerate block size.
        if update.keyspace.block_size == 0 {
            return Applied {
                outcome: ApplyOutcome::KeyspaceMismatch,
                last_seq: 0,
                epoch: 0,
                changed: false,
            };
        }
        let space = self.space_or_create(&update.keyspace);

        // SHARED-lock duplicate fast path: the multi-gateway steady
        // state is every gateway re-publishing hot placement chains.
        // A placement whose chain the holder fully holds changes
        // NOTHING — resolve it under the read lock (concurrent with
        // queries and with each other) and never touch the write side.
        // Conditions are exactly the shapes whose slow-path outcome is
        // Applied with changed=false: plain same-epoch placement Stored
        // for a known non-event-fed holder. One deliberate divergence:
        // the slow path's runaway-capacity truncate is skipped — a
        // fully-covered duplicate adds zero blocks, so the bound cannot
        // be NEWLY exceeded here; a bound already exceeded (possible
        // only via a later capacity shrink) is enforced on the next
        // non-duplicate store.
        if update.seq == 0 && update.added.is_none() && !update.dropped {
            if let [WireEvent::Stored { parent, blocks }] = update.events.as_slice() {
                let pairs: Vec<(u64, u64)> = blocks
                    .iter()
                    .map(|b| (b.seq_hash.0, b.content_hash.0))
                    .collect();
                // The read-side walk answers both placement questions:
                // fully covered => return without any exclusive work;
                // covered PREFIX => the exclusive section applies only
                // the suffix, re-anchored at the last plain-duplicate
                // key (a fresh 32-block tail on a hot 256-block prefix
                // holds the write lock for the 32, not the 288 — the
                // full-batch hold is what starved queries in the
                // multi-writer baseline).
                let mut split: Option<(u64, usize)> = None;
                {
                    let shared = space.read().expect(LOCK_MSG);
                    if let Some(holder) = shared.holders.get(&update.holder) {
                        if !holder.event_fed && !holder.dropped && update.epoch == holder.epoch {
                            let (run, all) =
                                shared
                                    .tree
                                    .dup_prefix(holder.id, parent.map(|p| p.0), &pairs);
                            if all {
                                holder
                                    .last_publish_ns
                                    .store(self.now_ns(), std::sync::atomic::Ordering::Relaxed);
                                return Applied {
                                    outcome: ApplyOutcome::Applied,
                                    last_seq: holder.last_seq,
                                    epoch: holder.epoch,
                                    changed: false,
                                };
                            }
                            if run > 0 {
                                split = Some((pairs[run as usize - 1].0, run as usize));
                            }
                        }
                    }
                }
                if let Some((anchor, run)) = split {
                    if let Some(result) = self.apply_placement_suffix(
                        &space,
                        update,
                        parent.map(|p| p.0),
                        &pairs,
                        anchor,
                        run,
                    ) {
                        return result;
                    }
                    // Posture changed between the locks (epoch bump,
                    // feed flip, retire): the general path re-resolves.
                }
            }

            // Digest fast path: a duplicate placement that carries NO
            // blocks, only {parent, tip, len}. Confirmed with one
            // read-lock key lookup (prefix-closed => tip at the
            // expected position proves full coverage); a miss returns
            // DigestMiss so the publisher resends the chain in full.
            if let [WireEvent::StoredDigest { parent, tip, len }] = update.events.as_slice() {
                let shared = space.read().expect(LOCK_MSG);
                if let Some(holder) = shared.holders.get(&update.holder) {
                    if holder.event_fed || holder.dropped || update.epoch != holder.epoch {
                        return Applied {
                            outcome: ApplyOutcome::DigestMiss,
                            last_seq: holder.last_seq,
                            epoch: holder.epoch,
                            changed: false,
                        };
                    }
                    let start_pos = match parent {
                        None => Some(0u32),
                        Some(p) => shared.tree.position_of(holder.id, p.0).map(|pp| pp + 1),
                    };
                    // `len` is wire-controlled: the tip-position compare
                    // uses checked arithmetic so an absurd length can
                    // never wrap into a false confirmation (a wrap in a
                    // release build would be the exact silent under-match
                    // digests are forbidden to produce); overflow => miss.
                    let confirmed = *len > 0
                        && start_pos.is_some_and(|start| {
                            start.checked_add(*len - 1).is_some_and(|tip_pos| {
                                shared.tree.position_of(holder.id, tip.0) == Some(tip_pos)
                            })
                        });
                    if confirmed {
                        holder
                            .last_publish_ns
                            .store(self.now_ns(), std::sync::atomic::Ordering::Relaxed);
                        return Applied {
                            outcome: ApplyOutcome::Applied,
                            last_seq: holder.last_seq,
                            epoch: holder.epoch,
                            changed: false,
                        };
                    }
                }
                // Unknown holder or unconfirmed: force a full resend.
                let (last_seq, epoch) = shared
                    .holders
                    .get(&update.holder)
                    .map_or((0, 0), |h| (h.last_seq, h.epoch));
                return Applied {
                    outcome: ApplyOutcome::DigestMiss,
                    last_seq,
                    epoch,
                    changed: false,
                };
            }
        }

        let mut space = space.write().expect(LOCK_MSG);
        self.apply_locked(&mut space, update)
    }

    /// Apply a run of updates, taking each keyspace's write lock ONCE
    /// for its whole same-keyspace run instead of once per update.
    /// Under a heavy event stream this cuts write-lock acquisitions
    /// ~batch-fold, so the shared-lock queries get real gaps between
    /// write bursts instead of ping-ponging against a per-event
    /// writer. The result is IDENTICAL to applying each update
    /// individually in order (only the lock granularity changes, never
    /// the semantics — asserted by `apply_batch_equals_sequential`).
    ///
    /// This path is for the SEQUENCED event feed; it deliberately does
    /// not run the read-lock placement/digest fast paths (those updates
    /// go through `apply`), so a digest reaching here is a miss.
    pub fn apply_batch(&self, updates: &[UpdateMsg]) -> Vec<Applied> {
        let mut out = Vec::with_capacity(updates.len());
        let mut i = 0;
        while i < updates.len() {
            if updates[i].keyspace.block_size == 0 {
                out.push(Applied {
                    outcome: ApplyOutcome::KeyspaceMismatch,
                    last_seq: 0,
                    epoch: 0,
                    changed: false,
                });
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < updates.len() && updates[j].keyspace == updates[i].keyspace {
                j += 1;
            }
            let space = self.space_or_create(&updates[i].keyspace);
            // Apply the same-keyspace run in bounded sub-batches, RELEASING
            // and re-acquiring the write lock between them, so shared-lock
            // queries get a window every SUB_BATCH applies instead of
            // waiting behind the whole run. Caps the query-latency tail
            // under a heavy write burst without a per-update lock grab.
            const SUB_BATCH: usize = 16;
            let mut k = i;
            while k < j {
                let end = (k + SUB_BATCH).min(j);
                {
                    let mut guard = space.write().expect(LOCK_MSG);
                    for u in &updates[k..end] {
                        out.push(self.apply_locked(&mut guard, u));
                    }
                }
                k = end;
            }
            i = j;
        }
        out
    }

    /// The exclusive-lock write path: apply one update to an
    /// already-write-locked keyspace.
    fn apply_locked(&self, space: &mut KeyspaceState, update: &UpdateMsg) -> Applied {
        if !space.holders.contains_key(&update.holder) {
            let id = space.tree.create_holder(&update.holder);
            space.holders.insert(
                update.holder.clone(),
                HolderState {
                    id,
                    epoch: update.epoch,
                    last_seq: 0,
                    event_fed: false,
                    capacity_blocks: self.cfg.default_capacity_blocks,
                    dropped: false,
                    last_publish_ns: std::sync::atomic::AtomicU64::new(self.now_ns()),
                },
            );
        }
        let tree = &mut space.tree;
        let holder = space
            .holders
            .get_mut(&update.holder)
            .expect("holder inserted above");

        let mut changed = false;

        // Control payloads (membership lifecycle) apply BEFORE the
        // epoch gate: the gateway's removal workflow publishes
        // `dropped` at whatever epoch it last saw, while the bridge
        // may have bumped the holder past it — a lifecycle signal
        // silently discarded on epoch mismatch is a holder leak (the
        // liveness review caught exactly this: `dropped` at epoch 1
        // vs a bridge on epoch 2 was Deduped).
        //
        // `changed` is gated on ACTUAL transitions, never on the
        // payload's presence: the relay forwards changing applies to
        // peers, and a peer re-applying an already-standing added/
        // dropped must be a no-op or symmetric replicas ping-pong the
        // lifecycle echo forever (the same loop the ~190x Stored echo
        // amplification came from).
        if let Some(added) = &update.added {
            // Zero means "no capacity claim" — leave the standing
            // value: a bare lifecycle re-announce must not clobber a
            // worker-declared capacity back to the default.
            if added.capacity_blocks != 0 && holder.capacity_blocks != added.capacity_blocks {
                holder.capacity_blocks = added.capacity_blocks;
                changed = true;
            }
            if added.event_fed && !holder.event_fed {
                holder.event_fed = true;
                changed = true;
            }
            if holder.dropped {
                holder.dropped = false;
                changed = true;
            }
        }
        if update.dropped && !holder.dropped {
            holder.dropped = true;
            changed = true;
        }

        // Epoch gate: higher epoch supersedes (implicit clear), lower is
        // dropped, equal proceeds. A bump is proof of life — a new feed
        // generation exists, so a standing soft-retire is healed.
        if update.epoch > holder.epoch {
            tree.clear(holder.id);
            holder.epoch = update.epoch;
            holder.last_seq = 0;
            if !update.dropped {
                holder.dropped = false;
            }
            changed = true;
        } else if update.epoch < holder.epoch {
            return Applied {
                outcome: ApplyOutcome::Deduped,
                last_seq: holder.last_seq,
                epoch: holder.epoch,
                changed,
            };
        }
        holder
            .last_publish_ns
            .store(self.now_ns(), std::sync::atomic::Ordering::Relaxed);

        // Sequenced = event feed; unsequenced = placement/control.
        let sequenced = update.seq != 0;
        if sequenced {
            if update.seq <= holder.last_seq {
                return Applied {
                    outcome: ApplyOutcome::Deduped,
                    last_seq: holder.last_seq,
                    epoch: holder.epoch,
                    changed,
                };
            }
            holder.last_seq = update.seq;
            if holder.dropped && !update.dropped {
                // Observed engine events are proof of life: heal a
                // stale (or wrongly relayed) soft-retire.
                holder.dropped = false;
                changed = true;
            }
        }

        let mut outcome = ApplyOutcome::Applied;
        for event in &update.events {
            match event {
                WireEvent::Stored { parent, blocks } => {
                    if !sequenced && holder.event_fed {
                        // D4: placements never pollute observed holders.
                        outcome = ApplyOutcome::FeedRejected;
                        continue;
                    }
                    if sequenced && !holder.event_fed {
                        // First sequenced traffic pins feed authority too:
                        // a holder with a real event stream is observed.
                        holder.event_fed = true;
                    }
                    let pairs: Vec<(u64, u64)> = blocks
                        .iter()
                        .map(|b| (b.seq_hash.0, b.content_hash.0))
                        .collect();
                    let stored = tree
                        .store(holder.id, parent.map(|p| p.0), &pairs)
                        .or_else(|e| match e {
                            // Unresolvable parent re-anchors at position 0,
                            // mirroring the gateway monitor's recovery.
                            StoreError::ParentNotFound => tree.store(holder.id, None, &pairs),
                            other => Err(other),
                        });
                    // `changed` comes from the OUTCOME, never a
                    // length delta: a MOVE (the re-anchor path above
                    // produces them) changes query answers while
                    // netting zero blocks, and a length-delta
                    // heuristic would suppress its relay — replica
                    // divergence (audit finding).
                    if let Ok(outcome) = &stored {
                        changed |= outcome.applied > 0;
                    }
                    if stored.is_ok() && !holder.event_fed {
                        // Capacity is RUNAWAY PROTECTION, not an
                        // eviction mirror: the placement feed carries
                        // no removal signal, so index-side eviction
                        // order can never match the worker's real
                        // order — truncating AT the worker's size
                        // races it and under-matches (measured: p95
                        // prediction error 0 -> 9216 tokens when the
                        // forest-correct accounting made the old
                        // at-capacity bound actually bind). Truncate
                        // only past 2x declared capacity; TTL remains
                        // the freshness bound.
                        let bound = holder.capacity_blocks.saturating_mul(2);
                        if tree.holder_blocks(holder.id) > bound {
                            tree.truncate_tail(holder.id, bound);
                        }
                    }
                }
                WireEvent::Removed { seq_hashes } => {
                    if !holder.event_fed {
                        holder.event_fed = true;
                    }
                    let keys: Vec<u64> = seq_hashes.iter().map(|h| h.0).collect();
                    changed |= tree.remove(holder.id, &keys) > 0;
                }
                WireEvent::Cleared => {
                    tree.clear(holder.id);
                    changed = true;
                }
                WireEvent::StoredDigest { .. } => {
                    // Digests are handled by the read-lock fast path and
                    // are only ever sent alone. Reaching the general
                    // loop (a mixed batch) means we cannot confirm it
                    // here without its blocks: force a full resend
                    // rather than silently drop it.
                    outcome = ApplyOutcome::DigestMiss;
                }
            }
        }
        Applied {
            outcome,
            last_seq: holder.last_seq,
            epoch: holder.epoch,
            changed,
        }
    }

    /// Split placement apply: exclusive work for ONLY the uncovered
    /// suffix, anchored at the last plain-duplicate key found by the
    /// shared-lock walk. Returns None when holder posture changed
    /// between the locks (the caller falls through to the general
    /// path). A concurrent clear/truncate can invalidate the anchor —
    /// then the FULL batch is stored instead (with the general path's
    /// dangling-parent re-anchor), which is exactly what a full-batch
    /// apply serialized after that clear would have produced: the
    /// split is linearizable, never lossy.
    fn apply_placement_suffix(
        &self,
        space: &Arc<std::sync::RwLock<KeyspaceState>>,
        update: &UpdateMsg,
        original_parent: Option<u64>,
        pairs: &[(u64, u64)],
        anchor: u64,
        run: usize,
    ) -> Option<Applied> {
        let mut space = space.write().expect(LOCK_MSG);
        let space = &mut *space;
        let holder = space.holders.get_mut(&update.holder)?;
        if holder.event_fed || holder.dropped || update.epoch != holder.epoch {
            return None;
        }
        holder
            .last_publish_ns
            .store(self.now_ns(), std::sync::atomic::Ordering::Relaxed);
        let tree = &mut space.tree;
        let stored = tree
            .store(holder.id, Some(anchor), &pairs[run..])
            .or_else(|e| match e {
                StoreError::ParentNotFound => tree
                    .store(holder.id, original_parent, pairs)
                    .or_else(|e| match e {
                        StoreError::ParentNotFound => tree.store(holder.id, None, pairs),
                        other => Err(other),
                    }),
                other => Err(other),
            });
        let mut changed = false;
        if let Ok(outcome) = &stored {
            changed |= outcome.applied > 0;
        }
        if stored.is_ok() {
            let bound = holder.capacity_blocks.saturating_mul(2);
            if tree.holder_blocks(holder.id) > bound {
                tree.truncate_tail(holder.id, bound);
            }
        }
        Some(Applied {
            outcome: ApplyOutcome::Applied,
            last_seq: holder.last_seq,
            epoch: holder.epoch,
            changed,
        })
    }

    /// TTL sweep: clear inferred holders idle beyond the window, and
    /// RETIRE dropped holders entirely (including event-fed ones —
    /// the lifecycle leak the old engine carried). Cheap per-holder
    /// timestamps; run from a timer.
    pub fn sweep_idle(&self) {
        let ttl = self.cfg.inferred_ttl;
        let event_ttl = self.cfg.event_ttl;
        let now = self.now_ns();
        let since = |stamp: &std::sync::atomic::AtomicU64| {
            Duration::from_nanos(
                now.saturating_sub(stamp.load(std::sync::atomic::Ordering::Relaxed)),
            )
        };
        for (key, space_arc) in self.all_spaces() {
            let now_empty = {
                let mut space = space_arc.write().expect(LOCK_MSG);
                let space = &mut *space;
                let mut retire: Vec<String> = Vec::new();
                for (name, holder) in space.holders.iter_mut() {
                    let idle = since(&holder.last_publish_ns) > ttl;
                    if holder.dropped && idle {
                        retire.push(name.clone());
                    } else if !holder.event_fed && idle {
                        space.tree.clear(holder.id);
                    } else if holder.event_fed
                        && !holder.dropped
                        && !event_ttl.is_zero()
                        && since(&holder.last_publish_ns) > event_ttl
                    {
                        // Liveness backstop: silence far beyond the
                        // event feed's cadence means the departure
                        // signal was lost. Soft-retire; a next event
                        // batch self-heals. Replicas converge on this
                        // independently — each observes the same
                        // silence on its own clock.
                        holder.dropped = true;
                    }
                }
                for name in retire {
                    if let Some(holder) = space.holders.remove(&name) {
                        space.tree.retire_holder(holder.id);
                    }
                }
                space.holders.is_empty()
            };
            // Keyspace GC (audit finding: any publisher can mint
            // keyspaces and they were never removed). A keyspace whose
            // last holder retired is unlinked — but only when nothing
            // else holds its Arc: a concurrent apply that already
            // cloned it would otherwise mutate an orphan and lose the
            // update. New clones need the map lock we hold, so the
            // count check cannot race; a skipped removal is retried
            // next sweep.
            if now_empty {
                let mut map = self.keyspaces.write().expect(LOCK_MSG);
                if let Some(arc) = map.get(&key) {
                    let unshared = Arc::strong_count(arc) == 2; // map + our iteration clone
                    if unshared && arc.read().expect(LOCK_MSG).holders.is_empty() {
                        map.remove(&key);
                    }
                }
            }
        }
    }

    /// Overlap query: per-holder matched prefix depth, dropped holders
    /// excluded. Missing keyspace = empty answer (advisory semantics).
    pub fn find_matches(&self, keyspace: &KeyspaceKey, hashes: &[ContentHash]) -> Vec<HolderScore> {
        let Some(space) = self.space(keyspace) else {
            return Vec::new();
        };
        // SHARED lock with THREAD-LOCAL scratch: queries run
        // concurrently with each other and with the duplicate fast
        // path; only state-changing applies exclude them. The scratch
        // (three Vecs + the answer buffer) is reused per worker thread
        // so the routing hot path does not pay a fresh allocation set
        // on every query.
        thread_local! {
            static QUERY_SCRATCH: std::cell::RefCell<(OverlapScratch, Vec<Overlap>, Vec<u64>)> =
                std::cell::RefCell::new((OverlapScratch::default(), Vec::new(), Vec::new()));
        }
        let space = space.read().expect(LOCK_MSG);
        let KeyspaceState { tree, holders } = &*space;
        QUERY_SCRATCH.with(|cell| {
            let (scratch, answers, chain) = &mut *cell.borrow_mut();
            chain.clear();
            chain.extend(hashes.iter().map(|h| h.0));
            tree.overlap(chain, scratch, answers);
            let mut scores = Vec::with_capacity(answers.len());
            for o in answers.iter() {
                let Some(name) = tree.holder_name(o.holder) else {
                    continue;
                };
                let Some(holder) = holders.get(name) else {
                    continue;
                };
                if holder.dropped || o.depth == 0 {
                    continue;
                }
                scores.push(HolderScore {
                    holder: name.to_string(),
                    matched_blocks: o.depth,
                    total_blocks: o.total_blocks,
                    event_fed: holder.event_fed,
                });
            }
            // Holder name as the tie key: equal depths sort identically
            // on every replica, so converged replicas answer the same
            // query with the same Vec (a stable sort alone leaves the
            // tie order at the mercy of per-replica insertion order).
            scores.sort_by(|a, b| {
                b.matched_blocks
                    .cmp(&a.matched_blocks)
                    .then_with(|| a.holder.cmp(&b.holder))
            });
            scores
        })
    }

    /// Serialize current state as synthetic Updates (for `Pull`): one
    /// Stored per holder carrying its blocks in position order, under
    /// the holder's current epoch, unsequenced-for-inferred /
    /// watermark-seq-for-observed so the puller lands with the same
    /// dedup posture. (Gap positions collapse to a contiguous chain,
    /// as before: bootstrap equivalence is scoped to gap-free
    /// holders; gapped ones converge through the feeds.)
    /// Consistency: the cut is per KEYSPACE, not global — sufficient
    /// because every UpdateMsg (and all dedup posture: epoch, seq,
    /// feed authority) is scoped to one holder in one keyspace, so a
    /// puller lands each holder exactly as some real moment saw it.
    /// Reconstruct holder state from a snapshot/Pull `Update`, bypassing
    /// the live feed's seq-dedup and placement/feed-authority rules.
    /// Snapshot chunks all carry the holder's TRUE `last_seq` (a holder
    /// larger than `SNAPSHOT_CHUNK` spans several), so applying them
    /// through `apply` would seq-dedup every chunk after the first and
    /// silently truncate the holder on the bootstrapping replica. This is
    /// authoritative state, not a feed: chunks arrive in order and
    /// parent-linked, and the posture (added/dropped/capacity/event_fed)
    /// rides the first chunk.
    pub fn apply_snapshot(&self, update: &UpdateMsg) {
        if update.keyspace.block_size == 0 {
            return;
        }
        let space = self.space_or_create(&update.keyspace);
        let mut space = space.write().expect(LOCK_MSG);
        let space = &mut *space;
        if !space.holders.contains_key(&update.holder) {
            let id = space.tree.create_holder(&update.holder);
            space.holders.insert(
                update.holder.clone(),
                HolderState {
                    id,
                    epoch: update.epoch,
                    last_seq: 0,
                    event_fed: false,
                    capacity_blocks: self.cfg.default_capacity_blocks,
                    dropped: false,
                    last_publish_ns: std::sync::atomic::AtomicU64::new(self.now_ns()),
                },
            );
        }
        let tree = &mut space.tree;
        let holder = space
            .holders
            .get_mut(&update.holder)
            .expect("holder inserted above");
        // Posture rides the first chunk (later chunks carry no control).
        if let Some(added) = &update.added {
            if added.capacity_blocks != 0 {
                holder.capacity_blocks = added.capacity_blocks;
            }
            if added.event_fed {
                holder.event_fed = true;
            }
        }
        holder.dropped = update.dropped;
        holder.epoch = update.epoch;
        holder.last_seq = update.seq;
        holder
            .last_publish_ns
            .store(self.now_ns(), std::sync::atomic::Ordering::Relaxed);
        // Store every chunk's blocks, parent-linked, with no feed
        // rejection or capacity truncation — reconstructed ground truth.
        for event in &update.events {
            if let WireEvent::Stored { parent, blocks } = event {
                let pairs: Vec<(u64, u64)> = blocks
                    .iter()
                    .map(|b| (b.seq_hash.0, b.content_hash.0))
                    .collect();
                let _ = tree
                    .store(holder.id, parent.map(|p| p.0), &pairs)
                    .or_else(|e| match e {
                        StoreError::ParentNotFound => tree.store(holder.id, None, &pairs),
                        other => Err(other),
                    });
            }
        }
    }

    pub fn snapshot(&self) -> Vec<UpdateMsg> {
        let mut out = Vec::new();
        for (key, space_arc) in self.all_spaces() {
            let space = space_arc.read().expect(LOCK_MSG);
            for (holder_key, holder) in &space.holders {
                let blocks: Vec<WireBlock> = space
                    .tree
                    .enumerate(holder.id)
                    .map(|(_pos, k, content)| WireBlock {
                        seq_hash: SequenceHash(k),
                        content_hash: ContentHash(content),
                    })
                    .collect();
                // CHUNKED: one giant Stored per holder blows through
                // gRPC message limits at production block counts
                // (audit finding: >4MiB past ~210k blocks). Chunks
                // are parent-linked so the puller reassembles the
                // exact chain; the control payload rides only the
                // first chunk.
                const SNAPSHOT_CHUNK: usize = 16_384;
                let mut first = true;
                let mut parent: Option<SequenceHash> = None;
                let mut chunks = blocks.chunks(SNAPSHOT_CHUNK).peekable();
                if chunks.peek().is_none() {
                    out.push(UpdateMsg {
                        keyspace: key.clone(),
                        holder: holder_key.clone(),
                        epoch: holder.epoch,
                        seq: if holder.event_fed { holder.last_seq } else { 0 },
                        events: Vec::new(),
                        added: Some(AddedControl {
                            capacity_blocks: holder.capacity_blocks,
                            event_fed: holder.event_fed,
                        }),
                        dropped: holder.dropped,
                    });
                }
                for chunk in chunks {
                    out.push(UpdateMsg {
                        keyspace: key.clone(),
                        holder: holder_key.clone(),
                        epoch: holder.epoch,
                        seq: if holder.event_fed { holder.last_seq } else { 0 },
                        events: vec![WireEvent::Stored {
                            parent,
                            blocks: chunk.to_vec(),
                        }],
                        added: first.then_some(AddedControl {
                            capacity_blocks: holder.capacity_blocks,
                            event_fed: holder.event_fed,
                        }),
                        dropped: holder.dropped,
                    });
                    parent = chunk.last().map(|b| b.seq_hash);
                    first = false;
                }
            }
        }
        out
    }

    /// Total indexed blocks across keyspaces (stats/tests).
    pub fn entry_count(&self) -> usize {
        self.all_spaces()
            .iter()
            .map(|(_, s)| s.read().expect(LOCK_MSG).tree.stats().distinct_entries as usize)
            .sum()
    }

    /// Point-in-time gauges for the metrics endpoint. One keyspace
    /// locked at a time; cheap relative to apply/query traffic.
    pub fn stats(&self) -> EngineStats {
        let spaces = self.all_spaces();
        let mut stats = EngineStats {
            keyspaces: spaces.len(),
            ..EngineStats::default()
        };
        for (_, space_arc) in &spaces {
            let space = space_arc.read().expect(LOCK_MSG);
            stats.blocks += space.tree.stats().distinct_entries as usize;
            for holder in space.holders.values() {
                stats.holders += 1;
                if holder.event_fed {
                    stats.event_fed_holders += 1;
                }
                if holder.dropped {
                    stats.dropped_holders += 1;
                }
            }
        }
        stats
    }
}

/// Point-in-time engine gauges (see [`Engine::stats`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineStats {
    pub keyspaces: usize,
    pub holders: usize,
    pub event_fed_holders: usize,
    pub dropped_holders: usize,
    pub blocks: usize,
}

/// Deterministic placement chain: content hashes -> position-chained
/// wire blocks, using the indexer's own rolling prefix hash so every
/// publisher synthesizes byte-identical chains for identical prefixes.
pub fn placement_chain(content_hashes: &[ContentHash]) -> Vec<WireBlock> {
    crate::wire_hash::placement_chain(content_hashes)
        .into_iter()
        .map(|(seq_hash, content_hash)| WireBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn move_only_apply_reports_changed_for_relay() {
        // A re-anchor MOVE changes query answers while netting zero
        // blocks; the relay gate must still forward it (audit
        // finding: a length-delta heuristic suppressed it and let
        // replicas diverge).
        let engine = Engine::new(EngineConfig::default());
        let changed = engine.apply(&placement("w1", 21, 4)).changed;
        assert!(changed);
        // Same blocks re-anchored under a DIFFERENT prefix: every
        // key moves, block count unchanged.
        let chain = placement_chain(&prefix_hashes(21, 4));
        let other_parent = placement_chain(&prefix_hashes(22, 2));
        let mut setup = UpdateMsg {
            keyspace: keyspace(),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: vec![WireEvent::Stored {
                parent: None,
                blocks: other_parent.clone(),
            }],
            added: None,
            dropped: false,
        };
        engine.apply(&setup);
        setup.events = vec![WireEvent::Stored {
            parent: Some(other_parent[1].seq_hash),
            blocks: chain.clone(),
        }];
        let before = engine.entry_count();
        let changed = engine.apply(&setup).changed;
        assert!(changed, "move-only apply must relay");
        // Re-applying the identical update is a true no-op and must
        // NOT relay (echo suppression).
        let changed = engine.apply(&setup).changed;
        assert!(!changed, "idempotent echo must not relay");
        let _ = before;
    }

    use rand::{rngs::StdRng, seq::SliceRandom, RngExt, SeedableRng};

    use super::*;
    use crate::wire_hash::content_hash as compute_content_hash;

    fn keyspace() -> KeyspaceKey {
        KeyspaceKey {
            model: "m".into(),
            symbol_kind: SymbolKind::Tokens,
            block_size: 4,
        }
    }

    /// Content hashes for a synthetic prefix: block i hashes tokens
    /// [seed, i] so shared prefixes share leading hashes.
    fn prefix_hashes(seed: u32, blocks: usize) -> Vec<ContentHash> {
        (0..blocks as u32)
            .map(|i| compute_content_hash(&[seed, i]))
            .collect()
    }

    fn placement(holder: &str, seed: u32, blocks: usize) -> UpdateMsg {
        UpdateMsg {
            keyspace: keyspace(),
            holder: holder.into(),
            epoch: 1,
            seq: 0,
            events: vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(seed, blocks)),
            }],
            added: None,
            dropped: false,
        }
    }

    fn event_batch(holder: &str, seq: u64, events: Vec<WireEvent>) -> UpdateMsg {
        UpdateMsg {
            keyspace: keyspace(),
            holder: holder.into(),
            epoch: 1,
            seq,
            events,
            added: None,
            dropped: false,
        }
    }

    fn digest(holder: &str, seed: u32, blocks: usize) -> UpdateMsg {
        let chain = placement_chain(&prefix_hashes(seed, blocks));
        UpdateMsg {
            keyspace: keyspace(),
            holder: holder.into(),
            epoch: 1,
            seq: 0,
            events: vec![WireEvent::StoredDigest {
                parent: None,
                tip: chain.last().expect("non-empty").seq_hash,
                len: blocks as u32,
            }],
            added: None,
            dropped: false,
        }
    }

    #[test]
    fn digest_confirms_held_chain_and_misses_otherwise() {
        // TTL >> refresh cadence: the freshness loop below refreshes
        // every 40ms against a 400ms TTL, so a CI-box stall of a few
        // hundred ms cannot spuriously sweep the holder.
        let engine = Engine::new(EngineConfig {
            inferred_ttl: Duration::from_millis(400),
            ..EngineConfig::default()
        });
        // Digest for a chain the index has never seen -> MISS (resend).
        let (o, changed) = {
            let r = engine.apply(&digest("w1", 71, 6));
            (r.outcome, r.changed)
        };
        assert_eq!(o, ApplyOutcome::DigestMiss);
        assert!(!changed);

        // Establish it with a full placement, then the identical digest
        // is confirmed as a no-op (Applied, no relay) — same observable
        // outcome as a full duplicate placement.
        engine.apply(&placement("w1", 71, 6));
        let (o, changed) = {
            let r = engine.apply(&digest("w1", 71, 6));
            (r.outcome, r.changed)
        };
        assert_eq!(o, ApplyOutcome::Applied);
        assert!(!changed, "confirmed digest must not relay");
        assert_eq!(scores(&engine, 71, 6).len(), 1);

        // Wrong length (chain only 6 deep) -> MISS.
        let o = engine.apply(&digest("w1", 71, 8)).outcome;
        assert_eq!(o, ApplyOutcome::DigestMiss);

        // A confirmed digest refreshes freshness under the TTL: a
        // digest-only-fed holder must not be swept.
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(40));
            assert_eq!(
                engine.apply(&digest("w1", 71, 6)).outcome,
                ApplyOutcome::Applied
            );
            engine.sweep_idle();
        }
        assert_eq!(
            scores(&engine, 71, 6).len(),
            1,
            "digest must keep holder fresh"
        );

        // Event-fed holders never accept placement digests.
        engine.apply(&event_batch(
            "w2",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(72, 4)),
            }],
        ));
        let o = engine.apply(&digest("w2", 72, 4)).outcome;
        assert_eq!(
            o,
            ApplyOutcome::DigestMiss,
            "event-fed holder rejects digest"
        );
    }

    fn scores(engine: &Engine, seed: u32, blocks: usize) -> Vec<(String, u32)> {
        engine
            .find_matches(&keyspace(), &prefix_hashes(seed, blocks))
            .into_iter()
            .map(|s| (s.holder, s.matched_blocks))
            .collect()
    }

    #[test]
    fn placement_chain_is_deterministic_and_positions_match() {
        let hashes = prefix_hashes(7, 6);
        assert_eq!(placement_chain(&hashes), placement_chain(&hashes));
        let engine = Engine::new(EngineConfig::default());
        engine.apply(&placement("w1", 7, 6));
        assert_eq!(scores(&engine, 7, 6), vec![("w1".into(), 6)]);
        // A shorter shared prefix matches its depth only.
        assert_eq!(scores(&engine, 7, 3), vec![("w1".into(), 3)]);
    }

    #[test]
    fn placements_are_idempotent_across_publishers() {
        let a = Engine::new(EngineConfig::default());
        let b = Engine::new(EngineConfig::default());
        // "Gateways" 1..4 place the same prefix repeatedly and extensions
        // of it, in different orders per replica.
        let mut updates = Vec::new();
        for _publisher in 0..4 {
            updates.push(placement("w1", 9, 4));
            updates.push(placement("w1", 9, 8)); // extension
            updates.push(placement("w2", 9, 2)); // shorter copy elsewhere
        }
        let mut rng = StdRng::seed_from_u64(1);
        let mut for_a = updates.clone();
        let mut for_b = updates.clone();
        for_a.shuffle(&mut rng);
        for_b.shuffle(&mut rng);
        for u in &for_a {
            a.apply(u);
        }
        for u in &for_b {
            b.apply(u);
        }
        assert_eq!(scores(&a, 9, 8), scores(&b, 9, 8));
        assert_eq!(a.entry_count(), b.entry_count());
        // And identical to the once-only application.
        let once = Engine::new(EngineConfig::default());
        once.apply(&placement("w1", 9, 8));
        once.apply(&placement("w2", 9, 2));
        assert_eq!(scores(&once, 9, 8), scores(&a, 9, 8));
    }

    #[test]
    fn replicas_converge_under_interleaving_and_duplication() {
        // Event holders: per-holder order preserved (the stream contract);
        // cross-holder interleaving and duplicates are free game.
        // Placement holders: any order, any duplication.
        let mut rng = StdRng::seed_from_u64(42);
        let mut per_holder: Vec<Vec<UpdateMsg>> = Vec::new();
        for h in 0..4u32 {
            let holder = format!("ev{h}");
            let mut seqs = Vec::new();
            let mut chain = placement_chain(&prefix_hashes(100 + h, 12));
            for (i, window) in chain.chunks(3).enumerate() {
                let parent = if i == 0 {
                    None
                } else {
                    Some(chain[i * 3 - 1].seq_hash)
                };
                let mut events = vec![WireEvent::Stored {
                    parent,
                    blocks: window.to_vec(),
                }];
                // Mixed batch: occasionally remove an old tail block in the
                // same seq as a store.
                if i == 3 {
                    events.push(WireEvent::Removed {
                        seq_hashes: vec![chain[11].seq_hash],
                    });
                }
                seqs.push(event_batch(&holder, (i + 1) as u64, events));
            }
            chain.clear();
            per_holder.push(seqs);
        }
        let placements: Vec<UpdateMsg> = (0..6u32)
            .map(|p| placement(&format!("pl{}", p % 2), 200 + (p % 3), 4 + (p % 5) as usize))
            .collect();

        let deliver = |engine: &Engine, rng: &mut StdRng| {
            // Merge per-holder event queues preserving each holder's order.
            let mut cursors = vec![0usize; per_holder.len()];
            let mut pending_placements = placements.clone();
            loop {
                let live: Vec<usize> = cursors
                    .iter()
                    .enumerate()
                    .filter(|(h, &c)| c < per_holder[*h].len())
                    .map(|(h, _)| h)
                    .collect();
                if live.is_empty() && pending_placements.is_empty() {
                    break;
                }
                if !pending_placements.is_empty() && (live.is_empty() || rng.random_bool(0.4)) {
                    let i = rng.random_range(0..pending_placements.len());
                    let u = pending_placements.swap_remove(i);
                    engine.apply(&u);
                    if rng.random_bool(0.3) {
                        engine.apply(&u); // duplicate
                    }
                } else {
                    let h = live[rng.random_range(0..live.len())];
                    let u = &per_holder[h][cursors[h]];
                    engine.apply(u);
                    if rng.random_bool(0.3) {
                        engine.apply(u); // duplicate (deduped by seq)
                    }
                    cursors[h] += 1;
                }
            }
        };

        let a = Engine::new(EngineConfig::default());
        let b = Engine::new(EngineConfig::default());
        deliver(&a, &mut rng);
        deliver(&b, &mut rng);
        for h in 0..4u32 {
            assert_eq!(
                scores(&a, 100 + h, 12),
                scores(&b, 100 + h, 12),
                "event holder {h} diverged"
            );
        }
        for p in 0..3u32 {
            assert_eq!(scores(&a, 200 + p, 8), scores(&b, 200 + p, 8));
        }
        assert_eq!(a.entry_count(), b.entry_count());
    }

    #[test]
    fn higher_epoch_clears_lower_and_stale_epochs_drop() {
        let engine = Engine::new(EngineConfig::default());
        engine.apply(&placement("w1", 5, 6));
        assert_eq!(scores(&engine, 5, 6), vec![("w1".into(), 6)]);

        // Restarted holder announces epoch 2 with a fresh (shorter) cache.
        let mut restarted = placement("w1", 5, 2);
        restarted.epoch = 2;
        let outcome = engine.apply(&restarted).outcome;
        assert_eq!(outcome, ApplyOutcome::Applied);
        assert_eq!(scores(&engine, 5, 6), vec![("w1".into(), 2)]);

        // A late epoch-1 update (relay stragglers) is dropped.
        let outcome = engine.apply(&placement("w1", 5, 6)).outcome;
        assert_eq!(outcome, ApplyOutcome::Deduped);
        assert_eq!(scores(&engine, 5, 6), vec![("w1".into(), 2)]);
    }

    #[test]
    fn tail_eviction_keeps_prefixes_closed() {
        let engine = Engine::new(EngineConfig {
            default_capacity_blocks: 4,
            ..EngineConfig::default()
        });
        // Capacity is runaway protection: truncation fires only past
        // 2x the declared value (racing the worker's own unobservable
        // eviction was measured to under-match), so 8 blocks at
        // capacity 4 stay put...
        engine.apply(&placement("w1", 3, 8));
        assert_eq!(scores(&engine, 3, 8), vec![("w1".into(), 8)]);
        // ...and 12 blocks truncate to the 2x bound, prefix-closed:
        // the HEAD survives (depth 8 from position 0), never a
        // mid-chain hole.
        engine.apply(&placement("w1", 3, 12));
        assert_eq!(scores(&engine, 3, 12), vec![("w1".into(), 8)]);
        assert_eq!(scores(&engine, 3, 4), vec![("w1".into(), 4)]);
        assert_eq!(engine.entry_count(), 8);
    }

    #[test]
    fn event_fed_holders_reject_placements_and_split_batches_do_not_lose_events() {
        let engine = Engine::new(EngineConfig::default());
        let chain = placement_chain(&prefix_hashes(11, 6));
        // One seq carrying Stored + Removed together (engine batch shape).
        engine.apply(&event_batch(
            "w1",
            1,
            vec![
                WireEvent::Stored {
                    parent: None,
                    blocks: chain.clone(),
                },
                WireEvent::Removed {
                    seq_hashes: vec![chain[5].seq_hash],
                },
            ],
        ));
        assert_eq!(scores(&engine, 11, 6), vec![("w1".into(), 5)]);

        // A placement for the now event-fed holder is rejected.
        let outcome = engine.apply(&placement("w1", 12, 3)).outcome;
        assert_eq!(outcome, ApplyOutcome::FeedRejected);
        assert!(scores(&engine, 12, 3).is_empty());

        // Duplicate seq is deduped even with different content.
        engine.apply(&event_batch(
            "w1",
            1,
            vec![WireEvent::Removed {
                seq_hashes: vec![chain[4].seq_hash],
            }],
        ));
        assert_eq!(scores(&engine, 11, 6), vec![("w1".into(), 5)]);
    }

    #[test]
    fn snapshot_bootstrap_reproduces_answers() {
        let a = Engine::new(EngineConfig::default());
        a.apply(&placement("w1", 21, 5));
        let chain = placement_chain(&prefix_hashes(22, 4));
        a.apply(&event_batch(
            "w2",
            3,
            vec![WireEvent::Stored {
                parent: None,
                blocks: chain,
            }],
        ));

        let b = Engine::new(EngineConfig::default());
        for update in a.snapshot() {
            b.apply_snapshot(&update);
        }
        assert_eq!(scores(&a, 21, 5), scores(&b, 21, 5));
        assert_eq!(scores(&a, 22, 4), scores(&b, 22, 4));
        assert_eq!(a.entry_count(), b.entry_count());

        // The bootstrapped replica keeps the event holder's dedup posture:
        // the watermark seq travels, so a replayed old batch is dropped.
        let outcome = b
            .apply(&event_batch("w2", 2, vec![WireEvent::Cleared]))
            .outcome;
        assert_eq!(outcome, ApplyOutcome::Deduped);
    }

    #[test]
    fn lifecycle_controls_apply_across_the_epoch_gate() {
        let engine = Engine::new(EngineConfig::default());
        // Bridge feeds the holder at epoch 3.
        let mut feed = event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(51, 4)),
            }],
        );
        feed.epoch = 3;
        engine.apply(&feed);
        assert_eq!(scores(&engine, 51, 4).len(), 1);

        // The gateway's removal workflow publishes `dropped` at the
        // stale epoch it last saw. It must apply anyway (a lifecycle
        // signal silently discarded on epoch mismatch is a holder
        // leak) — and it must RELAY (changed=true) so replicas learn.
        let drop_msg = UpdateMsg {
            keyspace: keyspace(),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: Vec::new(),
            added: None,
            dropped: true,
        };
        let changed = engine.apply(&drop_msg).changed;
        assert!(changed, "stale-epoch drop must still relay");
        assert!(scores(&engine, 51, 4).is_empty(), "dropped holder scored");

        // Observed event traffic is proof of life: the next sequenced
        // batch heals the soft-retire.
        let mut alive = event_batch(
            "w1",
            2,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(52, 2)),
            }],
        );
        alive.epoch = 3;
        engine.apply(&alive);
        assert_eq!(
            scores(&engine, 52, 2).len(),
            1,
            "event batch must heal drop"
        );
    }

    #[test]
    fn silent_event_holders_are_soft_retired_by_the_backstop() {
        let engine = Engine::new(EngineConfig {
            inferred_ttl: Duration::ZERO,
            event_ttl: Duration::from_nanos(1),
            ..EngineConfig::default()
        });
        engine.apply(&event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(53, 4)),
            }],
        ));
        assert_eq!(scores(&engine, 53, 4).len(), 1);
        std::thread::sleep(Duration::from_millis(2));
        // First sweep: silence past event_ttl soft-retires (stops
        // scoring); second sweep: dropped + idle retires entirely and
        // the emptied keyspace unlinks.
        engine.sweep_idle();
        assert!(
            scores(&engine, 53, 4).is_empty(),
            "backstop must stop scoring"
        );
        engine.sweep_idle();
        assert_eq!(engine.stats().holders, 0, "dropped idle holder must retire");
        assert_eq!(engine.stats().keyspaces, 0);
    }

    #[test]
    fn apply_batch_equals_sequential() {
        // A varied stream of sequenced event batches + placements across
        // several holders (no digests — those deliberately route through
        // `apply`). apply_batch over the whole stream must produce the
        // identical outcome vector AND identical final query answers as
        // applying each update one at a time.
        let mut stream: Vec<UpdateMsg> = Vec::new();
        for h in 0..6u32 {
            let name = format!("w{h}");
            // event chain, sequenced
            for seq in 1..=4u64 {
                stream.push(event_batch(
                    &name,
                    seq,
                    vec![WireEvent::Stored {
                        parent: None,
                        blocks: placement_chain(&prefix_hashes(100 + h, (seq * 2) as usize)),
                    }],
                ));
            }
            // a remove and a re-store
            let chain = placement_chain(&prefix_hashes(100 + h, 6));
            stream.push(event_batch(
                &name,
                5,
                vec![WireEvent::Removed {
                    seq_hashes: vec![chain[chain.len() - 1].seq_hash],
                }],
            ));
            // a placement for a different (inferred) holder
            stream.push(placement(&format!("p{h}"), 200 + h, 5));
        }
        // interleave holders so runs mix
        let seq_engine = Engine::new(EngineConfig::default());
        let mut seq_out = Vec::new();
        for u in &stream {
            seq_out.push(seq_engine.apply(u));
        }
        let batch_engine = Engine::new(EngineConfig::default());
        let batch_out = batch_engine.apply_batch(&stream);

        assert_eq!(seq_out, batch_out, "batch outcome vector diverged");
        for h in 0..6u32 {
            for depth in [4usize, 6, 8] {
                assert_eq!(
                    scores(&seq_engine, 100 + h, depth),
                    scores(&batch_engine, 100 + h, depth),
                    "event-holder query diverged at h{h} depth{depth}"
                );
            }
            assert_eq!(
                scores(&seq_engine, 200 + h, 5),
                scores(&batch_engine, 200 + h, 5),
                "placement-holder query diverged at h{h}"
            );
        }
        assert_eq!(seq_engine.entry_count(), batch_engine.entry_count());
    }

    #[test]
    fn duplicate_fast_path_matches_slow_path_and_keeps_holders_fresh() {
        // TTL >> refresh cadence (see digest_confirms_...): stall-proof
        // margins for a loaded CI box.
        let engine = Engine::new(EngineConfig {
            inferred_ttl: Duration::from_millis(400),
            ..EngineConfig::default()
        });
        let first = engine.apply(&placement("w1", 61, 6));
        assert_eq!(first.outcome, ApplyOutcome::Applied);
        assert!(first.changed, "fresh placement must relay");
        // Re-publish (another gateway routed the same prefix): covered
        // by the shared-lock fast path — same outcome shape, no relay.
        let dup = engine.apply(&placement("w1", 61, 6));
        assert_eq!(dup.outcome, ApplyOutcome::Applied);
        assert!(!dup.changed, "duplicate placement must not relay");
        assert_eq!(scores(&engine, 61, 6).len(), 1);

        // Duplicate-only traffic is still proof of freshness: the fast
        // path must refresh the TTL stamp without the write lock, or a
        // hot-but-duplicate-fed holder would be swept.
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(40));
            engine.apply(&placement("w1", 61, 6));
            engine.sweep_idle();
        }
        assert_eq!(
            scores(&engine, 61, 6).len(),
            1,
            "duplicate-fed holder must never TTL out"
        );

        // Event-fed holders still reject placements (the fast path
        // must not swallow the FeedRejected outcome).
        engine.apply(&event_batch(
            "w2",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(62, 4)),
            }],
        ));
        let mut evt_placement = placement("w2", 62, 4);
        evt_placement.epoch = 1;
        let outcome = engine.apply(&evt_placement).outcome;
        assert_eq!(outcome, ApplyOutcome::FeedRejected);
    }

    #[test]
    fn keyspace_gc_removes_emptied_keyspaces_only() {
        // Small real TTL (not ZERO: with ZERO the idle predicate is
        // elapsed_ns > 0, which can round to the same nanosecond as
        // creation under parallel test load and flake).
        let engine = Engine::new(EngineConfig {
            inferred_ttl: Duration::from_millis(1),
            ..EngineConfig::default()
        });
        engine.apply(&placement("w1", 31, 4));
        let mut other = placement("w2", 32, 4);
        other.keyspace.model = "other-model".into();
        engine.apply(&other);
        assert_eq!(engine.stats().keyspaces, 2);

        // Dropping w1 and sweeping past the TTL retires the holder AND
        // unlinks its now-empty keyspace; the live keyspace stays. w2
        // stays a holder (its blocks are cleared as idle-inferred, but
        // the holder — hence its keyspace — remains).
        let mut drop_w1 = placement("w1", 31, 0);
        drop_w1.events.clear();
        drop_w1.dropped = true;
        engine.apply(&drop_w1);
        std::thread::sleep(Duration::from_millis(3));
        engine.sweep_idle();
        let stats = engine.stats();
        assert_eq!(stats.keyspaces, 1);
        assert_eq!(stats.holders, 1);
        // Recreation after GC is a fresh first contact, not a resurrection.
        engine.apply(&placement("w1", 31, 4));
        assert_eq!(engine.stats().keyspaces, 2);
        assert!(!scores(&engine, 31, 4).is_empty());
    }

    #[test]
    fn per_keyspace_locking_survives_concurrent_mixed_traffic() {
        // Not a performance proof — a race smoke: applies to two
        // keyspaces race snapshot/stats/sweep from other threads, and
        // the end state must equal the sequential expectation.
        let engine = std::sync::Arc::new(Engine::new(EngineConfig::default()));
        let mut handles = Vec::new();
        for space_no in 0..2u32 {
            let engine = engine.clone();
            handles.push(std::thread::spawn(move || {
                for round in 0..200u32 {
                    let mut update = placement(&format!("w{space_no}"), 40 + space_no, 6);
                    if space_no == 1 {
                        update.keyspace.model = "other-model".into();
                    }
                    engine.apply(&update);
                    if round % 16 == 0 {
                        engine.sweep_idle();
                    }
                }
            }));
        }
        for _ in 0..2 {
            let engine = engine.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..200 {
                    let _ = engine.snapshot();
                    let _ = engine.stats();
                    let _ = engine.entry_count();
                }
            }));
        }
        for handle in handles {
            handle.join().expect("no thread may panic");
        }
        let stats = engine.stats();
        assert_eq!(stats.keyspaces, 2);
        assert_eq!(stats.holders, 2);
        assert_eq!(scores(&engine, 40, 6).len(), 1);
    }

    /// A holder larger than one snapshot chunk (16384 blocks) must fully
    /// reconstruct on a bootstrapping replica. Event-fed holders are the
    /// risk: their snapshot chunks all carry the same last_seq, so a naive
    /// per-chunk apply seq-dedups every chunk after the first — silently
    /// truncating the holder to 16384 blocks on the new replica.
    #[test]
    fn snapshot_bootstrap_reconstructs_large_event_fed_chain() {
        let src = Engine::new(EngineConfig::default());
        let big = 20_000usize; // > SNAPSHOT_CHUNK (16384) => multiple chunks
        let chain = placement_chain(&prefix_hashes(1, big));
        // Sequenced (event-fed) so the source is not capacity-truncated.
        src.apply(&event_batch(
            "w-big",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: chain,
            }],
        ));
        assert_eq!(scores(&src, 1, big).first().map(|s| s.1), Some(big as u32));

        // Multiple parent-linked chunks for the big holder.
        let snap = src.snapshot();
        let chunk_count = snap
            .iter()
            .filter(|u| {
                u.holder == "w-big" && matches!(u.events.as_slice(), [WireEvent::Stored { .. }])
            })
            .count();
        assert!(
            chunk_count >= 2,
            "a >16384-block holder must snapshot into multiple chunks, got {chunk_count}"
        );

        // Bootstrap via the reconstruction path (as bootstrap_from does).
        let dst = Engine::new(EngineConfig::default());
        for u in &snap {
            dst.apply_snapshot(u);
        }
        assert_eq!(
            scores(&dst, 1, big).first().map(|s| s.1),
            Some(big as u32),
            "a >16384-block event-fed holder must fully reconstruct from its chunked snapshot"
        );
        // last_seq is preserved (the source's, not bumped per chunk) so
        // the replica resumes the event feed correctly.
        assert_eq!(
            scores(&dst, 1, 16_385).first().map(|s| s.1),
            Some(16_385),
            "the chunk boundary reconstructs as one contiguous chain"
        );
    }

    fn ks_with_block(block_size: u32) -> KeyspaceKey {
        KeyspaceKey {
            model: "m".into(),
            symbol_kind: SymbolKind::Tokens,
            block_size,
        }
    }

    /// A degenerate `block_size == 0` keyspace is rejected, never minted —
    /// on both the single `apply` and the `apply_batch` run-boundary path,
    /// where the reject must also not mis-group a following valid update.
    #[test]
    fn zero_block_size_keyspace_is_rejected_not_minted() {
        let engine = Engine::new(EngineConfig::default());
        let bad = UpdateMsg {
            keyspace: ks_with_block(0),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(1, 2)),
            }],
            added: None,
            dropped: false,
        };
        let outcome = engine.apply(&bad).outcome;
        assert_eq!(outcome, ApplyOutcome::KeyspaceMismatch);
        assert!(
            engine.space(&ks_with_block(0)).is_none(),
            "no keyspace minted"
        );

        // apply_batch: [zero-size, valid, valid] — the zero one is rejected
        // while the two valid updates still group and apply.
        let v1 = placement("w2", 5, 3);
        let v2 = placement("w3", 6, 3);
        let out = engine.apply_batch(&[bad, v1, v2]);
        assert_eq!(out[0].outcome, ApplyOutcome::KeyspaceMismatch);
        assert_eq!(out[1].outcome, ApplyOutcome::Applied);
        assert_eq!(out[2].outcome, ApplyOutcome::Applied);
        assert_eq!(scores(&engine, 5, 3), vec![("w2".to_string(), 3)]);
        assert_eq!(scores(&engine, 6, 3), vec![("w3".to_string(), 3)]);
    }

    /// A bare lifecycle re-announce (`AddedControl.capacity_blocks == 0`)
    /// must NOT clobber a worker-declared capacity back to the default: if
    /// it did, the `capacity * 2` runaway bound would collapse to 0 and the
    /// next placement would truncate the holder to empty — silent cache loss.
    #[test]
    fn bare_readvertise_preserves_declared_capacity() {
        let engine = Engine::new(EngineConfig::default());
        let announce = |cap: u64| UpdateMsg {
            keyspace: keyspace(),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: vec![],
            added: Some(AddedControl {
                capacity_blocks: cap,
                event_fed: false,
            }),
            dropped: false,
        };
        engine.apply(&announce(100));
        engine.apply(&placement("w1", 7, 60));
        assert_eq!(scores(&engine, 7, 60), vec![("w1".to_string(), 60)]);

        // Bare re-announce with capacity 0, then a placement that extends
        // the chain (so it takes the store+truncate path, not the pure
        // duplicate fast path). Capacity 0 clobbered -> bound 0 -> truncate
        // to empty. Preserved (100) -> bound 200 -> full 61 retained.
        engine.apply(&announce(0));
        engine.apply(&placement("w1", 7, 61));
        assert_eq!(
            scores(&engine, 7, 61).first().map(|s| s.1),
            Some(61),
            "capacity must survive a zero re-announce (no truncation to empty)"
        );
    }

    /// The digest fast path's prefix-contiguity soundness rests on a holder
    /// with no mid-chain holes. A hole can only arrive via `Removed`, which
    /// pins the holder as event-fed — and event-fed holders reject every
    /// digest. So a digest can never falsely confirm a chain with a hole.
    #[test]
    fn event_fed_holder_with_a_hole_never_confirms_a_digest() {
        let engine = Engine::new(EngineConfig::default());
        let chain = placement_chain(&prefix_hashes(3, 3));
        engine.apply(&event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: chain.clone(),
            }],
        ));
        // Remove the middle block -> holder is now event-fed with a hole.
        engine.apply(&event_batch(
            "w1",
            2,
            vec![WireEvent::Removed {
                seq_hashes: vec![chain[1].seq_hash],
            }],
        ));
        // A digest that would "confirm" tip at len 3 must MISS, not confirm.
        let outcome = engine.apply(&digest("w1", 3, 3)).outcome;
        assert_eq!(
            outcome,
            ApplyOutcome::DigestMiss,
            "an event-fed holder must never confirm a digest against a holed chain"
        );
    }

    /// Lifecycle relay echo suppression: a control payload relays only on
    /// a REAL transition. Re-applying a standing `dropped` (or a standing
    /// re-announce) reports changed=false — otherwise symmetric replicas
    /// relay the lifecycle echo back and forth forever (the same loop the
    /// ~190x Stored echo amplification came from).
    #[test]
    fn lifecycle_echo_dies_in_one_hop() {
        let engine = Engine::new(EngineConfig::default());
        engine.apply(&placement("w1", 31, 4));
        let drop_msg = UpdateMsg {
            keyspace: keyspace(),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: vec![],
            added: None,
            dropped: true,
        };
        assert!(
            engine.apply(&drop_msg).changed,
            "first drop is a transition"
        );
        assert!(
            !engine.apply(&drop_msg).changed,
            "re-applying a standing drop must not relay (echo ping-pong)"
        );

        let readd = UpdateMsg {
            added: Some(AddedControl {
                capacity_blocks: 0,
                event_fed: false,
            }),
            dropped: false,
            ..drop_msg.clone()
        };
        assert!(engine.apply(&readd).changed, "un-drop is a transition");
        assert!(
            !engine.apply(&readd).changed,
            "re-announcing a live holder must not relay"
        );
    }

    /// The digest confirm uses checked position arithmetic: an absurd
    /// wire-controlled `len` must resolve DigestMiss, never wrap into a
    /// false confirmation (release builds) or panic (debug builds).
    #[test]
    fn digest_with_absurd_len_misses_instead_of_wrapping() {
        let engine = Engine::new(EngineConfig::default());
        engine.apply(&placement("w1", 41, 4));
        let chain = placement_chain(&prefix_hashes(41, 4));
        let absurd = UpdateMsg {
            keyspace: keyspace(),
            holder: "w1".into(),
            epoch: 1,
            seq: 0,
            events: vec![WireEvent::StoredDigest {
                parent: Some(chain[2].seq_hash),
                tip: chain[3].seq_hash,
                len: u32::MAX,
            }],
            added: None,
            dropped: false,
        };
        assert_eq!(engine.apply(&absurd).outcome, ApplyOutcome::DigestMiss);
    }

    /// `Applied.epoch` is the holder's STORED epoch, not an echo of the
    /// update's — the field the publisher ack carries so a restarted
    /// bridge (back at epoch 1) learns the surviving index's real epoch
    /// and adopts past it instead of feeding dead-on-arrival updates.
    #[test]
    fn apply_reports_the_stored_epoch_not_the_updates() {
        let engine = Engine::new(EngineConfig::default());
        // Feed at epoch 7.
        let mut e7 = event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(51, 2)),
            }],
        );
        e7.epoch = 7;
        assert_eq!(engine.apply(&e7).epoch, 7);

        // A restarted publisher at epoch 1: Deduped, and the result
        // carries the STORED epoch (7) — the adoption signal.
        let mut stale = event_batch("w1", 1, vec![WireEvent::Cleared]);
        stale.epoch = 1;
        let r = engine.apply(&stale);
        assert_eq!(r.outcome, ApplyOutcome::Deduped);
        assert_eq!(
            r.epoch, 7,
            "ack must carry the stored epoch, not the update's"
        );
    }

    /// A fresh-seq `Cleared` empties the holder AND relays (`changed`).
    /// The apply branch had never executed under test — every prior
    /// `Cleared` was seq-deduped before the events loop, so a cleared
    /// worker could have kept serving stale blocks to peers unnoticed.
    #[test]
    fn cleared_event_empties_the_holder_and_relays() {
        let engine = Engine::new(EngineConfig::default());
        engine.apply(&event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: placement_chain(&prefix_hashes(9, 4)),
            }],
        ));
        assert_eq!(scores(&engine, 9, 4), vec![("w1".to_string(), 4)]);

        // Fresh seq -> reaches the events loop (not seq-deduped).
        let changed = engine
            .apply(&event_batch("w1", 2, vec![WireEvent::Cleared]))
            .changed;
        assert!(changed, "a clear changes query answers and must relay");
        assert!(
            scores(&engine, 9, 4).is_empty(),
            "the cleared holder no longer matches"
        );
    }

    /// A `Removed` event drops the named blocks, shrinking the match, and
    /// relays. Only the happy Stored path was covered before.
    #[test]
    fn removed_event_drops_blocks_and_relays() {
        let engine = Engine::new(EngineConfig::default());
        let chain = placement_chain(&prefix_hashes(11, 4));
        engine.apply(&event_batch(
            "w1",
            1,
            vec![WireEvent::Stored {
                parent: None,
                blocks: chain.clone(),
            }],
        ));
        assert_eq!(scores(&engine, 11, 4), vec![("w1".to_string(), 4)]);

        // Remove the tail block -> prefix match shrinks to 3, and relays.
        let changed = engine
            .apply(&event_batch(
                "w1",
                2,
                vec![WireEvent::Removed {
                    seq_hashes: vec![chain[3].seq_hash],
                }],
            ))
            .changed;
        assert!(changed, "a removal changes query answers and must relay");
        assert_eq!(
            scores(&engine, 11, 4).first().map(|s| s.1),
            Some(3),
            "the removed tail no longer counts toward the prefix"
        );
    }

    /// The split-placement fast path anchors the exclusive suffix store at a
    /// plain-duplicate key found under the shared lock. A concurrent clear
    /// can delete that anchor before the write lock — the fallback then
    /// reconstructs the full chain from the original parent. This races the
    /// two ops hard and asserts the holder is never left corrupted (the
    /// "linearizable, never lossy" contract).
    #[test]
    fn placement_split_races_clear_without_corruption() {
        use std::thread;
        for seed in 0..8u32 {
            let engine = Arc::new(Engine::new(EngineConfig::default()));
            engine.apply(&placement("w1", seed, 256));

            let extender = {
                let e = Arc::clone(&engine);
                thread::spawn(move || {
                    // Extend the hot 256-prefix -> exercises the split path.
                    for _ in 0..2000 {
                        let _ = e.apply(&placement("w1", seed, 288));
                    }
                })
            };
            let clearer = {
                let e = Arc::clone(&engine);
                thread::spawn(move || {
                    for _ in 0..2000 {
                        let _ = e.apply(&event_batch("w1", 0, vec![WireEvent::Cleared]));
                        let _ = e.apply(&placement("w1", seed, 256));
                    }
                })
            };
            // A panic in either thread (e.g. a corrupt chain) fails the test.
            extender.join().unwrap();
            clearer.join().unwrap();

            // Settle: a full placement must land the whole chain — proof the
            // race never wedged the holder into a suffix-only orphan state.
            engine.apply(&event_batch("w1", 0, vec![WireEvent::Cleared]));
            engine.apply(&placement("w1", seed, 288));
            assert_eq!(
                scores(&engine, seed, 288).first().map(|s| s.1),
                Some(288),
                "after racing split vs clear, a settling full placement must reconstruct the chain"
            );
        }
    }

    /// Keyspace GC unlinks an emptied keyspace only when nothing else holds
    /// its `Arc` (strong_count guard). This races `sweep_idle` (retiring a
    /// dropped holder and GC-ing the keyspace) against placements that
    /// re-create holders in the same keyspace, and asserts no panic/deadlock
    /// and that a placement issued after the race stays queryable — the
    /// keyspace was never orphaned out from under a concurrent apply.
    #[test]
    fn keyspace_gc_races_placement_without_orphaning() {
        use std::thread;
        for seed in 0..8u32 {
            let engine = Arc::new(Engine::new(EngineConfig {
                inferred_ttl: Duration::ZERO,
                ..Default::default()
            }));
            // A dropped holder the sweep will retire, emptying the keyspace.
            engine.apply(&placement("seed", seed, 4));
            engine.apply(&UpdateMsg {
                keyspace: keyspace(),
                holder: "seed".into(),
                epoch: 1,
                seq: 0,
                events: vec![],
                added: None,
                dropped: true,
            });

            let sweeper = {
                let e = Arc::clone(&engine);
                thread::spawn(move || {
                    for _ in 0..3000 {
                        e.sweep_idle();
                    }
                })
            };
            let placer = {
                let e = Arc::clone(&engine);
                thread::spawn(move || {
                    for _ in 0..3000 {
                        let _ = e.apply(&placement("live", seed, 4));
                    }
                })
            };
            sweeper.join().unwrap();
            placer.join().unwrap();

            // A fresh placement (no sweep between apply and query) must be
            // visible: if GC had orphaned the keyspace under a concurrent
            // apply, the map lookup here would miss it.
            engine.apply(&placement("final", seed, 4));
            assert!(
                scores(&engine, seed, 4).iter().any(|(h, _)| h == "final"),
                "a placement after the GC race must be queryable (keyspace not orphaned)"
            );
        }
    }
}
