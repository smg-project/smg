//! The decision engine: the stages run over one request's candidate workers,
//! what a decision reads (the view) and what it returns, and the engine that
//! decides over the policy store. Behind the `in-process` feature,
//! `PolicyEngine` is the engine in a gateway's own process: it gives the
//! gateway's own placement its pick over the gateway's worker objects, read
//! in place.
//!
//! [`PromptHashes`] are the block hashes of a request's prompt, made under
//! its [`HashSeeds`]. A caller writes them into the request facts of the
//! view.
//!
//! The session table that a `sessions` rule reads lives in this folder too.
//! So does the warm-up ruling: a stage applies it, and a host names a pool's
//! warming rows by it.

pub(crate) mod decision;
pub(crate) mod decision_engine;
pub(crate) mod decision_slot;
pub(crate) mod demands;
pub(crate) mod framework;
pub(crate) mod pair_classes;
#[cfg(feature = "in-process")]
pub(crate) mod policy_engine;
pub(crate) mod prefix_pass;
pub(crate) mod prompt_hashes;
#[cfg(feature = "in-process")]
pub(crate) mod records;
pub(crate) mod sticky_sessions;
pub(crate) mod store_backed_engine;
pub(crate) mod view;
pub(crate) mod warm_up;

pub use decision::{
    Decision, LiftedVeto, Overlap, OverlapSource, Pair, Pick, Queue, ShadowKind, Shed, ShedReason,
    Survivors, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
};
#[cfg(feature = "in-process")]
pub use decision_engine::WorkerLists;
pub use decision_engine::{DecisionEngine, HeldDecision, RequestMaker};
pub use decision_slot::{DecisionSlot, DecisionSlotGuard};
pub use demands::{DemandFacts, Demands, References, UnmetDemand};
pub use framework::{
    AloneProfile, Binding, DecisionScope, DrawUnit, EncodeItems, EncodePlaced, EncodePool,
    EncoderFacts, Engine, Filter, InTheWay, NotAlone, OverloadThresholds, PairPools, PairSide,
    PairTable, PairTableError, Picker, Place, Plan, PoolWord, Profile, ScoreFront, ScoreSpread,
    Scorer, Scratch, Shortlist, TieBand, WeightedScorer, DRAW_BOUND, DRAW_FROM, RANK_FROM,
};
pub use pair_classes::PartnerClasses;
#[cfg(feature = "in-process")]
pub use policy_engine::overlap::{
    self, Affinity, CacheIndex, CacheIndexes, PoolAffinity, PrefixRead,
};
#[cfg(feature = "in-process")]
pub use policy_engine::pair::{prefills_full, PairAdmission, PairLists};
#[cfg(feature = "in-process")]
pub use policy_engine::{PolicyEngine, PoolMembers};
pub use prompt_hashes::{HashSeeds, PromptHashes};
#[cfg(feature = "in-process")]
pub use records::{decide, worker_id, DecisionClock, PoolRecords};
pub use sticky_sessions::{
    SessionExpiry, SessionPins, SessionRule, SessionTable, SessionVerdict,
    ABSENT_WORKER_PIN_TTL_MICROS,
};
pub use store_backed_engine::{
    DecideOutcome, DecisionExplanation, DecisionNames, DecisionRecord, DecisionRefused, Leased,
    Request, StoreBackedEngine, SweepOutcome, SHADOW_ROWS,
};
pub use view::{
    fingerprint, key_of_session, permille, AffinityFacts, AffinityHolder, AffinityRows, Candidate,
    Constraints, Engines, HashKind, Header, Health, Identity, KeyedRow, LoadRow, MediaCounts,
    ModelFingerprint, OverloadRegime, Phase, PinSource, PoolFacts, RequestFacts, Role, Session,
    SignalSource, WireModes, WorkerId, ZoneCapacity, TEXT_BYTES_PER_TOKEN,
};
pub use warm_up::{WarmUpRule, WarmUpState};
