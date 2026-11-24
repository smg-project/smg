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
pub(crate) mod explain;
pub(crate) mod framework;
#[cfg(feature = "in-process")]
pub(crate) mod lease_events;
#[cfg(feature = "in-process")]
pub(crate) mod list_cell;
pub(crate) mod pair_classes;
#[cfg(feature = "in-process")]
pub(crate) mod policy_engine;
pub(crate) mod prefix_pass;
pub(crate) mod prompt_hashes;
#[cfg(feature = "in-process")]
pub(crate) mod records;
pub(crate) mod shadow;
pub(crate) mod sticky_sessions;
pub(crate) mod view;
pub(crate) mod warm_up;

pub use decision::{
    Decision, LiftedVeto, Overlap, OverlapSource, Pair, Pick, PlacedBy, Queue, ShadowKind, Shed,
    ShedReason, Survivors, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
};
pub use decision_engine::{
    AlonePick, DecideOutcome, Decided, DecisionEngine, DecisionRefused, DecodeRefused,
    HeldDecision, HeldDecode, Leased, Plain, Request, RequestMaker,
};
#[cfg(feature = "in-process")]
pub use decision_engine::{TokenKey, WorkerLists};
pub use decision_slot::{DecisionSlot, DecisionSlotGuard};
pub use demands::{DemandFacts, Demands, References, UnmetDemand};
pub use explain::{DecisionNames, DecisionRecord};
pub use framework::{
    AloneProfile, Binding, DecisionScope, DrawUnit, EncodeItems, EncodePlaced, EncodePool,
    EncoderFacts, Engine, Filter, FirstPartners, FirstPartnersError, InTheWay, NotAlone,
    OverloadThresholds, PairPools, PairSide, PairTable, PairTableError, Picker, Place, Plan,
    PoolRows, PoolWord, Profile, ScoreFront, ScoreSpread, Scorer, Scratch, Shortlist, TieBand,
    WeightedScorer, DRAW_BOUND, DRAW_FROM, RANK_FROM,
};
#[cfg(feature = "in-process")]
pub use list_cell::{LentList, ListCell, ModelLists};
pub use pair_classes::PartnerClasses;
#[cfg(feature = "in-process")]
pub use policy_engine::encode::EncodeLists;
#[cfg(feature = "in-process")]
pub use policy_engine::membership::Members;
#[cfg(feature = "in-process")]
pub use policy_engine::overlap::{
    self, Affinity, CacheIndex, CacheIndexes, PoolAffinity, PrefixRead,
};
#[cfg(feature = "in-process")]
pub use policy_engine::pair::{prefills_full, PairAdmission, PairLists};
#[cfg(feature = "in-process")]
pub use policy_engine::place::{
    HeaderRead, PlaceRequest, Prompt, TokenEntry, Unplaced, WorkerRule,
};
#[cfg(feature = "in-process")]
pub use policy_engine::start::WorkerData;
#[cfg(feature = "in-process")]
pub use policy_engine::PolicyEngine;
pub use prompt_hashes::{HashSeeds, PromptHashes};
#[cfg(feature = "in-process")]
pub use records::{decide, worker_id, worker_rank, DecisionClock, Live, PoolRecords};
pub use shadow::SHADOW_ROWS;
pub use sticky_sessions::{
    DecodeVerdict, SessionExpiry, SessionPins, SessionRule, SessionTable, SessionVerdict,
    ABSENT_WORKER_PIN_TTL_MICROS,
};
pub use view::{
    fingerprint, key_of_session, permille, AffinityFacts, AffinityHolder, AffinityRows, Candidate,
    Constraints, Engines, FixedFacts, HashKind, Header, Health, Identity, LoadRow, MediaCounts,
    ModelFingerprint, OverloadRegime, Phase, PoolFacts, RequestFacts, Role, Session,
    SessionCountedUnder, SignalSource, WireModes, WorkerId, ZoneCapacity, TEXT_BYTES_PER_TOKEN,
};
pub use warm_up::{WarmUpRule, WarmUpState};
