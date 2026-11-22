//! The decision engine: the stages run over one request's candidate workers,
//! what a decision reads (the view) and what it returns, and the engine that
//! decides over the policy store.
//!
//! [`PromptHashes`] are the block hashes of a request's prompt, made under
//! its [`HashSeeds`]. A caller writes them into the request facts of the
//! view.
//!
//! The session table that a `sessions` rule reads lives in this folder too.
//! So does the warm-up ruling: a stage applies it, and a host names a pool's
//! warming rows by it.

pub(crate) mod decision;
pub(crate) mod decision_slot;
pub(crate) mod demands;
pub(crate) mod framework;
pub(crate) mod prefix_pass;
pub(crate) mod prompt_hashes;
pub(crate) mod sticky_sessions;
pub(crate) mod store_backed_engine;
pub(crate) mod view;
pub(crate) mod warm_up;

pub use decision::{
    Decision, LiftedVeto, Overlap, OverlapSource, Pair, Pick, Queue, ShadowKind, Shed, ShedReason,
    Survivors, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
};
pub use decision_slot::{DecisionSlot, DecisionSlotGuard};
pub use demands::{DemandFacts, Demands, References, UnmetDemand};
pub use framework::{
    Binding, DecisionScope, DrawUnit, EncodeItems, EncodePlaced, EncodePool, EncoderFacts, Engine,
    Filter, OverloadThresholds, PairPools, PairSide, PairTable, PairTableError, Picker, Plan,
    Profile, ScoreFront, ScoreSpread, Scorer, Scratch, Shortlist, TieBand, WeightedScorer,
    DRAW_BOUND, DRAW_FROM, RANK_FROM,
};
pub use prompt_hashes::{HashSeeds, PromptHashes};
pub use sticky_sessions::{
    SessionExpiry, SessionPins, SessionRule, SessionTable, SessionVerdict,
    ABSENT_WORKER_PIN_TTL_MICROS,
};
pub use store_backed_engine::{
    DecideOutcome, DecisionExplanation, DecisionNames, DecisionRecord, DecisionRefused, Leased,
    Request, RequestFromSnapshot, StoreBackedEngine, SweepOutcome, SHADOW_ROWS,
};
pub use view::{
    fingerprint, key_of_session, permille, AffinityFacts, AffinityHolder, AffinityRows, Candidate,
    Constraints, Engines, HashKind, Header, Health, Identity, KeyedRow, LoadRow, MediaCounts,
    ModelFingerprint, OverloadRegime, Phase, PinSource, PoolFacts, RequestFacts, Role, Session,
    SignalSource, WireModes, WorkerId, ZoneCapacity, TEXT_BYTES_PER_TOKEN,
};
pub use warm_up::{WarmUpRule, WarmUpState};
