//! The decision engine: the stages run over one request's candidate workers,
//! what a decision reads (the view) and what it returns, and the engine that
//! decides over the policy store.

pub(crate) mod decider;
pub(crate) mod decision;
pub(crate) mod demands;
pub(crate) mod door;
pub(crate) mod framework;
pub(crate) mod sticky_sessions;
pub(crate) mod terms;
pub(crate) mod view;

pub use decider::{Acting, Decider};
pub use decision::{
    Decision, Lifted, Overlap, OverlapSource, Pair, Pick, Queue, ShadowKind, Shed, ShedReason,
    Survivors, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
};
pub use demands::{DemandFacts, Demands, References, Unmet};
pub use door::{
    Decided, DecisionRecord, Door, Explained, Leased, Maker, Names, Refused, Request, Swept,
    SHADOW_ROWS,
};
pub use framework::{
    Binding, DecisionScope, EncodeItems, EncodePlaced, EncodePool, EncoderFacts, Engine, Filter,
    Front, Narrowed, OverloadThresholds, PairPools, PairSide, PairTable, PairTableError, Picker,
    Plan, Profile, Scorer, Scratch, Spread, TieBand, Unit, Weighted, DRAW_BOUND, DRAW_FROM,
    RANK_FROM,
};
pub use sticky_sessions::{
    Forgets, Pins, SessionRule, SessionTable, SessionVerdict, ABSENCE_MICROS,
};
pub use view::{
    fingerprint, key_of_session, permille, AffinityFacts, Candidate, Constraints, Engines,
    HashKind, Header, Health, Identity, KeyedRow, LoadRow, MediaCounts, ModelFingerprint,
    NamedAffinity, NamedHolder, OverloadRegime, Phase, PinSource, PoolFacts, RequestFacts, Role,
    Session, SignalSource, WireModes, WorkerId, ZoneCapacity, TEXT_BYTES_PER_TOKEN,
};
