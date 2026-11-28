//! The policy document: its schema and validator, where a request's routing key
//! comes from, the calls an operator makes on a document in force, and the
//! document a gateway's routing settings mean.
//!
//! A Rust caller outside this crate writes no literal of a record a policy
//! document states: the records gain fields, and a pattern of one ends with
//! `..`. No record a document states says so in its own doc, because that doc
//! is the record's description in the published schema. How such a caller
//! comes to nineteen of them:
//!
//! - Ten have no start: [`ResidentWorkParams`], [`LeastCostParams`],
//!   [`BestParams`], [`BestWithTemperatureParams`], [`HasRoomParams`],
//!   [`Bounds`], [`HeaderMatch`], [`SessionsSpec`], [`CanarySpec`] and
//!   [`SignalSpec`]. Code outside this crate builds none: it deserializes
//!   one, or takes one in a spec the crate made. A [`Bounds`] of a caller's
//!   own type needs that type's read. A start comes the day a caller needs
//!   one.
//! - Seven start from their `Default`, and the caller assigns what it states
//!   otherwise: [`IndexSpec`], [`BucketParams`], [`CostMargin`], [`Match`],
//!   [`OverloadSpec`], [`KeySourceSpec`] and [`BindingSpec`].
//! - Two start from a call that takes the one field a document must write:
//!   [`ProfileSpec::of`] and [`RoutingKeySpec::of`].

pub(crate) mod applied;
#[cfg(feature = "operator")]
pub mod operator;
pub(crate) mod routing_key;
pub(crate) mod routing_settings;
pub(crate) mod spec;

pub use applied::AppliedSpec;
pub use routing_key::{strip_lineage, DerivedKey, DerivedKeys, KeySource, KeySources};
pub use routing_settings::{
    ConnectionModeSetting, LoadMetricSetting, ModeSettings, PolicySettings,
    RoutingKeyOverrideSettings, RoutingSettings, RoutingSettingsError,
};
pub use spec::{
    AcrossRanks, AdmissionSpec, AffinitySource, Assignment, BestParams, BestWithTemperatureParams,
    BindingSpec, BoundedLoadParams, Bounds, BucketParams, CanarySpec, Capacity,
    ConsistentHashParams, CostMargin, DecodePick, Digest, ExplainSpec, FilterSpec, FilterStage,
    HasRoomParams, HeaderMatch, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
    LeastCostParams, LeastLoadParams, Match, Needs, OverloadSpec, OverloadSwitch, Override,
    Pairing, PdPairPlanSpec, PickSpec, PickStage, PlanName, PlansSpec, PolicySpec,
    PrefixAffinityParams, ProfileSpec, Recovery, ResidentWorkParams, RoutingKeySpec, RulePlaced,
    ScoreSpec, ScoreStage, SessionsSpec, ShadowSpec, SignalSpec, SpecError, SpillParams,
    Unreported, ViewListPositions, ViewLists, ViewSpec, WarmupSliceParams, Weight, API_VERSION,
    DENIED_HEADERS, DENIED_KEY_HEADERS, FILTERS, KEY_BYTES, MANUAL_SESSION_HEADER,
    MAX_BLOCK_HASHES, MAX_CLIENT_SOURCES, MAX_ENGINES, MAX_HEADERS, MAX_KEY_BYTES, MAX_KEY_SOURCES,
    MAX_LABELS, MAX_SIGNALS, MAX_ZONES, OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS, PICKERS,
    PRESETS, SCORERS,
};
