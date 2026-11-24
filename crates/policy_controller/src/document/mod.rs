//! The policy document: its schema and validator, where a request's routing key
//! comes from, the calls an operator makes on a document in force, and the
//! document a gateway's routing settings mean.

#[cfg(feature = "operator")]
pub mod operator;
pub(crate) mod routing_key;
pub(crate) mod routing_settings;
pub(crate) mod spec;

pub use routing_key::{strip_lineage, DerivedKey, DerivedKeys, KeySource, KeySources};
pub use routing_settings::{
    ConnectionModeSetting, LoadMetricSetting, ModeSettings, PolicySettings,
    RoutingKeyOverrideSettings, RoutingSettings, RoutingSettingsError,
};
pub use spec::{
    AcrossRanks, AdmissionSpec, AffinitySource, AppliedSpec, Assignment, BestParams,
    BestWithTemperatureParams, BindingSpec, BoundedLoadParams, Bounds, BucketParams, CanarySpec,
    Capacity, ConsistentHashParams, CostMargin, DecodePick, Digest, ExplainSpec, FilterSpec,
    FilterStage, HasRoomParams, HeaderMatch, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
    LeastCostParams, LeastLoadParams, Match, Needs, OverloadSpec, Override, Pairing,
    PdPairPlanSpec, PickSpec, PickStage, PlanName, PlansSpec, PolicySpec, PrefixAffinityParams,
    ProfileSpec, Recovery, ResidentWorkParams, RoutingKeySpec, RulePlaced, ScoreSpec, ScoreStage,
    SessionsSpec, ShadowSpec, SignalSpec, SpecError, SpillParams, Unreported, ViewListPositions,
    ViewLists, ViewSpec, WarmupSliceParams, Weight, API_VERSION, DENIED_HEADERS,
    DENIED_KEY_HEADERS, FILTERS, KEY_BYTES, MANUAL_SESSION_HEADER, MAX_BLOCK_HASHES,
    MAX_CLIENT_SOURCES, MAX_ENGINES, MAX_HEADERS, MAX_KEY_BYTES, MAX_KEY_SOURCES, MAX_LABELS,
    MAX_SIGNALS, MAX_ZONES, OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS, PICKERS, PRESETS,
    SCORERS,
};
